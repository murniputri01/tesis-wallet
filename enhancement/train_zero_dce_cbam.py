#!/usr/bin/env python3
"""
Script: enhancement/train_zero_dce_cbam.py
Deskripsi: Pelatihan Model Utama Penelitian Zero-DCE + Modifikasi CBAM (Skenario C).
           Menyimpan best_zero_dce_cbam.pth berdasarkan Validation Loss terendah.

Penggunaan:
    python enhancement/train_zero_dce_cbam.py --config configs/zero_dce_cbam.yaml
    python enhancement/train_zero_dce_cbam.py --debug --epochs 5
"""

import argparse               # Pembaca argumen baris perintah terminal (--config, --debug, dsb)
import random                 # Pengacak statistik nilai seed pengacakan
import sys                    # Pengendali eksekusi sistem & manipulasi modul sys.path
from pathlib import Path      # Pengelola jalur folder/file lintas sistem operasi (Windows/Linux)

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # Lokasi akar proyek
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))              # Sisipkan jalur akar proyek ke sys.path

import numpy as np            # NumPy untuk komputasi array & penguncian random seed
import yaml                   # PyYAML untuk membaca file konfigurasi .yaml
import torch                  # PyTorch untuk penanganan tensor & pelatihan model deep learning
import torch.optim as optim   # Optimizer PyTorch (Adam)
from torch.utils.data import DataLoader, Subset  # PyTorch DataLoader & Subset manager
from PIL import Image         # PIL (Pillow) untuk penyimpanan sampel grid visualisasi

# Import lokal
from enhancement.dataloader import LowLightDataset      # Dataset PyTorch citra low-light
from enhancement.models.zero_dce_cbam import ZeroDCE_CBAM  # Arsitektur model Zero-DCE + CBAM
from enhancement.losses import ZeroDCELoss              # Loss function gabungan Zero-DCE


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


import os


def safe_torch_save(obj, filepath: Path):
    """Menyimpan checkpoint PyTorch secara aman di Windows (mencegah RuntimeError 1224)."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = filepath.with_name(f".tmp_{filepath.name}")
    torch.save(obj, str(tmp_path))
    if tmp_path.exists():
        if filepath.exists():
            try:
                os.remove(str(filepath))
            except Exception:
                pass
        os.replace(str(tmp_path), str(filepath))


def save_visualization_grid(
    input_t: torch.Tensor,
    dark_map_t: torch.Tensor,
    spatial_attn_t: torch.Tensor,
    enhanced_t: torch.Tensor,
    save_path: Path,
    target_height: int = 1080
):
    """
    Menyimpan grid visualisasi 4 panel: [Input | Darkness Map | Spatial Attention | Enhanced Output] dengan resolusi tinggi.
    """
    save_path = Path(save_path).resolve()
    save_path.parent.mkdir(parents=True, exist_ok=True)

    inp = np.clip(input_t[0].detach().cpu().numpy().transpose(1, 2, 0) * 255.0, 0, 255).astype(np.uint8)

    dark = np.clip(dark_map_t[0, 0].detach().cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
    dark_rgb = np.stack([dark] * 3, axis=-1)

    sp_attn = np.clip(spatial_attn_t[0, 0].detach().cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
    sp_attn_rgb = np.stack([sp_attn] * 3, axis=-1)

    enh = np.clip(enhanced_t[0].detach().cpu().numpy().transpose(1, 2, 0) * 255.0, 0, 255).astype(np.uint8)

    grid = np.hstack([inp, dark_rgb, sp_attn_rgb, enh])
    pil_img = Image.fromarray(grid)

    if target_height is not None and pil_img.height < target_height:
        scale = target_height / pil_img.height
        new_w = int(pil_img.width * scale)
        pil_img = pil_img.resize((new_w, target_height), Image.Resampling.LANCZOS)

    with open(str(save_path), "wb") as f:
        pil_img.save(f, format="PNG")


def train_zero_dce_cbam(
    config_path: str = "configs/zero_dce_cbam.yaml",
    debug: bool = False,
    debug_epochs: int = 5,
    video_stem: str = None
):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = cfg["training"].get("seed", 42)
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using Device: {device}")

    # Dataset & Dataloader
    train_dir = cfg["dataset"]["train_dir"]
    val_dir = cfg["dataset"]["val_dir"]

    # Alias Roboflow 'valid' -> 'val' jika folder val_dir belum ada
    if not Path(val_dir).exists() and Path("data/splits/valid/images").exists():
        val_dir = "data/splits/valid/images"

    if video_stem:
        v_train = Path(train_dir) / video_stem
        v_val = Path(val_dir) / video_stem
        selected_stem = Path("data/selected") / video_stem

        if v_train.exists():
            train_dir = str(v_train)
        elif selected_stem.exists():
            train_dir = str(selected_stem)

        if v_val.exists():
            val_dir = str(v_val)
        elif selected_stem.exists():
            val_dir = str(selected_stem)

    img_size = tuple(cfg["dataset"]["image_size"])
    batch_size = cfg["dataset"]["batch_size"]

    train_ds = LowLightDataset(images_dir=train_dir, image_size=img_size)
    val_ds = LowLightDataset(images_dir=val_dir, image_size=img_size)

    # Fallback jika val_ds kosong agar perulangan validasi & sampel gambar tidak terlewati
    if len(val_ds) == 0 and len(train_ds) > 0:
        print("[INFO] Validation dataset kosong, menggunakan train dataset untuk visualisasi sampel.")
        val_ds = train_ds

    if debug:
        print("[DEBUG MODE] Menggunakan subset 5 gambar untuk verifikasi cepat.")
        train_ds = Subset(train_ds, list(range(min(5, len(train_ds)))))
        val_ds = Subset(val_ds, list(range(min(5, len(val_ds)))))
        epochs = debug_epochs
        batch_size = min(2, len(train_ds))
    else:
        epochs = cfg["training"]["epochs"]

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=cfg["dataset"]["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=cfg["dataset"]["num_workers"])

    # Model Initialization
    n_iters = cfg["model"]["n_iters"]
    nf = cfg["model"]["nf"]
    dark_threshold = cfg["model"].get("dark_threshold", None)
    model = ZeroDCE_CBAM(n_iters=n_iters, nf=nf, dark_threshold=dark_threshold).to(device)

    # Loss Initialization
    loss_cfg = cfg["loss_weights"]
    criterion = ZeroDCELoss(
        lambda_spa=loss_cfg["lambda_spa"],
        lambda_exp=loss_cfg["lambda_exp"],
        lambda_col=loss_cfg["lambda_col"],
        lambda_tvA=loss_cfg["lambda_tvA"],
        patch_size=loss_cfg.get("patch_size", 16),
        target_exposure=loss_cfg.get("target_exposure", 0.55),
        use_pixel_weighting=loss_cfg.get("use_pixel_weighting", True),
        gamma=loss_cfg.get("gamma", 1.0)
    ).to(device)

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"]
    )

    # Interval penyimpanan citra sample validasi (epoch)
    save_frequency = cfg["training"].get("save_frequency", 5)

    # Gradient Clipping (mengikuti implementasi resmi Zero-DCE: max_norm = 0.1)
    grad_clip_norm = cfg["training"].get("grad_clip_norm", 0.1)

    ckpt_dir = Path(cfg["training"]["checkpoint_dir"])
    if video_stem:
        video_ckpt_dir = ckpt_dir / video_stem
        video_ckpt_dir.mkdir(parents=True, exist_ok=True)
        best_ckpt_path = video_ckpt_dir / ("best_zero_dce_cbam_debug.pth" if debug else "best_zero_dce_cbam.pth")
    else:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        best_ckpt_path = ckpt_dir / ("best_zero_dce_cbam_debug.pth" if debug else "best_zero_dce_cbam.pth")

    global_ckpt_path = ckpt_dir / ("best_zero_dce_cbam_debug.pth" if debug else "best_zero_dce_cbam.pth")

    if video_stem:
        res_dir = Path(f"results/enhancement/training_samples/{video_stem}")
    else:
        res_dir = Path("results/enhancement/training_samples")
    res_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 65)
    print(f" MEMULAI TRAINING ZERO-DCE + MODIFIKASI CBAM (SKENARIO C){f' - VIDEO: {video_stem}' if video_stem else ''}")
    print("=" * 65)
    print(f" Total Train Samples : {len(train_ds)}")
    print(f" Total Val Samples   : {len(val_ds)}")
    print(f" Total Epochs        : {epochs}")
    print(f" Batch Size          : {batch_size}")
    print(f" Steps / Epoch       : {len(train_loader)}")
    print(f" Total Steps         : {len(train_loader) * epochs}")
    print(f" Grad Clip Norm      : {grad_clip_norm}")
    print(f" Pixel Weighting     : {criterion.loss_exp.use_pixel_weighting}")
    print(f" Samples Output Dir  : {res_dir.as_posix()}")
    print(f" Checkpoint Output   : {best_ckpt_path.as_posix()}")
    print("-" * 65)

    best_val_loss = float("inf")
    train_history = []
    val_history = []

    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        train_loss_accum = 0.0
        for batch_idx, img in enumerate(train_loader):
            img = img.to(device)
            optimizer.zero_grad()

            enhanced, A, D, Mc_dark, Ms_dark = model(img)
            loss, loss_components = criterion(img, enhanced, A)

            loss.backward()
            if grad_clip_norm is not None and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

            train_loss_accum += loss.item()

        avg_train_loss = train_loss_accum / max(1, len(train_loader))
        train_history.append(avg_train_loss)

        # Validation Phase
        model.eval()
        val_loss_accum = 0.0
        with torch.no_grad():
            for val_batch_idx, img_val in enumerate(val_loader):
                img_val = img_val.to(device)
                enhanced_val, A_val, D_val, Mc_val, Ms_val = model(img_val)
                loss_val, _ = criterion(img_val, enhanced_val, A_val)
                val_loss_accum += loss_val.item()

                if val_batch_idx == 0 and (epoch % save_frequency == 0 or epoch == epochs or debug):
                    save_visualization_grid(
                        img_val, D_val, Ms_val, enhanced_val,
                        res_dir / f"zero_dce_cbam_sample_epoch_{epoch:03d}.png"
                    )

        avg_val_loss = val_loss_accum / max(1, len(val_loader))
        val_history.append(avg_val_loss)

        print(
            f" Epoch [{epoch:03d}/{epochs:03d}] | "
            f"Train Loss: {avg_train_loss:.6f} | "
            f"Val Loss: {avg_val_loss:.6f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "config": cfg
            }
            safe_torch_save(state, best_ckpt_path)
            if best_ckpt_path != global_ckpt_path:
                safe_torch_save(state, global_ckpt_path)
            print(f"  [SAVED] Checkpoint terbaik diperbarui: {best_ckpt_path.as_posix()}")

    print("=" * 65)
    print(f" TRAINING SELESAI! Best Validation Loss: {best_val_loss:.6f}")
    print("=" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Training Model Zero-DCE + Modifikasi CBAM (Skenario C).")
    parser.add_argument("--config", type=str, default="configs/zero_dce_cbam.yaml", help="Path file konfigurasi YAML.")
    parser.add_argument("--debug", action="store_true", help="Aktifkan mode debug untuk verifikasi cepat.")
    parser.add_argument("--epochs", type=int, default=5, help="Jumlah epoch pada mode debug.")
    parser.add_argument("--video_stem", type=str, default=None, help="Nama/ID video sumber (misal: 'video01' atau 'ain').")

    args = parser.parse_args()

    train_zero_dce_cbam(
        config_path=args.config,
        debug=args.debug,
        debug_epochs=args.epochs,
        video_stem=args.video_stem
    )


if __name__ == "__main__":
    main()
