#!/usr/bin/env python3
"""
Script: scripts/check_upscale_quality.py
Deskripsi: Memverifikasi perbaikan pipeline upscaling pada enhancement/enhance.py dengan
           membandingkan dua jalur pemrosesan pada citra & model yang sama:

             OLD : output model -> kuantisasi uint8 -> resize (PIL)          [perilaku lama]
             NEW : output model -> resize float32 -> kuantisasi uint8 -> unsharp mask opsional

           Karena kedua jalur menghasilkan resolusi akhir yang identik, metriknya
           dapat dibandingkan secara langsung.

Metrik:
  - Laplacian Variance : energi frekuensi tinggi (proxy ketajaman global)
  - Tenengrad          : rata-rata magnitudo gradien Sobel (proxy kekuatan tepi)
  - Weber Contrast     : |mean(bbox) - mean(ring sekitar)| / mean(ring sekitar)
                         -> seberapa "kentara" walet terhadap latar plafon
  - Edge Contrast      : rata-rata magnitudo gradien di dalam bbox walet saja

Penggunaan:
    python scripts/check_upscale_quality.py --model zero_dce_cbam --checkpoint checkpoints/zero_dce_cbam/best_zero_dce_cbam.pth --scale 2.0 --sharpen 80
"""

import argparse               # Pembaca argumen baris perintah terminal (--checkpoint, --scale, dsb)
import sys                    # Pengendali eksekusi sistem & penambahan jalur direktori sys.path
from pathlib import Path      # Pengelola jalur folder/file lintas sistem operasi (Windows/Linux)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # Lokasi akar proyek
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))  # Sisipkan jalur akar proyek ke sys.path

import cv2                    # OpenCV untuk pemrosesan citra (Sobel, Laplacian, cvtColor)
import numpy as np            # NumPy untuk operasi matriks piksel & kalkulasi statistik
import pandas as pd           # Pandas untuk ekspor laporan metrik ke format CSV
import torch                  # PyTorch untuk penanganan tensor & evaluasi model deep learning
import torchvision.transforms as T  # Transformasi tensor PyTorch (ToTensor)
from PIL import Image, ImageFilter  # PIL (Pillow) untuk manipulasi citra, resample, & UnsharpMask
import matplotlib             # Matplotlib untuk rendering figur visualisasi
matplotlib.use("Agg")         # Set backend Matplotlib non-interaktif
import matplotlib.pyplot as plt  # Pyplot untuk membuat figur 3-kolom perbandingan crop

from enhancement.enhance import _resize_float, _resolve_target_size  # Impor fungsi penunjang upscaling
from enhancement.models.zero_dce_cbam import ZeroDCE_CBAM            # Impor arsitektur model Zero-DCE + CBAM


RESAMPLE_MAP = {
    "lanczos": Image.Resampling.LANCZOS,
    "bicubic": Image.Resampling.BICUBIC,
    "bilinear": Image.Resampling.BILINEAR,
}


def load_model(model_type: str, checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = checkpoint.get("config", {})
    n_iters = cfg.get("model", {}).get("n_iters", 8)
    nf = cfg.get("model", {}).get("nf", 32)

    dark_thresh = cfg.get("model", {}).get("dark_threshold", None)
    model = ZeroDCE_CBAM(n_iters=n_iters, nf=nf, dark_threshold=dark_thresh).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def sharpness_metrics(img_u8: np.ndarray) -> dict:
    """Metrik ketajaman global pada citra RGB uint8."""
    gray = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY).astype(np.float32)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    tenengrad = float(np.mean(np.sqrt(gx ** 2 + gy ** 2)))
    return {"lap_var": lap_var, "tenengrad": tenengrad}


def read_yolo_boxes(label_path: Path, img_w: int, img_h: int):
    """Membaca label YOLO ternormalisasi -> daftar (x1, y1, x2, y2) dalam piksel."""
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        _, xc, yc, bw, bh = (float(v) for v in parts[:5])
        x1 = int(round((xc - bw / 2) * img_w))
        y1 = int(round((yc - bh / 2) * img_h))
        x2 = int(round((xc + bw / 2) * img_w))
        y2 = int(round((yc + bh / 2) * img_h))
        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2, y2))
    return boxes


def bbox_metrics(img_u8: np.ndarray, box, ring: float = 1.0):
    """Weber contrast bbox vs ring latar di sekitarnya, plus kekuatan tepi dalam bbox."""
    h, w = img_u8.shape[:2]
    x1, y1, x2, y2 = box
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None

    gray = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY).astype(np.float32)
    inner = gray[y1:y2, x1:x2]

    mx, my = int((x2 - x1) * ring), int((y2 - y1) * ring)
    ox1, oy1 = max(0, x1 - mx), max(0, y1 - my)
    ox2, oy2 = min(w, x2 + mx), min(h, y2 + my)
    outer = gray[oy1:oy2, ox1:ox2].copy()
    # Buang wilayah bbox dari statistik latar
    outer[y1 - oy1:y2 - oy1, x1 - ox1:x2 - ox1] = np.nan
    bg_mean = float(np.nanmean(outer))
    if not np.isfinite(bg_mean) or bg_mean <= 1e-6:
        return None

    gx = cv2.Sobel(inner, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(inner, cv2.CV_32F, 0, 1, ksize=3)

    return {
        "weber": abs(float(np.mean(inner)) - bg_mean) / bg_mean,
        "edge": float(np.mean(np.sqrt(gx ** 2 + gy ** 2))),
    }


def save_crop_panel(native_u8, old_u8, new_u8, b_nat, b_old, b_new, out_path, name, args):
    """Menyimpan panel 3-kolom crop walet: NATIVE | OLD | NEW (zoom NEAREST, tanpa pemulusan)."""
    def crop(img, box, pad_ratio=1.2):
        h, w = img.shape[:2]
        x1, y1, x2, y2 = box
        px, py = int((x2 - x1) * pad_ratio), int((y2 - y1) * pad_ratio)
        return img[max(0, y1 - py):min(h, y2 + py), max(0, x1 - px):min(w, x2 + px)]

    crops = [crop(native_u8, b_nat), crop(old_u8, b_old), crop(new_u8, b_new)]
    sharpen_note = f" + unsharp {args.sharpen:.0f}%" if args.sharpen > 0 else ""
    titles = [
        "NATIVE (tanpa upscale)",
        f"OLD: uint8 -> resize {args.upscale_method}",
        "NEW: resize float -> uint8" + sharpen_note,
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
    for ax, c, t in zip(axes, crops, titles):
        if c.size:
            ax.imshow(c, interpolation="nearest")
        ax.set_title(t, fontsize=10)
        ax.axis("off")
    fig.suptitle(f"Perbandingan crop walet - {name}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Verifikasi kualitas pipeline upscaling enhancement.")
    parser.add_argument("--model", default="zero_dce_cbam", choices=["zero_dce_cbam"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input_dir", default="data/splits", help="Direktori split (dicari rekursif).")
    parser.add_argument("--scale", type=float, default=2.0)
    parser.add_argument("--target_size", type=int, nargs=2, default=None, metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--allow_stretch", action="store_true")
    parser.add_argument("--upscale_method", default="lanczos", choices=list(RESAMPLE_MAP))
    parser.add_argument("--sharpen", type=float, default=80.0)
    parser.add_argument("--limit", type=int, default=6, help="Jumlah gambar yang diperiksa.")
    parser.add_argument("--output_dir", default="results/upscale_check")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"[ERROR] Checkpoint tidak ditemukan: {checkpoint_path}")
        return

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model, checkpoint_path, device)
    resample_filter = RESAMPLE_MAP[args.upscale_method]
    target_size = tuple(args.target_size) if args.target_size else None

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    image_paths = sorted(
        p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in valid_exts
    )[: args.limit]

    if not image_paths:
        print(f"[WARNING] Tidak ada gambar di '{input_dir.as_posix()}'")
        return

    sharpen_label = "OFF" if args.sharpen <= 0 else f"{args.sharpen:.0f}%"
    print("\n" + "=" * 78)
    print(f" VERIFIKASI PIPELINE UPSCALING: {args.model.upper()} | {args.upscale_method.upper()}")
    print("=" * 78)
    print(f" Gambar diperiksa : {len(image_paths)}")
    print(f" Scale / Target   : {args.scale:.2f}x / {target_size}")
    print(f" Unsharp Mask     : {sharpen_label}")
    print(f" Device           : {device}")
    print("-" * 78)

    global_rows, box_rows = [], []
    panel_saved = False
    to_tensor = T.ToTensor()

    with torch.no_grad():
        for img_p in image_paths:
            with Image.open(img_p) as pil_img:
                rgb_img = pil_img.convert("RGB")
                orig_w, orig_h = rgb_img.size
                tensor_in = to_tensor(rgb_img).unsqueeze(0).to(device)

            enhanced_t, _, _, _, _ = model(tensor_in)

            enh_f = enhanced_t[0].clamp(0.0, 1.0).cpu().numpy().transpose(1, 2, 0).astype(np.float32)
            final_w, final_h = _resolve_target_size(
                orig_w, orig_h, args.scale, target_size, args.allow_stretch
            )
            if (final_w, final_h) == (orig_w, orig_h):
                print(f" [SKIP] {img_p.name}: resolusi target sama dengan asli, tidak ada upscale.")
                continue

            native_u8 = np.clip(enh_f * 255.0, 0, 255).astype(np.uint8)

            # OLD: kuantisasi dulu, baru resize (dan stretch penuh bila target_size diberikan)
            old_size = target_size if target_size is not None else (final_w, final_h)
            old_u8 = np.asarray(
                Image.fromarray(native_u8).resize(old_size, resample_filter), dtype=np.uint8
            )

            # NEW: resize float -> kuantisasi -> unsharp opsional
            resized_f = _resize_float(enh_f, (final_w, final_h), resample_filter)
            new_img = Image.fromarray(np.clip(resized_f * 255.0, 0, 255).astype(np.uint8))
            if args.sharpen > 0:
                new_img = new_img.filter(
                    ImageFilter.UnsharpMask(radius=1.0, percent=int(round(args.sharpen)), threshold=3)
                )
            new_u8 = np.asarray(new_img, dtype=np.uint8)

            m_old, m_new = sharpness_metrics(old_u8), sharpness_metrics(new_u8)
            global_rows.append({
                "filename": img_p.name,
                "native_size": f"{orig_w}x{orig_h}",
                "old_size": f"{old_u8.shape[1]}x{old_u8.shape[0]}",
                "new_size": f"{new_u8.shape[1]}x{new_u8.shape[0]}",
                "lap_var_old": m_old["lap_var"],
                "lap_var_new": m_new["lap_var"],
                "tenengrad_old": m_old["tenengrad"],
                "tenengrad_new": m_new["tenengrad"],
            })

            label_path = img_p.parent.parent / "labels" / f"{img_p.stem}.txt"
            boxes = read_yolo_boxes(label_path, orig_w, orig_h)
            sx, sy = final_w / orig_w, final_h / orig_h
            osx, osy = old_size[0] / orig_w, old_size[1] / orig_h

            for idx, box in enumerate(boxes):
                b_new = tuple(int(round(v * s)) for v, s in zip(box, (sx, sy, sx, sy)))
                b_old = tuple(int(round(v * s)) for v, s in zip(box, (osx, osy, osx, osy)))

                r_nat = bbox_metrics(native_u8, box)
                r_old = bbox_metrics(old_u8, b_old)
                r_new = bbox_metrics(new_u8, b_new)
                if not (r_nat and r_old and r_new):
                    continue

                box_rows.append({
                    "filename": img_p.name,
                    "box_id": idx,
                    "weber_native": r_nat["weber"],
                    "weber_old": r_old["weber"],
                    "weber_new": r_new["weber"],
                    "edge_native": r_nat["edge"],
                    "edge_old": r_old["edge"],
                    "edge_new": r_new["edge"],
                })

                if not panel_saved:
                    save_crop_panel(
                        native_u8, old_u8, new_u8, box, b_old, b_new,
                        output_dir / "crop_comparison.png", img_p.name, args
                    )
                    panel_saved = True

            print(f" [OK] {img_p.name}: {orig_w}x{orig_h} -> {final_w}x{final_h}, {len(boxes)} bbox")

    if not global_rows:
        print("\n[WARNING] Tidak ada gambar yang diproses (periksa --scale / --target_size).")
        return

    df_global = pd.DataFrame(global_rows)
    df_global.to_csv(output_dir / "global_metrics.csv", index=False)

    print("\n" + "-" * 78)
    print(" KETAJAMAN GLOBAL (resolusi akhir identik; makin tinggi makin tajam)")
    print("-" * 78)
    for col in ("lap_var", "tenengrad"):
        old_m, new_m = df_global[f"{col}_old"].mean(), df_global[f"{col}_new"].mean()
        delta = (new_m - old_m) / old_m * 100 if old_m else float("nan")
        print(f" {col:<12} OLD={old_m:10.3f}  NEW={new_m:10.3f}  ({delta:+.1f}%)")

    if box_rows:
        df_box = pd.DataFrame(box_rows)
        df_box.to_csv(output_dir / "bbox_metrics.csv", index=False)
        print("\n" + "-" * 78)
        print(f" KEKENTARAAN WALET PADA {len(df_box)} BBOX (NATIVE = tanpa upscale, sebagai acuan)")
        print("-" * 78)
        for col in ("weber", "edge"):
            nat_m = df_box[f"{col}_native"].mean()
            old_m, new_m = df_box[f"{col}_old"].mean(), df_box[f"{col}_new"].mean()
            d_old = (old_m - nat_m) / nat_m * 100
            d_new = (new_m - nat_m) / nat_m * 100
            print(
                f" {col:<8} NATIVE={nat_m:9.4f}   OLD={old_m:9.4f} ({d_old:+6.1f}%)"
                f"   NEW={new_m:9.4f} ({d_new:+6.1f}%)"
            )
    else:
        print("\n [INFO] Tidak ada bbox walet pada gambar yang diperiksa;")
        print("        hanya metrik ketajaman global yang tersedia.")

    print("\n Output: " + (output_dir / "global_metrics.csv").as_posix())
    if panel_saved:
        print("         " + (output_dir / "crop_comparison.png").as_posix())
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
