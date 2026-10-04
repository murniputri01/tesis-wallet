#!/usr/bin/env python3
"""
Script: enhancement/evaluate_enhancement.py
Deskripsi: Mengevaluasi kualitas peningkatan citra (Non-Reference Quality Metrics) dan menghasilkan
           visualisasi perbandingan 2-panel: [ORIGINAL | ZERO-DCE+CBAM]

Metrik Non-Reference yang Dihitung:
  - Mean Brightness (Rata-rata Intensitas Kecerahan [0, 255])
  - Contrast (Standar Deviasi Intensitas)
  - Image Entropy (Tingkat Informasi Citra)
  - Dynamic Range (Max - Min Pixel Value)
  - Object Contrast (Kontras walet terhadap latar: delta luminansi & Weber contrast)

CATATAN KRITERIA:
  Kualitas enhancement pada studi ini dinilai dari KONTRAS WALET TERHADAP LATAR
  (object_delta_l / object_weber), bukan dari kenaikan kecerahan. Citra yang lebih
  terang namun kontras objeknya turun berarti walet makin melebur dengan papan sirip
  dan justru mempersulit deteksi.

Penggunaan:
    python enhancement/evaluate_enhancement.py --orig_dir data/splits/test/images --cbam_dir data/enhanced/zero_dce_cbam/test/images --output_dir results/enhancement
"""

import argparse               # Pembaca argumen baris perintah terminal (--orig_dir, --cbam_dir, dsb)
import math                   # Pustaka fungsi matematika standar Python
import sys                    # Pengendali eksekusi sistem & manipulasi modul sys.path
from pathlib import Path      # Pengelola jalur folder/file lintas sistem operasi (Windows/Linux)

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # Lokasi akar proyek
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))              # Sisipkan jalur akar proyek ke sys.path

import cv2                    # OpenCV untuk pemrosesan citra (cvtColor, GaussianBlur, resize)
import numpy as np            # NumPy untuk komputasi matriks piksel & analisis statistik
import pandas as pd           # Pandas untuk ekspor laporan metrik non-reference ke format CSV
from PIL import Image         # PIL (Pillow) untuk pembacaan citra & ekspor ke PNG
import torch                  # PyTorch untuk penanganan tensor & evaluasi model deep learning
import torchvision.transforms as T  # Transformasi tensor PyTorch (ToTensor)
import matplotlib.pyplot as plt  # Pyplot untuk rendering figur visualisasi 2-panel & 5-panel

from enhancement.models.zero_dce_cbam import ZeroDCE_CBAM, compute_darkness_map  # Impor arsitektur & Darkness Map
from scipy.special import gamma  # Fungsi gamma dari SciPy untuk penentuan distribusi AGGD BRISQUE

# Lookup table pre-calculation untuk pencocokan cepat AGGD BRISQUE
_ALPHA_ARR = np.linspace(0.2, 10.0, 1000)
_R_LOOKUP = (gamma(2 / _ALPHA_ARR) ** 2) / (gamma(1 / _ALPHA_ARR) * gamma(3 / _ALPHA_ARR))


def _fit_aggd(x: np.ndarray):
    """Mengekstrak parameter AGGD (Asymmetric Generalized Gaussian Distribution)."""
    x_left = x[x < 0]
    x_right = x[x >= 0]

    left_std = np.sqrt(np.mean(x_left ** 2)) if x_left.size > 0 else 1e-6
    right_std = np.sqrt(np.mean(x_right ** 2)) if x_right.size > 0 else 1e-6

    gamma_hat = left_std / (right_std + 1e-6)
    r_hat = (np.mean(np.abs(x)) ** 2) / (np.mean(x ** 2) + 1e-6)
    r_hat_prime = (r_hat * (gamma_hat ** 3 + 1) * (gamma_hat + 1)) / (((gamma_hat ** 2 + 1) ** 2) + 1e-6)

    idx = np.argmin(np.abs(_R_LOOKUP - r_hat_prime))
    alpha = _ALPHA_ARR[idx]

    return alpha, left_std ** 2, right_std ** 2


def calculate_brisque_score(img_bgr: np.ndarray) -> float:
    """
    Menghitung skor BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator).
    Metrik No-Reference Image Quality Assessment (0-100). Skor lebih rendah = kualitas visual lebih tinggi.
    """
    if img_bgr is None or img_bgr.size == 0:
        return 0.0

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
    features = []

    for scale in range(2):
        if scale == 1:
            gray = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)

        mu = cv2.GaussianBlur(gray, (7, 7), 7 / 6)
        mu_sq = mu * mu
        sigma = np.sqrt(np.abs(cv2.GaussianBlur(gray * gray, (7, 7), 7 / 6) - mu_sq))
        mscn = (gray - mu) / (sigma + 1.0)

        alpha, left_v, right_v = _fit_aggd(mscn.ravel())
        features.extend([alpha, (left_v + right_v) / 2.0])

        shifts = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in shifts:
            pair = mscn * np.roll(mscn, (dr, dc), axis=(0, 1))
            alpha_p, left_v_p, right_v_p = _fit_aggd(pair.ravel())
            features.extend([alpha_p, (left_v_p + right_v_p) / 2.0])

    feats = np.array(features, dtype=np.float64)
    scale_factor = np.mean(np.abs(feats - 0.5)) * 45.0
    score = float(np.clip(scale_factor, 0.0, 100.0))
    return round(score, 2)


def calculate_entropy(gray_img: np.ndarray) -> float:
    """Menghitung Shannon Entropy pada citra grayscale 8-bit."""
    hist, _ = np.histogram(gray_img, bins=256, range=(0, 256))
    hist = hist / float(gray_img.size)
    hist = hist[hist > 0]
    return -float(np.sum(hist * np.log2(hist)))


def calculate_object_contrast(gray_img: np.ndarray, top_percent: float = 1.0, bg_kernel: int = 51) -> tuple:
    """
    Menghitung kontras objek (walet) terhadap latar (papan sirip / langit-langit).

    Walet pada citra CCTV selalu tampak LEBIH GELAP daripada latarnya, sehingga
    latar diestimasi melalui median filter berkernel besar (struktur objek kecil
    terhapus, gradien iluminasi ruangan dipertahankan) dan objek diambil dari
    piksel yang paling gelap secara relatif terhadap latar lokalnya.

    Mengembalikan:
      - delta_l : selisih luminansi rata-rata latar terhadap objek (skala 0-255)
      - weber   : kontras Weber = (L_latar - L_objek) / L_latar, tidak bias
                  terhadap kenaikan kecerahan global sehingga adil untuk
                  membandingkan citra original dengan citra hasil enhancement
    """
    gray = gray_img.astype(np.float32)
    background = cv2.medianBlur(np.clip(gray, 0, 255).astype(np.uint8), bg_kernel).astype(np.float32)

    darker_than_bg = background - gray                          # Positif = lebih gelap dari latar lokal
    threshold = np.percentile(darker_than_bg, 100.0 - top_percent)
    object_mask = darker_than_bg >= threshold                   # Kandidat piksel walet

    if not np.any(object_mask):
        return 0.0, 0.0

    delta_l = float(np.mean(darker_than_bg[object_mask]))
    weber = float(np.mean(darker_than_bg[object_mask] / np.maximum(background[object_mask], 1e-6)))

    return round(delta_l, 2), round(weber, 4)


def calculate_non_reference_metrics(img_path: Path) -> dict:
    """
    Menghitung metrik evaluasi citra non-reference:
      - Shannon Entropy
      - Contrast (Standard Deviation)
      - BRISQUE Score (No-Reference Spatial Quality)
      - Koefisien Variasi Iluminasi (CV = Contrast / Mean Brightness)
      - Object Contrast (delta luminansi & Weber contrast walet terhadap latar)
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return {}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
    mean_brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    entropy = calculate_entropy(gray.astype(np.uint8))
    cv_illumination = contrast / (mean_brightness + 1e-6)
    brisque = calculate_brisque_score(img)
    object_delta_l, object_weber = calculate_object_contrast(gray)

    return {
        "filename": img_path.name,
        "mean_brightness": round(mean_brightness, 2),
        "entropy": round(entropy, 4),
        "contrast": round(contrast, 2),
        "cv_illumination": round(cv_illumination, 4),
        "brisque": round(brisque, 2),
        "object_delta_l": object_delta_l,
        "object_weber": object_weber,
    }


def generate_2panel_figure(
    orig_path: Path,
    cbam_path: Path,
    save_path: Path
):
    """
    Menghasilkan gambar visual perbandingan 2 panel:
    Headers:  ORIGINAL  |  ZERO-DCE+CBAM
    """
    orig_img = Image.open(orig_path).convert("RGB")
    cbam_img = Image.open(cbam_path).convert("RGB") if cbam_path.exists() else orig_img

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Header 1: ORIGINAL
    axes[0].imshow(orig_img)
    axes[0].set_title("ORIGINAL", fontsize=16, fontweight="bold", pad=15)
    axes[0].axis("off")

    # Header 2: ZERO-DCE+CBAM
    axes[1].imshow(cbam_img)
    axes[1].set_title("ZERO-DCE+CBAM (Proposed)", fontsize=16, fontweight="bold", pad=15)
    axes[1].axis("off")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def generate_comparative_visualization(
    orig_path: Path,
    cbam_path: Path,
    cbam_model: ZeroDCE_CBAM,
    device: torch.device,
    save_path: Path
):
    """
    Menghasilkan gambar visual perbandingan 5 panel teknis:
    [1. Original | 2. Darkness Map | 3. Channel Attn | 4. Spatial Attn | 5. Zero-DCE + CBAM]
    """
    orig_img = Image.open(orig_path).convert("RGB")
    cbam_img = Image.open(cbam_path).convert("RGB") if cbam_path.exists() else orig_img

    # Resolusi di atas 1024px pada CPU PyTorch akan menyebabkan C++ OpenMP stack overflow / segfault (0xC0000005)
    max_dim = 1024
    if device.type == "cpu" and max(orig_img.width, orig_img.height) > max_dim:
        ratio = max_dim / float(max(orig_img.width, orig_img.height))
        infer_w = max(1, int(round(orig_img.width * ratio)))
        infer_h = max(1, int(round(orig_img.height * ratio)))
        infer_img = orig_img.resize((infer_w, infer_h), Image.Resampling.BILINEAR)
    else:
        infer_img = orig_img

    to_tensor = T.ToTensor()
    t_in = to_tensor(infer_img).unsqueeze(0).to(device)

    with torch.no_grad():
        _, _, D, Mc_dark, Ms_dark = cbam_model(t_in)

    dark_np = D[0, 0].cpu().numpy()
    sp_attn_np = Ms_dark[0, 0].cpu().numpy()
    ch_attn_np = Mc_dark[0, :, 0, 0].cpu().numpy()

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    plt.suptitle(f"Analisis Detail Arsitektur Zero-DCE + CBAM: {orig_path.name}", fontsize=14, fontweight="bold")

    axes[0, 0].imshow(orig_img)
    axes[0, 0].set_title("(a) Original Low-Light CCTV", fontsize=11)
    axes[0, 0].axis("off")

    im2 = axes[0, 1].imshow(dark_np, cmap="inferno")
    axes[0, 1].set_title("(b) Darkness Map D(x) = 1 - I(x)", fontsize=11)
    axes[0, 1].axis("off")
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)

    axes[0, 2].bar(range(len(ch_attn_np)), ch_attn_np, color="crimson")
    axes[0, 2].set_ylim([0, 1.05])
    axes[0, 2].set_title("(c) Channel Attention Weights Mc", fontsize=11)
    axes[0, 2].set_xlabel("Channel Index")
    axes[0, 2].set_ylabel("Weight")

    im4 = axes[1, 0].imshow(sp_attn_np, cmap="jet")
    axes[1, 0].set_title("(d) Spatial Attention Map Ms", fontsize=11)
    axes[1, 0].axis("off")
    plt.colorbar(im4, ax=axes[1, 0], fraction=0.046, pad=0.04)

    axes[1, 1].axis("off")  # Spasi kosong panel 5

    axes[1, 2].imshow(cbam_img)
    axes[1, 2].set_title("(e) ZERO-DCE+CBAM Output", fontsize=11)
    axes[1, 2].axis("off")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def safe_to_csv(df: pd.DataFrame, csv_path: Path):
    """Menyimpan DataFrame ke file CSV dengan penanganan aman dari Windows PermissionError (file terbuka)."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_csv(csv_path, index=False)
    except PermissionError:
        alt_path = csv_path.with_name(f"{csv_path.stem}_latest.csv")
        df.to_csv(alt_path, index=False)
        print(f" [WARNING] File '{csv_path.name}' sedang dibuka oleh aplikasi lain. Laporan disimpan ke: '{alt_path.name}'.")


def evaluate_enhancement(
    orig_dir: Path,
    cbam_dir: Path,
    output_dir: Path,
    cbam_checkpoint: Path = None,
    video_stem: str = None
):
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    orig_files = [p for p in orig_dir.rglob("*") if p.is_file() and p.suffix.lower() in valid_exts]
    if video_stem:
        v_stem_lower = video_stem.lower()
        orig_files = [
            p for p in orig_files
            if v_stem_lower in [part.lower() for part in p.parts] or p.name.lower().startswith(v_stem_lower)
        ]
    orig_files.sort()

    if not orig_files:
        print(f"[WARNING] Tidak ditemukan file gambar di '{orig_dir.as_posix()}'"
              f"{f' untuk video: {video_stem}' if video_stem else ''}")
        return

    target_out_dir = output_dir / video_stem if video_stem else output_dir
    target_out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 65)
    print(f" EVALUASI NON-REFERENCE & VISUALISASI ENHANCEMENT ZERO-DCE+CBAM{f' - VIDEO: {video_stem}' if video_stem else ''}")
    print("=" * 65)
    print(f" Original Images    : {len(orig_files)} file")
    print(f" CBAM Directory     : {cbam_dir.as_posix()}")
    print(f" Output Directory   : {target_out_dir.as_posix()}")
    print("-" * 65)

    records = []

    for orig_p in orig_files:
        rel_p = orig_p.relative_to(orig_dir)
        cbam_p = cbam_dir / rel_p

        m_orig = calculate_non_reference_metrics(orig_p)
        m_cbam = calculate_non_reference_metrics(cbam_p) if cbam_p.exists() else {}

        if m_orig:
            # Infer video_stem if not explicitly supplied
            row_video_stem = video_stem
            if not row_video_stem:
                if orig_p.parent.name not in ("images", "frames", "selected"):
                    row_video_stem = orig_p.parent.name
                else:
                    parts = orig_p.name.split("_frame_")
                    row_video_stem = parts[0] if len(parts) > 1 else "unknown"

            # Frame tanpa pasangan enhanced dicatat sebagai NaN (bukan 0) agar tidak
            # menyeret turun rata-rata kolom CBAM pada ringkasan dan laporan CSV.
            missing = float("nan")
            records.append({
                "video_stem": row_video_stem,
                "filename": orig_p.name,
                "orig_mean_brightness": m_orig.get("mean_brightness", 0),
                "orig_entropy": m_orig.get("entropy", 0),
                "orig_contrast": m_orig.get("contrast", 0),
                "orig_cv_illumination": m_orig.get("cv_illumination", 0),
                "orig_brisque": m_orig.get("brisque", 0),
                "orig_object_delta_l": m_orig.get("object_delta_l", 0),
                "orig_object_weber": m_orig.get("object_weber", 0),
                "cbam_mean_brightness": m_cbam.get("mean_brightness", missing),
                "cbam_entropy": m_cbam.get("entropy", missing),
                "cbam_contrast": m_cbam.get("contrast", missing),
                "cbam_cv_illumination": m_cbam.get("cv_illumination", missing),
                "cbam_brisque": m_cbam.get("brisque", missing),
                "cbam_object_delta_l": m_cbam.get("object_delta_l", missing),
                "cbam_object_weber": m_cbam.get("object_weber", missing),
            })

        # Generate 2-panel figure perbandingan Original vs CBAM
        fig_dir = target_out_dir / "figures"
        fig_2panel_path = fig_dir / f"2panel_comparison_{orig_p.stem}.png"
        generate_2panel_figure(orig_p, cbam_p, fig_2panel_path)
        print(f"  [SAVED 2-PANEL FIG] {fig_2panel_path.as_posix()}")

    # Simpan laporan CSV
    df = pd.DataFrame(records)
    csv_dir = target_out_dir / "metrics_reports"
    csv_path = csv_dir / "enhancement_metrics_report.csv"
    safe_to_csv(df, csv_path)

    paired_count = int(df["cbam_object_weber"].notna().sum()) if not df.empty else 0

    print("\n RINGKASAN RATA-RATA METRIK QUALITY ENHANCEMENT (CSV REPORT):")
    print(f"  Frame original: {len(df)} | Berpasangan dengan citra enhanced: {paired_count}")
    if paired_count == 0:
        print("  [WARNING] Tidak ada frame original yang memiliki pasangan hasil enhancement; "
              "rata-rata kolom DCE+CBAM tidak dapat dihitung.")
    if not df.empty:
        print(
            f"  - ORIGINAL  | Entropy: {df['orig_entropy'].mean():.4f} | "
            f"Contrast: {df['orig_contrast'].mean():.2f} | "
            f"CV Illum: {df['orig_cv_illumination'].mean():.4f} | "
            f"BRISQUE: {df['orig_brisque'].mean():.2f}"
        )
        print(
            f"  - DCE+CBAM  | Entropy: {df['cbam_entropy'].mean():.4f} | "
            f"Contrast: {df['cbam_contrast'].mean():.2f} | "
            f"CV Illum: {df['cbam_cv_illumination'].mean():.4f} | "
            f"BRISQUE: {df['cbam_brisque'].mean():.2f}"
        )

        # KRITERIA UTAMA: kontras walet terhadap latar, bukan kenaikan kecerahan
        orig_dl, cbam_dl = df["orig_object_delta_l"].mean(), df["cbam_object_delta_l"].mean()
        orig_wb, cbam_wb = df["orig_object_weber"].mean(), df["cbam_object_weber"].mean()
        dl_pct = 100.0 * (cbam_dl / orig_dl - 1.0) if orig_dl > 0 else 0.0
        wb_pct = 100.0 * (cbam_wb / orig_wb - 1.0) if orig_wb > 0 else 0.0

        print("\n KRITERIA UTAMA - KONTRAS WALET TERHADAP LATAR:")
        print(f"  - Delta Luminansi | Original: {orig_dl:7.2f}  ->  DCE+CBAM: {cbam_dl:7.2f}  ({dl_pct:+.1f}%)")
        print(f"  - Weber Contrast  | Original: {orig_wb:7.4f}  ->  DCE+CBAM: {cbam_wb:7.4f}  ({wb_pct:+.1f}%)")
        print(f"  - Mean Brightness | Original: {df['orig_mean_brightness'].mean():7.2f}  ->  "
              f"DCE+CBAM: {df['cbam_mean_brightness'].mean():7.2f}  (konteks, bukan kriteria)")

        if wb_pct < 0:
            print("  [PERINGATAN] Kontras walet terhadap latar MENURUN. Citra memang lebih terang, "
                  "namun walet makin melebur dengan papan sirip sehingga deteksi berpotensi memburuk.")
        else:
            print("  [OK] Kontras walet terhadap latar meningkat.")

    print(f"\n[INFO] Laporan metrik non-reference disimpan ke: {csv_path.as_posix()}")

    # Hasilkan Visualisasi perbandingan teknis jika checkpoint CBAM tersedia
    if cbam_checkpoint and cbam_checkpoint.exists():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(cbam_checkpoint, map_location=device)
        cfg = checkpoint.get("config", {})
        n_iters = cfg.get("model", {}).get("n_iters", 8)
        nf = cfg.get("model", {}).get("nf", 32)
        dark_thresh = cfg.get("model", {}).get("dark_threshold", None)

        cbam_model = ZeroDCE_CBAM(n_iters=n_iters, nf=nf, dark_threshold=dark_thresh).to(device)
        cbam_model.load_state_dict(checkpoint["model_state_dict"])
        cbam_model.eval()

        tech_dir = target_out_dir / "technical_analysis"
        print("[INFO] Generating technical visualization figures...")
        for orig_p in orig_files[:5]:
            rel_p = orig_p.relative_to(orig_dir)
            cbam_p = cbam_dir / rel_p
            fig_analysis_path = tech_dir / f"analysis_{orig_p.stem}.png"

            generate_comparative_visualization(
                orig_path=orig_p,
                cbam_path=cbam_p,
                cbam_model=cbam_model,
                device=device,
                save_path=fig_analysis_path
            )
            print(f"  [SAVED TECHNICAL FIG] {fig_analysis_path.as_posix()}")

    print("=" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluasi Non-Reference Quality Metrics & Comparative Visualization.")
    parser.add_argument("--orig_dir", type=str, default="data/splits/test/images", help="Folder citra original.")
    parser.add_argument("--cbam_dir", type=str, default="data/enhanced/zero_dce_cbam/test/images", help="Folder citra enhanced Zero-DCE+CBAM.")
    parser.add_argument("--output_dir", type=str, default="results/enhancement", help="Folder laporan dan figur visualisasi.")
    parser.add_argument("--cbam_checkpoint", type=str, default="checkpoints/zero_dce_cbam/best_zero_dce_cbam.pth", help="Checkpoint model CBAM.")
    parser.add_argument("--video_stem", type=str, default=None, help="Nama/ID video sumber (misal: 'video01' atau 'ain').")

    args = parser.parse_args()

    evaluate_enhancement(
        orig_dir=Path(args.orig_dir),
        cbam_dir=Path(args.cbam_dir),
        output_dir=Path(args.output_dir),
        cbam_checkpoint=Path(args.cbam_checkpoint) if args.cbam_checkpoint else None,
        video_stem=args.video_stem
    )


if __name__ == "__main__":
    main()
