#!/usr/bin/env python3
"""
Script: scripts/utils/compare_clahe_baseline.py
Deskripsi: Membandingkan metode usulan (Zero-DCE + CBAM) terhadap baseline murah CLAHE
           pada frame yang sama, memakai kriteria utama proyek ini: KONTRAS WALET
           TERHADAP LATAR (delta luminansi & Weber contrast), bukan kecerahan.

Latar belakang:
  Enhancement yang membuat citra lebih terang belum tentu membantu deteksi. Bila
  baseline CLAHE menghasilkan kontras walet yang lebih tinggi daripada Zero-DCE+CBAM,
  maka klaim "proposed method" tidak dapat disandarkan pada metrik enhancement saja
  dan harus dibuktikan lewat mAP deteksi.

Penggunaan:
    python scripts/utils/compare_clahe_baseline.py --orig_dir data/raw/frames/26_agustus --cbam_dir data/enhanced/zero_dce_cbam/26_agustus --output_dir results/enhancement/26_agustus
"""

import argparse               # Pembaca argumen baris perintah terminal (--orig_dir, --cbam_dir, dsb)
import sys                    # Pengendali eksekusi sistem & manipulasi modul sys.path
from pathlib import Path      # Pengelola jalur folder/file lintas sistem operasi (Windows/Linux)

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # Lokasi akar proyek
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))                      # Sisipkan jalur akar proyek ke sys.path

import cv2                    # OpenCV untuk CLAHE, konversi ruang warna, & pemrosesan citra
import numpy as np            # NumPy untuk komputasi matriks piksel & analisis statistik
import pandas as pd           # Pandas untuk ekspor tabel perbandingan ke format CSV
import matplotlib.pyplot as plt  # Pyplot untuk rendering figur perbandingan 3-panel

from enhancement.evaluate_enhancement import (  # Pakai ulang metrik resmi agar angka konsisten
    calculate_object_contrast,
    calculate_entropy,
    calculate_brisque_score,
    safe_to_csv,
)

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def apply_clahe(img_bgr: np.ndarray, clip_limit: float = 2.0, tile_grid: int = 8) -> np.ndarray:
    """
    Menerapkan CLAHE (Contrast Limited Adaptive Histogram Equalization) pada kanal
    luminansi L dari ruang warna LAB, sehingga kontras lokal naik tanpa menggeser warna.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    l_equalized = clahe.apply(l_channel)

    return cv2.cvtColor(cv2.merge((l_equalized, a_channel, b_channel)), cv2.COLOR_LAB2BGR)


def measure(img_bgr: np.ndarray) -> dict:
    """Menghitung metrik satu citra: kriteria utama (kontras objek) plus metrik pendukung."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
    delta_l, weber = calculate_object_contrast(gray)

    return {
        "mean_brightness": round(float(np.mean(gray)), 2),
        "contrast": round(float(np.std(gray)), 2),
        "entropy": round(calculate_entropy(gray.astype(np.uint8)), 4),
        "brisque": calculate_brisque_score(img_bgr),
        "object_delta_l": delta_l,
        "object_weber": weber,
    }


def generate_3panel_figure(orig: np.ndarray, clahe_img: np.ndarray, cbam: np.ndarray, save_path: Path):
    """Menghasilkan figur perbandingan 3 panel: ORIGINAL | CLAHE (baseline) | ZERO-DCE+CBAM."""
    panels = [
        (orig, "ORIGINAL"),
        (clahe_img, "CLAHE (Baseline)"),
        (cbam, "ZERO-DCE+CBAM (Proposed)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, (img, title) in zip(axes, panels):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=15, fontweight="bold", pad=15)
        ax.axis("off")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def compare(
    orig_dir: Path,
    cbam_dir: Path,
    output_dir: Path,
    clip_limit: float,
    tile_grid: int,
    save_clahe_dir: Path = None,
    max_figures: int = 3,
):
    # Basis iterasi adalah folder enhanced, karena hanya frame itu yang punya pasangan lengkap
    cbam_files = sorted(p for p in cbam_dir.rglob("*") if p.is_file() and p.suffix.lower() in VALID_EXTS)

    if not cbam_files:
        print(f"[WARNING] Tidak ditemukan citra enhanced di '{cbam_dir.as_posix()}'")
        return

    print("\n" + "=" * 70)
    print(" PERBANDINGAN BASELINE CLAHE vs ZERO-DCE+CBAM (KRITERIA: KONTRAS WALET)")
    print("=" * 70)
    print(f" Original Directory : {orig_dir.as_posix()}")
    print(f" Enhanced Directory : {cbam_dir.as_posix()}")
    print(f" CLAHE Parameter    : clipLimit={clip_limit}, tileGridSize=({tile_grid}, {tile_grid})")
    print("-" * 70)

    records = []
    figures_done = 0

    for cbam_p in cbam_files:
        rel_p = cbam_p.relative_to(cbam_dir)
        orig_p = orig_dir / rel_p

        if not orig_p.exists():
            print(f"  [SKIP] Pasangan original tidak ditemukan: {rel_p.as_posix()}")
            continue

        orig_img = cv2.imread(str(orig_p))
        cbam_img = cv2.imread(str(cbam_p))
        if orig_img is None or cbam_img is None:
            print(f"  [SKIP] Gagal membaca citra: {rel_p.as_posix()}")
            continue

        clahe_img = apply_clahe(orig_img, clip_limit=clip_limit, tile_grid=tile_grid)

        if save_clahe_dir is not None:
            clahe_out = save_clahe_dir / rel_p
            clahe_out.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(clahe_out), clahe_img)

        row = {"filename": orig_p.name}
        for prefix, img in (("orig", orig_img), ("clahe", clahe_img), ("cbam", cbam_img)):
            for key, val in measure(img).items():
                row[f"{prefix}_{key}"] = val
        records.append(row)

        print(f"  [OK] {rel_p.as_posix()} | Weber  orig={row['orig_object_weber']:.3f}  "
              f"clahe={row['clahe_object_weber']:.3f}  cbam={row['cbam_object_weber']:.3f}")

        if figures_done < max_figures:
            fig_path = output_dir / "figures" / f"3panel_clahe_vs_cbam_{orig_p.stem}.png"
            generate_3panel_figure(orig_img, clahe_img, cbam_img, fig_path)
            print(f"       [SAVED FIG] {fig_path.as_posix()}")
            figures_done += 1

    if not records:
        print("[WARNING] Tidak ada pasangan citra yang berhasil dibandingkan.")
        return

    df = pd.DataFrame(records)
    csv_path = output_dir / "metrics_reports" / "clahe_vs_cbam_report.csv"
    safe_to_csv(df, csv_path)

    base_weber = df["orig_object_weber"].mean()

    print(f"\n RINGKASAN RATA-RATA (n = {len(df)} frame):")
    header = f"  {'METODE':<22}{'Weber':>9}{'vs ORI':>10}{'DeltaL':>9}{'Bright':>9}{'Entropy':>9}{'BRISQUE':>9}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for prefix, label in (("orig", "ORIGINAL"), ("clahe", "CLAHE (baseline)"), ("cbam", "ZERO-DCE+CBAM")):
        weber = df[f"{prefix}_object_weber"].mean()
        pct = 100.0 * (weber / base_weber - 1.0) if base_weber > 0 else 0.0
        print(f"  {label:<22}{weber:9.4f}{pct:9.1f}%{df[f'{prefix}_object_delta_l'].mean():9.2f}"
              f"{df[f'{prefix}_mean_brightness'].mean():9.2f}{df[f'{prefix}_entropy'].mean():9.4f}"
              f"{df[f'{prefix}_brisque'].mean():9.2f}")

    clahe_weber = df["clahe_object_weber"].mean()
    cbam_weber = df["cbam_object_weber"].mean()

    print("\n KESIMPULAN:")
    if cbam_weber >= clahe_weber and cbam_weber >= base_weber:
        print("  Zero-DCE+CBAM unggul pada kriteria kontras walet terhadap latar.")
    elif clahe_weber > cbam_weber:
        print(f"  CLAHE baseline MENGUNGGULI Zero-DCE+CBAM ({clahe_weber:.4f} vs {cbam_weber:.4f}).")
        print("  Klaim keunggulan metode usulan tidak dapat disandarkan pada metrik enhancement;")
        print("  perlu dibuktikan melalui mAP / recall deteksi YOLO pada frame yang sama.")
    else:
        print(f"  Kedua metode berada di bawah citra original ({base_weber:.4f}).")
        print("  Enhancement pada dataset ini justru menurunkan keterpisahan walet dari latar.")

    print(f"\n[INFO] Laporan perbandingan disimpan ke: {csv_path.as_posix()}")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Perbandingan baseline CLAHE vs Zero-DCE+CBAM pada kriteria kontras walet.")
    parser.add_argument("--orig_dir", type=str, required=True, help="Folder citra original.")
    parser.add_argument("--cbam_dir", type=str, required=True, help="Folder citra hasil Zero-DCE+CBAM.")
    parser.add_argument("--output_dir", type=str, default="results/enhancement", help="Folder laporan dan figur.")
    parser.add_argument("--clip_limit", type=float, default=2.0, help="Parameter clipLimit CLAHE.")
    parser.add_argument("--tile_grid", type=int, default=8, help="Ukuran tileGridSize CLAHE (N x N).")
    parser.add_argument("--save_clahe_dir", type=str, default=None, help="Bila diisi, citra CLAHE ikut disimpan ke folder ini.")
    parser.add_argument("--max_figures", type=int, default=3, help="Jumlah figur 3-panel yang dihasilkan.")

    args = parser.parse_args()

    compare(
        orig_dir=Path(args.orig_dir),
        cbam_dir=Path(args.cbam_dir),
        output_dir=Path(args.output_dir),
        clip_limit=args.clip_limit,
        tile_grid=args.tile_grid,
        save_clahe_dir=Path(args.save_clahe_dir) if args.save_clahe_dir else None,
        max_figures=args.max_figures,
    )


if __name__ == "__main__":
    main()
