#!/usr/bin/env python3
"""
Script: select_frames.py
Deskripsi: Melakukan seleksi frame dari data/raw/frames/ ke data/selected/
           untuk menghilangkan frame duplikat/hampir identik akibat posisi kamera CCTV yang statis.

Fitur Seleksi:
  1. Stride / Subsampling (misal: ambil 1 dari setiap N frame).
  2. Deteksi Kemiripan Visual (SSIM, MSE, atau Histogram Correlation).
  3. Non-destruktif (mengopi file terpilih, tidak menghapus data mentah).

Penggunaan:
    python scripts/select_frames.py --input data/raw/frames --output data/selected --stride 2
    python scripts/select_frames.py --input data/raw/frames --output data/selected --method ssim --threshold 0.95
    python scripts/select_frames.py --input data/raw/frames --output data/selected --method mse --threshold 50.0
"""

import argparse            # Pembaca argumen terminal (--input, --method, --threshold)
import shutil              # Pustaka penyalinan/pemindahan berkas di disk (shutil.copy2)
import sys                 # Pengendali eksekusi sistem & penutup program jika error
from pathlib import Path   # Pengelola jalur folder/file lintas sistem operasi (Windows/Linux)
import cv2                 # OpenCV untuk membaca citra, konversi warna, & kalkulasi histogram
import numpy as np         # NumPy untuk komputasi matriks piksel & operasi matematika

try:
    from skimage.metrics import structural_similarity as ssim  # Fungsi hitung SSIM dari scikit-image
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


def calculate_mse(imgA: np.ndarray, imgB: np.ndarray) -> float:
    """Menghitung Mean Squared Error (MSE) antara dua gambar."""
    err = np.sum((imgA.astype("float") - imgB.astype("float")) ** 2)  # Menjumlahkan kuadrat selisih piksel
    err /= float(imgA.shape[0] * imgA.shape[1])                        # Membagi dengan total luas piksel (H x W)
    return err


def calculate_ssim(imgA: np.ndarray, imgB: np.ndarray) -> float:
    """Menghitung SSIM antara dua gambar grayscale."""
    if not HAS_SKIMAGE:
        return 0.0
    return ssim(imgA, imgB)  # Memanggil ssim() untuk mengukur kemiripan struktur visual antar citra


def calculate_hist_corr(imgA: np.ndarray, imgB: np.ndarray) -> float:
    """Menghitung korelasi histogram antara dua gambar."""
    histA = cv2.calcHist([imgA], [0], None, [256], [0, 256])  # Menghitung sebaran frekuensi piksel imgA
    histB = cv2.calcHist([imgB], [0], None, [256], [0, 256])  # Menghitung sebaran frekuensi piksel imgB
    cv2.normalize(histA, histA, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)  # Normalisasi histogram [0, 1]
    cv2.normalize(histB, histB, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)  # Normalisasi histogram [0, 1]
    return cv2.compareHist(histA, histB, cv2.HISTCMP_CORREL)                 # Korelasi kemiripan histogram


def parse_group_id(filename: str) -> str:
    """Meng-ekstrak ID grup/sumber video dari nama file (misal: 'video01' dari 'video01_frame_000001.jpg')."""
    if "_frame_" in filename:
        return filename.split("_frame_")[0]  # Memotong string nama file berdasarkan pemisah '_frame_'
    return "default_group"


def select_frames(
    input_dir: Path,
    output_dir: Path,
    stride: int = 1,
    method: str = "none",
    threshold: float = 0.95,
    max_resolution: int = 640
):
    """
    Menyeleksi frame berdasarkan stride atau metrik kemiripan citra.
    """
    if not input_dir.exists():  # Memeriksa apakah direktori input ada
        print(f"[ERROR] Direktori input tidak ditemukan: {input_dir}")
        return

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_paths = [  # Mencari seluruh gambar di direktori input
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in valid_extensions
    ]
    image_paths.sort()  # Mengurutkan file gambar berdasarkan nama secara alfabetis

    total_input = len(image_paths)
    if total_input == 0:
        print(f"[WARNING] Tidak ada file gambar di: {input_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)  # Membuat direktori tujuan jika belum ada

    print("\n" + "=" * 60)
    print(" MEMULAI SELEKSI FRAME")
    print("=" * 60)
    print(f" Direktori Input  : {input_dir.as_posix()}")
    print(f" Direktori Output : {output_dir.as_posix()}")
    print(f" Total Frame Awal : {total_input}")
    print(f" Stride Step      : {stride}")
    print(f" Metode Kemiripan : {method} (Threshold: {threshold})")
    print("-" * 60)

    selected_paths = []

    if method == "none":
        selected_paths = image_paths[::stride]  # Seleksi pencacahan berdasarkan kelipatan stride
    else:
        if method == "ssim" and not HAS_SKIMAGE:
            print("[WARNING] Package 'scikit-image' tidak ditemukan. Mengalihkan ke metode 'hist' (Histogram Correlation).")
            method = "hist"
            threshold = 0.98 if threshold == 0.95 else threshold

        prev_gray = None

        for idx, img_path in enumerate(image_paths):  # Loop berulang membaca indeks & path gambar
            if idx % stride != 0:
                continue

            img = cv2.imread(str(img_path))  # Membaca matriks piksel citra BGR dari disk
            if img is None:
                print(f"[WARNING] Gagal membaca {img_path.name}, dilewati.")
                continue

            h, w = img.shape[:2]  # Mengambil dimensi tinggi (h) dan lebar (w) gambar
            scale = max_resolution / max(h, w)
            if scale < 1.0:
                curr_gray = cv2.cvtColor(cv2.resize(img, (int(w * scale), int(h * scale))), cv2.COLOR_BGR2GRAY)  # Resize & konversi ke Grayscale
            else:
                curr_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                                               # Konversi langsung ke Grayscale

            if prev_gray is None:
                selected_paths.append(img_path)  # Frame pertama selalu disimpan
                prev_gray = curr_gray
            else:
                is_different = True
                if method == "ssim":
                    val = calculate_ssim(prev_gray, curr_gray)  # Hitung SSIM dengan frame sebelumnya
                    if val >= threshold:                         # Jika SSIM >= threshold -> Sangat mirip (buang)
                        is_different = False
                elif method == "mse":
                    val = calculate_mse(prev_gray, curr_gray)   # Hitung MSE dengan frame sebelumnya
                    if val <= threshold:                         # Jika MSE <= threshold -> Sangat mirip (buang)
                        is_different = False
                elif method == "hist":
                    val = calculate_hist_corr(prev_gray, curr_gray)  # Hitung Korelasi Histogram
                    if val >= threshold:                              # Jika korelasi >= threshold -> Sangat mirip (buang)
                        is_different = False

                if is_different:
                    selected_paths.append(img_path)
                    prev_gray = curr_gray

    copied_count = 0
    for p in selected_paths:
        group_id = parse_group_id(p.name)
        dest_dir = output_dir / group_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_file = dest_dir / p.name
        shutil.copy2(p, dest_file)  # Menyalin berkas citra terpilih ke folder tujuan beserta metadatanya
        copied_count += 1

    reduction = ((total_input - copied_count) / total_input) * 100 if total_input > 0 else 0.0

    print(f" Frame Sebelum Seleksi : {total_input}")
    print(f" Frame Setelah Seleksi : {copied_count}")
    print(f" Reduksi Frame         : {reduction:.2f}% (Tereliminasi: {total_input - copied_count} frame)")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(  # Penampung argumen terminal
        description="Seleksi frame CCTV untuk mengurangi redundansi visual."
    )
    parser.add_argument(               # Parameter input
        "--input",
        type=str,
        default="data/raw/frames",
        help="Direktori asal frame (default: data/raw/frames)."
    )
    parser.add_argument(               # Parameter output
        "--output",
        type=str,
        default="data/selected",
        help="Direktori tujuan penyimpanan frame terpilih (default: data/selected)."
    )
    parser.add_argument(               # Parameter stride
        "--stride",
        type=int,
        default=1,
        help="Ambil 1 frame setiap N frame (default: 1)."
    )
    parser.add_argument(               # Parameter metode kemiripan
        "--method",
        type=str,
        choices=["none", "ssim", "mse", "hist"],
        default="none",
        help="Metode eliminasi frame mirip: 'none', 'ssim', 'mse', 'hist' (default: none)."
    )
    parser.add_argument(               # Parameter threshold kemiripan
        "--threshold",
        type=float,
        default=0.95,
        help="Threshold kemiripan. Untuk SSIM/Hist: batasan kemiripan maksimum (misal 0.95). Untuk MSE: selisih minimum (misal 50.0)."
    )

    args = parser.parse_args()         # Mengekstrak argumen terminal ke objek args

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    select_frames(                     # Eksekusi fungsi seleksi frame
        input_dir=input_dir,
        output_dir=output_dir,
        stride=args.stride,
        method=args.method,
        threshold=args.threshold
    )


if __name__ == "__main__":             # Memastikan main() berjalan hanya jika dieksekusi langsung
    main()
