#!/usr/bin/env python3
"""
Script: analyze_dataset.py
Deskripsi: Menganalisis karakteristik dataset citra dan menjalankan pengecekan kualitas data (Data Quality Check)
           seperti gambar rusak, 0-byte, duplikat, mode warna, dan statistik resolusi.
           Hasil analisis disimpan ke results/dataset_analysis/dataset_report.csv.

Penggunaan:
    python scripts/analyze_dataset.py --input data/selected --output_dir results/dataset_analysis
    python scripts/analyze_dataset.py --input data/original/images --output_dir results/dataset_analysis
"""

import argparse                          # Pembaca argumen baris perintah terminal (--input, --output_dir)
import hashlib                           # Generator hash MD5 untuk pendeteksian citra duplikat persis
import sys                               # Pengendali eksekusi sistem & pemutus program jika error
from collections import Counter, defaultdict  # Penghitung frekuensi & kamus otomatis
from pathlib import Path                 # Pengelola jalur folder/file lintas sistem operasi (Windows/Linux)
import cv2                               # OpenCV untuk membaca matriks piksel citra & dimensi channel
import numpy as np                       # NumPy untuk komputasi rata-rata & statistik resolusi/ukuran
import pandas as pd                      # Pandas untuk ekspor laporan statistik ke format tabel CSV
from PIL import Image                    # PIL (Pillow) untuk verifikasi kerusakan file gambar & mode warna


def compute_md5(file_path: Path) -> str:
    """Menghitung hash MD5 untuk memilah file duplikat persis."""
    hash_md5 = hashlib.md5()             # Inisialisasi objek kalkulator hash MD5
    with open(file_path, "rb") as f:     # Membuka file gambar dalam mode pembacaan byte biner ("rb")
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)       # Memperbarui kalkulasi hash per chunk 4096 byte
    return hash_md5.hexdigest()          # Mengekstrak string representasi heksadesimal hash MD5


def parse_video_source(filename: str) -> str:
    """Meng-ekstrak nama sumber video jika menggunakan konvensi {video_name}_frame_{idx}."""
    if "_frame_" in filename:
        return filename.split("_frame_")[0]  # Memotong string nama file berdasarkan pemisah '_frame_'
    return "unknown_source"


def analyze_dataset(input_dir: Path, output_dir: Path, min_res_thresh: int = 128):
    """
    Melakukan pemeriksaan kualitas data dan perhitungan statistik citra.
    """
    if not input_dir.exists():           # Memeriksa keberadaan direktori input
        print(f"[ERROR] Direktori input tidak ditemukan: {input_dir}")
        return

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
    image_files = [                      # Mencari seluruh file gambar secara rekursif
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in valid_extensions
    ]
    image_files.sort()                   # Mengurutkan nama file gambar secara alfabetis

    total_images = len(image_files)
    if total_images == 0:
        print(f"[WARNING] Tidak ditemukan file gambar di: {input_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)  # Membuat folder tujuan laporan jika belum ada

    print("\n" + "=" * 65)
    print(" MEMULAI ANALISIS DAN QUALTIY CHECK DATASET")
    print("=" * 65)
    print(f" Target Folder : {input_dir.as_posix()}")
    print(f" Total Gambar  : {total_images}")
    print("-" * 65)

    corrupt_files = []
    zero_byte_files = []
    small_res_files = []
    color_modes = Counter()
    channel_counts = Counter()
    file_formats = Counter()
    resolutions = []
    file_sizes_kb = []
    hashes = defaultdict(list)
    video_sources = Counter()

    records = []

    for img_path in image_files:         # Loop berulang mengevaluasi setiap file citra
        file_size = img_path.stat().st_size  # mengambil ukuran file dalam satuan byte
        file_size_kb = file_size / 1024.0   # Konversi ukuran file ke kilobyte (KB)
        file_sizes_kb.append(file_size_kb)

        video_src = parse_video_source(img_path.name)
        video_sources[video_src] += 1

        if file_size == 0:               # Deteksi file rusak 0-byte
            zero_byte_files.append(img_path.name)
            corrupt_files.append((img_path.name, "0-byte file"))
            continue

        md5_hash = compute_md5(img_path)  # Hitung hash MD5 file gambar
        hashes[md5_hash].append(img_path.name)

        is_valid = True
        try:
            with Image.open(img_path) as pil_img:  # Buka gambar dengan PIL (Pillow)
                pil_img.verify()         # Verifikasi keutuhan berkas gambar (deteksi corrupt)
            
            with Image.open(img_path) as pil_img:  # Buka ulang PIL untuk ekstrak metadata
                format_name = pil_img.format       # Ambil format berkas (JPEG/PNG)
                mode = pil_img.mode               # Ambil mode warna (RGB/L/RGBA)
                width, height = pil_img.size      # Ambil dimensi resolusi (lebar, tinggi)
                file_formats[format_name] += 1
                color_modes[mode] += 1

            cv_img = cv2.imread(str(img_path))     # Buka gambar dengan OpenCV
            if cv_img is None:
                is_valid = False
                corrupt_files.append((img_path.name, "OpenCV cv2.imread return None"))
            else:
                channels = cv_img.shape[2] if len(cv_img.shape) == 3 else 1  # Ambil jumlah channel (3 atau 1)
                channel_counts[channels] += 1

        except Exception as e:
            is_valid = False
            corrupt_files.append((img_path.name, str(e)))

        if is_valid:
            resolutions.append((width, height))
            if width < min_res_thresh or height < min_res_thresh:  # Deteksi resolusi di bawah batas minimum
                small_res_files.append((img_path.name, f"{width}x{height}"))

            records.append({
                "filename": img_path.name,
                "video_source": video_src,
                "width": width,
                "height": height,
                "aspect_ratio": round(width / height, 3) if height > 0 else 0,
                "channels": channels,
                "color_mode": mode,
                "size_kb": round(file_size_kb, 2),
                "md5": md5_hash
            })

    duplicate_groups = {k: v for k, v in hashes.items() if len(v) > 1}  # Memetakan kelompok gambar duplikat
    total_duplicates = sum(len(v) - 1 for v in duplicate_groups.values())

    if resolutions:
        widths = [r[0] for r in resolutions]
        heights = [r[1] for r in resolutions]
        min_res = f"{min(widths)}x{min(heights)}"
        max_res = f"{max(widths)}x{max(heights)}"
        mean_res = f"{int(np.mean(widths))}x{int(np.mean(heights))}"  # Hitung rata-rata resolusi
    else:
        min_res, max_res, mean_res = "N/A", "N/A", "N/A"

    avg_size_kb = np.mean(file_sizes_kb) if file_sizes_kb else 0.0  # Hitung rata-rata ukuran file dalam KB

    print(" RINGKASAN ANALISIS DATASET:")
    print(f" - Valid Images      : {len(records)} / {total_images}")
    print(f" - Format File       : {dict(file_formats)}")
    print(f" - Mode Warna        : {dict(color_modes)}")
    print(f" - Channel Gambar    : {dict(channel_counts)}")
    print(f" - Ukuran File Rerata: {avg_size_kb:.2f} KB (Min: {min(file_sizes_kb):.2f} KB, Max: {max(file_sizes_kb):.2f} KB)")
    print(f" - Resolusi Min      : {min_res}")
    print(f" - Resolusi Max      : {max_res}")
    print(f" - Resolusi Rerata   : {mean_res}")
    print(f" - Sumber Video      : {len(video_sources)} video ({dict(video_sources)})")
    print("-" * 65)

    print(" LAPORAN QUALITY CHECK:")
    print(f" [!] File Corrupt/Rusak  : {len(corrupt_files)} file")
    for fname, err in corrupt_files[:5]:
        print(f"      -> {fname}: {err}")
    
    print(f" [!] File 0-Byte         : {len(zero_byte_files)} file")
    print(f" [!] Resolusi Terlalu Kecil (<{min_res_thresh}px): {len(small_res_files)} file")
    print(f" [!] Gambar Duplikat (MD5): {total_duplicates} file dalam {len(duplicate_groups)} grup duplikat")
    print("=" * 65)

    df_details = pd.DataFrame(records)         # Mengonversi list dictionary ke DataFrame Pandas
    details_csv = output_dir / "image_details.csv"
    df_details.to_csv(details_csv, index=False) # Menyimpan laporan detail ke berkas CSV

    summary_data = [
        {"metric": "total_images", "value": total_images},
        {"metric": "valid_images", "value": len(records)},
        {"metric": "corrupt_images", "value": len(corrupt_files)},
        {"metric": "zero_byte_images", "value": len(zero_byte_files)},
        {"metric": "duplicate_images", "value": total_duplicates},
        {"metric": "small_resolution_images", "value": len(small_res_files)},
        {"metric": "avg_size_kb", "value": round(avg_size_kb, 2)},
        {"metric": "min_resolution", "value": min_res},
        {"metric": "max_resolution", "value": max_res},
        {"metric": "mean_resolution", "value": mean_res},
        {"metric": "total_video_sources", "value": len(video_sources)},
    ]
    df_summary = pd.DataFrame(summary_data)     # Buat DataFrame ringkasan statistik
    summary_csv = output_dir / "dataset_report.csv"
    df_summary.to_csv(summary_csv, index=False) # Menyimpan laporan ringkasan ke berkas CSV

    print(f"\n[INFO] Laporan ringkasan disimpan ke : {summary_csv.as_posix()}")
    print(f"[INFO] Laporan per-gambar disimpan ke  : {details_csv.as_posix()}\n")


def main():
    parser = argparse.ArgumentParser(          # Membuat parser argumen baris perintah terminal
        description="Analisis karakteristik citra dataset dan pengecekan kualitas data."
    )
    parser.add_argument("--input", type=str, default="data/selected", help="Direktori citra yang dianalisis.")
    parser.add_argument("--output_dir", type=str, default="results/dataset_analysis", help="Direktori hasil laporan.")
    parser.add_argument("--min_res", type=int, default=128, help="Batas minimum resolusi citra.")

    args = parser.parse_args()                 # Mengekstrak argumen terminal ke objek args

    analyze_dataset(                           # Eksekusi fungsi analisis dataset utama
        input_dir=Path(args.input),
        output_dir=Path(args.output_dir),
        min_res_thresh=args.min_res
    )


if __name__ == "__main__":                     # Memastikan skrip berjalan hanya saat dipanggil langsung
    main()
