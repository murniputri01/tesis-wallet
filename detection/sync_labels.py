#!/usr/bin/env python3
"""
Script: detection/sync_labels.py
Deskripsi: Menyinkronkan file label YOLO (.txt) dari `data/splits/` ke direktori citra
           ter-enhance `data/enhanced/zero_dce_cbam/` agar Ultralytics YOLO dapat
           membaca pasangan `images` dan `labels` secara otomatis.
"""

import sys                 # Pengendali eksekusi sistem & manipulasi modul sys.path
import shutil              # Pustaka penyalinan berkas di disk (shutil.copy2)
from pathlib import Path   # Pengelola jalur folder/file lintas sistem operasi (Windows/Linux)

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # Lokasi akar proyek
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))              # Sisipkan jalur akar proyek ke sys.path


def sync_labels(
    splits_dir: Path = Path("data/splits"),
    enhanced_dir: Path = Path("data/enhanced/zero_dce_cbam")
):
    print("\n" + "=" * 65)
    print(" SINKRONISASI LABEL YOLO KE DIREKTORI ENHANCED DATASET")
    print("=" * 65)

    if not splits_dir.exists():                        # Memeriksa ketersediaan folder splits
        print(f"[ERROR] Folder splits '{splits_dir}' tidak ditemukan!")
        return

    subsets = ["train", "val", "test"]
    synced_count = 0

    for sub in subsets:
        split_lbl_dir = splits_dir / sub / "labels"
        enh_sub_dir = enhanced_dir / sub
        enh_lbl_dir = enh_sub_dir / "labels"

        if not split_lbl_dir.exists():                 # Memeriksa ketersediaan subfolder labels
            print(f" [SKIP] {sub}: '{split_lbl_dir}' tidak ada.")
            continue

        enh_lbl_dir.mkdir(parents=True, exist_ok=True) # Membuat direktori tujuan labels pada enhanced

        txt_files = list(split_lbl_dir.rglob("*.txt"))  # Mencari seluruh file label .txt secara rekursif
        for txt_p in txt_files:
            rel_p = txt_p.relative_to(split_lbl_dir)    # Mengambil jalur relatif file terhadap folder asal
            target_p = enh_lbl_dir / rel_p
            target_p.parent.mkdir(parents=True, exist_ok=True)  # Membuat subfolder tujuan untuk menyamakan struktur
            shutil.copy2(txt_p, target_p)               # Menyalin file label .txt ke direktori enhanced
            synced_count += 1

        print(f" [OK] {sub}: {len(txt_files)} file label disinkronkan ke '{enh_lbl_dir.as_posix()}'")

    print(f" Total label disinkronkan: {synced_count} file .txt")
    print("=" * 65 + "\n")


def main():
    sync_labels()                                      # Memanggil fungsi sinkronisasi label utama


if __name__ == "__main__":                             # Memastikan skrip berjalan hanya saat dipanggil langsung
    main()

