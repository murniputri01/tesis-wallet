#!/usr/bin/env python3
"""
Script: split_dataset.py
Deskripsi: Membagi dataset teranotasi (data/original/images & data/original/labels) menjadi
           70% Training, 15% Validation, dan 15% Testing.

Mekanisme Pencegahan Data Leakage:
  - Menggunakan Group Split (berdasarkan ID sumber video asal).
  - Seluruh frame yang berasal dari video CCTV yang sama dikelompokkan dan diletakkan secara utuh
    pada salah satu subset (Train, Val, atau Test) agar model tidak "menghafal" latar belakang kandang
    walet yang identik pada data uji.
  - Menggunakan fixed random seed (default: 42) agar pembagian reproducible.

Penggunaan:
    python scripts/split_dataset.py --images_dir data/original/images --labels_dir data/original/labels --output_dir data/splits --seed 42
"""

import argparse                               # Pembaca argumen baris perintah terminal (--images_dir, --seed, dsb)
import random                                 # Pengacak berbasis random seed acak yang reproducible
import shutil                                 # Penyalinan berkas terpasang (image & label .txt) ke folder tujuan
import sys                                    # Pengendali eksekusi sistem & penutup program jika error
from collections import Counter, defaultdict  # Struktur data penghitung frekuensi & kamus otomatis
from pathlib import Path                      # Pengelola jalur folder/file lintas sistem operasi (Windows/Linux)


def parse_group_id(filename: str) -> str:
    """Meng-ekstrak ID grup/sumber video dari nama file (misal: 'video01' dari 'video01_frame_000001.jpg')."""
    if "_frame_" in filename:
        return filename.split("_frame_")[0]  # Memotong string nama file berdasarkan pemisah '_frame_'
    return "default_group"


def split_dataset(
    images_dir: Path,
    labels_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    use_group_split: bool = True
):
    """
    Membagi dataset secara terstruktur dan terpasangkan (image & label) dengan rasio 70:15:15.
    """
    if not images_dir.exists() or not labels_dir.exists():  # Memeriksa ketersediaan folder images & labels
        print(f"[ERROR] Direktori images ({images_dir}) atau labels ({labels_dir}) tidak ditemukan.")
        return

    total_ratio = train_ratio + val_ratio + test_ratio        # Hitung total proporsi rasio
    train_r = train_ratio / total_ratio                       # Normalisasi rasio data training
    val_r = val_ratio / total_ratio                           # Normalisasi rasio data validasi
    test_r = test_ratio / total_ratio                         # Normalisasi rasio data testing

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    image_files = [                                           # Cari seluruh file gambar di direktori images
        p for p in images_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in valid_exts
    ]
    image_files.sort()                                        # Urutkan nama file gambar secara alfabetis

    total_images = len(image_files)
    if total_images == 0:
        print(f"[WARNING] Tidak ada file gambar ditemukan di {images_dir}.")
        return

    label_map = {                                             # Pemetaan seluruh file label .txt berbasis nama stem
        p.stem: p for p in labels_dir.rglob("*.txt")
        if p.is_file()
    }

    pairs = []
    missing_lbl_count = 0
    for img_p in image_files:                                 # Cocokkan pasangan gambar dan label .txt
        if img_p.stem in label_map:
            pairs.append((img_p, label_map[img_p.stem]))      # Simpan tuple pasangan (image_path, label_path)
        else:
            missing_lbl_count += 1
            print(f"[WARNING] Gambar '{img_p.name}' tidak memiliki label di {labels_dir}. File ini dilewati dari split.")

    if not pairs:
        print("[ERROR] Tidak ada pasangan gambar dan label yang valid untuk diproses.")
        return

    print("\n" + "=" * 65)
    print(" MEMULAI SPLIT DATASET (70% Train, 15% Val, 15% Test)")
    print("=" * 65)
    print(f" Total Pasangan Valid : {len(pairs)} (Missing label: {missing_lbl_count})")
    print(f" Random Seed           : {seed}")
    print(f" Group-based Splitting : {'AKTIF (Video-level Split)' if use_group_split else 'NON-AKTIF (Frame-level Split)'}")
    print("-" * 65)

    random.seed(seed)                                         # Mengunci random seed untuk pengacakan konsisten

    groups = defaultdict(list)                                # Kelompokkan pasangan data berdasarkan ID grup video
    for img_p, lbl_p in pairs:
        grp_id = parse_group_id(img_p.name) if use_group_split else img_p.name
        groups[grp_id].append((img_p, lbl_p))

    group_keys = sorted(list(groups.keys()))                  # Ambil daftar ID grup video terurut
    random.shuffle(group_keys)                                # Acak urutan grup video menggunakan random seed

    train_pairs, val_pairs, test_pairs = [], [], []

    if use_group_split and len(group_keys) >= 3:
        total_items = len(pairs)
        target_train = int(round(total_items * train_r))      # Hitung target jumlah item data training
        target_val = int(round(total_items * val_r))          # Hitung target jumlah item data validasi

        curr_train, curr_val = 0, 0

        for grp_key in group_keys:                            # Alokasikan seluruh isi video ke salah satu split
            grp_items = groups[grp_key]
            n_items = len(grp_items)

            if curr_train + n_items <= target_train or (curr_train == 0 and curr_val > 0):
                train_pairs.extend(grp_items)                  # Masukkan ke kelompok Training
                curr_train += n_items
            elif curr_val + n_items <= target_val or (curr_val == 0 and len(test_pairs) > 0):
                val_pairs.extend(grp_items)                    # Masukkan ke kelompok Validation
                curr_val += n_items
            else:
                test_pairs.extend(grp_items)                   # Masukkan sisanya ke kelompok Testing
    else:
        if use_group_split and len(group_keys) < 3:
            print(
                f"[WARNING] Hanya ditemukan {len(group_keys)} grup sumber video ('{group_keys}'). "
                f"Minimal dibutuhkan 3 video terpisah untuk Video-level Split (Train/Val/Test). "
                f"Sistem beralih ke Frame-level Random Split dengan seed={seed}.\n"
                f"PERHATIAN: Frame-level split pada video CCTV statis dapat berpotensi Data Leakage!"
            )
        
        all_pairs_shuffled = list(pairs)                      # Salin seluruh daftar pasangan data
        random.shuffle(all_pairs_shuffled)                    # Acak urutan frame secara individual

        n_total = len(all_pairs_shuffled)
        n_train = int(round(n_total * train_r))
        n_val = int(round(n_total * val_r))

        train_pairs = all_pairs_shuffled[:n_train]
        val_pairs = all_pairs_shuffled[n_train:n_train + n_val]
        test_pairs = all_pairs_shuffled[n_train + n_val:]

    splits_dict = {                                           # Kamus pemetaan subset data split
        "train": train_pairs,
        "val": val_pairs,
        "test": test_pairs
    }

    for split_name, split_list in splits_dict.items():         # Loop berulang membuat struktur folder split
        img_dest_dir = output_dir / split_name / "images"
        lbl_dest_dir = output_dir / split_name / "labels"
        img_dest_dir.mkdir(parents=True, exist_ok=True)
        lbl_dest_dir.mkdir(parents=True, exist_ok=True)

        for src_img, src_lbl in split_list:                    # Menyalin berkas citra & label ke folder split tujuan
            dest_img_path = img_dest_dir / src_img.name
            dest_lbl_path = lbl_dest_dir / src_lbl.name
            shutil.copy2(src_img, dest_img_path)               # Menyalin berkas citra langsung flat
            shutil.copy2(src_lbl, dest_lbl_path)               # Menyalin berkas label .txt langsung flat

    n_all = len(pairs)
    p_train = (len(train_pairs) / n_all) * 100
    p_val = (len(val_pairs) / n_all) * 100
    p_test = (len(test_pairs) / n_all) * 100

    print(" RINGKASAN PEMBAGIAN DATASET:")
    print(f" [Train Set] : {len(train_pairs):4d} gambar ({p_train:.1f}%) -> {output_dir.as_posix()}/train")
    print(f" [Val Set]   : {len(val_pairs):4d} gambar ({p_val:.1f}%) -> {output_dir.as_posix()}/val")
    print(f" [Test Set]  : {len(test_pairs):4d} gambar ({p_test:.1f}%) -> {output_dir.as_posix()}/test")
    print("-" * 65)

    if use_group_split:
        print(" DISTRIBUSI GRUP/VIDEO PER SPLIT:")
        for s_name, s_list in splits_dict.items():
            g_counts = Counter([parse_group_id(img_p.name) for img_p, _ in s_list])  # Hitung frekuensi distribusi video
            print(f"  - {s_name.upper():5s} : {dict(g_counts)}")
    print("=" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser(                         # Buat parser argumen baris perintah terminal
        description="Pembagian dataset berpasangan format YOLO ke subset Train, Val, dan Test."
    )
    parser.add_argument("--images_dir", type=str, default="data/original/images", help="Direktori asal citra.")
    parser.add_argument("--labels_dir", type=str, default="data/original/labels", help="Direktori asal label.")
    parser.add_argument("--output_dir", type=str, default="data/splits", help="Direktori tujuan penyimpanan split.")
    parser.add_argument("--train_ratio", type=float, default=0.70, help="Rasio data training (default: 0.70).")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Rasio data validasi (default: 0.15).")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="Rasio data testing (default: 0.15).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed untuk reproducibility (default: 42).")
    parser.add_argument("--no_group", action="store_true", help="Matikan group-based split.")

    args = parser.parse_args()                                # Membaca argumen terminal ke objek args

    split_dataset(                                            # Eksekusi fungsi pembagian dataset utama
        images_dir=Path(args.images_dir),
        labels_dir=Path(args.labels_dir),
        output_dir=Path(args.output_dir),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        use_group_split=not args.no_group
    )


if __name__ == "__main__":                                     # Memastikan skrip berjalan hanya saat dipanggil langsung
    main()
