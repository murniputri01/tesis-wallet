#!/usr/bin/env python3
"""
Script: scripts/augment_dataset.py
Deskripsi: Modul Augmentasi khusus Data Latih (Train Set) untuk citra dan label YOLO (.txt).
           Mendukung transformasi: Horizontal Flip, Random Brightness/Contrast, dan Blur ringan.
           Memastikan koordinat bounding box YOLO tetap 100% presisi setelah augmentasi.

Penggunaan:
    python scripts/augment_dataset.py --images_dir data/splits/train/images --labels_dir data/splits/train/labels --output_dir data/splits/train_augmented --num_aug 2
"""

import argparse               # Pembaca argumen baris perintah terminal (--images_dir, --num_aug, dsb)
import random                 # Pengacak statistik nilai parameter augmentasi (alpha, beta, ksize)
import sys                    # Pengendali eksekusi sistem & penambahan jalur direktori sys.path
from pathlib import Path      # Pengelola jalur folder/file lintas sistem operasi (Windows/Linux)
from typing import List, Tuple # Penentu tipe data statis tuple & list

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))  # Menambahkan root direktori proyek ke sys.path

import cv2                    # OpenCV untuk pemrosesan transformasi citra (flip, convertScaleAbs, GaussianBlur)
import numpy as np            # NumPy untuk komputasi larik matriks piksel


def flip_h_yolo_labels(labels: List[Tuple[int, float, float, float, float]]) -> List[Tuple[int, float, float, float, float]]:
    """Membalik koordinat YOLO secara horizontal (x_center -> 1.0 - x_center)."""
    aug_labels = []
    for cls_id, xc, yc, w, h in labels:
        new_xc = round(1.0 - xc, 6)   # Membalik koordinat x_center ter-normalisasi (1.0 - x)
        aug_labels.append((cls_id, new_xc, yc, w, h))
    return aug_labels


def read_yolo_file(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    labels = []
    if not label_path.exists():      # Memeriksa apakah file label .txt ada di disk
        return labels
    # read_text(): Membaca seluruh baris string dari file label .txt
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()  # Memotong string baris berdasarkan spasi
        if len(parts) >= 5:
            cls_id = int(parts[0])    # Mengambil class_id
            xc, yc, w, h = (float(v) for v in parts[1:5])  # Mengekstrak koordinat bbox [x, y, w, h]
            labels.append((cls_id, xc, yc, w, h))
    return labels


def save_yolo_file(label_path: Path, labels: List[Tuple[int, float, float, float, float]]):
    label_path.parent.mkdir(parents=True, exist_ok=True)  # Membuat folder induk label jika belum ada
    lines = [f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}" for cls_id, xc, yc, w, h in labels]
    # write_text(): Menuliskan baris-baris koordinat ke file .txt dengan encoding UTF-8
    label_path.write_text("\n".join(lines), encoding="utf-8")


def augment_image_and_labels(
    img: np.ndarray,
    labels: List[Tuple[int, float, float, float, float]],
    aug_type: str
) -> Tuple[np.ndarray, List[Tuple[int, float, float, float, float]]]:
    """Menerapkan augmentasi pada citra dan memperbarui label YOLO."""
    aug_img = img.copy()              # Salin matriks piksel citra asli
    aug_labels = list(labels)

    if aug_type == "flip_h":
        aug_img = cv2.flip(img, 1)    # cv2.flip(1): Membalik gambar secara horizontal (kiri-kanan)
        aug_labels = flip_h_yolo_labels(labels)  # Update koordinat x_center bbox YOLO
    elif aug_type == "brightness":
        alpha = random.uniform(0.85, 1.15)       # Mengacak faktor pengali kontras (gain)
        beta = random.randint(-15, 15)           # Mengacak faktor pergeseran kecerahan (bias)
        aug_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)  # Mengubah kecerahan & kontras piksel
    elif aug_type == "blur":
        ksize = random.choice([3, 5])            # Pilih ukuran kernel blur (3x3 atau 5x5)
        aug_img = cv2.GaussianBlur(img, (ksize, ksize), 0)  # cv2.GaussianBlur(): Efek kekaburan halus

    return aug_img, aug_labels


def parse_group_id(filename: str) -> str:
    """Meng-ekstrak ID grup/sumber video dari nama file (misal: 'video01' dari 'video01_frame_000001.jpg')."""
    if "_frame_" in filename:
        return filename.split("_frame_")[0]      # Memotong string nama file berdasarkan '_frame_'
    return "default_group"


def run_augmentation(
    images_dir: Path,
    labels_dir: Path,
    output_dir: Path,
    num_aug: int = 2
):
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    # rglob(): Mencari seluruh file gambar di direktori train secara rekursif
    img_paths = sorted([p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in valid_exts])

    if not img_paths:
        print(f"[WARNING] Tidak ada gambar ditemukan di '{images_dir.as_posix()}'")
        return

    label_map = {                     # Buat peta lokasi file label .txt berbasis nama stem
        p.stem: p for p in labels_dir.rglob("*.txt")
        if p.is_file()
    }

    out_img_dir = output_dir / "images"
    out_lbl_dir = output_dir / "labels"
    out_img_dir.mkdir(parents=True, exist_ok=True)  # Membuat direktori output images
    out_lbl_dir.mkdir(parents=True, exist_ok=True)  # Membuat direktori output labels

    print("\n" + "=" * 65)
    print(" MEMULAI MODUL AUGMENTASI DATA LATIH (TRAIN SET)")
    print("=" * 65)
    print(f" Input Images  : {len(img_paths)} file")
    print(f" Output Dir    : {output_dir.as_posix()}")
    print(f" Aug Factor    : {num_aug}x variasi per citra")
    print("-" * 65)

    saved_count = 0
    aug_types = ["flip_h", "brightness", "blur"]

    for img_p in img_paths:
        img = cv2.imread(str(img_p))  # cv2.imread(): Membaca gambar dari disk
        if img is None:
            continue

        grp_id = parse_group_id(img_p.name)
        sub_out_img_dir = out_img_dir / grp_id
        sub_out_lbl_dir = out_lbl_dir / grp_id
        sub_out_img_dir.mkdir(parents=True, exist_ok=True)  # Buat sub-folder gambar per grup
        sub_out_lbl_dir.mkdir(parents=True, exist_ok=True)  # Buat sub-folder label per grup

        lbl_p = label_map.get(img_p.stem, labels_dir / f"{img_p.stem}.txt")
        labels = read_yolo_file(lbl_p) # Membaca file koordinat label .txt

        # 1. Simpan gambar asli
        cv2.imwrite(str(sub_out_img_dir / img_p.name), img)  # cv2.imwrite(): Simpan citra asli
        save_yolo_file(sub_out_lbl_dir / f"{img_p.stem}.txt", labels)
        saved_count += 1

        # 2. Hasilkan versi augmentasi
        for i in range(num_aug):
            aug_mode = aug_types[i % len(aug_types)]
            aug_img, aug_lbls = augment_image_and_labels(img, labels, aug_mode)  # Terapkan transformasi

            out_stem = f"{img_p.stem}_aug_{aug_mode}_{i+1}"
            out_img_p = sub_out_img_dir / f"{out_stem}{img_p.suffix}"
            out_lbl_p = sub_out_lbl_dir / f"{out_stem}.txt"

            cv2.imwrite(str(out_img_p), aug_img)      # cv2.imwrite(): Simpan citra ter-augmentasi
            save_yolo_file(out_lbl_p, aug_lbls)       # Simpan file label ter-augmentasi
            saved_count += 1

    print(f" [SUKSES] Augmentasi Selesai! Total file dihasilkan: {saved_count} (citra + label)")
    print("=" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Modul Augmentasi Data Latih YOLO.")  # Buat parser terminal
    parser.add_argument("--images_dir", type=str, required=True, help="Folder gambar train.")
    parser.add_argument("--labels_dir", type=str, required=True, help="Folder label train.")
    parser.add_argument("--output_dir", type=str, required=True, help="Folder output dataset ter-augmentasi.")
    parser.add_argument("--num_aug", type=int, default=2, help="Jumlah variasi augmentasi per citra.")

    args = parser.parse_args()        # Mengekstrak argumen terminal ke objek args

    run_augmentation(                 # Eksekusi modul augmentasi utama
        images_dir=Path(args.images_dir),
        labels_dir=Path(args.labels_dir),
        output_dir=Path(args.output_dir),
        num_aug=args.num_aug
    )


if __name__ == "__main__":             # Memastikan skrip berjalan hanya saat dipanggil langsung
    main()
