#!/usr/bin/env python3
"""
Script: validate_dataset.py
Deskripsi: Memvalidasi konsistensi antara file gambar dan anotasi label format YOLO.
           Memeriksa ketersediaan file, nilai class_id (harus 0), rentang koordinat [0.0, 1.0],
           serta batas bounding box agar tidak keluar dari citra.

Penggunaan:
    python scripts/validate_dataset.py --images_dir data/original/images --labels_dir data/original/labels
    python scripts/validate_dataset.py --splits_dir data/splits
"""

import argparse               # Pembaca argumen baris perintah terminal (--images_dir, --splits_dir)
import sys                    # Pengendali eksekusi sistem & pemutus skrip (sys.exit)
from pathlib import Path      # Pengelola jalur folder/file lintas sistem operasi (Windows/Linux)


def validate_yolo_label(label_path: Path, max_class_id: int = 0):
    """
    Memvalidasi isi file label format YOLO.
    Format per baris: <class_id> <x_center> <y_center> <width> <height>
    """
    errors = []
    num_annotations = 0

    if not label_path.exists():  # Memeriksa apakah file label .txt ada di disk
        return False, 0, [f"File label tidak ditemukan: {label_path.name}"]

    try:
        with open(label_path, "r", encoding="utf-8") as f:  # Membaca seluruh isi baris file label .txt
            lines = [line.strip() for line in f.readlines() if line.strip()]

        if len(lines) == 0:     # File label kosong diizinkan (background image tanpa objek)
            return True, 0, []

        for line_num, line in enumerate(lines, start=1):  # Loop membaca setiap baris anotasi bbox
            tokens = line.split()  # Memotong string baris berdasarkan spasi
            if len(tokens) != 5:  # Memeriksa kelengkapan 5 komponen YOLO (class x y w h)
                errors.append(f"Baris {line_num}: Harus berisi 5 nilai (class x y w h), ditemukan {len(tokens)} nilai.")
                continue

            try:
                class_id = int(tokens[0])       # Konversi class_id ke integer
                x_center = float(tokens[1])     # Konversi x_center ke float
                y_center = float(tokens[2])     # Konversi y_center ke float
                width = float(tokens[3])        # Konversi width ke float
                height = float(tokens[4])       # Konversi height ke float
            except ValueError:
                errors.append(f"Baris {line_num}: Nilai token bukan angka valid ('{line}').")
                continue

            if class_id != max_class_id:        # Validasi class_id wajib bernilai 0 (walet)
                errors.append(f"Baris {line_num}: class_id={class_id} tidak valid. Hanya kelas 0 (walet) yang diizinkan.")

            if not (0.0 <= x_center <= 1.0):   # Validasi batas x_center ter-normalisasi [0, 1]
                errors.append(f"Baris {line_num}: x_center={x_center} di luar rentang [0, 1].")
            if not (0.0 <= y_center <= 1.0):   # Validasi batas y_center ter-normalisasi [0, 1]
                errors.append(f"Baris {line_num}: y_center={y_center} di luar rentang [0, 1].")
            if not (0.0 < width <= 1.0):       # Validasi batas lebar width ter-normalisasi (0, 1]
                errors.append(f"Baris {line_num}: width={width} harus di rentang (0, 1].")
            if not (0.0 < height <= 1.0):      # Validasi batas tinggi height ter-normalisasi (0, 1]
                errors.append(f"Baris {line_num}: height={height} harus di rentang (0, 1].")

            x_min = x_center - (width / 2.0)    # Hitung tepi kiri x_min bounding box
            x_max = x_center + (width / 2.0)    # Hitung tepi kanan x_max bounding box
            y_min = y_center - (height / 2.0)   # Hitung tepi atas y_min bounding box
            y_max = y_center + (height / 2.0)   # Hitung tepi bawah y_max bounding box

            eps = 1e-4                          # Toleransi kecil 1e-4 untuk pembulatan float
            if x_min < -eps or x_max > 1.0 + eps or y_min < -eps or y_max > 1.0 + eps:  # Cek batas citra
                errors.append(
                    f"Baris {line_num}: Bounding box melampaui batas citra "
                    f"(x_min={x_min:.4f}, x_max={x_max:.4f}, y_min={y_min:.4f}, y_max={y_max:.4f})."
                )

            num_annotations += 1

    except Exception as e:
        errors.append(f"Gagal membaca file label: {str(e)}")

    is_valid = len(errors) == 0  # Status validasi True jika tidak ada error sama sekali
    return is_valid, num_annotations, errors


def validate_dataset_folder(images_dir: Path, labels_dir: Path):
    """Memverifikasi direktori pasangan gambar dan label."""
    if not images_dir.exists():  # Memeriksa keberadaan folder gambar
        print(f"[ERROR] Direktori gambar tidak ditemukan: {images_dir}")
        return False

    valid_image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    image_files = {              # Cari seluruh file gambar di folder images
        p.stem: p for p in images_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in valid_image_exts
    }

    label_files = {}
    if labels_dir.exists():      # Cari seluruh file label .txt di folder labels
        label_files = {
            p.stem: p for p in labels_dir.rglob("*")
            if p.is_file() and p.suffix.lower() == ".txt"
        }

    all_stems = sorted(list(set(image_files.keys()) | set(label_files.keys())))  # Gabungkan seluruh nama stem

    total_images = len(image_files)
    images_with_labels = 0
    missing_labels = 0
    missing_images = 0
    invalid_labels = 0
    total_annotations = 0
    invalid_detail_list = []

    for stem in all_stems:       # Loop memeriksa keterhubungan pasangan citra & label
        has_img = stem in image_files
        has_lbl = stem in label_files

        if has_img and not has_lbl:
            missing_labels += 1  # Gambar tidak memiliki file label
            invalid_detail_list.append(f"[MISSING LABEL] Image '{image_files[stem].name}' tidak memiliki file label.")
        elif has_lbl and not has_img:
            missing_images += 1  # Label tidak memiliki gambar pasangannya
            invalid_detail_list.append(f"[ORPHAN LABEL] Label '{label_files[stem].name}' tidak memiliki gambar pasangannya.")
        else:
            images_with_labels += 1
            lbl_path = label_files[stem]
            is_valid, num_ann, errors = validate_yolo_label(lbl_path, max_class_id=0)  # Validasi isi file label .txt
            total_annotations += num_ann

            if not is_valid:
                invalid_labels += 1
                for err in errors:
                    invalid_detail_list.append(f"[INVALID LABEL] {lbl_path.name} -> {err}")

    print("\n" + "=" * 55)
    print(f" HASIL VALIDASI DATASET: {images_dir.as_posix()}")
    print("=" * 55)
    print(f" Total images       : {total_images}")
    print(f" Images with labels : {images_with_labels}")
    print(f" Missing labels     : {missing_labels}")
    print(f" Orphan labels      : {missing_images}")
    print(f" Invalid labels     : {invalid_labels}")
    print(f" Total annotations  : {total_annotations}")
    print(f" Total classes      : 1 (0: walet)")
    print("=" * 55)

    if invalid_detail_list:
        print("\n DETAIL MASALAH YANG DITEMUKAN:")
        for detail in invalid_detail_list[:15]:  # Tampilkan 15 rincian kesalahan pertama
            print(f"  - {detail}")
        if len(invalid_detail_list) > 15:
            print(f"  ... dan {len(invalid_detail_list) - 15} masalah lainnya.")
        print("-" * 55 + "\n")
    else:
        print("\n [STATUS] DATASET 100% VALID & KONSISTEN FORMAT YOLO!\n")

    return (missing_labels == 0 and invalid_labels == 0 and missing_images == 0)


def main():
    parser = argparse.ArgumentParser(          # Buat parser argumen baris perintah terminal
        description="Validasi konsistensi citra dan label format YOLO."
    )
    parser.add_argument("--images_dir", type=str, default="data/original/images", help="Direktori gambar.")
    parser.add_argument("--labels_dir", type=str, default="data/original/labels", help="Direktori label.")
    parser.add_argument("--splits_dir", type=str, default=None, help="Direktori splits (train, val, test).")

    args = parser.parse_args()                 # Mengekstrak argumen terminal ke objek args

    if args.splits_dir:
        splits_path = Path(args.splits_dir)
        sub_splits = ["train", "val", "test"]
        all_passed = True
        for sub in sub_splits:                 # Memvalidasi subset train, val, dan test secara berurutan
            img_p = splits_path / sub / "images"
            lbl_p = splits_path / sub / "labels"
            passed = validate_dataset_folder(img_p, lbl_p)
            if not passed:
                all_passed = False
        sys.exit(0 if all_passed else 1)      # Hentikan eksekusi dengan exit code 0 (sukses) atau 1 (gagal)
    else:
        passed = validate_dataset_folder(Path(args.images_dir), Path(args.labels_dir))
        sys.exit(0 if passed else 1)          # Hentikan eksekusi sesuai hasil validasi


if __name__ == "__main__":                     # Memastikan skrip berjalan hanya saat dipanggil langsung
    main()
