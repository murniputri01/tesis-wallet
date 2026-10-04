#!/usr/bin/env python3
"""
Script : scripts/run_tahap2.py
Tahap  : 2 — ANOTASI ROBOFLOW & PERSIAPAN DATASET
Alur   : Roboflow Research System Workflow (alur_sistem_penelitian_roboflow.png)

Menjalankan seluruh persiapan dataset setelah anotasi citra terang dari Roboflow secara berurutan:
  1. Standardisasi Label Roboflow (Konversi Format Polygon ke Bounding Box 5-kolom YOLO)
  2. Pembangunan Dataset Bersih 70 : 15 : 15 (Deduplikasi Frame & Anti-Leakage)
  3. Validasi Konsistensi Format Dataset YOLO
  4. Sinkronisasi Label Anotasi ke Citra Enhanced (Skenario B)
  5. Pembangunan Master Dataset Multi-Video (Skenario A & B)
  6. Validasi Konsistensi Format Master Dataset YOLO
  7. Augmentasi Data Latih (2 varian per citra train)

Penggunaan:
    python scripts/run_tahap2.py --video_stem video01
"""

import argparse           # Pembaca argumen baris perintah terminal (--video_stem, dsb)
import subprocess         # Pemanggil sub-proses eksekusi script Python eksternal
import sys                # Pengendali eksekusi sistem & penutup eksekusi jika error (sys.exit)
from pathlib import Path  # Pengelola jalur folder/file lintas sistem operasi (Windows/Linux)

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # Lokasi akar proyek


def print_friendly_error(title: str, cause_lines, fix_lines):
    """Menampilkan error dengan penyebab dan langkah perbaikan yang jelas."""
    print("\n" + "!" * 75)
    print(f" [ERROR] {title}")
    print("!" * 75)
    print("\n PENYEBAB:")
    for line in cause_lines:
        print(f"  - {line}")
    print("\n CARA MENGATASI:")
    for line in fix_lines:
        print(f"  - {line}")
    print("!" * 75 + "\n")


def find_stem_files(base_dir: Path, video_stem: str, suffixes):
    """Mencari file video_stem pada struktur folder nested atau flat."""
    nested_dir = base_dir / video_stem
    if nested_dir.exists():
        return [
            p for p in nested_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in suffixes
        ]

    return [
        p for p in base_dir.rglob("*")
        if (
            p.is_file()
            and p.suffix.lower() in suffixes
            and (p.name.startswith(f"{video_stem}_") or p.stem == video_stem)
        )
    ]


def run_cmd(cmd_list, description):
    """Mengeksekusi perintah subprocess dan menghentikan alur jika terjadi kegagalan."""
    print("\n" + "=" * 65)
    print(f" [TAHAP 2] {description}")
    print("=" * 65)
    print(f" Executing: {' '.join(cmd_list)}")
    print("-" * 65)
    result = subprocess.run(cmd_list, cwd=str(PROJECT_ROOT))  # subprocess.run(): Jalankan perintah CLI
    if result.returncode != 0:                                # Memeriksa status exit code hasil eksekusi
        print(f"\n[ERROR] Langkah '{description}' gagal dengan exit code {result.returncode}.")
        sys.exit(result.returncode)                           # sys.exit(): Hentikan program jika gagal


def main():
    parser = argparse.ArgumentParser(                         # Buat penampung argumen terminal
        description="Tahap 2: Anotasi Roboflow & Persiapan Dataset (clean split, validasi, sync, master, augmentasi)."
    )
    parser.add_argument(                                       # Argumen --video_stem (wajib)
        "--video_stem", type=str, required=True,
        help="Nama/ID video sumber tanpa ekstensi (misal: 'video01' atau '7_agustus')."
    )
    parser.add_argument(                                       # Argumen --num_aug (default 2)
        "--num_aug", type=int, default=2,
        help="Jumlah varian augmentasi per citra latih (default: 2)."
    )
    parser.add_argument(
        "--model_yolo", type=str, default=None,
        help="Kompatibilitas lama; diabaikan karena Tahap 2 memakai anotasi Roboflow/manual."
    )
    parser.add_argument(
        "--min_objects", type=int, default=0,
        help="Buang frame dengan jumlah objek kurang dari nilai ini saat build dataset bersih (default: 0)."
    )
    parser.add_argument(
        "--keep_empty", action="store_true",
        help="Pertahankan frame tanpa objek teranotasi saat build dataset bersih."
    )
    parser.add_argument(
        "--skip_master", action="store_true",
        help="Lewati pembangunan master dataset multi-video."
    )

    args = parser.parse_args()                                 # Membaca argumen terminal ke objek args
    video_stem = args.video_stem
    python_exe = sys.executable                                # Mengambil lokasi interpreter Python aktif

    print("\n" + "#" * 65)
    print(f" MEMULAI TAHAP 2: ANOTASI ROBOFLOW & PERSIAPAN DATASET")
    print(f" Video Stem : {video_stem}")
    print("#" * 65)

    step = 0

    orig_img_root = Path("data/original/images")
    orig_lbl_root = Path("data/original/labels")
    orig_img_dir = orig_img_root / video_stem
    orig_lbl_dir = orig_lbl_root / video_stem
    source_img_dir = orig_img_dir if orig_img_dir.exists() else orig_img_root
    source_lbl_dir = orig_lbl_dir if orig_lbl_dir.exists() else orig_lbl_root

    enhanced_img_root = Path("data/enhanced/zero_dce_cbam")
    img_files_for_stem = find_stem_files(orig_img_root, video_stem, {".jpg", ".jpeg", ".png", ".bmp"})
    enhanced_img_files_for_stem = find_stem_files(enhanced_img_root, video_stem, {".jpg", ".jpeg", ".png", ".bmp"})
    lbl_files_for_stem = find_stem_files(orig_lbl_root, video_stem, {".txt"})

    if not lbl_files_for_stem:
        print_friendly_error(
            f"BELUM ADA ANOTASI/LABEL ROBOFLOW UNTUK '{video_stem}'",
            [
                f"Ditemukan {len(img_files_for_stem)} gambar original dan {len(enhanced_img_files_for_stem)} gambar enhanced untuk '{video_stem}', tetapi tidak ada file label .txt yang cocok.",
                f"Contoh kasus: video '{video_stem}' sudah diproses, tetapi belum di-upload dan dianotasi di Roboflow.",
                "Tanpa label Roboflow, dataset bersih dan training YOLO tidak punya ground truth yang benar.",
            ],
            [
                "Upload citra terang/enhanced ke Roboflow lalu anotasi objek walet.",
                "Export dataset Roboflow dalam format YOLOv8/YOLOv12.",
                f"Pastikan gambar hasil export/citra original tersedia di data/original/images/{video_stem}/.",
                f"Letakkan label di data/original/labels/{video_stem}/ atau gunakan nama file {video_stem}_frame_*.txt.",
                f"Jalankan ulang: python scripts/run_tahap2.py --video_stem {video_stem}",
            ],
        )
        sys.exit(1)

    if not img_files_for_stem:
        print_friendly_error(
            f"CITRA ORIGINAL UNTUK '{video_stem}' BELUM TERSEDIA",
            [
                f"Label untuk '{video_stem}' ditemukan, tetapi gambar original belum ada di data/original/images.",
                "Tahap 2 membutuhkan pasangan gambar dan label dengan nama stem yang sama.",
            ],
            [
                f"Pastikan gambar tersedia di data/original/images/{video_stem}/ atau bernama {video_stem}_frame_*.jpg.",
                "Jika gambar baru ada di folder enhanced, salin/export pasangan gambar yang sesuai ke data/original/images sebelum menjalankan Tahap 2.",
            ],
        )
        sys.exit(1)

    # 1. Standardisasi Format Label Roboflow (Konversi Polygon Segmen Roboflow -> YOLO Bounding Box)
    step += 1
    run_cmd(
        [python_exe, "scripts/dataset/convert_polygon_to_bbox.py",
         "--labels_dir", str(source_lbl_dir)],
        f"{step}. Konversi Format Label Polygon (Roboflow Segment) ke YOLO Bounding Box"
    )

    # 2. Pembangunan Dataset Bersih 70:15:15 (Deduplikasi Frame & Anti-Leakage)
    step += 1
    clean_cmd = [
        python_exe, "scripts/dataset/build_clean_dataset.py",
        "--images_dir", str(source_img_dir),
        "--labels_dir", str(source_lbl_dir),
        "--output_dir", "data/splits",
        "--sources", video_stem,
        "--min_objects", str(args.min_objects),
    ]
    if args.keep_empty:
        clean_cmd.append("--keep_empty")
    run_cmd(
        clean_cmd,
        f"{step}. Build Dataset Bersih 70:15:15 ({video_stem})"
    )

    # 3. Validasi Konsistensi Dataset YOLO setelah pembersihan
    step += 1
    run_cmd(
        [python_exe, "scripts/dataset/validate_dataset.py",
         "--splits_dir", "data/splits"],
        f"{step}. Validasi Konsistensi Format Dataset YOLO"
    )

    # 4. Sinkronisasi Label Anotasi ke Direktori Enhanced (Skenario B)
    step += 1
    run_cmd(
        [python_exe, "detection/sync_labels.py"],                # Eksekusi script sinkronisasi label .txt
        f"{step}. Sinkronisasi Label Anotasi ke Skenario A & B"
    )

    # 5. Pembangunan Master Dataset Multi-Video (Skenario A & B)
    step += 1
    if not args.skip_master:
        run_cmd(
            [python_exe, "scripts/dataset/build_master_dataset.py", "--overwrite"],
            f"{step}. Build Master Dataset Multi-Video"
        )
    else:
        print(f"\n[INFO] Langkah {step}. Build Master Dataset dilewati (--skip_master).")

    # 6. Validasi Konsistensi Master Dataset YOLO
    step += 1
    if not args.skip_master:
        run_cmd(
            [python_exe, "scripts/dataset/validate_dataset.py",
             "--splits_dir", "data/master_dataset"],
            f"{step}. Validasi Master Dataset Original"
        )
        run_cmd(
            [python_exe, "scripts/dataset/validate_dataset.py",
             "--splits_dir", "data/enhanced/master_dataset"],
            f"{step}. Validasi Master Dataset Enhanced"
        )
    else:
        print(f"\n[INFO] Langkah {step}. Validasi Master Dataset dilewati (--skip_master).")

    # 7. Augmentasi Data Latih (Train Set Only)
    step += 1
    train_img_dir = Path("data/splits/train/images")
    train_lbl_dir = Path("data/splits/train/labels")
    if train_img_dir.exists():
        run_cmd(
            [python_exe, "scripts/dataset/augment_dataset.py",  # Eksekusi script augmentasi data latih
             "--images_dir", str(train_img_dir),
             "--labels_dir", str(train_lbl_dir),
             "--output_dir", "data/splits/train_augmented",
             "--num_aug", str(args.num_aug)],
            f"{step}. Augmentasi Data Latih ({video_stem})"
        )
    else:
        print(f"\n[WARNING] Folder train tidak ditemukan: {train_img_dir}. Langkah augmentasi dilewati.")

    print("\n" + "#" * 65)
    print(f" TAHAP 2 SELESAI [SUKSES] - {video_stem}")
    print("#" * 65)
    print(" Lokasi Output:")
    print(f"  [DIR] Citra + Label     : data/original/images/{video_stem}/")
    print(f"                            data/original/labels/{video_stem}/")
    print(f"  [DIR] Dataset Splits    : data/splits/  (train / val / test)")
    print(f"  [DIR] Enhanced Splits   : data/enhanced/zero_dce_cbam/  (train / val / test)")
    print(f"  [DIR] Data Augmentasi   : data/splits/train_augmented/")
    print()
    print(" Langkah Berikutnya (Tahap 3 - Training Deteksi YOLO12):")
    print(f"  -> Jalankan Tahap 3:")
    print(f"     python scripts/run_tahap3.py --video_stem {video_stem}")
    print("#" * 65 + "\n")


if __name__ == "__main__":                                     # Memastikan skrip berjalan hanya saat dipanggil langsung
    main()
