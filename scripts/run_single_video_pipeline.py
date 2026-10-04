#!/usr/bin/env python3
"""
Script: scripts/run_single_video_pipeline.py
Deskripsi: Master script otomatis untuk mengeksekusi pipeline dari 1 file video mentah
           hingga mendapatkan hasil penerangan citra Zero-DCE + CBAM dan anotasi YOLO.

Penggunaan:
    python scripts/run_single_video_pipeline.py --video data/raw/videos/video01.mp4
    python scripts/run_single_video_pipeline.py --video data/raw/videos/video01.mp4 --train --num_threads 4
"""

import argparse           # Pembaca argumen baris perintah terminal (--video, --train, dsb)
import subprocess         # Pemanggil sub-proses eksekusi script Python eksternal
import sys                # Pengendali eksekusi sistem & penutup eksekusi jika error (sys.exit)
from pathlib import Path  # Pengelola jalur folder/file lintas sistem operasi (Windows/Linux)

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # Lokasi akar proyek


def run_cmd(cmd_list, description):
    """Mengeksekusi perintah subprocess dan menghentikan alur jika terjadi kegagalan."""
    print("\n" + "=" * 65)
    print(f" [PIPELINE STEP] {description}")
    print("=" * 65)
    print(f" Executing: {' '.join(cmd_list)}")
    print("-" * 65)
    result = subprocess.run(cmd_list, cwd=str(PROJECT_ROOT))  # subprocess.run(): Jalankan perintah CLI
    if result.returncode != 0:                                # Memeriksa status exit code hasil eksekusi
        print(f"\n[ERROR] Tahap '{description}' gagal dengan exit code {result.returncode}.")
        sys.exit(result.returncode)                           # sys.exit(): Hentikan program jika gagal


def main():
    parser = argparse.ArgumentParser(                         # Buat penampung argumen terminal
        description="Master script otomatis pipeline end-to-end berdasar Alur Penelitian Roboflow."
    )
    parser.add_argument(                                       # Argumen --video (wajib)
        "--video",
        type=str,
        required=True,
        help="Path ke file video mentah (misal: data/raw/videos/video01.mp4)."
    )
    parser.add_argument(                                       # Argumen --model_yolo (default yolov8n.pt)
        "--model_yolo",
        type=str,
        default="yolov8n.pt",
        help="Model YOLO untuk auto-labeling (default: yolov8n.pt)."
    )
    parser.add_argument(                                       # Argumen --interval (default 1.0)
        "--interval",
        type=float,
        default=1.0,
        help="Interval ekstraksi frame dalam detik (default: 1.0)."
    )
    parser.add_argument(                                       # Flag --train (opsional)
        "--train",
        action="store_true",
        help="Jalankan juga proses retraining model Zero-DCE+CBAM & YOLO12 (opsional)."
    )
    parser.add_argument(                                       # Path checkpoint Zero-DCE+CBAM
        "--checkpoint",
        type=str,
        default="checkpoints/zero_dce_cbam/best_zero_dce_cbam.pth",
        help="Path checkpoint Zero-DCE+CBAM untuk enhancement generator."
    )
    parser.add_argument(                                       # Jumlah CPU threads PyTorch
        "--num_threads",
        type=int,
        default=4,
        help="Jumlah CPU threads PyTorch untuk mencegah laptop hang (default: 4)."
    )
    parser.add_argument(                                       # Flag --side_by_side (opsional)
        "--side_by_side",
        action="store_true",
        help="Hasilkan video perbandingan side-by-side (kiri: mentah, kanan: enhanced)."
    )
    parser.add_argument(                                       # Flag --skip_video_enhance (opsional)
        "--skip_video_enhance",
        action="store_true",
        help="Lewati pencerahan video full pada Tahap 1."
    )

    args = parser.parse_args()                                 # Membaca argumen terminal ke objek args

    video_path = Path(args.video)
    if not video_path.exists():                                # Memeriksa keberadaan file video di disk
        print(f"[ERROR] File video tidak ditemukan: {video_path}")
        sys.exit(1)

    video_stem = video_path.stem                               # Mengambil nama file video tanpa ekstensi

    print("\n" + "#" * 65)
    print(f" MEMULAI AUTOMATED ROBOFLOW PIPELINE UNTUK VIDEO: {video_path.name}")
    print("#" * 65)

    python_exe = sys.executable                                # Mengambil lokasi interpreter Python aktif

    # =========================================================================
    # TAHAP 1: EKSTRAKSI DATA & PREPROCESSING ENHANCEMENT (ZERO-DCE + CBAM)
    # =========================================================================
    tahap1_cmd = [python_exe, "scripts/run_tahap1.py", "--video", str(video_path), "--interval", str(args.interval), "--num_threads", str(args.num_threads)]
    if args.train:
        tahap1_cmd.append("--train")
    if args.side_by_side:
        tahap1_cmd.append("--side_by_side")
    if args.skip_video_enhance:
        tahap1_cmd.append("--skip_video_enhance")
    run_cmd(tahap1_cmd, "TAHAP 1: Ekstraksi Data & Preprocessing Enhancement Zero-DCE+CBAM")

    # =========================================================================
    # TAHAP 2: ANOTASI ROBOFLOW & PERSIAPAN DATASET
    # =========================================================================
    run_cmd(
        [python_exe, "scripts/run_tahap2.py", "--video_stem", video_stem, "--model_yolo", args.model_yolo],
        "TAHAP 2: Anotasi Roboflow & Persiapan Dataset (Split, Sync, Augmentasi, Validasi)"
    )

    # =========================================================================
    # TAHAP 3: PELATIHAN & EVALUASI DETEKSI OBJEK (YOLO12)
    # =========================================================================
    if args.train:
        run_cmd(
            [python_exe, "scripts/run_tahap3.py", "--video_stem", video_stem],
            "TAHAP 3: Pelatihan & Evaluasi Deteksi Objek YOLO12 (Skenario A vs B)"
        )
    else:
        run_cmd(
            [python_exe, "scripts/run_tahap3.py", "--video_stem", video_stem, "--only_eval"],
            "TAHAP 3: Evaluasi Metrik Deteksi Objek YOLO12 (Skenario A vs B)"
        )

    # =========================================================================
    # TAHAP 4: COUNTING, ESTIMASI POPULASI & KLASIFIKASI PERTUMBUHAN
    # =========================================================================
    run_cmd(
        [python_exe, "scripts/run_tahap4.py", "--video_stem", video_stem],
        "TAHAP 4: Perhitungan Objek, Error MAE/RMSE & Estimasi Populasi Walet"
    )

    print("\n" + "#" * 65)
    print(f" MASTER ROBOFLOW PIPELINE SELESAI [SUKSES] UNTUK VIDEO: {video_path.name}")
    print("#" * 65 + "\n")


if __name__ == "__main__":                                     # Memastikan skrip berjalan hanya saat dipanggil langsung
    main()

