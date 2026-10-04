#!/usr/bin/env python3
"""
Script : scripts/run_tahap4.py
Tahap  : 4 — PERHITUNGAN OBJEK (COUNTING), ESTIMASI POPULASI & KLASIFIKASI PERTUMBUHAN

Menjalankan seluruh proses counting, estimasi populasi, dan pembuatan video visualisasi:
  1. Perhitungan Jumlah Objek Presisi (Prediksi vs Ground Truth per Frame)
  2. Evaluasi Error Counting (MAE & RMSE)
  3. Perbandingan Skenario A (Original) vs Skenario B (Zero-DCE + CBAM Usulan)
  4. Estimasi Laju Pertumbuhan Populasi (Growth Rate %) & Klasifikasi Pertumbuhan
  5. Rendering Video Deteksi Bounding Box + Telemetry Real-Time Counter (H.264 mp4)

Penggunaan (Pilih Video / Master Dataset):
    # Evaluasi Master Dataset (Akurasi Presisi Tinggi --conf 0.05)
    python scripts/run_tahap4.py --use_master --conf 0.05

    # Evaluasi Video Spesifik (misal: 5_agustus atau video01)
    python scripts/run_tahap4.py --video_stem 5_agustus --conf 0.05

    # Generasi Video Counter Overlaid Side-by-Side (A vs B)
    python scripts/run_tahap4.py --video_stem 5_agustus --render_video --side_by_side --conf 0.05
"""

import argparse           # Pembaca argumen baris perintah terminal (--video_stem, --conf, dsb)
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


def count_files(folder: Path, suffixes, video_stem: str = "", use_master: bool = False):
    if not folder.exists():
        return 0
    files = [
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in suffixes
    ]
    if use_master or not video_stem:
        return len(files)
    return len([
        p for p in files
        if p.name.startswith(f"{video_stem}_") or video_stem.lower() in [part.lower() for part in p.parts]
    ])


def run_cmd(cmd_list, description):
    """Mengeksekusi perintah subprocess dan menghentikan alur jika terjadi kegagalan."""
    print("\n" + "=" * 70)
    print(f" [TAHAP 4] {description}")
    print("=" * 70)
    print(f" Executing: {' '.join(cmd_list)}")
    print("-" * 70)
    result = subprocess.run(cmd_list, cwd=str(PROJECT_ROOT))  # subprocess.run(): Jalankan perintah CLI
    if result.returncode != 0:                                # Memeriksa status exit code hasil eksekusi
        print_friendly_error(
            f"LANGKAH TAHAP 4 GAGAL: {description}",
            [
                f"Subprocess berhenti dengan exit code {result.returncode}.",
                "Penyebab umum: checkpoint YOLO belum ada, dataset test/label kosong, video render tidak ditemukan, atau dependency video codec bermasalah.",
            ],
            [
                "Baca pesan teknis tepat di atas blok ini untuk detail dari script yang gagal.",
                "Pastikan Tahap 3 sudah selesai dan menghasilkan checkpoint best.pt.",
                "Pastikan Tahap 2 sudah membuat data/splits atau master dataset yang tervalidasi.",
                "Jika gagal render video, cek --video_path atau gunakan tanpa --render_video.",
            ],
        )
        sys.exit(result.returncode)                           # sys.exit(): Hentikan program jika gagal


def main():
    parser = argparse.ArgumentParser(                         # Buat penampung argumen terminal
        description="Tahap 4: Perhitungan Objek (Counting), Estimasi Populasi & Klasifikasi Pertumbuhan."
    )
    parser.add_argument(                                       # Argumen --video_stem (opsional)
        "--video_stem", type=str, default="",
        help="Nama/ID video sumber (opsional, misal: 'video01' atau '7_agustus'). Jika kosong, seluruh test set dievaluasi."
    )
    parser.add_argument(                                       # Path weights Skenario A
        "--weights_a", type=str,
        default="checkpoints/yolo_original/best.pt",
        help="Path weights YOLO Skenario A (default: checkpoints/yolo_original/best.pt)."
    )
    parser.add_argument(                                       # Path weights Skenario B
        "--weights_b", type=str,
        default="checkpoints/yolo_zero_dce_cbam/best.pt",
        help="Path weights YOLO Skenario B (default: checkpoints/yolo_zero_dce_cbam/best.pt)."
    )
    parser.add_argument(                                       # Confidence threshold deteksi & counting
        "--conf", type=float, default=0.35,
        help="Confidence threshold untuk deteksi & counting (default: 0.35 untuk akurasi presisi tinggi)."
    )
    parser.add_argument(                                       # Resolusi inferensi YOLO
        "--imgsz", type=int, default=640,
        help="Resolusi inferensi YOLO (default: 640)."
    )
    parser.add_argument(                                       # Direktori hasil analisis counting
        "--output_dir", type=str, default="results/detection",
        help="Direktori penyimpanan hasil analisis & grafik counting (default: results/detection)."
    )
    parser.add_argument(                                       # Path file video sumber (opsional untuk rendering video)
        "--video_path", type=str, default=None,
        help="Path ke file video mentah (.mp4) untuk pembuatan video visualisasi deteksi & real-time counter."
    )
    parser.add_argument(                                       # Flag rendering video
        "--render_video", action="store_true",
        help="Aktifkan pembuatan video visualisasi deteksi & real-time counter."
    )
    parser.add_argument(                                       # Flag side_by_side untuk video
        "--side_by_side", action="store_true",
        help="Buat video perbandingan Side-by-Side (Skenario A vs Skenario B)."
    )
    parser.add_argument(                                       # Batas frame untuk uji coba cepat
        "--frame_limit", type=int, default=None,
        help="Batasi jumlah frame video yang diproses (opsional)."
    )
    parser.add_argument(                                       # Flag Master Dataset (Opsi 2)
        "--use_master", action="store_true",
        help="Gunakan Master Dataset Multi-Video Opsi 2."
    )

    args = parser.parse_args()                                 # Membaca argumen terminal ke objek args
    python_exe = sys.executable                                # Mengambil jalur interpreter Python aktif

    ckpt_a = Path(args.weights_a)
    ckpt_b = Path(args.weights_b)

    # Cek ketersediaan checkpoint spesifik per-video jika ada
    if args.video_stem:
        v_ckpt_a = PROJECT_ROOT / "checkpoints" / "yolo_original" / args.video_stem / "best.pt"
        v_ckpt_b = PROJECT_ROOT / "checkpoints" / "yolo_zero_dce_cbam" / args.video_stem / "best.pt"
        if v_ckpt_a.exists():                                  # Gunakan checkpoint spesifik video A jika ada
            ckpt_a = v_ckpt_a
        if v_ckpt_b.exists():                                  # Gunakan checkpoint spesifik video B jika ada
            ckpt_b = v_ckpt_b

    print("\n" + "#" * 70)
    print(f" MEMULAI TAHAP 4: PERHITUNGAN OBJEK & ESTIMASI POPULASI")
    if args.video_stem:
        print(f" Video Stem : {args.video_stem}")
    print(f" Weights A  : {ckpt_a}")
    print(f" Weights B  : {ckpt_b}")
    print("#" * 70)

    if not ckpt_a.exists() and not ckpt_b.exists():            # Memeriksa ketersediaan minimal 1 checkpoint
        print_friendly_error(
            "CHECKPOINT YOLO TAHAP 4 TIDAK DITEMUKAN",
            [
                f"Checkpoint Skenario A tidak ditemukan: {ckpt_a}",
                f"Checkpoint Skenario B tidak ditemukan: {ckpt_b}",
                "Tahap 4 membutuhkan minimal satu model YOLO hasil training Tahap 3 untuk counting.",
            ],
            [
                f"Jalankan Tahap 3 terlebih dahulu: python scripts/run_tahap3.py --video_stem {args.video_stem or '<nama_video>'}",
                "Jika memakai master dataset, jalankan Tahap 3 dengan --use_master.",
                "Atau berikan path checkpoint manual melalui --weights_a dan/atau --weights_b.",
            ],
        )
        sys.exit(1)

    if not ckpt_a.exists() or not ckpt_b.exists():
        missing_name = "Skenario A" if not ckpt_a.exists() else "Skenario B"
        missing_path = ckpt_a if not ckpt_a.exists() else ckpt_b
        print(f"\n[WARNING] Checkpoint {missing_name} tidak ditemukan: {missing_path}")
        print("          Counting tetap berjalan dengan checkpoint yang tersedia, tetapi perbandingan A vs B tidak lengkap.")

    if args.use_master:
        eval_img_dir = PROJECT_ROOT / "data" / "master_dataset" / "test" / "images"
        eval_lbl_dir = PROJECT_ROOT / "data" / "master_dataset" / "test" / "labels"
        fallback_img_dir = PROJECT_ROOT / "data" / "master_dataset" / "val" / "images"
        fallback_lbl_dir = PROJECT_ROOT / "data" / "master_dataset" / "val" / "labels"
    else:
        eval_img_dir = PROJECT_ROOT / "data" / "splits" / "test" / "images"
        eval_lbl_dir = PROJECT_ROOT / "data" / "splits" / "test" / "labels"
        fallback_img_dir = PROJECT_ROOT / "data" / "splits" / "val" / "images"
        fallback_lbl_dir = PROJECT_ROOT / "data" / "splits" / "val" / "labels"

    eval_images = count_files(eval_img_dir, {".jpg", ".jpeg", ".png", ".bmp"}, args.video_stem, args.use_master)
    eval_labels = count_files(eval_lbl_dir, {".txt"}, args.video_stem, args.use_master)
    fallback_images = count_files(fallback_img_dir, {".jpg", ".jpeg", ".png", ".bmp"}, args.video_stem, args.use_master)
    fallback_labels = count_files(fallback_lbl_dir, {".txt"}, args.video_stem, args.use_master)

    if (eval_images == 0 or eval_labels == 0) and (fallback_images == 0 or fallback_labels == 0):
        print_friendly_error(
            "DATASET EVALUASI TAHAP 4 BELUM SIAP",
            [
                f"Split test: {eval_images} gambar, {eval_labels} label.",
                f"Split val fallback: {fallback_images} gambar, {fallback_labels} label.",
                "Tahap 4 membutuhkan data evaluasi berlabel untuk menghitung MAE/RMSE dan estimasi populasi.",
            ],
            [
                f"Jalankan Tahap 2 terlebih dahulu: python scripts/run_tahap2.py --video_stem {args.video_stem or '<nama_video>'}",
                "Pastikan validasi dataset YOLO pada Tahap 2 lulus.",
                "Jika memakai --use_master, pastikan master dataset sudah dibangun dan tidak kosong.",
            ],
        )
        sys.exit(1)

    # 1. Eksekusi Laporan Counting CSV & Grafik Populasi
    cmd = [
        python_exe, "detection/count_population.py",          # Eksekusi script counting & estimasi populasi
        "--weights_a", str(ckpt_a),
        "--weights_b", str(ckpt_b),
        "--output_dir", args.output_dir,
        "--conf", str(args.conf)
    ]
    if args.video_stem:
        cmd += ["--video_stem", args.video_stem]
    if args.use_master:
        cmd.append("--use_master")

    run_cmd(cmd, "Counting Objek per-Frame, Metrik MAE/RMSE, & Estimasi Laju Pertumbuhan Populasi")

    # 2. Eksekusi Generasi Video Visualisasi (jika --render_video atau --video_path diberikan)
    target_video = args.video_path
    if not target_video and args.video_stem:
        # Coba cari file video mentah di data/raw/videos/
        potential_videos = list((PROJECT_ROOT / "data" / "raw" / "videos").glob(f"{args.video_stem}.*"))
        if potential_videos:
            target_video = str(potential_videos[0])

    if target_video or args.render_video:
        if not target_video:
            print_friendly_error(
                "VIDEO SUMBER UNTUK RENDER TAHAP 4 TIDAK DITEMUKAN",
                [
                    "--render_video aktif, tetapi --video_path tidak diberikan dan file video tidak ditemukan otomatis.",
                    f"Video dicari berdasarkan stem: {args.video_stem or '(kosong)'}.",
                ],
                [
                    "Berikan path video secara eksplisit dengan --video_path data/raw/videos/<nama_video>.mp4.",
                    "Atau letakkan video mentah di data/raw/videos/ dengan nama yang sama seperti --video_stem.",
                    "Jika hanya butuh CSV/grafik counting, jalankan tanpa --render_video.",
                ],
            )
            sys.exit(1)
        else:
            target_video_path = Path(target_video)
            if not target_video_path.exists():
                print_friendly_error(
                    "FILE VIDEO RENDER TAHAP 4 TIDAK DITEMUKAN",
                    [
                        f"Path video yang diberikan tidak ada: {target_video_path}",
                        "Video overlay real-time counter membutuhkan file video mentah yang valid.",
                    ],
                    [
                        "Cek kembali argumen --video_path.",
                        "Letakkan video mentah di data/raw/videos/.",
                        "Jika tidak ingin membuat video overlay, hilangkan flag --render_video.",
                    ],
                )
                sys.exit(1)

            vid_cmd = [
                python_exe, "detection/video_counter_visualizer.py",
                "--video", target_video,
                "--weights_a", str(ckpt_a),
                "--weights_b", str(ckpt_b),
                "--conf", str(args.conf),
                "--imgsz", str(args.imgsz)
            ]
            if args.side_by_side:
                vid_cmd.append("--side_by_side")
            if args.frame_limit:
                vid_cmd += ["--frame_limit", str(args.frame_limit)]

            run_cmd(vid_cmd, "Generasi Video Visualisasi Deteksi & Real-Time Counter Walet (H.264 mp4)")

    print("\n" + "#" * 70)
    print(f" TAHAP 4 SELESAI [SUKSES] - {args.video_stem if args.video_stem else 'All Datasets'}")
    print("#" * 70)
    print(" Lokasi Output:")
    out_path = Path(args.output_dir) / args.video_stem if args.video_stem else Path(args.output_dir)
    print(f"  [FILE] Detail Counting CSV  : {out_path / 'per_frame_counting_detail.csv'}")
    print(f"  [FILE] Summary Metrik CSV   : {out_path / 'counting_error_metrics.csv'}")
    print(f"  [IMG]  Grafik Tren Populasi : {out_path / 'population_growth_trend.png'}")
    print(f"  [IMG]  Grafik Error Bar     : {out_path / 'counting_error_comparison.png'}")
    if target_video or args.render_video:
        print(f"  [VIDEO] Video Counter Overlay : {out_path / 'video'}")
    print("#" * 70 + "\n")


if __name__ == "__main__":                                     # Memastikan skrip berjalan hanya saat dipanggil langsung
    main()
