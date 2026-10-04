#!/usr/bin/env python3
"""
Script : scripts/run_tahap1.py
Tahap  : 1 — EKSTRAKSI DATA & PREPROCESSING ENHANCEMENT (ZERO-DCE + CBAM)
Alur   : Roboflow Research System Workflow (alur_sistem_penelitian_roboflow.png)

Menjalankan seluruh proses ekstraksi dan pencerahan citra low-light secara berurutan:
  1. Ekstraksi Frame dari Video (1 frame/detik)
  2. Seleksi Frame / Eliminasi Redundansi (SSIM ≥ 0.95)
  3. (Opsional) Pelatihan Model Zero-DCE + CBAM (flag --train)
  4. Batch Enhancement Generator — Menerangkan citra low-light terpilih
  5. Evaluasi Kualitas Citra (Entropy, Contrast, BRISQUE, Koefisien Variasi Iluminasi)
  6. Pembaruan Dashboard Kualitas Citra Gabungan (results/enhancement/quality_metrics/)

Penggunaan:
    python scripts/run_tahap1.py --video data/raw/videos/video01.mp4
    python scripts/run_tahap1.py --video data/raw/videos/video01.mp4 --train --num_threads 4
"""

import argparse           # Pembaca argumen baris perintah terminal (--video, --train, dsb)
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


def count_files(folder: Path, suffixes):
    if not folder.exists():
        return 0
    return len([
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in suffixes
    ])


def run_cmd(cmd_list, description):
    """Mengeksekusi perintah subprocess dan menghentikan alur jika terjadi kegagalan."""
    print("\n" + "=" * 65)
    print(f" [TAHAP 1] {description}")
    print("=" * 65)
    print(f" Executing: {' '.join(cmd_list)}")
    print("-" * 65)
    result = subprocess.run(cmd_list, cwd=str(PROJECT_ROOT))  # subprocess.run(): Jalankan perintah CLI
    if result.returncode != 0:                                # Memeriksa status exit code hasil eksekusi
        print_friendly_error(
            f"LANGKAH TAHAP 1 GAGAL: {description}",
            [
                f"Subprocess berhenti dengan exit code {result.returncode}.",
                "Penyebab umum: video tidak bisa dibaca, frame hasil ekstraksi kosong, checkpoint enhancement tidak cocok, atau dependency OpenCV/PyTorch bermasalah.",
            ],
            [
                "Baca pesan error teknis tepat di atas blok ini untuk detail dari script yang gagal.",
                "Pastikan video input valid dan bisa dibuka pemutar video.",
                "Jika gagal pada enhancement, cek checkpoint .pth atau jalankan ulang dengan --train.",
                "Jika hanya ingin melewati video full, gunakan --skip_video_enhance.",
            ],
        )
        sys.exit(result.returncode)                           # sys.exit(): Hentikan program jika gagal


def main():
    parser = argparse.ArgumentParser(                         # Buat penampung argumen terminal
        description="Tahap 1: Ekstraksi Data & Preprocessing Enhancement Zero-DCE+CBAM."
    )
    parser.add_argument(                                       # Argumen --video (wajib)
        "--video", type=str, required=True,
        help="Path ke file video mentah (misal: data/raw/videos/video01.mp4)."
    )
    parser.add_argument(                                       # Argumen --interval (default 1.0)
        "--interval", type=float, default=1.0,
        help="Interval ekstraksi frame dalam detik (default: 1.0)."
    )
    parser.add_argument(                                       # Argumen --ssim_threshold (default 0.95)
        "--ssim_threshold", type=float, default=0.95,
        help="Threshold SSIM untuk eliminasi frame redundan (default: 0.95)."
    )
    parser.add_argument(                                       # Flag --train (opsional)
        "--train", action="store_true",
        help="Latih ulang model Zero-DCE + CBAM dari awal pada video baru (opsional)."
    )
    parser.add_argument(                                       # Path checkpoint model
        "--checkpoint", type=str,
        default="checkpoints/zero_dce_cbam/best_zero_dce_cbam.pth",
        help="Path checkpoint model Zero-DCE + CBAM."
    )
    parser.add_argument(                                       # Jumlah CPU threads PyTorch
        "--num_threads", type=int, default=4,
        help="Jumlah CPU threads PyTorch untuk batch enhance (default: 4)."
    )
    parser.add_argument(                                       # Flag --side_by_side (opsional)
        "--side_by_side", action="store_true",
        help="Hasilkan video perbandingan side-by-side (kiri: mentah, kanan: enhanced)."
    )
    parser.add_argument(                                       # Flag --skip_video_enhance (opsional)
        "--skip_video_enhance", action="store_true",
        help="Lewati pencerahan video full (hanya proses citra frame)."
    )
    parser.add_argument(                                       # Flag --skip_quality_viz (opsional)
        "--skip_quality_viz", action="store_true",
        help="Lewati pembaruan dashboard gabungan di results/enhancement/quality_metrics/."
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Folder tujuan penyimpanan hasil enhancement (default: otomatis ke zero_dce_cbam atau zero_dce_cbam_upscaled jika --scale > 1.0)."
    )
    # Argumen Upscaling & Resizing (Default: Auto-Upscale 2.0x + Sharpen 60%)
    parser.add_argument(
        "--scale", type=float, default=2.0,
        help="Faktor skala perbesaran/upscale resolusi (default: 2.0x perbesaran otomatis)."
    )
    parser.add_argument(
        "--target_size", type=int, nargs=2, default=None, metavar=("WIDTH", "HEIGHT"),
        help="Target resolusi spesifik (Lebar Tinggi), misal: --target_size 1280 1440."
    )
    parser.add_argument(
        "--upscale_method", type=str, default="lanczos", choices=["lanczos", "bicubic", "bilinear"],
        help="Metode interpolasi presisi float32 (default: lanczos)."
    )
    parser.add_argument(
        "--sharpen", type=float, default=60.0,
        help="Kekuatan unsharp mask setelah upscale (default: 60.0)."
    )
    parser.add_argument(
        "--allow_stretch", action="store_true",
        help="Izinkan --target_size mengubah aspect ratio (objek terdistorsi)."
    )

    args = parser.parse_args()                                 # Membaca argumen terminal ke objek args
    video_path = Path(args.video)

    if not video_path.exists():                                # Memeriksa keberadaan file video di disk
        print_friendly_error(
            "FILE VIDEO TAHAP 1 TIDAK DITEMUKAN",
            [
                f"Path video yang diberikan tidak ada: {video_path}",
                "Tahap 1 harus dimulai dari file video mentah seperti .mp4, .avi, atau .mov.",
            ],
            [
                "Cek kembali nama file dan folder video.",
                "Letakkan video mentah di data/raw/videos/.",
                "Contoh: python scripts/run_tahap1.py --video data/raw/videos/28_september.mp4",
            ],
        )
        sys.exit(1)

    valid_video_exts = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
    if video_path.suffix.lower() not in valid_video_exts:
        print_friendly_error(
            "FORMAT FILE VIDEO TAHAP 1 TIDAK DIDUKUNG",
            [
                f"File '{video_path.name}' berekstensi '{video_path.suffix}'.",
                "Tahap 1 hanya menerima file video umum, bukan gambar, zip, atau file dataset.",
            ],
            [
                "Gunakan file video dengan ekstensi .mp4, .avi, .mov, .mkv, atau .m4v.",
                "Jika input berupa kumpulan gambar, jalankan modul dataset secara manual sesuai kebutuhan.",
            ],
        )
        sys.exit(1)

    video_stem = video_path.stem                               # Mengambil nama file video tanpa ekstensi
    python_exe = sys.executable                                # Mengambil lokasi interpreter Python aktif
    ckpt_path = Path(args.checkpoint)

    # Tentukan folder output enhancement secara otomatis jika tidak ditentukan manual
    if args.output_dir:
        enhanced_base_dir = args.output_dir
    elif args.scale != 1.0 or args.target_size is not None:
        enhanced_base_dir = "data/enhanced/zero_dce_cbam_upscaled"
    else:
        enhanced_base_dir = "data/enhanced/zero_dce_cbam"

    print("\n" + "#" * 65)
    print(f" MEMULAI TAHAP 1: EKSTRAKSI DATA & PREPROCESSING ENHANCEMENT")
    print(f" Video : {video_path.name}  (stem: {video_stem})")
    print("#" * 65)

    step = 0

    # 1. Ekstraksi Frame
    step += 1
    run_cmd(
        [python_exe, "scripts/dataset/extract_frames.py",       # Eksekusi script ekstraksi frame
         "--input", str(video_path),
         "--output", "data/raw/frames",
         "--interval", str(args.interval)],
        f"{step}. Ekstraksi Frame dari {video_path.name}"
    )

    # 2. Seleksi Frame / Eliminasi Redundansi (SSIM)
    step += 1
    raw_frames_dir = Path("data/raw/frames") / video_stem
    raw_frame_count = count_files(raw_frames_dir, {".jpg", ".jpeg", ".png", ".bmp"})
    if raw_frame_count == 0:
        print_friendly_error(
            f"EKSTRAKSI FRAME KOSONG UNTUK '{video_stem}'",
            [
                f"Tidak ada frame gambar ditemukan di {raw_frames_dir.as_posix()}.",
                "Video mungkin rusak, tidak bisa dibaca OpenCV, durasinya terlalu pendek, atau interval ekstraksi terlalu besar.",
            ],
            [
                "Coba buka video secara manual untuk memastikan file tidak rusak.",
                "Coba jalankan ulang dengan interval lebih kecil, misalnya --interval 0.5.",
                "Pastikan codec video didukung oleh OpenCV/FFmpeg di environment ini.",
            ],
        )
        sys.exit(1)

    run_cmd(
        [python_exe, "scripts/dataset/select_frames.py",        # Eksekusi script seleksi SSIM
         "--input", str(raw_frames_dir),
         "--output", "data/selected",
         "--method", "ssim",
         "--threshold", str(args.ssim_threshold)],
        f"{step}. Seleksi & Eliminasi Redundansi Frame ({video_stem})"
    )

    # 3. (Opsional) Pelatihan Model Zero-DCE + CBAM
    if args.train:
        step += 1
        run_cmd(
            [python_exe, "enhancement/train_zero_dce_cbam.py",   # Eksekusi script pelatihan Zero-DCE+CBAM
             "--config", "configs/zero_dce_cbam.yaml",
             "--video_stem", video_stem],
            f"{step}. Pelatihan Model Zero-DCE + CBAM ({video_stem})"
        )
    else:
        print(f"\n[INFO] Flag --train tidak diaktifkan. Melewati pelatihan, langsung ke batch enhancement.")
        print(f"       Menggunakan checkpoint: {ckpt_path}")

    # 4. Batch Enhancement Generator pada Citra Terpilih (Resolusi Asli & Upscaled)
    if not ckpt_path.exists():                                 # Memeriksa ketersediaan checkpoint model .pth
        print_friendly_error(
            "CHECKPOINT ZERO-DCE + CBAM TIDAK DITEMUKAN",
            [
                f"File checkpoint tidak ada: {ckpt_path}",
                "Batch enhancement membutuhkan model .pth untuk menerangkan frame low-light.",
            ],
            [
                "Pastikan path --checkpoint sudah benar.",
                "Jika belum punya checkpoint, jalankan Tahap 1 dengan flag --train.",
                f"Contoh: python scripts/run_tahap1.py --video {video_path.as_posix()} --train",
            ],
        )
        sys.exit(1)

    selected_dir = Path("data/selected") / video_stem
    selected_frame_count = count_files(selected_dir, {".jpg", ".jpeg", ".png", ".bmp"})
    if selected_frame_count == 0:
        print_friendly_error(
            f"TIDAK ADA FRAME TERPILIH UNTUK '{video_stem}'",
            [
                f"Folder {selected_dir.as_posix()} kosong setelah seleksi SSIM.",
                "Threshold SSIM mungkin terlalu ketat, atau frame hasil ekstraksi tidak terbaca dengan baik.",
            ],
            [
                "Coba turunkan threshold, misalnya --ssim_threshold 0.90.",
                "Cek isi folder data/raw/frames/<video_stem>/ untuk memastikan frame mentah ada.",
                "Jika ingin semua frame dipertahankan, jalankan select_frames.py dengan parameter yang lebih longgar.",
            ],
        )
        sys.exit(1)

    # 4a. Generasi Citra Enhanced Resolusi Asli (1.0x) -> data/enhanced/zero_dce_cbam/
    step += 1
    enhanced_orig_base = "data/enhanced/zero_dce_cbam"
    run_cmd(
        [python_exe, "enhancement/enhance.py",
         "--checkpoint", str(ckpt_path),
         "--input_dir", str(selected_dir.parent),
         "--output_dir", enhanced_orig_base,
         "--video_stem", video_stem,
         "--num_threads", str(args.num_threads),
         "--scale", "1.0",
         "--sharpen", "0.0",
         "--overwrite"],
        f"{step}. Generator Batch Enhancement Resolusi Asli 1.0x ({video_stem})"
    )

    # 4b. Generasi Citra Enhanced Upscaled (2.0x + Sharpen) -> data/enhanced/zero_dce_cbam_upscaled/
    step += 1
    enhanced_upscale_base = "data/enhanced/zero_dce_cbam_upscaled"
    enhance_upscale_cmd = [
        python_exe, "enhancement/enhance.py",
        "--checkpoint", str(ckpt_path),
        "--input_dir", str(selected_dir.parent),
        "--output_dir", enhanced_upscale_base,
        "--video_stem", video_stem,
        "--num_threads", str(args.num_threads),
        "--scale", str(args.scale),
        "--upscale_method", str(args.upscale_method),
        "--sharpen", str(args.sharpen),
        "--overwrite"
    ]
    if args.target_size:
        enhance_upscale_cmd.extend(["--target_size", str(args.target_size[0]), str(args.target_size[1])])
    if args.allow_stretch:
        enhance_upscale_cmd.append("--allow_stretch")

    run_cmd(
        enhance_upscale_cmd,
        f"{step}. Generator Batch Enhancement Upscale {args.scale:.1f}x ({video_stem})"
    )

    # 5. Evaluasi Kualitas Citra Enhancement (Resolusi Asli & Upscaled)
    step += 1
    enhanced_dir_orig = Path(enhanced_orig_base) / video_stem
    run_cmd(
        [python_exe, "enhancement/evaluate_enhancement.py",
         "--orig_dir", str(selected_dir),
         "--cbam_dir", str(enhanced_dir_orig),
         "--output_dir", "results/enhancement",
         "--cbam_checkpoint", str(ckpt_path),
         "--video_stem", video_stem],
        f"{step}. Evaluasi Kualitas Citra Enhancement Resolusi Asli ({video_stem})"
    )

    step += 1
    enhanced_dir_upscale = Path(enhanced_upscale_base) / video_stem
    run_cmd(
        [python_exe, "enhancement/evaluate_enhancement.py",
         "--orig_dir", str(selected_dir),
         "--cbam_dir", str(enhanced_dir_upscale),
         "--output_dir", "results/enhancement_upscaled",
         "--cbam_checkpoint", str(ckpt_path),
         "--video_stem", video_stem],
        f"{step}. Evaluasi Kualitas Citra Enhancement Upscaled ({video_stem})"
    )

    # 6. Pembaruan Dashboard Gabungan Kualitas Citra
    # Laporan metrik video ini ditulis ke folder tersendiri (results/enhancement/<stem>/),
    # sehingga flag --merge dipakai agar dashboard memuat SELURUH video, bukan hanya
    # video terakhir yang diproses.
    if not args.skip_quality_viz:
        step += 1
        run_cmd(
            [python_exe, "scripts/utils/visualize_quality_metrics.py"],
            f"{step}. Pembaruan Dashboard Kualitas Citra Gabungan ({video_stem})"
        )
    else:
        print("\n[INFO] Flag --skip_quality_viz diaktifkan. Melewati pembaruan dashboard kualitas citra.")

    # 7. Pencerahan & Rekonstruksi Video Full
    video_out_path = None
    if not args.skip_video_enhance:
        step += 1
        suffix = "_comparison" if args.side_by_side else "_enhanced"
        video_out_dir = PROJECT_ROOT / "results" / "enhancement" / video_stem / "video"
        video_out_path = video_out_dir / f"{video_stem}{suffix}.mp4"

        video_cmd = [
            python_exe, "scripts/utils/enhance_video.py",     # Eksekusi generator pencerahan video full
            "--video", str(video_path),
            "--checkpoint", str(ckpt_path),
            "--output", str(video_out_path),
            "--num_threads", str(args.num_threads)
        ]
        if args.side_by_side:
            video_cmd.append("--side_by_side")

        run_cmd(
            video_cmd,
            f"{step}. Pencerahan & Rekonstruksi Video Full ({video_stem})"
        )
    else:
        print("\n[INFO] Flag --skip_video_enhance diaktifkan. Melewati pencerahan video full.")

    print("\n" + "#" * 65)
    print(f" TAHAP 1 SELESAI [SUKSES] - {video_path.name}")
    print("#" * 65)
    print(" Lokasi Output:")
    print(f"  [DIR] Frame Mentah          : data/raw/frames/{video_stem}/")
    print(f"  [DIR] Frame Terpilih        : data/selected/{video_stem}/")
    print(f"  [DIR] Citra Enhanced (1.0x) : data/enhanced/zero_dce_cbam/{video_stem}/")
    print(f"  [DIR] Citra Enhanced (2.0x) : data/enhanced/zero_dce_cbam_upscaled/{video_stem}/")
    print(f"  [FILE] Metrik CSV           : results/enhancement/{video_stem}/metrics_reports/enhancement_metrics_report.csv")
    if not args.skip_quality_viz:
        print(f"  [DIR] Dashboard Kualitas: results/enhancement/quality_metrics/  (dashboard.html - gabungan semua video)")
    if video_out_path and video_out_path.exists():
        rel_video_path = video_out_path.relative_to(PROJECT_ROOT)
        print(f"  [FILE] Video Enhanced   : {rel_video_path.as_posix()}")
    print()
    print(" Langkah Berikutnya (Tahap 2 - Anotasi Roboflow & Persiapan Dataset):")
    print(f"  1. Upload & Anotasi citra terang di: data/enhanced/zero_dce_cbam/{video_stem}/ ke Roboflow")
    print("  2. Export dataset YOLO ke data/original/images & data/original/labels")
    print(f"  3. Jalankan Tahap 2:")
    print(f"     python scripts/run_tahap2.py --video_stem {video_stem}")
    print("#" * 65 + "\n")


if __name__ == "__main__":                                     # Memastikan skrip berjalan hanya saat dipanggil langsung
    main()
