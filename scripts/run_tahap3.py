#!/usr/bin/env python3
"""
Script : scripts/run_tahap3.py
Tahap  : 3 — TRAINING DETEKSI OBJEK (YOLO12) & EVALUASI KOMPARATIF

Menjalankan seluruh proses training dan evaluasi deteksi secara berurutan:
  1. Training Model Deteksi YOLO12 — Skenario A (Citra Original Low-Light)
  2. Training Model Deteksi YOLO12 — Skenario B (Citra Enhanced Zero-DCE + CBAM)
  3. Evaluasi Metrik Deteksi (Precision, Recall, mAP@0.5, mAP@0.5:0.95, FPS)
     + Generasi Figur Visualisasi Bounding Box 2-Panel

Penggunaan:
    python scripts/run_tahap3.py --video_stem video01
    python scripts/run_tahap3.py --video_stem video01 --only_scenario A
    python scripts/run_tahap3.py --video_stem video01 --only_eval
"""

import argparse           # Pembaca argumen baris perintah terminal (--video_stem, --epochs, dsb)
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


def count_files(folder: Path, suffixes, video_stem: str = None, use_master: bool = False):
    if not folder.exists():
        return 0
    files = [
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in suffixes
    ]
    if use_master or not video_stem or video_stem == "master":
        return len(files)
    return len([
        p for p in files
        if p.name.startswith(f"{video_stem}_") or video_stem.lower() in [part.lower() for part in p.parts]
    ])


def validate_dataset_ready(name: str, image_dir: Path, label_dir: Path, video_stem: str, use_master: bool):
    """Memastikan dataset YOLO punya folder image dan label sebelum training/evaluasi."""
    image_count = count_files(image_dir, {".jpg", ".jpeg", ".png", ".bmp"}, video_stem, use_master)
    label_count = count_files(label_dir, {".txt"}, video_stem, use_master)

    if image_count == 0 or label_count == 0:
        dataset_mode = "master dataset" if use_master else "data/splits"
        print_friendly_error(
            f"DATASET {name} BELUM SIAP UNTUK TAHAP 3",
            [
                f"Mode dataset yang dipakai: {dataset_mode}.",
                f"Jumlah gambar pada {image_dir.as_posix()}: {image_count}.",
                f"Jumlah label pada {label_dir.as_posix()}: {label_count}.",
                "Training YOLO membutuhkan pasangan gambar dan label hasil Tahap 2.",
            ],
            [
                f"Jalankan Tahap 2 terlebih dahulu: python scripts/run_tahap2.py --video_stem {video_stem}",
                "Jika memakai --use_master, pastikan Tahap 2 tidak dijalankan dengan --skip_master.",
                "Jika error berasal dari label kosong, cek apakah anotasi Roboflow sudah diekspor dengan benar.",
            ],
        )
        sys.exit(1)


def validate_weights_ready(weights: str):
    """Memastikan pretrained weights tersedia lokal sebelum training."""
    weights_path = Path(weights)
    if not weights_path.is_absolute():
        weights_path = PROJECT_ROOT / weights_path

    if not weights_path.exists():
        print_friendly_error(
            "PRETRAINED WEIGHTS YOLO TIDAK DITEMUKAN",
            [
                f"File weights '{weights}' tidak ditemukan.",
                "Training YOLO membutuhkan bobot awal seperti yolo12n.pt atau path checkpoint yang valid.",
            ],
            [
                "Pastikan file yolo12n.pt ada di root project.",
                "Atau jalankan dengan argumen --weights yang mengarah ke file .pt yang tersedia.",
                f"Contoh: python scripts/run_tahap3.py --video_stem <nama_video> --weights yolo12n.pt",
            ],
        )
        sys.exit(1)


def run_cmd(cmd_list, description):
    """Mengeksekusi perintah subprocess dan menghentikan alur jika terjadi kegagalan."""
    print("\n" + "=" * 70)
    print(f" [TAHAP 3] {description}")
    print("=" * 70)
    print(f" Executing: {' '.join(cmd_list)}")
    print("-" * 70)
    result = subprocess.run(cmd_list, cwd=str(PROJECT_ROOT))  # subprocess.run(): Jalankan perintah CLI
    if result.returncode != 0:                                # Memeriksa status exit code hasil eksekusi
        print_friendly_error(
            f"LANGKAH TAHAP 3 GAGAL: {description}",
            [
                f"Subprocess berhenti dengan exit code {result.returncode}.",
                "Penyebab paling umum: dataset belum siap, config YAML salah, label YOLO tidak valid, weights tidak ditemukan, atau perangkat GPU/CPU bermasalah.",
            ],
            [
                "Baca pesan error tepat di atas blok ini untuk detail teknis dari script yang gagal.",
                "Jika error dataset/label, jalankan ulang Tahap 2 dan pastikan validasi dataset lulus.",
                "Jika error CUDA/GPU, coba jalankan dengan --device cpu.",
                "Jika error weights, pastikan file .pt yang dipakai benar-benar ada.",
            ],
        )
        sys.exit(result.returncode)                           # sys.exit(): Hentikan program jika gagal


def main():
    parser = argparse.ArgumentParser(                         # Buat penampung argumen terminal
        description="Tahap 3: Training Deteksi YOLO12 (Skenario A & B) + Evaluasi Komparatif."
    )
    parser.add_argument(                                       # Argumen --video_stem (wajib)
        "--video_stem", type=str, required=True,
        help="Nama/ID video sumber tanpa ekstensi (misal: 'video01' atau '7_agustus')."
    )
    parser.add_argument(                                       # Argumen --epochs (default 100)
        "--epochs", type=int, default=100,
        help="Jumlah epoch training YOLO (default: 100)."
    )
    parser.add_argument(                                       # Argumen --weights (default yolo12n.pt)
        "--weights", type=str, default="yolo12n.pt",
        help="Pretrained weights YOLO (default: yolo12n.pt)."
    )
    parser.add_argument(                                       # Argumen --batch (default 16)
        "--batch", type=int, default=16,
        help="Batch size training YOLO (default: 16)."
    )
    parser.add_argument(                                       # Argumen --imgsz (default 640)
        "--imgsz", type=int, default=640,
        help="Ukuran resolusi input citra YOLO (default: 640)."
    )
    parser.add_argument(                                       # Argumen --device (default GPU '0' atau 'cpu')
        "--device", type=str, default="",
        help="Perangkat komputasi ('0' untuk GPU, 'cpu' untuk CPU)."
    )
    parser.add_argument(                                       # Opsi training hanya 1 skenario ('A' atau 'B')
        "--only_scenario", type=str, default=None, choices=["A", "B", "a", "b"],
        help="Jalankan training hanya untuk 1 skenario ('A' atau 'B'). Default: keduanya."
    )
    parser.add_argument(                                       # Flag langsung ke evaluasi tanpa training
        "--only_eval", action="store_true",
        help="Lewati training, langsung jalankan evaluasi (kedua model sudah ada)."
    )
    parser.add_argument(                                       # Confidence threshold evaluasi
        "--conf", type=float, default=0.25,
        help="Confidence threshold untuk evaluasi & visualisasi (default: 0.25)."
    )
    parser.add_argument(                                       # Jumlah sampel visualisasi 2-panel
        "--limit", type=int, default=6,
        help="Jumlah sampel gambar untuk visualisasi 2-panel (default: 6)."
    )
    parser.add_argument(                                       # Flag Master Dataset (Opsi 2)
        "--use_master", action="store_true",
        help="Gunakan Master Dataset Multi-Video Opsi 2 (configs/dataset_master_*.yaml)."
    )

    args = parser.parse_args()                                 # Membaca argumen terminal ke objek args
    video_stem = args.video_stem
    python_exe = sys.executable                                # Mengambil jalur interpreter Python aktif

    ckpt_a = PROJECT_ROOT / "checkpoints" / "yolo_original" / "best.pt"
    ckpt_b = PROJECT_ROOT / "checkpoints" / "yolo_zero_dce_cbam" / "best.pt"

    print("\n" + "#" * 70)
    print(f" MEMULAI TAHAP 3: TRAINING DETEKSI OBJEK (YOLO12) & EVALUASI")
    print(f" Video Stem : {video_stem}{' (MASTER DATASET OPSI 2)' if args.use_master else ''}")
    print("#" * 70)

    step = 0
    run_a = not args.only_eval and (args.only_scenario is None or args.only_scenario.upper() == "A")
    run_b = not args.only_eval and (args.only_scenario is None or args.only_scenario.upper() == "B")

    if not args.only_eval:
        validate_weights_ready(args.weights)

    if args.use_master:
        dataset_a_img = PROJECT_ROOT / "data" / "master_dataset" / "train" / "images"
        dataset_a_lbl = PROJECT_ROOT / "data" / "master_dataset" / "train" / "labels"
        dataset_b_img = PROJECT_ROOT / "data" / "enhanced" / "master_dataset" / "train" / "images"
        dataset_b_lbl = PROJECT_ROOT / "data" / "enhanced" / "master_dataset" / "train" / "labels"
    else:
        dataset_a_img = PROJECT_ROOT / "data" / "splits" / "train" / "images"
        dataset_a_lbl = PROJECT_ROOT / "data" / "splits" / "train" / "labels"
        dataset_b_img = PROJECT_ROOT / "data" / "enhanced" / "zero_dce_cbam" / "train" / "images"
        dataset_b_lbl = PROJECT_ROOT / "data" / "enhanced" / "zero_dce_cbam" / "train" / "labels"

    if run_a or args.only_eval:
        validate_dataset_ready("SKENARIO A (ORIGINAL)", dataset_a_img, dataset_a_lbl, video_stem, args.use_master)
    if run_b or args.only_eval:
        validate_dataset_ready("SKENARIO B (ENHANCED)", dataset_b_img, dataset_b_lbl, video_stem, args.use_master)

    # 1. Training Skenario A (Original Low-Light)
    if run_a:
        step += 1
        cmd_a = [
            python_exe, "detection/train_yolo.py",             # Eksekusi training YOLO Skenario A
            "--scenario", "A",
            "--epochs", str(args.epochs),
            "--weights", args.weights,
            "--batch", str(args.batch),
            "--imgsz", str(args.imgsz),
            "--video_stem", video_stem,
        ]
        if args.use_master:
            cmd_a += ["--data", "configs/dataset_master_original.yaml"]
        if args.device:
            cmd_a += ["--device", args.device]
        run_cmd(cmd_a, f"{step}  Training Skenario A — Citra Original Low-Light ({video_stem})")

    # 2. Training Skenario B (Zero-DCE + CBAM Enhanced)
    if run_b:
        enhanced_dir = PROJECT_ROOT / "data" / "enhanced" / "zero_dce_cbam"
        if not enhanced_dir.exists():                          # Memeriksa ketersediaan folder citra enhanced
            print(f"\n[WARNING] Folder enhanced tidak ditemukan: {enhanced_dir}")
            print("          Jalankan Tahap 2 terlebih dahulu untuk menghasilkan citra enhanced.")
            print("          python scripts/run_tahap2.py --video_stem " + video_stem)
            sys.exit(1)

        step += 1
        cmd_b = [
            python_exe, "detection/train_yolo.py",             # Eksekusi training YOLO Skenario B
            "--scenario", "B",
            "--epochs", str(args.epochs),
            "--weights", args.weights,
            "--batch", str(args.batch),
            "--imgsz", str(args.imgsz),
            "--video_stem", video_stem,
        ]
        if args.use_master:
            cmd_b += ["--data", "configs/dataset_master_zero_dce_cbam.yaml"]
        if args.device:
            cmd_b += ["--device", args.device]
        run_cmd(cmd_b, f"{step}  Training Skenario B — Citra Enhanced Zero-DCE + CBAM ({video_stem})")

    # 3. Evaluasi Metrik Deteksi + Visualisasi 2-Panel
    skip_eval = args.only_scenario is not None and not args.only_eval
    if not skip_eval:
        missing = []
        if not ckpt_a.exists():                                # Memeriksa ketersediaan checkpoint Skenario A
            missing.append(f"  - Skenario A: {ckpt_a}")
        if not ckpt_b.exists():                                # Memeriksa ketersediaan checkpoint Skenario B
            missing.append(f"  - Skenario B: {ckpt_b}")

        if missing:
            print_friendly_error(
                "CHECKPOINT EVALUASI TAHAP 3 BELUM LENGKAP",
                [
                    "Evaluasi A vs B membutuhkan checkpoint Skenario A dan Skenario B.",
                    "Checkpoint berikut belum ditemukan:",
                    *missing,
                ],
                [
                    f"Jalankan training lengkap: python scripts/run_tahap3.py --video_stem {video_stem}",
                    "Jika hanya ingin training satu skenario, gunakan --only_scenario A atau --only_scenario B dan jangan pakai --only_eval.",
                    "Pastikan file best.pt tersimpan di checkpoints/yolo_original/ dan checkpoints/yolo_zero_dce_cbam/.",
                ],
            )
            sys.exit(1)
        else:
            eval_cmd = [
                python_exe, "detection/evaluate_detection.py",
                "--weights_a", str(ckpt_a),
                "--weights_b", str(ckpt_b),
                "--output_dir", "results/detection",
                "--conf", str(args.conf),
                "--limit", str(args.limit),
                "--video_stem", video_stem
            ]
            if args.use_master:
                eval_cmd.append("--use_master")
            run_cmd(
                eval_cmd,
                f"{step}  Evaluasi Metrik Deteksi + Visualisasi 2-Panel A vs B ({video_stem})"
            )

    print("\n" + "#" * 70)
    print(f" TAHAP 3 SELESAI [SUKSES] - {video_stem}")
    print("#" * 70)
    print(" Lokasi Output:")
    print(f"  [DIR] Checkpoint Skenario A : checkpoints/yolo_original/{video_stem}/best.pt")
    print(f"  [DIR] Checkpoint Skenario B : checkpoints/yolo_zero_dce_cbam/{video_stem}/best.pt")
    print(f"  [FILE] Metrik CSV           : results/detection/detection_metrics_report.csv")
    print(f"  [IMG]  Figur 2-Panel        : results/detection/2panel_detection_comparison_*.png")
    print()
    print(" Langkah Berikutnya:")
    print("  -> Jalankan Tahap 4:")
    print(f"     python scripts/run_tahap4.py --video_stem {video_stem}")
    print("#" * 70 + "\n")


if __name__ == "__main__":                                     # Memastikan skrip berjalan hanya saat dipanggil langsung
    main()
