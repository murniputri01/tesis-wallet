#!/usr/bin/env python3
"""
Script: detection/train_yolo.py
Deskripsi: Script utama pelatihan model deteksi objek YOLO12 untuk:
             - Skenario A (Original Low-Light Image)
             - Skenario B (Proposed Method: Zero-DCE + CBAM Enhanced Image)

Penggunaan:
    # Training Skenario A (Original Baseline) - 100 Epochs
    python detection/train_yolo.py --scenario A --epochs 100 --weights yolo12n.pt

    # Training Skenario B (Proposed Method: Zero-DCE + CBAM) - 100 Epochs
    python detection/train_yolo.py --scenario B --epochs 100 --weights yolo12n.pt
"""

import argparse               # Pembaca argumen baris perintah terminal (--scenario, --epochs, dsb)
import sys                    # Pengendali eksekusi sistem & pemutus program jika error
import shutil                 # Pustaka penyalinan berkas checkpoint best.pt ke folder target
from pathlib import Path      # Pengelola jalur folder/file lintas sistem operasi (Windows/Linux)

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # Lokasi akar proyek
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))              # Sisipkan jalur akar proyek ke sys.path

from ultralytics import YOLO  # Framework YOLO untuk inisialisasi & pelatihan model
from detection.sync_labels import sync_labels          # Fungsi sinkronisasi label .txt ke folder enhanced


def train_yolo(
    scenario: str,
    weights: str = "yolo12n.pt",
    epochs: int = 100,
    batch: int = 16,
    imgsz: int = 640,
    device: str = "",
    project: str = "runs/detect",
    name: str = None,
    video_stem: str = None,
    data: str = None
):
    scenario = scenario.upper()
    if scenario not in {"A", "B"}:
        raise ValueError(f"Skenario harus 'A' (Original) atau 'B' (Zero-DCE+CBAM). Diberikan: {scenario}")

    if scenario == "A":
        config_path = PROJECT_ROOT / "configs" / "dataset_original.yaml"
        exp_name = name or (f"yolo12_scenario_A_{video_stem}" if video_stem else "yolo12_scenario_A_original")
        base_ckpt_dir = PROJECT_ROOT / "checkpoints" / "yolo_original"
    else:
        # Skenario B (Zero-DCE + CBAM Proposed Method)
        sync_labels(
            splits_dir=PROJECT_ROOT / "data" / "splits",
            enhanced_dir=PROJECT_ROOT / "data" / "enhanced" / "zero_dce_cbam"
        )
        config_path = PROJECT_ROOT / "configs" / "dataset_zero_dce_cbam.yaml"
        exp_name = name or (f"yolo12_scenario_B_{video_stem}" if video_stem else "yolo12_scenario_B_zero_dce_cbam")
        base_ckpt_dir = PROJECT_ROOT / "checkpoints" / "yolo_zero_dce_cbam"

    target_ckpt_dir = base_ckpt_dir / video_stem if video_stem else base_ckpt_dir

    # Konfigurasi dataset kustom menimpa pilihan baku per skenario (misal dataset bersih 7_agustus)
    if data:
        custom = Path(data)
        config_path = custom if custom.is_absolute() else PROJECT_ROOT / custom

    if not config_path.exists():
        print(f"[ERROR] File konfigurasi dataset tidak ditemukan: {config_path}")
        sys.exit(1)

    target_ckpt_dir.mkdir(parents=True, exist_ok=True)
    base_ckpt_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print(f" MEMULAI TRAINING DETEKSI OBJEK YOLO12 | SKENARIO {scenario}{f' - VIDEO: {video_stem}' if video_stem else ''}")
    print("=" * 70)
    print(f" Model Weights : {weights}")
    print(f" Config File   : {config_path.as_posix()}")
    print(f" Epochs        : {epochs}")
    print(f" Batch Size    : {batch}")
    print(f" Image Size    : {imgsz}x{imgsz}")
    print(f" Exp Name      : {exp_name}")
    print("-" * 70)

    # Inisialisasi Model YOLO
    model = YOLO(weights)

    # Eksekusi Pelatihan
    train_args = {
        "data": str(config_path),
        "epochs": epochs,
        "batch": batch,
        "imgsz": imgsz,
        "project": str(PROJECT_ROOT / project),
        "name": exp_name,
        "exist_ok": True,
        "plots": True,
        "save": True,
        "verbose": True
    }

    if device:
        train_args["device"] = device

    results = model.train(**train_args)

    # Salin best.pt ke folder checkpoints/
    run_dir = PROJECT_ROOT / project / exp_name
    best_weights = run_dir / "weights" / "best.pt"
    last_weights = run_dir / "weights" / "last.pt"

    if best_weights.exists():
        target_best = target_ckpt_dir / "best.pt"
        shutil.copy2(best_weights, target_best)
        print(f"\n [SUCCESS] Checkpoint terbaik disimpan ke: {target_best.as_posix()}")
        if target_ckpt_dir != base_ckpt_dir:
            shutil.copy2(best_weights, base_ckpt_dir / "best.pt")

    if last_weights.exists():
        target_last = target_ckpt_dir / "last.pt"
        shutil.copy2(last_weights, target_last)
        print(f" [SUCCESS] Checkpoint terakhir disimpan ke: {target_last.as_posix()}")
        if target_ckpt_dir != base_ckpt_dir:
            shutil.copy2(last_weights, base_ckpt_dir / "last.pt")

    print("=" * 70 + "\n")
    return results


def main():
    parser = argparse.ArgumentParser(description="Script Pelatihan Deteksi Objek YOLO12 Skenario A & B.")
    parser.add_argument("--scenario", "-s", type=str, required=True, choices=["A", "B", "a", "b"], help="Skenario eksperimen ('A'=Original, 'B'=Zero-DCE+CBAM).")
    parser.add_argument("--weights", "-w", type=str, default="yolo12n.pt", help="Pretrained model weights (default: yolo12n.pt).")
    parser.add_argument("--epochs", "-e", type=int, default=100, help="Jumlah putaran training (default: 100).")
    parser.add_argument("--batch", "-b", type=int, default=16, help="Batch size (default: 16).")
    parser.add_argument("--imgsz", type=int, default=640, help="Ukuran resolusi citra input (default: 640).")
    parser.add_argument("--device", type=str, default="", help="Perangkat komputasi ('0', 'cpu', dll).")
    parser.add_argument("--project", type=str, default="runs/detect", help="Folder output runs (default: runs/detect).")
    parser.add_argument("--data", type=str, default=None, help="Config dataset kustom (.yaml), menimpa pilihan baku skenario.")
    parser.add_argument("--video_stem", type=str, default=None, help="Nama/ID video sumber (misal: 'video01' atau 'ain').")

    args = parser.parse_args()

    train_yolo(
        scenario=args.scenario,
        weights=args.weights,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        data=args.data,
        device=args.device,
        project=args.project,
        video_stem=args.video_stem
    )


if __name__ == "__main__":
    main()
