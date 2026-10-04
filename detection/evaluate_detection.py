#!/usr/bin/env python3
"""
Script: detection/evaluate_detection.py
Deskripsi: Script evaluasi komparatif metrik deteksi objek (Precision, Recall, mAP@0.5, mAP@0.5:0.95, FPS)
           serta pembuatan figur visualisasi perbandingan bounding box 2-panel antara:
             - Skenario A (Original Low-Light Image)
             - Skenario B (Proposed Method: Zero-DCE + CBAM Enhanced Image)

Penggunaan:
    python detection/evaluate_detection.py --weights_a checkpoints/yolo_original/best.pt --weights_b checkpoints/yolo_zero_dce_cbam/best.pt
"""

import argparse               # Pembaca argumen baris perintah terminal (--weights_a, --weights_b, dsb)
import sys                    # Pengendali eksekusi sistem & manipulasi modul sys.path
import time                   # Pengukur waktu eksekusi inferensi milidetik (ms)
import cv2                    # OpenCV untuk pemrosesan citra (imread, cvtColor, rectangle, putText)
import numpy as np            # NumPy untuk komputasi array matriks piksel
import pandas as pd           # Pandas untuk pemrosesan dataframe & ekspor laporan CSV metrik
from pathlib import Path      # Pengelola jalur folder/file lintas sistem operasi (Windows/Linux)

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # Lokasi akar proyek
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))              # Sisipkan jalur akar proyek ke sys.path

import matplotlib             # Matplotlib untuk rendering figur visualisasi
matplotlib.use("Agg")         # Set backend Matplotlib non-interaktif
import matplotlib.pyplot as plt  # Pyplot untuk figur visualisasi 2-panel A vs B
from ultralytics import YOLO  # Framework YOLO untuk evaluasi metrik val() & predict()


def draw_yolo_predictions(img_rgb: np.ndarray, results, conf_thresh: float = 0.25):
    """Menggambar bounding box dan label prediksi YOLO pada citra RGB."""
    img_draw = img_rgb.copy()
    h, w = img_draw.shape[:2]

    if not results or len(results) == 0:
        return img_draw

    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return img_draw

    for box in boxes:
        conf = float(box.conf[0])
        if conf < conf_thresh:
            continue
        cls_id = int(box.cls[0])
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())

        # Warna bounding box (Hijau neon)
        color = (0, 255, 127)
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, thickness=2)

        label_text = f"walet {conf:.2f}"
        font_scale = 0.5
        thickness = 1
        (t_w, t_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

        # Latar belakang label
        cv2.rectangle(img_draw, (x1, max(0, y1 - t_h - 4)), (x1 + t_w, max(t_h + 4, y1)), color, -1)
        cv2.putText(
            img_draw, label_text, (x1, max(t_h, y1 - 2)),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness, cv2.LINE_AA
        )

    return img_draw


def evaluate_detection(
    weights_a: Path,
    weights_b: Path,
    output_dir: Path = Path("results/detection"),
    conf_thresh: float = 0.25,
    num_samples: int = 6,
    video_stem: str = None,
    use_master: bool = False,
    data_a: str = None,
    data_b: str = None
):
    target_out_dir = output_dir / video_stem if video_stem else output_dir
    target_out_dir.mkdir(parents=True, exist_ok=True)

    if data_a:
        config_a = Path(data_a) if Path(data_a).is_absolute() else PROJECT_ROOT / data_a
    elif use_master or video_stem == "master":
        config_a = PROJECT_ROOT / "configs" / "dataset_master_original.yaml"
    else:
        config_a = PROJECT_ROOT / "configs" / "dataset_original.yaml"

    if data_b:
        config_b = Path(data_b) if Path(data_b).is_absolute() else PROJECT_ROOT / data_b
    elif use_master or video_stem == "master":
        config_b = PROJECT_ROOT / "configs" / "dataset_master_zero_dce_cbam.yaml"
    else:
        config_b = PROJECT_ROOT / "configs" / "dataset_zero_dce_cbam.yaml"

    print("\n" + "=" * 75)
    print(f" MEMULAI EVALUASI METRIK DETEKSI OBJEK (YOLO12): SKENARIO A vs B{f' - VIDEO: {video_stem}' if video_stem else ''}")
    print("=" * 75)
    print(f" Config Skenario A : {config_a.as_posix()}")
    print(f" Config Skenario B : {config_b.as_posix()}")
    print("-" * 75)

    # Tentukan split evaluasi (jika test tidak ada/kosong, fallback ke val)
    split_eval = "test"
    test_dir_check = PROJECT_ROOT / "data" / "splits" / "test" / "images"
    if use_master or video_stem == "master":
        test_dir_check = PROJECT_ROOT / "data" / "master_dataset" / "test" / "images"
    if not test_dir_check.exists() or len(list(test_dir_check.glob("*"))) == 0:
        split_eval = "val"
        print(f" [INFO] Split 'test' kosong/tidak ada. Menggunakan split '{split_eval}' untuk evaluasi.")

    metrics_list = []

    # 1. Evaluasi Skenario A (Original)
    if weights_a.exists():
        print(f"\n [1/2] Evaluasi Model Skenario A: {weights_a.as_posix()}")
        model_a = YOLO(str(weights_a))
        res_a = model_a.val(data=str(config_a), split=split_eval, verbose=False)

        p_a = float(res_a.results_dict.get("metrics/precision(B)", 0.0))
        r_a = float(res_a.results_dict.get("metrics/recall(B)", 0.0))
        map50_a = float(res_a.results_dict.get("metrics/mAP50(B)", 0.0))
        map5095_a = float(res_a.results_dict.get("metrics/mAP50-95(B)", 0.0))
        inference_time_a = res_a.speed.get("inference", 0.0)
        fps_a = 1000.0 / inference_time_a if inference_time_a > 0 else 0.0

        metrics_list.append({
            "Skenario": "Skenario A (Original Low-Light)",
            "Precision": round(p_a, 4),
            "Recall": round(r_a, 4),
            "mAP@0.5": round(map50_a, 4),
            "mAP@0.5:0.95": round(map5095_a, 4),
            "Inference_Time_ms": round(inference_time_a, 2),
            "FPS": round(fps_a, 2)
        })
    else:
        print(f" [WARNING] Checkpoint Skenario A '{weights_a}' tidak ditemukan.")

    # 2. Evaluasi Skenario B (Zero-DCE + CBAM)
    if weights_b.exists():
        print(f"\n [2/2] Evaluasi Model Skenario B: {weights_b.as_posix()}")
        model_b = YOLO(str(weights_b))
        res_b = model_b.val(data=str(config_b), split=split_eval, verbose=False)

        p_b = float(res_b.results_dict.get("metrics/precision(B)", 0.0))
        r_b = float(res_b.results_dict.get("metrics/recall(B)", 0.0))
        map50_b = float(res_b.results_dict.get("metrics/mAP50(B)", 0.0))
        map5095_b = float(res_b.results_dict.get("metrics/mAP50-95(B)", 0.0))
        inference_time_b = res_b.speed.get("inference", 0.0)
        fps_b = 1000.0 / inference_time_b if inference_time_b > 0 else 0.0

        metrics_list.append({
            "Skenario": "Skenario B (Zero-DCE + CBAM Usulan)",
            "Precision": round(p_b, 4),
            "Recall": round(r_b, 4),
            "mAP@0.5": round(map50_b, 4),
            "mAP@0.5:0.95": round(map5095_b, 4),
            "Inference_Time_ms": round(inference_time_b, 2),
            "FPS": round(fps_b, 2)
        })
    else:
        print(f" [WARNING] Checkpoint Skenario B '{weights_b}' tidak ditemukan.")

    # Simpan Laporan CSV
    if metrics_list:
        df = pd.DataFrame(metrics_list)
        csv_path = target_out_dir / "detection_metrics_report.csv"
        df.to_csv(csv_path, index=False)
        print("\n" + "-" * 75)
        print(" RINGKASAN HASIL EVALUASI METRIK DETEKSI OBJEK:")
        print("-" * 75)
        print(df.to_string(index=False))
        print(f"\n [SAVED] Laporan CSV disederhanakan di: {csv_path.as_posix()}")

    # 3. Generate Figur Visualisasi Perbandingan Bounding Box 2-Panel
    if weights_a.exists() and weights_b.exists():
        print("\n Generasi Figur Perbandingan Bounding Box 2-Panel (300 DPI)...")
        if use_master or video_stem == "master":
            sample_orig_dir = PROJECT_ROOT / "data" / "master_dataset" / split_eval / "images"
            sample_enh_dir = PROJECT_ROOT / "data" / "enhanced" / "master_dataset" / split_eval / "images"
            if not sample_orig_dir.exists() or len(list(sample_orig_dir.glob("*"))) == 0:
                sample_orig_dir = PROJECT_ROOT / "data" / "master_dataset" / "train" / "images"
                sample_enh_dir = PROJECT_ROOT / "data" / "enhanced" / "master_dataset" / "train" / "images"
        else:
            sample_orig_dir = PROJECT_ROOT / "data" / "splits" / split_eval / "images"
            sample_enh_dir = PROJECT_ROOT / "data" / "enhanced" / "zero_dce_cbam" / split_eval / "images"
            if not sample_orig_dir.exists() or len(list(sample_orig_dir.glob("*"))) == 0:
                sample_orig_dir = PROJECT_ROOT / "data" / "splits" / "train" / "images"
                sample_enh_dir = PROJECT_ROOT / "data" / "enhanced" / "zero_dce_cbam" / "train" / "images"

        valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
        orig_imgs = [p for p in sample_orig_dir.rglob("*") if p.is_file() and p.suffix.lower() in valid_exts]
        if video_stem and video_stem != "master":
            v_stem_lower = video_stem.lower()
            orig_imgs = [
                p for p in orig_imgs
                if v_stem_lower in [part.lower() for part in p.parts] or p.name.lower().startswith(v_stem_lower)
            ]
        orig_imgs = sorted(orig_imgs)[:num_samples]

        saved_figures = 0
        for orig_p in orig_imgs:
            enh_p = sample_enh_dir / orig_p.name

            if not enh_p.exists():
                enh_p = orig_p

            img_orig = cv2.imread(str(orig_p))
            img_enh = cv2.imread(str(enh_p))
            if img_orig is None or img_enh is None:
                continue

            img_orig_rgb = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
            img_enh_rgb = cv2.cvtColor(img_enh, cv2.COLOR_BGR2RGB)

            res_pred_a = model_a.predict(img_orig, conf=conf_thresh, verbose=False)
            res_pred_b = model_b.predict(img_enh, conf=conf_thresh, verbose=False)

            vis_a = draw_yolo_predictions(img_orig_rgb, res_pred_a, conf_thresh)
            vis_b = draw_yolo_predictions(img_enh_rgb, res_pred_b, conf_thresh)

            # Buat Panel Visualisasi 2-Kolom Resolusi Tinggi (300 DPI)
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))

            axes[0].imshow(vis_a)
            axes[0].set_title("Skenario A (Original Low-Light)", fontsize=11, fontweight="bold", pad=8)
            axes[0].axis("off")

            axes[1].imshow(vis_b)
            axes[1].set_title("Skenario B (Zero-DCE + CBAM Usulan)", fontsize=11, fontweight="bold", pad=8)
            axes[1].axis("off")

            fig.tight_layout()
            out_fig_p = target_out_dir / f"2panel_detection_comparison_{orig_p.stem}.png"
            fig.savefig(out_fig_p, dpi=300, bbox_inches="tight")
            plt.close(fig)
            saved_figures += 1

        print(f" [SUCCESS] Berhasil membuat {saved_figures} figur perbandingan deteksi 2-panel di '{target_out_dir.as_posix()}'")

    print("=" * 75 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Script Evaluasi Metrik Deteksi YOLO Skenario A vs B.")
    parser.add_argument("--weights_a", type=str, default="checkpoints/yolo_original/best.pt", help="Path checkpoint best.pt Skenario A.")
    parser.add_argument("--weights_b", type=str, default="checkpoints/yolo_zero_dce_cbam/best.pt", help="Path checkpoint best.pt Skenario B.")
    parser.add_argument("--output_dir", type=str, default="results/detection", help="Folder output laporan CSV dan gambar 2-panel.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25).")
    parser.add_argument("--limit", type=int, default=6, help="Jumlah sampel gambar visualisasi (default: 6).")
    parser.add_argument("--video_stem", type=str, default=None, help="Nama/ID video sumber (misal: 'video01' atau 'ain').")
    parser.add_argument("--use_master", action="store_true", help="Gunakan Master Dataset Multi-Video (configs/dataset_master_*.yaml).")
    parser.add_argument("--data_a", type=str, default=None, help="Path custom data config YAML Skenario A.")
    parser.add_argument("--data_b", type=str, default=None, help="Path custom data config YAML Skenario B.")

    args = parser.parse_args()

    evaluate_detection(
        weights_a=Path(args.weights_a),
        weights_b=Path(args.weights_b),
        output_dir=Path(args.output_dir),
        conf_thresh=args.conf,
        num_samples=args.limit,
        video_stem=args.video_stem,
        use_master=args.use_master,
        data_a=args.data_a,
        data_b=args.data_b
    )


if __name__ == "__main__":
    main()
