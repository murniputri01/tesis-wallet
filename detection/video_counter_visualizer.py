#!/usr/bin/env python3
"""
Script: detection/video_counter_visualizer.py
Deskripsi: Modul generasi video visualisasi deteksi objek walet (YOLO) dilengkapi
           dengan kotak deteksi (bounding box) dan panel telemetry real-time
           (timestamp video & jumlah walet terhitung secara live per frame).

Fitur Utama:
  - Pembacaan video mentah (.mp4).
  - Inferensi YOLO pada frame asli (Skenario A) dan/atau frame pencerahan Zero-DCE+CBAM (Skenario B).
  - Rendering Bounding Box & Label Confidence Score.
  - Telemetry Overlay Real-Time (Timestamp MM:SS & Counter Jumlah Walet).
  - Mode Tampilan Side-by-Side (Perbandingan Skenario A vs B secara berdampingan).
  - Ekspor Video H.264 (libx264, yuv420p) kompatibel web browser & media player.

Penggunaan:
    python detection/video_counter_visualizer.py --video data/raw/videos/video01.mp4
    python detection/video_counter_visualizer.py --video data/raw/videos/video01.mp4 --side_by_side
"""

import argparse               # Pembaca argumen baris perintah terminal (--video, --weights_a, dsb)
import gc                     # Garbage collector Python untuk pembersihan alokasi memori
import os                     # Pengaturan variabel lingkungan sistem (OpenMP/KMP)
import sys                    # Pengendali eksekusi sistem & manipulasi modul sys.path
import time                   # Pengukur durasi proses rendering video
from pathlib import Path      # Pengelola jalur folder/file lintas sistem operasi (Windows/Linux)
from typing import Optional, Tuple  # Penentu tipe data statis

# Mencegah tabrakan DLL OpenMP C++ pada PyTorch CPU Windows (0xC0000005 segfault)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # Lokasi akar proyek
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))              # Sisipkan jalur akar proyek ke sys.path

import cv2                    # OpenCV untuk ekstraksi frame, menggambar bbox/teks, & manipulasi matriks
import numpy as np            # NumPy untuk manipulasi array piksel citra
import torch                  # PyTorch untuk penanganan tensor & inferensi deep learning
from ultralytics import YOLO  # Framework YOLO untuk deteksi objek walet

from enhancement.models.zero_dce_cbam import ZeroDCE_CBAM  # Impor model pencerahan Zero-DCE + CBAM

# Batas resolusi inferensi pada CPU untuk mencegah PyTorch OpenMP C++ Stack Overflow (0xC0000005)
CPU_MAX_DIM = 720


def load_enhancement_model(checkpoint_path: Path, device: torch.device):
    """Memuat model Zero-DCE + CBAM terlatih."""
    if not checkpoint_path.exists():
        return None
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = checkpoint.get("config", {})
    n_iters = cfg.get("model", {}).get("n_iters", 8)
    nf = cfg.get("model", {}).get("nf", 32)
    dark_thresh = cfg.get("model", {}).get("dark_threshold", None)

    model = ZeroDCE_CBAM(n_iters=n_iters, nf=nf, dark_threshold=dark_thresh).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def enhance_frame(model, frame_bgr: np.ndarray, device: torch.device) -> np.ndarray:
    """Meng-enhance satu frame BGR menggunakan Zero-DCE + CBAM dengan jaminan memori aman."""
    if model is None:
        return frame_bgr
    orig_h, orig_w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb_f = np.ascontiguousarray(rgb.astype(np.float32) / 255.0)

    # Turunkan resolusi inferensi sementara jika berjalan di CPU untuk stabilitas OpenMP
    if device.type == "cpu" and max(orig_w, orig_h) > CPU_MAX_DIM:
        ratio = CPU_MAX_DIM / float(max(orig_w, orig_h))
        infer_w = max(1, int(round(orig_w * ratio)))
        infer_h = max(1, int(round(orig_h * ratio)))
        infer_f = cv2.resize(rgb_f, (infer_w, infer_h), interpolation=cv2.INTER_AREA)
    else:
        infer_f = rgb_f

    tensor_in = torch.from_numpy(
        np.ascontiguousarray(infer_f.transpose(2, 0, 1))
    ).unsqueeze(0).to(device)

    enhanced_t, _, _, _, _ = model(tensor_in)
    enh_f = enhanced_t[0].clamp(0.0, 1.0).cpu().numpy().transpose(1, 2, 0).astype(np.float32)

    del tensor_in, enhanced_t
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if (enh_f.shape[1], enh_f.shape[0]) != (orig_w, orig_h):
        enh_f = cv2.resize(
            np.ascontiguousarray(enh_f), (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4
        )

    enh_u8 = np.clip(enh_f * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(np.ascontiguousarray(enh_u8), cv2.COLOR_RGB2BGR)


def filter_boxes(
    boxes,
    frame_w: int,
    frame_h: int,
    max_area_frac: float,
    max_side_frac: float,
    min_side_px: int,
):
    """Menyaring hasil deteksi YOLO berdasarkan batas ukuran wajar objek walet.

    Statistik anotasi dataset menunjukkan walet adalah objek sangat kecil
    (luas bbox p99 ~0.18% dari luas frame, sisi terpanjang <=15% dimensi frame).
    Kotak yang jauh lebih besar dipastikan false positive dan dibuang di sini.

    Mengembalikan list tuple (x1, y1, x2, y2, conf) yang lolos penyaringan.
    """
    kept = []
    if boxes is None or len(boxes) == 0:
        return kept

    frame_area = float(max(1, frame_w * frame_h))
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        conf = float(box.conf[0].cpu().numpy())
        bw, bh = x2 - x1, y2 - y1

        if bw < min_side_px or bh < min_side_px:      # Terlalu kecil / noise piksel
            continue
        if (bw * bh) / frame_area > max_area_frac:    # Luas kotak melampaui batas walet
            continue
        if bw / float(frame_w) > max_side_frac:       # Kotak terlalu lebar
            continue
        if bh / float(frame_h) > max_side_frac:       # Kotak terlalu tinggi
            continue

        kept.append((x1, y1, x2, y2, conf))

    return kept


def draw_detections_and_hud(
    frame_bgr: np.ndarray,
    detections,
    title: str,
    timestamp_str: str,
    box_color: Tuple[int, int, int] = (0, 255, 0),
    hud_bg_color: Tuple[int, int, int] = (20, 20, 20),
    show_labels: bool = False,
) -> Tuple[np.ndarray, int]:
    """Menggambar Bounding Box deteksi dan Telemetry HUD Overlay di atas frame."""
    annotated = frame_bgr.copy()
    count = 0

    if detections:
        count = len(detections)
        for x1, y1, x2, y2, conf in detections:
            # Gambar Kotak Deteksi (Thickness 1, proporsional objek kecil)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 1)

            if not show_labels:
                continue

            # Label Confidence Score (opsional: teks mudah menutupi objek kecil)
            label_txt = f"Walet {conf:.2f}"
            (t_w, t_h), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)

            # Background label mini di atas box
            lbl_y1 = max(0, y1 - t_h - 4)
            cv2.rectangle(annotated, (x1, lbl_y1), (x1 + t_w + 4, lbl_y1 + t_h + 4), box_color, -1)
            cv2.putText(
                annotated,
                label_txt,
                (x1 + 2, lbl_y1 + t_h + 1),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

    # ---------------------------------------------------------
    # Telemetry HUD Overlay (Top-Left Box)
    # ---------------------------------------------------------
    hud_w, hud_h = 320, 95
    overlay = annotated.copy()

    # Panel Latar Belakang Transparan (Dark Glassmorphism effect)
    cv2.rectangle(overlay, (15, 15), (15 + hud_w, 15 + hud_h), hud_bg_color, -1)
    alpha = 0.75
    cv2.addWeighted(overlay, alpha, annotated, 1 - alpha, 0, annotated)

    # Bingkai Accent Panel HUD
    cv2.rectangle(annotated, (15, 15), (15 + hud_w, 15 + hud_h), box_color, 2)

    # Teks Telemetry
    # 1. Judul Skenario
    cv2.putText(
        annotated,
        title,
        (25, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    # 2. Waktu Video / Timestamp
    cv2.putText(
        annotated,
        f"Waktu  : {timestamp_str}",
        (25, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    # 3. Live Counter Jumlah Walet
    cv2.putText(
        annotated,
        f"Jumlah : {count} Ekor",
        (25, 93),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.80,
        box_color,
        2,
        cv2.LINE_AA,
    )

    return annotated, count


def render_video_counter(
    video_path: Path,
    weights_a: Path,
    weights_b: Optional[Path] = None,
    checkpoint_enhancer: Optional[Path] = None,
    output_path: Optional[Path] = None,
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.45,
    max_det: int = 1000,
    max_area_frac: float = 0.03,
    max_side_frac: float = 0.20,
    min_side_px: int = 3,
    show_labels: bool = False,
    imgsz: int = 640,
    side_by_side: bool = False,
    frame_limit: Optional[int] = None,
    num_threads: int = 4,
):
    if not video_path.exists():
        print(f"[ERROR] File video tidak ditemukan: {video_path}")
        return

    if num_threads > 0:
        torch.set_num_threads(max(1, num_threads))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Model YOLO
    model_a = YOLO(str(weights_a)) if weights_a and weights_a.exists() else None
    model_b = YOLO(str(weights_b)) if weights_b and weights_b.exists() else None

    if not model_a and not model_b:
        print("[ERROR] Tidak ditemukan checkpoint model YOLO!")
        return

    # Load Model Zero-DCE+CBAM (untuk Skenario B jika side_by_side atau Skenario B aktif)
    enh_model = None
    if checkpoint_enhancer and checkpoint_enhancer.exists():
        enh_model = load_enhancement_model(checkpoint_enhancer, device)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Tidak dapat membuka file video: {video_path}")
        return

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Tentukan resolusi aman keluaran video agar konsumsi RAM stabil (terutama pada video 2K/4K)
    max_dim = 960 if side_by_side else 1080
    out_w, out_h = src_w, src_h
    if max(src_w, src_h) > max_dim:
        ratio = max_dim / float(max(src_w, src_h))
        out_w = max(2, int(round(src_w * ratio)) // 2 * 2)
        out_h = max(2, int(round(src_h * ratio)) // 2 * 2)

    writer_w = out_w * 2 if side_by_side else out_w

    if output_path is None:
        video_stem = video_path.stem
        out_dir = PROJECT_ROOT / "results" / "detection" / video_stem / "video"
        out_dir.mkdir(parents=True, exist_ok=True)
        suffix = "_side_by_side_counter" if side_by_side else "_detection_counter"
        output_path = out_dir / f"{video_stem}{suffix}.mp4"
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Inisialisasi Penulis Video H.264
    imageio_writer = None
    cv_writer = None
    try:
        import imageio
        imageio_writer = imageio.get_writer(
            str(output_path),
            format="FFMPEG",
            mode="I",
            fps=src_fps,
            codec="libx264",
            pixelformat="yuv420p",
            macro_block_size=1,
        )
        print(" [INFO] Video Encoder: H.264 (FFMPEG libx264, yuv420p) - Web & Media Player Ready.")
    except Exception as e:
        print(f" [WARN] Gagal menginisialisasi imageio FFMPEG ({e}). Fallback ke OpenCV mp4v...")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        cv_writer = cv2.VideoWriter(str(output_path), fourcc, src_fps, (writer_w, out_h))
        if not cv_writer.isOpened():
            print(f"[ERROR] VideoWriter gagal dibuka untuk: {output_path}")
            cap.release()
            return

    print("\n" + "=" * 70)
    print(" RENDERING VIDEO VISUALISASI DETEKSI & REAL-TIME COUNTER WALET")
    print("=" * 70)
    print(f" Video Sumber   : {video_path.as_posix()}")
    print(f" Resolusi Asli  : {src_w} x {src_h}")
    print(f" Resolusi Render: {writer_w} x {out_h}{' (Side-by-Side)' if side_by_side else ''}")
    print(f" Mode Tampilan  : {'Side-by-Side (Skenario A vs B)' if side_by_side else 'Single Video'}")
    print(f" Perangkat      : {device}")
    print(f" Total Frame    : {total_frames if total_frames > 0 else 'Tidak diketahui'}")
    print(f" File Output    : {output_path.as_posix()}")
    print("-" * 70)

    idx = 0
    t_start = time.time()

    with torch.no_grad():
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_limit and idx >= frame_limit:
                break

            # Rescale frame awal ke resolusi out_w, out_h agar RAM dan CPU ringan & stabil
            if (frame.shape[1], frame.shape[0]) != (out_w, out_h):
                frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

            # Hitung timestamp video (MM:SS)
            seconds = int(idx / src_fps)
            m, s = divmod(seconds, 60)
            time_str = f"{m:02d}:{s:02d}"

            # --- Skenario A (Original) ---
            frame_a = frame.copy()
            dets_a = []
            if model_a:
                res_a = model_a.predict(
                    frame_a, imgsz=imgsz, conf=conf_thresh, iou=iou_thresh,
                    max_det=max_det, verbose=False,
                )
                boxes_a = res_a[0].boxes if (res_a and len(res_a) > 0) else None
                dets_a = filter_boxes(
                    boxes_a, out_w, out_h, max_area_frac, max_side_frac, min_side_px
                )

            annotated_a, count_a = draw_detections_and_hud(
                frame_a,
                dets_a,
                title="SKENARIO A: ORIGINAL",
                timestamp_str=time_str,
                box_color=(0, 165, 255),  # Oranye untuk Skenario A
                show_labels=show_labels,
            )

            if side_by_side:
                # --- Skenario B (Zero-DCE + CBAM) ---
                frame_b = enhance_frame(enh_model, frame, device)
                dets_b = []
                target_model_b = model_b if model_b else model_a
                if target_model_b:
                    res_b = target_model_b.predict(
                        frame_b, imgsz=imgsz, conf=conf_thresh, iou=iou_thresh,
                        max_det=max_det, verbose=False,
                    )
                    boxes_b = res_b[0].boxes if (res_b and len(res_b) > 0) else None
                    dets_b = filter_boxes(
                        boxes_b, out_w, out_h, max_area_frac, max_side_frac, min_side_px
                    )

                annotated_b, count_b = draw_detections_and_hud(
                    frame_b,
                    dets_b,
                    title="SKENARIO B: ZERO-DCE+CBAM",
                    timestamp_str=time_str,
                    box_color=(0, 255, 0),  # Hijau Terang untuk Skenario B
                    show_labels=show_labels,
                )

                out_frame = np.hstack([annotated_a, annotated_b])
                del frame_b, annotated_b
            else:
                out_frame = annotated_a

            del frame_a, annotated_a

            if imageio_writer is not None:
                rgb_frame = cv2.cvtColor(out_frame, cv2.COLOR_BGR2RGB)
                imageio_writer.append_data(rgb_frame)
                del rgb_frame
            elif cv_writer is not None:
                cv_writer.write(out_frame)

            del out_frame

            idx += 1
            if idx % 20 == 0 or idx == total_frames:
                elapsed = time.time() - t_start
                print(f"  [{idx}/{total_frames if total_frames > 0 else '?'}] "
                      f"Frame diproses ({elapsed / idx:.2f} detik/frame)", flush=True)
            gc.collect()

    cap.release()
    if imageio_writer is not None:
        imageio_writer.close()
    if cv_writer is not None:
        cv_writer.release()

    print("-" * 70)
    print(f" SELESAI: {idx} frame video berhasil dibuat dalam {time.time() - t_start:.1f} detik.")
    print(f" Video Output: {output_path.as_posix()}")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generasi Video Deteksi Objek & Real-Time Counter Walet Tahap 4."
    )
    parser.add_argument("--video", type=str, required=True, help="Path file video mentah (.mp4).")
    parser.add_argument("--weights_a", type=str, default="checkpoints/yolo_original/best.pt", help="Weights YOLO Skenario A.")
    parser.add_argument("--weights_b", type=str, default="checkpoints/yolo_zero_dce_cbam/best.pt", help="Weights YOLO Skenario B.")
    parser.add_argument("--checkpoint_enhancer", type=str, default="checkpoints/zero_dce_cbam/best_zero_dce_cbam.pth", help="Checkpoint model Zero-DCE+CBAM.")
    parser.add_argument("--output", type=str, default=None, help="Path video output (.mp4).")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold deteksi (default: 0.25).")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold NMS; lebih rendah = kotak tumpang tindih lebih agresif dibuang (default: 0.45).")
    parser.add_argument("--max_det", type=int, default=1000, help="Batas maksimum deteksi per frame (default: 1000).")
    parser.add_argument("--max_area_frac", type=float, default=0.03, help="Buang kotak yang luasnya melebihi fraksi luas frame ini (default: 0.03).")
    parser.add_argument("--max_side_frac", type=float, default=0.20, help="Buang kotak yang lebar/tingginya melebihi fraksi dimensi frame ini (default: 0.20).")
    parser.add_argument("--min_side_px", type=int, default=3, help="Buang kotak dengan sisi lebih kecil dari nilai ini dalam piksel (default: 3).")
    parser.add_argument("--show_labels", action="store_true", help="Tampilkan teks label confidence di atas setiap kotak.")
    parser.add_argument("--imgsz", type=int, default=640, help="Resolusi inferensi YOLO (default: 640).")
    parser.add_argument("--side_by_side", action="store_true", help="Tampilkan video perbandingan Side-by-Side (A vs B).")
    parser.add_argument("--frame_limit", type=int, default=None, help="Batasi jumlah frame (untuk uji cepat).")
    parser.add_argument("--num_threads", type=int, default=4, help="Jumlah thread CPU PyTorch.")

    args = parser.parse_args()

    render_video_counter(
        video_path=Path(args.video),
        weights_a=Path(args.weights_a),
        weights_b=Path(args.weights_b),
        checkpoint_enhancer=Path(args.checkpoint_enhancer),
        output_path=Path(args.output) if args.output else None,
        conf_thresh=args.conf,
        iou_thresh=args.iou,
        max_det=args.max_det,
        max_area_frac=args.max_area_frac,
        max_side_frac=args.max_side_frac,
        min_side_px=args.min_side_px,
        show_labels=args.show_labels,
        imgsz=args.imgsz,
        side_by_side=args.side_by_side,
        frame_limit=args.frame_limit,
        num_threads=args.num_threads,
    )


if __name__ == "__main__":
    main()
