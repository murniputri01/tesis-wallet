#!/usr/bin/env python3
"""
Script: scripts/utils/enhance_video.py
Deskripsi: Merekonstruksi video hasil peningkatan kualitas citra Zero-DCE + CBAM.
           Video mentah dibaca frame-per-frame, setiap frame di-enhance oleh model
           terlatih, lalu ditulis kembali menjadi file video (.mp4) dengan FPS
           yang sama seperti video sumber sehingga durasi gerak walet tidak berubah.

Penggunaan:
    python scripts/utils/enhance_video.py --video data/raw/videos/26_agustus.mp4
    python scripts/utils/enhance_video.py --video data/raw/videos/26_agustus.mp4 --side_by_side --max_dim 1280
"""

import argparse               # Pembaca argumen baris perintah terminal (--video, --fps, dsb)
import gc                     # Garbage collector Python untuk pembersihan alokasi memori
import sys                    # Pengendali eksekusi sistem & manipulasi modul sys.path
import time                   # Pengukur durasi proses enhancement per video
from pathlib import Path      # Pengelola jalur folder/file lintas sistem operasi (Windows/Linux)
from typing import Optional   # Penentu tipe data opsional

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # Lokasi akar proyek
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))                     # Sisipkan jalur akar proyek ke sys.path

import cv2                    # OpenCV untuk pembacaan video, interpolasi, & penulisan video
import numpy as np            # NumPy untuk komputasi matriks piksel & manipulasi array
import torch                  # PyTorch untuk penanganan tensor & inferensi model deep learning

from enhancement.models.zero_dce_cbam import ZeroDCE_CBAM     # Impor arsitektur model Zero-DCE + CBAM

# Batas resolusi inferensi pada CPU. Di atas nilai ini PyTorch CPU (OpenMP)
# berpotensi stack overflow / access violation 0xC0000005, sama seperti di enhance.py.
CPU_MAX_DIM = 1024


def load_model(checkpoint_path: Path, device: torch.device):
    """Memuat model Zero-DCE + CBAM beserta konfigurasi yang tersimpan di checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)   # Memuat checkpoint model .pth
    cfg = checkpoint.get("config", {})                              # Membaca dictionary konfigurasi
    n_iters = cfg.get("model", {}).get("n_iters", 8)                # Jumlah iterasi kurva pencerahan
    nf = cfg.get("model", {}).get("nf", 32)                         # Jumlah filter konvolusi awal
    dark_thresh = cfg.get("model", {}).get("dark_threshold", None)  # Ambang gelap adaptif (opsional)

    model = ZeroDCE_CBAM(n_iters=n_iters, nf=nf, dark_threshold=dark_thresh).to(device)  # Inisialisasi model
    model.load_state_dict(checkpoint["model_state_dict"])           # Memuat bobot hasil pelatihan
    model.eval()                                                    # Set model ke mode evaluasi/inferensi
    return model


def enhance_frame(model, frame_bgr: np.ndarray, device: torch.device) -> np.ndarray:
    """Meng-enhance satu frame BGR (uint8) dan mengembalikannya pada resolusi asli.

    Kuantisasi ke 8-bit sengaja ditunda sampai proses resize selesai agar galat
    pembulatan dan noise low-light tidak ikut disebar oleh filter interpolasi.
    """
    orig_h, orig_w = frame_bgr.shape[:2]                            # Dimensi asli frame video

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)                # Konversi BGR (OpenCV) ke RGB (model)
    rgb_f = rgb.astype(np.float32) / 255.0                          # Normalisasi piksel ke rentang 0..1

    # Turunkan resolusi sementara bila inferensi berjalan di CPU pada frame besar
    if device.type == "cpu" and max(orig_w, orig_h) > CPU_MAX_DIM:
        ratio = CPU_MAX_DIM / float(max(orig_w, orig_h))
        infer_w = max(1, int(round(orig_w * ratio)))
        infer_h = max(1, int(round(orig_h * ratio)))
        infer_f = cv2.resize(rgb_f, (infer_w, infer_h), interpolation=cv2.INTER_AREA)  # Downscale untuk inferensi
    else:
        infer_f = rgb_f

    tensor_in = torch.from_numpy(                                   # Ubah array ke tensor 4D [1, 3, H, W]
        np.ascontiguousarray(infer_f.transpose(2, 0, 1))
    ).unsqueeze(0).to(device)

    enhanced_t, _, _, _, _ = model(tensor_in)                       # Inferensi Zero-DCE + CBAM

    enh_f = enhanced_t[0].clamp(0.0, 1.0).cpu().numpy().transpose(1, 2, 0).astype(np.float32)  # Tensor ke NumPy

    del tensor_in, enhanced_t                                       # Hapus tensor sementara dari memori
    if device.type == "cuda":
        torch.cuda.empty_cache()                                    # Bersihkan memori cache GPU VRAM

    # Kembalikan ke resolusi asli frame bila sempat diturunkan untuk inferensi
    if (enh_f.shape[1], enh_f.shape[0]) != (orig_w, orig_h):
        enh_f = cv2.resize(
            np.ascontiguousarray(enh_f), (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4
        )

    enh_u8 = np.clip(enh_f * 255.0, 0, 255).astype(np.uint8)        # Kuantisasi float32 ke uint8 [0, 255]
    return cv2.cvtColor(enh_u8, cv2.COLOR_RGB2BGR)                  # Kembalikan ke BGR untuk VideoWriter


def enhance_video(
    video_path: Path,
    checkpoint_path: Path,
    output_path: Path,
    fps_override: Optional[float] = None,
    max_dim: Optional[int] = None,
    side_by_side: bool = False,
    frame_limit: Optional[int] = None,
    num_threads: int = 4,
):
    if not video_path.exists():                                     # Memeriksa keberadaan file video sumber
        print(f"[ERROR] File video tidak ditemukan: {video_path}")
        return
    if not checkpoint_path.exists():                                # Memeriksa ketersediaan checkpoint .pth
        print(f"[ERROR] Checkpoint model tidak ditemukan: {checkpoint_path}")
        return

    if num_threads > 0:
        torch.set_num_threads(max(1, num_threads))                  # Batasi CPU threads agar laptop tidak hang

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Deteksi perangkat GPU atau CPU
    model = load_model(checkpoint_path, device)                     # Muat model Zero-DCE + CBAM terlatih

    cap = cv2.VideoCapture(str(video_path))                         # Buka video sumber dengan OpenCV
    if not cap.isOpened():
        print(f"[ERROR] Video tidak dapat dibuka oleh OpenCV: {video_path}")
        return

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0                      # FPS asli video sumber
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)      # Jumlah total frame video sumber
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))                  # Lebar frame video sumber
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))                 # Tinggi frame video sumber

    out_fps = fps_override if fps_override else (src_fps if src_fps > 0 else 25.0)  # FPS video keluaran

    # Skala keluaran opsional agar ukuran file video hasil tetap wajar
    out_w, out_h = src_w, src_h
    if max_dim and max(src_w, src_h) > max_dim:
        ratio = max_dim / float(max(src_w, src_h))
        out_w = max(2, int(round(src_w * ratio)) // 2 * 2)          # Bulatkan genap (syarat encoder H.264)
        out_h = max(2, int(round(src_h * ratio)) // 2 * 2)

    writer_w = out_w * 2 if side_by_side else out_w                 # Lebar kanvas ganda untuk mode perbandingan

    output_path.parent.mkdir(parents=True, exist_ok=True)           # Buat direktori tujuan bila belum ada

    # Inisialisasi penulis video H.264 (libx264 yuv420p) via imageio agar kompatibel dengan Roboflow & HTML5 Web
    imageio_writer = None
    cv_writer = None
    try:
        import imageio
        imageio_writer = imageio.get_writer(
            str(output_path),
            fps=out_fps,
            codec="libx264",
            pixelformat="yuv420p",
            macro_block_size=1
        )
        print(f" [INFO] Penulis video: H.264 (libx264, yuv420p) - Kompatibel Roboflow & Web Browser.")
    except Exception as e:
        print(f" [WARN] Gagal menginisialisasi imageio H.264 ({e}). Fallback ke OpenCV VideoWriter (mp4v)...")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        cv_writer = cv2.VideoWriter(str(output_path), fourcc, out_fps, (writer_w, out_h))
        if not cv_writer.isOpened():
            print(f"[ERROR] VideoWriter gagal diinisialisasi untuk: {output_path}")
            cap.release()
            return

    print("\n" + "=" * 65)
    print(" REKONSTRUKSI VIDEO ENHANCED: ZERO-DCE + CBAM (H.264)")
    print("=" * 65)
    print(f" Video Sumber       : {video_path.as_posix()}")
    print(f" Perangkat          : {device}")
    print(f" Resolusi Sumber    : {src_w} x {src_h}")
    print(f" Resolusi Keluaran  : {writer_w} x {out_h}{' (side-by-side)' if side_by_side else ''}")
    print(f" FPS Keluaran       : {out_fps:.2f}")
    print(f" Total Frame        : {total_frames if total_frames > 0 else 'tidak diketahui'}")
    print(f" File Keluaran      : {output_path.as_posix()}")
    print("-" * 65)

    idx = 0
    t_start = time.time()
    with torch.no_grad():                                           # Non-aktifkan komputasi gradien autograd
        while True:
            ok, frame = cap.read()                                  # Baca satu frame dari video sumber
            if not ok:
                break
            if frame_limit and idx >= frame_limit:                  # Hentikan bila batas uji coba tercapai
                break

            enhanced = enhance_frame(model, frame, device)          # Enhance frame dengan Zero-DCE + CBAM

            if (out_w, out_h) != (src_w, src_h):                    # Sesuaikan ke resolusi keluaran
                enhanced = cv2.resize(enhanced, (out_w, out_h), interpolation=cv2.INTER_AREA)
                frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

            out_frame = np.hstack([frame, enhanced]) if side_by_side else enhanced  # Format frame keluaran BGR

            if imageio_writer is not None:
                rgb_frame = cv2.cvtColor(out_frame, cv2.COLOR_BGR2RGB) # Konversi BGR OpenCV ke RGB untuk imageio H.264
                imageio_writer.append_data(rgb_frame)
            elif cv_writer is not None:
                cv_writer.write(out_frame)                          # Tulis frame ke OpenCV VideoWriter

            idx += 1
            if idx % 10 == 0 or idx == total_frames:
                elapsed = time.time() - t_start
                print(f"  [{idx}/{total_frames if total_frames > 0 else '?'}] "
                      f"{elapsed / idx:.2f} detik/frame", flush=True)
            gc.collect()                                            # Eksekusi garbage collection pembersihan RAM

    cap.release()                                                   # Tutup pembaca video sumber
    if imageio_writer is not None:
        imageio_writer.close()                                      # Finalisasi & tutup penulis video imageio H.264
    if cv_writer is not None:
        cv_writer.release()                                         # Finalisasi & tutup file video OpenCV

    print("-" * 65)
    print(f" SELESAI: {idx} frame ditulis dalam {time.time() - t_start:.1f} detik.")
    print(f" Video hasil: {output_path.as_posix()}")
    print("=" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Rekonstruksi video hasil enhancement Zero-DCE + CBAM dari video mentah."
    )
    parser.add_argument("--video", type=str, required=True,
                        help="Path video mentah (misal: data/raw/videos/26_agustus.mp4).")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/zero_dce_cbam/best_zero_dce_cbam.pth",
                        help="Path checkpoint Zero-DCE+CBAM terlatih.")
    parser.add_argument("--output", type=str, default=None,
                        help="Path file video keluaran (default: results/enhancement/<video>/video/<video>_enhanced.mp4).")
    parser.add_argument("--fps", type=float, default=None,
                        help="Paksa FPS keluaran (default: mengikuti FPS video sumber).")
    parser.add_argument("--max_dim", type=int, default=None,
                        help="Batas sisi terpanjang frame keluaran, misal 1280 (default: resolusi asli).")
    parser.add_argument("--side_by_side", action="store_true",
                        help="Tulis video perbandingan: frame asli di kiri, hasil enhancement di kanan.")
    parser.add_argument("--frame_limit", type=int, default=None,
                        help="Batasi jumlah frame yang diproses (untuk uji coba cepat).")
    parser.add_argument("--num_threads", type=int, default=4,
                        help="Jumlah CPU threads PyTorch untuk mencegah laptop hang (default: 4).")

    args = parser.parse_args()                                      # Membaca argumen terminal ke objek args

    video_path = Path(args.video)
    if args.output:
        output_path = Path(args.output)
    else:
        suffix = "_comparison" if args.side_by_side else "_enhanced"
        output_path = (PROJECT_ROOT / "results" / "enhancement" / video_path.stem /
                       "video" / f"{video_path.stem}{suffix}.mp4")

    enhance_video(
        video_path=video_path,
        checkpoint_path=Path(args.checkpoint),
        output_path=output_path,
        fps_override=args.fps,
        max_dim=args.max_dim,
        side_by_side=args.side_by_side,
        frame_limit=args.frame_limit,
        num_threads=args.num_threads,
    )


if __name__ == "__main__":                                          # Memastikan skrip berjalan saat dipanggil langsung
    main()
