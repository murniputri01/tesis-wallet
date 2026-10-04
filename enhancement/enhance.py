#!/usr/bin/env python3
"""
Script: enhancement/enhance.py
Deskripsi: Mengeksekusi inferensi batch citra (Enhancement Generator) menggunakan model terlatih Zero-DCE + CBAM.
           Menyimpan citra hasil peningkatan kualitas ke data/enhanced/zero_dce_cbam/.

Penggunaan:
    python enhancement/enhance.py --checkpoint checkpoints/zero_dce_cbam/best_zero_dce_cbam.pth --input_dir data/splits --output_dir data/enhanced/zero_dce_cbam
"""

import argparse               # Pembaca argumen baris perintah terminal (--checkpoint, --scale, dsb)
import gc                     # Garbage collector Python untuk pembersihan alokasi memori
import os                     # Pustaka antarmuka sistem operasi
import sys                    # Pengendali eksekusi sistem & manipulasi modul sys.path
from pathlib import Path      # Pengelola jalur folder/file lintas sistem operasi (Windows/Linux)
from typing import Optional, Tuple  # Penentu tipe data opsional & tuple

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # Lokasi akar proyek
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))              # Sisipkan jalur akar proyek ke sys.path

import torch                  # PyTorch untuk penanganan tensor & inferensi model deep learning
import numpy as np            # NumPy untuk komputasi matriks piksel & manipulasi array
from PIL import Image, ImageFilter  # PIL (Pillow) untuk pembacaan citra, resample, & UnsharpMask
import torchvision.transforms as T  # Transformasi tensor PyTorch (ToTensor)

from enhancement.models.zero_dce_cbam import ZeroDCE_CBAM  # Impor arsitektur model Zero-DCE + CBAM


def _resolve_target_size(
    orig_w: int,
    orig_h: int,
    scale: float,
    target_size: Optional[Tuple[int, int]],
    allow_stretch: bool
) -> Tuple[int, int]:
    """Menentukan resolusi akhir.

    target_size berprioritas lebih tinggi dari scale. Secara default aspect ratio
    citra asli dipertahankan (target_size diperlakukan sebagai kotak pembatas),
    karena stretching mendistorsi bentuk objek kecil seperti walet.
    """
    if target_size is not None:
        tgt_w, tgt_h = target_size
        if allow_stretch:
            return tgt_w, tgt_h
        ratio = min(tgt_w / orig_w, tgt_h / orig_h)
        return max(1, int(round(orig_w * ratio))), max(1, int(round(orig_h * ratio)))
    if scale != 1.0:
        return max(1, int(round(orig_w * scale))), max(1, int(round(orig_h * scale)))
    return orig_w, orig_h


import cv2                    # OpenCV untuk pemrosesan interpolasi citra

def _resize_float(arr: np.ndarray, size: Tuple[int, int], upscale_method: str = "lanczos") -> np.ndarray:
    """Resize citra float32 [H, W, 3] (rentang 0..1) menggunakan OpenCV presisi tinggi.

    Interpolasi dilakukan SEBELUM kuantisasi ke uint8 agar galat pembulatan 8-bit
    dan noise low-light yang diangkat Zero-DCE tidak ikut disebar oleh filter.
    """
    final_w, final_h = size
    method_map = {
        "lanczos": cv2.INTER_LANCZOS4,
        "bicubic": cv2.INTER_CUBIC,
        "bilinear": cv2.INTER_LINEAR,
    }
    interp = method_map.get(upscale_method.lower(), cv2.INTER_LANCZOS4)
    # Memastikan array kontigu dalam memori C untuk menghindari C++ access violation / segfault
    arr_contiguous = np.ascontiguousarray(arr, dtype=np.float32)  # Pastikan array C-contiguous
    return cv2.resize(arr_contiguous, (final_w, final_h), interpolation=interp)  # Resize dengan cv2.resize


def enhance_dataset(
    model_type: str,
    checkpoint_path: Path,
    input_dir: Path,
    output_dir: Path,
    overwrite: bool = False,
    scale: float = 1.0,
    target_size: Optional[Tuple[int, int]] = None,
    upscale_method: str = "lanczos",
    sharpen: float = 0.0,
    allow_stretch: bool = False,
    num_threads: int = 4,
    video_stem: Optional[str] = None
):
    if not checkpoint_path.exists():                   # Memeriksa ketersediaan checkpoint model .pth
        print(f"[ERROR] Checkpoint model tidak ditemukan: {checkpoint_path}")
        return

    # Batasi CPU threads PyTorch agar CPU/laptop tidak hang 100%
    if num_threads > 0:
        torch.set_num_threads(max(1, num_threads))      # Set jumlah CPU threads PyTorch
        print(f"[INFO] PyTorch CPU threads dibatasi ke: {torch.get_num_threads()}")  # Cetak jumlah threads

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Deteksi perangkat GPU atau CPU
    print(f"[INFO] Running Enhancement Generator using {model_type} on {device}")

    # Load Model
    checkpoint = torch.load(checkpoint_path, map_location=device)  # Memuat checkpoint memori model .pth
    cfg = checkpoint.get("config", {})                 # Membaca dictionary konfigurasi dari checkpoint
    n_iters = cfg.get("model", {}).get("n_iters", 8)   # Jumlah iterasi kurva pencerahan (n = 8)
    nf = cfg.get("model", {}).get("nf", 32)            # Jumlah filter konvolusi awal (nf = 32)

    dark_thresh = cfg.get("model", {}).get("dark_threshold", None)
    model = ZeroDCE_CBAM(n_iters=n_iters, nf=nf, dark_threshold=dark_thresh).to(device)  # Inisialisasi model

    model.load_state_dict(checkpoint["model_state_dict"])  # Memuat bobot memori terlatikan
    model.eval()                                       # Set model ke mode evaluasi/inferensi

    # Mapping resampling filter
    resample_map = {
        "lanczos": Image.Resampling.LANCZOS,
        "bicubic": Image.Resampling.BICUBIC,
        "bilinear": Image.Resampling.BILINEAR,
    }
    resample_filter = resample_map.get(upscale_method.lower(), Image.Resampling.LANCZOS)

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}

    # Cari file gambar di input_dir (mendukung subfolder train/val/test)
    image_paths = [                                    # Mencari seluruh file gambar secara rekursif
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in valid_exts
    ]

    # Filter khusus 1 video jika video_stem ditentukan
    if video_stem:
        v_stem_lower = video_stem.lower()
        image_paths = [                                # Filter gambar berbasis video_stem
            p for p in image_paths
            if v_stem_lower in [part.lower() for part in p.parts] or p.name.lower().startswith(v_stem_lower)
        ]

    image_paths.sort()                                 # Mengurutkan file gambar secara alfabetis

    if not image_paths:
        print(f"[WARNING] Tidak ada file gambar ditemukan di '{input_dir.as_posix()}'"
              f"{f' untuk video: {video_stem}' if video_stem else ''}")
        return

    print("\n" + "=" * 65)
    print(f" PROSES ENHANCEMENT DATASET: {model_type.upper()}")
    print("=" * 65)
    print(f" Total Gambar Input : {len(image_paths)}")
    print(f" Target Video       : {video_stem if video_stem else 'SEMUA VIDEO'}")
    print(f" Source Directory   : {input_dir.as_posix()}")
    print(f" Target Directory   : {output_dir.as_posix()}")
    print(f" Scale Factor       : {scale:.2f}x")
    if target_size:
        print(f" Target Resolution  : {target_size[0]} x {target_size[1]}")
    print(f" Upscale Method     : {upscale_method.upper()} (float32 pre-quantization)")
    print(f" Unsharp Mask       : {'OFF' if sharpen <= 0 else f'{sharpen:.0f}%'}")
    print(f" Aspect Ratio       : {'STRETCH (distorsi)' if allow_stretch else 'DIPERTAHANKAN'}")
    print("-" * 65)

    processed_count = 0
    skipped_count = 0

    to_tensor = T.ToTensor()                           # Inisialisasi converter PIL Image ke Tensor [0, 1]

    with torch.no_grad():                              # Non-aktifkan komputasi gradien autograd
        for idx, img_p in enumerate(image_paths, 1):   # Loop membaca setiap gambar
            # Replikasi struktur relasi direktori (misal train/images/xxx.jpg)
            rel_path = img_p.relative_to(input_dir)    # Ambil jalur relatif citra
            out_path = output_dir / rel_path

            if out_path.exists() and not overwrite:
                skipped_count += 1
                continue

            print(f"  [{idx}/{len(image_paths)}] Enhancing: {rel_path.as_posix()}", flush=True)

            out_path.parent.mkdir(parents=True, exist_ok=True)  # Buat direktori subfolder tujuan

            with Image.open(img_p) as pil_img:         # Buka file gambar dengan PIL
                rgb_img = pil_img.convert("RGB")       # Konversi citra ke mode RGB
                orig_w, orig_h = rgb_img.size
                
                # Resolusi di atas 1024px pada CPU PyTorch akan menyebabkan C++ OpenMP stack overflow / segfault (0xC0000005)
                # Resizing tensor ke max_dim 1024px untuk inferensi model dan mengembalikannya ke resolusi asli.
                max_dim = 1024
                if device.type == "cpu" and max(orig_w, orig_h) > max_dim:
                    ratio = max_dim / float(max(orig_w, orig_h))
                    infer_w = max(1, int(round(orig_w * ratio)))
                    infer_h = max(1, int(round(orig_h * ratio)))
                    infer_img = rgb_img.resize((infer_w, infer_h), Image.Resampling.BILINEAR)  # Resize sementara untuk CPU
                else:
                    infer_img = rgb_img

                try:
                    tensor_in = to_tensor(infer_img).unsqueeze(0).to(device)  # Ubah ke tensor 4D [1, 3, H, W]
                    if model_type == "zero_dce":
                        enhanced_t, _ = model(tensor_in)  # Inferensi baseline Zero-DCE
                    else:
                        enhanced_t, _, _, _, _ = model(tensor_in)  # Inferensi Zero-DCE + CBAM
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        torch.cuda.empty_cache()       # Bersihkan cache VRAM GPU jika OOM
                        # Fallback ke CPU untuk citra resolusi sangat tinggi
                        model_cpu = model.to("cpu")
                        tensor_in_cpu = to_tensor(infer_img).unsqueeze(0).to("cpu")
                        if model_type == "zero_dce":
                            enhanced_t, _ = model_cpu(tensor_in_cpu)
                        else:
                            enhanced_t, _, _, _, _ = model_cpu(tensor_in_cpu)
                        model = model.to(device)
                    else:
                        raise e

                # Hasil model tetap float32 (0..1) sampai proses resize selesai
                enh_f = enhanced_t[0].clamp(0.0, 1.0).cpu().numpy().transpose(1, 2, 0).astype(np.float32)  # Konversi ke NumPy
                
                del tensor_in, enhanced_t              # Hapus variabel tensor sementara dari memori
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()           # Bersihkan memori cache GPU VRAM

                final_w, final_h = _resolve_target_size(  # Hitung dimensi akhir
                    orig_w, orig_h, scale, target_size, allow_stretch
                )

                if (enh_f.shape[1], enh_f.shape[0]) != (final_w, final_h):
                    enh_f = _resize_float(enh_f, (final_w, final_h), upscale_method)  # Resize presisi float32

                # Kuantisasi ke 8-bit dilakukan paling akhir
                enh_np = np.clip(enh_f * 255.0, 0, 255).astype(np.uint8)  # Kuantisasi float32 ke uint8 [0, 255]
                out_img = Image.fromarray(enh_np)      # Konversi array NumPy ke PIL Image

                # Interpolasi bersifat low-pass: unsharp mask mengembalikan sebagian
                # frekuensi tinggi sehingga tepi walet tetap kentara setelah upscale.
                if sharpen > 0 and (final_w, final_h) != (orig_w, orig_h):
                    out_img = out_img.filter(          # Terapkan filter penajaman UnsharpMask
                        ImageFilter.UnsharpMask(radius=1.0, percent=int(round(sharpen)), threshold=3)
                    )

                out_img.save(out_path)                 # Simpan gambar ter-enhance ke disk
                
                # Salin label .txt YOLO jika ada agar struktur dataset tetap utuh (images & labels)
                path_parts = img_p.parts
                if "images" in path_parts:
                    img_idx = path_parts.index("images")
                    base_dir = Path(*path_parts[:img_idx])
                    rel_path_from_images = Path(*path_parts[img_idx+1:])
                    lbl_src = base_dir / "labels" / rel_path_from_images.with_suffix(".txt")
                    if lbl_src.exists():               # Memeriksa ketersediaan file label .txt
                        out_parts = out_path.parts
                        if "images" in out_parts:
                            out_img_idx = out_parts.index("images")
                            out_base = Path(*out_parts[:out_img_idx])
                            lbl_dst = out_base / "labels" / rel_path_from_images.with_suffix(".txt")
                            lbl_dst.parent.mkdir(parents=True, exist_ok=True)  # Buat sub-folder labels tujuan
                            lbl_dst.write_text(lbl_src.read_text(encoding="utf-8"), encoding="utf-8")  # Menyalin label .txt

                processed_count += 1
                gc.collect()                           # Eksekusi garbage collection pembersihan RAM

    print(f"\n Processing Finished : {processed_count} gambar di-enhance, {skipped_count} dilewati (sudah ada).")
    print("=" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Batch Enhancement Generator untuk Zero-DCE+CBAM.")
    parser.add_argument(
        "--model",
        type=str,
        default="zero_dce_cbam",
        choices=["zero_dce_cbam"],
        help="Jenis model enhancement (default: 'zero_dce_cbam')."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path ke file checkpoint PyTorch (.pth)."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/splits",
        help="Direktori asal gambar mentah/split (default: data/splits)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Direktori tujuan penyimpanan gambar enhanced."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Timpa file jika sudah ada di folder tujuan."
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Faktor skala perbesaran resolusi (default: 1.0 = resolusi asli)."
    )
    parser.add_argument(
        "--target_size",
        type=int,
        nargs=2,
        default=None,
        metavar=("WIDTH", "HEIGHT"),
        help="Target resolusi spesifik (Lebar Tinggi), misal: --target_size 1280 1440. "
             "Secara default diperlakukan sebagai kotak pembatas (aspect ratio dipertahankan)."
    )
    parser.add_argument(
        "--allow_stretch",
        action="store_true",
        help="Izinkan --target_size mengubah aspect ratio (objek akan terdistorsi)."
    )
    parser.add_argument(
        "--upscale_method",
        type=str,
        default="lanczos",
        choices=["lanczos", "bicubic", "bilinear"],
        help="Metode interpolasi presisi tinggi (default: lanczos)."
    )
    parser.add_argument(
        "--sharpen",
        type=float,
        default=0.0,
        help="Kekuatan unsharp mask (persen) setelah upscale untuk memulihkan ketajaman tepi. "
             "0 = nonaktif, rekomendasi 60-100."
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=4,
        help="Jumlah CPU threads PyTorch (default: 4 untuk mencegah CPU 100% hang)."
    )
    parser.add_argument(
        "--video_stem",
        type=str,
        default=None,
        help="Hanya proses gambar milik nama video tertentu (misal: 'ain' atau 'video01')."
    )

    args = parser.parse_args()

    enhance_dataset(
        model_type=args.model,
        checkpoint_path=Path(args.checkpoint),
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        overwrite=args.overwrite,
        scale=args.scale,
        target_size=tuple(args.target_size) if args.target_size else None,
        upscale_method=args.upscale_method,
        sharpen=args.sharpen,
        allow_stretch=args.allow_stretch,
        num_threads=args.num_threads,
        video_stem=args.video_stem
    )


if __name__ == "__main__":
    main()
