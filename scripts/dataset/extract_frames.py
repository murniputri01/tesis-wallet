#!/usr/bin/env python3
"""
Script: extract_frames.py
Deskripsi: Membaca video CCTV inframerah dan mengekstrak frame berdasarkan interval waktu
           atau sampling rate yang dapat dikonfigurasi. Nama file hasil ekstraksi seragam dan rapi.

Penggunaan:
    python scripts/extract_frames.py --input data/raw/videos/video01.mp4 --output data/raw/frames --interval 1
    python scripts/extract_frames.py --input data/raw/videos/ --output data/raw/frames --interval 2
"""

import argparse             # Pembaca argumen baris perintah terminal (--input, --interval, dsb)
import sys                  # Modul sistem operasi untuk penanganan error & penghentian program (sys.exit)
from pathlib import Path    # Pengelola jalur folder/file lintas sistem operasi (Windows/Linux)
import cv2                  # Library OpenCV untuk pengolahan video & citra digital


def extract_frames_from_video(video_path: Path, output_dir: Path, interval_sec: float = 1.0, fps_sample: float = None):
    """
    Mengekstrak frame dari sebuah file video berdasarkan interval waktu (detik) atau fps_sample.
    """
    if not video_path.exists():  # Memeriksa keberadaan file video di disk
        print(f"[ERROR] File video tidak ditemukan: {video_path}")
        return 0

    cap = cv2.VideoCapture(str(video_path))   # Membuka dan membaca stream file video dari disk
    if not cap.isOpened():                    # Memeriksa apakah video berhasil dibuka oleh OpenCV
        print(f"[ERROR] Gagal membuka video: {video_path}. File mungkin corrupt atau format tidak didukung.")
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Mengambil jumlah total frame video
    video_fps = cap.get(cv2.CAP_PROP_FPS)                  # Mengambil kecepatan frame per detik (FPS) video

    if video_fps <= 0:
        print(f"[WARNING] FPS video tidak valid ({video_fps}) pada {video_path.name}. Menggunakan fallback FPS=25.")
        video_fps = 25.0

    duration_sec = total_frames / video_fps if total_frames > 0 else 0.0  # Menghitung total durasi video (detik)

    # Hitung interval loncatan frame (frame_step)
    if fps_sample is not None and fps_sample > 0:
        frame_step = max(1, int(round(video_fps / fps_sample)))
    else:
        frame_step = max(1, int(round(video_fps * interval_sec)))

    video_stem = video_path.stem                         # Mengambil nama file video tanpa ekstensi (misal: 'video01')
    video_out_dir = output_dir / video_stem              # Membuat objek lokasi folder tujuan penyimpanan frame
    video_out_dir.mkdir(parents=True, exist_ok=True)     # Membuat folder baru jika belum ada di disk

    print("\n" + "=" * 60)
    print(f" Memproses Video : {video_path.name}")       # Mengambil nama file video lengkap beserta ekstensinya
    print("=" * 60)
    print(f" Total Frame Video : {total_frames}")
    print(f" FPS Video         : {video_fps:.2f}")
    print(f" Durasi Video      : {duration_sec:.2f} detik ({duration_sec / 60:.2f} menit)")
    print(f" Interval Sampling : Setiap {interval_sec} detik (Frame step: {frame_step})")
    print("-" * 60)

    frame_count = 0
    extracted_count = 0

    while True:
        ret, frame = cap.read()  # Membaca 1 frame berikutnya (ret=status True/False, frame=matriks piksel)
        if not ret:              # Hentikan loop jika video sudah habis (ret == False)
            break

        if frame_count % frame_step == 0:  # Ambil frame sesuai interval loncatan frame_step
            extracted_count += 1
            filename = f"{video_stem}_frame_{extracted_count:06d}.jpg"
            save_path = video_out_dir / filename
            success = cv2.imwrite(str(save_path), frame)  # Menyimpan frame menjadi file gambar .jpg di disk
            if not success:
                print(f"[WARNING] Gagal menyimpan frame: {save_path}")

        frame_count += 1

    cap.release()  # Melepaskan memori dan menutup file video setelah selesai diproses

    print(f" Summary Extracted : {extracted_count} frame berhasil disimpan ke '{video_out_dir.as_posix()}'")  # String lokasi folder '/'
    print("=" * 60 + "\n")
    return extracted_count


def main():
    parser = argparse.ArgumentParser(  # Membuat penampung argumen baris perintah terminal
        description="Ekstraksi frame dari video CCTV kandang walet dengan interval konfigurasional."
    )
    parser.add_argument(               # Menentukan opsi parameter input
        "--input",
        type=str,
        default="data/raw/videos",
        help="Path ke file video tunggal atau direktori berisi file video."
    )
    parser.add_argument(               # Menentukan opsi parameter output
        "--output",
        type=str,
        default="data/raw/frames",
        help="Direktori tujuan penyimpanan frame hasil ekstraksi."
    )
    parser.add_argument(               # Menentukan opsi parameter interval ekstraksi (detik)
        "--interval",
        type=float,
        default=1.0,
        help="Interval waktu ekstraksi dalam detik (default: 1.0 detik)."
    )
    parser.add_argument(               # Menentukan opsi parameter FPS sampling rate
        "--fps_sample",
        type=float,
        default=None,
        help="Sampling rate dalam FPS (misal: 0.5 = 1 frame per 2 detik). Meng-override --interval jika diset."
    )

    args = parser.parse_args()          # Membaca & mengumpan nilai parameter dari terminal ke variabel args

    input_path = Path(args.input)       # Konversi string path input menjadi objek Path
    output_dir = Path(args.output)     # Konversi string path output menjadi objek Path

    supported_extensions = {".mp4", ".avi", ".mkv", ".mov", ".dav", ".flv", ".wmv"}

    if input_path.is_file():            # Memeriksa jika input mengarah ke file tunggal
        if input_path.suffix.lower() not in supported_extensions:  # Memeriksa ekstensi format video
            print(f"[ERROR] Format file {input_path.suffix} tidak didukung.")
            sys.exit(1)                 # Hentikan eksekusi dengan status error
        video_files = [input_path]
    elif input_path.is_dir():           # Memeriksa jika input mengarah ke sebuah folder/direktori
        video_files = [                 # Mencari seluruh file video secara rekursif di dalam folder
            p for p in input_path.rglob("*")
            if p.suffix.lower() in supported_extensions
        ]
        video_files.sort()              # Mengurutkan nama file video secara alfabetis
        if not video_files:
            print(f"[WARNING] Tidak ditemukan file video di dalam direktori: {input_path}")
            sys.exit(0)                 # Hentikan eksekusi secara normal
    else:
        print(f"[ERROR] Path input tidak valid: {input_path}")
        sys.exit(1)                     # Hentikan eksekusi dengan status error

    print(f"[INFO] Ditemukan {len(video_files)} file video untuk diproses.")

    total_extracted = 0
    for v_path in video_files:
        count = extract_frames_from_video(v_path, output_dir, interval_sec=args.interval, fps_sample=args.fps_sample)  # Panggil fungsi utama
        total_extracted += count

    print(f"[SELESAI] Total keseluruhan frame yang diekstrak: {total_extracted} frame.")


if __name__ == "__main__":             # Memastikan fungsi main() dipanggil hanya saat skrip dijalankan langsung
    main()
