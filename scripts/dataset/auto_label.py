#!/usr/bin/env python3
"""
Script: auto_label.py
Deskripsi: Melakukan auto-labeling (semi-otomatis) pada gambar di data/selected/
           menggunakan model YOLO terlatih (.pt) untuk menghasilkan file label YOLO (.txt).

Penggunaan:
    python scripts/auto_label.py --model path/to/best_yolo.pt --input data/selected --conf 0.25
"""

import argparse           # Pembaca argumen baris perintah terminal (--model, --input, --conf)
import shutil             # Penyalinan/pemindahan file citra ke folder tujuan di disk
import sys                # Pengendali sistem operasi & penutup eksekusi jika error (sys.exit)
from pathlib import Path  # Pengelola jalur folder/file lintas sistem operasi (Windows/Linux)

try:
    from ultralytics import YOLO  # Mengimpor modul arsitektur model deteksi YOLO dari ultralytics
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False


def parse_group_id(filename: str) -> str:
    """Meng-ekstrak ID grup/sumber video dari nama file (misal: 'video01' dari 'video01_frame_000001.jpg')."""
    if "_frame_" in filename:
        return filename.split("_frame_")[0]  # Memotong nama file berdasarkan pola pemisah '_frame_'
    return "default_group"


def auto_label_images(
    model_path: Path,
    input_dir: Path,
    images_output_dir: Path,
    labels_output_dir: Path,
    conf_threshold: float = 0.25
):
    if not HAS_ULTRALYTICS:  # Memeriksa apakah library ultralytics telah terinstall
        print("[ERROR] Library 'ultralytics' belum terinstall.")
        print("Silakan install dengan perintah: pip install ultralytics")
        sys.exit(1)           # Hentikan eksekusi dengan status error 1

    model_str = str(model_path)  # Mengonversi objek Path menjadi string lokasi file
    is_standard_model = model_str.startswith("yolo") and (model_str.endswith(".pt") or model_str.endswith(".yaml"))

    if not model_path.exists() and not is_standard_model:  # Memeriksa ketersediaan weights model YOLO
        print(f"[ERROR] File checkpoint model YOLO tidak ditemukan di: '{model_path}'")
        print("[PETUNJUK]:")
        print(" 1. 'path/to/yolov8_walet.pt' adalah teks contoh. Ganti dengan lokasi file .pt asli Anda (misal: 'checkpoints/best.pt').")
        print(" 2. Atau jika ingin menggunakan pretrained YOLOv8 resmi, jalankan: --model yolov8n.pt")
        sys.exit(1)

    if not input_dir.exists():  # Memeriksa apakah direktori input gambar ada
        print(f"[ERROR] Direktori input tidak ditemukan: {input_dir}")
        sys.exit(1)

    images_output_dir.mkdir(parents=True, exist_ok=True)  # Membuat folder output gambar jika belum ada
    labels_output_dir.mkdir(parents=True, exist_ok=True)  # Membuat folder output label .txt jika belum ada

    print("\n" + "=" * 65)
    print(" MEMULAI AUTO-LABELING (SEMI-OTOMATIS YOLO)")
    print("=" * 65)
    print(f" Model Weights     : {model_path.as_posix()}")  # Mengonversi path ke format POSIX dengan '/'
    print(f" Input Directory   : {input_dir.as_posix()}")
    print(f" Confidence Thresh : {conf_threshold}")
    print(f" Output Images Dir : {images_output_dir.as_posix()}")
    print(f" Output Labels Dir : {labels_output_dir.as_posix()}")
    print("-" * 65)

    model = YOLO(str(model_path))  # Memuat pretrained model/weights YOLO ke memori

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    image_paths = [  # Mencari seluruh file gambar di direktori input secara rekursif
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in valid_exts
    ]
    image_paths.sort()  # Mengurutkan nama file gambar secara alfabetis

    if not image_paths:
        print(f"[WARNING] Tidak ada file gambar ditemukan di {input_dir}")
        return

    labeled_count = 0
    empty_label_count = 0

    for img_p in image_paths:  # Loop berulang membaca setiap file citra
        # Predict dengan YOLO: Menjalankan inferensi deteksi objek pada 1 citra
        results = model.predict(source=str(img_p), conf=conf_threshold, verbose=False)

        group_id = parse_group_id(img_p.name)     # Mengekstrak ID grup/sumber video
        img_dest_dir = images_output_dir / group_id
        lbl_dest_dir = labels_output_dir / group_id
        img_dest_dir.mkdir(parents=True, exist_ok=True)  # Buat sub-folder gambar per grup
        lbl_dest_dir.mkdir(parents=True, exist_ok=True)  # Buat sub-folder label per grup

        lbl_filename = f"{img_p.stem}.txt"       # Menentukan nama file label .txt pasangan
        lbl_save_path = lbl_dest_dir / lbl_filename
        img_save_path = img_dest_dir / img_p.name

        shutil.copy2(img_p, img_save_path)        # Menyalin file gambar asli ke direktori tujuan data/original/images

        lines = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes              # Mengekstrak bounding box hasil prediksi model
            for box in boxes:
                cls_id = 0                        # Memetakan seluruh deteksi ke class_id 0 (walet)
                xywhn = box.xywhn[0].tolist()     # Mengambil koordinat bbox ter-normalisasi [x_center, y_center, w, h]
                x_center, y_center, w, h = xywhn
                lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")  # Format baris label YOLO

        # open(): Membuka dan menuliskan koordinat anotasi ke dalam file label .txt
        with open(lbl_save_path, "w", encoding="utf-8") as f:
            if lines:
                f.write("\n".join(lines) + "\n")   # Menuliskan baris-baris bounding box ke file .txt
                labeled_count += 1
            else:
                empty_label_count += 1

    print(" RINGKASAN AUTO-LABELING:")
    print(f" Total Gambar Diproses       : {len(image_paths)}")
    print(f" Gambar Ter-anotasi (BBox >0): {labeled_count}")
    print(f" Gambar Tanpa Objek (BBox 0) : {empty_label_count}")
    print("=" * 65)
    print("[PETUNJUK ML/ANNOTATOR]: Buka folder 'data/original/images' & 'data/original/labels'")
    print("di aplikasi LabelImg untuk memverifikasi & merevisi bounding box secara manual.\n")


def main():
    parser = argparse.ArgumentParser(description="Auto-Labeling semi-otomatis menggunakan model YOLO.")  # Buat parser terminal
    parser.add_argument("--model", type=str, required=True, help="Path ke model YOLO terlatih (.pt).")   # Argumen --model
    parser.add_argument("--input", type=str, default="data/selected", help="Direktori asal gambar mentah terpilih.")
    parser.add_argument("--images_out", type=str, default="data/original/images", help="Direktori tujuan penyimapan gambar.")
    parser.add_argument("--labels_out", type=str, default="data/original/labels", help="Direktori tujuan penyimpanan label YOLO .txt.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold deteksi (default: 0.25).")

    args = parser.parse_args()  # Mengekstrak argumen yang dimasukkan dari terminal

    auto_label_images(          # Memanggil fungsi eksekusi auto-labeling utama
        model_path=Path(args.model),
        input_dir=Path(args.input),
        images_output_dir=Path(args.images_out),
        labels_output_dir=Path(args.labels_out),
        conf_threshold=args.conf
    )


if __name__ == "__main__":      # Memastikan skrip berjalan hanya jika dieksekusi langsung
    main()
