#!/usr/bin/env python3
"""
Script: convert_polygon_to_bbox.py
Deskripsi: Mengonversi file label anotasi YOLO Segmentation (Polygon) berkoordinat banyak
           (class x1 y1 x2 y2 ... xn yn) menjadi format standar YOLO Bounding Box
           (class x_center y_center width height) ber-5 token.

Penggunaan:
    python scripts/dataset/convert_polygon_to_bbox.py --labels_dir data/original/labels
"""

import argparse
import sys
from pathlib import Path


def convert_polygon_line_to_bbox(line: str) -> str:
    """
    Mengonversi satu baris string anotasi polygon ke baris bounding box YOLO (5 token).
    Jika baris sudah berisi 5 token, dikembalikan tanpa perubahan.
    """
    tokens = line.strip().split()
    if not tokens:
        return line

    if len(tokens) == 5:
        return line.strip()

    if len(tokens) > 5 and len(tokens) % 2 == 1:
        cls_id = int(tokens[0])
        coords = [float(v) for v in tokens[1:]]
        xs = coords[0::2]
        ys = coords[1::2]

        x_min = max(0.0, min(xs))
        x_max = min(1.0, max(xs))
        y_min = max(0.0, min(ys))
        y_max = min(1.0, max(ys))

        width = x_max - x_min
        height = y_max - y_min
        x_center = x_min + (width / 2.0)
        y_center = y_min + (height / 2.0)

        # Pastikan batas berada dalam kisaran valid [0.0, 1.0]
        x_center = min(max(x_center, 0.0), 1.0)
        y_center = min(max(y_center, 0.0), 1.0)
        width = min(max(width, 1e-6), 1.0)
        height = min(max(height, 1e-6), 1.0)

        return f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

    return line.strip()


def convert_labels_in_dir(labels_dir: Path) -> tuple:
    """
    Memindai dan mengonversi seluruh file label .txt di labels_dir dari polygon ke bounding box.
    """
    if not labels_dir.exists():
        print(f"[ERROR] Direktori label tidak ditemukan: {labels_dir}")
        return 0, 0, 0

    txt_files = list(labels_dir.rglob("*.txt"))
    converted_files = 0
    converted_lines_total = 0

    print("\n" + "=" * 65)
    print(" MEMULAI KONVERSI LABEL POLYGON (YOLO SEGMENT) KE BOUNDING BOX")
    print("=" * 65)
    print(f" Direktori Label : {labels_dir.as_posix()}")
    print(f" Total File .txt : {len(txt_files)}")
    print("-" * 65)

    for txt_p in txt_files:
        try:
            with open(txt_p, "r", encoding="utf-8") as f:
                raw_lines = [l.strip() for l in f.readlines() if l.strip()]

            if not raw_lines:
                continue

            file_modified = False
            new_lines = []
            converted_lines_in_file = 0

            for line in raw_lines:
                converted_line = convert_polygon_line_to_bbox(line)
                if converted_line != line.strip():
                    file_modified = True
                    converted_lines_in_file += 1
                new_lines.append(converted_line)

            if file_modified:
                with open(txt_p, "w", encoding="utf-8") as f:
                    f.write("\n".join(new_lines) + "\n")
                converted_files += 1
                converted_lines_total += converted_lines_in_file
                print(f" [KONVERSI] {txt_p.name}: {converted_lines_in_file} baris polygon -> bbox")

        except Exception as e:
            print(f" [ERROR] Gagal memproses file {txt_p.name}: {e}")

    print("-" * 65)
    print(f" RINGKASAN KONVERSI:")
    print(f" File Terkonversi  : {converted_files} dari {len(txt_files)} file")
    print(f" Total Baris Ubah  : {converted_lines_total} baris polygon")
    print("=" * 65 + "\n")

    return len(txt_files), converted_files, converted_lines_total


def main():
    parser = argparse.ArgumentParser(
        description="Konversi label format YOLO Segmentation (Polygon) ke YOLO Bounding Box."
    )
    parser.add_argument(
        "--labels_dir", type=str, default="data/original/labels",
        help="Direktori berisi file label YOLO .txt."
    )
    args = parser.parse_args()

    convert_labels_in_dir(Path(args.labels_dir))


if __name__ == "__main__":
    main()
