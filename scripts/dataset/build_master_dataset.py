#!/usr/bin/env python3
"""
Script: scripts/dataset/build_master_dataset.py
Deskripsi: Script penggabung dataset dari berbagai video stem ke dalam 1 Master Dataset (Opsi 2).
           Mendukung pembentukan Master Dataset untuk:
             - Skenario A (Original Low-Light): data/master_dataset/
             - Skenario B (Proposed Zero-DCE + CBAM): data/enhanced/master_dataset/

Penggunaan:
    python scripts/dataset/build_master_dataset.py
    python scripts/dataset/build_master_dataset.py --overwrite
"""

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def clean_stem(name: str) -> str:
    """Membersihkan suffix hash dari export Roboflow (misal: '_jpg.rf.xxxx' -> '')."""
    if "_jpg.rf." in name:
        return name.split("_jpg.rf.")[0]
    elif ".rf." in name:
        return name.split(".rf.")[0]
    return name


def process_and_copy_label(src_path: Path, dest_path: Path):
    """
    Membaca label (bisa berformat Polygon / Segmentation / OBB / BBox),
    mengonversi Polygon menjadi Bounding Box standar 5 kolom (class_id x_center y_center width height),
    dan memastikan class_id selalu 0 (walet).
    """
    if not src_path.exists():
        return
    
    lines = src_path.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
    out_lines = []
    
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        
        try:
            vals = [float(p) for p in parts]
        except ValueError:
            continue
        
        if len(vals) < 5:
            continue
            
        cls_id = 0  # Paksa kelas 0 (walet)
        coords = vals[1:]
        
        if len(coords) == 4:
            # Sudah 5 kolom Bounding Box: [cls, xc, yc, w, h]
            xc, yc, w, h = coords
        elif len(coords) >= 6 and len(coords) % 2 == 0:
            # Format Polygon/Segmentation: [cls, x1, y1, x2, y2, ...]
            xs = coords[0::2]
            ys = coords[1::2]
            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)
            xc = (xmin + xmax) / 2.0
            yc = (ymin + ymax) / 2.0
            w = xmax - xmin
            h = ymax - ymin
        else:
            continue
        
        # Validasi batas koordinat [0.0, 1.0]
        xc = max(0.0, min(1.0, xc))
        yc = max(0.0, min(1.0, yc))
        w = max(0.0001, min(1.0, w))
        h = max(0.0001, min(1.0, h))
        
        out_lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
        
    dest_path.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")


def build_master_dataset(overwrite: bool = True):
    splits_src = PROJECT_ROOT / "data" / "splits"
    enh_src = PROJECT_ROOT / "data" / "enhanced" / "zero_dce_cbam"

    master_orig_dir = PROJECT_ROOT / "data" / "master_dataset"
    master_enh_dir = PROJECT_ROOT / "data" / "enhanced" / "master_dataset"

    print("\n" + "=" * 75)
    print(" BUILDER MASTER DATASET MULTI-VIDEO (OPSI 2 - FLAT STRUCTURE)")
    print("=" * 75)
    print(f" Source Original Splits : {splits_src.as_posix()}")
    print(f" Source Enhanced Images : {enh_src.as_posix()}")
    print(f" Target Master Original : {master_orig_dir.as_posix()}")
    print(f" Target Master Enhanced : {master_enh_dir.as_posix()}")
    print("-" * 75)

    if overwrite:
        for target_dir in (master_orig_dir, master_enh_dir):
            if target_dir.exists():
                shutil.rmtree(target_dir)

    subsets = ["train", "val", "test"]

    for subset in subsets:
        # -------------------------------------------------------------
        # 1. Master Original (Skenario A) - Flat Structure
        # -------------------------------------------------------------
        orig_img_out = master_orig_dir / subset / "images"
        orig_lbl_out = master_orig_dir / subset / "labels"
        orig_img_out.mkdir(parents=True, exist_ok=True)
        orig_lbl_out.mkdir(parents=True, exist_ok=True)

        src_sub_img = splits_src / subset / "images"
        src_sub_lbl = splits_src / subset / "labels"

        # Dukungan alias folder Roboflow ('valid' -> 'val')
        if subset == "val" and not src_sub_img.exists() and (splits_src / "valid" / "images").exists():
            src_sub_img = splits_src / "valid" / "images"
            src_sub_lbl = splits_src / "valid" / "labels"

        copied_orig_img = 0
        copied_orig_lbl = 0

        # Peta seluruh label yang ada di subset labels (termasuk .rf.xxxx)
        lbl_map = {}
        if src_sub_lbl.exists():
            for lbl_p in src_sub_lbl.rglob("*.txt"):
                c_stem = clean_stem(lbl_p.stem)
                # Prioritaskan label yang tidak kosong
                if c_stem not in lbl_map or lbl_p.stat().st_size > lbl_map[c_stem].stat().st_size:
                    lbl_map[c_stem] = lbl_p

        if src_sub_img.exists():
            img_files = [p for p in src_sub_img.rglob("*") if p.is_file() and p.suffix.lower() in VALID_EXTS]
            for img_p in img_files:
                base_stem = clean_stem(img_p.stem)
                clean_img_name = base_stem + img_p.suffix.lower()
                dest_img = orig_img_out / clean_img_name
                
                shutil.copy2(img_p, dest_img)
                copied_orig_img += 1

                # Cari label berpasangan
                if base_stem in lbl_map:
                    matched_lbl = lbl_map[base_stem]
                    clean_lbl_name = base_stem + ".txt"
                    dest_lbl = orig_lbl_out / clean_lbl_name
                    process_and_copy_label(matched_lbl, dest_lbl)
                    copied_orig_lbl += 1

        # -------------------------------------------------------------
        # 2. Master Enhanced Zero-DCE+CBAM (Skenario B) - Flat Structure
        # -------------------------------------------------------------
        enh_img_out = master_enh_dir / subset / "images"
        enh_lbl_out = master_enh_dir / subset / "labels"
        enh_img_out.mkdir(parents=True, exist_ok=True)
        enh_lbl_out.mkdir(parents=True, exist_ok=True)

        copied_enh_img = 0
        copied_enh_lbl = 0

        # Bangun peta gambar enhanced: stem_bersih -> path file
        all_enh_images = {}
        if enh_src.exists():
            for p in enh_src.rglob("*"):
                if p.is_file() and p.suffix.lower() in VALID_EXTS:
                    all_enh_images[clean_stem(p.stem)] = p

        # Iterasi HANYA berdasarkan gambar yang sudah diterima di Skenario A
        for orig_img_p in orig_img_out.glob("*"):
            if not orig_img_p.is_file():
                continue
            base_stem = orig_img_p.stem
            clean_img_name = orig_img_p.name
            clean_lbl_name = base_stem + ".txt"

            # 1. Salin versi Enhanced jika ada, jika tidak fallback ke gambar splits/
            if base_stem in all_enh_images:
                shutil.copy2(all_enh_images[base_stem], enh_img_out / clean_img_name)
            else:
                shutil.copy2(orig_img_p, enh_img_out / clean_img_name)
            copied_enh_img += 1

            # 2. Salin & normalisasi label presisi Roboflow yang sesuai
            orig_lbl_p = orig_lbl_out / clean_lbl_name
            if orig_lbl_p.exists():
                process_and_copy_label(orig_lbl_p, enh_lbl_out / clean_lbl_name)
                copied_enh_lbl += 1

        print(f" [{subset.upper():<5}] Original: {copied_orig_img} gambar, {copied_orig_lbl} label | "
              f"Enhanced: {copied_enh_img} gambar, {copied_enh_lbl} label")

    print("\n" + "=" * 75)
    print(" BUILD MASTER DATASET SELESAI [SUKSES]")
    print(" Config YAML Siap Digunakan:")
    print("   - Skenario A (Original) : configs/dataset_master_original.yaml")
    print("   - Skenario B (Enhanced) : configs/dataset_master_zero_dce_cbam.yaml")
    print("=" * 75 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Builder Master Dataset Multi-Video (Opsi 2).")
    parser.add_argument("--overwrite", action="store_true", help="Timpa file jika sudah ada di master folder.")
    args = parser.parse_args()
    build_master_dataset(overwrite=args.overwrite)


if __name__ == "__main__":
    main()
