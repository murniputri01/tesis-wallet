#!/usr/bin/env python3
"""
Script: scripts/dataset/build_clean_dataset.py
Deskripsi: Membangun split dataset YOLO yang bersih dari duplikasi anotasi dan
           kebocoran data (data leakage) antar split.

Latar Belakang Masalah:
  Satu frame fisik yang sama dapat memiliki lebih dari satu berkas anotasi, yaitu
  versi hasil pelabelan otomatis lama (umumnya kosong atau 0-2 objek) dan versi
  ekspor Roboflow hasil pelabelan manual (puluhan objek), dengan nama berkas
  berbeda, misal:
      7_agustus_frame_000001.jpg                    ->  1 objek  (auto label lama)
      7_agustus_frame_000001_jpg.rf.oIJ4vq....jpg   -> 74 objek  (Roboflow)
  Karena namanya berbeda, skrip split lama memperlakukan keduanya sebagai sampel
  independen sehingga frame yang sama tersebar di train, val, dan test sekaligus,
  dan model dilatih dengan dua kebenaran yang saling bertentangan.

Solusi:
  1. Identitas frame dinormalisasi (imbuhan '_jpg.rf.<hash>' dari Roboflow dibuang).
  2. Setiap identitas frame hanya diwakili satu varian: varian dengan jumlah objek
     anotasi terbanyak (anotasi manual Roboflow mengalahkan auto label kosong).
  3. Frame tanpa objek dibuang secara baku (--keep_empty untuk mempertahankannya)
     agar model tidak dilatih dengan sinyal negatif pada citra yang penuh objek.
  4. Split dilakukan atas identitas frame sehingga satu frame mustahil muncul di
     lebih dari satu split.

Penggunaan:
    python scripts/dataset/build_clean_dataset.py --output_dir data/splits_clean
    python scripts/dataset/build_clean_dataset.py --sources 7_agustus --output_dir data/splits_7agustus
"""

import argparse                              # Pembaca argumen baris perintah terminal
import random                                # Pengacak urutan sampel untuk pembagian split
import re                                    # Pencocokan pola imbuhan nama berkas Roboflow
import shutil                                # Penyalinan berkas citra & label ke folder tujuan
import sys                                   # Pengendali kode keluar program
from collections import defaultdict          # Kamus otomatis untuk pengelompokan frame
from pathlib import Path                     # Pengelola jalur folder/file lintas sistem operasi

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Lokasi akar proyek

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}  # Ekstensi citra yang dikenali

# Pola imbuhan berkas ekspor Roboflow, contoh: '_jpg.rf.oIJ4vqSNykuAX0uyyqVL'
ROBOFLOW_SUFFIX = re.compile(r"_(jpg|jpeg|png)\.rf\.[0-9a-zA-Z]+$", re.IGNORECASE)


def frame_identity(stem: str) -> str:
    """Menormalisasi nama berkas menjadi identitas frame fisik yang unik."""
    return ROBOFLOW_SUFFIX.sub("", stem)


def count_objects(label_path) -> int:
    """Menghitung jumlah baris anotasi bounding box valid pada satu berkas label YOLO."""
    if label_path is None or not label_path.exists():
        return 0
    n = 0
    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if len(line.split()) == 5:
            n += 1
    return n


def collect_candidates(images_dir: Path, labels_dir: Path, sources):
    """Mengumpulkan pasangan citra-label dan mengelompokkannya per identitas frame."""
    label_by_stem = {p.stem: p for p in labels_dir.rglob("*.txt")}
    groups = defaultdict(list)

    for img in sorted(images_dir.rglob("*")):
        if img.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        ident = frame_identity(img.stem)
        source = ident.split("_frame_")[0] if "_frame_" in ident else "default"
        if sources and source not in sources:
            continue
        label = label_by_stem.get(img.stem)
        groups[ident].append({
            "image": img,
            "label": label,
            "source": source,
            "n_obj": count_objects(label),
            "is_roboflow": bool(ROBOFLOW_SUFFIX.search(img.stem)),
        })

    return groups


def pick_best(variants):
    """Memilih satu varian terbaik per frame: anotasi terkaya, lalu ekspor Roboflow."""
    return sorted(variants, key=lambda v: (v["n_obj"], v["is_roboflow"]), reverse=True)[0]


def build(
    images_dir: Path,
    labels_dir: Path,
    output_dir: Path,
    sources,
    drop_empty: bool,
    min_objects: int,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> int:
    """Menjalankan seluruh proses deduplikasi, pembagian split, dan penyalinan berkas."""
    groups = collect_candidates(images_dir, labels_dir, sources)
    if not groups:
        print(f"[ERROR] Tidak ada citra ditemukan pada {images_dir}")
        return 1

    selected = []
    dropped_empty = []
    dropped_sparse = []
    discarded_dupes = 0

    for ident, variants in sorted(groups.items()):
        best = pick_best(variants)
        discarded_dupes += len(variants) - 1
        if drop_empty and best["n_obj"] == 0:
            dropped_empty.append(ident)
            continue
        # Frame dengan anotasi jauh lebih sedikit dari semestinya (sisa auto label lama)
        # diperlakukan sebagai anotasi tak lengkap dan dibuang agar tidak menjadi
        # sinyal negatif palsu bagi objek yang sebenarnya ada namun tidak dilabeli.
        if min_objects > 0 and best["n_obj"] < min_objects:
            dropped_sparse.append((ident, best["n_obj"]))
            continue
        best["ident"] = ident
        selected.append(best)

    print("=" * 74)
    print(" PEMBANGUNAN DATASET BERSIH (DEDUPLIKASI FRAME & ANTI-LEAKAGE)")
    print("=" * 74)
    print(f" Identitas frame unik     : {len(groups)}")
    print(f" Varian duplikat dibuang  : {discarded_dupes}")
    print(f" Frame kosong dibuang     : {len(dropped_empty)}")
    print(f" Frame anotasi tak lengkap: {len(dropped_sparse)}"
          + (f" -> {', '.join(f'{i} ({n} objek)' for i, n in dropped_sparse)}" if dropped_sparse else ""))
    print(f" Frame terpakai           : {len(selected)}")
    print(f" Total objek teranotasi   : {sum(s['n_obj'] for s in selected)}")
    print("-" * 74)

    if not selected:
        print("[ERROR] Tidak ada frame beranotasi yang tersisa.")
        return 1

    # Pembagian dilakukan per sumber agar tiap split tetap mewakili seluruh sumber video
    by_source = defaultdict(list)
    for item in selected:
        by_source[item["source"]].append(item)

    rng = random.Random(seed)
    assignment = {"train": [], "val": [], "test": []}

    for source, items in sorted(by_source.items()):
        items = sorted(items, key=lambda v: v["ident"])
        rng.shuffle(items)
        n = len(items)
        n_train = max(1, round(n * train_ratio)) if n > 2 else max(1, n - 1)
        remaining = n - n_train
        n_val = min(max(1, round(n * val_ratio)) if remaining >= 2 else remaining, remaining)
        assignment["train"].extend(items[:n_train])
        assignment["val"].extend(items[n_train:n_train + n_val])
        assignment["test"].extend(items[n_train + n_val:])

    # Penyalinan berkas ke struktur folder standar YOLO
    for split in ("train", "val", "test"):
        for sub in ("images", "labels"):
            d = output_dir / split / sub
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)

    for split, items in assignment.items():
        for item in items:
            dst_img = output_dir / split / "images" / f"{item['ident']}{item['image'].suffix}"
            dst_lbl = output_dir / split / "labels" / f"{item['ident']}.txt"
            shutil.copy2(item["image"], dst_img)
            if item["label"] is not None and item["label"].exists():
                shutil.copy2(item["label"], dst_lbl)
            else:
                dst_lbl.write_text("", encoding="utf-8")

    for split in ("train", "val", "test"):
        items = assignment[split]
        per_src = defaultdict(int)
        for it in items:
            per_src[it["source"]] += 1
        obj = sum(it["n_obj"] for it in items)
        rincian = ", ".join(f"{k}={v}" for k, v in sorted(per_src.items())) or "-"
        print(f" {split:<5}: {len(items):>3} citra | {obj:>5} objek | {rincian}")

    # Verifikasi tidak ada identitas frame yang bocor lintas split
    seen = {}
    leaks = []
    for split, items in assignment.items():
        for it in items:
            if it["ident"] in seen:
                leaks.append((it["ident"], seen[it["ident"]], split))
            seen[it["ident"]] = split

    status = "LOLOS (0 frame bocor)" if not leaks else f"GAGAL ({len(leaks)} frame bocor)"
    print("-" * 74)
    print(f" Verifikasi leakage       : {status}")
    print(f" Direktori keluaran       : {output_dir.as_posix()}")
    print("=" * 74)
    return 0 if not leaks else 1


def main():
    parser = argparse.ArgumentParser(
        description="Bangun split dataset YOLO bersih tanpa duplikasi anotasi & leakage antar split."
    )
    parser.add_argument("--images_dir", type=str, default="data/original/images", help="Direktori asal citra.")
    parser.add_argument("--labels_dir", type=str, default="data/original/labels", help="Direktori asal label.")
    parser.add_argument("--output_dir", type=str, default="data/splits_clean", help="Direktori tujuan split.")
    parser.add_argument("--sources", type=str, nargs="*", default=None, help="Batasi pada sumber tertentu, misal: 7_agustus.")
    parser.add_argument("--keep_empty", action="store_true", help="Pertahankan frame tanpa objek teranotasi.")
    parser.add_argument("--min_objects", type=int, default=0, help="Buang frame yang objek teranotasinya kurang dari nilai ini (sisa auto label lama).")
    parser.add_argument("--train_ratio", type=float, default=0.70, help="Rasio data training (default: 0.70).")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Rasio data validasi (default: 0.15).")
    parser.add_argument("--seed", type=int, default=42, help="Seed acak untuk reproducibility (default: 42).")
    args = parser.parse_args()

    def resolve(p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else PROJECT_ROOT / path

    sys.exit(build(
        images_dir=resolve(args.images_dir),
        labels_dir=resolve(args.labels_dir),
        output_dir=resolve(args.output_dir),
        sources=set(args.sources) if args.sources else None,
        drop_empty=not args.keep_empty,
        min_objects=args.min_objects,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    ))


if __name__ == "__main__":
    main()
