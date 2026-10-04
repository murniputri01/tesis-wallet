#!/usr/bin/env python3
"""
Script: detection/count_population.py
Deskripsi: Script utama Tahap 4 (Pengujian, Perhitungan Objek/Counting, Estimasi Populasi & Klasifikasi Pertumbuhan)
           Sesuai Diagram 4 Alur Penelitian:
             1. Counting (Perhitungan Jumlah Objek Prediksi vs Ground Truth per Frame)
             2. Evaluasi Error Counting: MAE (Mean Absolute Error) & RMSE (Root Mean Squared Error)
             3. Perbandingan Skenario A (Original) vs Skenario B (Zero-DCE + CBAM Usulan)
             4. Estimasi Laju Pertumbuhan Populasi: Growth(%) = (P_n - P_{n-1}) / P_{n-1} * 100
             5. Klasifikasi Pertumbuhan:
                - +2% s.d. +6% per interval/minggu = "Stabil"
                - < +2% per interval/minggu = "Tidak Baik"

Penggunaan:
    python detection/count_population.py --weights_a checkpoints/yolo_original/best.pt --weights_b checkpoints/yolo_zero_dce_cbam/best.pt
"""

import argparse               # Pembaca argumen baris perintah terminal (--weights_a, --weights_b, dsb)
import sys                    # Pengendali eksekusi sistem & manipulasi modul sys.path
import numpy as np            # NumPy untuk komputasi matriks, rata-rata, & akar kuadrat (sqrt)
import pandas as pd           # Pandas untuk pemrosesan dataframe, pencatatan data counting, & ekspor CSV
from pathlib import Path      # Pengelola jalur folder/file lintas sistem operasi (Windows/Linux)
from typing import Dict, List, Tuple  # Penentu tipe data statis tuple, list, & dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # Lokasi akar proyek
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))              # Sisipkan jalur akar proyek ke sys.path

import matplotlib             # Matplotlib untuk rendering figur visualisasi
matplotlib.use("Agg")         # Set backend Matplotlib non-interaktif
import matplotlib.pyplot as plt  # Pyplot untuk grafik bar chart MAE/RMSE & line chart tren populasi
from ultralytics import YOLO  # Framework YOLO untuk pemanggilan inferensi deteksi & counting


def read_gt_count(label_path: Path) -> int:
    """Membaca jumlah objek Ground Truth dari file label YOLO (.txt)."""
    if not label_path.exists():
        return 0
    lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return len(lines)


def parse_group_id(filename: str) -> str:
    """Meng-ekstrak ID kelompok video/waktu dari nama file (misal: 'video01' dari 'video01_frame_000001.jpg')."""
    if "_frame_" in filename:
        return filename.split("_frame_")[0]
    return "default_group"


def classify_growth(growth_pct: float) -> str:
    """Klasifikasi pertumbuhan populasi sesuai kriteria diagram alur penelitian."""
    if 2.0 <= growth_pct <= 6.0:
        return "Stabil"
    elif growth_pct < 2.0:
        return "Tidak Baik"
    else:
        return "Pertumbuhan Tinggi (> +6%)"


def evaluate_counting_and_population(
    weights_a: Path,
    weights_b: Path,
    output_dir: Path = Path("results/detection"),
    conf_thresh: float = 0.25,
    video_stem: str = None,
    use_master: bool = False
):
    target_out_dir = output_dir / video_stem if video_stem else output_dir
    target_out_dir.mkdir(parents=True, exist_ok=True)

    if use_master or video_stem == "master":
        base_orig_dir = PROJECT_ROOT / "data" / "master_dataset"
        base_enh_dir = PROJECT_ROOT / "data" / "enhanced" / "master_dataset"
    else:
        base_orig_dir = PROJECT_ROOT / "data" / "splits"
        base_enh_dir = PROJECT_ROOT / "data" / "enhanced" / "zero_dce_cbam"

    split_sub = "test"
    test_orig_dir = base_orig_dir / split_sub / "images"
    test_enh_dir = base_enh_dir / split_sub / "images"
    test_label_dir = base_orig_dir / split_sub / "labels"

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    img_paths = [p for p in test_orig_dir.rglob("*") if p.is_file() and p.suffix.lower() in valid_exts] if test_orig_dir.exists() else []

    if video_stem and video_stem != "master":
        v_stem_lower = video_stem.lower()
        img_paths = [
            p for p in img_paths
            if v_stem_lower in [part.lower() for part in p.parts] or p.name.lower().startswith(v_stem_lower)
        ]

    # Fallback ke val / train jika test set kosong
    if not img_paths:
        for fb_split in ["val", "train"]:
            fb_orig_dir = base_orig_dir / fb_split / "images"
            if fb_orig_dir.exists():
                candidates = [p for p in fb_orig_dir.rglob("*") if p.is_file() and p.suffix.lower() in valid_exts]
                if video_stem and video_stem != "master":
                    v_stem_lower = video_stem.lower()
                    candidates = [
                        p for p in candidates
                        if v_stem_lower in [part.lower() for part in p.parts] or p.name.lower().startswith(v_stem_lower)
                    ]
                if candidates:
                    img_paths = candidates
                    test_orig_dir = fb_orig_dir
                    test_enh_dir = base_enh_dir / fb_split / "images"
                    test_label_dir = base_orig_dir / fb_split / "labels"
                    print(f" [INFO] Split 'test' tidak memiliki sampel gambar. Menggunakan split '{fb_split}' untuk evaluasi.")
                    break

    print("\n" + "=" * 78)
    print(f" TAHAP 4: PERHITUNGAN OBJEK (COUNTING), MAE, RMSE & ESTIMASI POPULASI{f' - VIDEO: {video_stem}' if video_stem else ''}")
    print("=" * 78)

    img_paths.sort()

    if not img_paths:
        print(f"[ERROR] Tidak ada citra pengujian ditemukan di '{test_orig_dir.as_posix()}'"
              f"{f' untuk video: {video_stem}' if video_stem else ''}")
        return

    model_a = YOLO(str(weights_a)) if weights_a.exists() else None
    model_b = YOLO(str(weights_b)) if weights_b.exists() else None

    if not model_a and not model_b:
        print("[ERROR] Tidak ada checkpoint model yang ditemukan!")
        return

    records = []

    for img_p in img_paths:
        rel_p = img_p.relative_to(test_orig_dir)
        enh_p = test_enh_dir / rel_p
        lbl_p = test_label_dir / rel_p.with_suffix(".txt")

        gt_count = read_gt_count(lbl_p)

        # Prediksi Skenario A
        count_a = 0
        if model_a:
            res_a = model_a.predict(str(img_p), conf=conf_thresh, verbose=False)
            boxes_a = res_a[0].boxes if (res_a and len(res_a) > 0) else None
            count_a = len(boxes_a) if boxes_a is not None else 0

        # Prediksi Skenario B
        count_b = 0
        target_img_b = str(enh_p) if enh_p.exists() else str(img_p)
        if model_b:
            res_b = model_b.predict(target_img_b, conf=conf_thresh, verbose=False)
            boxes_b = res_b[0].boxes if (res_b and len(res_b) > 0) else None
            count_b = len(boxes_b) if boxes_b is not None else 0

        group_id = parse_group_id(img_p.name)

        records.append({
            "filename": img_p.name,
            "group_id": group_id,
            "gt_count": gt_count,
            "pred_count_A": count_a,
            "pred_count_B": count_b,
            "err_A": count_a - gt_count,
            "err_B": count_b - gt_count,
            "abs_err_A": abs(count_a - gt_count),
            "abs_err_B": abs(count_b - gt_count),
            "sq_err_A": (count_a - gt_count) ** 2,
            "sq_err_B": (count_b - gt_count) ** 2
        })

    df_detail = pd.DataFrame(records)
    detail_csv = target_out_dir / "per_frame_counting_detail.csv"
    df_detail.to_csv(detail_csv, index=False)

    # ---------------------------------------------------------
    # 1. Kalkulasi Metrik Error Counting: MAE & RMSE
    # ---------------------------------------------------------
    mae_a = float(df_detail["abs_err_A"].mean())
    rmse_a = float(np.sqrt(df_detail["sq_err_A"].mean()))

    mae_b = float(df_detail["abs_err_B"].mean())
    rmse_b = float(np.sqrt(df_detail["sq_err_B"].mean()))

    df_counting_summary = pd.DataFrame([
        {
            "Skenario": "Skenario A (Original Low-Light)",
            "Total_Citra": len(df_detail),
            "Total_GT_Object": int(df_detail["gt_count"].sum()),
            "Total_Pred_Object": int(df_detail["pred_count_A"].sum()),
            "MAE": round(mae_a, 4),
            "RMSE": round(rmse_a, 4)
        },
        {
            "Skenario": "Skenario B (Zero-DCE + CBAM Usulan)",
            "Total_Citra": len(df_detail),
            "Total_GT_Object": int(df_detail["gt_count"].sum()),
            "Total_Pred_Object": int(df_detail["pred_count_B"].sum()),
            "MAE": round(mae_b, 4),
            "RMSE": round(rmse_b, 4)
        }
    ])

    counting_csv = target_out_dir / "counting_error_metrics.csv"
    df_counting_summary.to_csv(counting_csv, index=False)

    print("\n" + "-" * 78)
    print(" METRIK EVALUASI KESALAHAN PERHITUNGAN (COUNTING ERROR): MAE & RMSE")
    print("-" * 78)
    print(df_counting_summary.to_string(index=False))

    # ---------------------------------------------------------
    # 2. Perhitungan Laju Pertumbuhan Populasi (Growth %) & Klasifikasi
    # ---------------------------------------------------------
    grouped = df_detail.groupby("group_id").agg({
        "gt_count": "sum",
        "pred_count_A": "sum",
        "pred_count_B": "sum"
    }).reset_index()

    population_rows = []
    groups = list(grouped["group_id"])

    for idx, row in grouped.iterrows():
        grp = row["group_id"]
        pop_gt = row["gt_count"]
        pop_a = row["pred_count_A"]
        pop_b = row["pred_count_B"]

        if idx == 0:
            growth_gt_pct = 0.0
            growth_a_pct = 0.0
            growth_b_pct = 0.0
            status_gt = "Baseline (Periode 1)"
            status_a = "Baseline (Periode 1)"
            status_b = "Baseline (Periode 1)"
        else:
            prev_gt = grouped.loc[idx - 1, "gt_count"]
            prev_a = grouped.loc[idx - 1, "pred_count_A"]
            prev_b = grouped.loc[idx - 1, "pred_count_B"]

            growth_gt_pct = ((pop_gt - prev_gt) / prev_gt * 100.0) if prev_gt > 0 else 0.0
            growth_a_pct = ((pop_a - prev_a) / prev_a * 100.0) if prev_a > 0 else 0.0
            growth_b_pct = ((pop_b - prev_b) / prev_b * 100.0) if prev_b > 0 else 0.0

            status_gt = classify_growth(growth_gt_pct)
            status_a = classify_growth(growth_a_pct)
            status_b = classify_growth(growth_b_pct)

        population_rows.append({
            "Group_ID": grp,
            "GT_Population": pop_gt,
            "Pred_Pop_Skenario_A": pop_a,
            "Pred_Pop_Skenario_B": pop_b,
            "Growth_GT(%)": round(growth_gt_pct, 2),
            "Growth_Skenario_A(%)": round(growth_a_pct, 2),
            "Growth_Skenario_B(%)": round(growth_b_pct, 2),
            "Klasifikasi_GT": status_gt,
            "Klasifikasi_Skenario_A": status_a,
            "Klasifikasi_Skenario_B": status_b
        })

    df_pop = pd.DataFrame(population_rows)
    pop_csv = target_out_dir / "population_estimation_report.csv"
    df_pop.to_csv(pop_csv, index=False)

    print("\n" + "-" * 78)
    print(" ESTIMASI POPULASI, LAJU PERTUMBUHAN GROWTH(%), DAN KLASIFIKASI PERTUMBUHAN")
    print("-" * 78)
    print(df_pop.to_string(index=False))

    # ---------------------------------------------------------
    # 3. Generasi Grafik Figur Evaluasi (300 DPI)
    # ---------------------------------------------------------
    # A. Bar Chart MAE & RMSE
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(2)
    width = 0.35

    rects1 = ax.bar(x - width/2, [mae_a, rmse_a], width, label="Skenario A (Original)", color="#e74c3c")
    rects2 = ax.bar(x + width/2, [mae_b, rmse_b], width, label="Skenario B (Zero-DCE+CBAM)", color="#2ecc71")

    ax.set_ylabel("Nilai Error (Makin Rendah Makin Baik)", fontsize=10, fontweight="bold")
    ax.set_title("Perbandingan Error Counting: MAE vs RMSE", fontsize=12, fontweight="bold", pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(["MAE", "RMSE"], fontsize=10, fontweight="bold")
    ax.legend(frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for rect in rects1 + rects2:
        height = rect.get_height()
        ax.annotate(f"{height:.2f}",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

    fig.tight_layout()
    chart_err_p = target_out_dir / "counting_error_comparison.png"
    fig.savefig(chart_err_p, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # B. Line Chart Tren Populasi
    fig, ax = plt.subplots(figsize=(8, 4.5))
    group_labels = df_pop["Group_ID"]

    ax.plot(group_labels, df_pop["GT_Population"], marker="o", linewidth=2.5, label="Ground Truth", color="#34495e")
    ax.plot(group_labels, df_pop["Pred_Pop_Skenario_A"], marker="s", linestyle="--", linewidth=2, label="Skenario A (Original)", color="#e74c3c")
    ax.plot(group_labels, df_pop["Pred_Pop_Skenario_B"], marker="^", linestyle="-.", linewidth=2, label="Skenario B (Zero-DCE+CBAM)", color="#2ecc71")

    ax.set_xlabel("Kelompok Video / Interval Waktu", fontsize=10, fontweight="bold")
    ax.set_ylabel("Total Estimasi Populasi (Ekor)", fontsize=10, fontweight="bold")
    ax.set_title("Tren Estimasi Populasi Burung Walet antar Interval Waktu", fontsize=12, fontweight="bold", pad=10)
    ax.legend(frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5)

    for i, txt in enumerate(df_pop["Klasifikasi_Skenario_B"]):
        ax.annotate(f"{df_pop['Pred_Pop_Skenario_B'].iloc[i]} ({txt})",
                    (group_labels.iloc[i], df_pop["Pred_Pop_Skenario_B"].iloc[i]),
                    textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8, fontweight="bold")

    fig.tight_layout()
    chart_pop_p = target_out_dir / "population_growth_trend.png"
    fig.savefig(chart_pop_p, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("\n" + "=" * 78)
    print(f" [SUCCESS] Laporan CSV & Figur Grafik Tahap 4 Berhasil Disimpan:")
    print(f"  - Laporan Error Counting CSV : {counting_csv.as_posix()}")
    print(f"  - Laporan Populasi CSV       : {pop_csv.as_posix()}")
    print(f"  - Grafik Error Bar Chart     : {chart_err_p.as_posix()}")
    print(f"  - Grafik Tren Populasi       : {chart_pop_p.as_posix()}")
    print("=" * 78 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Script Tahap 4: Perhitungan Objek, MAE/RMSE, & Estimasi Populasi Walet.")
    parser.add_argument("--weights_a", type=str, default="checkpoints/yolo_original/best.pt", help="Path checkpoint best.pt Skenario A.")
    parser.add_argument("--weights_b", type=str, default="checkpoints/yolo_zero_dce_cbam/best.pt", help="Path checkpoint best.pt Skenario B.")
    parser.add_argument("--output_dir", type=str, default="results/detection", help="Folder output laporan dan grafik.")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold (default: 0.35).")
    parser.add_argument("--video_stem", type=str, default=None, help="Nama/ID video sumber (misal: 'video01' atau '5_agustus').")
    parser.add_argument("--use_master", action="store_true", help="Gunakan Master Dataset Multi-Video (Opsi 2).")

    args = parser.parse_args()

    evaluate_counting_and_population(
        weights_a=Path(args.weights_a),
        weights_b=Path(args.weights_b),
        output_dir=Path(args.output_dir),
        conf_thresh=args.conf,
        video_stem=args.video_stem,
        use_master=args.use_master
    )


if __name__ == "__main__":
    main()
