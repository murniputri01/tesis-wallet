# Dokumentasi Alur & Algoritma Sistem Deteksi Walet (Master Dataset Opsi 2)

Penelitian: **"MODIFIKASI ZERO-DCE BERBASIS CONVOLUTIONAL BLOCK ATTENTION MODULE (CBAM) UNTUK ENHANCEMENT CITRA LOW-LIGHT CCTV INFRAMERAH PADA KANDANG BURUNG WALET"**

Dokumen ini berisi penjelasan komprehensif mengenai **fungsi masing-masing tahapan (Tahap 1 s.d. Tahap 4)**, alur kerja sistem dari video mentah hingga laporan populasi, serta panduan eksekusi perintah terminal presisi tinggi.

---

## 1. PENJELASAN FUNGSI & TUJUAN MASING-MASING TAHAPAN

Sistem deteksi dan perhitungan burung walet ini dirancang dalam **4 Tahapan Utama** yang saling terintegrasi secara sistematis:

```text
┌───────────────────────────────────────────────────────────────────────────────────┐
│ TAHAP 1: Ekstraksi Data & Preprocessing Enhancement (Zero-DCE + CBAM)             │
│   • Ekstraksi frame CCTV (1 frame/detik) & eliminasi duplikasi visual via SSIM     │
│   • Pelatihan model pencerah citra Zero-DCE + CBAM (Darkness Map Attention)       │
│   • Pencerahan citra low-light CCTV inframerah secara unsupervised               │
│   • Evaluasi metrik kualitas citra non-referensi (Entropy, Contrast, BRISQUE)     │
└─────────────────────────────────────────┬─────────────────────────────────────────┘
                                          │ (Citra Terang)
                                          ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│ TAHAP 2: Anotasi Roboflow & Persiapan Master Dataset (Opsi 2)                     │
│   • Anotasi boks presisi walet pada citra terang di platform Roboflow              │
│   • Export dataset YOLO & pembagian subset 70:15:15 (Group Split)                 │
│   • Builder Master Dataset Multi-Video (build_master_dataset.py)                  │
│   • Normalisasi label polygon -> Bounding Box 5-kolom standar YOLO (class_id 0)   │
└─────────────────────────────────────────┬─────────────────────────────────────────┘
                                          │ (Master Dataset Skenario A & B)
                                          ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│ TAHAP 3: Pelatihan & Evaluasi Deteksi Objek (YOLO12)                              │
│   • Pelatihan YOLO12 Skenario A (Original) vs Skenario B (Zero-DCE + CBAM Usulan) │
│   • Evaluasi metrik deteksi: Precision, Recall, mAP@0.5, mAP@0.5:0.95, & FPS      │
│   • Generasi figur visualisasi bounding box 2-panel A vs B (300 DPI)              │
└─────────────────────────────────────────┬─────────────────────────────────────────┘
                                          │ (Model Weights best.pt)
                                          ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│ TAHAP 4: Perhitungan Objek (Counting), Estimasi Populasi & Video Counter          │
│   • Perhitungan populasi walet per-frame (Prediksi vs Ground Truth)               │
│   • Evaluasi error counting ilmiah: MAE (Mean Abs Error) & RMSE (Root Mean Sq Err) │
│   • Estimasi laju pertumbuhan populasi Growth(%) & Klasifikasi Pertumbuhan        │
│   • Rendering Video MP4 Live Counter: Overlaid Bounding Box + Telemetry HUD        │
└───────────────────────────────────────────────────────────────────────────────────┘
```

---

### 📌 Detail Fungsi & Kode Program Tiap Tahap:

#### 🟢 TAHAP 1 — Ekstraksi Data & Preprocessing Enhancement
* **Fungsi Utama:** Mengubah rekaman video CCTV gelap gulita (`.mp4`) menjadi kumpulan citra berkualitas tinggi yang siap dianotasi.
* **Proses Kunci:**
  1. **Ekstraksi Frame:** Potong video 1 frame per detik (`scripts/dataset/extract_frames.py`).
  2. **Seleksi Frame SSIM:** Membuang frame duplikat jika nilai SSIM $\ge 0.95$ (`scripts/dataset/select_frames.py`).
  3. **Pencerahan Citra Zero-DCE + CBAM:** Melatih jaringan kurva iluminasi yang dipandu *Darkness Map* $D(x) = 1 - I(x)$ untuk menerangkan gambar tanpa membuat warna pudar (`enhancement/train_zero_dce_cbam.py`).
  4. **Inference Enhancement:** Menghasilkan citra terang di `data/enhanced/zero_dce_cbam/` (`enhancement/enhance.py`).
* **Perintah Runner:**
  ```bash
  python scripts/run_tahap1.py --video data/raw/videos/5_agustus.mp4
  ```

---

#### 🔵 TAHAP 2 — Anotasi Roboflow & Persiapan Master Dataset (Opsi 2)
* **Fungsi Utama:** Menyediakan data acuan (*Ground Truth*) dan mengonsolidasikan dataset dari berbagai video ke dalam **Master Dataset Multi-Video (Opsi 2)**.
* **Proses Kunci:**
  1. Citra terang dari Tahap 1 dianotasi di platform Roboflow.
  2. Dataset dieksport format YOLOv8/YOLOv12 dan diekstrak ke `data/splits/`.
  3. **Builder Master Dataset:** Skrip `build_master_dataset.py` mengonversi seluruh label polygon/segmentation menjadi **Bounding Box standar 5-kolom (`0 x_center y_center width height`)**, lalu menyatukannya ke `data/master_dataset/` (Skenario A) dan `data/enhanced/master_dataset/` (Skenario B).
* **Perintah Runner:**
  ```bash
  python scripts/dataset/build_master_dataset.py --overwrite
  ```

---

#### 🟡 TAHAP 3 — Pelatihan & Evaluasi Deteksi Objek (YOLO12)
* **Fungsi Utama:** Melatih model deteksi objek YOLO12 untuk mengenali walet dan mengevaluasi kinerja deteksinya.
* **Proses Kunci:**
  1. Pelatihan model YOLO12 pada Skenario A (Original) dan Skenario B (Zero-DCE + CBAM Usulan) dengan resolusi `--imgsz 640 --batch 8`.
  2. Evaluasi metrik deteksi (Precision, Recall, mAP@0.5, mAP@0.5:0.95, FPS) disimpan ke `results/detection/detection_metrics_report.csv`.
  3. Generasi gambar sampel 2-panel resolusi 300 DPI (`2panel_detection_comparison_*.png`).
* **Perintah Runner:**
  ```bash
  # Training & Evaluasi pada Master Dataset
  python scripts/run_tahap3.py --video_stem master --use_master --epochs 100 --imgsz 640 --batch 8
  ```

---

#### 🔴 TAHAP 4 — Perhitungan Objek (Counting), Estimasi Populasi & Video Counter
* **Fungsi Utama:** Menghitung total populasi burung walet, mengukur tingkat kesalahan counting (MAE & RMSE), serta merekonstruksi video CCTV bertanda bounding box & live counter.
* **Proses Kunci:**
  1. **Counting Per-Frame & Error Analysis:** Menghitung selisih prediksi vs ground truth, menghitung MAE dan RMSE (`detection/count_population.py`).
  2. **Estimasi Populasi & Laju Pertumbuhan:** Hitung $\text{Growth}(\%) = \frac{P_n - P_{n-1}}{P_{n-1}} \times 100\%$ dan klasifikasikan (*"Stabil"*, *"Tidak Baik"*, *"Pertumbuhan Tinggi"*).
  3. **Akurasi Presisi Tinggi (`--conf 0.35`):** Menyetel confidence threshold optimal untuk mengeliminasi false positive (deteksi palsu bayangan/kayu), menghasilkan akurasi **~94% s.d. 97%+**.
  4. **Rendering Video Visualizer:** Menghasilkan video `.mp4` dengan overlay bounding box walet + **Telemetry HUD (Timestamp MM:SS & Live Counter "Jumlah: X Ekor")** secara side-by-side maupun single video (`detection/video_counter_visualizer.py`).
* **Perintah Runner:**
  ```bash
  # 1. Evaluasi Perhitungan & Populasi Master Dataset (Akurasi ~94%-97%)
  python scripts/run_tahap4.py --use_master --conf 0.35

  # 2. Rendering Video Counter Side-by-Side (Berdampingan A vs B)
  python scripts/run_tahap4.py --video_stem 5_agustus --render_video --side_by_side --conf 0.35
  ```

---

## 2. METRIK AKURASI PERHITUNGAN & ANALISIS ERROR (TAHAP 4)

### A. Formula Akurasi Perhitungan (*Counting Accuracy Rate*)
$$\text{Counting Accuracy (\%)} = \left( 1 - \frac{|\text{Prediksi} - \text{GT}|}{\text{GT}} \right) \times 100\%$$

* **Contoh Kasus 1 (Prediksi 87, GT 82):**
  $$\text{Accuracy} = \left( 1 - \frac{|87 - 82|}{82} \right) \times 100\% = \left( 1 - \frac{5}{82} \right) \times 100\% = \mathbf{93.90\%}$$
* **Contoh Kasus 2 (Prediksi 84, GT 82):**
  $$\text{Accuracy} = \left( 1 - \frac{|84 - 82|}{82} \right) \times 100\% = \left( 1 - \frac{2}{82} \right) \times 100\% = \mathbf{97.56\%}$$

---

### B. Formula Mean Absolute Error (MAE) & Root Mean Squared Error (RMSE)
* **MAE:** $\text{MAE} = \frac{1}{M} \sum_{i=1}^{M} |\hat{N}_i - N_i|$
* **RMSE:** $\text{RMSE} = \sqrt{\frac{1}{M} \sum_{i=1}^{M} (\hat{N}_i - N_i)^2}$

---

## 3. LOKASI OUTPUT & HASIL EVALUASI

 Seluruh file hasil eksekusi tersimpan secara otomatis di folder `results/detection/`:
- 🎬 **Video Overlaid Counter MP4:** `results/detection/<video_stem>/video/<video_stem>_side_by_side_counter.mp4`
- 📈 **Detail Per-Frame Counting (CSV):** `results/detection/<video_stem>/per_frame_counting_detail.csv`
- 📊 **Metrik MAE & RMSE (CSV):** `results/detection/<video_stem>/counting_error_metrics.csv`
- 📑 **Laporan Populasi & Growth % (CSV):** `results/detection/<video_stem>/population_estimation_report.csv`
- 🖼️ **Grafik Bar Chart Error (300 DPI):** `results/detection/<video_stem>/counting_error_comparison.png`
- 🖼️ **Grafik Tren Populasi (300 DPI):** `results/detection/<video_stem>/population_growth_trend.png`

---

## 4. VISUALISASI ARSITEKTUR & DIAGRAM

- 🌐 **Dashboard HTML Interaktif**: [results/alur_sistem_master_opsi2.html](file:///c:/MyPrograms/swallow_detection/results/alur_sistem_master_opsi2.html)
- 🖼️ **Diagram Flowchart (300 DPI)**: [explanation/png/alur_sistem_master_opsi2.png](file:///c:/MyPrograms/swallow_detection/explanation/png/alur_sistem_master_opsi2.png)
