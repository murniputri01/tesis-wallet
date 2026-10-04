# Pipeline Pengolahan Dataset & Low-Light Enhancement CCTV Inframerah Kandang Burung Walet

Proyek ini berisi pipeline 4 tahap end-to-end lengkap dengan dukungan **Master Dataset Multi-Video (Opsi 2)** untuk inferensi otomatis video baru secara instan:

1. **Tahap 1: Ekstraksi Data & Preprocessing Enhancement (Zero-DCE + CBAM)**: Ekstraksi frame, seleksi frame SSIM, pencerahan citra low-light dengan model Zero-DCE + CBAM modifikasi, dan evaluasi kualitas citra (Entropy, BRISQUE, Contrast).
2. **Tahap 2: Anotasi Roboflow & Persiapan Dataset**: Anotasi walet pada citra terang di platform Roboflow (Ground Truth), export dataset YOLO (.txt), pembagian dataset 70:15:15 bebas data leakage, sinkronisasi label ke citra original (Skenario A) & enhanced (Skenario B), pembuatan Master Dataset multi-video (`build_master_dataset.py`), serta augmentasi data latih.
3. **Tahap 3: Pelatihan & Evaluasi Deteksi Objek (YOLO12)**: Pelatihan model deteksi objek YOLO12 pada dataset original (Skenario A) vs enhanced (Skenario Proposed B) menggunakan `--imgsz 1024`, serta evaluasi komparatif metrik mAP, Precision, Recall, dan FPS.
4. **Tahap 4: Perhitungan Objek (Counting), Estimasi Populasi & Klasifikasi Pertumbuhan**: Perhitungan populasi walet per-frame pada test set, evaluasi error counting (MAE & RMSE), estimasi laju pertumbuhan populasi, klasifikasi tingkat pertumbuhan, serta visualisasi video counter real-time.

---

## STRUKTUR DIREKTORI PROYEK

```
swallow_detection/
├── .venv/                               # Virtual environment (lingkungan Python terisolasi)
├── checkpoints/                         # Bobot memori model hasil pelatihan
│   ├── yolo_original/                   # Weights YOLO12 (Skenario A - Baseline Original Low-Light)
│   ├── yolo_zero_dce_cbam/              # Weights YOLO12 (Skenario B - Proposed Zero-DCE + CBAM)
│   └── zero_dce_cbam/                   # Weights model pencerah citra (best_zero_dce_cbam.pth)
│
├── configs/                             # File konfigurasi dataset & hyperparameter
│   ├── dataset.yaml                     # Config utama dataset YOLO
│   ├── dataset_master_original.yaml     # Config Master Dataset Skenario A (Original Low-Light)
│   ├── dataset_master_zero_dce_cbam.yaml # Config Master Dataset Skenario B (Proposed Method)
│   ├── dataset_original.yaml            # Config dataset Skenario A per-video
│   ├── dataset_zero_dce_cbam.yaml       # Config dataset Skenario B per-video
│   └── zero_dce_cbam.yaml               # Config hyperparameter model Zero-DCE + CBAM
│
├── data/                                # Pengelolaan dataset citra & label multi-tahap
│   ├── raw/
│   │   ├── videos/                      # Rekaman CCTV mentah (.mp4, .avi)
│   │   └── frames/                      # Frame hasil ekstraksi potongan video
│   ├── selected/                        # Frame terpilih (penyaringan bebas kemiripan visual SSIM)
│   ├── original/
│   │   ├── images/                      # Citra terpilih untuk diawali anotasi
│   │   └── labels/                      # Label anotasi kotak boks format YOLO (.txt)
│   ├── splits/                          # Subset dataset per-video (70:15:15)
│   ├── master_dataset/                  # Master Dataset Multi-Video Skenario A (Original)
│   └── enhanced/                        # Output citra hasil enhancement Tahap 2
│       ├── zero_dce_cbam/               # Citra ter-pencerah per-video (resolusi asli 1:1)
│       └── master_dataset/              # Master Dataset Multi-Video Skenario B (Enhanced)
│
├── detection/                           # Pipeline Tahap 3 & 4 (Deteksi Objek & Counter)
│   ├── train_yolo.py                    # Script pelatihan model deteksi YOLO12 per-skenario
│   ├── evaluate_detection.py            # Script evaluasi metrik deteksi (mAP) & visualisasi 2-panel
│   ├── count_population.py              # Script penghitungan objek per-frame & MAE/RMSE
│   ├── video_counter_visualizer.py      # Generator video visualisasi live counter & HUD telemetry
│   └── sync_labels.py                   # Sinkronisasi file label anotasi YOLO
│
├── enhancement/                         # Pipeline Tahap 2 (Model Enhancement Low-Light)
│   ├── dataloader.py                    # PyTorch Dataset & DataLoader
│   ├── train_zero_dce_cbam.py           # Script pelatihan model Zero-DCE + CBAM
│   ├── enhance.py                       # Generator pencerah citra otomatis (batch inference)
│   ├── evaluate_enhancement.py          # Evaluasi metrik kualitas citra non-reference (Entropy, BRISQUE)
│   ├── models/                          # Arsitektur jaringan neural (Zero-DCE + Dark-CBAM)
│   └── losses/                          # Loss functions (Spatial, Exposure, Color, Illumination)
│
├── explanation/                         # Dokumentasi teori, alur sistem, & file referensi
│   ├── md/                              # Dokumentasi teks Markdown
│   │   ├── ALUR_DAN_ALGORITMA_OPSI2_MASTER_DATASET.md # Dokumentasi alur Master Dataset Opsi 2
│   │   └── ...
│   ├── html/                            # Dashboard visualisasi alur interaktif
│   │   └── alur_sistem_master_opsi2.html
│   └── png/                             # Diagram flowchart 300 DPI
│       └── alur_sistem_master_opsi2.png
│
├── results/                             # Hasil eksekusi, laporan CSV, & visualisasi
│   ├── dataset_analysis/                # Laporan CSV & statistik kualitas dataset Tahap 1
│   ├── enhancement/                     # Laporan metrik pencerahan citra Tahap 2 (Terorganisir)
│   │   ├── by_video/                    # Output enhancement terisolasi per-video (7_agustus, video01, dll)
│   │   ├── global_reports/              # Laporan CSV, figures, & training samples global
│   │   ├── quality_metrics/             # Dashboard HTML interaktif & grafik distribusi
│   │   └── test_samples/                # Sampel citra uji sementara
│   ├── detection/                       # Laporan metrik YOLO & visualisasi 2-panel Tahap 3-4
│   └── alur_sistem_master_opsi2.html    # Dashboard Visualisasi Alur Sistem Interaktif Opsi 2
│
├── scripts/                             # Script pembantu pemrosesan data & runner pipeline
│   ├── dataset/
│   │   ├── build_master_dataset.py      # Builder penggabung Master Dataset Multi-Video (Opsi 2)
│   │   ├── extract_frames.py            # Ekstraksi frame citra dari video CCTV
│   │   ├── select_frames.py             # Seleksi eliminasi frame duplikat via SSIM
│   │   ├── split_dataset.py             # Pembagian dataset 70:15:15 Group Split
│   │   └── validate_dataset.py          # Validasi format file label YOLO
│   └── utils/
│       ├── enhance_video.py             # Rekonstruksi video ter-pencerah (.mp4)
│       └── visualize_quality_metrics.py # Pembuat dashboard HTML & grafik evaluasi
│
├── yolo12n.pt                           # Pretrained weights YOLO12 Nano
├── yolov8n.pt                           # Pretrained weights YOLOv8 Nano
├── requirements.txt                     # Daftar pustaka/library Python
└── README.md                            # Dokumentasi utama proyek
```

---

## 📖 Glosarium Istilah Teknis

- **Master Dataset (Opsi 2)**: Gabungan dataset dari berbagai sampel video CCTV untuk melatih *Master Model* sekali saja di awal, sehingga saat ada video baru di masa depan, sistem dapat mendeteksi walet secara otomatis (*zero-shot inference*) tanpa perlu menyentuh Roboflow lagi.
- **Weights (Bobot Memori Model)**: File berisi kumpulan angka/parameter yang dipelajari oleh AI selama pelatihan (`.pth` untuk pencerah citra, `.pt` untuk deteksi walet).
- **Enhancement (Pencerahan Citra)**: Proses peningkatan kualitas citra low-light (remang/gelap) CCTV inframerah menggunakan Zero-DCE + CBAM agar detail objek terlihat lebih jelas.
- **Loss Functions**: Rumus matematika yang mengukur tingkat kesalahan tebakan model saat pelatihan.

---

## 🚀 PANDUAN PENGGUNAAN UTAMA

### 📋 PANDUAN STEP-BY-STEP PENGUMPULAN & PEMBENTUKAN DATASET

Bagi Anda yang ingin mengumpulkan dataset baru (dari video CCTV mentah hingga menjadi **Master Dataset Siap Latih**):

1. **Langkah 1: Pencerahan Video Mentah ke Citra Terang**  
   Masukkan file video mentah (`.mp4`) ke folder `data/raw/videos/`, lalu jalankan:
   ```bash
   python scripts/run_tahap1.py --video data/raw/videos/7_agustus.mp4
   ```
   *(Hasil gambar terang tersimpan otomatis di: `data/enhanced/zero_dce_cbam/7_agustus/`)*

2. **Langkah 2: Anotasi Manual Walet di Roboflow**  
   - Ambil foto-foto citra terang dari folder `data/enhanced/zero_dce_cbam/7_agustus/`.
   - Upload ke platform **Roboflow**, lalu buat anotasi boks walet secara presisi.
   - Export dataset dari Roboflow dalam format **YOLOv8 / YOLOv12 PyTorch** (`.zip`).

3. **Langkah 3: Ekstrak Hasil Export Roboflow ke `data/splits/`**  
   - Ekstrak seluruh isi file `.zip` Roboflow langsung ke folder `data/splits/`  
     *(Sehingga folder `data/splits/` terisi folder `train/`, `val/`, dan `test/`)*.

4. **Langkah 4: Jalankan Builder Master Dataset (Otomatis)**  
   ```bash
   python scripts/dataset/build_master_dataset.py
   ```
   *(Skrip secara otomatis menyatukan gambar terang & label Roboflow tanpa hash `.rf.xxxx` ke `data/enhanced/master_dataset/`)*.

---

### 1. FASE 1: PELATIHAN MASTER MODEL (Cukup 1 Kali di Awal)

FASE 1 bertujuan untuk melatih satu **"Otak Utama" (Master Model)** yang dapat mengenali burung walet pada citra ter-pencerah (Zero-DCE + CBAM). FASE 1 terdiri dari **3 Sub-Langkah utama**:

#### 🔍 Penjelasan Detail 3 Sub-Langkah FASE 1

1. **Sub-Langkah 1: Konsolidasi Master Dataset (`build_master_dataset.py`)**
   - **Tujuan**: Menggabungkan & menyinkronkan seluruh dataset citra + label dari berbagai video ke folder terpusat `data/enhanced/master_dataset/`.
   - **Input**: Dataset per-video (`data/splits/` dan `data/enhanced/zero_dce_cbam/`).
   - **Output**: Master Dataset terstruktur dengan file konfigurasi `configs/dataset_master_zero_dce_cbam.yaml`.
   - **Kapan Dijalankan?**: Saat pertama kali setup atau ketika ada penambahan video/data latih baru ke master.

2. **Sub-Langkah 2: Pelatihan Model Utama YOLO12 (`train_yolo.py`)**
   - **Tujuan**: Melatih model deteksi YOLO12 Skenario B (Zero-DCE + CBAM) pada resolusi tinggi 1024px.
   - **Input**: Master dataset dari Sub-Langkah 1 dan bobot awal (`yolo12n.pt`).
   - **Output**: Bobot memori "Otak Utama" tersimpan di `checkpoints/yolo_zero_dce_cbam/master/best.pt` & `checkpoints/yolo_zero_dce_cbam/best.pt`.
   - **Kapan Dijalankan?**: Setelah Sub-Langkah 1 selesai, atau ketika ingin mencoba hyperparameter training baru (misal mengubah `--epochs` atau `--batch`).

3. **Sub-Langkah 3: Evaluasi Metrik & Visualisasi Bounding Box (`evaluate_detection.py`)**
   - **Tujuan**: Menguji akurasi deteksi model pada subset `test` (metrik mAP50, mAP50-95, Precision, Recall) serta menghasilkan gambar sampel bounding box.
   - **Input**: Bobot terlatih `checkpoints/yolo_zero_dce_cbam/master/best.pt`.
   - **Output**: Laporan metrik CSV & gambar visualisasi bounding box di folder `results/detection/`.
   - **Kapan Dijalankan?**: Setelah pelatihan selesai untuk verifikasi performa model.

---

#### 📌 Cara A: Otomatis Sekali Jalan (All-in-One)
```bash
# Jalankan Step 1 -> 2 -> 3 secara berurutan
python scripts/run_fase1_master_training.py --epochs 100 --imgsz 1024 --batch 8
```

#### 📌 Cara B: Jalankan Sub-Langkah Spesifik (`--step`)
```bash
# Sub-Langkah 1: Konsolidasi Master Dataset Multi-Video saja
python scripts/run_fase1_master_training.py --step 1
# Atau: python scripts/run_fase1_master_training.py --only_build_dataset

# Sub-Langkah 2: Pelatihan Model Utama YOLO12 (1024px) saja
python scripts/run_fase1_master_training.py --step 2 --epochs 100 --batch 8
# Atau: python scripts/run_fase1_master_training.py --only_train

# Sub-Langkah 3: Evaluasi Metrik & Visualisasi Bounding Box saja
python scripts/run_fase1_master_training.py --step 3
# Atau: python scripts/run_fase1_master_training.py --only_eval
```

#### 📌 Cara C: Jalankan Per-Script Manual (Command Langsung)
```bash
# 1. Konsolidasi Dataset
python scripts/dataset/build_master_dataset.py --overwrite

# 2. Training YOLO12 (Proposed Skenario B: Zero-DCE + CBAM)
python detection/train_yolo.py --scenario B --data configs/dataset_master_zero_dce_cbam.yaml --epochs 100 --imgsz 1024 --batch 8 --video_stem master

# 3. Evaluasi Metrik & Bounding Box
python detection/evaluate_detection.py --weights_b checkpoints/yolo_zero_dce_cbam/master/best.pt --output_dir results/detection --video_stem master
```

---

### 2. FASE 2: DETEKSI OTOMATIS & RENDERING VIDEO COUNTER (PILIH VIDEO)

Setiap kali Anda memiliki file video baru di `data/raw/videos/` (misal: `5_agustus.mp4`, `7_agustus.mp4`, `11_agustus.mp4`, `video01.mp4`), Anda dapat memilih video dan mengeksekusi perhitungan presisi tinggi (Akurasi ~94%-97%, `--conf 0.35`):

#### 🎥 Option A: Generasi Video Counter Real-Time (Side-by-Side Bounding Box + Telemetry HUD)
```bash
# Hasilkan Video MP4 Bounding Box + Counter Real-Time (Side-by-Side Skenario A vs B)
python scripts/run_tahap4.py --video_stem 5_agustus --render_video --side_by_side --conf 0.35
```

#### 📊 Option B: Evaluasi Laporan Counting & Estimasi Populasi
```bash
# Hitung MAE, RMSE, dan Laju Pertumbuhan Populasi pada Master Dataset
python scripts/run_tahap4.py --use_master --conf 0.35
```

**Hasil Otomatis**:
- 🎬 **Video Counter MP4**: `results/detection/<video_stem>/video/<video_stem>_side_by_side_counter.mp4`
- 📈 **Laporan Per-Frame CSV**: `results/detection/<video_stem>/per_frame_counting_detail.csv`
- 📊 **Metrik MAE & RMSE CSV**: `results/detection/<video_stem>/counting_error_metrics.csv`
- 🖼️ **Grafik Tren Populasi & Error**: `results/detection/<video_stem>/population_growth_trend.png`

---

## 📊 DASHBOARD & VISUALISASI ARSITEKTUR

Untuk melihat arsitektur dan alur sistem secara visual:
- 🌐 **Dashboard HTML Interaktif**: [results/alur_sistem_master_opsi2.html](file:///c:/MyPrograms/swallow_detection/results/alur_sistem_master_opsi2.html)
- 🖼️ **Diagram Flowchart Sistem Saat Ini (300 DPI)**: [explanation/png/alur_sistem_penelitian_saat_ini.png](file:///c:/MyPrograms/swallow_detection/explanation/png/alur_sistem_penelitian_saat_ini.png)
- 📄 **Dokumentasi Lengkap Opsi 2**: [explanation/md/ALUR_DAN_ALGORITMA_OPSI2_MASTER_DATASET.md](file:///c:/MyPrograms/swallow_detection/explanation/md/ALUR_DAN_ALGORITMA_OPSI2_MASTER_DATASET.md)
