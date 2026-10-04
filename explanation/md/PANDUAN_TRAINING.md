# Panduan Eksekusi Training, Debugging, dan Evaluasi Tahap 2

Penelitian: **"MODIFIKASI ZERO-DCE BERBASIS CONVOLUTIONAL BLOCK ATTENTION MODULE (CBAM) UNTUK ENHANCEMENT CITRA LOW-LIGHT CCTV INFRAMERAH PADA KANDANG BURUNG WALET"**

Dokumen ini berisi penjelasan sintaks perintah terminal, perbedaan mode pengujian (*Debug Mode* vs *Full Training*), serta panduan langkah-demi-langkah dari pelatihan model hingga pembuatan figur visualisasi 2-panel dan analisis teknis.

---

## 1. PENJELASAN SINTAKS PERINTAH TERMINAL

Memahami arti dari setiap parameter perintah terminal:

$$\text{\texttt{python}} \quad \underbrace{\text{\texttt{enhancement/train\_zero\_dce\_cbam.py}}}_{\text{File Script Utama}} \quad \underbrace{\text{\texttt{--config configs/zero\_dce\_cbam.yaml}}}_{\text{File Konfigurasi Hyperparameter}} \quad \underbrace{\text{\texttt{--debug}}}_{\text{Mode Uji Cepat}} \quad \underbrace{\text{\texttt{--epochs 5}}}_{\text{Jumlah Putaran Debug}}$$

1. **`python`**: Memanggil *interpreter* bahasa Python untuk mengeksekusi script.
2. **`enhancement/train_zero_dce_cbam.py`**: Lokasi file script pelatihan untuk Zero-DCE + CBAM Modifikasi.
3. **`--config configs/zero_dce_cbam.yaml`**: Memuat seluruh hyperparameter pelatihan penuh dari file konfigurasi YAML (600 Epochs, lr 0.0001, batch size 4, grad clip norm 0.1, pixel weighting, dsb).
4. **`--debug`**: Flag mode uji coba cepat (*sanity check / dry run*). Program mengambil subset kecil (5 gambar saja) untuk memastikan kode, memori, dan pembuatan gambar sampel berjalan tanpa error.
5. **`--epochs 5`**: Membatasi perulangan pelatihan hanya 5 kali putaran pada mode debug (selesai dalam 3–5 detik).
6. **`--overwrite`**: Menimpa file hasil *enhancement* jika file dengan nama yang sama sudah ada di folder tujuan.

## 2. SKENARIO PENGUJIAN EKSPERIMEN

Dalam pipeline eksperimen penelitian ini, dataset dan evaluasi difokuskan pada:

- **Skenario A (Original / Benchmark Tanpa Enhancement)**: Citra mentah asli pada [`data/splits/test/images`](file:///c:/MyPrograms/swallow_detection/data/splits/test/images) langsung digunakan untuk deteksi objek (YOLO) tanpa proses enhancement.
- **Skenario Proposed Method (Zero-DCE + CBAM Modifikasi)**: Citra di-enhance menggunakan Zero-DCE yang dilengkapi modul CBAM Modifikasi berbasis Darkness Map dan disimpan di [`data/enhanced/zero_dce_cbam/`](file:///c:/MyPrograms/swallow_detection/data/enhanced/zero_dce_cbam/).

---

## 3. PERBANDINGAN MODE EKSEKUSI

| Mode Perintah | Perintah Terminal | Tujuan & Kegunaan | Waktu Eksekusi |
| :--- | :--- | :--- | :--- |
| **Uji Coba Cepat (Debug Mode)** | `python enhancement/train_zero_dce_cbam.py --debug --epochs 5` | **Untuk Tes Cepat**: Memastikan kode aman, memori CPU/GPU cukup, & sampel pratinjau gambar 1080p terbentuk. | ⚡ ~3 Detik |
| **Pelatihan Penuh (Full Training)** | `python enhancement/train_zero_dce_cbam.py --config configs/zero_dce_cbam.yaml` | **Untuk Hasil Tesis**: Melatih jaringan 600 Epochs pada seluruh dataset train/val hingga konvergen penuh. | ⏳ ~2–5 Menit |

---

## ⚡ METODE PALING EFISIEN: 1-COMMAND AUTOMATED MASTER PIPELINE (RECOMMENDED)

Untuk memproses file video baru dari mentah hingga pencerahan citra secara **paling efisien dan praktis**, gunakan script master pipeline 1-command:

```bash
# Eksekusi pipeline otomatis (ekstraksi, seleksi, split, hingga enhancement generator)
python scripts/run_single_video_pipeline.py --video data/raw/videos/video01.mp4 --num_threads 4

# Eksekusi pipeline otomatis sekaligus retraining model Zero-DCE+CBAM
python scripts/run_single_video_pipeline.py --video data/raw/videos/video01.mp4 --train --num_threads 4
```

---

## 4. LANGKAH-LANGKAH EKSEKUSI MANUAL TAHAP 2

### LANGKAH 0: Pengecekan Prasyarat Dataset
Pastikan folder dataset `data/splits/` sudah siap dari Tahap 1:
- `data/splits/train/images`
- `data/splits/val/images`
- `data/splits/test/images`

Terminal dibuka pada direktori utama project: `C:\MyPrograms\swallow_detection`

---

### LANGKAH 1: UJI COBA CEPAT (DEBUG TEST - 5 EPOCHS)

Sebelum melatih penuh, lakukan tes cepat 5 epoch untuk memverifikasi bahwa kode dan pembuatan pratinjau sampel berjalan lancar:

```bash
# Tes Cepat Skenario Zero-DCE + CBAM Modifikasi (misal untuk video 'video01')
python enhancement/train_zero_dce_cbam.py --debug --epochs 5 --video_stem video01
```

- *Hasil Sampel Pratinjau tersimpan di*:
  - `results/enhancement/training_samples/video01/zero_dce_cbam_sample_epoch_005.png`

---

### LANGKAH 2: PELATIHAN PENUH MODEL (FULL TRAINING - 600 EPOCHS)

Jalankan pelatihan penuh pada seluruh dataset untuk menghasilkan bobot model terbaik (*checkpoint*):

```bash
python enhancement/train_zero_dce_cbam.py --config configs/zero_dce_cbam.yaml --video_stem video01
```
- *Checkpoint Terbaik Tersimpan di*: `checkpoints/zero_dce_cbam/video01/best_zero_dce_cbam.pth`

---

### LANGKAH 3: GENERATE CITRA HASIL ENHANCEMENT (BATCH INFERENCE)

Setelah model selesai dilatih, hasilkan citra *enhanced* pada subfolder video terpilih di dataset (`train`, `val`, `test`):

```bash
python enhancement/enhance.py --checkpoint checkpoints/zero_dce_cbam/best_zero_dce_cbam.pth --input_dir data/splits --output_dir data/enhanced/zero_dce_cbam --video_stem video01 --num_threads 4 --overwrite
```

- *Hasil Dataset Skenario Utama tersimpan di*: `data/enhanced/zero_dce_cbam/`

---

### LANGKAH 4: EVALUASI METRIK & GENERATE FIGUR VISUALISASI 2-PANEL DAN TEKNIS

Jalankan perintah evaluasi untuk menghitung metrik *Non-Reference* (Brightness, Contrast, Shannon Entropy, Dynamic Range) serta membuat figur 2-panel (`ORIGINAL | ZERO-DCE+CBAM`) resolusi tinggi (300 DPI) dan figur analisis teknis:

```bash
python enhancement/evaluate_enhancement.py --orig_dir data/splits/test/images --cbam_dir data/enhanced/zero_dce_cbam/test/images --output_dir results/enhancement --cbam_checkpoint checkpoints/zero_dce_cbam/best_zero_dce_cbam.pth --video_stem video01
```

#### File Keluaran Akhir:
1. **Laporan Metrik Non-Reference CSV**:
   `results/enhancement/video01/metrics_reports/enhancement_metrics_report.csv`
2. **Figur Visualisasi 2-Panel (300 DPI)**:
   `results/enhancement/video01/figures/2panel_comparison_*.png`
3. **Figur Analisis Teknis (Peta Attention & Darkness Map)**:
   `results/enhancement/video01/technical_analysis/analysis_*.png`

---

## 5. CARA MENGHASILKAN VISUALISASI SEPERTI `zero_dce_cbam_sample_epoch_600.png`

File gambar sampel pratinjau seperti `zero_dce_cbam_sample_epoch_600.png` dihasilkan secara otomatis oleh script pelatihan selama proses training berlangsung pada interval epoch tertentu.

### A. Cara Kerja Pembuatan Gambar Sampel Visualisasi:
1. **Interval Penyimpanan (`save_frequency`)**:
   Dalam file [`configs/zero_dce_cbam.yaml`](file:///c:/MyPrograms/swallow_detection/configs/zero_dce_cbam.yaml), terdapat parameter:
   ```yaml
   training:
     epochs: 600           # Total putaran training
     save_frequency: 100   # Simpan sampel gambar setiap 100 epoch
   ```
   Setiap kali epoch mencapai kelipatan 100 (misalnya Epoch 100, 200, 300, 400, 500, 600), program akan otomatis mengambil 1 citra dari validation set dan menyimpannya di folder `results/enhancement/training_samples/{video_stem}/`.

2. **Format Gambar yang Dihasilkan**:
   - **Zero-DCE + CBAM Proposed Method**: Format 4 panel `[INPUT ORIGINAL | DARKNESS MAP | SPATIAL ATTENTION | CBAM ENHANCED]`
     Nama file: `results/enhancement/training_samples/{video_stem}/zero_dce_cbam_sample_epoch_600.png`

3. **Resolusi Kualitas Tinggi (High-Resolution 1080p)**:
   Gambar pratinjau otomatis di-upscale secara presisi (*Lanczos Interpolation*) ke tinggi 1080px agar saat dibuka tampak besar, tajam, dan tidak pecah.

## 6. LANGKAH-LANGKAH EKSEKUSI TAHAP 3 (TRAINING & EVALUASI DETEKSI OBJEK YOLO12)

Setelah Tahap 2 (Model Enhancement) selesai dan menghasilkan citra ter-enhance pada `data/enhanced/zero_dce_cbam/`, jalankan pelatihan dan evaluasi komparatif deteksi objek YOLO12:

### LANGKAH 1: Training Skenario A (Original Low-Light Image)
```bash
python detection/train_yolo.py --scenario A --epochs 100 --weights yolo12n.pt --video_stem video01
```
- *Checkpoint Terbaik*: `checkpoints/yolo_original/video01/best.pt`

### LANGKAH 2: Training Skenario B (Proposed Method: Zero-DCE + CBAM)
```bash
python detection/train_yolo.py --scenario B --epochs 100 --weights yolo12n.pt --video_stem video01
```
- *Checkpoint Terbaik*: `checkpoints/yolo_zero_dce_cbam/video01/best.pt`

### LANGKAH 3: Evaluasi Metrik & Figur Visualisasi 2-Panel Bounding Box (300 DPI)
```bash
python detection/evaluate_detection.py --weights_a checkpoints/yolo_original/best.pt --weights_b checkpoints/yolo_zero_dce_cbam/best.pt --output_dir results/detection --video_stem video01
```
- *Output Laporan CSV*: `results/detection/video01/detection_metrics_report.csv` (Precision, Recall, mAP@0.5, mAP@0.5:0.95, FPS)
- *Output Figur 2-Panel*: `results/detection/video01/2panel_detection_comparison_*.png`

---

## 7. LANGKAH-LANGKAH EKSEKUSI TAHAP 4 (COUNTING, MAE/RMSE, ESTIMASI POPULASI & RENDERING VIDEO COUNTER)

Jalankan evaluasi kesalahan perhitungan objek (counting error MAE & RMSE), estimasi pertumbuhan populasi burung walet, serta pembuatan video counter visualizer:

```bash
# 1. Evaluasi Counting & Estimasi Populasi pada Master Dataset (Akurasi ~94%-97%, --conf 0.35)
python scripts/run_tahap4.py --use_master --conf 0.35

# 2. Evaluasi Video Spesifik (misal: 5_agustus atau video01)
python scripts/run_tahap4.py --video_stem 5_agustus --conf 0.35

# 3. Generasi Video Visualisasi Deteksi & Real-Time Counter (Side-by-Side Bounding Box HUD)
python scripts/run_tahap4.py --video_stem 5_agustus --render_video --side_by_side --conf 0.35
```

- *Detail Per-Frame (CSV)*: `results/detection/5_agustus/per_frame_counting_detail.csv`
- *Metrik Error MAE & RMSE (CSV)*: `results/detection/5_agustus/counting_error_metrics.csv`
- *Laporan Estimasi Populasi & Klasifikasi (CSV)*: `results/detection/5_agustus/population_estimation_report.csv`
- *Grafik Error Bar Chart (300 DPI)*: `results/detection/5_agustus/counting_error_comparison.png`
- *Grafik Tren Populasi & Klasifikasi (300 DPI)*: `results/detection/5_agustus/population_growth_trend.png`
- *Video Overlaid Counter MP4*: `results/detection/5_agustus/video/5_agustus_side_by_side_counter.mp4`

---

## 8. RINGKASAN CHEAT SHEET PERINTAH TERMINAL

```bash
# 1. Automated Master Pipeline (1-Command Pipeline)
python scripts/run_single_video_pipeline.py --video data/raw/videos/video01.mp4 --num_threads 4

# 2. Full Training Proposed Method Enhancement (Zero-DCE + CBAM)
python enhancement/train_zero_dce_cbam.py --config configs/zero_dce_cbam.yaml --video_stem video01

# 3. Batch Enhancement Generator
python enhancement/enhance.py --checkpoint checkpoints/zero_dce_cbam/best_zero_dce_cbam.pth --input_dir data/splits --output_dir data/enhanced/zero_dce_cbam --video_stem video01 --num_threads 4 --overwrite

# 4. Evaluasi Metrik & Visualisasi Enhancement
python enhancement/evaluate_enhancement.py --orig_dir data/splits/test/images --cbam_dir data/enhanced/zero_dce_cbam/test/images --output_dir results/enhancement --cbam_checkpoint checkpoints/zero_dce_cbam/best_zero_dce_cbam.pth --video_stem video01

# 5. Training Deteksi Objek YOLO12 Skenario A (Original)
python detection/train_yolo.py --scenario A --epochs 100 --weights yolo12n.pt --video_stem video01 --imgsz 640 --batch 8

# 6. Training Deteksi Objek YOLO12 Skenario B (Zero-DCE + CBAM)
python detection/train_yolo.py --scenario B --epochs 100 --weights yolo12n.pt --video_stem video01 --imgsz 640 --batch 8

# 7. Evaluasi Metrik Deteksi & Figur Visualisasi 2-Panel Bounding Box
python detection/evaluate_detection.py --weights_a checkpoints/yolo_original/best.pt --weights_b checkpoints/yolo_zero_dce_cbam/best.pt --output_dir results/detection --video_stem video01

# 8. Counting Presisi Tinggi (MAE/RMSE) & Rendering Video Overlaid Counter
python scripts/run_tahap4.py --video_stem 5_agustus --render_video --side_by_side --conf 0.35
```
