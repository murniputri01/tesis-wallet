# Dokumen Alur dan Algoritma Tahap 3: Training Deteksi Objek (YOLO12) & Evaluasi Komparatif

Penelitian: **"MODIFIKASI ZERO-DCE BERBASIS CONVOLUTIONAL BLOCK ATTENTION MODULE (CBAM) UNTUK ENHANCEMENT CITRA LOW-LIGHT CCTV INFRAMERAH PADA KANDANG BURUNG WALET"**

Dokumen ini menyajikan panduan sistematis, teknis, dan matematis mengenai alur pemrosesan deep learning di dalam **Tahap 3 (Pelatihan Model Deteksi Objek YOLO12 Skenario A vs B dan Evaluasi Komparatif)**. Setiap bagian dipisahkan secara tegas dan terstruktur antara **Penjelasan Konsep & Formulasi Matematika** dengan **Lokasi & Potongan Kode Program Implementasi**.

---

## 1. DIAGRAM ALUR TAHAP 3

```text
Dataset Citra Low-Light (data/splits/) & Enhanced (data/enhanced/zero_dce_cbam/)
                                │
                                ▼
┌──────────────────────────────────────────────────────────────┐
│ 1. SINKRONISASI LABEL SKENARIO B                             │  (sync_labels.py)
│    Salin struktur label (.txt) dari data/splits/             │
│    ke data/enhanced/zero_dce_cbam/                           │
└──────────────────────────────┬───────────────────────────────┘
                               │
            ┌──────────────────┴──────────────────┐
            ▼                                     ▼
┌──────────────────────────────┐       ┌──────────────────────────────┐
│ 2A. TRAINING SKENARIO A      │       │ 2B. TRAINING SKENARIO B      │  (train_yolo.py)
│     Input: Original Low-Light│       │     Input: Zero-DCE + CBAM │
│     Config: dataset_original │       │     Config: dataset_zero_dce │
│     Output: best.pt (A)      │       │     Output: best.pt (B)      │
└──────────────┬───────────────┘       └──────────────┬───────────────┘
               │                                      │
               └──────────────────┬───────────────────┘
                                  │
                                  ▼ Checkpoints (checkpoints/yolo_original/ & checkpoints/yolo_zero_dce_cbam/)
┌──────────────────────────────────────────────────────────────┐
│ 3. EVALUASI METRIK DETEKSI KOMPARATIF                       │  (evaluate_detection.py)
│    Hitung Metrik: Precision, Recall, mAP@0.5, mAP@0.5:0.95,  │
│                   Inference Time (ms), & FPS                 │
│    Simpan Laporan: results/detection/detection_metrics_report.csv│
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│ 4. GENERASI FIGUR VISUALISASI BOUNDING BOX 2-PANEL (300 DPI) │  (evaluate_detection.py)
│    Panel Kiri  : Predictions Skenario A (Original)           │
│    Panel Kanan : Predictions Skenario B (Zero-DCE + CBAM)    │
│    Output Image: results/detection/2panel_detection_comparison_*.png
└──────────────────────────────────────────────────────────────┘
```

---

## 2. DETAIL ALGORITMA DAN IMPLEMENTASI PROGRAM TIAP KOMPONEN

---

### A. Sinkronisasi Struktur Label Skenario B

#### 📘 1. Penjelasan Konsep & Formulasi Matematika

- **Konsep & Algoritma**:
  Citra pencerahan hasil Tahap 2 disimpan pada direktori `data/enhanced/zero_dce_cbam/`. Sebelum melatih model deteksi objek YOLO12 Skenario B, struktur folder dan file anotasi bounding box `.txt` dari `data/splits/` disinkronkan ke dalam folder citra enhanced agar struktur format dataset YOLO terpenuhi secara utuh (memiliki subfolder `images/` dan `labels/`).

---

#### 💻 2. Lokasi Implementasi & Potongan Kode Program

- **File Program**: [`detection/sync_labels.py`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/detection/sync_labels.py)
- **Fungsi Utama**: [`sync_labels()`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/detection/sync_labels.py#L18-L60)
- **Letak Baris Kode**: Baris 38 – 58

```python
# [detection/sync_labels.py] Baris 38-58
for split_name in ["train", "val", "test", "train_augmented"]:
    src_label_dir = splits_dir / split_name / "labels"
    dst_label_dir = enhanced_dir / split_name / "labels"
    if src_label_dir.exists():
        shutil.copytree(src_label_dir, dst_label_dir, dirs_exist_ok=True)
```

---

### B. Arsitektur Model Deteksi Objek YOLO12 & Formulasi Loss Function

#### 📘 1. Penjelasan Konsep & Formulasi Matematika

- **Konsep & Algoritma**:
  YOLO12 mengadopsi mekanisme *Area Attention* yang efisien untuk menangkap pola fitur burung walet pada latar belakang kandang yang gelap. Model dioptimasi memprediksi koordinat bounding box $(x_c, y_c, w, h)$ dan kelas objek tunggal `walet` (`class_id = 0`).

- **Formulasi Total Loss YOLO12**:
  $$L_{\text{YOLO}} = \lambda_{\text{box}} L_{\text{box}} + \lambda_{\text{cls}} L_{\text{cls}} + \lambda_{\text{dfl}} L_{\text{dfl}}$$

  1. **Complete Intersection over Union Loss ($L_{\text{box}}$ - CIoU Loss)**:
     $$\text{CIoU} = \text{IoU} - \left( \frac{\rho^2(b, b^{gt})}{c^2} + \alpha v \right), \quad L_{\text{box}} = 1 - \text{CIoU}$$
     di mana $\rho(b, b^{gt})$ adalah jarak Euclidean pusat box, $c$ adalah panjang diagonal box penutup terkecil, dan $v$ adalah ukuran konsistensi rasio aspek.

  2. **Binary Cross-Entropy Loss ($L_{\text{cls}}$ - Classification Loss)**:
     $$L_{\text{cls}} = -\sum_{i} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$

  3. **Distribution Focal Loss ($L_{\text{dfl}}$)**:
     Memperhalus regresi batas tepi bounding box ketika objek walet berukuran kecil dan berkontur samar.

---

#### 💻 2. Lokasi Implementasi & Potongan Kode Program

- **File Program**: [`detection/train_yolo.py`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/detection/train_yolo.py)
- **Fungsi Utama**: [`train_yolo()`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/detection/train_yolo.py#L23-L120)
- **Konfigurasi Dataset A**: [`configs/dataset_original.yaml`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/configs/dataset_original.yaml)
- **Konfigurasi Dataset B**: [`configs/dataset_zero_dce_cbam.yaml`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/configs/dataset_zero_dce_cbam.yaml)
- **Letak Baris Kode**: Baris 82 – 98

```python
# [detection/train_yolo.py] Baris 82-98
model = YOLO(model_cfg)  # Inisialisasi arsitektur YOLO12
results = model.train(
    data=str(data_config_path),
    epochs=epochs,
    batch=batch_size,
    imgsz=img_size,
    device=device,
    project=str(output_dir.parent),
    name=output_dir.name
)
```

---

### C. Metrik Evaluasi Deteksi Komparatif & Visualisasi 2-Panel

#### 📘 1. Penjelasan Konsep & Formulasi Matematika

1. **Precision ($P$)**:
   Tingkat akurasi deteksi positif dibanding seluruh prediksi positif:
   $$P = \frac{TP}{TP + FP}$$

2. **Recall ($R$)**:
   Tingkat keberhasilan model menemukan seluruh objek walet nyata pada frame:
   $$R = \frac{TP}{TP + FN}$$

3. **Mean Average Precision ($mAP@0.5$ dan $mAP@0.5:0.95$)**:
   $$AP = \int_{0}^{1} P(R) \, dR$$
   - $mAP@0.5$: Rata-rata $AP$ pada threshold $\text{IoU} = 0.50$.
   - $mAP@0.5:0.95$: Rata-rata $AP$ pada 10 threshold IoU ($0.50, 0.55, \dots, 0.95$).

4. **Kecepatan Inferensi & FPS**:
   $$\text{FPS} = \frac{1000}{t_{\text{inference\_ms}}}$$

---

#### 💻 2. Lokasi Implementasi & Potongan Kode Program

- **File Program**: [`detection/evaluate_detection.py`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/detection/evaluate_detection.py)
- **Fungsi Utama**: [`evaluate_detection()`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/detection/evaluate_detection.py#L69-L215) dan [`draw_yolo_predictions()`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/detection/evaluate_detection.py#L31-L67)
- **Letak Baris Kode**:
  - Ekstraksi Metrik Skenario A & B via `val()`: Baris 90 – 125
  - Kalkulasi FPS & Pembuatan DataFrame Laporan: Baris 98 – 140
  - Rendering Visualisasi Bounding Box 2-Panel: Baris 31 – 67 & 160 – 205

```python
# [detection/evaluate_detection.py] Baris 90-125 & 98-140
# Evaluasi inferensi Skenario B
res_b = model_b.val(data=str(config_b), split="test", verbose=False)
p_b = float(res_b.results_dict.get("metrics/precision(B)", 0.0))
r_b = float(res_b.results_dict.get("metrics/recall(B)", 0.0))
map50_b = float(res_b.results_dict.get("metrics/mAP50(B)", 0.0))
map5095_b = float(res_b.results_dict.get("metrics/mAP50-95(B)", 0.0))

# Hitung FPS
inf_time = res_b.speed.get("inference", 0.0)
fps_b = 1000.0 / inf_time if inf_time > 0 else 0.0
```

---

## 3. TABEL RANGKUMAN PEMETAAN FILE KODE PROGRAM

| Komponen Algoritma | File Kode Program | Kelas / Fungsi Utama | Letak Baris Implementasi Formula |
| :--- | :--- | :--- | :--- |
| **Runner Utama Tahap 3** | [`scripts/run_tahap3.py`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/scripts/run_tahap3.py) | `main()` | Runner otomatis eksekusi Tahap 3 |
| **Sinkronisasi Label Dataset** | [`detection/sync_labels.py`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/detection/sync_labels.py) | [`sync_labels()`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/detection/sync_labels.py#L18-L60) | [`L38-L58`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/detection/sync_labels.py#L38-L58) (Salin anotasi label `.txt`) |
| **Pusat Training YOLO12** | [`detection/train_yolo.py`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/detection/train_yolo.py) | [`train_yolo()`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/detection/train_yolo.py#L23-L120) | [`L82-L98`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/detection/train_yolo.py#L82-L98) (Optimasi $L_{\text{box}}, L_{\text{cls}}, L_{\text{dfl}}$) |
| **Konfigurasi Dataset A** | [`configs/dataset_original.yaml`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/configs/dataset_original.yaml) | Konfigurasi YAML | Jalur citra original low-light |
| **Konfigurasi Dataset B** | [`configs/dataset_zero_dce_cbam.yaml`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/configs/dataset_zero_dce_cbam.yaml) | Konfigurasi YAML | Jalur citra enhanced Zero-DCE + CBAM |
| **Evaluasi Deteksi & 2-Panel** | [`detection/evaluate_detection.py`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/detection/evaluate_detection.py) | [`evaluate_detection()`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/detection/evaluate_detection.py#L69-L215), [`draw_yolo_predictions()`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/detection/evaluate_detection.py#L31-L67) | [`L95-L100`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/detection/evaluate_detection.py#L95-L100), [`L115-L120`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/detection/evaluate_detection.py#L115-L120) ($P, R, mAP, \text{FPS}$), [`L31-L67`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/detection/evaluate_detection.py#L31-L67) (Visualisasi BBox) |

---

## 4. PERINTAH UTAMA EKSEKUSI TAHAP 3

Untuk menjalankan Tahap 3 secara otomatis:

```bash
# Jalankan seluruh proses (Training Skenario A, Skenario B, & Evaluasi Komparatif)
python scripts/run_tahap3.py --video_stem video01

# Jalankan hanya proses evaluasi komparatif (jika checkpoint best.pt sudah ada)
python scripts/run_tahap3.py --video_stem video01 --only_eval

# Custom hyperparameter training
python scripts/run_tahap3.py --video_stem video01 --epochs 100 --batch 16 --imgsz 640
```

Hasil output akan tersimpan di:
- `checkpoints/yolo_original/video01/best.pt` (Checkpoint Weights Skenario A)
- `checkpoints/yolo_zero_dce_cbam/video01/best.pt` (Checkpoint Weights Skenario B)
- `results/detection/detection_metrics_report.csv` (Laporan Metrik Deteksi CSV)
- `results/detection/2panel_detection_comparison_*.png` (Figur Visualisasi Bounding Box 2-Panel 300 DPI)
