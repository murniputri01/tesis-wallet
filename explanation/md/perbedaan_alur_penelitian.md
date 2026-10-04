# Perbandingan Alur Sistem Penelitian: Sebelum vs Sekarang

> Berdasarkan:
> - **Sebelum (Awal)**: [`alu_sistem_penelitian_sebelum.jpeg`](../png/alu_sistem_penelitian_sebelum.jpeg)
> - **Sekarang (Awal)**: [`alur_sistem_penelitian_sekarang.png`](../png/alur_sistem_penelitian_sekarang.png)
> - **Revisi Terbaru (Roboflow Flow)**: [`alur_sistem_penelitian_roboflow.png`](../png/alur_sistem_penelitian_roboflow.png)

---

## Ringkasan Cepat

| Aspek | Sebelum (Awal) | Revisi Sekarang (Roboflow Flow) |
|---|---|---|
| Judul Kolom 1 | **1 DATA** | **1 EKSTRAKSI DATA & ENHANCEMENT ZERO-DCE+CBAM** |
| Judul Kolom 2 | **2 ENHANCEMENT & PREPROCESSING** | **2 ANOTASI ROBOFLOW & PERSIAPAN DATASET** |
| Judul Kolom 3 | **3 TRAINING DETEKSI OBJEK** | **3 TRAINING DETEKSI OBJEK (YOLO12)** |
| Judul Kolom 4 | **4 PENGUJIAN & ESTIMASI POPULASI** | **4 COUNTING & ESTIMASI POPULASI** |
| Jumlah Skenario | **3 Skenario** (A, B, C) | **2 Skenario** (A, B) |
| Posisi Enhancement | Di Kolom 2 (setelah labeling) | **Dipindah ke Kolom 1** (sebelum labeling) |
| Posisi Labeling & Split | Di Kolom 2 (bawah) | **Di Kolom 2** (pada citra terang via Roboflow) |
| Detail Arsitektur | Ringkas (garis besar) | Eksplisit step-by-step (Anotasi Terang + Sync Label) |

---

## Perubahan per Kolom

---

### 📦 Kolom 1 — DATA → PERSIAPAN & PENGOLAHAN DATA

**SEBELUM** — hanya 3 langkah sederhana:

```
Mulai
  → Pengumpulan Data Rekaman CCTV Inframerah
  → Ekstraksi Frame Video
  → Resize Citra          ← (output terakhir, masuk ke Kol-2)
```

**SEKARANG** — 7 langkah, lebih lengkap dan mandiri:

```
Mulai
  → Rekaman CCTV Inframerah (data/raw/videos/*.mp4)
  → Ekstraksi Frame Video (1 frame/detik)
  → Seleksi Frame (SSIM ≥ 0.95)
  → Auto-Labeling YOLO (bounding box walet)
  → [Diamond] Label & bbox valid?
        Tidak → (loop kembali)
        Ya    → Split Dataset 70:15:15 (Video Group Split)
  → Augmentasi → Data Latih saja (2 varian/citra)
```

**Alasan perubahan Kolom 1:**

| # | Perubahan | Alasan |
|---|---|---|
| 1 | `Resize Citra` **dihilangkan** | Resize sekarang dilakukan secara internal di dalam pipeline training enhancement (256×256) — bukan tahap dataset terpisah |
| 2 | `Auto-Labeling YOLO` **ditambahkan** | Secara logis, anotasi adalah bagian dari **persiapan data**, bukan preprocessing Enhancement |
| 3 | `Diamond "Label & bbox valid?"` **ditambahkan** | Mengeksplisitkan langkah QC (Quality Control) manual — operator harus memverifikasi label sebelum lanjut |
| 4 | `Split Dataset` **dipindah** dari Kol-2 ke Kol-1 | Split adalah persiapan data, bukan bagian dari proses enhancement |
| 5 | `Augmentasi` **dipindah** dari Kol-2 ke Kol-1 | Augmentasi adalah operasi dataset, logikanya ikut persiapan data |

---

### ⚙️ Kolom 2 — ENHANCEMENT & PREPROCESSING → ENHANCEMENT ZERO-DCE + CBAM

**SEBELUM** — hanya garis besar training:

```
Training Zero-DCE + CBAM
  (L_spa + L_exp + L_col + L_tvA)  ← bobot tidak disebutkan
  → [Diamond] Loss konvergen?
       Tidak → (loop)
       Ya    → Model Enhancement Zero-DCE + CBAM  (silinder)
  → Enhancement Seluruh Citra (resolusi asli dipertahankan)
  → Evaluasi Kualitas Citra (Entropy, Contrast, BRISQUE, CV_illum)
  → Anotasi & Labeling pada Citra Hasil Enhancement  ← (dipindah ke Kol-1)
  → Split Data 70:15:15                              ← (dipindah ke Kol-1)
  → Augmentasi → Data Latih saja                    ← (dipindah ke Kol-1)
```

**SEKARANG** — step-by-step arsitektur Zero-DCE + CBAM:

```
Input Citra Latih (resize 256×256, batch 4)
  → Darkness Map  D(x) = 1 − I(x)
  → Dark-Aware Channel Attention
    Dark-Guided Spatial Attention (kernel 7×7)
  → Estimasi Kurva Zero-DCE (8 iterasi LE-curve, 32 filter)
  → [Kotak hijau] Non-Reference Loss
      L_spa×10 + L_exp×10 + L_col×5 + L_tvA×200
  → [Diamond] Loss konvergen?  (maks. 600 epoch)
       Tidak → (loop)
       Ya    → Model Enhancement best_zero_dce_cbam.pth  (silinder)
  → Enhancement Seluruh Split (resolusi asli 1:1 + upscaling 2×)
  → Evaluasi Kualitas Citra (Entropy, Contrast, BRISQUE, CV_illum)
```

**Alasan perubahan Kolom 2:**

| # | Perubahan | Alasan |
|---|---|---|
| 1 | Step `Darkness Map → Channel Attn → Spatial Attn → Kurva` **ditambahkan** | Menunjukkan **arsitektur internal model** secara eksplisit sesuai kode `dark_attention.py` dan `zero_dce_cbam.py` |
| 2 | Bobot loss `×10, ×10, ×5, ×200` **disebutkan** | Bobot sudah dikalibrasi di `configs/zero_dce_cbam.yaml` — parameter penting penelitian |
| 3 | Batas epoch `maks. 600` **ditambahkan** | Info konkret dari konfigurasi training `train_zero_dce_cbam.py` |
| 4 | `+ upscaling 2×` **ditambahkan** di enhancement output | Sesuai implementasi aktual di `enhance.py` |
| 5 | Anotasi, Split, Augmentasi **dipindah keluar** ke Kol-1 | Diurai ke tahap yang lebih tepat secara logis |

---

### 🎯 Kolom 3 — TRAINING DETEKSI OBJEK → TRAINING DETEKSI OBJEK (YOLO12)

**SEBELUM** — 3 skenario komparatif:

```
[Box biru] Tiga Skenario Eksperimen (label identik):
  A. Citra Original
  B. Zero-DCE Baseline          ← ADA
  C. Zero-DCE + CBAM (Usulan)  ← ADA

→ Input Data Latih + Validasi
→ Training Model Object Detection (YOLO12)
→ Validasi Model
→ [Diamond] mAP memadai?
       Tidak → Hyperparameter Tuning (loop)
       Ya    → Model Deteksi Walet (silinder)
```

**SEKARANG** — 2 skenario, informasi lebih teknis:

```
[Box biru] Dua Skenario Uji (label identik):
  A. Citra Original Low-Light
  B. Citra Enhanced Zero-DCE + CBAM   ← Skenario B Baseline DIHILANGKAN

→ Input Data Latih + Validasi (train 70% - val 15%)
→ Training Model Deteksi YOLO12n (pretrained COCO, 100 epoch)
→ Validasi Model per-Epoch
→ [Diamond] mAP memadai?
       Tidak → Hyperparameter Tuning (loop)
       Ya    → Model Deteksi Walet (best.pt per skenario)
            ↓
       Evaluasi Metrik Deteksi        ← BARU, sebelumnya di Kol-4
       (Precision, Recall, mAP@0.5, mAP@0.5:0.95, FPS)
```

**Alasan perubahan Kolom 3:**

| # | Perubahan | Alasan |
|---|---|---|
| 1 | Skenario **dari 3 → 2** (Skenario B Zero-DCE Baseline **dihilangkan**) | Penelitian fokus hanya membandingkan **A (tanpa enhancement)** vs **B (Zero-DCE + CBAM usulan)** |
| 2 | Detail `pretrained COCO, 100 epoch` **ditambahkan** | Info teknis konkret sesuai script `detection/train_yolo.py` |
| 3 | `Evaluasi Metrik Deteksi` **dipindah dari Kol-4 ke Kol-3** | Detection metrics (Precision, Recall, mAP, FPS) lebih tepat berada **di bawah model yang sama** yang menghasilkannya |
| 4 | `FPS` **ditambahkan** sebagai metrik | Relevan untuk menilai kelayakan real-time pada kamera CCTV |

---

### 📊 Kolom 4 — PENGUJIAN & ESTIMASI POPULASI → COUNTING & ESTIMASI POPULASI

**SEBELUM** — evaluasi gabungan detection + counting:

```
Input Data Uji
→ Deteksi Objek (Walet)
→ Perhitungan Objek (Counting)
→ Evaluasi Performa            ← gabung detection + counting
    Precision, Recall, mAP@0.5, mAP@0.5:0.95, MAE, RMSE
→ Perbandingan Skenario A vs B vs C
→ Estimasi Populasi (Growth%)
→ Klasifikasi Pertumbuhan
→ Selesai
```

**SEKARANG** — fokus murni pada counting & estimasi:

```
Input Data Uji (test 15%, belum dilihat model)
→ Deteksi Objek Walet
→ Perhitungan Objek per-Frame (counting)
→ Evaluasi Error Counting      ← hanya MAE & RMSE
    (Precision/Recall sudah dipindah ke Kol-3)
→ Perbandingan Skenario A vs B  ← 3 → 2 skenario
→ Estimasi Populasi
    Growth(%) = (Pn - Pn-1) / Pn-1 x 100
→ Klasifikasi Pertumbuhan
→ Selesai
```

**Alasan perubahan Kolom 4:**

| # | Perubahan | Alasan |
|---|---|---|
| 1 | Judul `PENGUJIAN` → `COUNTING` | Menekankan bahwa Kol-4 **khusus counting & populasi**, bukan evaluasi deteksi (yang sudah di Kol-3) |
| 2 | Evaluasi gabungan **dipecah**: detection metrics → Kol-3, counting metrics → Kol-4 | Pemisahan **tanggung jawab yang lebih bersih** — setiap kolom punya scope jelas |
| 3 | Skenario perbandingan **dari A/B/C → A/B** | Konsisten dengan pengurangan skenario di Kol-3 |
| 4 | `per-Frame` **ditambahkan** pada Counting | Lebih presisi secara teknis sesuai implementasi `count_population.py` |
| 5 | Keterangan `test 15%, belum dilihat model` **ditambahkan** | Menegaskan prinsip **data leakage prevention** (model tidak pernah melihat test set saat training) |

---

## Diagram Perbedaan Struktural

```
SEBELUM                              SEKARANG
──────────────────────────────────────────────────────────────────
Kol-1 (DATA)                         Kol-1 (PERSIAPAN & PENGOLAHAN DATA)
  Mulai                                Mulai
  Pengumpulan Data                     Rekaman CCTV
  Ekstraksi Frame                      Ekstraksi Frame
  Resize Citra ─────────┐              Seleksi Frame (SSIM)
                        │              Auto-Labeling YOLO
                        │              [Diamond] QC Label valid?
                        │              Split Dataset 70:15:15  ← dipindah dari Kol-2
                        │              Augmentasi              ← dipindah dari Kol-2
                        ▼
Kol-2 (ENHANCEMENT & PREPROCESSING)   Kol-2 (ENHANCEMENT ZERO-DCE + CBAM)
  Training (ringkas)                   Input Citra (256×256)
  [Diamond] Loss konvergen?            Darkness Map D(x) = 1 - I(x)  ← baru
  Model Enhancement (silinder)         Channel Attention + Spatial Attn 7×7  ← baru
  Enhancement Citra                    Kurva Zero-DCE 8-iter  ← baru
  Evaluasi Kualitas                    Non-Reference Loss ×bobot eksplisit  ← baru
  Anotasi & Labeling ────┐             [Diamond] Loss konvergen? (600 ep)
  Split Data 70:15:15 ───┤ dipindah    Model Enhancement (silinder)
  Augmentasi ────────────┘ ke Kol-1   Enhancement + upscaling 2×  ← baru
                                       Evaluasi Kualitas

Kol-3 (TRAINING DETEKSI OBJEK)       Kol-3 (TRAINING DETEKSI OBJEK YOLO12)
  3 Skenario A/B/C                     2 Skenario A/B  ← Skenario B Baseline dihapus
  Input Data Latih                     Input Data Latih + Validasi
  Training YOLO12                      Training YOLO12n (COCO, 100 ep)  ← detail
  Validasi Model                       Validasi per-Epoch
  [Diamond] mAP memadai?               [Diamond] mAP memadai?
  Hyperparameter Tuning (loop)         Hyperparameter Tuning (loop)
  Model Deteksi Walet                  Model Deteksi Walet
                                       Evaluasi Metrik Deteksi  ← dipindah dari Kol-4

Kol-4 (PENGUJIAN & ESTIMASI)          Kol-4 (COUNTING & ESTIMASI POPULASI)
  Input Data Uji                       Input Data Uji (test 15%, unseen)
  Deteksi Objek                        Deteksi Objek
  Counting                             Counting per-Frame  ← per-frame
  Evaluasi Performa                    Evaluasi Error Counting  ← MAE & RMSE saja
    (P, R, mAP, MAE, RMSE) ─────         (Precision/Recall sudah di Kol-3)
  Perbandingan A vs B vs C             Perbandingan A vs B  ← 2 skenario
  Estimasi Populasi Growth(%)          Estimasi Populasi Growth(%)
  Klasifikasi Pertumbuhan              Klasifikasi Pertumbuhan
  Selesai                              Selesai
```

---

> **Perubahan paling signifikan**: Skenario dari **3 → 2**. Skenario B (Zero-DCE Baseline tanpa CBAM)
> dihilangkan, sehingga eksperimen hanya membandingkan **A (Original/tanpa enhancement)**
> vs **B (Zero-DCE + CBAM Modifikasi)**. Ini memfokuskan kontribusi utama pada modifikasi CBAM
> berbasis Darkness Map.

> **Perubahan lainnya** bersifat klarifikasi dan pemindahan — bukan perubahan metodologi fundamental.
> Konten teknis (loss function, SSIM threshold, rasio split, formula Growth%) tetap sama,
> hanya lebih diperjelas dan diletakkan di posisi yang lebih logis.
