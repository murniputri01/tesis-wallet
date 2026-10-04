# Dokumentasi Algoritma dan Pemetaan Kode Program

Penelitian: **"MODIFIKASI ZERO-DCE BERBASIS CONVOLUTIONAL BLOCK ATTENTION MODULE (CBAM) UNTUK ENHANCEMENT CITRA LOW-LIGHT CCTV INFRAMERAH PADA KANDANG BURUNG WALET"**

Dokumen ini menjelaskan secara teknis dan matematis alur arsitektur deep learning yang digunakan dalam penelitian ini (Zero-DCE dan Modifikasi CBAM berbasis Darkness Map) sesuai diagram alur **Algoritma Zero-DCE + CBAM**, beserta pemetaan presisi lokasi baris kode di dalam struktur project.

---

## 1. DIAGRAM ALUR ARSITEKTUR ALGORITMA ZERO-DCE + CBAM

Berdasarkan rancangan alur arsitektur jaringan deep learning:

```text
Input Image I(x) [B, 3, H, W] ────────┐
   │                                  │
   ▼                                  ▼
Feature Extraction CNN             Darkness Map Extractor
(DCE-Net Layers 1-6)               D(x) = 1 - Mean(I(x))
   │                                  │
   ├──► Feature F [B, 32, H, W] ◄─────┤
   │                                  │
   │   ┌──────────────────────────────┴───────────────────────────┐
   │   │             DARK-AWARE CHANNEL ATTENTION                 │
   │   │                                                          │
   │   │   Input Feature F                                        │
   │   │     ├──► Max_Pool_dark (F * D) ──┐                       │
   │   │     │                            ├─► Shared MLP ─► Sigmoid ─► Mc_dark
   │   │     └──► Avg_Pool_dark (F * D) ──┘                           │
   │   │                                                              ▼
   │   │   Channel-refined Feature Fc ◄────────────────────────────── ⊗
   │   └──────────────────────────────┬───────────────────────────┘
   │                                  │
   │   ┌──────────────────────────────┴───────────────────────────┐
   │   │             DARK-GUIDED SPATIAL ATTENTION                │
   │   │                                                          │
   │   │   Channel-refined Feature Fc                             │
   │   │     └──► [Avg_Pool_c, Max_Pool_c] ─► Conv (7x7) ─► Sigmoid ─► Spatial Map
   │   │                                                               │
   │   │   Guided Spatial Attention Ms_dark ◄───────────────── ⊙ D(x) ┘
   │   │                                                               │
   │   │   CBAM Residual Feature Fcs = Fc * (1 + Ms_dark) ◄──────────── ⊗
   │   └──────────────────────────────┬───────────────────────────┘
   │                                  │
   ▼                                  ▼
Adaptive Curve Estimation Head A(x) = Tanh(Conv7(Concat(x1, Fcs)))
   │
   ▼
Zero-DCE Iterative Enhancement Curve
E_n(x) = E_{n-1}(x) + A_n(x) * E_{n-1}(x) * (1 - E_{n-1}(x))  (n = 1..8)
   │
   ▼
Output Enhanced Image [B, 3, H, W]
```

---

## 2. SKENARIO EKSPERIMEN PENELITIAN (SKENARIO A DAN B)

Penelitian ini membagi pengujian ke dalam 2 skenario komparatif untuk menguji dampak peningkatan kualitas citra terhadap deteksi objek burung walet:

1. **Skenario A (Original / Benchmark Tanpa Enhancement)**:
   - **Pipeline**: `Original Low-Light CCTV` $\rightarrow$ `Deteksi Objek (YOLO)`
   - **Deskripsi**: Menggunakan citra mentah asli dari CCTV inframerah tanpa proses *enhancement* (langsung dari `data/splits/test/images`). Berguna sebagai *baseline* awal untuk mengukur sejauh mana pencahayaan rendah merusak akurasi deteksi.
2. **Skenario B (Proposed Method - Zero-DCE + CBAM Modifikasi)**:
   - **Pipeline**: `Original Low-Light CCTV` $\rightarrow$ `Zero-DCE + CBAM Modifikasi` $\rightarrow$ `Deteksi Objek (YOLO)`
   - **Deskripsi**: Metode utama yang diusulkan. Memproses citra menggunakan Zero-DCE yang dimodifikasi dengan *Darkness Map* $D(x)$, *Dark-Aware Channel Attention*, dan *Dark-Guided Spatial Attention* untuk memfokuskan peningkatan kualitas pada area sangat gelap di dalam kandang walet.

---

## 3. ALGORITMA ZERO-DCE (DASAR TEORITIS)

Zero-Reference Deep Curve Estimation (Zero-DCE) adalah metode *deep learning* peningkatan kualitas citra pencahayaan rendah tanpa memerlukan *paired ground-truth* (pasangan foto terang-gelap).

### A. DCE-Net (Feature Extractor & Parameter Estimator)
DCE-Net menggunakan 7 layer konvolusi simetris dengan *skip connections* untuk mengestimasi peta parameter kurva $A(x)$.

- **Formulasi Output Layer**:
  $$A(x) = \tanh(\text{Conv}_7(\text{Concat}(x_1, x_6)))$$
  Aktivasi `tanh` digunakan untuk membatasi parameter kurva $A(x)$ di rentang $[-1.0, 1.0]$.
- **Dimensi Output**: $[B, 24, H, W]$ di mana $24 = 8 \text{ iterasi} \times 3 \text{ channel RGB}$.

### B. Iterative Quadratic Enhancement Curve
Fungsi peningkat kecerahan citra secara bertahap menggunakan kurva kuadratik:
$$E_0(x) = I(x)$$
$$E_n(x) = E_{n-1}(x) + A_n(x) \cdot E_{n-1}(x) \cdot (1 - E_{n-1}(x)) \quad \text{untuk } n = 1, 2, \dots, 8$$

---

## 4. ALGORITMA MODIFIKASI CBAM BERBASIS DARKNESS MAP (SKENARIO B)

CBAM (*Convolutional Block Attention Module*) dimodifikasi dengan memasukkan informasi tingkat kegelapan (*Darkness Map*) $D(x) = 1 - I(x)$ agar model memfokuskan perhatian secara spesifik pada area gelap kandang walet.

### A. Darkness Map Extractor
Mengekstrak peta tingkat kegelapan lokal dari citra input:
$$D(x) = 1.0 - \text{Mean}_{\text{channel}}(I(x))$$
Nilai $D(x) \in [0.0, 1.0]$, di mana $1.0$ merepresentasikan piksel sangat gelap dan $0.0$ merepresentasikan piksel terang.

### B. Dark-Aware Channel Attention Module
Memperhitungkan bobot channel dengan menggabungkan Max Pooling dan Average Pooling yang dibobot secara khusus oleh $D(x)$:

1. **Pooling Berbobot Kegelapan**:
   $$\text{AvgPool}_{\text{dark}} = \frac{\sum_{x} [F(x) \cdot D(x)]}{\sum_{x} D(x) + \epsilon}$$
   $$\text{MaxPool}_{\text{dark}} = \max_{x} [F(x) \cdot D(x)]$$
2. **Shared MLP Layer ($1 \times 1$ Conv)**:
   $$\text{MLP}(v) = \text{Conv}_{1 \times 1}^{(2)}\left(\text{ReLU}\left(\text{Conv}_{1 \times 1}^{(1)}(v)\right)\right)$$
3. **Channel Attention Map $M_{c\_dark}$**:
   $$M_{c\_dark} = \sigma\left(\text{MLP}(\text{AvgPool}_{\text{dark}}) + \text{MLP}(\text{MaxPool}_{\text{dark}})\right)$$
4. **Channel-refined Feature $F_c$**:
   $$F_c = M_{c\_dark} \otimes F$$

### C. Dark-Guided Spatial Attention Module
Menggabungkan Max Pooling dan Average Pooling sepanjang channel axis, diproses via Konvolusi $7 \times 7$, dan dipandu secara langsung oleh $D(x)$:

1. **Pooling Spasial**: $[\text{AvgPool}_c(F_c), \text{MaxPool}_c(F_c)]$ sepanjang channel axis.
2. **Spatial Attention Map $M_{s\_dark}$**:
   $$M_{s\_dark} = \sigma\left(\text{Conv}_{7 \times 7}([\text{AvgPool}_c(F_c), \text{MaxPool}_c(F_c)])\right) \odot D(x)$$
3. **Residual Gating Refinement ($F_{cs}$)**:
   $$F_{cs} = F_c \otimes (1.0 + M_{s\_dark})$$
   *Catatan Formulasi*: Penggunaan residual gating $(1.0 + M_{s\_dark})$ memastikan attention menguatkan fitur pada area gelap tanpa memadamkan fitur pada area yang sudah cukup terang.

---

## 5. FUNGSI KERUGIAN (NON-REFERENCE LOSS FUNCTIONS)

Pelatihan dilakukan tanpa citra referensi (*zero-reference*) menggunakan kombinasi 4 fungsi kerugian:

$$L_{\text{total}} = \lambda_{\text{spa}} L_{\text{spa}} + \lambda_{\text{exp}} L_{\text{exp}} + \lambda_{\text{col}} L_{\text{col}} + \lambda_{\text{tvA}} L_{\text{tvA}}$$

1. **Spatial Consistency Loss ($L_{\text{spa}}$)**: Mempertahankan kontras lokal 4-arah (kiri, kanan, atas, bawah) antara citra input dan output.
2. **Exposure Control Loss ($L_{\text{exp}}$)**: Mengukur jarak kuadrat daerah lokal patch $16 \times 16$ terhadap nilai paparan target ($E = 0.55$). Pada Skenario B, dilengkapi *Adaptive Pixel-Wise Dark Weighting* berbasis $D(x)^\gamma$.
3. **Color Constancy Loss ($L_{\text{col}}$)**: Berdasarkan prinsip Gray-World untuk mencegah pergeseran warna yang tidak alami.
4. **Illumination Smoothness Loss ($L_{\text{tvA}}$)**: Menjaga kehalusan gradien spasial pada peta kurva $A(x)$ menggunakan Total Variation (TV) loss.

---

## 6. PEMETAAN LOKASI ALGORITMA DALAM FILE PROGRAM

Berikut adalah tabel pemetaan lengkap lokasi algoritma dan fungsi matematika dalam file kode Python (diperbarui sesuai struktur repository):

| Komponen Algoritma | Nama File Kode | Nama Class / Fungsi | Nomor Baris Kode |
| :--- | :--- | :--- | :--- |
| **Darkness Map Calculation** | [`enhancement/models/dark_attention.py`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/enhancement/models/dark_attention.py) | `compute_darkness_map()` | Baris 17 – 36 |
| **Channel Attention Module** | [`enhancement/models/dark_attention.py`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/enhancement/models/dark_attention.py) | `DarkAwareChannelAttention` | Baris 39 – 84 |
| **Spatial Attention Module** | [`enhancement/models/dark_attention.py`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/enhancement/models/dark_attention.py) | `DarkGuidedSpatialAttention` | Baris 87 – 127 |
| **Modul CBAM Lengkap** | [`enhancement/models/dark_attention.py`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/enhancement/models/dark_attention.py) | `DarkCBAMModule` | Baris 130 – 148 |
| **Integrasi Zero-DCE + CBAM** | [`enhancement/models/zero_dce_cbam.py`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/enhancement/models/zero_dce_cbam.py) | `ZeroDCE_CBAM` | Baris 20 – 94 |
| **Adaptive Curve Estimation Head** | [`enhancement/models/zero_dce_cbam.py`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/enhancement/models/zero_dce_cbam.py) | `ZeroDCE_CBAM.forward()` | Baris 84 |
| **Spatial Consistency Loss ($L_{\text{spa}}$)** | [`enhancement/losses/spatial_consistency.py`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/enhancement/losses/spatial_consistency.py) | `SpatialConsistencyLoss` | Baris 12 – 53 |
| **Exposure Control Loss ($L_{\text{exp}}$)** | [`enhancement/losses/exposure_control.py`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/enhancement/losses/exposure_control.py) | `ExposureControlLoss` | Baris 11 – 37 |
| **Color Constancy Loss ($L_{\text{col}}$)** | [`enhancement/losses/color_constancy.py`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/enhancement/losses/color_constancy.py) | `ColorConstancyLoss` | Baris 11 – 28 |
| **Illumination Smoothness Loss ($L_{\text{tvA}}$)** | [`enhancement/losses/illumination_smoothness.py`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/enhancement/losses/illumination_smoothness.py) | `IlluminationSmoothnessLoss` | Baris 11 – 30 |
| **Total Non-Reference Loss** | [`enhancement/losses/__init__.py`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/enhancement/losses/__init__.py) | `ZeroDCELoss` | Baris 15 – 72 |
| **Training Pipeline Zero-DCE+CBAM** | [`enhancement/train_zero_dce_cbam.py`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/enhancement/train_zero_dce_cbam.py) | `train_zero_dce_cbam()` | Baris 32 – 175 |
| **Batch Enhancement Generator** | [`enhancement/enhance.py`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/enhancement/enhance.py) | `enhance_dataset()` | Baris 42 – 180 |
| **2-Panel & Technical Evaluator** | [`enhancement/evaluate_enhancement.py`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/enhancement/evaluate_enhancement.py) | `evaluate_enhancement()` | Baris 128 – 250 |
