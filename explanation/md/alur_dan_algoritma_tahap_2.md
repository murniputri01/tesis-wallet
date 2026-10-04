# Dokumen Alur dan Algoritma Tahap 2: Enhancement Zero-DCE + Modifikasi CBAM

Penelitian: **"MODIFIKASI ZERO-DCE BERBASIS CONVOLUTIONAL BLOCK ATTENTION MODULE (CBAM) UNTUK ENHANCEMENT CITRA LOW-LIGHT CCTV INFRAMERAH PADA KANDANG BURUNG WALET"**

Dokumen ini menyajikan panduan sistematis, teknis, dan matematis mengenai alur pemrosesan deep learning di dalam **Tahap 2 (Model Enhancement Zero-DCE + CBAM Modifikasi)**. Setiap komponen dipisahkan secara tegas dan terstruktur antara **Penjelasan Konsep & Formulasi Matematika** dengan **Lokasi & Potongan Kode Program Implementasi**.

---

## 1. DIAGRAM ALUR TAHAP 2

```text
Citra Input Low-Light I(x) [B, 3, H, W] (data/splits/)
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. PELATIHAN MODEL ZERO-DCE + MODIFIKASI CBAM (SKENARIO C) │  (train_zero_dce_cbam.py)
│                                                             │
│    Input Image I(x) ─────────────────────────┐              │
│       │                                      │              │
│       ▼                                      ▼              │
│    Feature Extractor CNN             Darkness Map Extractor │
│    (DCE-Net Layers 1-6)              D(x) = 1 - Mean(I(x))  │
│       │                                      │              │
│       ├──► Feature F [B, 32, H, W] ◄─────────┤              │
│       │                                      │              │
│       │   ┌──────────────────────────────────┴──────────┐   │
│       │   │     DARK-AWARE CHANNEL ATTENTION            │   │
│       │   │  AvgPool_dark(F*D), MaxPool_dark(F*D)       │   │
│       │   │  Channel-refined Feature Fc ◄───────────────┘   │
│       │   │                                  │              │
│       │   │   ┌──────────────────────────────┴──────────┐   │
│       │   │   │     DARK-GUIDED SPATIAL ATTENTION       │   │
│       │   │   │  Ms_dark = Conv7x7([Avg, Max]) * D(x)   │   │
│       │   │   │  Fcs = Fc * (1.0 + Ms_dark) ◄───────────┘   │
│       │   └──────────────────────────────────┬──────────┘   │
│       ▼                                      ▼              │
│    Adaptive Curve Estimation Head A(x) = Tanh(Conv7(Concat)) │
│       │                                                     │
│       ▼                                                     │
│    Zero-DCE Iterative Enhancement Curve (n = 1 .. 8)         │
│    E_n(x) = E_{n-1}(x) + A_n(x) * E_{n-1}(x) * (1 - E_{n-1}(x))│
│       │                                                     │
│       ▼                                                     │
│    Optimasi 4 Non-Reference Loss Functions:                 │
│    L_total = λ_spa*L_spa + λ_exp*L_exp + λ_col*L_col + λ_tvA*L_tvA
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼ Checkpoint (.pth)
┌─────────────────────────────────────────────────────────────┐
│ 2. BATCH ENHANCEMENT GENERATOR                              │  (enhance.py)
│    Batch inference citra di Train, Val, dan Test set        │
│    Output: data/enhanced/zero_dce_cbam/                     │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. EVALUASI METRIK KUALITAS CITRA NON-REFERENCE             │  (evaluate_enhancement.py)
│    Hitung: Entropy, Contrast, BRISQUE, & CV_illum           │
│    Generate Figur Visualisasi 2-Panel (Original vs Enhanced)│
│    Output: results/enhancement/                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. DETAIL ALGORITMA DAN IMPLEMENTASI PROGRAM TIAP KOMPONEN

---

### A. Darkness Map Extractor $D(x)$

#### 📘 1. Penjelasan Konsep & Formulasi Matematika

- **Konsep & Algoritma**:
  Mengekstrak peta distribusi tingkat kegelapan citra input $I(x)$ secara piksel demi piksel. Nilai $D(x)$ merepresentasikan bobot atensi tingkat rendahnya intensitas cahaya pada area tertentu di ruang kandang walet.

- **Formulasi Matematika**:
  $$D(x) = 1.0 - \text{Mean}_{\text{channel}}(I(x))$$
  - Di mana $I(x) \in [0.0, 1.0]$ adalah matriks intensitas citra input.
  - Nilai $D(x) \in [0.0, 1.0]$: Piksel mendekati $1.0$ menunjukkan sudut paling gelap, sedangkan mendekati $0.0$ menunjukkan area yang telah cukup terang.

---

#### 💻 2. Lokasi Implementasi & Potongan Kode Program

- **File Program**: [`enhancement/models/dark_attention.py`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/enhancement/models/dark_attention.py)
- **Fungsi Utama**: [`compute_darkness_map()`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/enhancement/models/dark_attention.py#L17-L37)
- **Pemanggilan Model**: [`ZeroDCE_CBAM.forward()#L67-L69`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/enhancement/models/zero_dce_cbam.py#L67-L69)
- **Letak Baris Kode**: Baris 17 – 37

```python
# [enhancement/models/dark_attention.py] Baris 17-37
def compute_darkness_map(x: torch.Tensor, threshold: float = None) -> torch.Tensor:
    # Rata-rata 3 channel RGB sepanjang axis channel (dim=1)
    intensity = torch.mean(x, dim=1, keepdim=True)
    D = 1.0 - intensity  # D(x) = 1.0 - I(x)

    if threshold is not None:
        D = torch.where(D >= threshold, D, torch.zeros_like(D))

    return torch.clamp(D, 0.0, 1.0)
```

---

### B. Dark-Aware Channel Attention Module

#### 📘 1. Penjelasan Konsep & Formulasi Matematika

- **Konsep & Algoritma**:
  Mekanisme *Channel Attention* tradisional dimodifikasi dengan memasukkan pembobotan langsung dari peta kegelapan $D(x)$ pada operasi pooling spasial, sehingga jaringan memprioritaskan fitur saluran yang membawa informasi pada bagian ruang yang gelap.

- **Formulasi Matematika**:
  1. **Pooling Berbobot Kegelapan**:
     $$\text{AvgPool}_{\text{dark}} = \frac{\sum_{x} [F(x) \cdot D(x)]}{\sum_{x} D(x) + \epsilon}$$
     $$\text{MaxPool}_{\text{dark}} = \max_{x} [F(x) \cdot D(x)]$$
  2. **Channel Attention Map ($M_{c\text{\_dark}}$)**:
     $$M_{c\text{\_dark}} = \sigma\left(\text{MLP}(\text{AvgPool}_{\text{dark}}) + \text{MLP}(\text{MaxPool}_{\text{dark}})\right)$$
  3. **Channel-Refined Feature ($F_c$)**:
     $$F_c = M_{c\text{\_dark}} \otimes F$$

---

#### 💻 2. Lokasi Implementasi & Potongan Kode Program

- **File Program**: [`enhancement/models/dark_attention.py`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/enhancement/models/dark_attention.py)
- **Kelas Utama**: [`DarkAwareChannelAttention`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/enhancement/models/dark_attention.py#L39-L85)
- **Letak Baris Kode**: Baris 69 – 84

```python
# [enhancement/models/dark_attention.py] Baris 69-84
# Dark-Aware Average Pooling & Max Pooling
D_sum = torch.sum(D, dim=[2, 3], keepdim=True) + 1e-6
avg_dark = torch.sum(F_in * D, dim=[2, 3], keepdim=True) / D_sum

max_dark, _ = torch.max(F_in * D, dim=3, keepdim=True)
max_dark, _ = torch.max(max_dark, dim=2, keepdim=True)

# Shared MLP + Sigmoid Activation
mlp_avg = self.mlp(avg_dark)
mlp_max = self.mlp(max_dark)
Mc_dark = self.sigmoid(mlp_avg + mlp_max)

# Channel Refinement
Fc = Mc_dark * F_in
```

---

### C. Dark-Guided Spatial Attention Module

#### 📘 1. Penjelasan Konsep & Formulasi Matematika

- **Konsep & Algoritma**:
  Atensi spasial $7 \times 7$ dipandu secara langsung oleh $D(x)$ dan digabungkan dengan mekanisme *Residual Gating* $(1.0 + M_{s\text{\_dark}})$. Pendekatan ini secara selektif menguatkan representasi fitur pada sudut gelap kandang walet tanpa merusak area yang telah terang.

- **Formulasi Matematika**:
  1. **Spatial Attention Map ($M_{s\text{\_dark}}$)**:
     $$M_{s\text{\_dark}} = \sigma\left(\text{Conv}_{7 \times 7}([\text{AvgPool}_c(F_c), \text{MaxPool}_c(F_c)])\right) \odot D(x)$$
  2. **Residual Gating Refinement ($F_{cs}$)**:
     $$F_{cs} = F_c \otimes (1.0 + M_{s\text{\_dark}})$$

---

#### 💻 2. Lokasi Implementasi & Potongan Kode Program

- **File Program**: [`enhancement/models/dark_attention.py`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/enhancement/models/dark_attention.py)
- **Kelas Utama**: [`DarkGuidedSpatialAttention`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/enhancement/models/dark_attention.py#L87-L124)
- **Letak Baris Kode**: Baris 112 – 123

```python
# [enhancement/models/dark_attention.py] Baris 112-123
avg_out = torch.mean(Fc, dim=1, keepdim=True)
max_out, _ = torch.max(Fc, dim=1, keepdim=True)
concat = torch.cat([avg_out, max_out], dim=1)

# Spatial Attention Map dipandu langsung oleh Darkness Map D(x)
spatial_map = self.sigmoid(self.conv(concat))
Ms_dark = spatial_map * D

# Residual Gating Feature Refinement
Fcs = Fc * (1.0 + Ms_dark)
```

---

### D. Kurva Pencerahan Kuadratik Iteratif ($n = 1 \dots 8$)

#### 📘 1. Penjelasan Konsep & Formulasi Matematika

- **Konsep & Algoritma**:
  Menerapkan transformasi kurva kuadratik secara bertingkat sebanyak $n = 8$ iterasi. Mengubah pencahayaan citra secara adaptif dan kontinu tanpa menyebabkan overexposure maupun perubahan artefak warna yang drastis.

- **Formulasi Matematika**:
  $$E_0(x) = I(x)$$
  $$E_n(x) = E_{n-1}(x) + A_n(x) \cdot E_{n-1}(x) \cdot (1 - E_{n-1}(x)) \quad \text{untuk } n = 1, 2, \dots, 8$$
  - $A_n(x) \in [-1.0, 1.0]$: Peta parameter kurva 24-channel yang diestimasi oleh DCE-Net.

---

#### 💻 2. Lokasi Implementasi & Potongan Kode Program

- **File Program**: [`enhancement/models/zero_dce_cbam.py`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/enhancement/models/zero_dce_cbam.py)
- **Method Utama**: [`ZeroDCE_CBAM.enhance_curve()`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/enhancement/models/zero_dce_cbam.py#L45-L51)
- **Letak Baris Kode**: Baris 45 – 51

```python
# [enhancement/models/zero_dce_cbam.py] Baris 45-51
def enhance_curve(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    enhanced = x
    for i in range(self.n_iters):  # n_iters = 8
        a_n = A[:, i * 3:(i + 1) * 3, :, :]
        enhanced = enhanced + a_n * enhanced * (1.0 - enhanced)
    enhanced = torch.clamp(enhanced, 0.0, 1.0)
    return enhanced
```

---

### E. Fungsi Kerugian Tanpa Referensi (*Non-Reference Loss Functions*)

#### 📘 1. Penjelasan Konsep & Formulasi Matematika

Proses pelatihan tidak memerlukan data pasangan foto terang (*unpaired training*), dioptimalkan menggunakan kombinasi 4 loss functions:

$$L_{\text{total}} = \lambda_{\text{spa}} L_{\text{spa}} + \lambda_{\text{exp}} L_{\text{exp}} + \lambda_{\text{col}} L_{\text{col}} + \lambda_{\text{tvA}} L_{\text{tvA}}$$

Bobot default: $\lambda_{\text{spa}} = 5.0$, $\lambda_{\text{exp}} = 25.0$, $\lambda_{\text{col}} = 5.0$, $\lambda_{\text{tvA}} = 200.0$.

1. **Spatial Consistency Loss ($L_{\text{spa}}$)**:
   Mempertahankan gradien kontras lokal 4-arah (kiri, kanan, atas, bawah) antara citra input dan output:
   $$L_{\text{spa}} = \frac{1}{K} \sum_{i=1}^{K} \sum_{j \in \Omega(i)} (|Y_i - Y_j| - |I_i - I_j|)^2$$

2. **Exposure Control Loss ($L_{\text{exp}}$)**:
   Mengontrol intensitas rata-rata patch $16 \times 16$ agar mendekati nilai paparan target ($E = 0.55$):
   $$L_{\text{exp}} = \frac{1}{M} \sum_{k=1}^{M} |Y_k - E|^2$$

3. **Color Constancy Loss ($L_{\text{col}}$)**:
   Berdasarkan hipotesis *Gray-World* untuk mencegah penyimpangan warna:
   $$L_{\text{col}} = \sum_{\forall (p, q) \in \{(R,G), (R,B), (G,B)\}} (J^p - J^q)^2$$

4. **Illumination Smoothness Loss ($L_{\text{tvA}}$)**:
   Menjaga keteraturan dan kehalusan gradien peta kurva $A(x)$ menggunakan Total Variation:
   $$L_{\text{tvA}} = \frac{1}{N} \sum_{n=1}^{N} \sum_{c \in \{R,G,B\}} (|\nabla_x A_n^c| + |\nabla_y A_n^c|)^2$$

---

#### 💻 2. Lokasi Implementasi & Potongan Kode Program

- **File Gabungan Loss**: [`enhancement/losses/__init__.py`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/enhancement/losses/__init__.py) (`ZeroDCELoss`)
- **Implementasi Masing-Masing Loss**:

```python
# 1. Spatial Consistency Loss [enhancement/losses/spatial_consistency.py] Baris 38-53
D_org_left = F.conv2d(in_pool, self.weight_left, padding=1)
D_enh_left = F.conv2d(enh_pool, self.weight_left, padding=1)
loss_left = torch.pow(D_org_left - D_enh_left, 2)
# Rata-rata 4 arah: left, right, up, down
return torch.mean(loss_left + loss_right + loss_up + loss_down)

# 2. Exposure Control Loss [enhancement/losses/exposure_control.py] Baris 24-37
gray_enh = torch.mean(enhanced_img, dim=1, keepdim=True)
patch_mean = self.pool(gray_enh)  # AvgPool2d(patch_size=16)
loss = torch.mean(torch.pow(patch_mean - self.target_exposure, 2))

# 3. Color Constancy Loss [enhancement/losses/color_constancy.py] Baris 19-28
r_mean = torch.mean(enhanced_img[:, 0, :, :], dim=[1, 2])
g_mean = torch.mean(enhanced_img[:, 1, :, :], dim=[1, 2])
b_mean = torch.mean(enhanced_img[:, 2, :, :], dim=[1, 2])
loss = torch.mean(torch.sqrt(torch.pow(r_mean - g_mean, 2) + torch.pow(r_mean - b_mean, 2) + torch.pow(g_mean - b_mean, 2) + 1e-6))

# 4. Illumination Smoothness Loss [enhancement/losses/illumination_smoothness.py] Baris 27-30
h_tv = torch.pow((A[:, :, 1:, :] - A[:, :, :h_x - 1, :]), 2).sum()
w_tv = torch.pow((A[:, :, :, 1:] - A[:, :, :, :w_x - 1]), 2).sum()
return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size
```

---

### F. Metrik Evaluasi Kualitas Citra Non-Reference

#### 📘 1. Penjelasan Konsep & Formulasi Matematika

1. **Shannon Entropy ($H$)**: Mengukur kepadatan detail informasi gradasi:
   $$H = -\sum_{i=0}^{255} P(i) \log_2 P(i)$$
2. **Contrast ($\sigma$) & Koefisien Variasi Iluminasi ($CV_{\text{illum}}$)**:
   $$\sigma = \sqrt{\frac{1}{N}\sum_{i=1}^N (I_i - \mu)^2}, \quad CV_{\text{illum}} = \frac{\sigma}{\mu}$$
3. **BRISQUE Score**: Metrik kualitas citra berbasis distribusi MSCN dan AGGD ($0-100$, semakin rendah skornya semakin tinggi kualitas persepsinya).

---

#### 💻 2. Lokasi Implementasi & Potongan Kode Program

- **File Program**: [`enhancement/evaluate_enhancement.py`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/enhancement/evaluate_enhancement.py)
- **Letak Baris Kode**:
  - Entropy: Baris 96 – 101
  - Contrast & $CV_{\text{illum}}$: Baris 116 – 122
  - BRISQUE: Baris 61 – 94

```python
# [enhancement/evaluate_enhancement.py] Baris 96-101 & 116-122
def calculate_entropy(gray_img: np.ndarray) -> float:
    hist, _ = np.histogram(gray_img, bins=256, range=(0, 256))
    hist = hist / float(gray_img.size)
    hist = hist[hist > 0]
    return -float(np.sum(hist * np.log2(hist)))

# Kalkulasi Contrast dan Koefisien Variasi Iluminasi
mean_brightness = float(np.mean(gray))
contrast = float(np.std(gray))
cv_illumination = contrast / (mean_brightness + 1e-6)
```

---

## 3. TABEL RANGKUMAN PEMETAAN FILE KODE PROGRAM

| Komponen Algoritma | File Kode Program | Kelas / Fungsi Utama | Letak Baris Implementasi Formula |
| :--- | :--- | :--- | :--- |
| **Runner Utama Tahap 2** | [`scripts/run_tahap2.py`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/scripts/run_tahap2.py) | `main()` | Runner otomatis eksekusi Tahap 2 |
| **Darkness Map $D(x)$** | [`enhancement/models/dark_attention.py`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/enhancement/models/dark_attention.py) | [`compute_darkness_map()`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/enhancement/models/dark_attention.py#L17-L37) | [`L28-L36`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/enhancement/models/dark_attention.py#L28-L36) ($D = 1.0 - \text{Mean}(I)$) |
| **Dark-Aware Channel Attention** | [`enhancement/models/dark_attention.py`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/enhancement/models/dark_attention.py) | [`DarkAwareChannelAttention`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/enhancement/models/dark_attention.py#L39-L85) | [`L69-L84`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/enhancement/models/dark_attention.py#L69-L84) ($\text{Pool}_{\text{dark}}$, $M_c$, $F_c$) |
| **Dark-Guided Spatial Attention** | [`enhancement/models/dark_attention.py`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/enhancement/models/dark_attention.py) | [`DarkGuidedSpatialAttention`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/enhancement/models/dark_attention.py#L87-L124) | [`L113-L123`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/enhancement/models/dark_attention.py#L113-L123) ($M_{s\text{\_dark}}$, $1 + M_s$) |
| **Arsitektur Zero-DCE + CBAM** | [`enhancement/models/zero_dce_cbam.py`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/enhancement/models/zero_dce_cbam.py) | [`ZeroDCE_CBAM`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/enhancement/models/zero_dce_cbam.py#L19-L95) | [`L45-L51`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/enhancement/models/zero_dce_cbam.py#L45-L51) ($E_n$ kurva kuadratik), [`L67-L94`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/enhancement/models/zero_dce_cbam.py#L67-L94) |
| **Spatial Consistency Loss ($L_{\text{spa}}$)** | [`enhancement/losses/spatial_consistency.py`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/enhancement/losses/spatial_consistency.py) | [`SpatialConsistencyLoss`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/enhancement/losses/spatial_consistency.py#L12-L54) | [`L38-L53`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/enhancement/losses/spatial_consistency.py#L38-L53) (Gradien 4 arah) |
| **Exposure Control Loss ($L_{\text{exp}}$)** | [`enhancement/losses/exposure_control.py`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/enhancement/losses/exposure_control.py) | [`ExposureControlLoss`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/enhancement/losses/exposure_control.py#L11-L38) | [`L24-L37`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/enhancement/losses/exposure_control.py#L24-L37) (Patch pooling & $E = 0.55$) |
| **Color Constancy Loss ($L_{\text{col}}$)** | [`enhancement/losses/color_constancy.py`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/enhancement/losses/color_constancy.py) | [`ColorConstancyLoss`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/enhancement/losses/color_constancy.py#L11-L29) | [`L19-L28`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/enhancement/losses/color_constancy.py#L19-L28) (Prinsip Gray-World) |
| **Illumination Smoothness Loss ($L_{\text{tvA}}$)** | [`enhancement/losses/illumination_smoothness.py`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/enhancement/losses/illumination_smoothness.py) | [`IlluminationSmoothnessLoss`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/enhancement/losses/illumination_smoothness.py#L11-L34) | [`L27-L30`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/enhancement/losses/illumination_smoothness.py#L27-L30) (Total variation $A$) |
| **Total Loss Function** | [`enhancement/losses/__init__.py`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/enhancement/losses/__init__.py) | [`ZeroDCELoss`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/enhancement/losses/__init__.py#L15-L73) | [`L52-L72`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/enhancement/losses/__init__.py#L52-L72) (Kombinasi 4 komponen loss) |
| **Training Pipeline** | [`enhancement/train_zero_dce_cbam.py`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/enhancement/train_zero_dce_cbam.py) | [`train_zero_dce_cbam()`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/enhancement/train_zero_dce_cbam.py#L32-L175) | Training loop model Zero-DCE + CBAM |
| **Batch Generator Enhancement** | [`enhancement/enhance.py`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/enhancement/enhance.py) | [`enhance_dataset()`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/enhancement/enhance.py#L42-L180) | Generator dataset batch inferensi |
| **Evaluasi Metrik Citra & 2-Panel** | [`enhancement/evaluate_enhancement.py`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/enhancement/evaluate_enhancement.py) | [`evaluate_enhancement()`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/enhancement/evaluate_enhancement.py#L128-L250) | [`L61-L125`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/enhancement/evaluate_enhancement.py#L61-L125) (Entropy, BRISQUE, $CV_{\text{illum}}$) |

---

## 4. PERINTAH UTAMA EKSEKUSI TAHAP 2

Untuk menjalankan Tahap 2 secara otomatis:

```bash
# Generator enhancement & evaluasi metrik citra (Menggunakan Checkpoint Pre-trained)
python scripts/run_tahap2.py --video_stem video01

# Generator + training ulang model dari awal (Retraining khusus video tertentu)
python scripts/run_tahap2.py --video_stem video01 --train --num_threads 4
```

Hasil output akan tersimpan di:
- `checkpoints/zero_dce_cbam/video01/best_zero_dce_cbam.pth` (Weights Model)
- `data/enhanced/zero_dce_cbam/` (Citra Hasil Pencerahan)
- `results/enhancement/enhancement_metrics_report.csv` (Metrik Kualitas Citra CSV)
- `results/enhancement/2panel_comparison_*.png` (Figur 2-Panel Original vs Enhanced)
