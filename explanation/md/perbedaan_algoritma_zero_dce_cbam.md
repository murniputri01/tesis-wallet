# Perbandingan Diagram Algoritma Zero-DCE + CBAM: Sebelum vs Sekarang

> Berdasarkan:
> - **Sebelum**: [`algoritma_zero_dce_cbam_sebelum.jpeg`](../png/algoritma_zero_dce_cbam_sebelum.jpeg) — CBAM standar (Woo et al., 2018) yang ditempelkan ke Zero-DCE
> - **Sekarang**: [`algoritma_zero_dce_cbam_sekarang.png`](../png/algoritma_zero_dce_cbam_sekarang.png) — CBAM modifikasi berbasis *darkness map*, sesuai implementasi nyata pada kode
>
> Sumber kode acuan:
> - [`enhancement/models/zero_dce_cbam.py`](../../enhancement/models/zero_dce_cbam.py)
> - [`enhancement/models/dark_attention.py`](../../enhancement/models/dark_attention.py)

---

## Ringkasan Cepat

| Aspek | Sebelum (CBAM standar) | Sekarang (sesuai kode) |
|---|---|---|
| Darkness Map `D(x)` | **Tidak ada** | **Ada**, jadi cabang tersendiri dari input |
| Channel pooling | `Max_Pool` & `Avg_Pool` biasa | `MaxPool_dark` & `AvgPool_dark` (berbobot `D(x)`) |
| Shared layer | "FC1 / FC2, Shared Dense Layer" | Conv 1×1 `32 → 2 → 32` (reduction ratio 16) |
| Spatial attention | `Ms = Sigmoid(Conv2D([Max, Avg]))` | `Ms_dark = Sigmoid(Conv7×7([Max_c, Avg_c])) · D(x)` |
| Penggabungan akhir | Perkalian murni `F'' = Ms ⊗ F'` | **Residual gating** `Fcs = Fc · (1 + Ms_dark)` |
| Posisi modul atensi | Digambar di ujung feature extractor | Dipasang khusus pada fitur `x6` |
| Curve head | Kotak "Adaptive curve estimation A(x)" tanpa rumus | `A(x) = tanh(Conv3×3([x1, Fcs]))`, 24 map |
| Feature extractor | 3 kubus tanpa keterangan | conv1–conv6 (3×3, 32 ch, ReLU) + skip-concat |
| Rumus iterasi | Terpotong, tanpa `n` dan tanpa clamp | Utuh + `E_0 = I(x)`, `n = 1…8`, `clamp[0,1]` |
| Citra panel | Gambar contoh (bukan dari pipeline) | Frame asli proyek, darkness map dihitung langsung |

---

## Perubahan per Komponen

---

### 1. Darkness Map `D(x)` — komponen yang sama sekali tidak ada di diagram lama

**SEBELUM** — input citra langsung masuk ke feature extraction, lalu ke CBAM. Tidak ada sinyal apa pun yang memberi tahu modul atensi di mana letak area gelap.

**SEKARANG** — input bercabang dua: satu ke feature extractor, satu lagi ke ekstraksi darkness map.

```
Input I(x)
  ├──→ DCE-Net conv1–conv6 ──→ Feature Map F
  └──→ D(x) = 1 − mean_c I(x) ──→ dipakai di channel & spatial attention
```

Implementasi: [`dark_attention.py:30`](../../enhancement/models/dark_attention.py#L30)

```python
intensity = torch.mean(x, dim=1, keepdim=True)
D = 1.0 - intensity
```

**Alasan perubahan:** inilah kontribusi utama penelitian. Tanpa `D(x)`, diagram hanya menggambarkan CBAM generik — siapa pun yang menempelkan CBAM ke Zero-DCE akan menghasilkan gambar yang persis sama, sehingga novelty penelitian tidak terlihat.

---

### 2. Channel Attention — dari pooling biasa menjadi *dark-aware pooling*

**SEBELUM** — pooling global standar atas seluruh citra:

```
F → Max_Pool  ┐
              ├→ Shared Dense Layer (FC1, FC2) → ⊕ → Sigmoid → Mc
F → Avg_Pool  ┘
```

**SEKARANG** — pooling dibobot darkness map, sehingga statistik channel didominasi piksel gelap:

```
F, D(x) → MaxPool_dark = max[D(x) · F(x)]                    ┐
                                                             ├→ Shared MLP (Conv1×1 32→2→32) → ⊕ → Sigmoid → Mc_dark
F, D(x) → AvgPool_dark = Σ[D(x) · F(x)] / (Σ D(x) + ε)       ┘

Fc = Mc_dark ⊙ F
```

Implementasi: [`dark_attention.py:71-83`](../../enhancement/models/dark_attention.py#L71-L83)

```python
D_sum = torch.sum(D, dim=[2, 3], keepdim=True) + 1e-6
avg_dark = torch.sum(F_in * D, dim=[2, 3], keepdim=True) / D_sum

max_dark, _ = torch.max(F_in * D, dim=3, keepdim=True)
max_dark, _ = torch.max(max_dark, dim=2, keepdim=True)

Mc_dark = self.sigmoid(self.mlp(avg_dark) + self.mlp(max_dark))
Fc = Mc_dark * F_in
```

**Alasan perubahan:**

| # | Perubahan | Alasan |
|---|---|---|
| 1 | Pooling dibobot `D(x)` | Pada citra CCTV inframerah, sebagian besar piksel adalah latar sarang yang relatif terang. Pooling biasa membuat statistik channel didominasi latar, sehingga atensi tidak fokus ke area gelap tempat walet berada |
| 2 | Normalisasi `Σ D + ε` | Menjaga skala tetap seperti rata-rata (bukan jumlah), dan mencegah pembagian nol saat citra nyaris terang seluruhnya |
| 3 | "Shared Dense Layer" → Conv 1×1 | Implementasi memakai `nn.Conv2d(kernel_size=1)`, bukan `nn.Linear`, agar bekerja langsung pada tensor `[B, C, 1, 1]` tanpa reshape |
| 4 | Ditulis `32 → 2 → 32` | Jumlah channel `nf = 32` dengan `reduction_ratio = 16`, dan ada pengaman `max(4, C // r)` — angka aslinya perlu tampil agar diagram bisa diverifikasi terhadap kode ([`dark_attention.py:47`](../../enhancement/models/dark_attention.py#L47)) |

---

### 3. Spatial Attention — dipandu `D(x)` dan memakai *residual gating*

**SEBELUM** — spatial attention CBAM standar, hasil akhirnya perkalian murni:

```
F' → [Max_Pool, Avg_Pool] → Conv2D → Sigmoid → Ms → F'' = Ms ⊗ F'
```

**SEKARANG** — peta spasial dikalikan `D(x)` dulu, lalu digabung secara residual:

```
Fc → [MaxPool_c, AvgPool_c] → Conv 7×7 → Sigmoid → ⊗ D(x) → Ms_dark
Fcs = Fc ⊙ (1 + Ms_dark)
```

Implementasi: [`dark_attention.py:118-126`](../../enhancement/models/dark_attention.py#L118-L126)

```python
spatial_map = self.sigmoid(self.conv(concat))
Ms_dark = spatial_map * D
Fcs = Fc * (1.0 + Ms_dark)
```

**Alasan perubahan:**

| # | Perubahan | Alasan |
|---|---|---|
| 1 | `Ms_dark = Sigmoid(...) · D(x)` | Atensi spasial tidak hanya belajar sendiri di mana harus fokus, tetapi secara eksplisit dipandu peta kegelapan |
| 2 | Perkalian murni → `(1 + Ms_dark)` | Karena `Ms_dark ≤ 1`, bentuk `Fcs = Ms_dark · Fc` **meredam seluruh fitur**; akibatnya `A(x)` menyusut dan hasil enhancement justru lebih lemah daripada baseline Zero-DCE. Bentuk residual menguatkan area gelap hingga 2× (`D → 1`) tanpa memadamkan area terang (`D → 0`, fitur diteruskan apa adanya). Alasan ini didokumentasikan langsung di komentar kode ([`dark_attention.py:121-125`](../../enhancement/models/dark_attention.py#L121-L125)) |
| 3 | Kernel ditulis eksplisit `7×7` | Diagram lama hanya menulis "Conv2D"; kode mengunci kernel ke 3 atau 7 dan konfigurasi memakai 7 ([`dark_attention.py:94`](../../enhancement/models/dark_attention.py#L94)) |

---

### 4. Posisi Modul CBAM di Dalam Jaringan

**SEBELUM** — CBAM digambar seolah berdiri di ujung feature extractor, dan "CBAM feature" langsung diteruskan ke curve estimation.

**SEKARANG** — modul dipasang khusus pada fitur layer ke-6, dan keluarannya masih digabung lagi dengan skip connection `x1`:

```
x1 = ReLU(conv1(x))                      ← disimpan untuk skip
...
x6 = ReLU(conv6([x2, x5]))               ← Feature Map F, [B, 32, H, W]
Fcs, Mc_dark, Ms_dark = dark_cbam(x6, D)
A = tanh(conv7([x1, Fcs]))               ← skip x1 ikut masuk ke curve head
```

Implementasi: [`zero_dce_cbam.py:80-83`](../../enhancement/models/zero_dce_cbam.py#L80-L83)

**Alasan perubahan:** skip connection `x1` adalah bagian dari arsitektur DCE-Net asli dan tetap dipertahankan. Kalau tidak digambar, pembaca akan mengira curve head hanya menerima keluaran CBAM, padahal inputnya adalah gabungan `[x1, Fcs]` dengan `nf * 2 = 64` channel.

---

### 5. Curve Estimation Head dan Kurva Iteratif

**SEBELUM** — kotak "Adaptive curva estimation A (x)" tanpa rumus, dan rumus iterasi ditulis terpotong dua baris:

```
E_n(x) = E_{n-1}(x) + A_n(x)
E_{n-1}(x) * (1 - E_{n-1}(x))
```

**SEKARANG** — kedua bagian ditulis lengkap:

```
A(x) = tanh( Conv3×3( [x1, Fcs] ) )      →  [B, 24, H, W] = 8 iterasi × 3 channel RGB

E_n(x) = E_{n-1}(x) + A_n(x) · E_{n-1}(x) · (1 − E_{n-1}(x))
E_0(x) = I(x),  n = 1 … 8,  clamp[0, 1]
```

Implementasi: [`zero_dce_cbam.py:40`](../../enhancement/models/zero_dce_cbam.py#L40) dan [`zero_dce_cbam.py:49`](../../enhancement/models/zero_dce_cbam.py#L49)

**Alasan perubahan:**

| # | Perubahan | Alasan |
|---|---|---|
| 1 | Rumus `A(x)` ditampilkan | Menunjukkan bahwa `A(x)` dihasilkan konvolusi 3×3 dengan aktivasi `tanh`, sehingga nilainya berada di `[-1, 1]` (bisa menggelapkan maupun menerangkan) |
| 2 | Dimensi `[B, 24, H, W]` ditulis | 24 = `n_iters × 3` = 8 × 3; angka ini menjelaskan mengapa tiap iterasi punya parameter kurva RGB sendiri |
| 3 | `n = 1 … 8` dan `clamp[0,1]` ditambahkan | Jumlah iterasi adalah hyperparameter di [`configs/zero_dce_cbam.yaml`](../../configs/zero_dce_cbam.yaml) (`n_iters: 8`), dan clamp memastikan output tetap citra valid |
| 4 | Label "zero-reference" ditambahkan | Menegaskan bahwa pelatihan tidak memakai citra referensi terang — ciri khas Zero-DCE |

---

### 6. Citra pada Panel Input, Darkness Map, dan Output

**SEBELUM** — citra contoh yang tidak berasal dari pipeline penelitian.

**SEKARANG** — memakai frame asli dataset (`7_agustus_frame_000013`):

- **Input**: `data/original/images/7_agustus/7_agustus_frame_000013.jpg`
- **Output**: `data/enhanced/zero_dce_cbam/train/images/7_agustus/7_agustus_frame_000013.jpg`
- **Darkness map**: dihitung langsung di dalam skrip dengan rumus yang sama seperti di model (`D = 1 − mean_c I`), jadi yang tampil benar-benar `D(x)` dari frame tersebut, bukan ilustrasi

**Alasan perubahan:** panel darkness map hanya meyakinkan kalau nilainya dihitung, bukan digambar. Ini sekaligus memperlihatkan bahwa pada citra CCTV inframerah, tubuh walet memang jatuh di area dengan `D(x)` tinggi.

---

## Kesimpulan untuk Penulisan Skripsi

Tiga hal berikut adalah pembeda yang harus terlihat jelas pada diagram, dan hanya ada di versi sekarang:

1. **Darkness map `D(x) = 1 − I(x)`** sebagai sinyal pemandu yang diekstraksi langsung dari citra input, tanpa parameter yang perlu dilatih.
2. **Dark-aware channel attention** — pooling channel yang dibobot `D(x)`, sehingga deskriptor channel mewakili area gelap, bukan latar terang yang mendominasi.
3. **Dark-guided spatial attention dengan residual gating** — `Ms_dark` dikalikan `D(x)`, lalu digabung sebagai `Fc · (1 + Ms_dark)` agar penguatan bersifat aditif dan tidak meredam fitur area terang.

Diagram versi sebelum menggambarkan CBAM standar, sehingga tidak dapat dipakai untuk mengklaim kontribusi ketiga poin di atas.
