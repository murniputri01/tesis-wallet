## Apakah algoritma ini hanya berfokus pada B H W, tidak pada R G B?

Penjelasan sederhananya: **TIDAK**.

Algoritma ini **tidak hanya berfokus pada lokasi/posisi gambar ($H, W$)**, tetapi **sangat memperhatikan warna ($R, G, B$)**.

---

### 💡 Pemahaman Sederhana Notasi $[B, C, H, W]$

Di dunia kecerdasan buatan (_AI / Deep Learning_), gambar diwakili oleh tensor 4 dimensi:

1. **$B$ (_Batch_)**: Jumlah lembar foto yang diproses sekaligus dalam satu putaran.
2. **$C$ (_Channels_)**: Warna gambar. Untuk gambar berwarna (RGB), jumlahnya ada **3 channel** yaitu **$R$ (Red/Merah), $G$ (Green/Hijau), dan $B$ (Blue/Biru)**.
3. **$H$ (_Height_)**: Tinggi foto (posisi vertikal piksel).
4. **$W$ (_Width_)**: Lebar foto (posisi horisontal piksel).

---

### 🧩 Bagaimana Algoritma Membagi Fokusnya?

Algoritma usulan ini membagi tugasnya menjadi **2 perhatian khusus**:

#### 1. Perhatian pada Warna ($R, G, B$) $\rightarrow$ _Channel Attention_

- Modul ini memeriksa channel warna ($R, G, B$).
- **Tujuannya**: Memastikan bahwa saat gambar gelap diterangkan, warna asli objek (burung walet dan dinding kandang) **tidak menjadi pudar, aneh, atau berubah warna**.

#### 2. Perhatian pada Posisi/Letak ($H, W$) $\rightarrow$ _Spatial Attention_

- Modul ini memeriksa titik-titik koordinat tinggi dan lebar foto ($H, W$).
- **Tujuannya**: Mencari di mana letak **sudut-sudut kandang walet yang sangat gelap** (misalnya pojok atas atau selipan kayu) agar AI menerangkan area gelap tersebut lebih kuat dibanding area tengah yang sudah cukup terang.

---

### 💻 Implementasi Kode Program dalam Proyek

Berikut adalah bukti implementasi kode Python tempat algoritma memproses dimensi **Warna ($R, G, B$)** dan **Posisi ($H, W$)**:

#### A. Ekstraksi Darkness Map $D(x)$ dari Channel Warna ($R, G, B$)
File: [`enhancement/models/dark_attention.py`](file:///c:/MyPrograms/swallow_detection/enhancement/models/dark_attention.py#L28-L36)

```python
intensity = torch.mean(x, dim=1, keepdim=True)  # Rata-rata intensitas 3 channel RGB (dim=1)
D = 1.0 - intensity                             # Hitung peta kegelapan piksel demi piksel
```

---

#### B. Perhatian pada Warna ($R, G, B$) / Channel Attention ($C$)
File: [`enhancement/models/dark_attention.py`](file:///c:/MyPrograms/swallow_detection/enhancement/models/dark_attention.py#L69-L84)

```python
# Mempersatukan dimensi posisi (H, W) untuk mengisolasi bobot tiap channel warna (C)
D_sum = torch.sum(D, dim=[2, 3], keepdim=True) + 1e-6                   # Sum sepanjang H dan W
avg_dark = torch.sum(F_in * D, dim=[2, 3], keepdim=True) / D_sum        # Average pooling berbobot kegelapan
max_dark, _ = torch.max(F_in * D, dim=3, keepdim=True)                  # Max pooling sepanjang W
max_dark, _ = torch.max(max_dark, dim=2, keepdim=True)                 # Max pooling sepanjang H

Mc_dark = self.sigmoid(self.mlp(avg_dark) + self.mlp(max_dark))         # Peta perhatian channel warna [B, C, 1, 1]
Fc = Mc_dark * F_in                                                     # Penguatan fitur channel warna
```

---

#### C. Perhatian pada Posisi/Letak ($H, W$) / Spatial Attention ($H, W$)
File: [`enhancement/models/dark_attention.py`](file:///c:/MyPrograms/swallow_detection/enhancement/models/dark_attention.py#L113-L127)

```python
# Mempersatukan channel warna (dim=1) untuk mengisolasi koordinat posisi spasial (H, W)
avg_out = torch.mean(Fc, dim=1, keepdim=True)                           # Rata-rata sepanjang channel warna
max_out, _ = torch.max(Fc, dim=1, keepdim=True)                         # Nilai maksimum sepanjang channel warna
concat = torch.cat([avg_out, max_out], dim=1)                           # Gabungkan fitur posisi [B, 2, H, W]

spatial_map = self.sigmoid(self.conv(concat))                           # Peta atensi lokasi [B, 1, H, W]
Ms_dark = spatial_map * D                                               # Dipandu langsung oleh Darkness Map D(x)
Fcs = Fc * (1.0 + Ms_dark)                                              # Penguatan posisi gelap di koordinat (H, W)
```

---

### 🎨 Bukti Bahwa Algoritma Sangat Menjaga Warna ($R, G, B$)

Algoritma ini memiliki rumus khusus bernama **Color Constancy Loss ($L_{\text{col}}$)**.

File: [`enhancement/losses/color_constancy.py`](file:///c:/MyPrograms/swallow_detection/enhancement/losses/color_constancy.py#L18-L28)

```python
# Menghitung rata-rata intensitas masing-masing channel warna RGB
r_mean = torch.mean(enhanced_img[:, 0, :, :], dim=[1, 2])               # Rata-rata channel Merah (R)
g_mean = torch.mean(enhanced_img[:, 1, :, :], dim=[1, 2])               # Rata-rata channel Hijau (G)
b_mean = torch.mean(enhanced_img[:, 2, :, :], dim=[1, 2])               # Rata-rata channel Biru (B)

# Menghitung selisih keseimbangan antar pasang warna (Prinsip Gray-World)
d_rg = torch.pow(r_mean - g_mean, 2)                                    # Jarak warna Merah vs Hijau
d_rb = torch.pow(r_mean - b_mean, 2)                                    # Jarak warna Merah vs Biru
d_gb = torch.pow(g_mean - b_mean, 2)                                    # Jarak warna Hijau vs Biru

loss = torch.mean(torch.sqrt(d_rg + d_rb + d_gb + 1e-6))               # Penjaga agar warna tetap alami & seimbang
```

- Jika warna Merah tiba-tiba menjadi terlalu dominan (gambar jadi kemerahan), AI akan otomatis menegurnya dan mengembalikan keseimbangan warnanya.
- Hasilnya, foto CCTV inframerah yang tadinya gelap gulita akan menjadi terang dengan **warna yang tetap alami dan tidak belang-belang**.
