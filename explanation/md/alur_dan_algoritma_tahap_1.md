# Dokumen Alur dan Algoritma Tahap 1: Persiapan dan Pengolahan Dataset

Penelitian: **"MODIFIKASI ZERO-DCE BERBASIS CONVOLUTIONAL BLOCK ATTENTION MODULE (CBAM) UNTUK ENHANCEMENT CITRA LOW-LIGHT CCTV INFRAMERAH PADA KANDANG BURUNG WALET"**

Dokumen ini menyajikan panduan sistematis, teknis, dan matematis mengenai alur pemrosesan data di dalam **Tahap 1 (Persiapan dan Pengolahan Dataset)**. Setiap tahapan dipisahkan secara tegas dan terstruktur antara **Penjelasan Konsep & Formulasi Matematika** dengan **Lokasi & Potongan Kode Program Implementasi**.

---

## 1. DIAGRAM ALUR TAHAP 1

```text
Rekaman Video CCTV Mentah (.mp4 / .avi)
                  │
                  ▼
   ┌─────────────────────────────┐
   │ 1. Ekstraksi Frame Temporal │  (extract_frames.py)
   │    Sampling interval Δt = 1s│
   └──────────────┬──────────────┘
                  │
                  ▼
   ┌─────────────────────────────┐
   │ 2. Seleksi Redundansi SSIM  │  (select_frames.py)
   │    Threshold SSIM ≥ 0.95    │
   └──────────────┬──────────────┘
                  │
                  ▼
   ┌─────────────────────────────┐
   │ 3. Auto-Labeling YOLO       │  (auto_label.py)
   │    Pemetaan Class ID = 0    │
   └──────────────┬──────────────┘
                  │
                  ▼
   ┌─────────────────────────────┐
   │ 4. Split Dataset 70:15:15   │  (split_dataset.py)
   │    Video Group Split        │
   └──────────────┬──────────────┘
                  │
                  ▼
   ┌─────────────────────────────┐
   │ 5. Augmentasi Data Latih    │  (augment_dataset.py)
   │    Horisontal Flip, Bright  │
   └──────────────┬──────────────┘
                  │
                  ▼
   ┌─────────────────────────────┐
   │ 6. Validasi Format YOLO     │  (validate_dataset.py)
   │    Checking Label & Image   │
   └──────────────┬──────────────┘
                  │
                  ▼
Dataset Siap untuk Tahap 2 & 3 (Train, Val, Test)
```

---

## 2. DETAIL ALGORITMA DAN IMPLEMENTASI PROGRAM TIAP TAHAP

---

### Langkah 1: Ekstraksi Frame dari Rekaman Video CCTV

#### 📘 1. Penjelasan Konsep & Formulasi Matematika

- **Konsep & Algoritma**:
  Kamera CCTV merekam video berdurasi panjang dengan frame rate tinggi. Untuk membuat dataset citra, dilakukan ekstraksi frame-by-frame menggunakan *OpenCV VideoCapture* dengan interval waktu sampling $\Delta t$ konfigurasional (misalnya 1 frame per detik).

- **Formulasi Waktu Index Frame**:
  $$\text{Target Frame Index } k = \text{round}(k \cdot \text{FPS} \cdot \Delta t)$$
  $$\text{frame\_step} = \max(1, \text{round}(\text{FPS} \cdot \Delta t))$$
  - $\text{FPS}$ (*Frames Per Second*): Kecepatan frame bawaan video CCTV.
  - $\Delta t$: Interval sampling waktu yang ditentukan (default: 1.0 detik).
  - $\text{frame\_step}$: Nilai loncatan frame (misal FPS=25 dan $\Delta t=1.0$, maka frame diambil tiap kelipatan 25).

- **Tujuan**: Mengubah berkas video utuh menjadi citra diam (*still frames*) yang siap dianalisis dan dianotasi.

---

#### 💻 2. Lokasi Implementasi & Potongan Kode Program

- **File Program**: [`scripts/dataset/extract_frames.py`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/scripts/dataset/extract_frames.py)
- **Fungsi Utama**: [`extract_frames_from_video()`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/scripts/dataset/extract_frames.py#L18-L82)
- **Letak Baris Kode**:
  - Penentuan `frame_step`: Baris 40 – 45
  - Ekstraksi dan penyimpanan citra: Baris 62 – 76

```python
# [scripts/dataset/extract_frames.py] Baris 40-45 & 62-76
# Hitung interval loncatan frame (frame_step)
if fps_sample is not None and fps_sample > 0:
    frame_step = max(1, int(round(video_fps / fps_sample)))
else:
    frame_step = max(1, int(round(video_fps * interval_sec)))

# Ekstraksi citra frame sesuai loncatan frame_step
while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_step == 0:
        extracted_count += 1
        filename = f"{video_stem}_frame_{extracted_count:06d}.jpg"
        save_path = video_out_dir / filename
        cv2.imwrite(str(save_path), frame)

    frame_count += 1
```

---

### Langkah 2: Seleksi & Eliminasi Redundansi Visual (SSIM)

#### 📘 1. Penjelasan Konsep & Formulasi Matematika

- **Konsep & Algoritma**:
  Karena posisi kamera CCTV bersifat statis, banyak frame yang memiliki tampilan visual identik/duplikat ketika walet tidak bergerak atau tidak ada di dalam jangkauan kamera. Algoritma **Structural Similarity Index Measure (SSIM)** digunakan untuk mengukur kemiripan struktur antar frame berturut-turut.

- **Formulasi Matematika SSIM**:
  $$\text{SSIM}(x, y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$$
  - $\mu_x, \mu_y$: Rata-rata intensitas piksel citra $x$ dan citra $y$.
  - $\sigma_x^2, \sigma_y^2$: Varians intensitas citra $x$ dan citra $y$.
  - $\sigma_{xy}$: Kovariansi antara citra $x$ dan citra $y$.
  - $C_1 = (K_1 L)^2, C_2 = (K_2 L)^2$: Konstanta penstabil komputasi ($K_1=0.01, K_2=0.03, L=255$).

- **Aturan Keputusan Seleksi**:
  $$\text{Keputusan}: \begin{cases} \text{Simpan Frame } y, & \text{jika } \text{SSIM}(x, y) < 0.95 \quad (\text{Terdapat perubahan visual}) \\ \text{Eliminasi / Drop}, & \text{jika } \text{SSIM}(x, y) \ge 0.95 \quad (\text{Redundan / Duplikat}) \end{cases}$$

- **Tujuan**: Meringkas dataset secara cerdas dan mencegah *overfitting* tanpa kehilangan informasi penting.

---

#### 💻 2. Lokasi Implementasi & Potongan Kode Program

- **File Program**: [`scripts/dataset/select_frames.py`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/scripts/dataset/select_frames.py)
- **Fungsi Utama**: [`calculate_ssim()`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/scripts/dataset/select_frames.py#L39-L44) dan [`select_frames()`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/scripts/dataset/select_frames.py#L62-L166)
- **Letak Baris Kode**:
  - Perhitungan SSIM: Baris 39 – 44
  - Logika seleksi kemiripan: Baris 133 – 150

```python
# [scripts/dataset/select_frames.py] Baris 39-44 & 133-150
def calculate_ssim(imgA: np.ndarray, imgB: np.ndarray) -> float:
    if not HAS_SKIMAGE:
        return 0.0
    return ssim(imgA, imgB)

# Evaluasi ambang batas SSIM >= 0.95
if method == "ssim":
    val = calculate_ssim(prev_gray, curr_gray)
    if val >= threshold:  # val >= 0.95: Gambar redundan
        is_different = False

if is_different:
    selected_paths.append(img_path)
    prev_gray = curr_gray
```

---

### Langkah 3: Auto-Labeling Anotasi Semi-Otomatis YOLO

#### 📘 1. Penjelasan Konsep & Formulasi Matematika

- **Konsep & Algoritma**:
  Melakukan deteksi awal semi-otomatis menggunakan model YOLO terlatih untuk menemukan koordinat objek burung walet pada citra-citra yang lolos seleksi SSIM.

- **Formulasi Bounding Box Ter-Normalisasi**:
  Setiap bounding box hasil prediksi $(x_{\text{min}}, y_{\text{min}}, x_{\text{max}}, y_{\text{max}})$ dikonversi ke sistem koordinat relatif $[0.0, 1.0]$:
  $$x_{\text{center}} = \frac{x_{\text{min}} + x_{\text{max}}}{2 \cdot W}, \quad y_{\text{center}} = \frac{y_{\text{min}} + y_{\text{max}}}{2 \cdot H}$$
  $$w = \frac{x_{\text{max}} - x_{\text{min}}}{W}, \quad h = \frac{y_{\text{max}} - y_{\text{min}}}{H}$$
  - $W, H$: Lebar dan tinggi piksel citra asli.

- **Aturan Pemetaan Kelas Tunggal (Single-Class)**:
  Seluruh bounding box dipetakan secara eksklusif ke **`class_id = 0`** (`walet`):
  $$\text{\texttt{0 <x\_center> <y\_center> <w> <h>}}$$

- **Tujuan**: Mempercepat proses anotasi data awal secara masif sebelum verifikasi akhir menggunakan aplikasi LabelImg.

---

#### 💻 2. Lokasi Implementasi & Potongan Kode Program

- **File Program**: [`scripts/dataset/auto_label.py`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/scripts/dataset/auto_label.py)
- **Fungsi Utama**: [`auto_label_images()`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/scripts/dataset/auto_label.py#L30-L125)
- **Letak Baris Kode**:
  - Inferensi Model YOLO: Baris 86 – 88
  - Format Label YOLO & Normalisasi Koordinat: Baris 101 – 116

```python
# [scripts/dataset/auto_label.py] Baris 86-88 & 101-116
results = model.predict(source=str(img_p), conf=conf_threshold, verbose=False)
lines = []

if len(results) > 0 and results[0].boxes is not None:
    for box in results[0].boxes:
        cls_id = 0  # Pemetaan khusus class walet (ID=0)
        x_center, y_center, w, h = box.xywhn[0].tolist()
        lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

with open(lbl_save_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
```

---

### Langkah 4: Pembagian Dataset 70:15:15 Bebas Data Leakage (Video Group Split)

#### 📘 1. Penjelasan Konsep & Formulasi Matematika

- **Konsep & Algoritma**:
  Membagi data ke dalam 3 subset: **70% Training**, **15% Validation**, dan **15% Testing**. Digunakan pendekatan **Group-based Stratified Split** berdasarkan sumber rekaman video asal (`group_id`).

- **Pencegahan Data Leakage (Kebocoran Data)**:
  Pada rekaman CCTV sudut tetap, latar belakang dinding dan sarang walet pada detik ke-1 dan detik ke-2 sangat identik. Jika data dibagi secara acak per-frame (*Random Split*), model akan menghafal latar belakang yang sama di data uji sehingga metrik akurasi menjadi semu (*overoptimistic*). Video Group Split menjamin seluruh frame dari video tertentu hanya berada di salah satu subset secara eksklusif.

---

#### 💻 2. Lokasi Implementasi & Potongan Kode Program

- **File Program**: [`scripts/dataset/split_dataset.py`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/scripts/dataset/split_dataset.py)
- **Fungsi Utama**: [`split_dataset()`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/scripts/dataset/split_dataset.py#L33-L198) dan [`parse_group_id()`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/scripts/dataset/split_dataset.py#L26-L30)
- **Letak Baris Kode**:
  - Ekstraksi Group ID Video: Baris 26 – 30
  - Pengelompokan & Pembagian Rasio 70:15:15: Baris 95 – 155

```python
# [scripts/dataset/split_dataset.py] Baris 95-155
# Grouping pasangan citra dan label per sumber video
groups = defaultdict(list)
for img_p, lbl_p in pairs:
    grp_id = parse_group_id(img_p.name) if use_group_split else img_p.name
    groups[grp_id].append((img_p, lbl_p))

# Distribusi utuh kelompok video ke subset Train, Val, dan Test
for grp_id in group_keys:
    items = groups[grp_id]
    if train_count < target_train:
        train_pairs.extend(items)
        train_count += len(items)
    elif val_count < target_val:
        val_pairs.extend(items)
        val_count += len(items)
    else:
        test_pairs.extend(items)
        test_count += len(items)
```

---

### Langkah 5: Augmentasi Data Latih (*Train Set Augmentation*)

#### 📘 1. Penjelasan Konsep & Formulasi Matematika

- **Konsep & Algoritma**:
  Augmentasi dilakukan **hanya pada Train Set** untuk melipatgandakan variasi data latih tanpa mengubah keaslian Validation dan Test set.

- **Varian Transformasi & Formulasi Bounding Box**:
  1. **Horizontal Flip**:
     $$x_{\text{center, baru}} = 1.0 - x_{\text{center, lama}}$$
  2. **Penyesuaian Kecerahan & Kontras (Brightness & Contrast Shifting)**:
     $$I_{\text{aug}}(x) = \text{clip}(\alpha \cdot I(x) + \beta, 0, 255)$$
     dengan $\alpha \in [0.85, 1.15]$ (faktor kontras) dan $\beta \in [-15, 15]$ (faktor pergeseran kecerahan).
  3. **Gaussian Blur Ringan**: Filter konvolusi Gaussian kernel $3 \times 3$ atau $5 \times 5$.

---

#### 💻 2. Lokasi Implementasi & Potongan Kode Program

- **File Program**: [`scripts/dataset/augment_dataset.py`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/scripts/dataset/augment_dataset.py)
- **Fungsi Utama**: [`flip_h_yolo_labels()`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/scripts/dataset/augment_dataset.py#L26-L33) dan [`augment_image_and_labels()`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/scripts/dataset/augment_dataset.py#L56-L76)
- **Letak Baris Kode**:
  - Transformasi koordinat flip horizontal: Baris 26 – 33
  - Penerapan transformasi citra dan label: Baris 56 – 76

```python
# [scripts/dataset/augment_dataset.py] Baris 26-33 & 65-72
def flip_h_yolo_labels(labels):
    aug_labels = []
    for cls_id, xc, yc, w, h in labels:
        new_xc = round(1.0 - xc, 6)  # Pembalikan koordinat x_center
        aug_labels.append((cls_id, new_xc, yc, w, h))
    return aug_labels

if aug_type == "flip_h":
    aug_img = cv2.flip(img, 1)
    aug_labels = flip_h_yolo_labels(labels)
elif aug_type == "brightness":
    alpha = random.uniform(0.85, 1.15)
    beta = random.randint(-15, 15)
    aug_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
```

---

### Langkah 6: Validasi Konsistensi Format Dataset YOLO

#### 📘 1. Penjelasan Konsep & Formulasi Matematika

- **Konsep & Algoritma**:
  Pemeriksaan otomatis struktur format dataset sebelum proses *training* dimulai. Memastikan tidak ada file label yang hilang, class ID salah, atau bounding box yang koordinatnya melebihi dimensi citra.

- **Kriteria Validasi**:
  1. **Integritas Pasangan**: Setiap citra `.jpg`/`.png` wajib memiliki file label `.txt` pasangan.
  2. **Batas Normalisasi**: $0.0 \le x_c, y_c \le 1.0$ dan $0.0 < w, h \le 1.0$.
  3. **Validasi Single-Class**: Nilai `class_id` wajib sama dengan 0.
  4. **Pemeriksaan Batas Tepi**:
     $$x_{\text{min}} = x_c - \frac{w}{2} \ge 0, \quad x_{\text{max}} = x_c + \frac{w}{2} \le 1$$
     $$y_{\text{min}} = y_c - \frac{h}{2} \ge 0, \quad y_{\text{max}} = y_c + \frac{h}{2} \le 1$$

---

#### 💻 2. Lokasi Implementasi & Potongan Kode Program

- **File Program**: [`scripts/dataset/validate_dataset.py`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/scripts/dataset/validate_dataset.py)
- **Fungsi Utama**: [`validate_yolo_label()`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/scripts/dataset/validate_dataset.py#L18-L83) dan [`validate_dataset_folder()`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/scripts/dataset/validate_dataset.py#L85-L162)
- **Letak Baris Kode**:
  - Pemeriksaan batas koordinat & kelas: Baris 52 – 76
  - Verifikasi kelengkapan berkas: Baris 91 – 150

```python
# [scripts/dataset/validate_dataset.py] Baris 52-76
if class_id != max_class_id:
    errors.append(f"class_id={class_id} tidak valid. Hanya kelas 0 (walet) yang diizinkan.")

if not (0.0 <= x_center <= 1.0) or not (0.0 <= y_center <= 1.0):
    errors.append("Koordinat x_center atau y_center di luar rentang [0, 1].")

if not (0.0 < width <= 1.0) or not (0.0 < height <= 1.0):
    errors.append("Dimensi width atau height di luar rentang (0, 1].")

x_min = x_center - (width / 2.0)
x_max = x_center + (width / 2.0)
if x_min < -1e-4 or x_max > 1.0 + 1e-4:
    errors.append("Bounding box melampaui batas tepi citra.")
```

---

## 3. TABEL RANGKUMAN PEMETAAN FILE KODE PROGRAM

| Langkah Tahapan | File Kode Program | Kelas / Fungsi Utama | Letak Baris Implementasi Formula |
| :--- | :--- | :--- | :--- |
| **Runner Utama Tahap 1** | [`scripts/run_tahap1.py`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/scripts/run_tahap1.py) | `main()` | Eksekusi pipeline otomatis langkah 1-6 |
| **1. Ekstraksi Frame** | [`scripts/dataset/extract_frames.py`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/scripts/dataset/extract_frames.py) | [`extract_frames_from_video()`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/scripts/dataset/extract_frames.py#L18-L82) | [`L40-L45`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/scripts/dataset/extract_frames.py#L40-L45) (Sampling frame_step) |
| **2. Seleksi SSIM** | [`scripts/dataset/select_frames.py`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/scripts/dataset/select_frames.py) | [`calculate_ssim()`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/scripts/dataset/select_frames.py#L39-L44), [`select_frames()`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/scripts/dataset/select_frames.py#L62-L166) | [`L39-L44`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/scripts/dataset/select_frames.py#L39-L44), [`L134-L150`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/scripts/dataset/select_frames.py#L134-L150) (SSIM $\ge 0.95$) |
| **3. Auto-Labeling** | [`scripts/dataset/auto_label.py`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/scripts/dataset/auto_label.py) | [`auto_label_images()`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/scripts/dataset/auto_label.py#L30-L125) | [`L105-L109`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/scripts/dataset/auto_label.py#L105-L109) (BBox xywhn & class 0) |
| **4. Split Dataset** | [`scripts/dataset/split_dataset.py`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/scripts/dataset/split_dataset.py) | [`split_dataset()`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/scripts/dataset/split_dataset.py#L33-L198) | [`L95-L155`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/scripts/dataset/split_dataset.py#L95-L155) (Group Split 70:15:15) |
| **5. Augmentasi Data** | [`scripts/dataset/augment_dataset.py`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/scripts/dataset/augment_dataset.py) | [`flip_h_yolo_labels()`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/scripts/dataset/augment_dataset.py#L26-L33), [`augment_image_and_labels()`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/scripts/dataset/augment_dataset.py#L56-L76) | [`L26-L33`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/scripts/dataset/augment_dataset.py#L26-L33) (Flip $1.0 - x_c$), [`L68-L72`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/scripts/dataset/augment_dataset.py#L68-L72) ($\alpha I + \beta$) |
| **6. Validasi Dataset** | [`scripts/dataset/validate_dataset.py`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/scripts/dataset/validate_dataset.py) | [`validate_yolo_label()`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/scripts/dataset/validate_dataset.py#L18-L83) | [`L52-L76`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/scripts/dataset/validate_dataset.py#L52-L76) (BBox coordinate bounds) |
| **7. Analisis Dataset** | [`scripts/dataset/analyze_dataset.py`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/scripts/dataset/analyze_dataset.py) | [`analyze_dataset()`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/scripts/dataset/analyze_dataset.py#L25-L180) | [`L82-L95`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/scripts/dataset/analyze_dataset.py#L82-L95) (Brightness & contrast stats) |
| **8. Pencerahan Video Full** | [`scripts/utils/enhance_video.py`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/scripts/utils/enhance_video.py) | [`enhance_video()`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/scripts/utils/enhance_video.py#L93-L189) | [`L93-L189`](file:///c:/Users/MURNI/OneDrive/Documents/GitHub/swallow_detection/scripts/utils/enhance_video.py#L93-L189) (Rekonstruksi video .mp4) |

---

## 4. PERINTAH UTAMA EKSEKUSI TAHAP 1

Untuk menjalankan seluruh rangkaian alur Tahap 1 secara otomatis dari 1 file video:

```bash
python scripts/run_tahap1.py --video data/raw/videos/video01.mp4
```

Hasil output pemrosesan akan tersimpan rapi pada direktori:
- `data/raw/frames/video01/` (Frame mentah hasil ekstraksi)
- `data/selected/video01/` (Frame terpilih seleksi SSIM)
- `data/original/images/video01/` & `data/original/labels/video01/` (Citra + Label YOLO)
- `data/splits/` (`train`, `val`, `test`)
- `data/splits/train_augmented/` (Data latih ter-augmentasi)
- `results/enhancement/video01/video/video01_enhanced.mp4` (Video ter-pencerah hasil rekonstruksi)
