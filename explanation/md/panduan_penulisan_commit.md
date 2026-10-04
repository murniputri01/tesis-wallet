# PANDUAN PENULISAN COMMIT SEBELUM PUSH

Dokumen ini menjadi acuan singkat untuk menulis pesan commit yang rapi, konsisten, dan mudah dipahami saat melakukan `git commit` dan `git push` pada proyek **Swallow Detection**.

---

## 1. Prinsip Utama

Commit harus menjelaskan **apa yang berubah** dan **mengapa perubahan itu dibuat**.

Gunakan commit kecil dan fokus. Satu commit idealnya hanya berisi satu tujuan, misalnya:

- memperbaiki bug validasi dataset,
- menambah argumen baru pada script,
- memperbarui dokumentasi,
- merapikan konfigurasi training.

Hindari commit besar yang mencampur banyak hal sekaligus, misalnya memperbaiki bug, mengubah README, menambah model, dan menghapus file data dalam satu commit.

---

## 2. Format Pesan Commit

Gunakan format berikut:

```bash
git commit -m "<tipe>: <ringkasan singkat>"
```

Contoh:

```bash
git commit -m "fix: perbaiki alur validasi dataset tahap 2"
```

Jika butuh penjelasan lebih panjang:

```bash
git commit -m "feat: tambahkan build clean dataset pada tahap 2" -m "Dataset kini dibersihkan dari duplikasi frame sebelum master dataset dibangun."
```

---

## 3. Daftar Tipe Commit

| Tipe | Fungsi | Contoh |
| :--- | :--- | :--- |
| `feat` | Menambah fitur baru | `feat: tambahkan opsi skip master dataset` |
| `fix` | Memperbaiki bug | `fix: perbaiki path augmentasi train tahap 2` |
| `docs` | Mengubah dokumentasi | `docs: tambah panduan penulisan commit` |
| `refactor` | Merapikan kode tanpa mengubah perilaku utama | `refactor: sederhanakan pemilihan path dataset` |
| `test` | Menambah atau memperbaiki pengujian | `test: tambah validasi format label yolo` |
| `chore` | Tugas teknis kecil/non-fitur | `chore: rapikan konfigurasi gitignore` |
| `config` | Mengubah file konfigurasi | `config: update dataset master yaml` |
| `perf` | Meningkatkan performa | `perf: percepat pencarian file label dataset` |
| `revert` | Membatalkan commit sebelumnya | `revert: batalkan perubahan split dataset lama` |

---

## 4. Aturan Penulisan Ringkasan

Gunakan bahasa Indonesia yang jelas dan aktif.

Disarankan:

```text
fix: perbaiki validasi master dataset enhanced
feat: tambahkan argumen min objects pada tahap 2
docs: tambah panduan commit sebelum push
```

Hindari:

```text
update
fix bug
commit baru
perubahan
revisi final
final banget
```

Ringkasan commit sebaiknya:

- diawali huruf kecil setelah tipe,
- tidak terlalu panjang,
- tidak diakhiri titik,
- menyebut area yang berubah jika memungkinkan.

---

## 5. Contoh Commit untuk Proyek Ini

### Dataset dan Tahap 2

```bash
git commit -m "feat: tambahkan build clean dataset pada run tahap 2"
git commit -m "fix: validasi master dataset setelah proses overwrite"
git commit -m "fix: sesuaikan path augmentasi train dataset"
```

### YOLO dan Deteksi

```bash
git commit -m "feat: tambah konfigurasi training yolo master dataset"
git commit -m "fix: sinkronkan label enhanced sebelum training skenario b"
git commit -m "perf: kurangi overhead pembacaan label saat evaluasi"
```

### Enhancement

```bash
git commit -m "feat: tambahkan evaluasi brisque pada hasil enhancement"
git commit -m "fix: perbaiki penyimpanan citra zero dce cbam"
git commit -m "refactor: rapikan fungsi loading checkpoint enhancement"
```

### Dokumentasi

```bash
git commit -m "docs: perbarui alur tahap 2 dataset yolo"
git commit -m "docs: tambah panduan training master model"
git commit -m "docs: jelaskan fungsi validasi dataset yolo"
```

---

## 6. Checklist Sebelum Commit

Jalankan pengecekan berikut sebelum membuat commit:

```bash
git status
git diff
```

Pastikan:

- hanya file yang relevan yang masuk commit,
- tidak ada file besar seperti `.pt`, `.pth`, `.mp4`, `.zip`, atau folder `runs/`,
- tidak ada path lokal pribadi seperti `C:\Users\nama_user\...`,
- kode yang diedit sudah diuji minimal dengan command yang sesuai,
- dokumentasi diperbarui jika alur penggunaan berubah.

Jika hanya ingin memasukkan file tertentu:

```bash
git add scripts/run_tahap2.py
git add scripts/dataset/build_master_dataset.py
```

Hindari `git add .` jika belum yakin semua perubahan memang perlu ikut commit.

---

## 7. Checklist Sebelum Push

Sebelum `git push`, cek ulang riwayat commit:

```bash
git log --oneline -5
git status
```

Lalu push branch:

```bash
git push -u origin nama-branch
```

Untuk push berikutnya di branch yang sama:

```bash
git push
```

Jika branch utama (`main`) diproteksi, jangan push langsung ke `main`. Push ke branch kerja lalu buat Pull Request.

---

## 8. Format Nama Branch

Gunakan nama branch yang menggambarkan jenis pekerjaan.

```text
feature/build-clean-dataset
fix/run-tahap2-master-validation
docs/panduan-commit
refactor/dataset-pipeline
experiment/yolo-imgsz-1024
```

Hindari:

```text
coba
branch-baru
revisi
punya-saya
final
```

---

## 9. Template Commit Multi-Baris

Untuk perubahan penting, gunakan pesan commit dengan body:

```bash
git commit -m "fix: perbaiki alur run tahap 2" -m "Dataset kini dibersihkan dengan build_clean_dataset sebelum master dataset dibuat. Validasi dilakukan pada data splits dan master dataset untuk mencegah leakage atau label tidak konsisten."
```

Struktur ideal:

```text
<tipe>: <ringkasan>

<penjelasan singkat kenapa perubahan diperlukan>
<dampak penting atau catatan penggunaan>
```

---

## 10. Contoh Alur Lengkap

```bash
git switch -c fix/run-tahap2-clean-dataset
git status
git diff
git add scripts/run_tahap2.py scripts/dataset/build_master_dataset.py
git commit -m "fix: perbaiki alur clean dataset tahap 2"
git log --oneline -5
git push -u origin fix/run-tahap2-clean-dataset
```

Jika perubahan berupa dokumentasi:

```bash
git switch -c docs/panduan-commit
git add explanation/md/panduan_penulisan_commit.md
git commit -m "docs: tambah panduan penulisan commit"
git push -u origin docs/panduan-commit
```

---

## 11. Pola yang Direkomendasikan

Gunakan pola ini saat bingung menulis pesan:

```text
<tipe>: <aksi> <objek> <konteks>
```

Contoh:

```text
fix: perbaiki validasi dataset tahap 2
feat: tambahkan opsi skip master dataset
docs: jelaskan aturan commit sebelum push
refactor: rapikan builder master dataset
```

Prinsip akhirnya sederhana: ketika orang lain membaca `git log`, mereka harus bisa memahami riwayat perubahan proyek tanpa membuka semua isi file satu per satu.
