# PANDUAN KOLABORASI TIM DENGAN GIT & GITHUB

Panduan ini disusun khusus untuk anggota tim pengembang dan peneliti proyek **Swallow Detection** agar dapat berkolaborasi secara efektif, aman, dan rapi menggunakan **Git** dan **GitHub / GitLab** tanpa saling menimpa kode (*code overwriting*) atau merusak cabang utama (*main branch*).

---

## 📋 DAFTAR ISI

1. [Konsep & Arsitektur Kolaborasi Tim](#1-konsep--arsitektur-kolaborasi-tim)
2. [Konfigurasi Awal Anggota Tim Baru (Onboarding)](#2-konfigurasi-awal-anggota-tim-baru-onboarding)
3. [Alur Kerja Kolaborasi Harian (Daily Team Workflow)](#3-alur-kerja-kolaborasi-harian-daily-team-workflow)
4. [Proses Pull Request (PR) & Code Review](#4-proses-pull-request-pr--code-review)
5. [Penanganan Merge Conflict dalam Tim](#5-penanganan-merge-conflict-dalam-tim)
6. [Sinkronisasi Branch: `git rebase` vs `git merge`](#6-sinkronisasi-branch-git-rebase-vs-git-merge)
7. [Aturan Proteksi Branch (Branch Protection Rules)](#7-aturan-proteksi-branch-branch-protection-rules)
8. [Protokol Khusus Kolaborasi Tim AI / Deep Learning](#8-protokol-khusus-kolaborasi-tim-ai--deep-learning)
9. [Pencegahan Kesalahan & Penyelamatan Kode Tim](#9-pencegahan-kesalahan--penyelamatan-kode-tim)
10. [Rangkuman Perintah Kolaborasi (Team Cheat Sheet)](#10-rangkuman-perintah-kolaborasi-team-cheat-sheet)

---

## 1. KONSEP & ARSITEKTUR KOLABORASI TIM

Dalam pengembangan software dan riset AI berbasis tim, cabang utama (`main`) dianggap sebagai **production-ready / stable code**. Dilarang keras melakukan commit atau push perubahan secara langsung (*direct push*) ke branch `main`.

### Diagram Alur Kolaborasi Tim (Feature Branch Workflow)

```
[ Remote origin/main ]  ---------------------------------------------------> [ Merged PR ] (Stable)
                                  \                                       /
                                   \  git switch -c feature/yolo12-eval  /
                                    v                                   v
[ Local Feature Branch ] ----------------> [ Commit 1 ] --> [ Commit 2 ] --> [ Pull Request & Review ]
```

### Peran & Tanggung Jawab dalam Tim:
1. **Developer / Researcher**:
   - Mengerjakan fitur, eksperimen, atau bug fix di branch khusus (`feature/*`, `fix/*`, `experiment/*`).
   - Membuat Pull Request (PR) saat pekerjaan selesai atau siap di-review.
2. **Maintainer / Lead Reviewer**:
   - Memeriksa kebenaran kode, hasil eksperimen, dan kepatuhan `.gitignore`.
   - Menyetujui (*Approve*) dan menggabungkan (*Merge*) PR ke branch `main`.

---

## 2. KONFIGURASI AWAL ANGGOTA TIM BARU (ONBOARDING)

Setiap anggota tim yang baru bergabung wajib mengatur identitas lokal agar riwayat commit pada GitHub dapat diidentifikasi secara jelas.

```bash
# 1. Atur Nama Lengkap dan Email yang terdaftar di akun GitHub/GitLab
git config --global user.name "Nama Lengkap Anda"
git config --global user.email "email.anda@domain.com"

# 2. Atur editor teks default & nama branch default
git config --global init.defaultBranch main
git config --global core.autocrlf true   # Khusus pengguna Windows

# 3. Kloning repositori tim ke komputer lokal
git clone https://github.com/username/swallow_detection.git
cd swallow_detection

# 4. Verifikasi remote repository
git remote -v
```

---

## 3. ALUR KERJA KOLABORASI HARIAN (DAILY TEAM WORKFLOW)

Ikuti 6 langkah standar ini setiap kali Anda akan mulai bekerja atau membuat fitur baru:

### Langkah 1: Pastikan Branch `main` Lokal Anda Paling Update
Sebelum membuat branch baru, selalu ambil pembaharuan kode terbaru yang dikerjakan oleh rekan tim lainnya:
```bash
git switch main
git pull origin main
```

> 💡 **Apa yang terjadi saat kita melakukan `git pull`?**
> - Perintah `git pull` adalah gabungan dari dua proses: **`git fetch`** (mengunduh data commit terbaru dari semua branch server) + **`git merge`** (menggabungkan kode).
> - **Apakah branch lain ikut ter-update?** **TIDAK**. Kode hanya akan diperbarui pada **branch yang sedang aktif Anda buka saat itu** (misal: `main`). Branch lokal lainnya (seperti `feature/darkness-cbam`) kodenya tidak akan berubah secara otomatis sampai Anda berpindah ke branch tersebut dan melakukan `git pull` atau `git merge main`.

### Langkah 2: Buat Branch Spesifik untuk Fitur / Tugas Anda
Gunakan penamaan branch yang jelas sesuai tugas:
- `feature/nama-fitur` : Penambahan fitur baru (misal: `feature/darkness-cbam`)
- `fix/nama-bug`       : Perbaikan kesalahan kode (misal: `fix/ssim-threshold-bug`)
- `docs/nama-dok`      : Perbaikan dokumentasi (misal: `docs/update-readme`)

```bash
# Membuat sekaligus berpindah ke branch baru
git switch -c feature/zero-dce-enhancement
```

### Langkah 3: Kerjakan Kode & Lakukan Commit Berkala
Buat commit yang kecil (*atomic commits*) dan jelas. Jangan menumpuk perubahan besar dalam 1 commit.

```bash
# Cek status file yang diedit
git status

# Masukkan file yang relevan ke Staging Area
git add enhancement/models/zero_dce_cbam.py configs/zero_dce_cbam.yaml

# Commit dengan format Conventional Commits
git commit -m "feat: tambahkan modul Dark-Aware CBAM pada arsitektur Zero-DCE"
```

### Langkah 4: Sinkronkan Kembali Perubahan dari Tim
Jika saat Anda bekerja ada rekan tim lain yang sudah melakukan *merge* ke `main`, ambil perubahan terbaru mereka ke branch fitur Anda:

```bash
# Ambil info terbaru dari remote server
git fetch origin

# Gabungkan kode terbaru dari origin/main ke branch fitur Anda
git merge origin/main
```

### Langkah 5: Push Branch Fitur ke Remote Repository
Kirimkan branch fitur lokal Anda ke GitHub agar bisa dilihat oleh tim:

```bash
# Push pertama kali (menghubungkan lokal branch dengan remote branch)
git push -u origin feature/zero-dce-enhancement

# Push berikutnya cukup:
git push
```

### Langkah 6: Buka Pull Request (PR) di GitHub
Buka repositori di browser (GitHub), lalu klik tombol **"Compare & pull request"**.

---

## 4. PROSES PULL REQUEST (PR) & CODE REVIEW

Pull Request (PR) adalah tempat tim berdiskusi, mengecek kualitas kode, dan memastikan eksperimen aman sebelum digabungkan ke kode utama.

### A. Template Deskripsi PR yang Baik
Saat membuat PR di GitHub, isi deskripsi dengan format berikut:

```markdown
## 📌 Ringkasan Perubahan
- Menambahkan modul `Darkness Map` dan `Dark-Aware CBAM` pada pipeline Tahap 2.
- Memperbarui file konfigurasi `configs/zero_dce_cbam.yaml`.

## 🧪 Pengujian & Hasil Eksperimen
- [x] Lulus uji coba training 5 epoch (`python enhancement/train_zero_dce_cbam.py --debug`).
- Metrik Entropy meningkat dari 5.21 menjadi 6.84 pada sampel CCTV gelap.

## ⚠️ Perhatian Khusus untuk Reviewer
- Mohon diperiksa apakah alokasi GPU PyTorch sudah efisien pada `dark_attention.py`.
```

### B. Checklist Bagi Code Reviewer (Rekan Tim)
Sebagai reviewer, pastikan mengecek hal-hal berikut sebelum mengklik **Approve**:
1. ❌ **Tidak ada file biner/weights besar** yang ikut ter-commit (seperti `.pt`, `.pth`, `.mp4`).
2. ❌ **Tidak ada hardcoded path lokal** (contoh: `C:\Users\nama_user\...`). Jalur folder harus relatif.
3. ✅ Kode berjalan tanpa error saat diuji di environment terisolasi `.venv`.
4. ✅ Pesan commit jelas dan dokumentasi pendukung telah diperbarui.

### C. Menanggapi Feedback Review & Update PR
Jika reviewer meminta perbaikan kode:
1. Perbaiki kode langsung di komputer lokal Anda pada branch fitur tersebut.
2. Lakukan `git add .` dan `git commit -m "fix: atur ulang alokasi memori GPU sesuai feedback review"`.
3. Lakukan `git push origin feature/zero-dce-enhancement`.
4. **Pull Request di GitHub akan ter-update secara otomatis!** Anda tidak perlu membuat PR baru.

---

## 5. PENANGANAN MERGE CONFLICT DALAM TIM

Merge Conflict terjadi ketika 2 anggota tim mengubah baris kode yang sama pada berkas yang sama dan Git memerlukan keputusan manusia untuk memilih versi mana yang benar.

### Skema Terjadinya Konflik:
- **Rekan Tim A** mengubah `SSIM threshold = 0.95` pada `select_frames.py` dan mem-push ke `main`.
- **Anda** mengubah `SSIM threshold = 0.90` pada `select_frames.py` di branch fitur Anda.
- Saat Anda melakukan `git merge origin/main`, Git akan menandai konflik!

### Langkah Menyelesaikan Konflik Tim:

1. **Jalankan `git status`** untuk melihat daftar file yang bentrok (*Unmerged paths*).
2. **Buka file konflik di Code Editor (VS Code / Antigravity IDE)**. Anda akan melihat penanda konflik:
   ```python
   <<<<<<< HEAD (Kode di Branch Fitur Anda)
   threshold = 0.90
   =======
   threshold = 0.95  # Diubah oleh Rekan Tim A di main
   >>>>>>> origin/main
   ```
3. **Komunikasi dengan Tim**: Tanyakan atau diskusikan dengan rekan tim nilai mana yang akan digunakan untuk eksperimen bersama.
4. **Edit File**: Hapus baris yang tidak digunakan beserta penanda `<<<<<<<`, `=======`, `>>>>>>>`.
5. **Simpan File & Selesaikan Merge**:
   ```bash
   # Tandai file konflik sudah selesai diperbaiki
   git add scripts/dataset/select_frames.py

   # Commit penyelesaian konflik
   git commit -m "fix: selesaikan merge conflict threshold SSIM dengan menetapkan nilai 0.95"

   # Push kembali ke remote
   git push
   ```

### B. Skenario Kasus Nyata: Si A & Si B Sama-sama Commit di `main` (File yang Sama)

#### Kronologi Kejadian:
1. **Si A** dan **Si B** berada di branch `main` lokal masing-masing.
2. Keduanya mengedit file yang sama (`select_frames.py`).
3. **Si A** melakukan commit dan **berhasil melakukan `git push origin main` lebih dulu**.
4. **Si B** melakukan commit di lokalnya, lalu mencoba `git push origin main`. Push **DITOLAK** oleh Git dengan pesan error:
   `! [rejected] main -> main (fetch first / non-fast-forward)`

#### Langkah Penyelesaian oleh Si B (Pihak yang Push-nya Ditolak):

##### Opsi 1: Direct Merge di Main (Solusi Langsung)
```bash
# 1. Si B menarik perubahan commit Si A yang sudah ada di remote main
git pull origin main
# Git akan menampilkan: CONFLICT (content): Merge conflict in scripts/dataset/select_frames.py

# 2. Buka file select_frames.py di VS Code / Editor. Pilih kode yang benar dan hapus penanda <<<<<<<, =======, >>>>>>>.

# 3. Tandai file selesai diperbaiki
git add scripts/dataset/select_frames.py

# 4. Commit hasil penggabungan
git commit -m "fix: selesaikan konflik commit di main antara Si A dan Si B"

# 5. Push kembali ke remote main
git push origin main
```
*Setelah ini, Si A cukup menjalankan `git pull origin main` di komputernya untuk mendapatkan versi gabungan terbaru.*

##### Opsi 2: Pindahkan Commit Si B ke Branch Baru (Best Practice)
Jika ingin kode Si B di-review dulu dan tidak langsung menimpa `main`:
```bash
# 1. Si B memindahkan commit lokalnya ke branch fitur baru
git switch -c feature/perbaikan-si-b

# 2. Kembalikan branch main lokal Si B ke kondisi remote yang bersih
git switch main
git fetch origin
git reset --hard origin/main

# 3. Gabungkan perubahan main terbaru ke branch fitur Si B & selesaikan konflik di sana
git switch feature/perbaikan-si-b
git merge main

# 4. Selesaikan konflik pada select_frames.py, lalu commit & push branch fitur untuk dibuka Pull Request (PR)
git add scripts/dataset/select_frames.py
git commit -m "fix: selesaikan konflik pada branch feature/perbaikan-si-b"
git push -u origin feature/perbaikan-si-b
```

---

## 6. SINKRONISASI BRANCH: `GIT REBASE` VS `GIT MERGE`

Ada 2 metode untuk menyelaraskan branch fitur Anda dengan perubahan terbaru dari `main`:

### Opsi A: `git merge origin/main` (Paling Aman untuk Pemula)
- **Kelebihan**: Membuat commit gabungan (*merge commit*) baru, mempertahankan riwayat historis asli.
- **Kekurangan**: Pohon riwayat Git (*git log graph*) terlihat agak bercabang-cabang.
```bash
git switch feature/yolo-eval
git fetch origin
git merge origin/main
```

### Opsi B: `git rebase origin/main` (Untuk Riwayat Git yang Rapi & Linear)
- **Kelebihan**: Menaruh commit fitur Anda di paling atas perubahan `main` terbaru, menghasilkan riwayat garis lurus yang sangat bersih.
- **Kekurangan**: Mengubah hash commit lokal. **Dilarang melakukan rebase pada branch yang dipakai bersama oleh anggota tim lain!**
```bash
git switch feature/yolo-eval
git fetch origin
git rebase origin/main
```

---

## 7. ATURAN PROTEKSI BRANCH (BRANCH PROTECTION RULES)

Untuk menjamin kualitas repositori tim di GitHub, Admin/Lead Project harus mengaktifkan **Branch Protection Rules** pada branch `main`:

### Pengaturan yang Direkomendasikan di GitHub Settings:
1. **Require a pull request before merging**: Mencegah siapa pun mem-push langsung ke `main` via terminal tanpa lewat PR.
2. **Require approvals (Minimal 1-2 Reviewer)**: PR hanya bisa di-merge jika sudah disetujui oleh anggota tim lain.
3. **Dismiss stale pull request approvals when new commits are pushed**: Menghapus persetujuan otomatis jika ada commit perbaikan baru yang masuk.
4. **Require status checks to pass before merging**: Memastikan pengujian otomatis (CI/CD) berhasil terlebih dahulu.

### ⚠️ Penanganan Error `GH013: Repository rule violations`
Jika Anda mengalami pesan error berikut saat melakukan `git push origin main`:
```text
remote: error: GH013: Repository rule violations found for refs/heads/main.
remote: - Changes must be made through a pull request.
! [remote rejected] main -> main (push declined due to repository rule violations)
```
**Penyebab**: Repositori GitHub Anda diproteksi dengan aturan *Require Pull Request*. Push langsung (*direct push*) ke `main` ditolak oleh GitHub.

**Solusi (Mengikuti Alur PR)**:
```bash
# 1. Pindahkan commit Anda di main ke branch fitur baru
git switch -c feature/update-terbaru

# 2. Kembalikan main lokal ke kondisi remote yang bersih
git switch main
git fetch origin
git reset --hard origin/main

# 3. Push branch fitur Anda ke GitHub
git switch feature/update-terbaru
git push -u origin feature/update-terbaru

# 4. Buka halaman GitHub & buat Pull Request (PR) dari feature/update-terbaru ke main, lalu klik Merge.
```

---

## 8. PROTOKOL KHUSUS KOLABORASI TIM AI / DEEP LEARNING

Proyek riset Deep Learning seperti **Swallow Detection** memiliki tantangan khusus mengenai pengelolaan file besar dan data eksperimen.

### 🚫 1. Dilarang Memasukkan File Berat ke Repositori Git
File biner berukuran besar akan membuat repositori Git menjadi lambat dan membengkak.

**File yang WAJIB berada di `.gitignore`**:
- Model Weights (`*.pt`, `*.pth`, `*.onnx`) $\rightarrow$ *Simpan di Google Drive Tim / Cloud Storage / DVC / HuggingFace.*
- Video & Images Mentah (`data/raw/`, `data/splits/`) $\rightarrow$ *Bagikan link folder cloud ke tim.*
- Virtual Environment (`.venv/`, `venv/`) $\rightarrow$ *Bagikan file `requirements.txt` agar setiap orang menginstal sendiri di komputernya.*

### 🛠️ 2. Berbagi Perubahan Lingkungan Python (`requirements.txt`)
Jika Anda menginstal pustaka baru saat membuat fitur (misal: `pip install albumentations`):
```bash
# Perbarui file requirements.txt
pip freeze > requirements.txt

# Commit & Push agar rekan tim bisa ikut menginstalnya
git add requirements.txt
git commit -m "chore: tambahkan library albumentations ke requirements.txt"
```

Rekan tim yang menarik kode cukup menjalankan:
```bash
git pull origin main
pip install -r requirements.txt
```

---

## 9. PENCEGAHAN KESALAHAN & PENYELAMATAN KODE TIM

### A. Rekan Tim Minta Bantuan Mendadak? Gunakan `git stash`
Jika Anda sedang mengetik kode setengah jadi di branch Anda, dan rekan tim meminta bantuan mengecek bug di `main`:

```bash
# Simpan draft sementara tanpa perlu commit setengah jadi
git stash

# Pindah ke main untuk bantu tes kode rekan tim
git switch main
git pull origin main

# Setelah selesai, kembali ke branch Anda dan kembalikan draft tadi
git switch feature/darkness-cbam
git stash pop
```

### B. Membatalkan Commit yang Sudah Terlanjur Dikanut Tim (`git revert`)
**Jangan gunakan `git reset --hard` pada commit yang sudah di-push ke remote**, karena akan merusak history repositori anggota tim lain!

Gunakan `git revert` yang akan membuat commit baru untuk membatalkan perubahan secara aman:
```bash
# Membatalkan efek dari commit tertentu secara aman
git revert <hash-commit-yang-salah>

# Push commit pembatalan ke remote
git push origin main
```

---

## 10. RANGKUMAN PERINTAH KOLABORASI (TEAM CHEAT SHEET)

| Skenario Kolaborasi | Perintah Git |
| :--- | :--- |
| **Ambil Update Tim di `main`** | `git switch main` lalu `git pull origin main` |
| **Buat Branch Fitur Baru** | `git switch -c feature/nama-fitur` |
| **Cek Perubahan Berkas** | `git status` |
| **Simpan Commit Lokal** | `git add .` lalu `git commit -m "feat: pesan"` |
| **Update Branch Fitur dari `main`** | `git fetch origin` lalu `git merge origin/main` |
| **Kirim Branch ke Remote GitHub** | `git push -u origin feature/nama-fitur` |
| **Simpan Draft Kerja Sementara** | `git stash` (Kembalikan: `git stash pop`) |
| **Lihat Grafik Percabangan Tim** | `git log --oneline --graph --all` |
| **Batalkan Commit di Remote (Aman)** | `git revert <hash-commit>` |

---

> 💡 **Prinsip Utama Kolaborasi Tim**: *"Pull Frequently, Commit Small, Push via Pull Request, Never Break Main!"*
