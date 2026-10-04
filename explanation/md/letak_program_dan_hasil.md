# Letak Program dan Hasil Proyek

Dokumen ini mencatat posisi file program, konfigurasi, model, dataset, dan hasil yang sudah dibuat pada proyek `swallow_detection`.

## Lokasi Utama Proyek

- Folder proyek: `C:\MyPrograms\swallow_detection`
- Dokumentasi utama: `README.md`
- Kebutuhan library: `requirements.txt`
- Bobot awal YOLO: `yolo12n.pt`, `yolov8n.pt`

## Peta Folder Program dan Hasil

| Folder | Isi utama | Fungsi |
|---|---|---|
| `scripts/` | Runner pipeline dan utilitas dataset | Menjalankan tahap 1 sampai 4 serta pekerjaan bantu seperti ekstraksi frame, split dataset, augmentasi, dan visualisasi metrik |
| `scripts/dataset/` | Script dataset | Ekstraksi frame, seleksi SSIM, split dataset, validasi label, auto label, augmentasi, dan pembuatan master dataset |
| `scripts/utils/` | Script utilitas | Enhancement video, visualisasi kualitas citra, pengecekan upscale, dan perbandingan baseline |
| `enhancement/` | Program Zero-DCE + CBAM | Training, inference enhancement, evaluasi kualitas citra, model, dan loss function |
| `enhancement/models/` | Arsitektur model | `zero_dce_cbam.py` dan `dark_attention.py` |
| `enhancement/losses/` | Loss function | Spatial consistency, exposure control, color constancy, dan illumination smoothness |
| `detection/` | Program deteksi dan counting | Training YOLO, evaluasi deteksi, sinkronisasi label, counting populasi, dan visualisasi video counter |
| `configs/` | File konfigurasi | Konfigurasi dataset original, enhanced, master dataset, dan Zero-DCE + CBAM |
| `data/` | Dataset | Video/frame mentah, frame terpilih, dataset original, split, master dataset, dan dataset hasil enhancement |
| `checkpoints/` | Model terlatih | Checkpoint `.pth` Zero-DCE + CBAM dan `.pt` YOLO |
| `results/` | Hasil analisis dan laporan | CSV, PNG, HTML dashboard, video hasil enhancement, video counter, dan laporan evaluasi |
| `runs/` | Output training/validasi YOLO | Grafik training, confusion matrix, PR curve, hasil validasi, dan weights dari Ultralytics |
| `explanation/` | Dokumentasi pendukung | Markdown, PDF, dan gambar alur/algoritma |

## Letak Deliverable

| No | Deliverable | Letak program | Letak hasil yang sudah dibuat |
|---:|---|---|---|
| 1 | Pipeline dataset | `scripts/dataset/extract_frames.py`, `scripts/dataset/select_frames.py`, `scripts/dataset/analyze_dataset.py`, `scripts/dataset/validate_dataset.py`, `scripts/dataset/split_dataset.py`, `scripts/dataset/build_master_dataset.py`, `scripts/run_tahap1.py`, `scripts/run_tahap2.py` | `data/raw/frames/`, `data/selected/`, `data/original/`, `data/splits/`, `data/master_dataset/`, `data/enhanced/master_dataset/` |
| 2 | Model Zero-DCE baseline | `enhancement/train_zero_dce_cbam.py`, `enhancement/enhance.py`, `enhancement/evaluate_enhancement.py`, `enhancement/models/zero_dce_cbam.py`, `enhancement/losses/` | `checkpoints/zero_dce_cbam/best_zero_dce_cbam.pth`, `checkpoints/zero_dce_cbam/*/best_zero_dce_cbam.pth`, `data/enhanced/zero_dce_cbam/`, `results/enhancement/` |
| 3 | Model Zero-DCE + CBAM | `enhancement/models/zero_dce_cbam.py`, `enhancement/models/dark_attention.py`, `enhancement/train_zero_dce_cbam.py`, `configs/zero_dce_cbam.yaml` | `checkpoints/zero_dce_cbam/best_zero_dce_cbam.pth`, `checkpoints/zero_dce_cbam/best_zero_dce_cbam_debug.pth`, `results/enhancement/training_samples/` |
| 4 | Hasil training video dan pembuatan dataset | `scripts/run_single_video_pipeline.py`, `scripts/run_tahap1.py`, `scripts/run_tahap2.py`, `scripts/utils/enhance_video.py`, `detection/video_counter_visualizer.py` | `results/enhancement/*/video/`, `results/enhancement_upscaled/*/video/`, `results/detection/*/video/`, `data/raw/videos/`, `data/raw/frames/`, `data/selected/` |
| 5 | Dataset hasil enhancement | `enhancement/enhance.py`, `scripts/utils/generate_val_enhanced.py`, `scripts/dataset/build_master_dataset.py` | `data/enhanced/zero_dce_cbam/`, `data/enhanced/zero_dce_cbam_upscaled/`, `data/enhanced/master_dataset/` |
| 6 | Laporan kualitas citra | `enhancement/evaluate_enhancement.py`, `scripts/utils/visualize_quality_metrics.py`, `scripts/utils/check_upscale_quality.py` | `results/enhancement/*/metrics_reports/`, `results/enhancement/quality_metrics/`, `results/enhancement/quality_metrics/dashboard.html`, `results/enhancement_upscaled/*/metrics_reports/` |
| 7 | Figur perbandingan citra | `enhancement/evaluate_enhancement.py`, `scripts/utils/compare_clahe_baseline.py`, `scripts/utils/visualize_quality_metrics.py` | `results/enhancement/*/figures/`, `results/enhancement/*/technical_analysis/`, `results/enhancement_upscaled/*/figures/`, `results/enhancement/quality_metrics/*.png` |
| 8 | Modul augmentasi | `scripts/dataset/augment_dataset.py`, `configs/dataset_scenario_a.yaml`, `configs/dataset_scenario_b.yaml`, `configs/dataset_scenario_c.yaml` | `data/enhanced/zero_dce_cbam/train_augmented/`, `data/enhanced/zero_dce_cbam_upscaled/train_augmented/` |
| 9 | Model deteksi terlatih | `detection/train_yolo.py`, `scripts/run_tahap3.py`, `configs/dataset_master_original.yaml`, `configs/dataset_master_zero_dce_cbam.yaml` | `checkpoints/yolo_original/best.pt`, `checkpoints/yolo_original/master/best.pt`, `checkpoints/yolo_zero_dce_cbam/best.pt`, `checkpoints/yolo_zero_dce_cbam/master/best.pt`, `runs/detect/yolo12_scenario_A_master/weights/`, `runs/detect/yolo12_scenario_B_master/weights/` |
| 10 | Laporan evaluasi deteksi | `detection/evaluate_detection.py`, `scripts/run_tahap3.py` | `results/detection/master/detection_metrics_report.csv`, `runs/detect/yolo12_scenario_A_master/results.csv`, `runs/detect/yolo12_scenario_B_master/results.csv`, `runs/detect/yolo12_scenario_A_master/BoxPR_curve.png`, `runs/detect/yolo12_scenario_B_master/BoxPR_curve.png`, `runs/detect/yolo12_scenario_A_master/confusion_matrix.png`, `runs/detect/yolo12_scenario_B_master/confusion_matrix.png` |
| 11 | Modul counting dan estimasi populasi | `detection/count_population.py`, `detection/video_counter_visualizer.py`, `scripts/run_tahap4.py` | `results/detection/population_estimation_report.csv`, `results/detection/per_frame_counting_detail.csv`, `results/detection/counting_error_metrics.csv`, `results/detection/population_growth_trend.png`, `results/detection/*/population_estimation_report.csv`, `results/detection/*/video/` |
| 12 | Dokumentasi | `README.md`, `explanation/md/`, `explanation/png/`, `explanation/pdf/` | `explanation/md/ALGORITMA.md`, `explanation/md/ALUR_DAN_ALGORITMA_OPSI2_MASTER_DATASET.md`, `explanation/md/alur_dan_algoritma_tahap_1.md`, `explanation/md/alur_dan_algoritma_tahap_2.md`, `explanation/md/alur_dan_algoritma_tahap_3.md`, `explanation/md/alur_dan_algoritma_tahap_4.md`, `explanation/pdf/CARA JALANKAN KODE.pdf`, `explanation/pdf/Dokumentasi_Algoritma_ZeroDCE_CBAM.pdf`, `explanation/png/` |

## Ringkasan Alur Penyimpanan Hasil

1. Video mentah masuk ke `data/raw/videos/`.
2. Frame hasil ekstraksi masuk ke `data/raw/frames/`.
3. Frame terpilih hasil seleksi masuk ke `data/selected/`.
4. Dataset original dan label YOLO ada di `data/original/`, `data/splits/`, dan `data/master_dataset/`.
5. Dataset hasil enhancement ada di `data/enhanced/zero_dce_cbam/`, `data/enhanced/zero_dce_cbam_upscaled/`, dan `data/enhanced/master_dataset/`.
6. Model enhancement tersimpan di `checkpoints/zero_dce_cbam/`.
7. Model deteksi YOLO tersimpan di `checkpoints/yolo_original/` dan `checkpoints/yolo_zero_dce_cbam/`.
8. Hasil training dan validasi YOLO tersimpan di `runs/detect/`.
9. Laporan kualitas citra, figur, dashboard, dan analisis teknis tersimpan di `results/enhancement/` dan `results/enhancement_upscaled/`.
10. Laporan deteksi, counting, estimasi populasi, grafik, dan video counter tersimpan di `results/detection/`.

