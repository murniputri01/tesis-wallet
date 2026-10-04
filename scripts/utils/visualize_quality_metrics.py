#!/usr/bin/env python3
"""
Script: scripts/utils/visualize_quality_metrics.py
Deskripsi: Membuat visualisasi data komprehensif untuk Evaluasi Kualitas Citra:
           1. Entropy (Tingkat Informasi & Detail Tekstur, bits)
           2. BRISQUE-like proxy (skor spasial no-reference, lebih rendah = lebih baik)
           3. Koefisien Variasi Iluminasi (CV = Std / Mean Brightness, keseragaman pencahayaan)
           4. Kontras Objek Weber (kontras walet terhadap latar - kriteria utama proyek)

Metodologi Multi-Video:
  - Otomatis memuat dan menggabungkan seluruh laporan CSV per-video yang ada di results/enhancement/
  - Menghasilkan visualisasi gabungan (overall) DAN perbandingan antar-video (video_comparison_bar.png)
  - Menghasilkan folder visualisasi khusus untuk masing-masing video di:
    results/enhancement/quality_metrics/by_video/<video_stem>/
  - Menyiapkan laporan interaktif dashboard.html dengan fitur dropdown filter per-video.

Output:
results/enhancement/quality_metrics/
    * comparison_bar.png       - perbandingan rata-rata tiap metrik (gabungan)
    * distribution_box.png     - sebaran nilai antar frame uji (gabungan)
    * paired_slope.png         - perubahan per-frame (gabungan)
    * summary_dashboard.png    - dashboard ringkasan + kesimpulan per-video
    * video_comparison_bar.png - perbandingan performa antar file video side-by-side
    * dashboard.html           - laporan interaktif dengan filter video (Chart.js)
    * by_video/<video_stem>/   - folder visualisasi khusus untuk masing-masing video
"""

import argparse
import html as html_lib
import json
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Set style for publication quality plots
plt.style.use('ggplot' if 'ggplot' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 1.0

# Seed tetap agar jitter pada box plot reproducible untuk lampiran laporan
RNG = np.random.default_rng(42)

COLOR_BETTER = '#27ae60'
COLOR_WORSE = '#e74c3c'
COLOR_ORIG = '#2c3e50'
COLOR_CBAM = '#16a085'

# Spesifikasi metrik terpusat.
METRICS = [
    {
        'key': 'entropy',
        'label': 'Entropy',
        'unit': 'bits',
        'higher_better': True,
        'decimals': 4,
        'note': 'Lebih tinggi = informasi/tekstur lebih kaya',
        'desc': 'Shannon entropy distribusi tingkat keabuan citra.',
    },
    {
        'key': 'brisque',
        'label': 'BRISQUE-like (proxy)',
        'unit': 'skor 0-100',
        'higher_better': False,
        'decimals': 2,
        'note': 'Lebih rendah = kualitas spasial lebih alami',
        'desc': ('Skor spasial no-reference berbasis fitur MSCN/AGGD. Proxy heuristik, '
                 'bukan BRISQUE standar dengan model SVR terlatih.'),
    },
    {
        'key': 'cv_illumination',
        'label': 'Koef. Variasi Iluminasi',
        'unit': 'rasio Std/Mean',
        'higher_better': False,
        'decimals': 4,
        'note': 'Lebih rendah = pencahayaan lebih seragam',
        'desc': ('CV = Std / Mean Brightness. Nilainya ikut turun bila kecerahan rata-rata '
                 'naik, jadi harus dibaca bersama kontras, bukan sendirian.'),
    },
    {
        'key': 'object_weber',
        'label': 'Kontras Objek (Weber)',
        'unit': 'rasio',
        'higher_better': True,
        'decimals': 4,
        'note': 'Lebih tinggi = walet lebih kontras dari latar',
        'desc': ('Kontras Weber walet terhadap latar lokal. Kriteria utama kualitas '
                 'enhancement pada proyek ini karena tidak bias terhadap kecerahan global.'),
    },
]

CONTEXT_METRICS = [
    {'key': 'contrast', 'label': 'Kontras Global (Std)', 'higher_better': True},
    {'key': 'mean_brightness', 'label': 'Kecerahan Rata-rata', 'higher_better': None},
]


# --------------------------------------------------------------------------- #
# Helper perhitungan
# --------------------------------------------------------------------------- #
def pct_change(orig: float, cbam: float) -> float:
    """Perubahan relatif (%) dengan guard pembagian nol; NaN bila tidak terdefinisi."""
    if orig is None or cbam is None:
        return float('nan')
    if not np.isfinite(orig) or not np.isfinite(cbam) or abs(orig) < 1e-12:
        return float('nan')
    return (cbam - orig) / orig * 100.0


def improvement_pct(orig: float, cbam: float, higher_better: bool) -> float:
    """Persentase perubahan yang sudah diorientasikan: positif = lebih baik."""
    change = pct_change(orig, cbam)
    if not np.isfinite(change):
        return float('nan')
    return change if higher_better else -change


def is_improved(orig: float, cbam: float, higher_better: bool) -> bool:
    return (cbam > orig) if higher_better else (cbam < orig)


def metric_summary(df: pd.DataFrame, spec: dict) -> dict:
    """Ringkasan satu metrik: rata-rata, perubahan berorientasi, dan jumlah frame membaik."""
    orig_col, cbam_col = f"orig_{spec['key']}", f"cbam_{spec['key']}"
    orig_mean = float(df[orig_col].mean()) if not df.empty else 0.0
    cbam_mean = float(df[cbam_col].mean()) if not df.empty else 0.0
    if spec['higher_better']:
        improved_mask = df[cbam_col] > df[orig_col]
    else:
        improved_mask = df[cbam_col] < df[orig_col]
    return {
        **spec,
        'orig_col': orig_col,
        'cbam_col': cbam_col,
        'orig_mean': orig_mean,
        'cbam_mean': cbam_mean,
        'raw_change': pct_change(orig_mean, cbam_mean),
        'improvement': improvement_pct(orig_mean, cbam_mean, spec['higher_better']),
        'n_improved': int(improved_mask.sum()),
        'n_total': int(len(df)),
        'improved': is_improved(orig_mean, cbam_mean, spec['higher_better']),
    }


def available_metrics(df: pd.DataFrame) -> list:
    """Metrik dari METRICS yang kolomnya benar-benar ada dan tidak seluruhnya NaN."""
    out = []
    for spec in METRICS:
        oc, cc = f"orig_{spec['key']}", f"cbam_{spec['key']}"
        if oc in df.columns and cc in df.columns and not df[oc].isna().all() and not df[cc].isna().all():
            out.append(metric_summary(df, spec))
    return out


def fmt(value: float, decimals: int) -> str:
    if value is None or not np.isfinite(value):
        return 'n/a'
    return f"{value:.{decimals}f}"


def fmt_pct(value: float) -> str:
    return 'n/a' if not np.isfinite(value) else f"{value:+.2f}%"


# --------------------------------------------------------------------------- #
# Pemuatan data
# --------------------------------------------------------------------------- #
def load_metrics_data(results_dir: Path, csv_path: Path = None, merge_all: bool = True) -> pd.DataFrame:
    """
    Memuat data metrik evaluasi.
    Secara default (merge_all=True), menggabungkan seluruh laporan CSV per-video yang
    ditemukan di subfolder results_dir dan memastikan kolom `video_stem` tersedia.
    """
    if csv_path is not None:
        csv_paths = [csv_path]
    else:
        found = sorted(
            results_dir.glob("**/enhancement_metrics_report*.csv"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        csv_paths = found if merge_all else found[:1]
        if found[1:] and not merge_all:
            print(f" [INFO] {len(found) - 1} laporan lain diabaikan (pakai --merge untuk menggabungkan).")

    if not csv_paths:
        print("[ERROR] Tidak ditemukan file enhancement_metrics_report.csv!")
        return pd.DataFrame()

    required = {'filename', 'orig_entropy', 'cbam_entropy'}
    dfs = []
    for csv_p in csv_paths:
        try:
            df_temp = pd.read_csv(csv_p)
        except Exception as e:
            print(f"[WARNING] Gagal membaca {csv_p}: {e}")
            continue

        missing = required - set(df_temp.columns)
        if df_temp.empty or missing:
            detail = ', '.join(sorted(missing)) if missing else 'file kosong'
            print(f"[WARNING] Lewati {csv_p.name}: kolom wajib tidak lengkap ({detail}).")
            continue

        # Determine video_stem if missing in CSV
        if 'video_stem' not in df_temp.columns or df_temp['video_stem'].isna().all():
            v_stem = "default"
            if csv_p.parent.name == "metrics_reports" and csv_p.parent.parent != results_dir:
                v_stem = csv_p.parent.parent.name
            elif csv_p.parent != results_dir:
                v_stem = csv_p.parent.name

            inferred = []
            for fname in df_temp['filename']:
                sf = str(fname)
                if "_frame_" in sf:
                    inferred.append(sf.split("_frame_")[0])
                elif v_stem != "default":
                    inferred.append(v_stem)
                else:
                    inferred.append("default")
            df_temp['video_stem'] = inferred

        try:
            shown = csv_p.resolve().relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            shown = csv_p.as_posix()
        print(f" [INFO] Memuat {shown} ({len(df_temp)} frame, video: {df_temp['video_stem'].iloc[0]})")
        dfs.append(df_temp)

    if not dfs:
        return pd.DataFrame()

    # Concatenate and deduplicate based on (video_stem, filename)
    df_all = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=['video_stem', 'filename'], keep='first')

    core_cols = [f"{p}_{k}" for k in ('entropy', 'brisque', 'cv_illumination')
                 for p in ('orig', 'cbam') if f"{p}_{k}" in df_all.columns]

    df_clean = df_all.dropna(subset=core_cols).copy()
    dropped = len(df_all) - len(df_clean)
    if dropped:
        print(f" [INFO] {dropped} baris dibuang karena metrik inti NaN.")

    return df_clean.sort_values(['video_stem', 'filename']).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Figure 1: Bar chart perbandingan rata-rata (Combined)
# --------------------------------------------------------------------------- #
def plot_grouped_bars(summaries: list, n: int, output_dir: Path, subtitle_suffix: str = ""):
    """Bar chart perbandingan rata-rata seluruh metrik utama."""
    ncols = len(summaries)
    fig, axes = plt.subplots(1, ncols, figsize=(5.3 * ncols, 5.2))
    axes = np.atleast_1d(axes)
    title_text = f'Evaluasi Kualitas Citra: Original vs ZERO-DCE+CBAM (n = {n} frame)'
    if subtitle_suffix:
        title_text += f' - {subtitle_suffix}'
    fig.suptitle(title_text, fontsize=16, fontweight='bold', y=1.03)

    for ax, s in zip(axes, summaries):
        bars = ax.bar(['Original', 'ZERO-DCE+CBAM'], [s['orig_mean'], s['cbam_mean']],
                      color=[COLOR_ORIG, COLOR_CBAM], width=0.55, edgecolor='black', alpha=0.85)

        for bar in bars:
            height = bar.get_height()
            ax.annotate(fmt(height, s['decimals']),
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 5), textcoords="offset points",
                        ha='center', va='bottom', fontsize=11, fontweight='bold')

        change_color = COLOR_BETTER if s['improved'] else COLOR_WORSE
        verdict = 'MEMBAIK' if s['improved'] else 'MEMBURUK'
        ax.set_title(f"{s['label']} ({s['unit']})\n" + "\n".join(textwrap.wrap(s['note'], width=36)),
                     fontsize=11.5, fontweight='bold')
        ax.text(0.5, 0.90,
                f"{verdict} {fmt_pct(s['improvement'])}\n{s['n_improved']}/{s['n_total']} frame membaik",
                transform=ax.transAxes, ha='center', va='center', fontsize=10.5, fontweight='bold',
                color=change_color,
                bbox=dict(boxstyle='round,pad=0.4', facecolor=change_color, alpha=0.12, edgecolor=change_color))

        top = max(s['orig_mean'], s['cbam_mean'], 1e-9)
        ax.set_ylim(0, top * 1.32)
        ax.grid(axis='y', linestyle='--', alpha=0.5)

    fig.text(0.5, -0.04,
             'Persentase ditampilkan sebagai PENINGKATAN: positif = lebih baik menurut arah masing-masing metrik.',
             ha='center', fontsize=9.5, style='italic', color='#555555')

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    save_p = output_dir / "comparison_bar.png"
    plt.savefig(save_p, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" [SAVED] {save_p.as_posix()}")


# --------------------------------------------------------------------------- #
# Figure 1b: Bar chart perbandingan per-video side-by-side
# --------------------------------------------------------------------------- #
def plot_video_comparison_bars(df: pd.DataFrame, summaries: list, output_dir: Path):
    """
    Bar chart perbandingan per-video (Original vs ZERO-DCE+CBAM) untuk setiap metrik utama.
    Memungkinkan melihat performa masing-masing file video secara berdampingan.
    """
    v_stems = sorted(df['video_stem'].unique().tolist())
    if len(v_stems) <= 1:
        return

    ncols = len(summaries)
    fig, axes = plt.subplots(1, ncols, figsize=(6.2 * ncols, 5.8))
    axes = np.atleast_1d(axes)
    fig.suptitle(f'Perbandingan Metrik Kualitas Citra Per-Video (n = {len(df)} frame, {len(v_stems)} video)',
                 fontsize=16, fontweight='bold', y=1.04)

    x_labels = v_stems + ['Gabungan']
    x = np.arange(len(x_labels))
    width = 0.36

    for ax, s in zip(axes, summaries):
        orig_means = []
        cbam_means = []
        for v in v_stems:
            df_v = df[df['video_stem'] == v]
            orig_means.append(float(df_v[s['orig_col']].mean()) if not df_v.empty else 0.0)
            cbam_means.append(float(df_v[s['cbam_col']].mean()) if not df_v.empty else 0.0)

        orig_means.append(s['orig_mean'])
        cbam_means.append(s['cbam_mean'])

        rects1 = ax.bar(x - width / 2, orig_means, width, label='Original', color=COLOR_ORIG, edgecolor='black', alpha=0.85)
        rects2 = ax.bar(x + width / 2, cbam_means, width, label='ZERO-DCE+CBAM', color=COLOR_CBAM, edgecolor='black', alpha=0.85)

        ax.set_title(f"{s['label']} ({s['unit']})\n" + "\n".join(textwrap.wrap(s['note'], width=32)),
                     fontsize=11.5, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=30, ha='right', fontsize=9.5, fontweight='bold')
        ax.legend(fontsize=9, loc='upper left')
        ax.grid(axis='y', linestyle='--', alpha=0.5)

        top = max(max(orig_means), max(cbam_means), 1e-9)
        ax.set_ylim(0, top * 1.35)

        for rect in rects1:
            h = rect.get_height()
            ax.annotate(fmt(h, s['decimals']), xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8, rotation=90)
        for rect in rects2:
            h = rect.get_height()
            ax.annotate(fmt(h, s['decimals']), xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold', rotation=90)

    fig.text(0.5, -0.06,
             'Menampilkan nilai rata-rata Original vs ZERO-DCE+CBAM untuk masing-masing file video secara berdampingan.',
             ha='center', fontsize=10, style='italic', color='#555555')

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    save_p = output_dir / "video_comparison_bar.png"
    plt.savefig(save_p, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" [SAVED MULTI-VIDEO FIG] {save_p.as_posix()}")


# --------------------------------------------------------------------------- #
# Figure 2: Box plot distribusi
# --------------------------------------------------------------------------- #
def plot_box_distributions(df: pd.DataFrame, summaries: list, output_dir: Path):
    """Box plot distribusi nilai metrik per sample frame."""
    ncols = len(summaries)
    fig, axes = plt.subplots(1, ncols, figsize=(5.3 * ncols, 5.2))
    axes = np.atleast_1d(axes)
    fig.suptitle(f'Distribusi Statistik Metrik Kualitas Citra (n = {len(df)} frame uji)',
                 fontsize=15, fontweight='bold')

    for ax, s in zip(axes, summaries):
        orig_data = df[s['orig_col']].values
        cbam_data = df[s['cbam_col']].values

        bp = ax.boxplot([orig_data, cbam_data], tick_labels=['Original', 'ZERO-DCE+CBAM'],
                        patch_artist=True, widths=0.4)
        for patch, color in zip(bp['boxes'], ['#5d6d7e', '#48c9b0']):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)

        for j, data in enumerate([orig_data, cbam_data]):
            x = RNG.normal(j + 1, 0.04, size=len(data))
            ax.scatter(x, data, color='black', alpha=0.6, zorder=3, s=25)

        ax.set_title(f"{s['label']}\n" + "\n".join(textwrap.wrap(s['note'], width=36)),
                     fontsize=12, fontweight='bold')
        ax.set_ylabel(s['unit'], fontsize=11)
        ax.grid(axis='y', linestyle='--', alpha=0.5)

    fig.text(0.5, -0.03,
             'Catatan: sampel frame dari video yang berbeda ditampilkan dalam satu distribusi.',
             ha='center', fontsize=9.5, style='italic', color='#555555')

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    save_p = output_dir / "distribution_box.png"
    plt.savefig(save_p, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" [SAVED] {save_p.as_posix()}")


# --------------------------------------------------------------------------- #
# Figure 3: Slope plot per frame
# --------------------------------------------------------------------------- #
def plot_paired_slopes(df: pd.DataFrame, summaries: list, output_dir: Path):
    """Slope plot untuk melihat transisi tiap frame secara individual."""
    ncols = len(summaries)
    fig, axes = plt.subplots(1, ncols, figsize=(5.3 * ncols, 5.2))
    axes = np.atleast_1d(axes)
    fig.suptitle('Perubahan Per-Frame Citra (Original -> ZERO-DCE+CBAM)', fontsize=15, fontweight='bold')

    for ax, s in zip(axes, summaries):
        for _, row in df.iterrows():
            y_orig, y_cbam = row[s['orig_col']], row[s['cbam_col']]
            color = COLOR_BETTER if is_improved(y_orig, y_cbam, s['higher_better']) else COLOR_WORSE
            ax.plot([0, 1], [y_orig, y_cbam], marker='o', color=color, alpha=0.6, linewidth=1.8)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Original', 'ZERO-DCE+CBAM'], fontsize=11, fontweight='bold')
        ax.set_title(f"{s['label']}\n" + "\n".join(textwrap.wrap(s['note'], width=36)),
                     fontsize=12, fontweight='bold')
        ax.set_ylabel(s['unit'], fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.4)

    fig.text(0.5, -0.03, 'Hijau = frame membaik, merah = frame memburuk (menurut arah masing-masing metrik).',
             ha='center', fontsize=9.5, style='italic', color='#555555')

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    save_p = output_dir / "paired_slope.png"
    plt.savefig(save_p, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" [SAVED] {save_p.as_posix()}")


# --------------------------------------------------------------------------- #
# Figure 4: Dashboard ringkasan
# --------------------------------------------------------------------------- #
def build_conclusion_text(df: pd.DataFrame, summaries: list) -> str:
    """Menyusun teks kesimpulan sepenuhnya dari data."""
    v_stems = sorted(df['video_stem'].unique().tolist())
    lines = [f"KESIMPULAN (n = {len(df)} frame, {len(v_stems)} video):", ""]

    for i, s in enumerate(summaries, start=1):
        arrow = 'naik' if s['cbam_mean'] > s['orig_mean'] else 'turun'
        verdict = 'MEMBAIK' if s['improved'] else 'MEMBURUK'
        lines.append(f"{i}. {s['label']}:")
        lines.append(f"   {fmt(s['orig_mean'], s['decimals'])} -> {fmt(s['cbam_mean'], s['decimals'])}"
                     f"  ({arrow}, {verdict})")
        lines.append(f"   {fmt_pct(s['improvement'])} peningkatan | "
                     f"{s['n_improved']}/{s['n_total']} frame membaik")
        lines.append("")

    if len(v_stems) > 1:
        lines.append("Per-Video Breakdown (Weber Contrast):")
        for v in v_stems:
            df_v = df[df['video_stem'] == v]
            o_wb = df_v['orig_object_weber'].mean() if 'orig_object_weber' in df_v.columns else 0
            c_wb = df_v['cbam_object_weber'].mean() if 'cbam_object_weber' in df_v.columns else 0
            imp = improvement_pct(o_wb, c_wb, True)
            lines.append(f"   * {v} ({len(df_v)}f): Weber {fmt_pct(imp)}")
        lines.append("")

    ctx = []
    for c in CONTEXT_METRICS:
        oc, cc = f"orig_{c['key']}", f"cbam_{c['key']}"
        if oc in df.columns and cc in df.columns and not df[cc].isna().all():
            ctx.append(f"   {c['label']}: {fmt_pct(pct_change(df[oc].mean(), df[cc].mean()))}")
    if ctx:
        lines.append("Konteks (bukan penilaian baik/buruk):")
        lines.extend(ctx)

    return "\n".join(lines).rstrip()


def plot_summary_dashboard(df: pd.DataFrame, summaries: list, output_dir: Path):
    """Dashboard ringkasan multi-panel."""
    n_bar = len(summaries)
    ncols = max(n_bar, 4)
    fig = plt.figure(figsize=(4.3 * ncols, 11))
    gs = gridspec.GridSpec(2, ncols, figure=fig, height_ratios=[1, 1.1], hspace=0.38, wspace=0.38)

    v_stems = sorted(df['video_stem'].unique().tolist())
    subtitle_v = f" ({len(v_stems)} video)" if len(v_stems) > 1 else ""
    fig.suptitle(f'DASHBOARD EVALUASI KUALITAS CITRA (ZERO-DCE + CBAM) - n = {len(df)} frame{subtitle_v}',
                 fontsize=16, fontweight='bold', y=0.97)

    # Baris 1: satu bar chart per metrik utama
    for i, s in enumerate(summaries):
        ax = fig.add_subplot(gs[0, i])
        accent = COLOR_BETTER if s['improved'] else COLOR_WORSE
        bars = ax.bar(['Original', 'DCE+CBAM'], [s['orig_mean'], s['cbam_mean']],
                      color=[COLOR_ORIG, accent], width=0.5, edgecolor='black', alpha=0.9)
        ax.set_title(f"{i + 1}. {s['label']}\n" + "\n".join(textwrap.wrap(s['note'], width=28)),
                     fontsize=10, fontweight='bold')
        ax.set_ylabel(s['unit'], fontsize=10)
        top = max(s['orig_mean'], s['cbam_mean'], 1e-9)
        ax.set_ylim(0, top * 1.28)
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + top * 0.03,
                    fmt(b.get_height(), s['decimals']), ha='center', fontweight='bold', fontsize=9.5)
        ax.grid(axis='y', linestyle='--', alpha=0.4)

    # Panel peningkatan
    split = ncols - 1
    ax4 = fig.add_subplot(gs[1, :split])

    impact = [(s['label'], s['improvement']) for s in summaries]
    for c in CONTEXT_METRICS:
        oc, cc = f"orig_{c['key']}", f"cbam_{c['key']}"
        if c['higher_better'] is not None and oc in df.columns and cc in df.columns and not df[cc].isna().all():
            impact.append((c['label'], improvement_pct(df[oc].mean(), df[cc].mean(), c['higher_better'])))

    keys = [k for k, _ in impact][::-1]
    vals = [v for _, v in impact][::-1]
    bar_colors = [COLOR_BETTER if (np.isfinite(v) and v >= 0) else COLOR_WORSE for v in vals]

    bars4 = ax4.barh(keys, [0.0 if not np.isfinite(v) else v for v in vals],
                     color=bar_colors, height=0.55, edgecolor='black', alpha=0.85)
    ax4.axvline(0, color='black', linewidth=1)
    ax4.set_title('Peningkatan Kualitas (%) - POSITIF = LEBIH BAIK',
                  fontsize=12, fontweight='bold')
    ax4.set_xlabel('Peningkatan (%) relatif terhadap citra original')

    span = max((abs(v) for v in vals if np.isfinite(v)), default=1.0)
    ax4.set_xlim(-span * 1.45, span * 1.45)
    for bar, v in zip(bars4, vals):
        w = bar.get_width()
        offset = span * 0.04 if w >= 0 else -span * 0.04
        ax4.text(w + offset, bar.get_y() + bar.get_height() / 2, fmt_pct(v),
                 va='center', ha='left' if w >= 0 else 'right', fontweight='bold', fontsize=10)

    # Panel kesimpulan tekstual
    ax5 = fig.add_subplot(gs[1, split:])
    ax5.axis('off')
    ax5.text(0.02, 0.98, build_conclusion_text(df, summaries), transform=ax5.transAxes,
             fontsize=8.2, va='top',
             bbox=dict(boxstyle='round,pad=0.7', facecolor='#eef2f7', edgecolor='#bdc3c7', alpha=0.9))

    output_dir.mkdir(parents=True, exist_ok=True)
    save_p = output_dir / "summary_dashboard.png"
    plt.savefig(save_p, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" [SAVED] {save_p.as_posix()}")


# --------------------------------------------------------------------------- #
# Laporan HTML Interaktif (Multi-Video Dropdown)
# --------------------------------------------------------------------------- #
def generate_html_visualization(df: pd.DataFrame, summaries: list, html_path: Path):
    """Laporan HTML interaktif dengan Chart.js dan filter dropdown per-video."""
    v_stems = sorted(df['video_stem'].unique().tolist())
    
    # Prepare serializable records for client-side JS filtering
    records_js = []
    for _, row in df.iterrows():
        rec = {
            'video_stem': str(row['video_stem']),
            'filename': str(row['filename']),
        }
        for s in METRICS:
            oc, cc = f"orig_{s['key']}", f"cbam_{s['key']}"
            if oc in row and pd.notna(row[oc]):
                rec[oc] = round(float(row[oc]), 6)
            if cc in row and pd.notna(row[cc]):
                rec[cc] = round(float(row[cc]), 6)
        records_js.append(rec)

    metrics_meta_js = [
        {
            'key': s['key'],
            'label': s['label'],
            'unit': s['unit'],
            'higher_better': s['higher_better'],
            'decimals': s['decimals'],
            'note': s['note'],
            'desc': s['desc'],
            'orig_col': f"orig_{s['key']}",
            'cbam_col': f"cbam_{s['key']}",
        }
        for s in METRICS
    ]

    # Build dropdown HTML
    video_options = ['<option value="ALL">Semua Video (Gabungan)</option>']
    for v in v_stems:
        cnt = (df['video_stem'] == v).sum()
        video_options.append(f'<option value="{html_lib.escape(v)}">{html_lib.escape(v)} ({cnt} frame)</option>')
    video_select_html = f"""
    <div class="video-filter-bar">
        <label for="videoSelect"><strong>&#127916; Filter Video:</strong></label>
        <select id="videoSelect" onchange="onVideoChange(this.value)">
            {"".join(video_options)}
        </select>
        <span class="video-count-badge" id="videoCountBadge">{len(df)} frame ({len(v_stems)} video)</span>
    </div>
    """

    cards_html = []
    for idx, s in enumerate(METRICS):
        canvas_id = f"chart_{s['key']}"
        cards_html.append(f"""
        <div class="card" id="card_{s['key']}">
            <h3>{idx + 1}. {html_lib.escape(s['label'])}</h3>
            <div class="metric-val" id="val_{s['key']}">- <span class="unit">{html_lib.escape(s['unit'])}</span></div>
            <div class="metric-change" id="change_{s['key']}">Memuat data...</div>
            <p class="desc">{html_lib.escape(s['note'])}. {html_lib.escape(s['desc'])}</p>
            <canvas id="{canvas_id}"></canvas>
        </div>""")

    cards_str = "\n".join(cards_html)

    header_cells = "".join(
        f"<th>Orig {html_lib.escape(s['label'])}</th><th>CBAM {html_lib.escape(s['label'])}</th>"
        for s in METRICS
    )

    html_content = f"""<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Visualisasi Evaluasi Kualitas Citra - Swallow Detection</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-color: #0f172a;
            --card-bg: #1e293b;
            --text-main: #f8fafc;
            --text-sub: #94a3b8;
            --accent-primary: #38bdf8;
            --accent-success: #34d399;
            --accent-danger: #f87171;
            --border-color: #334155;
        }}
        body {{
            font-family: 'Plus Jakarta Sans', sans-serif;
            background-color: var(--bg-color);
            color: var(--text-main);
            margin: 0;
            padding: 30px;
        }}
        .header {{ text-align: center; margin-bottom: 24px; }}
        .header h1 {{ font-size: 2.2rem; margin-bottom: 8px; }}
        .header p {{ color: var(--text-sub); font-size: 1.05rem; margin-top: 0; }}
        .video-filter-bar {{
            max-width: 900px; margin: 0 auto 24px auto; padding: 14px 20px;
            background: var(--card-bg); border: 1px solid var(--border-color);
            border-radius: 12px; display: flex; align-items: center; justify-content: space-between;
            gap: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }}
        .video-filter-bar label {{ font-size: 1rem; color: var(--accent-primary); }}
        .video-filter-bar select {{
            background: #0f172a; color: #f8fafc; border: 1px solid var(--border-color);
            padding: 8px 16px; border-radius: 8px; font-size: 0.95rem; font-weight: 600;
            outline: none; cursor: pointer; min-width: 250px;
        }}
        .video-filter-bar select:focus {{ border-color: var(--accent-primary); }}
        .video-count-badge {{
            background: rgba(56, 189, 248, 0.15); color: var(--accent-primary);
            padding: 6px 14px; border-radius: 20px; font-size: 0.88rem; font-weight: 600;
        }}
        .verdict {{
            max-width: 900px; margin: 0 auto 32px auto; padding: 16px 20px;
            background: var(--card-bg); border: 1px solid var(--border-color);
            border-left: 4px solid var(--accent-primary); border-radius: 10px;
            color: var(--text-sub); font-size: 0.95rem; line-height: 1.6;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
            gap: 24px; margin-bottom: 40px;
        }}
        .card {{
            background: var(--card-bg); border: 1px solid var(--border-color);
            border-radius: 16px; padding: 24px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        }}
        .card h3 {{
            margin-top: 0; font-size: 1.15rem; color: var(--accent-primary);
            border-bottom: 1px solid var(--border-color); padding-bottom: 12px;
        }}
        .metric-val {{ font-size: 1.8rem; font-weight: 700; margin: 15px 0 5px 0; }}
        .metric-val .unit {{ font-size: 0.95rem; color: var(--text-sub); font-weight: 400; }}
        .metric-change {{ font-size: 0.92rem; font-weight: 600; }}
        .desc {{ font-size: 0.85rem; color: var(--text-sub); margin-top: 10px; line-height: 1.5; }}
        .positive {{ color: var(--accent-success); }}
        .negative {{ color: var(--accent-danger); }}
        .cell-better {{ color: var(--accent-success); font-weight: 600; }}
        .cell-worse {{ color: var(--accent-danger); font-weight: 600; }}
        canvas {{ max-height: 240px; margin-top: 12px; }}
        table {{
            width: 100%; border-collapse: collapse; margin-top: 20px;
            background: var(--card-bg); border-radius: 12px; overflow: hidden;
            font-size: 0.88rem;
        }}
        th, td {{
            padding: 10px 14px; text-align: left;
            border-bottom: 1px solid var(--border-color); white-space: nowrap;
        }}
        th {{ background-color: #0f172a; color: var(--accent-primary); font-weight: 600; }}
        tr:hover {{ background-color: rgba(255, 255, 255, 0.03); }}
        .footnote {{ color: var(--text-sub); font-size: 0.85rem; line-height: 1.6; max-width: 900px; margin: 0 auto; }}
        .v-badge {{
            display: inline-block; background: #334155; color: #f8fafc;
            font-size: 0.75rem; padding: 2px 6px; border-radius: 4px; margin-right: 6px;
        }}
    </style>
</head>
<body>

    <div class="header">
        <h1>&#128202; Dashboard Visualisasi Kualitas Citra</h1>
        <p>Evaluasi No-Reference: Entropy, BRISQUE-like, Koefisien Variasi Iluminasi, dan Kontras Objek Weber</p>
    </div>

    {video_select_html}

    <div class="verdict" id="verdictBox">
        Memuat status evaluasi...
    </div>

    <div class="grid">
{cards_str}
    </div>

    <div class="card" style="margin-bottom: 40px;">
        <h3>&#128203; Detail Metrik Per-Frame Citra</h3>
        <div style="overflow-x: auto;">
            <table>
                <thead>
                    <tr><th>Video</th><th>Filename</th>{header_cells}</tr>
                </thead>
                <tbody id="tableBody">
                </tbody>
            </table>
        </div>
        <p class="desc">Kolom CBAM berwarna hijau bila frame tersebut membaik, merah bila memburuk.</p>
    </div>

    <div class="footnote">
        <p><strong>Catatan metodologis.</strong> Skor BRISQUE pada proyek ini merupakan proxy heuristik
        berbasis fitur MSCN/AGGD, bukan BRISQUE standar dengan model SVR terlatih pada dataset LIVE.
        Koefisien Variasi Iluminasi (Std/Mean) ikut mengecil ketika kecerahan rata-rata naik,
        sehingga penurunannya tidak otomatis berarti kualitas deteksi meningkat dan harus dibaca bersama
        Kontras Objek Weber.</p>
    </div>

    <script>
        const ALL_DATA = {json.dumps(records_js)};
        const METRICS_META = {json.dumps(metrics_meta_js)};
        let chartInstances = {{}};

        const baseOptions = {{
            responsive: true,
            maintainAspectRatio: false,
            plugins: {{
                legend: {{ labels: {{ color: '#f8fafc', boxWidth: 12 }} }},
                tooltip: {{ mode: 'index', intersect: false }}
            }},
            scales: {{
                x: {{ ticks: {{ color: '#94a3b8', maxRotation: 60, minRotation: 45, font: {{ size: 9 }} }},
                      grid: {{ color: '#334155' }} }},
                y: {{ ticks: {{ color: '#94a3b8' }}, grid: {{ color: '#334155' }} }}
            }}
        }};

        function fmtNum(val, decimals) {{
            if (val === null || val === undefined || isNaN(val)) return 'n/a';
            return val.toFixed(decimals);
        }}

        function fmtPctJS(val) {{
            if (val === null || val === undefined || isNaN(val)) return 'n/a';
            const sign = val >= 0 ? '+' : '';
            return sign + val.toFixed(2) + '%';
        }}

        function onVideoChange(selectedVideo) {{
            const filteredData = selectedVideo === 'ALL'
                ? ALL_DATA
                : ALL_DATA.filter(d => d.video_stem === selectedVideo);

            const nTotal = filteredData.length;
            document.getElementById('videoCountBadge').innerText = nTotal + ' frame' + (selectedVideo === 'ALL' ? ' (gabungan)' : '');

            let improvedCount = 0;
            let metricVerdicts = [];

            METRICS_META.forEach(m => {{
                const origVals = filteredData.map(d => d[m.orig_col]).filter(v => v !== undefined && !isNaN(v));
                const cbamVals = filteredData.map(d => d[m.cbam_col]).filter(v => v !== undefined && !isNaN(v));

                const origMean = origVals.length ? (origVals.reduce((a, b) => a + b, 0) / origVals.length) : 0;
                const cbamMean = cbamVals.length ? (cbamVals.reduce((a, b) => a + b, 0) / cbamVals.length) : 0;

                const rawChange = origMean !== 0 ? ((cbamMean - origMean) / origMean * 100.0) : 0;
                const improvement = m.higher_better ? rawChange : -rawChange;
                const isBetter = m.higher_better ? (cbamMean > origMean) : (cbamMean < origMean);

                let nImproved = 0;
                filteredData.forEach(d => {{
                    const o = d[m.orig_col];
                    const c = d[m.cbam_col];
                    if (o !== undefined && c !== undefined) {{
                        if (m.higher_better ? (c > o) : (c < o)) nImproved++;
                    }}
                }});

                if (isBetter) improvedCount++;
                metricVerdicts.push((isBetter ? 'MEMBAIK: ' : 'MEMBURUK: ') + m.label);

                // Update Card
                const valElem = document.getElementById('val_' + m.key);
                valElem.innerHTML = fmtNum(cbamMean, m.decimals) + ' <span class="unit">' + m.unit + '</span>';

                const changeElem = document.getElementById('change_' + m.key);
                const verdictTxt = isBetter ? 'Membaik' : 'Memburuk';
                const clsName = isBetter ? 'positive' : 'negative';
                changeElem.className = 'metric-change ' + clsName;
                changeElem.innerHTML = 'Original: ' + fmtNum(origMean, m.decimals) + ' &middot; ' +
                    verdictTxt + ' ' + fmtPctJS(improvement) + ' &middot; ' +
                    nImproved + '/' + nTotal + ' frame membaik';

                // Update Chart
                const labels = filteredData.map(d => (selectedVideo === 'ALL' ? d.video_stem + '/' : '') + d.filename);
                const chartId = 'chart_' + m.key;

                if (chartInstances[chartId]) {{
                    chartInstances[chartId].destroy();
                }}

                chartInstances[chartId] = new Chart(document.getElementById(chartId), {{
                    type: 'line',
                    data: {{
                        labels: labels,
                        datasets: [
                            {{ label: 'Original', data: origVals, borderColor: '#94a3b8', backgroundColor: '#94a3b8', tension: 0.2, pointRadius: 2.5 }},
                            {{ label: 'ZERO-DCE+CBAM', data: cbamVals, borderColor: '#38bdf8', backgroundColor: '#38bdf8', tension: 0.2, pointRadius: 2.5 }}
                        ]
                    }},
                    options: Object.assign({{}}, baseOptions, {{
                        plugins: Object.assign({{}}, baseOptions.plugins, {{
                            title: {{ display: true, text: m.label + ' (' + m.unit + ')', color: '#94a3b8', font: {{ size: 11 }} }}
                        }})
                    }})
                }});
            }});

            // Verdict Box Update
            const verdictBox = document.getElementById('verdictBox');
            verdictBox.innerHTML = '<strong>' + improvedCount + ' dari ' + METRICS_META.length + ' metrik membaik.</strong> ' +
                'Nilai persentase memakai konvensi: <em>positif = lebih baik</em>. Jumlah sampel: <strong>' + nTotal + ' frame</strong>.';

            // Table Update
            const tbody = document.getElementById('tableBody');
            let rowsHtml = '';
            filteredData.forEach(d => {{
                let cellsHtml = '';
                METRICS_META.forEach(m => {{
                    const o = d[m.orig_col];
                    const c = d[m.cbam_col];
                    const isB = m.higher_better ? (c > o) : (c < o);
                    const cls = isB ? 'cell-better' : 'cell-worse';
                    cellsHtml += '<td>' + fmtNum(o, m.decimals) + '</td><td class="' + cls + '">' + fmtNum(c, m.decimals) + '</td>';
                }});
                rowsHtml += '<tr><td><span class="v-badge">' + d.video_stem + '</span></td><td><code>' + d.filename + '</code></td>' + cellsHtml + '</tr>';
            }});
            tbody.innerHTML = rowsHtml;
        }}

        window.onload = function() {{
            onVideoChange('ALL');
        }};
    </script>
</body>
</html>
"""
    html_path.parent.mkdir(parents=True, exist_ok=True)
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f" [SAVED HTML DASHBOARD] {html_path.as_posix()}")


# --------------------------------------------------------------------------- #
# Main Execution
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Visualisasi metrik evaluasi kualitas citra.")
    parser.add_argument('--csv', type=Path, default=None,
                        help="Path eksplisit ke enhancement_metrics_report.csv (default: gabungkan seluruh laporan).")
    parser.add_argument('--merge', action='store_true', default=True,
                        help="Gabungkan seluruh laporan per-video (diaktifkan secara default).")
    parser.add_argument('--no-merge', action='store_true',
                        help="Hanya muat 1 laporan terbaru (bukan gabungan).")
    args = parser.parse_args()

    results_dir = PROJECT_ROOT / "results" / "enhancement"
    output_dir = results_dir / "quality_metrics"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(" VISUALISASI METRIK KUALITAS CITRA (MULTI-VIDEO EVALUATION)")
    print("=" * 70)

    df = load_metrics_data(results_dir, args.csv, merge_all=not args.no_merge)
    if df.empty:
        print("[ERROR] Data metrik tidak ditemukan. Jalankan evaluate_enhancement.py terlebih dahulu.")
        return 1

    summaries = available_metrics(df)
    if not summaries:
        print("[ERROR] Tidak ada kolom metrik yang bisa divisualisasikan.")
        return 1

    v_stems = sorted(df['video_stem'].unique().tolist())
    print(f" [INFO] {len(df)} frame dari {len(v_stems)} video ({', '.join(v_stems)}) siap divisualisasikan.\n")
    for s in summaries:
        verdict = 'MEMBAIK ' if s['improved'] else 'MEMBURUK'
        print(f"   {s['label']:26s} {fmt(s['orig_mean'], s['decimals']):>10s} -> "
              f"{fmt(s['cbam_mean'], s['decimals']):>10s}  "
              f"{fmt_pct(s['improvement']):>9s}  {verdict}  ({s['n_improved']}/{s['n_total']} frame)")
    print()

    # 1. Output Visualisasi Gabungan (Overall)
    plot_grouped_bars(summaries, len(df), output_dir)
    plot_video_comparison_bars(df, summaries, output_dir)
    plot_box_distributions(df, summaries, output_dir)
    plot_paired_slopes(df, summaries, output_dir)
    plot_summary_dashboard(df, summaries, output_dir)
    generate_html_visualization(df, summaries, output_dir / "dashboard.html")

    # 2. Output Visualisasi Per-Video Individual (by_video/<video_stem>/)
    if len(v_stems) > 0:
        by_video_dir = output_dir / "by_video"
        for v in v_stems:
            df_v = df[df['video_stem'] == v].copy()
            if df_v.empty:
                continue
            v_dir = by_video_dir / v
            v_summaries = available_metrics(df_v)
            plot_grouped_bars(v_summaries, len(df_v), v_dir, subtitle_suffix=f"Video: {v}")
            plot_box_distributions(df_v, v_summaries, v_dir)
            plot_paired_slopes(df_v, v_summaries, v_dir)
            plot_summary_dashboard(df_v, v_summaries, v_dir)
            generate_html_visualization(df_v, v_summaries, v_dir / "dashboard.html")

    print("\n [SUKSES] Seluruh visualisasi data multi-video berhasil dibuat!")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
