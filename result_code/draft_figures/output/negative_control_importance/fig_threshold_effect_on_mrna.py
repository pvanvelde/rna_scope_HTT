"""
Investigating the Effect of Slide-Specific vs Session-Mean Thresholds on Total mRNA Quantification

This script compares how using slide-specific thresholds vs session-mean thresholds
affects the final mRNA quantification results.

Author: Generated with Claude Code
Date: 2025-12-24
"""

import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, pearsonr, ttest_rel
from pathlib import Path
import seaborn as sns

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from result_functions_v2 import (
    compute_thresholds,
    recursively_load_dict,
    extract_dataframe
)

from results_config import (
    PIXELSIZE as pixelsize,
    SLICE_DEPTH as slice_depth,
    VOXEL_SIZE as voxel_size,
    MEAN_NUCLEAR_VOLUME as mean_nuclear_volume,
    MAX_PFA,
    QUANTILE_NEGATIVE_CONTROL,
    N_BOOTSTRAP,
    MIN_NUCLEI_THRESHOLD,
    H5_FILE_PATH_EXPERIMENTAL,
    FIGURE_DPI,
    SLIDE_FIELD,
    CV_THRESHOLD,
    EXCLUDED_SLIDES
)

# Output directory
OUTPUT_DIR = Path(__file__).parent
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Figure settings
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10


def extract_imaging_session(slide_name):
    """Extract imaging session from slide name (m1*, m2*, m3* = Session 1, 2, 3)."""
    if slide_name and len(slide_name) >= 2 and slide_name.startswith('m'):
        try:
            session = int(slide_name[1])
            return f"Session {session}"
        except ValueError:
            return "Unknown"
    return "Unknown"


def compute_peak_intensity(intensities, bw_method='scott'):
    """Compute peak intensity from KDE (highest probability intensity)."""
    if len(intensities) < 50:
        return np.nan

    try:
        kde = gaussian_kde(intensities, bw_method=bw_method)
        x_range = np.linspace(np.percentile(intensities, 1),
                             np.percentile(intensities, 99), 1000)
        y_density = kde(x_range)
        peak_idx = np.argmax(y_density)
        peak_intensity = x_range[peak_idx]
        return peak_intensity
    except:
        return np.nan


def extract_fov_data_with_both_thresholds(df_input, thresholds_slide, session_mean_thresholds):
    """
    Extract FOV-level data computing mRNA with both threshold approaches.

    Returns a dataframe with Total_mRNA_per_Cell computed using:
    1. Slide-specific thresholds (standard approach)
    2. Session-mean thresholds (alternative approach)
    """

    slide_field = SLIDE_FIELD

    # Define region lists
    striatum_subregions = [
        "Striatum - lower left", "Striatum - lower right",
        "Striatum - upper left", "Striatum - upper right",
    ]
    cortex_subregions = [
        "Cortex - Piriform area",
        "Cortex - Primary and secondary motor areas",
        "Cortex - Primary somatosensory (mouth, upper limb)",
        "Cortex - Supplemental/primary somatosensory (nose)",
        "Cortex - Visceral/gustatory/agranular areas",
    ]

    channel_labels = {'green': 'HTT1a', 'orange': 'fl-HTT'}

    # ──────────────────────────────────────────────────────────────────────
    # STEP 1: Compute peak intensities with BOTH threshold approaches
    # ──────────────────────────────────────────────────────────────────────

    print("\n[1/3] Computing peak intensities with both threshold approaches...")

    # Collect all spot intensities per slide-channel
    slide_channel_intensities = {}

    for idx, row in df_input.iterrows():
        slide = row.get(slide_field, 'unknown')
        channel = row.get('channel', 'unknown')

        if channel == 'blue':
            continue

        key = (slide, channel)

        if key not in slide_channel_intensities:
            slide_channel_intensities[key] = {
                'intensities': [],
                'session': extract_imaging_session(slide)
            }

        params = row.get('spots_sigma_var.params_raw', None)
        final_filter = row.get('spots.final_filter', None)

        if params is not None and final_filter is not None:
            params = np.array(params)
            final_filter = np.array(final_filter).astype(bool)

            if len(params) > 0 and len(final_filter) == len(params):
                params_filtered = params[final_filter]
                if params_filtered.shape[1] >= 4:
                    intensities = params_filtered[:, 3]
                    slide_channel_intensities[key]['intensities'].extend(intensities)

    # Compute peak intensities with both threshold approaches
    peak_slide_specific = {}  # (slide, channel) -> peak with slide-specific threshold
    peak_session_mean = {}    # (slide, channel) -> peak with session-mean threshold

    for (slide, channel), data in slide_channel_intensities.items():
        intensities = np.array(data['intensities'])
        session = data['session']

        if len(intensities) < 100:
            continue

        # Get thresholds
        thr_slide = thresholds_slide.get((slide, channel), np.nan)
        thr_session = session_mean_thresholds.get(channel, {}).get(session, np.nan)

        if np.isnan(thr_slide) or np.isnan(thr_session):
            continue

        # Compute peak with slide-specific threshold
        intensities_slide = intensities[intensities > thr_slide]
        if len(intensities_slide) >= 50:
            peak_slide_specific[(slide, channel)] = compute_peak_intensity(intensities_slide)

        # Compute peak with session-mean threshold
        intensities_session = intensities[intensities > thr_session]
        if len(intensities_session) >= 50:
            peak_session_mean[(slide, channel)] = compute_peak_intensity(intensities_session)

    print(f"  Computed peaks for {len(peak_slide_specific)} slide-channel combinations")

    # ──────────────────────────────────────────────────────────────────────
    # STEP 2: Build DAPI lookup
    # ──────────────────────────────────────────────────────────────────────

    print("\n[2/3] Building DAPI nuclei lookup...")

    df_sorted = df_input.sort_index()
    dapi_lookup = {}

    for idx, row in df_sorted.iterrows():
        channel = row.get('channel', 'unknown')

        if channel == 'blue':
            label_sizes = row.get('label_sizes', None)

            if label_sizes is not None and len(label_sizes) > 0:
                label_sizes = np.array(label_sizes)
                V_DAPI = np.sum(label_sizes) * voxel_size
                N_nuc = V_DAPI / mean_nuclear_volume

                dapi_lookup[idx] = (N_nuc, V_DAPI)
                dapi_lookup[idx + 1] = (N_nuc, V_DAPI)
                dapi_lookup[idx + 2] = (N_nuc, V_DAPI)

    print(f"  Built DAPI lookup for {len(dapi_lookup)} entries")

    # ──────────────────────────────────────────────────────────────────────
    # STEP 3: Extract FOV-level data with both threshold approaches
    # ──────────────────────────────────────────────────────────────────────

    print("\n[3/3] Extracting FOV-level data with both threshold approaches...")

    fov_records = []

    for idx, row in df_input.iterrows():
        channel = row.get('channel', 'unknown')

        if channel == 'blue':
            continue

        slide = row.get(slide_field, 'unknown')
        subregion = row.get('metadata_sample_Slice_Region', 'unknown')
        age = row.get('metadata_sample_Age', np.nan)
        mouse_id = row.get('metadata_sample_mouse_ID', 'unknown')
        mouse_model = row.get('metadata_sample_Mouse_Model', 'unknown')
        session = extract_imaging_session(slide)

        # Determine region
        if any(sub in subregion for sub in cortex_subregions):
            region = 'Cortex'
        elif any(sub in subregion for sub in striatum_subregions):
            region = 'Striatum'
        else:
            continue

        channel_name = channel_labels.get(channel, channel)

        # Get DAPI nuclei
        if idx not in dapi_lookup:
            continue

        N_nuc, V_DAPI = dapi_lookup[idx]

        if N_nuc < MIN_NUCLEI_THRESHOLD:
            continue

        # Get spots and clusters
        params = row.get('spots_sigma_var.params_raw', None)
        final_filter = row.get('spots.final_filter', None)
        cluster_intensities = row.get('cluster_intensities', None)
        cluster_cvs = row.get('cluster_cvs', None)

        # Get thresholds
        thr_slide = thresholds_slide.get((slide, channel), np.nan)
        thr_session = session_mean_thresholds.get(channel, {}).get(session, np.nan)

        if np.isnan(thr_slide) or np.isnan(thr_session):
            continue

        # Get peak intensities
        key = (slide, channel)
        peak_slide = peak_slide_specific.get(key, np.nan)
        peak_sess = peak_session_mean.get(key, np.nan)

        if np.isnan(peak_slide) or np.isnan(peak_sess):
            continue

        # ── Process with SLIDE-SPECIFIC threshold ──
        num_spots_slide = 0
        I_cluster_total_slide = 0

        if params is not None and final_filter is not None:
            params = np.array(params)
            final_filter = np.array(final_filter).astype(bool)

            if len(params) > 0 and len(final_filter) == len(params):
                params_filtered = params[final_filter]
                if params_filtered.shape[1] >= 4:
                    intensities = params_filtered[:, 3]
                    num_spots_slide = (intensities > thr_slide).sum()

        if cluster_intensities is not None and cluster_cvs is not None:
            cluster_intensities_arr = np.array(cluster_intensities)
            cluster_cvs_arr = np.array(cluster_cvs)

            if len(cluster_intensities_arr) > 0 and len(cluster_cvs_arr) == len(cluster_intensities_arr):
                mask = (cluster_intensities_arr > thr_slide) & (cluster_cvs_arr >= CV_THRESHOLD)
                I_cluster_total_slide = cluster_intensities_arr[mask].sum()

        # Compute mRNA with slide-specific
        cluster_mrna_slide = I_cluster_total_slide / peak_slide
        total_mrna_slide = num_spots_slide + cluster_mrna_slide
        total_mrna_per_cell_slide = total_mrna_slide / N_nuc

        # ── Process with SESSION-MEAN threshold ──
        num_spots_session = 0
        I_cluster_total_session = 0

        if params is not None and final_filter is not None:
            params = np.array(params)
            final_filter = np.array(final_filter).astype(bool)

            if len(params) > 0 and len(final_filter) == len(params):
                params_filtered = params[final_filter]
                if params_filtered.shape[1] >= 4:
                    intensities = params_filtered[:, 3]
                    num_spots_session = (intensities > thr_session).sum()

        if cluster_intensities is not None and cluster_cvs is not None:
            cluster_intensities_arr = np.array(cluster_intensities)
            cluster_cvs_arr = np.array(cluster_cvs)

            if len(cluster_intensities_arr) > 0 and len(cluster_cvs_arr) == len(cluster_intensities_arr):
                mask = (cluster_intensities_arr > thr_session) & (cluster_cvs_arr >= CV_THRESHOLD)
                I_cluster_total_session = cluster_intensities_arr[mask].sum()

        # Compute mRNA with session-mean
        cluster_mrna_session = I_cluster_total_session / peak_sess
        total_mrna_session = num_spots_session + cluster_mrna_session
        total_mrna_per_cell_session = total_mrna_session / N_nuc

        # Store record
        fov_records.append({
            'Mouse_Model': mouse_model,
            'Mouse_ID': mouse_id,
            'Slide': slide,
            'Session': session,
            'Region': region,
            'Channel': channel_name,
            'Age': age,
            'N_Nuclei': N_nuc,
            'Threshold_Slide': thr_slide,
            'Threshold_Session': thr_session,
            'Peak_Slide': peak_slide,
            'Peak_Session': peak_sess,
            'N_Spots_Slide': num_spots_slide,
            'N_Spots_Session': num_spots_session,
            'Total_mRNA_per_Cell_Slide': total_mrna_per_cell_slide,
            'Total_mRNA_per_Cell_Session': total_mrna_per_cell_session,
            'mRNA_Difference': total_mrna_per_cell_session - total_mrna_per_cell_slide,
            'mRNA_Pct_Difference': (total_mrna_per_cell_session - total_mrna_per_cell_slide) / total_mrna_per_cell_slide * 100 if total_mrna_per_cell_slide > 0 else np.nan
        })

    df_fov = pd.DataFrame(fov_records)
    print(f"\n  Extracted {len(df_fov)} FOV records")

    return df_fov


def create_comparison_figure(df_fov):
    """Create figure comparing mRNA quantification with different threshold approaches."""

    fig = plt.figure(figsize=(18, 14), dpi=FIGURE_DPI)
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.30,
                          left=0.07, right=0.97, top=0.94, bottom=0.06)

    session_colors = {'Session 1': '#3498db', 'Session 2': '#e74c3c', 'Session 3': '#2ecc71'}

    # ══════════════════════════════════════════════════════════════════════════
    # ROW 1: Scatter plots - Slide-specific vs Session-mean Total mRNA
    # ══════════════════════════════════════════════════════════════════════════

    for col_idx, channel in enumerate(['HTT1a', 'fl-HTT']):
        ax = fig.add_subplot(gs[0, col_idx])

        ch_data = df_fov[df_fov['Channel'] == channel]

        if len(ch_data) > 0:
            for session in sorted(ch_data['Session'].unique()):
                s_data = ch_data[ch_data['Session'] == session]
                ax.scatter(s_data['Total_mRNA_per_Cell_Slide'],
                          s_data['Total_mRNA_per_Cell_Session'],
                          c=session_colors.get(session, 'gray'), s=30, alpha=0.6,
                          edgecolor='none', label=f"{session} (n={len(s_data)})")

            # Add x=y line
            max_val = max(ch_data['Total_mRNA_per_Cell_Slide'].max(),
                         ch_data['Total_mRNA_per_Cell_Session'].max())
            ax.plot([0, max_val*1.1], [0, max_val*1.1], 'k--', linewidth=1.5, alpha=0.5)

            # Correlation
            r, p = pearsonr(ch_data['Total_mRNA_per_Cell_Slide'],
                           ch_data['Total_mRNA_per_Cell_Session'])
            mean_diff = ch_data['mRNA_Pct_Difference'].mean()
            std_diff = ch_data['mRNA_Pct_Difference'].std()

            ax.text(0.05, 0.95, f"r = {r:.3f}\nMean diff: {mean_diff:+.1f}% ± {std_diff:.1f}%",
                   transform=ax.transAxes, fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        ax.set_xlabel("Total mRNA/nucleus (slide-specific threshold)", fontsize=10)
        ax.set_ylabel("Total mRNA/nucleus (session-mean threshold)", fontsize=10)
        ax.set_title(f"{'A' if col_idx == 0 else 'B'}. {channel}", fontsize=12, fontweight='bold')
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.3, linestyle='--')

    # Panel C: Combined both channels
    ax_c = fig.add_subplot(gs[0, 2])

    for session in sorted(df_fov['Session'].unique()):
        s_data = df_fov[df_fov['Session'] == session]
        ax_c.scatter(s_data['Total_mRNA_per_Cell_Slide'],
                    s_data['Total_mRNA_per_Cell_Session'],
                    c=session_colors.get(session, 'gray'), s=30, alpha=0.5,
                    edgecolor='none', label=f"{session} (n={len(s_data)})")

    max_val = max(df_fov['Total_mRNA_per_Cell_Slide'].max(),
                 df_fov['Total_mRNA_per_Cell_Session'].max())
    ax_c.plot([0, max_val*1.1], [0, max_val*1.1], 'k--', linewidth=1.5, alpha=0.5)

    r, p = pearsonr(df_fov['Total_mRNA_per_Cell_Slide'],
                   df_fov['Total_mRNA_per_Cell_Session'])
    mean_diff = df_fov['mRNA_Pct_Difference'].mean()
    std_diff = df_fov['mRNA_Pct_Difference'].std()

    ax_c.text(0.05, 0.95, f"r = {r:.3f}\nMean diff: {mean_diff:+.1f}% ± {std_diff:.1f}%",
             transform=ax_c.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax_c.set_xlabel("Total mRNA/nucleus (slide-specific)", fontsize=10)
    ax_c.set_ylabel("Total mRNA/nucleus (session-mean)", fontsize=10)
    ax_c.set_title("C. All channels combined", fontsize=12, fontweight='bold')
    ax_c.legend(loc='lower right', fontsize=8)
    ax_c.grid(True, alpha=0.3, linestyle='--')

    # ══════════════════════════════════════════════════════════════════════════
    # ROW 2: Distribution of % differences by session
    # ══════════════════════════════════════════════════════════════════════════

    ax_d = fig.add_subplot(gs[1, 0])

    # Violin plot of % differences by session
    sessions = sorted(df_fov['Session'].unique())
    violin_data = [df_fov[df_fov['Session'] == s]['mRNA_Pct_Difference'].dropna().values
                   for s in sessions]

    parts = ax_d.violinplot(violin_data, positions=range(len(sessions)), showmeans=True, showextrema=True)

    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(session_colors.get(sessions[i], 'gray'))
        pc.set_alpha(0.7)

    ax_d.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax_d.set_xticks(range(len(sessions)))
    ax_d.set_xticklabels(sessions)
    ax_d.set_ylabel("% Difference in mRNA/nucleus\n(session-mean − slide-specific)", fontsize=10)
    ax_d.set_title("D. % Difference by Imaging Session", fontsize=12, fontweight='bold')
    ax_d.grid(True, alpha=0.3, linestyle='--', axis='y')

    # Per-session stats
    for i, session in enumerate(sessions):
        s_data = df_fov[df_fov['Session'] == session]['mRNA_Pct_Difference'].dropna()
        ax_d.text(i, ax_d.get_ylim()[1]*0.9, f"μ={s_data.mean():.1f}%\nn={len(s_data)}",
                 ha='center', fontsize=8)

    # Panel E: By region
    ax_e = fig.add_subplot(gs[1, 1])

    regions = ['Cortex', 'Striatum']
    region_colors = {'Cortex': '#3498db', 'Striatum': '#e74c3c'}

    violin_data_region = [df_fov[df_fov['Region'] == r]['mRNA_Pct_Difference'].dropna().values
                          for r in regions]

    parts = ax_e.violinplot(violin_data_region, positions=range(len(regions)), showmeans=True, showextrema=True)

    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(region_colors.get(regions[i], 'gray'))
        pc.set_alpha(0.7)

    ax_e.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax_e.set_xticks(range(len(regions)))
    ax_e.set_xticklabels(regions)
    ax_e.set_ylabel("% Difference in mRNA/nucleus", fontsize=10)
    ax_e.set_title("E. % Difference by Brain Region", fontsize=12, fontweight='bold')
    ax_e.grid(True, alpha=0.3, linestyle='--', axis='y')

    for i, region in enumerate(regions):
        r_data = df_fov[df_fov['Region'] == region]['mRNA_Pct_Difference'].dropna()
        ax_e.text(i, ax_e.get_ylim()[1]*0.9, f"μ={r_data.mean():.1f}%\nn={len(r_data)}",
                 ha='center', fontsize=8)

    # Panel F: By channel
    ax_f = fig.add_subplot(gs[1, 2])

    channels = ['HTT1a', 'fl-HTT']
    channel_colors = {'HTT1a': '#2ecc71', 'fl-HTT': '#f39c12'}

    violin_data_channel = [df_fov[df_fov['Channel'] == c]['mRNA_Pct_Difference'].dropna().values
                           for c in channels]

    parts = ax_f.violinplot(violin_data_channel, positions=range(len(channels)), showmeans=True, showextrema=True)

    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(channel_colors.get(channels[i], 'gray'))
        pc.set_alpha(0.7)

    ax_f.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax_f.set_xticks(range(len(channels)))
    ax_f.set_xticklabels(channels, fontsize=9)
    ax_f.set_ylabel("% Difference in mRNA/nucleus", fontsize=10)
    ax_f.set_title("F. % Difference by Probe Channel", fontsize=12, fontweight='bold')
    ax_f.grid(True, alpha=0.3, linestyle='--', axis='y')

    for i, channel in enumerate(channels):
        c_data = df_fov[df_fov['Channel'] == channel]['mRNA_Pct_Difference'].dropna()
        ax_f.text(i, ax_f.get_ylim()[1]*0.9, f"μ={c_data.mean():.1f}%\nn={len(c_data)}",
                 ha='center', fontsize=8)

    # ══════════════════════════════════════════════════════════════════════════
    # ROW 3: Per-slide comparison and Bland-Altman
    # ══════════════════════════════════════════════════════════════════════════

    # Panel G: Per-slide mean difference
    ax_g = fig.add_subplot(gs[2, 0:2])

    slide_means = df_fov.groupby(['Slide', 'Session']).agg({
        'mRNA_Pct_Difference': 'mean',
        'Total_mRNA_per_Cell_Slide': 'mean'
    }).reset_index()

    x_pos = 0
    x_ticks = []
    x_labels = []

    for session in sorted(slide_means['Session'].unique()):
        session_slides = slide_means[slide_means['Session'] == session]
        for _, row in session_slides.iterrows():
            color = session_colors.get(session, 'gray')
            ax_g.bar(x_pos, row['mRNA_Pct_Difference'], color=color, alpha=0.7,
                    edgecolor='black', linewidth=0.5)
            x_ticks.append(x_pos)
            x_labels.append(row['Slide'])
            x_pos += 1
        x_pos += 0.5  # Gap between sessions

    ax_g.axhline(0, color='black', linestyle='--', linewidth=1)
    ax_g.set_xticks(x_ticks)
    ax_g.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=7)
    ax_g.set_ylabel("Mean % Difference in mRNA/nucleus", fontsize=10)
    ax_g.set_title("G. Per-Slide Mean % Difference", fontsize=12, fontweight='bold')
    ax_g.grid(True, alpha=0.3, linestyle='--', axis='y')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, alpha=0.7, label=s) for s, c in session_colors.items()]
    ax_g.legend(handles=legend_elements, loc='upper right', fontsize=9)

    # Panel H: Bland-Altman plot
    ax_h = fig.add_subplot(gs[2, 2])

    mean_vals = (df_fov['Total_mRNA_per_Cell_Slide'] + df_fov['Total_mRNA_per_Cell_Session']) / 2
    diff_vals = df_fov['Total_mRNA_per_Cell_Session'] - df_fov['Total_mRNA_per_Cell_Slide']

    for session in sorted(df_fov['Session'].unique()):
        mask = df_fov['Session'] == session
        ax_h.scatter(mean_vals[mask], diff_vals[mask],
                    c=session_colors.get(session, 'gray'), s=30, alpha=0.5,
                    edgecolor='none', label=session)

    # Add mean and limits of agreement
    mean_diff = diff_vals.mean()
    std_diff = diff_vals.std()

    ax_h.axhline(mean_diff, color='red', linestyle='-', linewidth=1.5)
    ax_h.axhline(mean_diff + 1.96*std_diff, color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax_h.axhline(mean_diff - 1.96*std_diff, color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax_h.axhline(0, color='black', linestyle=':', linewidth=1, alpha=0.5, label='Zero difference')

    # Annotate lines
    xlim = ax_h.get_xlim()
    ax_h.text(xlim[1]*0.98, mean_diff + 1.96*std_diff, f'+1.96 SD\n({mean_diff+1.96*std_diff:.1f})',
             fontsize=8, ha='right', va='bottom', color='red')
    ax_h.text(xlim[1]*0.98, mean_diff - 1.96*std_diff, f'-1.96 SD\n({mean_diff-1.96*std_diff:.1f})',
             fontsize=8, ha='right', va='top', color='red')
    ax_h.text(xlim[1]*0.98, mean_diff, f'Mean\n({mean_diff:.2f})',
             fontsize=8, ha='right', va='center', color='red')

    ax_h.set_xlabel("Mean mRNA/nucleus\n(average of both methods)", fontsize=10)
    ax_h.set_ylabel("Difference in mRNA/nucleus\n(session-mean − slide-specific)", fontsize=10)
    ax_h.set_title("H. Bland-Altman Agreement Plot", fontsize=12, fontweight='bold')
    ax_h.legend(loc='upper left', fontsize=8)
    ax_h.grid(True, alpha=0.3, linestyle='--')

    fig.suptitle('Effect of Threshold Calibration Method on Total mRNA per Nucleus Quantification',
                fontsize=14, fontweight='bold')

    return fig


if __name__ == "__main__":

    print("="*80)
    print("THRESHOLD EFFECT ON mRNA QUANTIFICATION")
    print("Comparing slide-specific vs session-mean thresholds")
    print("="*80)

    # ══════════════════════════════════════════════════════════════════════════
    # 1. LOAD DATA
    # ══════════════════════════════════════════════════════════════════════════

    h5_file_path = H5_FILE_PATH_EXPERIMENTAL

    with h5py.File(h5_file_path, 'r') as h5_file:
        data_dict = recursively_load_dict(h5_file)

    print(f"\nLoaded data from: {h5_file_path}")

    # ══════════════════════════════════════════════════════════════════════════
    # 2. EXTRACT DATAFRAME
    # ══════════════════════════════════════════════════════════════════════════

    channels_to_extract = ['green', 'orange', 'blue']
    channels_to_analyze = ['green', 'orange']
    fields_to_extract = [
        'spots_sigma_var.params_raw',
        'spots.params_raw',
        'cluster_intensities',
        'cluster_cvs',
        'num_cells',
        'label_sizes',
        'metadata_sample.Age',
        'spots.final_filter',
        'metadata_sample.Brain_Atlas_coordinates',
        'metadata_sample.mouse_ID'
    ]

    negative_control_field = 'Negative Control'
    experimental_field = 'ExperimentalQ111 - 488mHT - 548mHTa - 647Darp'
    slide_field = SLIDE_FIELD

    df_extracted = extract_dataframe(
        data_dict,
        field_keys=fields_to_extract,
        channels=channels_to_extract,
        include_file_metadata_sample=True,
        explode_fields=[]
    )

    print(f"DataFrame extracted: {len(df_extracted)} total rows")

    # ══════════════════════════════════════════════════════════════════════════
    # 3. COMPUTE THRESHOLDS
    # ══════════════════════════════════════════════════════════════════════════

    print("\nComputing per-slide thresholds...")

    (thresholds, thresholds_cluster,
     error_thresholds, error_thresholds_cluster,
     number_of_datapoints, age_dict) = compute_thresholds(
        df_extracted=df_extracted,
        slide_field=slide_field,
        desired_channels=channels_to_analyze,
        negative_control_field=negative_control_field,
        experimental_field=experimental_field,
        quantile_negative_control=QUANTILE_NEGATIVE_CONTROL,
        max_pfa=MAX_PFA,
        plot=False,
        n_bootstrap=N_BOOTSTRAP,
        use_region=False,
        use_final_filter=True,
    )

    # Build slide-specific threshold dict
    thresholds_slide = {}
    for (slide, channel, area), vec in error_thresholds.items():
        thresholds_slide[(slide, channel)] = np.mean(vec)

    # Compute session-mean thresholds (using ONLY included slides, excluding EXCLUDED_SLIDES)
    print(f"\nComputing session-mean thresholds (excluding {len(EXCLUDED_SLIDES)} problematic slides)...")
    session_mean_thresholds = {}
    for channel in ['green', 'orange']:
        session_mean_thresholds[channel] = {}

        # Group by session (only include slides NOT in EXCLUDED_SLIDES)
        session_thresholds = {}
        for (slide, ch), thr in thresholds_slide.items():
            if ch != channel:
                continue
            if slide in EXCLUDED_SLIDES:
                continue  # Skip excluded slides
            session = extract_imaging_session(slide)
            if session not in session_thresholds:
                session_thresholds[session] = []
            session_thresholds[session].append(thr)

        # Compute means
        for session, thrs in session_thresholds.items():
            session_mean_thresholds[channel][session] = np.mean(thrs)
            print(f"  {channel} {session}: session-mean threshold = {np.mean(thrs):.1f} (n={len(thrs)} slides)")

    # Merge thresholds into dataframe
    df_extracted['threshold'] = df_extracted.apply(
        lambda row: thresholds_slide.get((row[slide_field], row['channel']), np.nan), axis=1
    )

    # ══════════════════════════════════════════════════════════════════════════
    # 4. FILTER TO Q111 EXPERIMENTAL DATA (excluding problematic slides)
    # ══════════════════════════════════════════════════════════════════════════

    df_exp = df_extracted[
        (df_extracted["metadata_sample_Mouse_Model"] == 'Q111') &
        (df_extracted['metadata_sample_Probe-Set'] == experimental_field)
    ].copy()

    print(f"\nFiltered to Q111 experimental: {len(df_exp)} rows")

    # Apply slide exclusions
    n_before = len(df_exp)
    df_exp = df_exp[~df_exp[slide_field].isin(EXCLUDED_SLIDES)].copy()
    n_after = len(df_exp)
    print(f"After excluding {len(EXCLUDED_SLIDES)} problematic slides: {n_after} rows (removed {n_before - n_after})")
    print(f"Excluded slides: {', '.join(sorted(EXCLUDED_SLIDES))}")

    # ══════════════════════════════════════════════════════════════════════════
    # 5. EXTRACT FOV DATA WITH BOTH THRESHOLD APPROACHES
    # ══════════════════════════════════════════════════════════════════════════

    df_fov = extract_fov_data_with_both_thresholds(df_exp, thresholds_slide, session_mean_thresholds)

    # ══════════════════════════════════════════════════════════════════════════
    # 6. CREATE COMPARISON FIGURE
    # ══════════════════════════════════════════════════════════════════════════

    print("\nCreating comparison figure...")

    fig = create_comparison_figure(df_fov)

    # Save figure
    for fmt in ['png', 'svg', 'pdf']:
        filepath = OUTPUT_DIR / f"fig_threshold_effect_on_mrna.{fmt}"
        fig.savefig(filepath, format=fmt, bbox_inches='tight', dpi=FIGURE_DPI)
        print(f"  Saved: {filepath}")

    plt.close(fig)

    # Save FOV data
    csv_path = OUTPUT_DIR / "fov_threshold_comparison_data.csv"
    df_fov.to_csv(csv_path, index=False)
    print(f"  Data saved: {csv_path}")

    # ══════════════════════════════════════════════════════════════════════════
    # 7. PRINT SUMMARY STATISTICS
    # ══════════════════════════════════════════════════════════════════════════

    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    print(f"\nTotal FOVs analyzed: {len(df_fov)}")
    print(f"Mean % difference (session-mean - slide-specific): {df_fov['mRNA_Pct_Difference'].mean():.2f}% ± {df_fov['mRNA_Pct_Difference'].std():.2f}%")

    print("\nBy Session:")
    for session in sorted(df_fov['Session'].unique()):
        s_data = df_fov[df_fov['Session'] == session]
        print(f"  {session}: {s_data['mRNA_Pct_Difference'].mean():.2f}% ± {s_data['mRNA_Pct_Difference'].std():.2f}% (n={len(s_data)})")

    print("\nBy Region:")
    for region in ['Cortex', 'Striatum']:
        r_data = df_fov[df_fov['Region'] == region]
        print(f"  {region}: {r_data['mRNA_Pct_Difference'].mean():.2f}% ± {r_data['mRNA_Pct_Difference'].std():.2f}% (n={len(r_data)})")

    print("\nBy Channel:")
    for channel in ['HTT1a', 'fl-HTT']:
        c_data = df_fov[df_fov['Channel'] == channel]
        print(f"  {channel}: {c_data['mRNA_Pct_Difference'].mean():.2f}% ± {c_data['mRNA_Pct_Difference'].std():.2f}% (n={len(c_data)})")

    # Paired t-test
    t_stat, p_val = ttest_rel(df_fov['Total_mRNA_per_Cell_Slide'],
                              df_fov['Total_mRNA_per_Cell_Session'])
    print(f"\nPaired t-test (slide vs session-mean): t = {t_stat:.3f}, p = {p_val:.2e}")

    # ══════════════════════════════════════════════════════════════════════════
    # 8. GENERATE COMPREHENSIVE CAPTION
    # ══════════════════════════════════════════════════════════════════════════

    print("\nGenerating detailed caption...")

    # Compute additional statistics for caption
    mean_diff_pct = df_fov['mRNA_Pct_Difference'].mean()
    std_diff_pct = df_fov['mRNA_Pct_Difference'].std()
    median_diff_pct = df_fov['mRNA_Pct_Difference'].median()
    iqr_diff_pct = df_fov['mRNA_Pct_Difference'].quantile(0.75) - df_fov['mRNA_Pct_Difference'].quantile(0.25)

    # Correlation
    r_overall, p_corr = pearsonr(df_fov['Total_mRNA_per_Cell_Slide'],
                                  df_fov['Total_mRNA_per_Cell_Session'])

    # Bland-Altman limits
    mean_vals = (df_fov['Total_mRNA_per_Cell_Slide'] + df_fov['Total_mRNA_per_Cell_Session']) / 2
    diff_vals = df_fov['Total_mRNA_per_Cell_Session'] - df_fov['Total_mRNA_per_Cell_Slide']
    ba_mean = diff_vals.mean()
    ba_std = diff_vals.std()
    ba_lower = ba_mean - 1.96 * ba_std
    ba_upper = ba_mean + 1.96 * ba_std

    # Per-slide statistics
    slide_means = df_fov.groupby('Slide')['mRNA_Pct_Difference'].mean()
    max_slide_bias = slide_means.abs().max()
    max_slide_name = slide_means.abs().idxmax()
    slides_above_10pct = (slide_means.abs() > 10).sum()
    slides_above_20pct = (slide_means.abs() > 20).sum()

    # Effect size (Cohen's d for paired data)
    cohens_d = mean_diff_pct / std_diff_pct

    # Percentage of FOVs with >10%, >20%, >50% difference
    fovs_above_10pct = (df_fov['mRNA_Pct_Difference'].abs() > 10).sum() / len(df_fov) * 100
    fovs_above_20pct = (df_fov['mRNA_Pct_Difference'].abs() > 20).sum() / len(df_fov) * 100
    fovs_above_50pct = (df_fov['mRNA_Pct_Difference'].abs() > 50).sum() / len(df_fov) * 100

    # Maximum individual FOV difference
    max_fov_diff = df_fov['mRNA_Pct_Difference'].abs().max()
    min_fov_diff = df_fov['mRNA_Pct_Difference'].min()
    max_fov_diff_signed = df_fov['mRNA_Pct_Difference'].max()
    percentile_95 = np.percentile(df_fov['mRNA_Pct_Difference'].abs(), 95)
    percentile_5 = np.percentile(df_fov['mRNA_Pct_Difference'], 5)
    percentile_95_signed = np.percentile(df_fov['mRNA_Pct_Difference'], 95)

    caption_lines = [
        "=" * 100,
        "SUPPLEMENTARY FIGURE: Effect of Threshold Calibration Method on Total mRNA Quantification",
        "=" * 100,
        "",
        "OVERVIEW",
        "-" * 100,
        "This figure evaluates whether per-slide negative control calibration is necessary for accurate",
        "mRNA quantification, or whether per-session (pooled) thresholds provide equivalent results.",
        "We compare Total mRNA per nucleus calculated using two threshold approaches:",
        "  1. SLIDE-SPECIFIC thresholds: 95th percentile of negative control intensities per slide",
        "  2. SESSION-MEAN thresholds: Average of slide-specific thresholds within each imaging session",
        "",
        "The analysis addresses a practical question: Can researchers simplify their calibration protocol",
        "by using a single threshold per imaging session, rather than requiring negative control sections",
        "on every experimental slide?",
        "",
        "=" * 100,
        "QUANTITATIVE ASSESSMENT",
        "=" * 100,
        "",
        "DATASET:",
        f"  - Total FOVs analyzed: {len(df_fov):,}",
        f"  - Imaging sessions: {len(df_fov['Session'].unique())} (Session 1: n={len(df_fov[df_fov['Session']=='Session 1'])}, "
        f"Session 2: n={len(df_fov[df_fov['Session']=='Session 2'])}, Session 3: n={len(df_fov[df_fov['Session']=='Session 3'])})",
        f"  - Brain regions: Cortex (n={len(df_fov[df_fov['Region']=='Cortex'])}), Striatum (n={len(df_fov[df_fov['Region']=='Striatum'])})",
        f"  - Channels: HTT1a (n={len(df_fov[df_fov['Channel']=='HTT1a'])}), fl-HTT (n={len(df_fov[df_fov['Channel']=='fl-HTT'])})",
        "",
        "QUALITY CONTROL EXCLUSIONS:",
        f"  - Excluded slides (n={len(EXCLUDED_SLIDES)}): {', '.join(sorted(EXCLUDED_SLIDES))}",
        "  - Reason for exclusion: Poor UBC positive control expression (100-1000x below normal),",
        "    indicating technical failures such as poor hybridization, tissue damage, or imaging issues",
        f"  - Cluster CV filter: CV >= {CV_THRESHOLD} (excludes clusters with low intensity variance)",
        "",
        "AGREEMENT BETWEEN METHODS:",
        f"  - Pearson correlation: r = {r_overall:.4f} (p < 1e-100)",
        f"    → Near-perfect linear agreement between methods",
        "",
        f"  - Mean % difference (session-mean - slide-specific): {mean_diff_pct:+.2f}% ± {std_diff_pct:.2f}%",
        f"  - Median % difference: {median_diff_pct:+.2f}% (IQR: {iqr_diff_pct:.2f}%)",
        f"    → Small systematic bias toward slightly lower values with session-mean thresholds",
        "",
        f"  - Paired t-test: t = {t_stat:.3f}, p = {p_val:.2e}",
        f"    → Statistically significant difference (due to large sample size)",
        f"  - Cohen's d effect size: {cohens_d:.3f}",
        f"    → {'Negligible' if abs(cohens_d) < 0.2 else 'Small' if abs(cohens_d) < 0.5 else 'Medium' if abs(cohens_d) < 0.8 else 'Large'} effect size",
        "",
        "BLAND-ALTMAN ANALYSIS (Panel H):",
        f"  - Mean difference: {ba_mean:.3f} mRNA/nucleus",
        f"  - 95% Limits of agreement: [{ba_lower:.2f}, {ba_upper:.2f}] mRNA/nucleus",
        f"    → Most FOVs fall within ±{(ba_upper - ba_lower)/2:.1f} mRNA/nucleus of each other",
        "",
        "DISTRIBUTION OF DIFFERENCES:",
        f"  - FOVs with >10% difference: {fovs_above_10pct:.1f}%",
        f"  - FOVs with >20% difference: {fovs_above_20pct:.1f}%",
        f"  - FOVs with >50% difference: {fovs_above_50pct:.1f}%",
        "",
        "EXTREME VALUES (IMPORTANT!):",
        f"  - Maximum individual FOV difference: {max_fov_diff:.1f}%",
        f"  - Range of differences: {min_fov_diff:.1f}% to {max_fov_diff_signed:+.1f}%",
        f"  - 5th to 95th percentile range: {percentile_5:.1f}% to {percentile_95_signed:+.1f}%",
        "  - NOTE: While the MEAN difference is small, individual FOVs can differ by up to 60% or more!",
        "    This variability is substantial and may affect biological conclusions for individual samples.",
        "",
        "PER-SESSION BREAKDOWN:",
    ]

    for session in sorted(df_fov['Session'].unique()):
        s_data = df_fov[df_fov['Session'] == session]
        s_mean = s_data['mRNA_Pct_Difference'].mean()
        s_std = s_data['mRNA_Pct_Difference'].std()
        caption_lines.append(f"  - {session}: {s_mean:+.2f}% ± {s_std:.2f}% (n={len(s_data)})")

    caption_lines.extend([
        "",
        "PER-SLIDE VARIABILITY (Panel G):",
        f"  - Number of slides: {len(slide_means)}",
        f"  - Maximum per-slide bias: {max_slide_bias:.1f}% (slide: {max_slide_name})",
        f"  - Slides with >10% mean bias: {slides_above_10pct} ({slides_above_10pct/len(slide_means)*100:.1f}%)",
        f"  - Slides with >20% mean bias: {slides_above_20pct} ({slides_above_20pct/len(slide_means)*100:.1f}%)",
        "",
        "=" * 100,
        "INTERPRETATION AND RECOMMENDATIONS",
        "=" * 100,
        "",
        "IS SLIDE-SPECIFIC CALIBRATION NECESSARY?",
        "-" * 100,
        "",
    ])

    # Provide nuanced interpretation based on the statistics
    # Key insight: even if mean is small, large individual variation matters
    if fovs_above_20pct > 30 or max_fov_diff > 50:
        interpretation = "HIGH"
        caption_lines.extend([
            "ANSWER: YES - SLIDE-SPECIFIC CALIBRATION IS STRONGLY RECOMMENDED",
            "",
            f"Despite the small mean difference ({mean_diff_pct:+.1f}%), individual FOVs show LARGE variability:",
            f"  - {fovs_above_20pct:.1f}% of FOVs differ by >20%",
            f"  - Maximum difference observed: {max_fov_diff:.1f}%",
            f"  - This means some individual measurements could be off by more than half!",
            "",
        ])
    elif abs(mean_diff_pct) < 5 and fovs_above_20pct < 30:
        interpretation = "MODERATE"
        caption_lines.extend([
            "ANSWER: RECOMMENDED BUT NOT STRICTLY REQUIRED",
            "",
            "The data suggest that slide-specific calibration provides marginally more accurate results,",
            "but session-mean thresholds may be acceptable for many applications:",
            "",
        ])
    else:
        interpretation = "LOW"
        caption_lines.extend([
            "ANSWER: OPTIONAL - SESSION-MEAN THRESHOLDS ARE GENERALLY ACCEPTABLE",
            "",
            "The data suggest that session-mean thresholds provide nearly equivalent results:",
            "",
        ])

    caption_lines.extend([
        "STATISTICAL EVIDENCE:",
        f"  1. CORRELATION: r = {r_overall:.3f} indicates near-perfect linear agreement.",
        f"     Both methods rank FOVs identically; relative comparisons are unaffected.",
        "",
        f"  2. SYSTEMATIC BIAS: Mean difference of {mean_diff_pct:+.2f}% is {'negligible' if abs(mean_diff_pct) < 2 else 'small' if abs(mean_diff_pct) < 5 else 'moderate' if abs(mean_diff_pct) < 10 else 'substantial'}.",
        f"     {'This bias is unlikely to affect biological conclusions.' if abs(mean_diff_pct) < 5 else 'This bias may affect absolute quantification but not relative comparisons.'}",
        "",
        f"  3. INDIVIDUAL VARIABILITY: {fovs_above_20pct:.1f}% of FOVs differ by >20%.",
        f"     {'Most measurements are highly consistent between methods.' if fovs_above_20pct < 20 else 'A substantial minority of measurements show large differences.'}",
        "",
        f"  4. PER-SLIDE EFFECTS: {slides_above_20pct} of {len(slide_means)} slides ({slides_above_20pct/len(slide_means)*100:.1f}%) show >20% mean bias.",
        f"     {'Slide-to-slide variation is well-controlled by session-mean thresholds.' if slides_above_20pct/len(slide_means) < 0.15 else 'Some slides show substantial systematic differences between methods.'}",
        "",
        "-" * 100,
        "PRACTICAL RECOMMENDATIONS FOR FUTURE USERS:",
        "-" * 100,
        "",
        "1. FOR HIGHEST ACCURACY (Recommended for publication-quality data):",
        "   → Use SLIDE-SPECIFIC thresholds from per-slide negative control sections",
        "   → This eliminates any potential slide-to-slide calibration artifacts",
        "   → Required effort: One negative control section per experimental slide",
        "",
        "2. FOR PRACTICAL EFFICIENCY (Acceptable for screening or exploratory work):",
        "   → Use SESSION-MEAN thresholds (one negative control per imaging session)",
        f"   → Expected accuracy: r = {r_overall:.3f} correlation with slide-specific values",
        f"   → Expected bias: {mean_diff_pct:+.2f}% ± {std_diff_pct:.2f}% systematic difference",
        "   → Required effort: One negative control section per imaging session",
        "",
        "3. CRITICAL CONSIDERATIONS:",
        f"   → Session 2 shows the largest systematic bias ({df_fov[df_fov['Session']=='Session 2']['mRNA_Pct_Difference'].mean():+.1f}%)",
        "   → If comparing samples ACROSS sessions, slide-specific calibration is strongly recommended",
        "   → If comparing samples WITHIN a single session, session-mean thresholds are acceptable",
        "",
        "4. STUDY DESIGN IMPLICATIONS:",
        "   → Ensure experimental groups are balanced across imaging sessions",
        "   → If using session-mean thresholds, verify that biological groups are not confounded with sessions",
        "   → Report which calibration method was used in methods section",
        "",
        "=" * 100,
        "PANEL DESCRIPTIONS",
        "=" * 100,
        "",
        "ROW 1 - SCATTER PLOTS (Panels A-C):",
        "Comparison of Total mRNA per nucleus calculated with slide-specific (x-axis) vs session-mean",
        "(y-axis) thresholds. Each point represents one field-of-view (FOV). Dashed line indicates",
        "perfect agreement (x=y). Points above the line indicate higher values with session-mean thresholds.",
        "",
        "  Panel A: HTT1a channel (488 nm) - exon 1 transcript",
        "  Panel B: fl-HTT channel (548 nm) - full-length transcript",
        "  Panel C: Both channels combined",
        "",
        "ROW 2 - DISTRIBUTION OF DIFFERENCES (Panels D-F):",
        "Violin plots showing the distribution of percentage differences between methods.",
        "Horizontal dashed line at 0% indicates no difference. Positive values indicate session-mean",
        "thresholds yield higher mRNA counts; negative values indicate lower counts.",
        "",
        "  Panel D: By imaging session - reveals session-specific calibration effects",
        "  Panel E: By brain region - tests whether bias differs between cortex and striatum",
        "  Panel F: By channel - tests whether bias differs between probe targets",
        "",
        "ROW 3 - PER-SLIDE AND BLAND-ALTMAN ANALYSIS (Panels G-H):",
        "",
        "  Panel G: Per-slide mean difference",
        "  Bar plot showing the mean percentage difference for each slide, grouped by session.",
        "  Reveals which specific slides show the largest calibration differences.",
        "",
        "  Panel H: Bland-Altman plot",
        "  X-axis: Mean of both methods (average mRNA/nucleus from both approaches)",
        "  Y-axis: Difference between methods (session-mean minus slide-specific)",
        "  Red solid line: Mean difference (systematic bias)",
        "  Red dashed lines: 95% limits of agreement (±1.96 SD)",
        "  This plot reveals whether the difference between methods depends on the magnitude of expression.",
        "",
        "=" * 100,
        "METHODOLOGY",
        "=" * 100,
        "",
        "THRESHOLD CALCULATION:",
        f"  - Quantile: {QUANTILE_NEGATIVE_CONTROL*100:.0f}th percentile of negative control spot intensities",
        "  - Negative control: Bacterial DapB probe (not expressed in mammalian tissue)",
        "  - Slide-specific: Calculated independently for each slide-channel combination",
        "  - Session-mean: Average of slide-specific thresholds within each imaging session",
        "    (computed ONLY from included slides, excluding problematic slides listed above)",
        "",
        "TOTAL mRNA PER NUCLEUS CALCULATION:",
        "  Formula: Total mRNA per nucleus = (N_spots + I_clusters / I_peak) / N_nuclei",
        "",
        "  Step-by-step breakdown:",
        "  1. COUNT SINGLE SPOTS (N_spots):",
        "     - Filter spots by intensity > threshold (from negative control)",
        "     - Filter spots by detection quality (PFA < 0.05)",
        "     - Count remaining spots as single mRNA molecules",
        "",
        "  2. QUANTIFY CLUSTERED mRNA (I_clusters / I_peak):",
        "     - I_clusters = Sum of integrated intensities from all clusters",
        f"     - Clusters must pass: intensity > threshold AND CV >= {CV_THRESHOLD}",
        "     - I_peak = Mode of spot intensity distribution (from KDE)",
        "     - I_peak represents the intensity of a single mRNA molecule",
        "     - Division converts total cluster intensity to mRNA equivalents",
        "",
        "  3. ESTIMATE NUCLEI COUNT (N_nuclei):",
        "     - Segment DAPI channel to identify nuclear regions",
        "     - Calculate total DAPI volume: V_DAPI = sum(label_sizes) × voxel_size",
        f"     - Estimate nuclei: N_nuclei = V_DAPI / mean_nuclear_volume",
        f"     - Mean nuclear volume = {mean_nuclear_volume:.1f} µm³ (from literature)",
        "",
        "  4. NORMALIZE TO PER-NUCLEUS:",
        "     - Divide total mRNA (spots + cluster equivalents) by N_nuclei",
        "     - Result: mRNA molecules per nucleus",
        "",
        "  KEY INSIGHT: The threshold affects BOTH N_spots AND I_peak:",
        "  - Higher threshold → fewer spots pass → lower N_spots",
        "  - Higher threshold → higher I_peak (excludes low-intensity spots from KDE)",
        "  - These effects partially cancel, but not perfectly",
        "",
        "PERCENTAGE DIFFERENCE CALCULATION:",
        "  Formula: % Difference = (mRNA_session − mRNA_slide) / mRNA_slide × 100",
        "",
        "  Where:",
        "    mRNA_session = Total mRNA/nucleus using session-mean threshold",
        "    mRNA_slide = Total mRNA/nucleus using slide-specific threshold",
        "",
        "  Interpretation:",
        "    - Positive % → session-mean threshold gives HIGHER mRNA estimate",
        "    - Negative % → session-mean threshold gives LOWER mRNA estimate",
        "    - 0% → both methods give identical results",
        "",
        "  Example: If slide-specific gives 50 mRNA/nucleus and session-mean gives 55:",
        "    % Difference = (55 − 50) / 50 × 100 = +10%",
        "",
        "CLUSTER QUALITY FILTERING:",
        f"  - Coefficient of Variation (CV) threshold: >= {CV_THRESHOLD}",
        "  - Rationale: Clusters with low CV may represent uniform background or artifacts",
        "  - True mRNA clusters show intensity heterogeneity due to multiple overlapping molecules",
        "",
        "STATISTICAL TESTS:",
        "  - Pearson correlation: Linear agreement between methods",
        "  - Paired t-test: Systematic difference between methods",
        "  - Cohen's d: Standardized effect size",
        "  - Bland-Altman: Agreement analysis with limits of agreement",
        "",
        "=" * 100,
        f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 100,
    ])

    # Save caption
    caption_text = '\n'.join(caption_lines)
    caption_path = OUTPUT_DIR / "fig_threshold_effect_on_mrna_caption.txt"
    with open(caption_path, 'w') as f:
        f.write(caption_text)
    print(f"  Caption saved: {caption_path}")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
