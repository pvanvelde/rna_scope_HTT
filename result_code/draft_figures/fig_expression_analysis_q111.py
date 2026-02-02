"""
Comprehensive Expression Analysis Figure for Q111 Mice
Shows fl-HTT and HTT1a expression levels across cortex and striatum
Including total, single, and clustered mRNA per cell, plus per-mouse breakdowns.

Author: Generated with Claude Code
Date: 2025-11-16
"""

import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, ttest_rel, linregress, pearsonr, gaussian_kde, mannwhitneyu
from pathlib import Path
import seaborn as sns

import sys
# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from result_functions_v2 import (
    compute_thresholds,
    recursively_load_dict,
    extract_dataframe
)

# Import centralized configuration
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
    FIGURE_FORMAT,
    EXCLUDED_SLIDES,
    CV_THRESHOLD
)

# Output directory
OUTPUT_DIR = Path(__file__).parent / "output" / "expression_analysis_q111"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Figure settings
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9


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


def extract_fov_level_data(df_input, mouse_model_name):
    """
    Extract FOV-level data with expression metrics.
    This function computes slide-specific peak intensities,
    then extracts FOV-level expression data.
    """

    # Configuration
    slide_field = 'metadata_sample_slide_name_std'

    # Define region lists
    striatum_subregions = [
        "Striatum - lower left",
        "Striatum - lower right",
        "Striatum - upper left",
        "Striatum - upper right",
    ]
    cortex_subregions = [
        "Cortex - Piriform area",
        "Cortex - Primary and secondary motor areas",
        "Cortex - Primary somatosensory (mouth, upper limb)",
        "Cortex - Supplemental/primary somatosensory (nose)",
        "Cortex - Visceral/gustatory/agranular areas",
    ]

    # Channel labels
    channel_labels_exp = {
        'green': 'HTT1a',
        'orange': 'fl-HTT'
    }

    # ──────────────────────────────────────────────────────────────────────
    # STEP 1: Compute slide-specific peak intensities for normalization
    # ──────────────────────────────────────────────────────────────────────

    print("\n[1/3] Computing slide-specific peak intensities...")

    spot_peaks = {}  # (slide, channel) -> peak_intensity

    for idx, row in df_input.iterrows():
        slide = row.get(slide_field, 'unknown')
        channel = row.get('channel', 'unknown')

        # Skip DAPI channel
        if channel == 'blue':
            continue

        key = (slide, channel)

        # Only compute once per slide-channel combo
        if key not in spot_peaks:
            # Collect all single spot intensities for this slide-channel
            all_intensities = []

            for idx2, row2 in df_input.iterrows():
                if (row2.get(slide_field, 'unknown') == slide and
                    row2.get('channel', 'unknown') == channel):

                    params = row2.get('spots_sigma_var.params_raw', None)
                    final_filter = row2.get('spots.final_filter', None)
                    threshold_val = row2.get('threshold', np.nan)

                    if params is not None and final_filter is not None:
                        params = np.array(params)
                        final_filter = np.array(final_filter).astype(bool)

                        if len(params) > 0 and len(final_filter) == len(params):
                            params_filtered = params[final_filter]
                            if params_filtered.shape[1] >= 4:
                                intensities = params_filtered[:, 3]
                                # Apply threshold filter
                                if not np.isnan(threshold_val):
                                    intensities = intensities[intensities > threshold_val]
                                all_intensities.extend(intensities)

            # Compute peak intensity
            if len(all_intensities) >= 50:
                peak_intensity = compute_peak_intensity(np.array(all_intensities))
                if not np.isnan(peak_intensity):
                    spot_peaks[key] = peak_intensity
                    print(f"  {slide} - {channel}: peak = {peak_intensity:.1f}")

    # ──────────────────────────────────────────────────────────────────────
    # STEP 2: Build DAPI lookup (blue channel nuclei counts)
    # ──────────────────────────────────────────────────────────────────────

    print("\n[2/3] Building DAPI nuclei lookup...")

    # Sort by index to ensure blue/green/orange triplets are together
    df_input_sorted = df_input.sort_index()

    dapi_lookup = {}  # idx -> (N_nuc, V_DAPI)

    for idx, row in df_input_sorted.iterrows():
        channel = row.get('channel', 'unknown')

        if channel == 'blue':
            label_sizes = row.get('label_sizes', None)

            if label_sizes is not None and len(label_sizes) > 0:
                label_sizes = np.array(label_sizes)
                V_DAPI = np.sum(label_sizes) * voxel_size
                N_nuc = V_DAPI / mean_nuclear_volume

                # Assign to blue, green, and orange (idx, idx+1, idx+2)
                dapi_lookup[idx] = (N_nuc, V_DAPI)
                dapi_lookup[idx + 1] = (N_nuc, V_DAPI)  # green
                dapi_lookup[idx + 2] = (N_nuc, V_DAPI)  # orange

    print(f"  Built DAPI lookup for {len(dapi_lookup)} entries")

    # ──────────────────────────────────────────────────────────────────────
    # STEP 3: Extract FOV-level expression data
    # ──────────────────────────────────────────────────────────────────────

    print("\n[3/3] Extracting FOV-level expression data...")

    fov_records = []

    for idx, row in df_input.iterrows():
        # Get channel
        channel = row.get('channel', 'unknown')

        # Skip blue channel
        if channel == 'blue':
            continue

        # Get metadata
        slide = row.get(slide_field, 'unknown')
        subregion = row.get('metadata_sample_Slice_Region', 'unknown')
        age = row.get('metadata_sample_Age', np.nan)
        atlas_coord = row.get('metadata_sample_Brain_Atlas_coordinates', np.nan)
        mouse_id = row.get('metadata_sample_mouse_ID', 'unknown')
        threshold = row.get('threshold', np.nan)

        # Determine merged region
        if any(sub in subregion for sub in cortex_subregions):
            region = 'Cortex'
        elif any(sub in subregion for sub in striatum_subregions):
            region = 'Striatum'
        else:
            continue  # Skip unknown regions

        # Map channel to readable name
        channel_name = channel_labels_exp.get(channel, channel)

        # Get DAPI-based nuclei count
        if idx not in dapi_lookup:
            continue

        N_nuc, V_DAPI = dapi_lookup[idx]

        # Apply QC filter
        if N_nuc < MIN_NUCLEI_THRESHOLD:
            continue

        # Get spots and clusters
        params = row.get('spots_sigma_var.params_raw', None)
        final_filter = row.get('spots.final_filter', None)
        cluster_intensities = row.get('cluster_intensities', None)
        cluster_cvs = row.get('cluster_cvs', None)

        # Initialize metrics
        num_spots = 0
        num_clusters = 0
        I_cluster_total = 0
        single_spot_intensities = []
        cluster_intensities_list = []

        # Process spots
        if params is not None and final_filter is not None:
            params = np.array(params)
            final_filter = np.array(final_filter).astype(bool)

            if len(params) > 0 and len(final_filter) == len(params):
                params_filtered = params[final_filter]

                if params_filtered.shape[1] >= 4:
                    intensities = params_filtered[:, 3]
                    # Apply threshold filter
                    if not np.isnan(threshold):
                        above_threshold = intensities > threshold
                        num_spots = above_threshold.sum()
                        single_spot_intensities = intensities[above_threshold].tolist()
                    else:
                        num_spots = len(intensities)
                        single_spot_intensities = intensities.tolist()

        # Process clusters (with intensity AND CV filtering)
        if cluster_intensities is not None:
            cluster_intensities = np.array(cluster_intensities)
            if len(cluster_intensities) > 0:
                # Apply threshold filter
                if not np.isnan(threshold):
                    # Intensity threshold
                    intensity_mask = cluster_intensities > threshold
                    # CV threshold (CV >= CV_THRESHOLD means good quality)
                    # CV data is required - no fallback
                    if cluster_cvs is None:
                        raise ValueError("CV data missing for cluster filtering")
                    cluster_cvs = np.array(cluster_cvs)
                    if len(cluster_cvs) != len(cluster_intensities):
                        raise ValueError(f"CV data length mismatch: {len(cluster_cvs)} vs {len(cluster_intensities)}")
                    cv_mask = cluster_cvs >= CV_THRESHOLD
                    above_threshold = intensity_mask & cv_mask
                    num_clusters = above_threshold.sum()
                    I_cluster_total = cluster_intensities[above_threshold].sum()
                    cluster_intensities_list = cluster_intensities[above_threshold].tolist()
                else:
                    num_clusters = len(cluster_intensities)
                    I_cluster_total = np.sum(cluster_intensities)
                    cluster_intensities_list = cluster_intensities.tolist()

        # Get peak intensity for this slide-channel
        key = (slide, channel)
        peak_intensity = spot_peaks.get(key, np.nan)

        if np.isnan(peak_intensity):
            continue

        # Compute mRNA equivalents
        cluster_mrna_equiv = I_cluster_total / peak_intensity
        total_mrna_equiv = num_spots + cluster_mrna_equiv

        # Compute per-cell metrics
        single_mrna_per_cell = num_spots / N_nuc
        clustered_mrna_per_cell = cluster_mrna_equiv / N_nuc
        total_mrna_per_cell = total_mrna_equiv / N_nuc
        clusters_per_cell = num_clusters / N_nuc

        # Store record
        fov_record = {
            'Mouse_Model': mouse_model_name,
            'Mouse_ID': mouse_id,
            'Slide': slide,
            'Region': region,
            'Subregion': subregion,
            'Channel': channel_name,
            'Age': age,
            'Brain_Atlas_Coord': atlas_coord,
            'Threshold': threshold,
            'N_Nuclei': N_nuc,
            'V_DAPI': V_DAPI,
            'N_Spots': num_spots,
            'N_Clusters': num_clusters,
            'I_Cluster_Total': I_cluster_total,
            'I_Single_Peak': peak_intensity,
            'Cluster_mRNA_Equiv': cluster_mrna_equiv,
            'Total_mRNA_Equiv': total_mrna_equiv,
            'Single_mRNA_per_Cell': single_mrna_per_cell,
            'Clustered_mRNA_per_Cell': clustered_mrna_per_cell,
            'Total_mRNA_per_Cell': total_mrna_per_cell,
            'Clusters_per_Cell': clusters_per_cell,
            'Single_Spot_Intensities': single_spot_intensities,
            'Cluster_Intensities': cluster_intensities_list
        }

        fov_records.append(fov_record)

    df_fov = pd.DataFrame(fov_records)
    print(f"\n  Extracted {len(df_fov)} FOV records")

    return df_fov


def create_comprehensive_expression_figure(df_fov):
    """
    Create a comprehensive multi-panel figure showing:
    - Total mRNA per cell
    - Single mRNA per cell
    - Clustered mRNA per cell
    - Total expression per mouse ID

    Layout: 4 rows x 2 columns
    Left column: Cortex vs Striatum comparisons
    Right column: Per-mouse ID breakdowns
    """

    # Filter for Q111 mice only
    df_q111 = df_fov[df_fov['Mouse_Model'] == 'Q111'].copy()

    # Create figure
    fig = plt.figure(figsize=(20, 26), dpi=FIGURE_DPI)
    gs = fig.add_gridspec(4, 2, hspace=0.40, wspace=0.30,
                          left=0.08, right=0.95, top=0.96, bottom=0.04)

    # Color scheme
    color_cortex = '#3498db'  # Blue
    color_striatum = '#e74c3c'  # Red
    color_mhtt1a = '#2ecc71'  # Green
    color_full = '#f39c12'  # Orange

    # Get unique mouse IDs and ages for color mapping
    mouse_ids = sorted(df_q111['Mouse_ID'].unique())
    ages = sorted(df_q111['Age'].unique())

    # Create age-based color palette
    age_colors = {2.0: '#3498db', 6.0: '#9b59b6', 12.0: '#e74c3c'}

    # ══════════════════════════════════════════════════════════════════════
    # ROW 1: TOTAL mRNA PER CELL
    # ══════════════════════════════════════════════════════════════════════

    # Left panel: Cortex vs Striatum by channel
    ax1 = fig.add_subplot(gs[0, 0])

    # Prepare data for grouped box plot
    plot_data = []
    for region in ['Cortex', 'Striatum']:
        for channel in ['HTT1a', 'fl-HTT']:
            subset = df_q111[(df_q111['Region'] == region) &
                            (df_q111['Channel'] == channel)]
            for val in subset['Total_mRNA_per_Cell']:
                plot_data.append({
                    'Region': region,
                    'Channel': channel,
                    'Value': val
                })

    df_plot = pd.DataFrame(plot_data)

    # Create grouped box plot
    positions = []
    labels = []
    bp_data = []
    colors = []

    for i, region in enumerate(['Cortex', 'Striatum']):
        for j, channel in enumerate(['HTT1a', 'fl-HTT']):
            subset = df_plot[(df_plot['Region'] == region) &
                            (df_plot['Channel'] == channel)]
            bp_data.append(subset['Value'].values)
            pos = i * 3 + j
            positions.append(pos)
            labels.append(f"{region}\n{channel}")

            # Color based on channel
            if channel == 'HTT1a':
                colors.append(color_mhtt1a)
            else:
                colors.append(color_full)

    bp = ax1.boxplot(bp_data, positions=positions, widths=0.6,
                     patch_artist=True, showfliers=False,
                     medianprops=dict(color='black', linewidth=2),
                     boxprops=dict(linewidth=1.5),
                     whiskerprops=dict(linewidth=1.5),
                     capprops=dict(linewidth=1.5))

    # Color boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Add individual points
    for i, (pos, data, color) in enumerate(zip(positions, bp_data, colors)):
        x = np.random.normal(pos, 0.04, size=len(data))
        ax1.scatter(x, data, alpha=0.3, s=20, color=color, edgecolor='none')

    ax1.set_xticks(positions)
    ax1.set_xticklabels(labels, fontsize=9, rotation=0, ha='center')
    ax1.set_ylabel('Total mRNA per Cell', fontsize=11, fontweight='bold')
    ax1.set_title('A) Total mRNA Expression: Cortex vs Striatum',
                  fontsize=12, fontweight='bold', pad=10)
    ax1.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax1.set_xlim(-0.5, 5.5)

    # Add statistics
    cortex_mhtt1a = df_q111[(df_q111['Region'] == 'Cortex') &
                            (df_q111['Channel'] == 'HTT1a')]['Total_mRNA_per_Cell']
    striatum_mhtt1a = df_q111[(df_q111['Region'] == 'Striatum') &
                              (df_q111['Channel'] == 'HTT1a')]['Total_mRNA_per_Cell']
    cortex_full = df_q111[(df_q111['Region'] == 'Cortex') &
                          (df_q111['Channel'] == 'fl-HTT')]['Total_mRNA_per_Cell']
    striatum_full = df_q111[(df_q111['Region'] == 'Striatum') &
                            (df_q111['Channel'] == 'fl-HTT')]['Total_mRNA_per_Cell']

    _, p_mhtt1a = mannwhitneyu(cortex_mhtt1a, striatum_mhtt1a, alternative='two-sided')
    _, p_full = mannwhitneyu(cortex_full, striatum_full, alternative='two-sided')

    ax1.text(0.02, 0.85, f"HTT1a: Cortex vs Striatum p={p_mhtt1a:.3e}\n"
                         f"Full-length: Cortex vs Striatum p={p_full:.3e}",
             transform=ax1.transAxes, fontsize=8, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Right panel: Per mouse ID
    ax2 = fig.add_subplot(gs[0, 1])

    # Aggregate by mouse ID, region, and channel
    mouse_stats = []
    for mouse_id in mouse_ids:
        for region in ['Cortex', 'Striatum']:
            for channel in ['HTT1a', 'fl-HTT']:
                subset = df_q111[(df_q111['Mouse_ID'] == mouse_id) &
                                (df_q111['Region'] == region) &
                                (df_q111['Channel'] == channel)]
                if len(subset) > 0:
                    age = subset['Age'].iloc[0]
                    mean_val = subset['Total_mRNA_per_Cell'].mean()
                    sem_val = subset['Total_mRNA_per_Cell'].sem()
                    mouse_stats.append({
                        'Mouse_ID': mouse_id,
                        'Age': age,
                        'Region': region,
                        'Channel': channel,
                        'Mean': mean_val,
                        'SEM': sem_val
                    })

    df_mouse_stats = pd.DataFrame(mouse_stats)

    # Plot bars grouped by mouse ID
    x_positions = np.arange(len(mouse_ids))
    bar_width = 0.18

    for i, region in enumerate(['Cortex', 'Striatum']):
        for j, channel in enumerate(['HTT1a', 'fl-HTT']):
            offset = (i * 2 + j - 1.5) * bar_width

            means = []
            sems = []
            bar_colors = []

            for mouse_id in mouse_ids:
                subset = df_mouse_stats[(df_mouse_stats['Mouse_ID'] == mouse_id) &
                                       (df_mouse_stats['Region'] == region) &
                                       (df_mouse_stats['Channel'] == channel)]
                if len(subset) > 0:
                    means.append(subset['Mean'].iloc[0])
                    sems.append(subset['SEM'].iloc[0])
                    age = subset['Age'].iloc[0]
                    bar_colors.append(age_colors.get(age, '#95a5a6'))
                else:
                    means.append(0)
                    sems.append(0)
                    bar_colors.append('#95a5a6')

            # Determine base color
            if channel == 'HTT1a':
                base_color = color_mhtt1a
            else:
                base_color = color_full

            # Make cortex lighter and striatum darker
            if region == 'Cortex':
                alpha = 0.5
            else:
                alpha = 0.9

            ax2.bar(x_positions + offset, means, bar_width,
                   label=f"{region} - {channel}",
                   color=base_color, alpha=alpha, edgecolor='black', linewidth=0.5,
                   yerr=sems, capsize=3, error_kw={'linewidth': 1})

    ax2.set_xlabel('Mouse ID', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Total mRNA per Cell (mean ± SEM)', fontsize=11, fontweight='bold')
    ax2.set_title('B) Total mRNA Expression: Per Mouse ID',
                  fontsize=12, fontweight='bold', pad=10)
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels([mid.replace('Q111 ', '') for mid in mouse_ids],
                        rotation=45, ha='right', fontsize=8)
    ax2.legend(loc='upper left', fontsize=7, ncol=2)
    ax2.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)

    # ══════════════════════════════════════════════════════════════════════
    # ROW 2: SINGLE mRNA PER CELL
    # ══════════════════════════════════════════════════════════════════════

    # Left panel: Cortex vs Striatum by channel
    ax3 = fig.add_subplot(gs[1, 0])

    # Prepare data
    plot_data = []
    for region in ['Cortex', 'Striatum']:
        for channel in ['HTT1a', 'fl-HTT']:
            subset = df_q111[(df_q111['Region'] == region) &
                            (df_q111['Channel'] == channel)]
            for val in subset['Single_mRNA_per_Cell']:
                plot_data.append({
                    'Region': region,
                    'Channel': channel,
                    'Value': val
                })

    df_plot = pd.DataFrame(plot_data)

    # Create grouped box plot
    bp_data = []
    colors = []

    for i, region in enumerate(['Cortex', 'Striatum']):
        for j, channel in enumerate(['HTT1a', 'fl-HTT']):
            subset = df_plot[(df_plot['Region'] == region) &
                            (df_plot['Channel'] == channel)]
            bp_data.append(subset['Value'].values)

            if channel == 'HTT1a':
                colors.append(color_mhtt1a)
            else:
                colors.append(color_full)

    bp = ax3.boxplot(bp_data, positions=positions, widths=0.6,
                     patch_artist=True, showfliers=False,
                     medianprops=dict(color='black', linewidth=2),
                     boxprops=dict(linewidth=1.5),
                     whiskerprops=dict(linewidth=1.5),
                     capprops=dict(linewidth=1.5))

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    for i, (pos, data, color) in enumerate(zip(positions, bp_data, colors)):
        x = np.random.normal(pos, 0.04, size=len(data))
        ax3.scatter(x, data, alpha=0.3, s=20, color=color, edgecolor='none')

    ax3.set_xticks(positions)
    ax3.set_xticklabels(labels, fontsize=9, rotation=0, ha='center')
    ax3.set_ylabel('Single mRNA per Cell', fontsize=11, fontweight='bold')
    ax3.set_title('C) Single mRNA Expression: Cortex vs Striatum',
                  fontsize=12, fontweight='bold', pad=10)
    ax3.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax3.set_xlim(-0.5, 5.5)

    # Add statistics
    cortex_mhtt1a = df_q111[(df_q111['Region'] == 'Cortex') &
                            (df_q111['Channel'] == 'HTT1a')]['Single_mRNA_per_Cell']
    striatum_mhtt1a = df_q111[(df_q111['Region'] == 'Striatum') &
                              (df_q111['Channel'] == 'HTT1a')]['Single_mRNA_per_Cell']
    cortex_full = df_q111[(df_q111['Region'] == 'Cortex') &
                          (df_q111['Channel'] == 'fl-HTT')]['Single_mRNA_per_Cell']
    striatum_full = df_q111[(df_q111['Region'] == 'Striatum') &
                            (df_q111['Channel'] == 'fl-HTT')]['Single_mRNA_per_Cell']

    _, p_mhtt1a = mannwhitneyu(cortex_mhtt1a, striatum_mhtt1a, alternative='two-sided')
    _, p_full = mannwhitneyu(cortex_full, striatum_full, alternative='two-sided')

    ax3.text(0.02, 0.85, f"HTT1a: Cortex vs Striatum p={p_mhtt1a:.3e}\n"
                         f"Full-length: Cortex vs Striatum p={p_full:.3e}",
             transform=ax3.transAxes, fontsize=8, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Right panel: Per mouse ID
    ax4 = fig.add_subplot(gs[1, 1])

    # Aggregate by mouse ID
    mouse_stats = []
    for mouse_id in mouse_ids:
        for region in ['Cortex', 'Striatum']:
            for channel in ['HTT1a', 'fl-HTT']:
                subset = df_q111[(df_q111['Mouse_ID'] == mouse_id) &
                                (df_q111['Region'] == region) &
                                (df_q111['Channel'] == channel)]
                if len(subset) > 0:
                    age = subset['Age'].iloc[0]
                    mean_val = subset['Single_mRNA_per_Cell'].mean()
                    sem_val = subset['Single_mRNA_per_Cell'].sem()
                    mouse_stats.append({
                        'Mouse_ID': mouse_id,
                        'Age': age,
                        'Region': region,
                        'Channel': channel,
                        'Mean': mean_val,
                        'SEM': sem_val
                    })

    df_mouse_stats = pd.DataFrame(mouse_stats)

    for i, region in enumerate(['Cortex', 'Striatum']):
        for j, channel in enumerate(['HTT1a', 'fl-HTT']):
            offset = (i * 2 + j - 1.5) * bar_width

            means = []
            sems = []

            for mouse_id in mouse_ids:
                subset = df_mouse_stats[(df_mouse_stats['Mouse_ID'] == mouse_id) &
                                       (df_mouse_stats['Region'] == region) &
                                       (df_mouse_stats['Channel'] == channel)]
                if len(subset) > 0:
                    means.append(subset['Mean'].iloc[0])
                    sems.append(subset['SEM'].iloc[0])
                else:
                    means.append(0)
                    sems.append(0)

            if channel == 'HTT1a':
                base_color = color_mhtt1a
            else:
                base_color = color_full

            if region == 'Cortex':
                alpha = 0.5
            else:
                alpha = 0.9

            ax4.bar(x_positions + offset, means, bar_width,
                   color=base_color, alpha=alpha, edgecolor='black', linewidth=0.5,
                   yerr=sems, capsize=3, error_kw={'linewidth': 1})

    ax4.set_xlabel('Mouse ID', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Single mRNA per Cell (mean ± SEM)', fontsize=11, fontweight='bold')
    ax4.set_title('D) Single mRNA Expression: Per Mouse ID',
                  fontsize=12, fontweight='bold', pad=10)
    ax4.set_xticks(x_positions)
    ax4.set_xticklabels([mid.replace('Q111 ', '') for mid in mouse_ids],
                        rotation=45, ha='right', fontsize=8)
    ax4.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)

    # ══════════════════════════════════════════════════════════════════════
    # ROW 3: CLUSTERED mRNA PER CELL
    # ══════════════════════════════════════════════════════════════════════

    # Left panel: Cortex vs Striatum by channel
    ax5 = fig.add_subplot(gs[2, 0])

    # Prepare data
    plot_data = []
    for region in ['Cortex', 'Striatum']:
        for channel in ['HTT1a', 'fl-HTT']:
            subset = df_q111[(df_q111['Region'] == region) &
                            (df_q111['Channel'] == channel)]
            for val in subset['Clustered_mRNA_per_Cell']:
                plot_data.append({
                    'Region': region,
                    'Channel': channel,
                    'Value': val
                })

    df_plot = pd.DataFrame(plot_data)

    # Create grouped box plot
    bp_data = []
    colors = []

    for i, region in enumerate(['Cortex', 'Striatum']):
        for j, channel in enumerate(['HTT1a', 'fl-HTT']):
            subset = df_plot[(df_plot['Region'] == region) &
                            (df_plot['Channel'] == channel)]
            bp_data.append(subset['Value'].values)

            if channel == 'HTT1a':
                colors.append(color_mhtt1a)
            else:
                colors.append(color_full)

    bp = ax5.boxplot(bp_data, positions=positions, widths=0.6,
                     patch_artist=True, showfliers=False,
                     medianprops=dict(color='black', linewidth=2),
                     boxprops=dict(linewidth=1.5),
                     whiskerprops=dict(linewidth=1.5),
                     capprops=dict(linewidth=1.5))

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    for i, (pos, data, color) in enumerate(zip(positions, bp_data, colors)):
        x = np.random.normal(pos, 0.04, size=len(data))
        ax5.scatter(x, data, alpha=0.3, s=20, color=color, edgecolor='none')

    ax5.set_xticks(positions)
    ax5.set_xticklabels(labels, fontsize=9, rotation=0, ha='center')
    ax5.set_ylabel('Clustered mRNA per Cell', fontsize=11, fontweight='bold')
    ax5.set_title('E) Clustered mRNA Expression: Cortex vs Striatum',
                  fontsize=12, fontweight='bold', pad=10)
    ax5.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax5.set_xlim(-0.5, 5.5)

    # Add statistics
    cortex_mhtt1a = df_q111[(df_q111['Region'] == 'Cortex') &
                            (df_q111['Channel'] == 'HTT1a')]['Clustered_mRNA_per_Cell']
    striatum_mhtt1a = df_q111[(df_q111['Region'] == 'Striatum') &
                              (df_q111['Channel'] == 'HTT1a')]['Clustered_mRNA_per_Cell']
    cortex_full = df_q111[(df_q111['Region'] == 'Cortex') &
                          (df_q111['Channel'] == 'fl-HTT')]['Clustered_mRNA_per_Cell']
    striatum_full = df_q111[(df_q111['Region'] == 'Striatum') &
                            (df_q111['Channel'] == 'fl-HTT')]['Clustered_mRNA_per_Cell']

    _, p_mhtt1a = mannwhitneyu(cortex_mhtt1a, striatum_mhtt1a, alternative='two-sided')
    _, p_full = mannwhitneyu(cortex_full, striatum_full, alternative='two-sided')

    ax5.text(0.02, 0.85, f"HTT1a: Cortex vs Striatum p={p_mhtt1a:.3e}\n"
                         f"Full-length: Cortex vs Striatum p={p_full:.3e}",
             transform=ax5.transAxes, fontsize=8, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Right panel: Per mouse ID
    ax6 = fig.add_subplot(gs[2, 1])

    # Aggregate by mouse ID
    mouse_stats = []
    for mouse_id in mouse_ids:
        for region in ['Cortex', 'Striatum']:
            for channel in ['HTT1a', 'fl-HTT']:
                subset = df_q111[(df_q111['Mouse_ID'] == mouse_id) &
                                (df_q111['Region'] == region) &
                                (df_q111['Channel'] == channel)]
                if len(subset) > 0:
                    age = subset['Age'].iloc[0]
                    mean_val = subset['Clustered_mRNA_per_Cell'].mean()
                    sem_val = subset['Clustered_mRNA_per_Cell'].sem()
                    mouse_stats.append({
                        'Mouse_ID': mouse_id,
                        'Age': age,
                        'Region': region,
                        'Channel': channel,
                        'Mean': mean_val,
                        'SEM': sem_val
                    })

    df_mouse_stats = pd.DataFrame(mouse_stats)

    for i, region in enumerate(['Cortex', 'Striatum']):
        for j, channel in enumerate(['HTT1a', 'fl-HTT']):
            offset = (i * 2 + j - 1.5) * bar_width

            means = []
            sems = []

            for mouse_id in mouse_ids:
                subset = df_mouse_stats[(df_mouse_stats['Mouse_ID'] == mouse_id) &
                                       (df_mouse_stats['Region'] == region) &
                                       (df_mouse_stats['Channel'] == channel)]
                if len(subset) > 0:
                    means.append(subset['Mean'].iloc[0])
                    sems.append(subset['SEM'].iloc[0])
                else:
                    means.append(0)
                    sems.append(0)

            if channel == 'HTT1a':
                base_color = color_mhtt1a
            else:
                base_color = color_full

            if region == 'Cortex':
                alpha = 0.5
            else:
                alpha = 0.9

            ax6.bar(x_positions + offset, means, bar_width,
                   color=base_color, alpha=alpha, edgecolor='black', linewidth=0.5,
                   yerr=sems, capsize=3, error_kw={'linewidth': 1})

    ax6.set_xlabel('Mouse ID', fontsize=11, fontweight='bold')
    ax6.set_ylabel('Clustered mRNA per Cell (mean ± SEM)', fontsize=11, fontweight='bold')
    ax6.set_title('F) Clustered mRNA Expression: Per Mouse ID',
                  fontsize=12, fontweight='bold', pad=10)
    ax6.set_xticks(x_positions)
    ax6.set_xticklabels([mid.replace('Q111 ', '') for mid in mouse_ids],
                        rotation=45, ha='right', fontsize=8)
    ax6.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)

    # ══════════════════════════════════════════════════════════════════════
    # ROW 4: COMPREHENSIVE SUMMARY BY MOUSE ID
    # ══════════════════════════════════════════════════════════════════════

    # Left panel: Stacked bar showing single vs clustered contribution
    ax7 = fig.add_subplot(gs[3, 0])

    # Aggregate data
    summary_stats = []
    for mouse_id in mouse_ids:
        for region in ['Cortex', 'Striatum']:
            for channel in ['HTT1a', 'fl-HTT']:
                subset = df_q111[(df_q111['Mouse_ID'] == mouse_id) &
                                (df_q111['Region'] == region) &
                                (df_q111['Channel'] == channel)]
                if len(subset) > 0:
                    age = subset['Age'].iloc[0]
                    single_mean = subset['Single_mRNA_per_Cell'].mean()
                    clustered_mean = subset['Clustered_mRNA_per_Cell'].mean()
                    total_mean = subset['Total_mRNA_per_Cell'].mean()
                    summary_stats.append({
                        'Mouse_ID': mouse_id,
                        'Age': age,
                        'Region': region,
                        'Channel': channel,
                        'Single': single_mean,
                        'Clustered': clustered_mean,
                        'Total': total_mean
                    })

    df_summary = pd.DataFrame(summary_stats)

    # Create stacked bar plot - focus on one condition for clarity
    # Show HTT1a in Cortex and Striatum
    subset_cortex = df_summary[(df_summary['Channel'] == 'HTT1a') &
                               (df_summary['Region'] == 'Cortex')]
    subset_striatum = df_summary[(df_summary['Channel'] == 'HTT1a') &
                                 (df_summary['Region'] == 'Striatum')]

    x_pos = np.arange(len(mouse_ids))
    width = 0.35

    # Cortex bars
    single_cortex = [subset_cortex[subset_cortex['Mouse_ID'] == mid]['Single'].iloc[0]
                     if len(subset_cortex[subset_cortex['Mouse_ID'] == mid]) > 0 else 0
                     for mid in mouse_ids]
    clustered_cortex = [subset_cortex[subset_cortex['Mouse_ID'] == mid]['Clustered'].iloc[0]
                        if len(subset_cortex[subset_cortex['Mouse_ID'] == mid]) > 0 else 0
                        for mid in mouse_ids]

    # Striatum bars
    single_striatum = [subset_striatum[subset_striatum['Mouse_ID'] == mid]['Single'].iloc[0]
                       if len(subset_striatum[subset_striatum['Mouse_ID'] == mid]) > 0 else 0
                       for mid in mouse_ids]
    clustered_striatum = [subset_striatum[subset_striatum['Mouse_ID'] == mid]['Clustered'].iloc[0]
                          if len(subset_striatum[subset_striatum['Mouse_ID'] == mid]) > 0 else 0
                          for mid in mouse_ids]

    # Plot stacked bars
    ax7.bar(x_pos - width/2, single_cortex, width, label='Single (Cortex)',
           color='#3498db', alpha=0.5, edgecolor='black', linewidth=0.5)
    ax7.bar(x_pos - width/2, clustered_cortex, width, bottom=single_cortex,
           label='Clustered (Cortex)', color='#3498db', alpha=0.9,
           edgecolor='black', linewidth=0.5)

    ax7.bar(x_pos + width/2, single_striatum, width, label='Single (Striatum)',
           color='#e74c3c', alpha=0.5, edgecolor='black', linewidth=0.5)
    ax7.bar(x_pos + width/2, clustered_striatum, width, bottom=single_striatum,
           label='Clustered (Striatum)', color='#e74c3c', alpha=0.9,
           edgecolor='black', linewidth=0.5)

    ax7.set_xlabel('Mouse ID', fontsize=11, fontweight='bold')
    ax7.set_ylabel('HTT1a mRNA per Cell', fontsize=11, fontweight='bold')
    ax7.set_title('G) HTT1a: Single vs Clustered Contribution by Mouse ID',
                  fontsize=12, fontweight='bold', pad=10)
    ax7.set_xticks(x_pos)
    ax7.set_xticklabels([mid.replace('Q111 ', '') for mid in mouse_ids],
                        rotation=45, ha='right', fontsize=8)
    ax7.legend(loc='upper left', fontsize=7, ncol=2)
    ax7.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)

    # Right panel: Similar for fl-HTT
    ax8 = fig.add_subplot(gs[3, 1])

    subset_cortex = df_summary[(df_summary['Channel'] == 'fl-HTT') &
                               (df_summary['Region'] == 'Cortex')]
    subset_striatum = df_summary[(df_summary['Channel'] == 'fl-HTT') &
                                 (df_summary['Region'] == 'Striatum')]

    single_cortex = [subset_cortex[subset_cortex['Mouse_ID'] == mid]['Single'].iloc[0]
                     if len(subset_cortex[subset_cortex['Mouse_ID'] == mid]) > 0 else 0
                     for mid in mouse_ids]
    clustered_cortex = [subset_cortex[subset_cortex['Mouse_ID'] == mid]['Clustered'].iloc[0]
                        if len(subset_cortex[subset_cortex['Mouse_ID'] == mid]) > 0 else 0
                        for mid in mouse_ids]

    single_striatum = [subset_striatum[subset_striatum['Mouse_ID'] == mid]['Single'].iloc[0]
                       if len(subset_striatum[subset_striatum['Mouse_ID'] == mid]) > 0 else 0
                       for mid in mouse_ids]
    clustered_striatum = [subset_striatum[subset_striatum['Mouse_ID'] == mid]['Clustered'].iloc[0]
                          if len(subset_striatum[subset_striatum['Mouse_ID'] == mid]) > 0 else 0
                          for mid in mouse_ids]

    ax8.bar(x_pos - width/2, single_cortex, width, label='Single (Cortex)',
           color='#3498db', alpha=0.5, edgecolor='black', linewidth=0.5)
    ax8.bar(x_pos - width/2, clustered_cortex, width, bottom=single_cortex,
           label='Clustered (Cortex)', color='#3498db', alpha=0.9,
           edgecolor='black', linewidth=0.5)

    ax8.bar(x_pos + width/2, single_striatum, width, label='Single (Striatum)',
           color='#e74c3c', alpha=0.5, edgecolor='black', linewidth=0.5)
    ax8.bar(x_pos + width/2, clustered_striatum, width, bottom=single_striatum,
           label='Clustered (Striatum)', color='#e74c3c', alpha=0.9,
           edgecolor='black', linewidth=0.5)

    ax8.set_xlabel('Mouse ID', fontsize=11, fontweight='bold')
    ax8.set_ylabel('fl-fl-HTT mRNA per Cell', fontsize=11, fontweight='bold')
    ax8.set_title('H) fl-HTT: Single vs Clustered Contribution by Mouse ID',
                  fontsize=12, fontweight='bold', pad=10)
    ax8.set_xticks(x_pos)
    ax8.set_xticklabels([mid.replace('Q111 ', '') for mid in mouse_ids],
                        rotation=45, ha='right', fontsize=8)
    ax8.legend(loc='upper left', fontsize=7, ncol=2)
    ax8.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)

    # ══════════════════════════════════════════════════════════════════════
    # GENERATE COMPREHENSIVE CAPTION
    # ══════════════════════════════════════════════════════════════════════

    caption_lines = []
    caption_lines.append("=" * 80)
    caption_lines.append("FIGURE: Comprehensive mRNA Expression Analysis (Q111 mice)")
    caption_lines.append("=" * 80)
    caption_lines.append("")

    caption_lines.append("OVERVIEW:")
    caption_lines.append("-" * 80)
    caption_lines.append("This figure presents a detailed analysis of HTT1a and fl-HTT transcript")
    caption_lines.append("expression in Q111 transgenic mice across cortical and striatal regions. The figure")
    caption_lines.append("is organized into three rows, each examining a different quantification metric:")
    caption_lines.append("total mRNA, single mRNA, and clustered mRNA per cell. For each metric, regional")
    caption_lines.append("comparisons (cortex vs striatum) and per-mouse breakdowns are provided.")
    caption_lines.append("")

    # Dataset statistics - Q111 only for this figure
    df_q111 = df_fov[df_fov['Mouse_Model'] == 'Q111']

    caption_lines.append("DATASET STATISTICS:")
    caption_lines.append("-" * 80)
    caption_lines.append(f"Total Q111 FOVs: {len(df_q111)}")
    caption_lines.append(f"  Cortex: {len(df_q111[df_q111['Region'] == 'Cortex'])} FOVs")
    caption_lines.append(f"  Striatum: {len(df_q111[df_q111['Region'] == 'Striatum'])} FOVs")
    caption_lines.append("")

    # Mouse IDs
    q111_mice = sorted(df_q111['Mouse_ID'].unique())
    caption_lines.append(f"Q111 mice analyzed (n={len(q111_mice)}):")
    for mouse_id in q111_mice:
        mouse_fovs = len(df_q111[df_q111['Mouse_ID'] == mouse_id])
        age = df_q111[df_q111['Mouse_ID'] == mouse_id]['Age'].iloc[0]
        caption_lines.append(f"  {mouse_id}: {mouse_fovs} FOVs ({age:.1f} months)")
    caption_lines.append("")

    caption_lines.append("PANEL ORGANIZATION:")
    caption_lines.append("-" * 80)
    caption_lines.append("Row 1 - TOTAL mRNA PER CELL (Panels A-B):")
    caption_lines.append("  Panel A: Box plots comparing cortex vs striatum for HTT1a and full-length")
    caption_lines.append("  Panel B: Bar plots showing per-mouse expression levels")
    caption_lines.append("")
    caption_lines.append("Row 2 - SINGLE mRNA PER CELL (Panels C-D):")
    caption_lines.append("  Panel C: Box plots comparing cortex vs striatum for HTT1a and full-length")
    caption_lines.append("  Panel D: Bar plots showing per-mouse expression levels")
    caption_lines.append("")
    caption_lines.append("Row 3 - CLUSTERED mRNA PER CELL (Panels E-F):")
    caption_lines.append("  Panel E: Box plots comparing cortex vs striatum for HTT1a and full-length")
    caption_lines.append("  Panel F: Bar plots showing per-mouse expression levels")
    caption_lines.append("")

    # Detailed statistics per metric
    metrics = [
        ('Total_mRNA_per_Cell', 'Total mRNA', 'A-B'),
        ('Single_mRNA_per_Cell', 'Single mRNA', 'C-D'),
        ('Clustered_mRNA_per_Cell', 'Clustered mRNA', 'E-F')
    ]

    for metric_col, metric_name, panels in metrics:
        caption_lines.append(f"{metric_name.upper()} STATISTICS (Panels {panels}):")
        caption_lines.append("-" * 80)

        for region in ['Cortex', 'Striatum']:
            caption_lines.append(f"\n{region}:")
            for channel in ['HTT1a', 'fl-HTT']:
                subset = df_q111[
                    (df_q111['Region'] == region) &
                    (df_q111['Channel'] == channel)
                ][metric_col]

                if len(subset) > 0:
                    mean_val = subset.mean()
                    std_val = subset.std()
                    median_val = subset.median()
                    q25 = subset.quantile(0.25)
                    q75 = subset.quantile(0.75)
                    n_val = len(subset)

                    caption_lines.append(
                        f"  {channel}: n={n_val}, mean={mean_val:.2f}±{std_val:.2f}, "
                        f"median={median_val:.2f}, IQR=[{q25:.2f}, {q75:.2f}]"
                    )

        # Per-mouse statistics
        caption_lines.append(f"\nPer-Mouse Means (Panel {panels.split('-')[1]}):")
        for mouse_id in sorted(df_q111['Mouse_ID'].unique()):
            mouse_subset = df_q111[df_q111['Mouse_ID'] == mouse_id]
            for region in ['Cortex', 'Striatum']:
                region_subset = mouse_subset[mouse_subset['Region'] == region]
                for channel in ['HTT1a', 'fl-HTT']:
                    channel_subset = region_subset[region_subset['Channel'] == channel]
                    if len(channel_subset) > 0:
                        mean_val = channel_subset[metric_col].mean()
                        sem_val = channel_subset[metric_col].sem()
                        n_val = len(channel_subset)
                        caption_lines.append(
                            f"  {mouse_id} | {region} | {channel}: "
                            f"mean={mean_val:.2f}±{sem_val:.2f} (n={n_val})"
                        )
        caption_lines.append("")

    caption_lines.append("COLOR SCHEME:")
    caption_lines.append("-" * 80)
    caption_lines.append("HTT1a: Green")
    caption_lines.append("fl-HTT: Orange")
    caption_lines.append("Cortex: Higher opacity (alpha=0.7)")
    caption_lines.append("Striatum: Lower opacity (alpha=0.5)")
    caption_lines.append("")

    caption_lines.append("QUALITY CONTROL:")
    caption_lines.append("-" * 80)
    caption_lines.append(f"Excluded slides (n={len(EXCLUDED_SLIDES)}): {', '.join(sorted(EXCLUDED_SLIDES))}")
    caption_lines.append(f"  (Slides excluded due to poor UBC positive control expression indicating technical failures)")
    caption_lines.append(f"CV threshold for cluster filtering: CV >= {CV_THRESHOLD}")
    caption_lines.append(f"Minimum nuclei per FOV: {MIN_NUCLEI_THRESHOLD}")
    caption_lines.append(f"Intensity threshold: Per-slide, determined from negative control at quantile={QUANTILE_NEGATIVE_CONTROL}, max PFA={MAX_PFA}")
    caption_lines.append("")

    caption_lines.append("METHODOLOGY:")
    caption_lines.append("-" * 80)
    caption_lines.append("Total mRNA per cell:")
    caption_lines.append("  Total = N_spots + (I_cluster_total / I_single_peak) / N_nuclei")
    caption_lines.append("  Where I_single_peak is slide-specific peak intensity from KDE")
    caption_lines.append("")
    caption_lines.append("Single mRNA per cell:")
    caption_lines.append("  Single = N_spots / N_nuclei")
    caption_lines.append("  Represents diffraction-limited individual transcripts")
    caption_lines.append("")
    caption_lines.append("Clustered mRNA per cell:")
    caption_lines.append("  Clustered = Total - Single")
    caption_lines.append("  Represents aggregated/high-density transcript populations")
    caption_lines.append("")
    caption_lines.append("Peak intensity normalization:")
    caption_lines.append("  - Kernel Density Estimation (KDE) with Scott's bandwidth")
    caption_lines.append("  - Slide-specific normalization accounts for technical variation")
    caption_lines.append("  - Minimum 50 spots required for reliable peak estimation")
    caption_lines.append("")

    caption_lines.append("STATISTICAL NOTES:")
    caption_lines.append("-" * 80)
    caption_lines.append("Box plots (Panels A, C, E):")
    caption_lines.append("  - Box: Interquartile range (IQR, 25th-75th percentile)")
    caption_lines.append("  - Median: Black horizontal line")
    caption_lines.append("  - Whiskers: Extend to 1.5×IQR")
    caption_lines.append("  - Individual points: Overlaid with transparency for distribution visualization")
    caption_lines.append("")
    caption_lines.append("Bar plots (Panels B, D, F):")
    caption_lines.append("  - Height: Mean expression across FOVs for each mouse")
    caption_lines.append("  - Error bars: SEM (standard error of the mean)")
    caption_lines.append("  - Each mouse contributes multiple FOVs")
    caption_lines.append("")

    caption_lines.append("=" * 80)
    caption_lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    caption_lines.append("=" * 80)

    # Save caption
    caption_path = OUTPUT_DIR / "fig_expression_analysis_q111_caption.txt"
    with open(caption_path, 'w') as f:
        f.write('\n'.join(caption_lines))
    print(f"  Saved caption: {caption_path}")

    # Save figure
    for fmt in ['png', 'svg', 'pdf']:
        output_path = OUTPUT_DIR / f"fig_expression_analysis_q111.{fmt}"
        fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"  Saved: {output_path}")

    plt.close(fig)

    return df_summary


def create_total_expression_figure(df_fov):
    """
    Create a focused figure showing just total mRNA expression
    across regions and channels with cleaner visualization.
    """

    # Filter for Q111 mice only
    df_q111 = df_fov[df_fov['Mouse_Model'] == 'Q111'].copy()

    # Color scheme
    color_mhtt1a = '#2ecc71'  # Green
    color_full = '#f39c12'  # Orange

    # Create figure with 2 panels
    fig = plt.figure(figsize=(16, 6), dpi=FIGURE_DPI)
    gs = fig.add_gridspec(1, 2, hspace=0.3, wspace=0.25,
                          left=0.08, right=0.95, top=0.90, bottom=0.15)

    # ══════════════════════════════════════════════════════════════════════
    # PANEL A: Regional Comparison (Cortex vs Striatum)
    # ══════════════════════════════════════════════════════════════════════

    ax1 = fig.add_subplot(gs[0, 0])

    # Prepare data for violin plots
    plot_data = []
    for region in ['Cortex', 'Striatum']:
        for channel in ['HTT1a', 'fl-HTT']:
            subset = df_q111[(df_q111['Region'] == region) &
                            (df_q111['Channel'] == channel)]
            for val in subset['Total_mRNA_per_Cell']:
                plot_data.append({
                    'Region': region,
                    'Channel': channel,
                    'Value': val
                })

    df_plot = pd.DataFrame(plot_data)

    # Create violin plots
    positions = [0, 1, 3, 4]
    colors = [color_mhtt1a, color_full, color_mhtt1a, color_full]
    labels = ['HTT1a\nCortex', 'Full-length\nCortex',
              'HTT1a\nStriatum', 'Full-length\nStriatum']

    violin_data = []
    for i, (region, channel) in enumerate([('Cortex', 'HTT1a'),
                                            ('Cortex', 'fl-HTT'),
                                            ('Striatum', 'HTT1a'),
                                            ('Striatum', 'fl-HTT')]):
        subset = df_plot[(df_plot['Region'] == region) &
                        (df_plot['Channel'] == channel)]
        violin_data.append(subset['Value'].values)

    parts = ax1.violinplot(violin_data, positions=positions, widths=0.7,
                           showmeans=True, showmedians=True)

    # Color the violins
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
        pc.set_edgecolor('black')
        pc.set_linewidth(1.5)

    # Add scatter points
    for i, (pos, data, color) in enumerate(zip(positions, violin_data, colors)):
        # Subsample if too many points
        if len(data) > 500:
            indices = np.random.choice(len(data), 500, replace=False)
            data_plot = data[indices]
        else:
            data_plot = data

        x = np.random.normal(pos, 0.04, size=len(data_plot))
        ax1.scatter(x, data_plot, alpha=0.2, s=10, color=color, edgecolor='none')

    ax1.set_xticks(positions)
    ax1.set_xticklabels(labels, fontsize=10)
    ax1.set_ylabel('Total mRNA per Cell', fontsize=12, fontweight='bold')
    ax1.set_title('A) Total fl-HTT Expression: Regional Comparison',
                  fontsize=13, fontweight='bold', pad=15)
    ax1.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax1.set_xlim(-0.5, 4.5)

    # Add statistics
    cortex_mhtt1a = df_q111[(df_q111['Region'] == 'Cortex') &
                            (df_q111['Channel'] == 'HTT1a')]['Total_mRNA_per_Cell']
    striatum_mhtt1a = df_q111[(df_q111['Region'] == 'Striatum') &
                              (df_q111['Channel'] == 'HTT1a')]['Total_mRNA_per_Cell']
    cortex_full = df_q111[(df_q111['Region'] == 'Cortex') &
                          (df_q111['Channel'] == 'fl-HTT')]['Total_mRNA_per_Cell']
    striatum_full = df_q111[(df_q111['Region'] == 'Striatum') &
                            (df_q111['Channel'] == 'fl-HTT')]['Total_mRNA_per_Cell']

    _, p_mhtt1a = mannwhitneyu(cortex_mhtt1a, striatum_mhtt1a, alternative='two-sided')
    _, p_full = mannwhitneyu(cortex_full, striatum_full, alternative='two-sided')

    stats_text = f"HTT1a: Cortex vs Striatum\n"
    stats_text += f"  Cortex: {cortex_mhtt1a.mean():.1f} ± {cortex_mhtt1a.std():.1f} (N={len(cortex_mhtt1a)})\n"
    stats_text += f"  Striatum: {striatum_mhtt1a.mean():.1f} ± {striatum_mhtt1a.std():.1f} (N={len(striatum_mhtt1a)})\n"
    stats_text += f"  p = {p_mhtt1a:.3e}\n\n"
    stats_text += f"fl-HTT: Cortex vs Striatum\n"
    stats_text += f"  Cortex: {cortex_full.mean():.1f} ± {cortex_full.std():.1f} (N={len(cortex_full)})\n"
    stats_text += f"  Striatum: {striatum_full.mean():.1f} ± {striatum_full.std():.1f} (N={len(striatum_full)})\n"
    stats_text += f"  p = {p_full:.3e}"

    ax1.text(0.02, 0.98, stats_text,
             transform=ax1.transAxes, fontsize=7, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
             family='monospace')

    # ══════════════════════════════════════════════════════════════════════
    # PANEL B: Age Comparison
    # ══════════════════════════════════════════════════════════════════════

    ax2 = fig.add_subplot(gs[0, 1])

    # Group by age, region, and channel
    age_stats = []
    for age in sorted(df_q111['Age'].unique()):
        for region in ['Cortex', 'Striatum']:
            for channel in ['HTT1a', 'fl-HTT']:
                subset = df_q111[(df_q111['Age'] == age) &
                                (df_q111['Region'] == region) &
                                (df_q111['Channel'] == channel)]
                if len(subset) > 0:
                    age_stats.append({
                        'Age': age,
                        'Region': region,
                        'Channel': channel,
                        'Mean': subset['Total_mRNA_per_Cell'].mean(),
                        'SEM': subset['Total_mRNA_per_Cell'].sem(),
                        'N': len(subset)
                    })

    df_age = pd.DataFrame(age_stats)

    # Plot lines for each region-channel combination
    ages = sorted(df_q111['Age'].unique())
    x_pos = np.arange(len(ages))

    for i, region in enumerate(['Cortex', 'Striatum']):
        for j, channel in enumerate(['HTT1a', 'fl-HTT']):
            subset = df_age[(df_age['Region'] == region) &
                           (df_age['Channel'] == channel)]

            if channel == 'HTT1a':
                color = color_mhtt1a
                marker = 'o'
            else:
                color = color_full
                marker = 's'

            if region == 'Cortex':
                linestyle = '-'
                alpha = 0.9
            else:
                linestyle = '--'
                alpha = 0.7

            # Plot with error bars
            means = [subset[subset['Age'] == age]['Mean'].iloc[0]
                    if len(subset[subset['Age'] == age]) > 0 else np.nan
                    for age in ages]
            sems = [subset[subset['Age'] == age]['SEM'].iloc[0]
                   if len(subset[subset['Age'] == age]) > 0 else np.nan
                   for age in ages]

            ax2.errorbar(x_pos, means, yerr=sems,
                        label=f"{channel} ({region})",
                        color=color, marker=marker, markersize=8,
                        linestyle=linestyle, linewidth=2, alpha=alpha,
                        capsize=5, capthick=2)

    ax2.set_xlabel('Age (months)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Total mRNA per Cell (mean ± SEM)', fontsize=12, fontweight='bold')
    ax2.set_title('B) Total fl-HTT Expression: Age-Dependent Trends',
                  fontsize=13, fontweight='bold', pad=15)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f'{int(age)}' for age in ages], fontsize=11)
    ax2.legend(loc='upper left', fontsize=8, ncol=2)
    ax2.grid(axis='both', alpha=0.3, linestyle='--', linewidth=0.5)

    # ══════════════════════════════════════════════════════════════════════
    # GENERATE COMPREHENSIVE CAPTION
    # ══════════════════════════════════════════════════════════════════════

    caption_lines = []
    caption_lines.append("=" * 80)
    caption_lines.append("FIGURE: Total mRNA Expression Summary (Q111 mice)")
    caption_lines.append("=" * 80)
    caption_lines.append("")

    caption_lines.append("OVERVIEW:")
    caption_lines.append("-" * 80)
    caption_lines.append("This summary figure provides a streamlined view of total fl-HTT mRNA expression in")
    caption_lines.append("Q111 transgenic mice. Panel A shows regional comparisons (cortex vs striatum) for")
    caption_lines.append("both transcript types, while Panel B displays expression levels broken down by")
    caption_lines.append("individual mouse. This figure focuses exclusively on total mRNA (single +")
    caption_lines.append("clustered), the most comprehensive quantification metric.")
    caption_lines.append("")

    # Dataset statistics
    df_q111 = df_fov[df_fov['Mouse_Model'] == 'Q111']

    caption_lines.append("DATASET STATISTICS:")
    caption_lines.append("-" * 80)
    caption_lines.append(f"Total Q111 FOVs: {len(df_q111)}")
    caption_lines.append(f"  Cortex: {len(df_q111[df_q111['Region'] == 'Cortex'])} FOVs")
    caption_lines.append(f"  Striatum: {len(df_q111[df_q111['Region'] == 'Striatum'])} FOVs")
    caption_lines.append("")

    q111_mice = sorted(df_q111['Mouse_ID'].unique())
    caption_lines.append(f"Q111 mice (n={len(q111_mice)}): {', '.join([m.split()[-1] for m in q111_mice])}")
    caption_lines.append("")

    caption_lines.append("PANEL A - REGIONAL COMPARISON:")
    caption_lines.append("-" * 80)
    caption_lines.append("Box plots with overlaid scatter points comparing expression between cortex and")
    caption_lines.append("striatum for both HTT1a and fl-HTT transcripts.")
    caption_lines.append("")

    for region in ['Cortex', 'Striatum']:
        caption_lines.append(f"{region}:")
        for channel in ['HTT1a', 'fl-HTT']:
            subset = df_q111[
                (df_q111['Region'] == region) &
                (df_q111['Channel'] == channel)
            ]['Total_mRNA_per_Cell']

            if len(subset) > 0:
                mean_val = subset.mean()
                std_val = subset.std()
                median_val = subset.median()
                q25 = subset.quantile(0.25)
                q75 = subset.quantile(0.75)
                n_val = len(subset)

                caption_lines.append(
                    f"  {channel}: n={n_val}, mean={mean_val:.2f}±{std_val:.2f}, "
                    f"median={median_val:.2f}, IQR=[{q25:.2f}, {q75:.2f}]"
                )
    caption_lines.append("")

    # Statistical comparison
    cortex_mhtt1a = df_q111[
        (df_q111['Region'] == 'Cortex') &
        (df_q111['Channel'] == 'HTT1a')
    ]['Total_mRNA_per_Cell']
    striatum_mhtt1a = df_q111[
        (df_q111['Region'] == 'Striatum') &
        (df_q111['Channel'] == 'HTT1a')
    ]['Total_mRNA_per_Cell']
    cortex_full = df_q111[
        (df_q111['Region'] == 'Cortex') &
        (df_q111['Channel'] == 'fl-HTT')
    ]['Total_mRNA_per_Cell']
    striatum_full = df_q111[
        (df_q111['Region'] == 'Striatum') &
        (df_q111['Channel'] == 'fl-HTT')
    ]['Total_mRNA_per_Cell']

    caption_lines.append("Regional differences:")
    caption_lines.append(f"  HTT1a: Cortex mean={cortex_mhtt1a.mean():.2f}, Striatum mean={striatum_mhtt1a.mean():.2f}")
    caption_lines.append(f"    Fold change: {cortex_mhtt1a.mean()/striatum_mhtt1a.mean():.2f}x (Cortex/Striatum)")
    caption_lines.append(f"  full-length: Cortex mean={cortex_full.mean():.2f}, Striatum mean={striatum_full.mean():.2f}")
    caption_lines.append(f"    Fold change: {cortex_full.mean()/striatum_full.mean():.2f}x (Cortex/Striatum)")
    caption_lines.append("")

    caption_lines.append("PANEL B - PER-MOUSE BREAKDOWN:")
    caption_lines.append("-" * 80)
    caption_lines.append("Bar plots showing mean±SEM total mRNA expression for each individual mouse,")
    caption_lines.append("separately for cortex and striatum. This reveals inter-individual variability")
    caption_lines.append("and identifies potential outlier mice.")
    caption_lines.append("")

    for mouse_id in sorted(q111_mice):
        mouse_subset = df_q111[df_q111['Mouse_ID'] == mouse_id]
        age = mouse_subset['Age'].iloc[0]
        total_fovs = len(mouse_subset) // 2  # Divide by 2 for channels

        caption_lines.append(f"{mouse_id} ({age:.1f}mo, {total_fovs} FOVs):")

        for region in ['Cortex', 'Striatum']:
            region_subset = mouse_subset[mouse_subset['Region'] == region]
            for channel in ['HTT1a', 'fl-HTT']:
                channel_subset = region_subset[region_subset['Channel'] == channel]
                if len(channel_subset) > 0:
                    mean_val = channel_subset['Total_mRNA_per_Cell'].mean()
                    sem_val = channel_subset['Total_mRNA_per_Cell'].sem()
                    n_val = len(channel_subset)
                    caption_lines.append(
                        f"  {region} - {channel}: {mean_val:.2f}±{sem_val:.2f} (n={n_val} FOVs)"
                    )
    caption_lines.append("")

    caption_lines.append("COLOR SCHEME:")
    caption_lines.append("-" * 80)
    caption_lines.append("HTT1a: Green")
    caption_lines.append("fl-HTT: Orange")
    caption_lines.append("Panel A: Cortex (alpha=0.7), Striatum (alpha=0.5)")
    caption_lines.append("Panel B: Bars grouped by mouse, with cortex/striatum distinguished by alpha")
    caption_lines.append("")

    caption_lines.append("QUALITY CONTROL:")
    caption_lines.append("-" * 80)
    caption_lines.append(f"Excluded slides (n={len(EXCLUDED_SLIDES)}): {', '.join(sorted(EXCLUDED_SLIDES))}")
    caption_lines.append(f"  (Slides excluded due to poor UBC positive control expression indicating technical failures)")
    caption_lines.append(f"CV threshold for cluster filtering: CV >= {CV_THRESHOLD}")
    caption_lines.append(f"Minimum nuclei per FOV: {MIN_NUCLEI_THRESHOLD}")
    caption_lines.append(f"Intensity threshold: Per-slide, determined from negative control at quantile={QUANTILE_NEGATIVE_CONTROL}, max PFA={MAX_PFA}")
    caption_lines.append("")

    caption_lines.append("METHODOLOGY:")
    caption_lines.append("-" * 80)
    caption_lines.append("Total mRNA per cell:")
    caption_lines.append("  Total = N_spots + (I_cluster_total / I_single_peak) / N_nuclei")
    caption_lines.append("")
    caption_lines.append("Peak intensity normalization:")
    caption_lines.append("  - Slide-specific I_single_peak determined via KDE")
    caption_lines.append("  - Scott's bandwidth method for KDE")
    caption_lines.append("  - Minimum 50 spots required for peak estimation")
    caption_lines.append("")
    caption_lines.append("This metric combines:")
    caption_lines.append("  1. Single mRNA: Individual diffraction-limited spots")
    caption_lines.append("  2. Clustered mRNA: Aggregated signal normalized by peak intensity")
    caption_lines.append("")

    caption_lines.append("INTERPRETATION:")
    caption_lines.append("-" * 80)
    caption_lines.append("Total mRNA per cell represents the most comprehensive quantification of transcript")
    caption_lines.append("abundance, capturing both dispersed single molecules and high-density aggregates.")
    caption_lines.append("Regional differences may reflect:")
    caption_lines.append("  - Intrinsic transcriptional differences")
    caption_lines.append("  - Cell-type composition variations")
    caption_lines.append("  - Disease-specific vulnerability patterns")
    caption_lines.append("")
    caption_lines.append("Inter-mouse variability may be driven by:")
    caption_lines.append("  - Age differences")
    caption_lines.append("  - Treatment conditions (aCSF, NTC, UNT)")
    caption_lines.append("  - Biological heterogeneity within genotype")
    caption_lines.append("")

    caption_lines.append("=" * 80)
    caption_lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    caption_lines.append("=" * 80)

    # Save caption
    caption_path = OUTPUT_DIR / "fig_total_expression_summary_caption.txt"
    with open(caption_path, 'w') as f:
        f.write('\n'.join(caption_lines))
    print(f"  Saved caption: {caption_path}")

    # Save figure
    for fmt in ['png', 'svg', 'pdf']:
        output_path = OUTPUT_DIR / f"fig_total_expression_summary.{fmt}"
        fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"  Saved: {output_path}")

    plt.close(fig)

    return df_age


# ══════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("="*70)
    print("Q111 fl-HTT EXPRESSION ANALYSIS")
    print("="*70)

    # ──────────────────────────────────────────────────────────────────────
    # SECTION 1: LOAD DATA
    # ──────────────────────────────────────────────────────────────────────

    print("\n" + "="*70)
    print("SECTION 1: LOADING DATA")
    print("="*70)

    # Load HDF5 data
    with h5py.File(H5_FILE_PATH_EXPERIMENTAL, 'r') as h5_file:
        data_dict = recursively_load_dict(h5_file)

    print(f"\nLoaded data from: {H5_FILE_PATH_EXPERIMENTAL}")

    # Extract DataFrame
    desired_channels = ['blue', 'green', 'orange']
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

    df_extracted = extract_dataframe(
        data_dict,
        field_keys=fields_to_extract,
        channels=desired_channels,
        include_file_metadata_sample=True,
        explode_fields=[]
    )

    print(f"Extracted {len(df_extracted)} total records")

    # Filter for experimental data (Q111)
    experimental_field = 'ExperimentalQ111 - 488mHT - 548mHTa - 647Darp'
    negative_control_field = 'Negative control'

    # Compute thresholds using the comprehensive function
    slide_field = 'metadata_sample_slide_name_std'

    (thresholds, thresholds_cluster,
     error_thresholds, error_thresholds_cluster,
     number_of_datapoints, age) = compute_thresholds(
        df_extracted=df_extracted,
        slide_field=slide_field,
        desired_channels=['green', 'orange'],
        negative_control_field=negative_control_field,
        experimental_field=experimental_field,
        quantile_negative_control=QUANTILE_NEGATIVE_CONTROL,
        max_pfa=MAX_PFA,
        plot=False,
        n_bootstrap=N_BOOTSTRAP,
        use_region=False,
        use_final_filter=True
    )

    # Build threshold lookup
    thr_rows = []
    for (slide, channel, area), vec in error_thresholds.items():
        thr_rows.append({
            "slide": slide,
            "channel": channel,
            "thr": np.mean(vec)
        })
    thr_df = (
        pd.DataFrame(thr_rows)
        .drop_duplicates(["slide", "channel"])
    )

    # Merge thresholds into main dataframe
    df_extracted = df_extracted.merge(
        thr_df,
        how="left",
        left_on=[slide_field, "channel"],
        right_on=["slide", "channel"]
    )
    df_extracted.rename(columns={"thr": "threshold"}, inplace=True)
    df_extracted.drop(columns=["slide"], inplace=True, errors='ignore')

    # Filter for experimental data (Q111 and Wildtype)
    df_experimental = df_extracted[
        df_extracted['metadata_sample_Probe-Set'] == experimental_field
    ].copy()

    # Check what mouse models are available
    print(f"\nAvailable mouse models: {df_experimental['metadata_sample_Mouse_Model'].unique()}")
    print(f"Mouse model counts:\n{df_experimental['metadata_sample_Mouse_Model'].value_counts()}")

    # Keep both Q111 and Wildtype (filter out any other models if present)
    df_experimental = df_experimental[
        df_experimental["metadata_sample_Mouse_Model"].isin(['Q111', 'Wildtype'])
    ].copy()

    print(f"Filtered to {len(df_experimental)} experimental records (Q111 + Wildtype)")

    # Exclude slides with technical failures
    if len(EXCLUDED_SLIDES) > 0:
        print(f"\nExcluding {len(EXCLUDED_SLIDES)} slides with technical failures: {EXCLUDED_SLIDES}")
        before_count = len(df_experimental)
        df_experimental = df_experimental[
            ~df_experimental[slide_field].isin(EXCLUDED_SLIDES)
        ].copy()
        after_count = len(df_experimental)
        print(f"  Removed {before_count - after_count} records")
        print(f"  Remaining: {after_count} records")

    # ──────────────────────────────────────────────────────────────────────
    # SECTION 2: EXTRACT FOV-LEVEL DATA
    # ──────────────────────────────────────────────────────────────────────

    print("\n" + "="*70)
    print("SECTION 2: EXTRACTING FOV-LEVEL DATA")
    print("="*70)

    # Process each mouse model separately and combine
    df_fov_list = []
    for model in df_experimental['metadata_sample_Mouse_Model'].unique():
        print(f"\nProcessing {model} data...")
        df_model = df_experimental[df_experimental['metadata_sample_Mouse_Model'] == model].copy()
        df_fov_model = extract_fov_level_data(df_model, model)
        df_fov_list.append(df_fov_model)

    df_fov = pd.concat(df_fov_list, ignore_index=True)

    # Save FOV data
    csv_path = OUTPUT_DIR / "fov_level_data.csv"
    df_fov.to_csv(csv_path, index=False)
    print(f"\nSaved FOV data to: {csv_path}")

    # ──────────────────────────────────────────────────────────────────────
    # SECTION 3: CREATE COMPREHENSIVE FIGURE
    # ──────────────────────────────────────────────────────────────────────

    print("\n" + "="*70)
    print("SECTION 3: CREATING COMPREHENSIVE EXPRESSION FIGURE")
    print("="*70)

    df_summary = create_comprehensive_expression_figure(df_fov)

    # Save summary statistics
    summary_path = OUTPUT_DIR / "summary_statistics.csv"
    df_summary.to_csv(summary_path, index=False)
    print(f"\nSaved summary statistics to: {summary_path}")

    # ──────────────────────────────────────────────────────────────────────
    # SECTION 4: CREATE TOTAL EXPRESSION SUMMARY FIGURE
    # ──────────────────────────────────────────────────────────────────────

    print("\n" + "="*70)
    print("SECTION 4: CREATING TOTAL EXPRESSION SUMMARY FIGURE")
    print("="*70)

    df_age = create_total_expression_figure(df_fov)

    # Save age statistics
    age_path = OUTPUT_DIR / "age_statistics.csv"
    df_age.to_csv(age_path, index=False)
    print(f"\nSaved age statistics to: {age_path}")

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
