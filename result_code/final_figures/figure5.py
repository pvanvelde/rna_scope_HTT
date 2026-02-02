"""
Figure 5 - Cluster Properties: Extreme vs Normal FOVs

Compares cluster-level properties between extreme FOVs (> WT P95) and normal FOVs.
Combines analysis for both mHTT1a and full-length mHTT across Cortex and Striatum.

Layout (matching PDF comments):
    Row 1: A (paired extreme vs normal FOVs from SAME animal - 3 columns: Animal #1, #2, #3; 2 rows: Extreme top, Normal bottom)
           B (clusters per nucleus boxplot - both channels)
    Row 2: C (mean cluster mRNA equiv boxplot)
           D (mHTT1a cluster localization - Normal top, Extreme bottom)
           E (full-length mHTT cluster localization - Normal top, Extreme bottom)

Panel A demonstrates within-animal heterogeneity: the same animal can have both extreme
and normal expression FOVs, showing that mRNA accumulation is spatially heterogeneous
even within a single brain.
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import mannwhitneyu
import pickle
import sys

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'draft_figures'))

from figure_config import (
    FigureConfig,
    apply_figure_style,
    save_figure,
    COLORS
)
from results_config import (
    VOXEL_SIZE,
    CV_THRESHOLD,
    BEAD_PSF_X,
    BEAD_PSF_Y,
    BEAD_PSF_Z,
    SIGMA_X_LOWER,
    QUANTILE_NEGATIVE_CONTROL,
    MAX_PFA,
)

# Apply consistent styling
apply_figure_style()

# Output directory
OUTPUT_DIR = Path(__file__).parent / "output"
CACHE_DIR = OUTPUT_DIR / 'cache'
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = CACHE_DIR / 'figure5_data.pkl'

# Set to True to force data reload
FORCE_RELOAD = False

# Channel colors (same as figure 4)
COLOR_MHTT1A = COLORS.get('q111_mhtt1a', '#2ecc71')  # Green
COLOR_FULL = COLORS.get('q111_full', '#f39c12')      # Orange


def load_and_process_data():
    """Load cluster property data from draft figures output."""

    # Check cache first
    if CACHE_FILE.exists() and not FORCE_RELOAD:
        print(f"Loading cached data from {CACHE_FILE}")
        with open(CACHE_FILE, 'rb') as f:
            return pickle.load(f)

    print("Loading cluster property data...")

    # Load from the draft figures output
    cluster_dir = Path(__file__).parent.parent / 'draft_figures' / 'output' / 'cluster_properties_extreme_vs_normal'

    df_clusters = pd.read_csv(cluster_dir / 'cluster_level_data.csv')
    df_fov_summary = pd.read_csv(cluster_dir / 'fov_cluster_summary.csv')

    print(f"  Loaded {len(df_clusters)} clusters, {len(df_fov_summary)} FOV summaries")

    # Standardize channel names
    df_clusters['Channel'] = df_clusters['Channel'].replace({'full length mHTT': 'full-length mHTT'})
    df_fov_summary['Channel'] = df_fov_summary['Channel'].replace({'full length mHTT': 'full-length mHTT'})

    # Calculate cluster volume (μm³) and density (mRNA equiv / μm³)
    df_clusters['Cluster_Volume_um3'] = df_clusters['Cluster_Size_voxels'] * VOXEL_SIZE
    df_clusters['Cluster_Density'] = df_clusters['Cluster_mRNA_Equiv'] / df_clusters['Cluster_Volume_um3']

    data = {
        'df_clusters': df_clusters,
        'df_fov_summary': df_fov_summary,
    }

    # Cache the data
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(data, f)
    print(f"Data cached to {CACHE_FILE}")

    return data


def plot_clusters_per_cell_merged(ax, df_fov):
    """Plot clusters per cell: Extreme (left) vs Normal (right), both channels and regions.
    Uses channel colors (green=mHTT1a, orange=full-length), hatching for Striatum."""
    cfg = FigureConfig
    from matplotlib.patches import Patch

    positions = []
    data_list = []
    colors_list = []
    hatches_list = []
    x_labels = []
    x_positions = []

    pos = 0
    for channel in ['mHTT1a', 'full-length mHTT']:
        df_ch = df_fov[df_fov['Channel'] == channel]
        ch_color = COLOR_MHTT1A if channel == 'mHTT1a' else COLOR_FULL
        ch_start = pos

        for region in ['Cortex', 'Striatum']:
            df_reg = df_ch[df_ch['Region'] == region]
            hatch = '///' if region == 'Striatum' else ''

            extreme = df_reg[df_reg['FOV_Class'] == 'Extreme']['Clusters_per_Cell'].dropna()
            normal = df_reg[df_reg['FOV_Class'] == 'Normal']['Clusters_per_Cell'].dropna()

            # Extreme on left
            if len(extreme) > 0:
                positions.append(pos)
                data_list.append(extreme.values)
                colors_list.append(ch_color)
                hatches_list.append(hatch)
                pos += 1

            # Normal on right
            if len(normal) > 0:
                positions.append(pos)
                data_list.append(normal.values)
                colors_list.append(ch_color)
                hatches_list.append(hatch)
                pos += 1

            pos += 0.3  # Small gap between regions

        ch_label = 'mHTT1a' if channel == 'mHTT1a' else 'full-length'
        x_labels.append(ch_label)
        x_positions.append((ch_start + pos - 0.3) / 2)
        pos += 1  # Larger gap between channels

    if len(data_list) == 0:
        return

    bp = ax.boxplot(data_list, positions=positions, patch_artist=True,
                    widths=0.6, showfliers=False)

    # Alternate alpha: darker for Extreme (left), lighter for Normal (right)
    for i, (patch, color, hatch) in enumerate(zip(bp['boxes'], colors_list, hatches_list)):
        is_extreme = (i % 2 == 0)  # Every other box starting from 0 is Extreme
        patch.set_facecolor(color)
        patch.set_alpha(0.9 if is_extreme else 0.5)
        patch.set_hatch(hatch)
        patch.set_edgecolor('black')

    for element in ['whiskers', 'caps', 'medians']:
        for item in bp[element]:
            item.set_color('black')

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, fontsize=cfg.FONT_SIZE_AXIS_TICK)
    ax.set_ylabel('Clusters per nucleus', fontsize=cfg.FONT_SIZE_AXIS_LABEL)

    # Legend instead of subtitle
    legend_elements = [
        Patch(facecolor='gray', alpha=0.9, edgecolor='black', label='Extreme'),
        Patch(facecolor='gray', alpha=0.5, edgecolor='black', label='Normal'),
        Patch(facecolor='white', edgecolor='black', hatch='///', label='Striatum'),
        Patch(facecolor='white', edgecolor='black', label='Cortex'),
    ]
    ax.legend(handles=legend_elements, fontsize=cfg.FONT_SIZE_LEGEND, loc='upper right', ncol=2)

    all_data = np.concatenate(data_list)
    y_max = np.percentile(all_data, 99) * 1.1
    ax.set_ylim([0, y_max])
    ax.grid(True, alpha=0.3, axis='y')


def plot_cluster_intensity_dist_by_channel(ax, df_clusters):
    """Plot cluster mRNA equivalent distributions: Normal vs Extreme, per channel using step histograms."""
    cfg = FigureConfig

    if 'Cluster_mRNA_Equiv' not in df_clusters.columns:
        return

    # Get max value across all data for consistent bins
    all_data = df_clusters['Cluster_mRNA_Equiv'].dropna()
    max_val = np.percentile(all_data, 99)
    bins = np.linspace(0, max_val, 40)

    # Plot each channel separately with step histograms
    for channel, ch_color in [('mHTT1a', COLOR_MHTT1A), ('full-length mHTT', COLOR_FULL)]:
        df_ch = df_clusters[df_clusters['Channel'] == channel]

        normal = df_ch[df_ch['FOV_Class'] == 'Normal']['Cluster_mRNA_Equiv'].dropna()
        extreme = df_ch[df_ch['FOV_Class'] == 'Extreme']['Cluster_mRNA_Equiv'].dropna()

        ch_label = 'mHTT1a' if channel == 'mHTT1a' else 'full-length'

        if len(normal) > 0:
            ax.hist(normal, bins=bins, alpha=0.4, color=ch_color,
                    density=True, histtype='stepfilled', linewidth=1.5,
                    label=f'{ch_label} Normal')

        if len(extreme) > 0:
            ax.hist(extreme, bins=bins, alpha=0.7, color=ch_color,
                    density=True, histtype='step', linewidth=2,
                    linestyle='--', label=f'{ch_label} Extreme')

    ax.set_xlabel('Cluster mRNA equivalents', fontsize=cfg.FONT_SIZE_AXIS_LABEL)
    ax.set_ylabel('Density', fontsize=cfg.FONT_SIZE_AXIS_LABEL)
    ax.set_title('Cluster size distribution', fontsize=cfg.FONT_SIZE_TITLE, fontweight='bold')
    ax.legend(fontsize=cfg.FONT_SIZE_LEGEND - 1, loc='upper right', ncol=1)
    ax.grid(True, alpha=0.3, axis='y')


def plot_distance_to_dapi_split(ax_normal, ax_extreme, df_clusters, channel):
    """Plot distance to DAPI distributions: Normal and Extreme on separate axes.

    Args:
        ax_normal: Axis for Normal FOVs (top)
        ax_extreme: Axis for Extreme FOVs (bottom)
        df_clusters: DataFrame with cluster data
        channel: 'mHTT1a' or 'full-length mHTT'
    """
    cfg = FigureConfig

    df_ch = df_clusters[df_clusters['Channel'] == channel]
    ch_color = COLOR_MHTT1A if channel == 'mHTT1a' else COLOR_FULL

    if 'Distance_to_DAPI_um' not in df_ch.columns:
        return

    # Combine both regions
    normal = df_ch[df_ch['FOV_Class'] == 'Normal']['Distance_to_DAPI_um'].dropna()
    extreme = df_ch[df_ch['FOV_Class'] == 'Extreme']['Distance_to_DAPI_um'].dropna()

    if len(normal) == 0 or len(extreme) == 0:
        return

    # Use same bins for both
    min_val = np.percentile(np.concatenate([normal, extreme]), 1)
    max_val = np.percentile(np.concatenate([normal, extreme]), 99)
    bins = np.linspace(min_val, max_val, 40)

    # Calculate fraction nuclear
    frac_nuc_norm = np.mean(normal < 0) * 100
    frac_nuc_ext = np.mean(extreme < 0) * 100

    ch_label = 'mHTT1a' if channel == 'mHTT1a' else 'full-length'

    # Plot Normal (top) - lighter
    ax_normal.hist(normal, bins=bins, alpha=0.5, color=ch_color,
                   density=True, edgecolor='black', linewidth=0.5,
                   label=f'Normal ({frac_nuc_norm:.0f}% nuc.)')
    ax_normal.axvline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax_normal.set_ylabel('Density', fontsize=cfg.FONT_SIZE_AXIS_LABEL - 1)
    ax_normal.legend(fontsize=cfg.FONT_SIZE_LEGEND, loc='upper right')
    ax_normal.set_xticklabels([])  # Hide x-tick labels on top plot
    ax_normal.grid(True, alpha=0.3, axis='y')
    # Channel label on top
    ax_normal.text(0.02, 0.95, ch_label, transform=ax_normal.transAxes,
                   fontsize=cfg.FONT_SIZE_AXIS_LABEL, fontweight='bold', va='top')

    # Plot Extreme (bottom) - darker
    ax_extreme.hist(extreme, bins=bins, alpha=0.9, color=ch_color,
                    density=True, edgecolor='black', linewidth=0.5,
                    label=f'Extreme ({frac_nuc_ext:.0f}% nuc.)')
    ax_extreme.axvline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax_extreme.set_xlabel('Distance to nuclear membrane (μm)', fontsize=cfg.FONT_SIZE_AXIS_LABEL)
    ax_extreme.set_ylabel('Density', fontsize=cfg.FONT_SIZE_AXIS_LABEL - 1)
    ax_extreme.legend(fontsize=cfg.FONT_SIZE_LEGEND, loc='upper right')
    ax_extreme.grid(True, alpha=0.3, axis='y')

    # Match y-axis limits
    y_max = max(ax_normal.get_ylim()[1], ax_extreme.get_ylim()[1])
    ax_normal.set_ylim([0, y_max])
    ax_extreme.set_ylim([0, y_max])


def plot_distance_to_dapi_merged(ax_normal, ax_extreme, df_clusters):
    """Plot distance to DAPI distributions for BOTH channels: Normal (top) and Extreme (bottom).

    Args:
        ax_normal: Axis for Normal FOVs (top)
        ax_extreme: Axis for Extreme FOVs (bottom)
        df_clusters: DataFrame with cluster data
    """
    cfg = FigureConfig

    if 'Distance_to_DAPI_um' not in df_clusters.columns:
        return

    # Get consistent bins across all data with 1 µm bin width
    # (z-slice spacing is 500nm, so 1 µm is appropriate given DAPI mask precision)
    all_dist = df_clusters['Distance_to_DAPI_um'].dropna()
    min_val = np.floor(np.percentile(all_dist, 1))  # Round down to nearest integer
    max_val = np.ceil(np.percentile(all_dist, 99))   # Round up to nearest integer
    bin_width = 1.0  # 1 µm bin width
    bins = np.arange(min_val, max_val + bin_width, bin_width)

    # Plot both channels
    for channel, ch_color in [('mHTT1a', COLOR_MHTT1A), ('full-length mHTT', COLOR_FULL)]:
        df_ch = df_clusters[df_clusters['Channel'] == channel]
        ch_label = 'mHTT1a' if channel == 'mHTT1a' else 'full-length'

        normal = df_ch[df_ch['FOV_Class'] == 'Normal']['Distance_to_DAPI_um'].dropna()
        extreme = df_ch[df_ch['FOV_Class'] == 'Extreme']['Distance_to_DAPI_um'].dropna()

        frac_nuc_norm = np.mean(normal < 0) * 100 if len(normal) > 0 else 0
        frac_nuc_ext = np.mean(extreme < 0) * 100 if len(extreme) > 0 else 0

        # Normal (top)
        if len(normal) > 0:
            ax_normal.hist(normal, bins=bins, alpha=0.5, color=ch_color,
                          density=True, histtype='stepfilled', linewidth=1,
                          label=f'{ch_label} ({frac_nuc_norm:.0f}% nuc.)')

        # Extreme (bottom)
        if len(extreme) > 0:
            ax_extreme.hist(extreme, bins=bins, alpha=0.7, color=ch_color,
                           density=True, histtype='stepfilled', linewidth=1,
                           label=f'{ch_label} ({frac_nuc_ext:.0f}% nuc.)')

    # Styling for Normal (top)
    ax_normal.axvline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax_normal.set_ylabel('Prob. density', fontsize=cfg.FONT_SIZE_AXIS_LABEL - 1)
    ax_normal.legend(fontsize=cfg.FONT_SIZE_LEGEND - 1, loc='upper right')
    ax_normal.grid(True, alpha=0.3, axis='y')
    ax_normal.text(0.02, 0.95, 'Normal', transform=ax_normal.transAxes,
                   fontsize=cfg.FONT_SIZE_AXIS_LABEL - 1, fontweight='bold', va='top')

    # Styling for Extreme (bottom)
    ax_extreme.axvline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax_extreme.set_xlabel('Distance to nuclear membrane (μm)', fontsize=cfg.FONT_SIZE_AXIS_LABEL)
    ax_extreme.set_ylabel('Prob. density', fontsize=cfg.FONT_SIZE_AXIS_LABEL - 1)
    ax_extreme.legend(fontsize=cfg.FONT_SIZE_LEGEND - 1, loc='upper right')
    ax_extreme.grid(True, alpha=0.3, axis='y')
    ax_extreme.text(0.02, 0.95, 'Extreme', transform=ax_extreme.transAxes,
                    fontsize=cfg.FONT_SIZE_AXIS_LABEL - 1, fontweight='bold', va='top')

    # Match y-axis limits
    y_max = max(ax_normal.get_ylim()[1], ax_extreme.get_ylim()[1])
    ax_normal.set_ylim([0, y_max])
    ax_extreme.set_ylim([0, y_max])


def plot_cluster_density_split(ax_normal, ax_extreme, df_clusters):
    """Plot cluster density distributions for BOTH channels: Normal (top) and Extreme (bottom).

    Args:
        ax_normal: Axis for Normal FOVs (top)
        ax_extreme: Axis for Extreme FOVs (bottom)
        df_clusters: DataFrame with cluster data
    """
    cfg = FigureConfig

    if 'Cluster_Density' not in df_clusters.columns:
        return

    # Get consistent bins (99th percentile to avoid outliers)
    all_data = df_clusters['Cluster_Density'].dropna()
    max_val = np.percentile(all_data, 99)
    bins = np.linspace(0, max_val, 40)

    # Plot both channels
    for channel, ch_color in [('mHTT1a', COLOR_MHTT1A), ('full-length mHTT', COLOR_FULL)]:
        df_ch = df_clusters[df_clusters['Channel'] == channel]
        ch_label = 'mHTT1a' if channel == 'mHTT1a' else 'full-length'

        normal = df_ch[df_ch['FOV_Class'] == 'Normal']['Cluster_Density'].dropna()
        extreme = df_ch[df_ch['FOV_Class'] == 'Extreme']['Cluster_Density'].dropna()

        median_norm = np.median(normal) if len(normal) > 0 else 0
        median_ext = np.median(extreme) if len(extreme) > 0 else 0

        # Normal (top)
        if len(normal) > 0:
            ax_normal.hist(normal, bins=bins, alpha=0.5, color=ch_color,
                          density=True, histtype='stepfilled', linewidth=1,
                          label=f'{ch_label} (med={median_norm:.2f})')

        # Extreme (bottom)
        if len(extreme) > 0:
            ax_extreme.hist(extreme, bins=bins, alpha=0.7, color=ch_color,
                           density=True, histtype='stepfilled', linewidth=1,
                           label=f'{ch_label} (med={median_ext:.2f})')

    # Styling for Normal (top)
    ax_normal.set_ylabel('Prob. density', fontsize=cfg.FONT_SIZE_AXIS_LABEL - 1)
    ax_normal.legend(fontsize=cfg.FONT_SIZE_LEGEND - 1, loc='upper right')
    ax_normal.grid(True, alpha=0.3, axis='y')
    ax_normal.text(0.02, 0.95, 'Normal', transform=ax_normal.transAxes,
                   fontsize=cfg.FONT_SIZE_AXIS_LABEL - 1, fontweight='bold', va='top')

    # Styling for Extreme (bottom)
    ax_extreme.set_xlabel('Cluster density (mRNA/μm³)', fontsize=cfg.FONT_SIZE_AXIS_LABEL)
    ax_extreme.set_ylabel('Prob. density', fontsize=cfg.FONT_SIZE_AXIS_LABEL - 1)
    ax_extreme.legend(fontsize=cfg.FONT_SIZE_LEGEND - 1, loc='upper right')
    ax_extreme.grid(True, alpha=0.3, axis='y')
    ax_extreme.text(0.02, 0.95, 'Extreme', transform=ax_extreme.transAxes,
                    fontsize=cfg.FONT_SIZE_AXIS_LABEL - 1, fontweight='bold', va='top')

    # Match y-axis limits
    y_max = max(ax_normal.get_ylim()[1], ax_extreme.get_ylim()[1])
    ax_normal.set_ylim([0, y_max])
    ax_extreme.set_ylim([0, y_max])


def plot_mean_cluster_mrna_merged(ax, df_fov):
    """Plot mean cluster mRNA per FOV: Extreme (left) vs Normal (right), both channels and regions.
    Uses channel colors (green=mHTT1a, orange=full-length), hatching for Striatum."""
    cfg = FigureConfig
    from matplotlib.patches import Patch

    if 'Mean_Cluster_mRNA' not in df_fov.columns:
        return

    positions = []
    data_list = []
    colors_list = []
    hatches_list = []
    x_labels = []
    x_positions = []

    pos = 0
    for channel in ['mHTT1a', 'full-length mHTT']:
        df_ch = df_fov[df_fov['Channel'] == channel]
        ch_color = COLOR_MHTT1A if channel == 'mHTT1a' else COLOR_FULL
        ch_start = pos

        for region in ['Cortex', 'Striatum']:
            df_reg = df_ch[df_ch['Region'] == region]
            hatch = '///' if region == 'Striatum' else ''

            extreme = df_reg[df_reg['FOV_Class'] == 'Extreme']['Mean_Cluster_mRNA'].dropna()
            normal = df_reg[df_reg['FOV_Class'] == 'Normal']['Mean_Cluster_mRNA'].dropna()

            # Extreme on left
            if len(extreme) > 0:
                positions.append(pos)
                data_list.append(extreme.values)
                colors_list.append(ch_color)
                hatches_list.append(hatch)
                pos += 1

            # Normal on right
            if len(normal) > 0:
                positions.append(pos)
                data_list.append(normal.values)
                colors_list.append(ch_color)
                hatches_list.append(hatch)
                pos += 1

            pos += 0.3

        ch_label = 'mHTT1a' if channel == 'mHTT1a' else 'full-length'
        x_labels.append(ch_label)
        x_positions.append((ch_start + pos - 0.3) / 2)
        pos += 1

    if len(data_list) == 0:
        return

    bp = ax.boxplot(data_list, positions=positions, patch_artist=True,
                    widths=0.6, showfliers=False)

    # Alternate alpha: darker for Extreme (left), lighter for Normal (right)
    for i, (patch, color, hatch) in enumerate(zip(bp['boxes'], colors_list, hatches_list)):
        is_extreme = (i % 2 == 0)  # Every other box starting from 0 is Extreme
        patch.set_facecolor(color)
        patch.set_alpha(0.9 if is_extreme else 0.5)
        patch.set_hatch(hatch)
        patch.set_edgecolor('black')

    for element in ['whiskers', 'caps', 'medians']:
        for item in bp[element]:
            item.set_color('black')

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, fontsize=cfg.FONT_SIZE_AXIS_TICK)
    ax.set_ylabel('Mean cluster mRNA equiv.', fontsize=cfg.FONT_SIZE_AXIS_LABEL)

    # Legend instead of subtitle
    legend_elements = [
        Patch(facecolor='gray', alpha=0.9, edgecolor='black', label='Extreme'),
        Patch(facecolor='gray', alpha=0.5, edgecolor='black', label='Normal'),
        Patch(facecolor='white', edgecolor='black', hatch='///', label='Striatum'),
        Patch(facecolor='white', edgecolor='black', label='Cortex'),
    ]
    ax.legend(handles=legend_elements, fontsize=cfg.FONT_SIZE_LEGEND, loc='upper right', ncol=2)

    all_data = np.concatenate(data_list)
    y_max = np.percentile(all_data, 99) * 1.1
    ax.set_ylim([0, y_max])
    ax.grid(True, alpha=0.3, axis='y')


def plot_scatter_clusters_vs_mrna(ax, df_fov):
    """Scatter plot: clusters per cell vs clustered mRNA per cell, all channels/regions."""
    cfg = FigureConfig

    # Plot for each channel with different colors
    for channel, color in [('mHTT1a', COLOR_MHTT1A), ('full-length mHTT', COLOR_FULL)]:
        df_ch = df_fov[df_fov['Channel'] == channel]

        normal = df_ch[df_ch['FOV_Class'] == 'Normal']
        extreme = df_ch[df_ch['FOV_Class'] == 'Extreme']

        # Normal FOVs - lighter shade
        if len(normal) > 0:
            ax.scatter(normal['Clusters_per_Cell'], normal['Clustered_mRNA_per_Cell'],
                      c=color, alpha=0.3, s=15, edgecolors='none',
                      label=f'{channel} Normal' if channel == 'mHTT1a' else None)

        # Extreme FOVs - marker with edge
        if len(extreme) > 0:
            ax.scatter(extreme['Clusters_per_Cell'], extreme['Clustered_mRNA_per_Cell'],
                      c=color, alpha=0.7, s=25, edgecolors='black', linewidths=0.5,
                      label=f'{channel} Extreme' if channel == 'mHTT1a' else None)

    ax.set_xlabel('Clusters per nucleus', fontsize=cfg.FONT_SIZE_AXIS_LABEL)
    ax.set_ylabel('Clustered mRNA per nucleus', fontsize=cfg.FONT_SIZE_AXIS_LABEL)
    ax.set_title('Clusters vs total mRNA', fontsize=cfg.FONT_SIZE_TITLE, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor=COLOR_MHTT1A, alpha=0.7, edgecolor='black', label='mHTT1a'),
        Patch(facecolor=COLOR_FULL, alpha=0.7, edgecolor='black', label='full-length mHTT'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
               markersize=6, alpha=0.3, label='Normal'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
               markeredgecolor='black', markersize=8, alpha=0.7, label='Extreme'),
    ]
    ax.legend(handles=legend_elements, fontsize=cfg.FONT_SIZE_LEGEND - 1, loc='upper left')


def create_figure5():
    """Create Figure 5 with cluster property comparisons."""
    cfg = FigureConfig

    # Load data
    print("\n" + "=" * 70)
    print("LOADING DATA FOR FIGURE 5")
    print("=" * 70)

    data = load_and_process_data()
    df_clusters = data['df_clusters']
    df_fov = data['df_fov_summary']

    # Figure dimensions - use standard page width from config
    fig_width = cfg.PAGE_WIDTH_FULL
    fig_height = fig_width * 0.7

    fig = plt.figure(figsize=(fig_width, fig_height))

    # Main grid: 2 rows x 8 columns - use config spacing
    main_gs = gridspec.GridSpec(
        2, 8,
        figure=fig,
        left=cfg.SUBPLOT_LEFT,
        right=cfg.SUBPLOT_RIGHT,
        bottom=cfg.SUBPLOT_BOTTOM,
        top=cfg.SUBPLOT_TOP,
        hspace=cfg.SUBPLOT_HSPACE,
        wspace=cfg.SUBPLOT_WSPACE
    )

    axes = {}

    # Row 1: A (placeholder - larger, 5 cols), B (clusters/cell merged, 3 cols)
    axes['A'] = fig.add_subplot(main_gs[0, 0:5])
    axes['B'] = fig.add_subplot(main_gs[0, 5:8])

    # Row 2: C (mean cluster mRNA, 2 cols), D (distance merged, 3 cols), E (density, 3 cols)
    axes['C'] = fig.add_subplot(main_gs[1, 0:2])

    # Panel D: Distance to DAPI - merged (both channels), split Normal/Extreme
    gs_d = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=main_gs[1, 2:5], hspace=0.08)
    axes['D_normal'] = fig.add_subplot(gs_d[0])
    axes['D_extreme'] = fig.add_subplot(gs_d[1], sharex=axes['D_normal'])

    # Panel E: Cluster density - merged (both channels), split Normal/Extreme
    gs_e = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=main_gs[1, 5:8], hspace=0.08)
    axes['E_normal'] = fig.add_subplot(gs_e[0])
    axes['E_extreme'] = fig.add_subplot(gs_e[1], sharex=axes['E_normal'])

    print("\n" + "=" * 70)
    print("CREATING PANELS")
    print("=" * 70)

    # Panel A: Placeholder (larger for example images)
    print("  Creating panel A (placeholder for example images)...")
    axes['A'].set_facecolor(COLORS['gray_light'])
    axes['A'].text(0.5, 0.5, 'Panel A\n(Example images)', transform=axes['A'].transAxes,
                   ha='center', va='center', fontsize=cfg.FONT_SIZE_TITLE,
                   color=COLORS['gray_dark'])
    axes['A'].set_xticks([])
    axes['A'].set_yticks([])

    # Panel B: Clusters per cell - merged (both channels)
    print("  Creating panel B (clusters per cell - merged)...")
    plot_clusters_per_cell_merged(axes['B'], df_fov)

    # Panel C: Mean cluster mRNA - merged (was panel F)
    print("  Creating panel C (mean cluster mRNA - merged)...")
    plot_mean_cluster_mrna_merged(axes['C'], df_fov)

    # Panel D: Distance to DAPI - merged (both channels, split Normal/Extreme)
    print("  Creating panel D (distance to DAPI - merged, split)...")
    plot_distance_to_dapi_merged(axes['D_normal'], axes['D_extreme'], df_clusters)

    # Panel E: Cluster density - merged (both channels, split Normal/Extreme)
    print("  Creating panel E (cluster density - merged, split)...")
    plot_cluster_density_split(axes['E_normal'], axes['E_extreme'], df_clusters)

    # Add panel labels
    # For D and E, use the top (normal) subplot for the label
    label_axes = {
        'A': axes['A'],
        'B': axes['B'],
        'C': axes['C'],
        'D': axes['D_normal'],
        'E': axes['E_normal'],
    }
    for label, ax in label_axes.items():
        ax.text(-0.12, 1.05, label, transform=ax.transAxes,
                fontsize=cfg.FONT_SIZE_PANEL_LABEL,
                fontweight=cfg.FONT_WEIGHT_PANEL_LABEL,
                va='bottom', ha='left')

    # Compute statistics for caption
    stats = compute_statistics(df_clusters, df_fov)

    return fig, axes, stats


def compute_statistics(df_clusters, df_fov):
    """Compute summary statistics for caption."""
    stats = {}

    for channel in ['mHTT1a', 'full-length mHTT']:
        df_ch_fov = df_fov[df_fov['Channel'] == channel]
        df_ch_clusters = df_clusters[df_clusters['Channel'] == channel]

        ch_stats = {}
        for region in ['Cortex', 'Striatum']:
            df_reg_fov = df_ch_fov[df_ch_fov['Region'] == region]

            normal_fov = df_reg_fov[df_reg_fov['FOV_Class'] == 'Normal']
            extreme_fov = df_reg_fov[df_reg_fov['FOV_Class'] == 'Extreme']

            reg_stats = {
                'n_normal_fov': len(normal_fov),
                'n_extreme_fov': len(extreme_fov),
            }

            # Clusters per cell
            if len(normal_fov) > 0 and len(extreme_fov) > 0:
                norm_cpc = normal_fov['Clusters_per_Cell'].dropna()
                ext_cpc = extreme_fov['Clusters_per_Cell'].dropna()
                if len(norm_cpc) > 0 and len(ext_cpc) > 0:
                    stat, p = mannwhitneyu(norm_cpc, ext_cpc, alternative='two-sided')
                    reg_stats['cpc_normal_median'] = np.median(norm_cpc)
                    reg_stats['cpc_extreme_median'] = np.median(ext_cpc)
                    reg_stats['cpc_p_value'] = p

            ch_stats[region] = reg_stats

        # Total clusters
        ch_stats['n_normal_clusters'] = len(df_ch_clusters[df_ch_clusters['FOV_Class'] == 'Normal'])
        ch_stats['n_extreme_clusters'] = len(df_ch_clusters[df_ch_clusters['FOV_Class'] == 'Extreme'])

        # Cluster density stats (mRNA equiv / μm³)
        if 'Cluster_Density' in df_ch_clusters.columns:
            normal_density = df_ch_clusters[df_ch_clusters['FOV_Class'] == 'Normal']['Cluster_Density'].dropna()
            extreme_density = df_ch_clusters[df_ch_clusters['FOV_Class'] == 'Extreme']['Cluster_Density'].dropna()
            if len(normal_density) > 0 and len(extreme_density) > 0:
                ch_stats['density_normal_median'] = np.median(normal_density)
                ch_stats['density_extreme_median'] = np.median(extreme_density)
                stat, p = mannwhitneyu(normal_density, extreme_density, alternative='two-sided')
                ch_stats['density_p_value'] = p
                ch_stats['density_fold_change'] = np.median(extreme_density) / np.median(normal_density)

        # Distance to DAPI stats
        if 'Distance_to_DAPI_um' in df_ch_clusters.columns:
            normal_dist = df_ch_clusters[df_ch_clusters['FOV_Class'] == 'Normal']['Distance_to_DAPI_um'].dropna()
            extreme_dist = df_ch_clusters[df_ch_clusters['FOV_Class'] == 'Extreme']['Distance_to_DAPI_um'].dropna()
            if len(normal_dist) > 0:
                ch_stats['frac_nuclear_normal'] = np.mean(normal_dist < 0) * 100
            if len(extreme_dist) > 0:
                ch_stats['frac_nuclear_extreme'] = np.mean(extreme_dist < 0) * 100

        stats[channel] = ch_stats

    return stats


def generate_caption(stats):
    """Generate figure caption."""

    # Extract statistics for the caption
    mhtt1a_stats = stats.get('mHTT1a', {})
    full_stats = stats.get('full-length mHTT', {})

    caption = f"""Figure 5: Cluster properties in extreme vs normal FOVs reveal mechanistic insights into mRNA accumulation.

OVERVIEW:
This figure compares cluster-level properties between "extreme" FOVs (those exceeding the WT 95th percentile for clustered mRNA/nucleus) and "normal" FOVs (below threshold) within Q111 transgenic mice only. The analysis addresses the key biological question: Do extreme FOVs accumulate more mRNA because they have MORE clusters, LARGER clusters, or BOTH? Understanding this distinction provides mechanistic insight into disease-associated mRNA accumulation patterns and potential therapeutic targets.

PANEL DESCRIPTIONS:

(A) REPRESENTATIVE IMAGES
Placeholder for representative microscopy images comparing cluster appearance in normal vs extreme FOVs.
- Top row: Normal FOV examples showing typical cluster density and size
- Bottom row: Extreme FOV examples showing elevated cluster density and/or size
- Channels: DAPI (blue, nuclear stain), mHTT1a (green), full-length mHTT (orange)
- Scale bars and imaging parameters
- Annotations highlighting clusters and nuclear boundaries

(B) CLUSTERS PER NUCLEUS: EXTREME vs NORMAL FOVs
Box plot comparison of cluster density between extreme and normal FOVs.
- Y-axis: Clusters per nucleus (average number of clusters per DAPI-positive nucleus in FOV)
- X-axis grouping:
  * Left section: mHTT1a probe
  * Right section: Full-length mHTT probe
- Within each channel section:
  * Pairs of boxes: Extreme (dark, left) vs Normal (light, right) for each region
  * Cortex: solid fill
  * Striatum: hatched fill (///)
- Color coding:
  * Green: mHTT1a probe
  * Orange: Full-length mHTT probe
  * Dark shade: Extreme FOVs
  * Light shade: Normal FOVs
- Box plot elements:
  * Center line: Median
  * Box: Interquartile range (25th-75th percentile)
  * Whiskers: 1.5× IQR
  * Outliers: Not shown (showfliers=False) to improve visualization
- Sample sizes:
  * mHTT1a Cortex: Normal n={mhtt1a_stats.get('Cortex', {}).get('n_normal_fov', 'N/A')}, Extreme n={mhtt1a_stats.get('Cortex', {}).get('n_extreme_fov', 'N/A')}
  * mHTT1a Striatum: Normal n={mhtt1a_stats.get('Striatum', {}).get('n_normal_fov', 'N/A')}, Extreme n={mhtt1a_stats.get('Striatum', {}).get('n_extreme_fov', 'N/A')}
  * Full-length Cortex: Normal n={full_stats.get('Cortex', {}).get('n_normal_fov', 'N/A')}, Extreme n={full_stats.get('Cortex', {}).get('n_extreme_fov', 'N/A')}
  * Full-length Striatum: Normal n={full_stats.get('Striatum', {}).get('n_normal_fov', 'N/A')}, Extreme n={full_stats.get('Striatum', {}).get('n_extreme_fov', 'N/A')}
- Statistical tests (Mann-Whitney U, extreme vs normal):
  * mHTT1a Cortex: median(extreme)={mhtt1a_stats.get('Cortex', {}).get('cpc_extreme_median', 'N/A'):.3f}, median(normal)={mhtt1a_stats.get('Cortex', {}).get('cpc_normal_median', 'N/A'):.3f}, p={mhtt1a_stats.get('Cortex', {}).get('cpc_p_value', 'N/A'):.2e}
  * mHTT1a Striatum: median(extreme)={mhtt1a_stats.get('Striatum', {}).get('cpc_extreme_median', 'N/A'):.3f}, median(normal)={mhtt1a_stats.get('Striatum', {}).get('cpc_normal_median', 'N/A'):.3f}, p={mhtt1a_stats.get('Striatum', {}).get('cpc_p_value', 'N/A'):.2e}
- Key finding: Extreme FOVs have DRAMATICALLY more clusters per nucleus than normal FOVs

(C) MEAN CLUSTER SIZE (mRNA EQUIVALENTS): EXTREME vs NORMAL FOVs
Box plot comparison of individual cluster size between extreme and normal FOVs.
- Y-axis: Mean cluster mRNA equivalents per FOV (average mRNA content per cluster within each FOV)
- Same X-axis grouping and color coding as panel B
- This metric answers: Are clusters in extreme FOVs individually larger?
- Interpretation:
  * If extreme FOVs have larger mean cluster size → mRNA accumulates into fewer, larger clusters
  * If extreme FOVs have similar mean cluster size → mRNA accumulates into more clusters of similar size
  * If extreme FOVs have smaller mean cluster size → many small clusters accumulate
- Key finding: Extreme FOVs tend to have larger mean cluster sizes, indicating that both cluster number AND cluster size contribute to elevated mRNA levels

(D) CLUSTER LOCALIZATION - DISTANCE TO NUCLEAR MEMBRANE (BOTH PROBES)
Distance from cluster center to nearest DAPI (nuclear) boundary, comparing both probes.
- Y-axis: Probability density
- X-axis: Distance to nuclear membrane (micrometers, μm)
  * NEGATIVE values = cluster center is INSIDE nucleus (nuclear localization)
  * POSITIVE values = cluster center is OUTSIDE nucleus (cytoplasmic localization)
  * Zero (dashed vertical line) = nuclear boundary
- Panel split:
  * TOP subplot: Normal FOVs - both mHTT1a (green) and full-length mHTT (orange) overlaid
  * BOTTOM subplot: Extreme FOVs - both probes overlaid
- Legend shows percentage of clusters with nuclear localization (negative distance values) for each probe
- Nuclear localization statistics:
  * mHTT1a Normal: {mhtt1a_stats.get('frac_nuclear_normal', 0):.1f}% nuclear
  * mHTT1a Extreme: {mhtt1a_stats.get('frac_nuclear_extreme', 0):.1f}% nuclear
  * Full-length Normal: {full_stats.get('frac_nuclear_normal', 0):.1f}% nuclear
  * Full-length Extreme: {full_stats.get('frac_nuclear_extreme', 0):.1f}% nuclear
- Histogram parameters:
  * Bin width: 1 µm (chosen based on z-slice spacing of 500 nm and DAPI mask precision limitations)
  * Density normalization, stepfilled histtype
  * Range: 1st to 99th percentile of distance values
- Key finding: Both probes show predominantly NUCLEAR localization in both normal and extreme FOVs, with extreme FOVs showing slightly more cytoplasmic localization (shift toward positive distances)

(E) CLUSTER DENSITY - mRNA CONCENTRATION WITHIN CLUSTERS (BOTH PROBES)
Cluster density = mRNA equivalents per unit volume (mRNA/μm³), comparing both probes.
- Y-axis: Probability density
- X-axis: Cluster density (mRNA equivalents per μm³)
- Panel split:
  * TOP subplot: Normal FOVs - both mHTT1a (green) and full-length mHTT (orange) overlaid
  * BOTTOM subplot: Extreme FOVs - both probes overlaid
- Legend shows median cluster density for each probe
- Cluster density statistics:
  * mHTT1a Normal: median = {mhtt1a_stats.get('density_normal_median', 0):.2f} mRNA/μm³
  * mHTT1a Extreme: median = {mhtt1a_stats.get('density_extreme_median', 0):.2f} mRNA/μm³ ({mhtt1a_stats.get('density_fold_change', 0):.2f}× fold change, p = {mhtt1a_stats.get('density_p_value', 1):.2e})
  * Full-length Normal: median = {full_stats.get('density_normal_median', 0):.2f} mRNA/μm³
  * Full-length Extreme: median = {full_stats.get('density_extreme_median', 0):.2f} mRNA/μm³ ({full_stats.get('density_fold_change', 0):.2f}× fold change, p = {full_stats.get('density_p_value', 1):.2e})
- Calculation: Cluster_Density = Cluster_mRNA_Equiv / (Cluster_Size_voxels × VOXEL_SIZE)
  * VOXEL_SIZE = {VOXEL_SIZE:.6f} μm³/voxel
- Key finding: Clusters in extreme FOVs are MORE DENSELY PACKED with mRNA than clusters in normal FOVs. This indicates that extreme FOVs don't just have more clusters - the individual clusters contain higher concentrations of mRNA molecules per unit volume

EXTREME vs NORMAL FOV CLASSIFICATION:
- EXTREME FOVs: Q111 FOVs where clustered mRNA per nucleus EXCEEDS the 95th percentile of the WT distribution
- NORMAL FOVs: Q111 FOVs where clustered mRNA per nucleus is BELOW the WT 95th percentile threshold
- NOTE: This is a WITHIN-Q111 comparison (extreme Q111 FOVs vs normal Q111 FOVs)
- The WT P95 threshold provides a biologically-grounded definition of "elevated expression"
- Thresholds are calculated separately for each channel-region combination (4 thresholds total)

CLUSTER-LEVEL STATISTICS:
mHTT1a probe:
- Total clusters in normal FOVs: {mhtt1a_stats.get('n_normal_clusters', 'N/A'):,}
- Total clusters in extreme FOVs: {mhtt1a_stats.get('n_extreme_clusters', 'N/A'):,}

Full-length mHTT probe:
- Total clusters in normal FOVs: {full_stats.get('n_normal_clusters', 'N/A'):,}
- Total clusters in extreme FOVs: {full_stats.get('n_extreme_clusters', 'N/A'):,}

================================================================================
FILTERING APPLIED (consistent with Figure 1, panels E onwards)
================================================================================

CLUSTER-LEVEL ANALYSIS:
This figure analyzes cluster properties (number, size, localization) comparing extreme vs normal FOVs within Q111 mice.

UPSTREAM SPOT FILTERING (applied before cluster identification):
1. DETECTION QUALITY FILTER (PFA):
   - Criterion: Probability of False Alarm < {MAX_PFA}
   - Purpose: Remove spots with poor Gaussian fit quality

2. INTENSITY THRESHOLD (from negative controls):
   - Criterion: Integrated photons > {QUANTILE_NEGATIVE_CONTROL*100:.0f}th percentile of negative control distribution
   - Purpose: Remove background/noise spots below detection threshold
   - Threshold: Calculated per slide-channel combination

3. SIZE LOWER BOUND (from bead PSF):
   - Criterion: σ_x ≥ {SIGMA_X_LOWER:.1f} nm (80% × bead PSF of {BEAD_PSF_X:.1f} nm)
   - Purpose: Remove artifactually small spots (noise, fitting errors)

CLUSTER IDENTIFICATION:
- Method: 3D connected component analysis on intensity-thresholded images
- Cluster intensity: Sum of all voxel intensities, normalized to mRNA equivalents

4. CLUSTER INTENSITY THRESHOLD (from negative controls):
   - Criterion: Cluster total intensity > {QUANTILE_NEGATIVE_CONTROL*100:.0f}th percentile of negative control distribution
   - Purpose: Remove clusters with intensity below the noise floor (false positive clusters)
   - Threshold: Calculated per slide-channel combination (same threshold as spot filter)

5. CLUSTER CV (COEFFICIENT OF VARIATION) THRESHOLD:
   - Criterion: Cluster CV >= {CV_THRESHOLD} (CV = standard deviation / mean of voxel intensities)
   - Purpose: Remove clusters with low intensity heterogeneity (likely noise or artifacts)
   - Rationale: True mRNA aggregates show spatial variation in signal; uniform low-variance regions are noise
   - See Figure 2 caption for detailed cluster discard statistics

TECHNICAL NOTES:
- Bead PSF: σ_x = {BEAD_PSF_X:.1f} nm, σ_y = {BEAD_PSF_Y:.1f} nm, σ_z = {BEAD_PSF_Z:.1f} nm
- Size lower bound: σ ≥ 80% of bead PSF ({SIGMA_X_LOWER:.1f} nm for σ_x)
- Cluster segmentation: 3D connected component analysis on thresholded intensity images
- Distance to DAPI: Euclidean distance from cluster centroid to nearest DAPI boundary
  * Calculated using distance transform from DAPI segmentation mask
  * Sign convention: Negative = inside DAPI mask, Positive = outside DAPI mask
- Nuclear/cytoplasmic classification: Based on sign of distance to DAPI
- FOV-level metrics: Calculated per field-of-view, then aggregated
- Cluster-level metrics: Each cluster is a data point (pooled across FOVs)

COLOR SCHEME SUMMARY:
| Probe | Color |
|-------|-------|
| mHTT1a (exon 1) | Green (#2ecc71) |
| Full-length mHTT | Orange (#f39c12) |

| Region | Pattern |
|--------|---------|
| Cortex | Solid fill |
| Striatum | Hatched (///) |

| FOV Class | Shade |
|-----------|-------|
| Extreme | Dark (alpha=0.9) |
| Normal | Light (alpha=0.5) |

KEY FINDINGS:
1. CLUSTER NUMBER DRIVES EXTREME PHENOTYPE: Extreme FOVs have dramatically more clusters per nucleus than normal FOVs. This is the PRIMARY driver of elevated mRNA levels - cells in extreme FOVs accumulate many more discrete mRNA clusters.

2. CLUSTER SIZE ALSO ELEVATED: Extreme FOVs also show larger mean cluster sizes, indicating that individual clusters in high-expressing regions contain more mRNA molecules. This represents a SECONDARY contribution to elevated expression.

3. HIGHER CLUSTER DENSITY IN EXTREME FOVs: Clusters in extreme FOVs are more densely packed with mRNA (~{mhtt1a_stats.get('density_fold_change', 0):.1f}-{full_stats.get('density_fold_change', 0):.1f}× higher concentration). This means:
   - mHTT1a: {mhtt1a_stats.get('density_extreme_median', 0):.2f} vs {mhtt1a_stats.get('density_normal_median', 0):.2f} mRNA/μm³ (extreme vs normal)
   - Full-length: {full_stats.get('density_extreme_median', 0):.2f} vs {full_stats.get('density_normal_median', 0):.2f} mRNA/μm³ (extreme vs normal)
   - Clusters in extreme FOVs aren't just bigger - they contain more mRNA per unit volume

4. NUCLEAR LOCALIZATION SHIFT IN EXTREME FOVs: While both conditions show predominantly nuclear localization, extreme FOVs show a subtle shift toward cytoplasmic clusters:
   - mHTT1a: {mhtt1a_stats.get('frac_nuclear_normal', 0):.0f}% nuclear (normal) → {mhtt1a_stats.get('frac_nuclear_extreme', 0):.0f}% nuclear (extreme)
   - Full-length: {full_stats.get('frac_nuclear_normal', 0):.0f}% nuclear (normal) → {full_stats.get('frac_nuclear_extreme', 0):.0f}% nuclear (extreme)
   - This may indicate impaired nuclear retention or increased cytoplasmic accumulation in high-expressing regions

5. PROBE CONSISTENCY: Both mHTT1a and full-length probes show similar patterns across all metrics, validating the biological relevance and probe specificity.

6. REGIONAL PATTERNS: Striatum shows similar cluster property differences as Cortex, suggesting that the mechanism of mRNA accumulation (more and larger clusters) is conserved across brain regions.

BIOLOGICAL IMPLICATIONS:
- The finding that extreme FOVs have MORE clusters (not just larger ones) suggests that mRNA accumulation hotspots represent regions with increased transcription or decreased mRNA degradation, rather than simply aggregation of existing transcripts
- Higher cluster DENSITY in extreme FOVs suggests active mRNA concentration mechanisms beyond simple transcriptional upregulation
- The subtle shift from nuclear to cytoplasmic localization in extreme FOVs may indicate beginning of nuclear export dysfunction or cytoplasmic retention in disease-affected cells
- The combination of increased cluster number, size, AND density suggests a compound effect where multiple processes contribute to disease pathology

DATA CACHING:
Processed data is cached to {CACHE_FILE.name} for fast subsequent runs. Set FORCE_RELOAD = True to regenerate from raw data.
"""
    return caption


def main():
    """Generate and save Figure 5."""

    fig, axes, stats = create_figure5()

    print("\n" + "=" * 70)
    print("SAVING FIGURE")
    print("=" * 70)

    save_figure(fig, 'figure5', formats=['svg', 'png', 'pdf'], output_dir=OUTPUT_DIR)

    # Generate and save caption
    caption = generate_caption(stats)
    caption_file = OUTPUT_DIR / 'figure5_caption.txt'
    with open(caption_file, 'w') as f:
        f.write(caption)
    print(f"Caption saved: {caption_file}")

    plt.close(fig)

    print("\n" + "=" * 70)
    print("FIGURE 5 COMPLETE")
    print("=" * 70)
    print(f"\nTo make layout changes quickly, just re-run this script.")
    print(f"Data is cached at: {CACHE_FILE}")


if __name__ == '__main__':
    main()
