"""
Figure 4 - FOV Extremes Analysis

Layout (matching PDF comments):
    Row 1: A (WT vs Q111 FOV comparison images - WT low expression left, Q111 high expression right)
           B (mHTT1a probe: distribution histogram)
           C (full-length mHTT probe: distribution histogram)
    Row 2: D (age breakdown - all channels)
           E (atlas coordinate breakdown - all channels)
    Row 3: F (mouse ID breakdown - all channels, full width)

Panel A shows 3 pairs of FOVs (columns: WT left, Q111 right) demonstrating
the contrast between low WT expression and elevated Q111 expression.
Expression levels annotated in green (mHTT1a) and orange (full-length mHTT).

Data sources: fig_fov_extremes_story.py analysis

Data caching: Processed data is cached to disk for fast layout iterations.
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import mannwhitneyu, gaussian_kde
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
    OUTPUT_DIR_COMPREHENSIVE,
    EXCLUDED_SLIDES,
    MOUSE_LABEL_MAP_Q111,
    MOUSE_LABEL_MAP_WT,
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

# Output and cache directories
OUTPUT_DIR = Path(__file__).parent / "output"
CACHE_DIR = OUTPUT_DIR / 'cache'
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = CACHE_DIR / 'figure4_data.pkl'

# Set to True to force data reload
FORCE_RELOAD = False

# Import colors from figure 3 scheme
from results_config import CHANNEL_COLORS

# Color scheme matching Figure 3
COLOR_MHTT1A_Q111 = CHANNEL_COLORS.get('green', '#2ecc71')  # Green for Q111 mHTT1a
COLOR_FULL_Q111 = CHANNEL_COLORS.get('orange', '#f39c12')   # Orange for Q111 full-length
COLOR_MHTT1A_WT = '#3498db'   # Blue for WT mHTT1a
COLOR_FULL_WT = '#9b59b6'     # Purple for WT full-length

# Darker shades for extreme FOVs (per channel) - distinguishes extreme by channel/region
COLOR_MHTT1A_Q111_EXTREME = '#1a5c2e'  # Dark green for Q111 mHTT1a extreme
COLOR_FULL_Q111_EXTREME = '#a35c00'    # Dark orange for Q111 full-length extreme


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_and_process_data():
    """Load and process all data for Figure 4. Returns cached data if available."""

    if CACHE_FILE.exists() and not FORCE_RELOAD:
        print(f"Loading cached data from {CACHE_FILE}")
        with open(CACHE_FILE, 'rb') as f:
            return pickle.load(f)

    print("\n" + "=" * 70)
    print("LOADING DATA FOR FIGURE 4")
    print("=" * 70)

    # Load FOV-level data
    print("Loading FOV-level data...")
    df_fov = pd.read_csv(OUTPUT_DIR_COMPREHENSIVE / 'fov_level_data.csv')

    # Filter for experimental data and exclude bad slides
    df_exp = df_fov[df_fov['Mouse_Model'].isin(['Q111', 'Wildtype'])].copy()
    df_exp = df_exp[~df_exp['Slide'].isin(EXCLUDED_SLIDES)]

    print(f"  Total FOVs after exclusions: {len(df_exp)}")
    print(f"  Q111 FOVs: {len(df_exp[df_exp['Mouse_Model'] == 'Q111'])}")
    print(f"  Wildtype FOVs: {len(df_exp[df_exp['Mouse_Model'] == 'Wildtype'])}")

    # Compute thresholds and statistics
    print("Computing thresholds and statistics...")

    thresholds = {}
    distribution_stats = {}

    for ch in ['mHTT1a', 'full-length mHTT']:
        # Map channel names (note: CSV uses 'full-length mHTT' with hyphen)
        ch_data = 'mHTT1a' if ch == 'mHTT1a' else 'full-length mHTT'

        for region in ['Cortex', 'Striatum']:
            # Get data
            q111_data = df_exp[(df_exp['Mouse_Model'] == 'Q111') &
                               (df_exp['Channel'] == ch_data) &
                               (df_exp['Region'] == region)]['Clustered_mRNA_per_Cell'].dropna().values

            wt_data = df_exp[(df_exp['Mouse_Model'] == 'Wildtype') &
                             (df_exp['Channel'] == ch_data) &
                             (df_exp['Region'] == region)]['Clustered_mRNA_per_Cell'].dropna().values

            if len(q111_data) == 0 or len(wt_data) == 0:
                continue

            # Calculate WT P95 threshold
            wt_p95 = np.percentile(wt_data, 95)
            thresholds[(ch, region)] = wt_p95

            # Count extreme FOVs
            n_q111_extreme = np.sum(q111_data > wt_p95)
            n_q111_total = len(q111_data)
            frac_extreme = n_q111_extreme / n_q111_total

            # Statistics
            u_stat, p_mwu = mannwhitneyu(q111_data, wt_data, alternative='two-sided')

            distribution_stats[(ch, region)] = {
                'q111_data': q111_data,
                'wt_data': wt_data,
                'wt_p95': wt_p95,
                'n_q111_extreme': n_q111_extreme,
                'n_q111_total': n_q111_total,
                'frac_extreme': frac_extreme,
                'q111_median': np.median(q111_data),
                'wt_median': np.median(wt_data),
                'q111_iqr': np.percentile(q111_data, 75) - np.percentile(q111_data, 25),
                'wt_iqr': np.percentile(wt_data, 75) - np.percentile(wt_data, 25),
                'p_mwu': p_mwu
            }

            print(f"  {ch} - {region}: {n_q111_extreme}/{n_q111_total} extreme ({100*frac_extreme:.1f}%)")

    # Compute age breakdown data (Q111 and WT)
    print("Computing age breakdown data...")
    age_data = {}
    for ch in ['mHTT1a', 'full-length mHTT']:
        ch_data = 'mHTT1a' if ch == 'mHTT1a' else 'full-length mHTT'
        for region in ['Cortex', 'Striatum']:
            wt_p95 = thresholds.get((ch, region))
            if wt_p95 is None:
                continue

            # Q111 data
            df_q111 = df_exp[(df_exp['Mouse_Model'] == 'Q111') &
                             (df_exp['Channel'] == ch_data) &
                             (df_exp['Region'] == region)]

            ages = sorted(df_q111['Age'].unique())
            n_extreme_list = []
            n_total_list = []

            for age in ages:
                df_age = df_q111[df_q111['Age'] == age]
                n_total = len(df_age)
                n_extreme = np.sum(df_age['Clustered_mRNA_per_Cell'] > wt_p95)
                n_extreme_list.append(n_extreme)
                n_total_list.append(n_total)

            # WT data - calculate actual extreme counts per age
            df_wt = df_exp[(df_exp['Mouse_Model'] == 'Wildtype') &
                           (df_exp['Channel'] == ch_data) &
                           (df_exp['Region'] == region)]

            wt_total_list = []
            wt_extreme_list = []
            for age in ages:
                df_age_wt = df_wt[df_wt['Age'] == age]
                wt_total_list.append(len(df_age_wt))
                wt_extreme = np.sum(df_age_wt['Clustered_mRNA_per_Cell'] > wt_p95)
                wt_extreme_list.append(wt_extreme)

            age_data[(ch, region)] = {
                'ages': ages,
                'n_extreme': n_extreme_list,
                'n_total': n_total_list,
                'wt_total': wt_total_list,
                'wt_extreme': wt_extreme_list
            }

    # Compute atlas breakdown data (Q111 and WT)
    print("Computing atlas breakdown data...")
    atlas_data = {}
    for ch in ['mHTT1a', 'full-length mHTT']:
        ch_data = 'mHTT1a' if ch == 'mHTT1a' else 'full-length mHTT'
        for region in ['Cortex', 'Striatum']:
            wt_p95 = thresholds.get((ch, region))
            if wt_p95 is None:
                continue

            # Q111 data
            df_q111 = df_exp[(df_exp['Mouse_Model'] == 'Q111') &
                             (df_exp['Channel'] == ch_data) &
                             (df_exp['Region'] == region)]

            atlas_coords = sorted(df_q111['Brain_Atlas_Coord'].unique())
            n_extreme_list = []
            n_total_list = []

            for atlas in atlas_coords:
                df_atlas = df_q111[df_q111['Brain_Atlas_Coord'] == atlas]
                n_total = len(df_atlas)
                n_extreme = np.sum(df_atlas['Clustered_mRNA_per_Cell'] > wt_p95)
                n_extreme_list.append(n_extreme)
                n_total_list.append(n_total)

            # WT data - calculate actual extreme counts per coordinate
            df_wt = df_exp[(df_exp['Mouse_Model'] == 'Wildtype') &
                           (df_exp['Channel'] == ch_data) &
                           (df_exp['Region'] == region)]

            wt_total_list = []
            wt_extreme_list = []
            for atlas in atlas_coords:
                df_atlas_wt = df_wt[df_wt['Brain_Atlas_Coord'] == atlas]
                wt_total_list.append(len(df_atlas_wt))
                wt_extreme = np.sum(df_atlas_wt['Clustered_mRNA_per_Cell'] > wt_p95)
                wt_extreme_list.append(wt_extreme)

            atlas_data[(ch, region)] = {
                'coords': atlas_coords,
                'n_extreme': n_extreme_list,
                'n_total': n_total_list,
                'wt_total': wt_total_list,
                'wt_extreme': wt_extreme_list
            }

    # Compute mouse ID breakdown data (Q111 and WT) with ages
    print("Computing mouse ID breakdown data...")
    mouse_data = {}
    for ch in ['mHTT1a', 'full-length mHTT']:
        ch_data = 'mHTT1a' if ch == 'mHTT1a' else 'full-length mHTT'
        for region in ['Cortex', 'Striatum']:
            wt_p95 = thresholds.get((ch, region))
            if wt_p95 is None:
                continue

            # Q111 data
            df_q111 = df_exp[(df_exp['Mouse_Model'] == 'Q111') &
                             (df_exp['Channel'] == ch_data) &
                             (df_exp['Region'] == region)]

            mouse_ids = sorted(df_q111['Mouse_ID'].unique())
            n_extreme_list = []
            n_total_list = []
            mouse_ages = []  # Store age for each mouse

            for mouse_id in mouse_ids:
                df_mouse = df_q111[df_q111['Mouse_ID'] == mouse_id]
                n_total = len(df_mouse)
                n_extreme = np.sum(df_mouse['Clustered_mRNA_per_Cell'] > wt_p95)
                n_extreme_list.append(n_extreme)
                n_total_list.append(n_total)
                # Get the age for this mouse (should be same for all FOVs of a mouse)
                mouse_ages.append(df_mouse['Age'].iloc[0] if len(df_mouse) > 0 else np.nan)

            # WT data - calculate actual extreme counts per mouse
            df_wt = df_exp[(df_exp['Mouse_Model'] == 'Wildtype') &
                           (df_exp['Channel'] == ch_data) &
                           (df_exp['Region'] == region)]

            wt_mouse_ids = sorted(df_wt['Mouse_ID'].unique())
            wt_total_list = []
            wt_extreme_list = []
            wt_ages = []  # Store age for each WT mouse
            for mouse_id in wt_mouse_ids:
                df_mouse_wt = df_wt[df_wt['Mouse_ID'] == mouse_id]
                wt_total_list.append(len(df_mouse_wt))
                wt_extreme = np.sum(df_mouse_wt['Clustered_mRNA_per_Cell'] > wt_p95)
                wt_extreme_list.append(wt_extreme)
                wt_ages.append(df_mouse_wt['Age'].iloc[0] if len(df_mouse_wt) > 0 else np.nan)

            mouse_data[(ch, region)] = {
                'mouse_ids': mouse_ids,
                'n_extreme': n_extreme_list,
                'n_total': n_total_list,
                'mouse_ages': mouse_ages,  # NEW: ages for Q111 mice
                'wt_mouse_ids': wt_mouse_ids,
                'wt_total': wt_total_list,
                'wt_extreme': wt_extreme_list,
                'wt_ages': wt_ages,  # NEW: ages for WT mice
            }

    cache_data = {
        'thresholds': thresholds,
        'distribution_stats': distribution_stats,
        'age_data': age_data,
        'atlas_data': atlas_data,
        'mouse_data': mouse_data,
    }

    # Save cache
    print(f"Saving cache to {CACHE_FILE}")
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(cache_data, f)

    return cache_data


# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def plot_distribution(ax, stats, channel, region):
    """Plot overall distribution using Figure 3 color scheme."""
    cfg = FigureConfig

    q111_data = stats['q111_data']
    wt_data = stats['wt_data']
    wt_p95 = stats['wt_p95']

    # Select colors based on channel - same as Figure 3
    if channel == 'mHTT1a':
        color_q111 = COLOR_MHTT1A_Q111  # Green
        color_wt = COLOR_MHTT1A_WT      # Blue
    else:
        color_q111 = COLOR_FULL_Q111    # Orange
        color_wt = COLOR_FULL_WT        # Purple

    # Create histogram bins
    max_val = np.percentile(q111_data, 99.5)
    bins = np.linspace(0, max_val, 40)

    # Plot WT distribution
    ax.hist(wt_data, bins=bins, alpha=0.7, color=color_wt,
            label=f'WT (n={len(wt_data)})',
            edgecolor='black', linewidth=0.5, density=True, zorder=5)

    # Plot ALL Q111 in same color (matching Figure 3)
    ax.hist(q111_data, bins=bins, alpha=0.6, color=color_q111,
            label=f'Q111 (n={len(q111_data)})',
            edgecolor='black', linewidth=0.5, density=True, zorder=3)

    # Mark WT P95 threshold
    ax.axvline(wt_p95, color='black', linestyle='--', linewidth=2,
               label=f'WT P95={wt_p95:.1f}')

    ax.set_xlabel('Clustered mRNA/nucleus', fontsize=cfg.FONT_SIZE_AXIS_LABEL)
    ax.set_ylabel('Density', fontsize=cfg.FONT_SIZE_AXIS_LABEL)

    channel_label = 'mHTT1a' if channel == 'mHTT1a' else 'full-length mHTT'
    ax.set_title(f'{channel_label} - {region}', fontsize=cfg.FONT_SIZE_TITLE, fontweight='bold')
    ax.legend(fontsize=cfg.FONT_SIZE_LEGEND - 1, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')


def plot_combined_distribution(ax, stats_cortex, stats_striatum, channel):
    """Plot combined distribution for both regions using Figure 3 color scheme.

    Bar plots with a final ">X" bin to compress the tail.
    Cortex = solid bars, Striatum = hatched bars (like Figure 3)
    Q111 = green/orange, WT = blue/purple
    """
    cfg = FigureConfig

    # Select colors based on channel - same as Figure 3
    if channel == 'mHTT1a':
        color_q111 = COLOR_MHTT1A_Q111  # Green
        color_wt = COLOR_MHTT1A_WT      # Blue
        tail_cutoff = 40  # Cutoff for tail bin
    else:
        color_q111 = COLOR_FULL_Q111    # Orange
        color_wt = COLOR_FULL_WT        # Purple
        tail_cutoff = 50  # Cutoff for tail bin

    # Get all data
    q111_cortex = stats_cortex['q111_data']
    q111_striatum = stats_striatum['q111_data']
    wt_cortex = stats_cortex['wt_data']
    wt_striatum = stats_striatum['wt_data']

    # Get WT P95 thresholds
    wt_p95_cortex = stats_cortex['wt_p95']
    wt_p95_striatum = stats_striatum['wt_p95']

    # Create bins up to cutoff, then one final ">cutoff" bin
    n_bins = 10  # Wider bins for better visualization
    bin_width_val = tail_cutoff / n_bins
    bins = np.linspace(0, tail_cutoff, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Add position for the ">X" bin (with extra spacing)
    tail_bin_center = tail_cutoff + bin_width_val * 2
    all_bin_centers = np.append(bin_centers, tail_bin_center)

    bar_width = bin_width_val * 0.22  # Each bar takes 22% of bin width

    def compute_hist_with_tail(data, bins, cutoff):
        """Compute histogram counts with a tail bin for values > cutoff."""
        # Regular histogram for values <= cutoff
        data_below = data[data <= cutoff]
        counts, _ = np.histogram(data_below, bins=bins)
        # Count values above cutoff
        tail_count = np.sum(data > cutoff)
        # Normalize to density (divide by total count and bin width)
        total = len(data)
        density = counts / (total * bin_width_val)
        tail_density = tail_count / (total * bin_width_val)
        return np.append(density, tail_density)

    # Compute histogram counts with tail bin
    q111_cortex_counts = compute_hist_with_tail(q111_cortex, bins, tail_cutoff)
    q111_striatum_counts = compute_hist_with_tail(q111_striatum, bins, tail_cutoff)
    wt_cortex_counts = compute_hist_with_tail(wt_cortex, bins, tail_cutoff)
    wt_striatum_counts = compute_hist_with_tail(wt_striatum, bins, tail_cutoff)

    # Plot grouped bars: 4 bars per bin
    offsets = [-1.5 * bar_width, -0.5 * bar_width, 0.5 * bar_width, 1.5 * bar_width]

    # Q111 Cortex (solid)
    ax.bar(all_bin_centers + offsets[0], q111_cortex_counts, width=bar_width,
           color=color_q111, alpha=0.8, edgecolor='black', linewidth=0.5,
           label=f'Q111 Cortex (n={len(q111_cortex)})')

    # Q111 Striatum (hatched)
    ax.bar(all_bin_centers + offsets[1], q111_striatum_counts, width=bar_width,
           color=color_q111, alpha=0.8, edgecolor='black', linewidth=0.5,
           hatch='///', label=f'Q111 Striatum (n={len(q111_striatum)})')

    # WT Cortex (solid)
    ax.bar(all_bin_centers + offsets[2], wt_cortex_counts, width=bar_width,
           color=color_wt, alpha=0.8, edgecolor='black', linewidth=0.5,
           label=f'WT Cortex (n={len(wt_cortex)})')

    # WT Striatum (hatched)
    ax.bar(all_bin_centers + offsets[3], wt_striatum_counts, width=bar_width,
           color=color_wt, alpha=0.8, edgecolor='black', linewidth=0.5,
           hatch='///', label=f'WT Striatum (n={len(wt_striatum)})')

    # Add vertical line to separate tail bin
    ax.axvline(tail_cutoff, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)

    # Add WT P95 threshold lines
    # Use average of Cortex and Striatum P95 for simplicity, or plot both
    wt_p95_avg = (wt_p95_cortex + wt_p95_striatum) / 2
    ax.axvline(wt_p95_avg, color='black', linestyle='--', linewidth=2,
               label=f'WT P95={wt_p95_avg:.1f}', zorder=15)

    ax.set_xlabel('Clustered mRNA/nucleus', fontsize=cfg.FONT_SIZE_AXIS_LABEL)
    ax.set_ylabel('prob. Density', fontsize=cfg.FONT_SIZE_AXIS_LABEL)
    ax.set_xlim(-bin_width_val, tail_bin_center + bin_width_val * 2)

    # Custom x-ticks with ">X" label for tail
    tick_positions = [0, tail_cutoff/2, tail_cutoff, tail_bin_center]
    tick_labels = ['0', f'{int(tail_cutoff/2)}', '', f'>{int(tail_cutoff)}']
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=cfg.FONT_SIZE_AXIS_TICK)

    channel_label = 'mHTT1a' if channel == 'mHTT1a' else 'full-length mHTT'
    ax.set_title(f'{channel_label}', fontsize=cfg.FONT_SIZE_TITLE, fontweight='bold')
    ax.legend(fontsize=cfg.FONT_SIZE_LEGEND - 1, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')


def plot_age_breakdown(ax, age_info, channel, region):
    """Plot age breakdown as stacked bars."""
    cfg = FigureConfig

    ages = age_info['ages']
    n_extreme = age_info['n_extreme']
    n_total = age_info['n_total']

    x_pos = np.arange(len(ages))
    n_normal = [t - e for t, e in zip(n_total, n_extreme)]
    frac_list = [e / t if t > 0 else 0 for e, t in zip(n_extreme, n_total)]

    # Stacked bar
    ax.bar(x_pos, n_normal, width=0.6, color=COLOR_Q111,
           alpha=0.6, label='Normal', edgecolor='darkgreen')
    ax.bar(x_pos, n_extreme, width=0.6, bottom=n_normal,
           color=COLOR_Q111_EXTREME, alpha=0.8, label='Extreme',
           edgecolor='black', hatch='///')

    # Add percentage labels
    max_tot = max(n_total) if n_total else 1
    for i, (ext, tot, frac) in enumerate(zip(n_extreme, n_total, frac_list)):
        ax.text(i, tot + max_tot * 0.02, f'{100*frac:.0f}%',
                ha='center', va='bottom', fontsize=cfg.FONT_SIZE_AXIS_TICK - 1,
                fontweight='bold', color=COLOR_Q111_EXTREME)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{int(a)}mo' for a in ages], fontsize=cfg.FONT_SIZE_AXIS_TICK)
    ax.set_xlabel('Age', fontsize=cfg.FONT_SIZE_AXIS_LABEL)
    ax.set_ylabel('# FOVs', fontsize=cfg.FONT_SIZE_AXIS_LABEL)

    channel_label = 'mHTT1a' if channel == 'mHTT1a' else 'full-length mHTT'
    ax.set_title(f'{channel_label} - {region}', fontsize=cfg.FONT_SIZE_TITLE - 1, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')


def plot_atlas_breakdown(ax, atlas_info, channel, region):
    """Plot atlas coordinate breakdown."""
    cfg = FigureConfig

    coords = atlas_info['coords']
    n_extreme = atlas_info['n_extreme']
    n_total = atlas_info['n_total']

    x_pos = np.arange(len(coords))
    frac_list = [e / t if t > 0 else 0 for e, t in zip(n_extreme, n_total)]

    # Color by fraction
    colors = plt.cm.RdYlGn_r([f for f in frac_list])
    ax.bar(x_pos, n_extreme, color=colors, alpha=0.8,
           edgecolor='black', linewidth=0.5)

    # Add percentage labels
    max_ext = max(n_extreme) if max(n_extreme) > 0 else 1
    for i, (ext, frac) in enumerate(zip(n_extreme, frac_list)):
        if ext > 0:
            ax.text(i, ext + max_ext * 0.02, f'{100*frac:.0f}%',
                    ha='center', va='bottom', fontsize=cfg.FONT_SIZE_AXIS_TICK - 2,
                    fontweight='bold')

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{int(c)}' for c in coords], fontsize=cfg.FONT_SIZE_AXIS_TICK - 1, rotation=45)
    ax.set_xlabel('Atlas coord (25μm)', fontsize=cfg.FONT_SIZE_AXIS_LABEL)
    ax.set_ylabel('# Extreme FOVs', fontsize=cfg.FONT_SIZE_AXIS_LABEL)

    channel_label = 'mHTT1a' if channel == 'mHTT1a' else 'full-length mHTT'
    ax.set_title(f'{channel_label} - {region}', fontsize=cfg.FONT_SIZE_TITLE - 1, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')


# ══════════════════════════════════════════════════════════════════════════════
# MAIN FIGURE CREATION
# ══════════════════════════════════════════════════════════════════════════════

def plot_age_breakdown_all_channels(ax, age_data):
    """Plot age breakdown showing % extreme FOVs, with WT on right."""
    cfg = FigureConfig

    # Get data for all combinations
    age_mhtt1a_cortex = age_data.get(('mHTT1a', 'Cortex'))
    age_mhtt1a_striatum = age_data.get(('mHTT1a', 'Striatum'))
    age_full_cortex = age_data.get(('full-length mHTT', 'Cortex'))
    age_full_striatum = age_data.get(('full-length mHTT', 'Striatum'))

    if not all([age_mhtt1a_cortex, age_mhtt1a_striatum, age_full_cortex, age_full_striatum]):
        return

    # Q111 ages
    q111_ages = age_mhtt1a_cortex['ages']
    n_q111 = len(q111_ages)

    # WT ages (where wt_total > 0 for any channel/region)
    def get_wt_ages_data(age_info):
        """Get WT ages and their extreme percentages."""
        wt_ages = []
        wt_pct = {}
        for i, age in enumerate(age_info['ages']):
            wt_total = age_info.get('wt_total', [])[i] if i < len(age_info.get('wt_total', [])) else 0
            wt_extreme = age_info.get('wt_extreme', [])[i] if i < len(age_info.get('wt_extreme', [])) else 0
            if wt_total > 0:
                wt_ages.append(age)
                wt_pct[age] = 100 * wt_extreme / wt_total
        return wt_ages, wt_pct

    wt_ages_mc, wt_pct_mc_dict = get_wt_ages_data(age_mhtt1a_cortex)
    wt_ages_ms, wt_pct_ms_dict = get_wt_ages_data(age_mhtt1a_striatum)
    wt_ages_fc, wt_pct_fc_dict = get_wt_ages_data(age_full_cortex)
    wt_ages_fs, wt_pct_fs_dict = get_wt_ages_data(age_full_striatum)
    wt_ages = sorted(set(wt_ages_mc) | set(wt_ages_ms) | set(wt_ages_fc) | set(wt_ages_fs))
    n_wt = len(wt_ages)

    # Combined: Q111 ages first, then WT ages
    all_ages = list(q111_ages) + wt_ages
    n_ages = len(all_ages)
    x = np.arange(n_ages)
    width = 0.2

    # Calculate percentage of extreme FOVs for Q111
    def get_extreme_pct(age_info):
        n_total = age_info['n_total']
        n_extreme = age_info['n_extreme']
        pct_extreme = [100 * e / t if t > 0 else 0 for e, t in zip(n_extreme, n_total)]
        return pct_extreme

    # Q111 data (for Q111 ages only, then zeros for WT section)
    pct_ext_mc = get_extreme_pct(age_mhtt1a_cortex) + [0] * n_wt
    pct_ext_ms = get_extreme_pct(age_mhtt1a_striatum) + [0] * n_wt
    pct_ext_fc = get_extreme_pct(age_full_cortex) + [0] * n_wt
    pct_ext_fs = get_extreme_pct(age_full_striatum) + [0] * n_wt

    # WT data - use actual extreme percentages (not hardcoded 5%)
    wt_pct_mc = [0] * n_q111 + [wt_pct_mc_dict.get(a, 0) for a in wt_ages]
    wt_pct_ms = [0] * n_q111 + [wt_pct_ms_dict.get(a, 0) for a in wt_ages]
    wt_pct_fc = [0] * n_q111 + [wt_pct_fc_dict.get(a, 0) for a in wt_ages]
    wt_pct_fs = [0] * n_q111 + [wt_pct_fs_dict.get(a, 0) for a in wt_ages]

    offsets = [-1.5*width, -0.5*width, 0.5*width, 1.5*width]

    # Q111 bars (green/orange)
    ax.bar(x + offsets[0], pct_ext_mc, width, color=COLOR_MHTT1A_Q111,
           edgecolor='black', linewidth=0.5, label='mHTT1a Cortex')
    ax.bar(x + offsets[1], pct_ext_ms, width, color=COLOR_MHTT1A_Q111,
           edgecolor='black', linewidth=0.5, hatch='///', label='mHTT1a Striatum')
    ax.bar(x + offsets[2], pct_ext_fc, width, color=COLOR_FULL_Q111,
           edgecolor='black', linewidth=0.5, label='full mHTT Cortex')
    ax.bar(x + offsets[3], pct_ext_fs, width, color=COLOR_FULL_Q111,
           edgecolor='black', linewidth=0.5, hatch='///', label='full mHTT Striatum')

    # WT bars (blue/purple) - at ~5%
    ax.bar(x + offsets[0], wt_pct_mc, width, color=COLOR_MHTT1A_WT,
           edgecolor='black', linewidth=0.5)
    ax.bar(x + offsets[1], wt_pct_ms, width, color=COLOR_MHTT1A_WT,
           edgecolor='black', linewidth=0.5, hatch='///')
    ax.bar(x + offsets[2], wt_pct_fc, width, color=COLOR_FULL_WT,
           edgecolor='black', linewidth=0.5)
    ax.bar(x + offsets[3], wt_pct_fs, width, color=COLOR_FULL_WT,
           edgecolor='black', linewidth=0.5, hatch='///')

    # Add vertical separator between Q111 and WT
    if n_q111 > 0 and n_wt > 0:
        ax.axvline(n_q111 - 0.5, color='black', linestyle='--', linewidth=1.5, alpha=0.7)

    # X-axis labels
    x_labels = [f'{int(a)}mo' for a in q111_ages] + [f'{int(a)}mo' for a in wt_ages]
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=cfg.FONT_SIZE_AXIS_TICK)
    ax.set_xlabel('Age', fontsize=cfg.FONT_SIZE_AXIS_LABEL)
    ax.set_ylabel('% extreme FOVs', fontsize=cfg.FONT_SIZE_AXIS_LABEL)
    ax.set_ylim(0, 100)
    ax.set_title('Age breakdown', fontsize=cfg.FONT_SIZE_TITLE, fontweight='bold')
    ax.legend(fontsize=cfg.FONT_SIZE_LEGEND - 2, loc='upper left', ncol=2)
    ax.grid(True, alpha=0.3, axis='y')


def plot_atlas_breakdown_all_channels(ax, atlas_data):
    """Plot atlas coordinate breakdown showing % extreme FOVs, with WT on right."""
    cfg = FigureConfig

    # Get data for all combinations
    atlas_mhtt1a_cortex = atlas_data.get(('mHTT1a', 'Cortex'))
    atlas_mhtt1a_striatum = atlas_data.get(('mHTT1a', 'Striatum'))
    atlas_full_cortex = atlas_data.get(('full-length mHTT', 'Cortex'))
    atlas_full_striatum = atlas_data.get(('full-length mHTT', 'Striatum'))

    if not all([atlas_mhtt1a_cortex, atlas_mhtt1a_striatum, atlas_full_cortex, atlas_full_striatum]):
        return

    # Get Q111 coordinates
    q111_coords = sorted(set(atlas_mhtt1a_cortex['coords']) |
                         set(atlas_mhtt1a_striatum['coords']) |
                         set(atlas_full_cortex['coords']) |
                         set(atlas_full_striatum['coords']))

    # Get WT coordinates (where wt_total > 0) and their actual extreme percentages
    def get_wt_coords_data(atlas_info):
        wt_coords = []
        wt_pct = {}
        for i, c in enumerate(atlas_info['coords']):
            wt_total = atlas_info.get('wt_total', [])[i] if i < len(atlas_info.get('wt_total', [])) else 0
            wt_extreme = atlas_info.get('wt_extreme', [])[i] if i < len(atlas_info.get('wt_extreme', [])) else 0
            if wt_total > 0:
                wt_coords.append(c)
                wt_pct[c] = 100 * wt_extreme / wt_total
        return wt_coords, wt_pct

    wt_coords_mc, wt_pct_mc_dict = get_wt_coords_data(atlas_mhtt1a_cortex)
    wt_coords_ms, wt_pct_ms_dict = get_wt_coords_data(atlas_mhtt1a_striatum)
    wt_coords_fc, wt_pct_fc_dict = get_wt_coords_data(atlas_full_cortex)
    wt_coords_fs, wt_pct_fs_dict = get_wt_coords_data(atlas_full_striatum)

    wt_coords = sorted(set(wt_coords_mc) | set(wt_coords_ms) | set(wt_coords_fc) | set(wt_coords_fs))

    # Combined: Q111 coords first, then WT coords
    n_q111 = len(q111_coords)
    n_wt = len(wt_coords)
    all_coords = q111_coords + wt_coords
    n_coords = len(all_coords)
    x = np.arange(n_coords)
    width = 0.2

    # Build aligned data for Q111 - only % extreme
    def get_aligned_q111_data(atlas_info, coords):
        coord_to_idx = {c: i for i, c in enumerate(atlas_info['coords'])}
        n_extreme = []
        n_total = []
        for c in coords:
            if c in coord_to_idx:
                idx = coord_to_idx[c]
                n_extreme.append(atlas_info['n_extreme'][idx])
                n_total.append(atlas_info['n_total'][idx])
            else:
                n_extreme.append(0)
                n_total.append(0)
        pct_extreme = [100 * e / t if t > 0 else 0 for e, t in zip(n_extreme, n_total)]
        return pct_extreme

    # Q111 data (for Q111 coords)
    pct_ext_mc = get_aligned_q111_data(atlas_mhtt1a_cortex, q111_coords) + [0] * n_wt
    pct_ext_ms = get_aligned_q111_data(atlas_mhtt1a_striatum, q111_coords) + [0] * n_wt
    pct_ext_fc = get_aligned_q111_data(atlas_full_cortex, q111_coords) + [0] * n_wt
    pct_ext_fs = get_aligned_q111_data(atlas_full_striatum, q111_coords) + [0] * n_wt

    # WT data - use actual extreme percentages (not hardcoded 5%)
    wt_pct_mc = [0] * n_q111 + [wt_pct_mc_dict.get(c, 0) for c in wt_coords]
    wt_pct_ms = [0] * n_q111 + [wt_pct_ms_dict.get(c, 0) for c in wt_coords]
    wt_pct_fc = [0] * n_q111 + [wt_pct_fc_dict.get(c, 0) for c in wt_coords]
    wt_pct_fs = [0] * n_q111 + [wt_pct_fs_dict.get(c, 0) for c in wt_coords]

    offsets = [-1.5*width, -0.5*width, 0.5*width, 1.5*width]

    # Q111 bars (green/orange)
    ax.bar(x + offsets[0], pct_ext_mc, width, color=COLOR_MHTT1A_Q111,
           edgecolor='black', linewidth=0.5, label='mHTT1a Cortex')
    ax.bar(x + offsets[1], pct_ext_ms, width, color=COLOR_MHTT1A_Q111,
           edgecolor='black', linewidth=0.5, hatch='///', label='mHTT1a Striatum')
    ax.bar(x + offsets[2], pct_ext_fc, width, color=COLOR_FULL_Q111,
           edgecolor='black', linewidth=0.5, label='full mHTT Cortex')
    ax.bar(x + offsets[3], pct_ext_fs, width, color=COLOR_FULL_Q111,
           edgecolor='black', linewidth=0.5, hatch='///', label='full mHTT Striatum')

    # WT bars (blue/purple) - at ~5%
    ax.bar(x + offsets[0], wt_pct_mc, width, color=COLOR_MHTT1A_WT,
           edgecolor='black', linewidth=0.5)
    ax.bar(x + offsets[1], wt_pct_ms, width, color=COLOR_MHTT1A_WT,
           edgecolor='black', linewidth=0.5, hatch='///')
    ax.bar(x + offsets[2], wt_pct_fc, width, color=COLOR_FULL_WT,
           edgecolor='black', linewidth=0.5)
    ax.bar(x + offsets[3], wt_pct_fs, width, color=COLOR_FULL_WT,
           edgecolor='black', linewidth=0.5, hatch='///')

    # Add vertical separator between Q111 and WT
    if n_q111 > 0 and n_wt > 0:
        ax.axvline(n_q111 - 0.5, color='black', linestyle='--', linewidth=1.5, alpha=0.7)

    # X-axis labels
    x_labels = [f'{int(c)}' for c in q111_coords] + [f'{int(c)}' for c in wt_coords]
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=cfg.FONT_SIZE_AXIS_TICK - 2, rotation=45)
    ax.set_xlabel('Atlas coord (25μm)', fontsize=cfg.FONT_SIZE_AXIS_LABEL)
    ax.set_ylabel('% extreme FOVs', fontsize=cfg.FONT_SIZE_AXIS_LABEL)
    ax.set_ylim(0, 100)
    ax.set_title('Atlas breakdown', fontsize=cfg.FONT_SIZE_TITLE, fontweight='bold')
    ax.legend(fontsize=cfg.FONT_SIZE_LEGEND - 2, loc='upper right', ncol=2)
    ax.grid(True, alpha=0.3, axis='y')


def plot_mouse_breakdown_all_channels(ax, mouse_data):
    """Plot mouse ID breakdown showing % extreme FOVs, grouped by age brackets.

    Uses anonymized IDs (Q#1, Q#2, W#1, W#2) to match Figure 3 format.
    Data is organized as: [Age 2mo: Q111 mice | WT mice] | [Age 6mo: Q111 mice | WT mice] | [Age 12mo: ...]
    Returns the mouse ID mappings for use in caption generation.
    """
    cfg = FigureConfig

    # Get data for all combinations
    mouse_mhtt1a_cortex = mouse_data.get(('mHTT1a', 'Cortex'))
    mouse_mhtt1a_striatum = mouse_data.get(('mHTT1a', 'Striatum'))
    mouse_full_cortex = mouse_data.get(('full-length mHTT', 'Cortex'))
    mouse_full_striatum = mouse_data.get(('full-length mHTT', 'Striatum'))

    if not all([mouse_mhtt1a_cortex, mouse_mhtt1a_striatum, mouse_full_cortex, mouse_full_striatum]):
        return {}, {}

    # Helper to extract numeric label for sorting
    def get_label_num(mouse_id, label_map):
        label = label_map.get(mouse_id, '#999')
        return int(label.replace('#', ''))

    # Build sub-index mapping: #1.1, #1.2, etc. for slides within same mouse
    def build_slide_sublabels(mouse_ids, id_map):
        """Create labels with sub-indices for each slide within a mouse."""
        sorted_ids = sorted(mouse_ids, key=lambda x: (get_label_num(x, id_map), x))
        sublabels = {}
        current_mouse = None
        sub_idx = 0
        for mid in sorted_ids:
            mouse_num = id_map.get(mid, '#?')
            if mouse_num != current_mouse:
                current_mouse = mouse_num
                sub_idx = 1
            else:
                sub_idx += 1
            sublabels[mid] = f"{mouse_num}.{sub_idx}"
        return sublabels

    # Build mouse_id -> age mapping from the data
    def build_mouse_age_map(mouse_info, is_wt=False):
        """Build a dictionary mapping mouse_id to age."""
        age_map = {}
        if is_wt:
            mouse_ids = mouse_info.get('wt_mouse_ids', [])
            ages = mouse_info.get('wt_ages', [])
        else:
            mouse_ids = mouse_info.get('mouse_ids', [])
            ages = mouse_info.get('mouse_ages', [])
        for i, mid in enumerate(mouse_ids):
            if i < len(ages):
                age_map[mid] = ages[i]
        return age_map

    # Build mouse_id -> pct_extreme mapping
    def build_mouse_pct_map(mouse_info, is_wt=False):
        """Build a dictionary mapping mouse_id to % extreme."""
        pct_map = {}
        if is_wt:
            mouse_ids = mouse_info.get('wt_mouse_ids', [])
            totals = mouse_info.get('wt_total', [])
            extremes = mouse_info.get('wt_extreme', [])
        else:
            mouse_ids = mouse_info.get('mouse_ids', [])
            totals = mouse_info.get('n_total', [])
            extremes = mouse_info.get('n_extreme', [])
        for i, mid in enumerate(mouse_ids):
            if i < len(totals) and totals[i] > 0:
                pct_map[mid] = 100 * extremes[i] / totals[i]
            else:
                pct_map[mid] = 0
        return pct_map

    # Collect all Q111 mouse IDs and their ages
    q111_mouse_ids_set = (set(mouse_mhtt1a_cortex['mouse_ids']) |
                          set(mouse_mhtt1a_striatum['mouse_ids']) |
                          set(mouse_full_cortex['mouse_ids']) |
                          set(mouse_full_striatum['mouse_ids']))

    # Collect all WT mouse IDs
    wt_mouse_ids_set = (set(mouse_mhtt1a_cortex.get('wt_mouse_ids', [])) |
                        set(mouse_mhtt1a_striatum.get('wt_mouse_ids', [])) |
                        set(mouse_full_cortex.get('wt_mouse_ids', [])) |
                        set(mouse_full_striatum.get('wt_mouse_ids', [])))

    # Use sublabel mappings (matching Figure 3 format: #1.1, #1.2, etc.)
    mouse_id_map_q111 = build_slide_sublabels(q111_mouse_ids_set, MOUSE_LABEL_MAP_Q111)
    mouse_id_map_wt = build_slide_sublabels(wt_mouse_ids_set, MOUSE_LABEL_MAP_WT)

    # Build age maps from all channel data (use first available)
    q111_age_map = build_mouse_age_map(mouse_mhtt1a_cortex, is_wt=False)
    wt_age_map = build_mouse_age_map(mouse_mhtt1a_cortex, is_wt=True)

    # Build pct_extreme maps for all channel-region combinations
    pct_maps = {
        ('mHTT1a', 'Cortex', 'Q111'): build_mouse_pct_map(mouse_mhtt1a_cortex, is_wt=False),
        ('mHTT1a', 'Striatum', 'Q111'): build_mouse_pct_map(mouse_mhtt1a_striatum, is_wt=False),
        ('full-length mHTT', 'Cortex', 'Q111'): build_mouse_pct_map(mouse_full_cortex, is_wt=False),
        ('full-length mHTT', 'Striatum', 'Q111'): build_mouse_pct_map(mouse_full_striatum, is_wt=False),
        ('mHTT1a', 'Cortex', 'WT'): build_mouse_pct_map(mouse_mhtt1a_cortex, is_wt=True),
        ('mHTT1a', 'Striatum', 'WT'): build_mouse_pct_map(mouse_mhtt1a_striatum, is_wt=True),
        ('full-length mHTT', 'Cortex', 'WT'): build_mouse_pct_map(mouse_full_cortex, is_wt=True),
        ('full-length mHTT', 'Striatum', 'WT'): build_mouse_pct_map(mouse_full_striatum, is_wt=True),
    }

    # Get all unique ages, sorted
    all_ages = sorted(set(q111_age_map.values()) | set(wt_age_map.values()))

    # Track positions and labels
    x_positions = []
    x_labels = []
    age_separators = []
    genotype_separators = []
    age_labels = []

    width = 0.2
    offsets = [-1.5*width, -0.5*width, 0.5*width, 1.5*width]
    pos = 0

    for age_idx, age in enumerate(all_ages):
        age_start = pos

        # Get Q111 mice for this age
        q111_mice_this_age = sorted(
            [m for m in q111_mouse_ids_set if q111_age_map.get(m) == age],
            key=lambda x: (get_label_num(x, MOUSE_LABEL_MAP_Q111), x)
        )

        # Plot Q111 mice for this age
        for mouse_id in q111_mice_this_age:
            mouse_label = mouse_id_map_q111.get(mouse_id, mouse_id)

            # Get pct_extreme for each channel-region
            pct_mc = pct_maps[('mHTT1a', 'Cortex', 'Q111')].get(mouse_id, 0)
            pct_ms = pct_maps[('mHTT1a', 'Striatum', 'Q111')].get(mouse_id, 0)
            pct_fc = pct_maps[('full-length mHTT', 'Cortex', 'Q111')].get(mouse_id, 0)
            pct_fs = pct_maps[('full-length mHTT', 'Striatum', 'Q111')].get(mouse_id, 0)

            # Plot bars for this mouse
            ax.bar(pos + offsets[0], pct_mc, width, color=COLOR_MHTT1A_Q111,
                   edgecolor='black', linewidth=0.5)
            ax.bar(pos + offsets[1], pct_ms, width, color=COLOR_MHTT1A_Q111,
                   edgecolor='black', linewidth=0.5, hatch='///')
            ax.bar(pos + offsets[2], pct_fc, width, color=COLOR_FULL_Q111,
                   edgecolor='black', linewidth=0.5)
            ax.bar(pos + offsets[3], pct_fs, width, color=COLOR_FULL_Q111,
                   edgecolor='black', linewidth=0.5, hatch='///')

            x_positions.append(pos)
            x_labels.append(f'Q{mouse_label}')
            pos += 1

        # Add genotype separator (dashed line between Q111 and WT within this age)
        if len(q111_mice_this_age) > 0:
            genotype_separators.append(pos - 0.5)

        # Get WT mice for this age
        wt_mice_this_age = sorted(
            [m for m in wt_mouse_ids_set if wt_age_map.get(m) == age],
            key=lambda x: (get_label_num(x, MOUSE_LABEL_MAP_WT), x)
        )

        # Plot WT mice for this age
        for mouse_id in wt_mice_this_age:
            mouse_label = mouse_id_map_wt.get(mouse_id, mouse_id)

            # Get pct_extreme for each channel-region
            pct_mc = pct_maps[('mHTT1a', 'Cortex', 'WT')].get(mouse_id, 0)
            pct_ms = pct_maps[('mHTT1a', 'Striatum', 'WT')].get(mouse_id, 0)
            pct_fc = pct_maps[('full-length mHTT', 'Cortex', 'WT')].get(mouse_id, 0)
            pct_fs = pct_maps[('full-length mHTT', 'Striatum', 'WT')].get(mouse_id, 0)

            # Plot bars for this mouse
            ax.bar(pos + offsets[0], pct_mc, width, color=COLOR_MHTT1A_WT,
                   edgecolor='black', linewidth=0.5)
            ax.bar(pos + offsets[1], pct_ms, width, color=COLOR_MHTT1A_WT,
                   edgecolor='black', linewidth=0.5, hatch='///')
            ax.bar(pos + offsets[2], pct_fc, width, color=COLOR_FULL_WT,
                   edgecolor='black', linewidth=0.5)
            ax.bar(pos + offsets[3], pct_fs, width, color=COLOR_FULL_WT,
                   edgecolor='black', linewidth=0.5, hatch='///')

            x_positions.append(pos)
            x_labels.append(f'W{mouse_label}')
            pos += 1

        # Store age label position (center of this age bracket)
        age_end = pos
        if age_start < age_end:  # Only add label if there's data for this age
            age_labels.append(((age_start + age_end - 1) / 2, f'{int(age)}mo'))

        # Add age separator (solid line) unless this is the last age
        if age_idx < len(all_ages) - 1 and (len(q111_mice_this_age) > 0 or len(wt_mice_this_age) > 0):
            age_separators.append(pos - 0.5)
            pos += 0.3  # Add spacing between age groups

    # Add dashed vertical lines between Q111 and WT within each age bracket
    for sep_pos in genotype_separators:
        ax.axvline(x=sep_pos, color='gray', linestyle='--', alpha=0.5, linewidth=1.0)

    # Add solid vertical lines between age brackets
    for sep_pos in age_separators:
        ax.axvline(x=sep_pos, color='black', linestyle='-', alpha=0.7, linewidth=1.5)

    # Add age bracket labels at the top
    for age_pos, age_label in age_labels:
        ax.text(age_pos, 95, age_label, ha='center', va='bottom',
                fontsize=cfg.FONT_SIZE_AXIS_LABEL, fontweight='bold')

    # Set x-ticks
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, fontsize=cfg.FONT_SIZE_AXIS_TICK - 1, rotation=45)

    # Add legend (only once, using dummy bars)
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=COLOR_MHTT1A_Q111, edgecolor='black', linewidth=0.5, label='mHTT1a Cortex'),
        plt.Rectangle((0, 0), 1, 1, facecolor=COLOR_MHTT1A_Q111, edgecolor='black', linewidth=0.5, hatch='///', label='mHTT1a Striatum'),
        plt.Rectangle((0, 0), 1, 1, facecolor=COLOR_FULL_Q111, edgecolor='black', linewidth=0.5, label='full mHTT Cortex'),
        plt.Rectangle((0, 0), 1, 1, facecolor=COLOR_FULL_Q111, edgecolor='black', linewidth=0.5, hatch='///', label='full mHTT Striatum'),
    ]
    ax.legend(handles=legend_handles, fontsize=cfg.FONT_SIZE_LEGEND - 2, loc='upper right', ncol=2)

    ax.set_xlabel('Mouse by Age', fontsize=cfg.FONT_SIZE_AXIS_LABEL)
    ax.set_ylabel('% extreme FOVs', fontsize=cfg.FONT_SIZE_AXIS_LABEL)
    ax.set_ylim(0, 100)
    ax.set_title('Mouse ID breakdown by age', fontsize=cfg.FONT_SIZE_TITLE, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Return the ID mappings for caption generation
    return mouse_id_map_q111, mouse_id_map_wt


def create_figure4():
    """Create Figure 4 with 3-row layout.

    Row 1: A (Placeholder), B (mHTT1a dist), C (full-length dist)
    Row 2: D (Age breakdown - all channels), E (Atlas breakdown - all channels)
    Row 3: F (Mouse ID breakdown - all channels)
    """
    cfg = FigureConfig

    # Load data
    data = load_and_process_data()
    distribution_stats = data['distribution_stats']
    age_data = data['age_data']
    atlas_data = data['atlas_data']
    mouse_data = data['mouse_data']

    # Figure dimensions - 3 rows (use proper page width from config)
    fig_width = cfg.PAGE_WIDTH_FULL
    fig_height = fig_width * 1.3  # Taller for 3 rows

    fig = plt.figure(figsize=(fig_width, fig_height))

    # 3-row layout
    # Row 1: 3 panels (A, B, C)
    # Row 2: 2 panels (D, E)
    # Row 3: 1 panel (F)
    main_gs = gridspec.GridSpec(
        3, 6,  # Use 6 columns for flexible layout
        figure=fig,
        left=cfg.SUBPLOT_LEFT + 0.02,
        right=cfg.SUBPLOT_RIGHT - 0.01,
        bottom=cfg.SUBPLOT_BOTTOM + 0.06,
        top=cfg.SUBPLOT_TOP - 0.02,
        hspace=0.45,
        wspace=0.4
    )

    axes = {}

    # Row 1: 3 panels spanning 2 columns each
    axes['A'] = fig.add_subplot(main_gs[0, 0:2])   # Placeholder
    axes['B'] = fig.add_subplot(main_gs[0, 2:4])   # mHTT1a distributions
    axes['C'] = fig.add_subplot(main_gs[0, 4:6], sharey=axes['B'])   # full-length distributions (shares y with B)

    # Row 2: D narrower (age has fewer x-ticks), E wider (atlas has many coords)
    axes['D'] = fig.add_subplot(main_gs[1, 0:2])   # Age breakdown (all channels)
    axes['E'] = fig.add_subplot(main_gs[1, 2:6], sharey=axes['D'])   # Atlas breakdown (shares y with D)

    # Row 3: F spans entire width (mouse ID breakdown)
    axes['F'] = fig.add_subplot(main_gs[2, 0:6])   # Mouse ID breakdown (all channels)

    # ══════════════════════════════════════════════════════════════════════════
    # FILL PANELS - ROW 1
    # ══════════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 70)
    print("CREATING PANELS")
    print("=" * 70)

    # Panel A: Placeholder
    print("  Creating panel A (placeholder)...")
    axes['A'].set_facecolor('#d3d3d3')
    axes['A'].text(0.5, 0.5, 'Panel A', ha='center', va='center',
                   fontsize=cfg.FONT_SIZE_TITLE, color='#666666',
                   transform=axes['A'].transAxes)
    axes['A'].set_xticks([])
    axes['A'].set_yticks([])

    # Panel B: mHTT1a distributions
    print("  Creating panel B (mHTT1a distributions)...")
    if ('mHTT1a', 'Cortex') in distribution_stats and ('mHTT1a', 'Striatum') in distribution_stats:
        plot_combined_distribution(
            axes['B'],
            distribution_stats[('mHTT1a', 'Cortex')],
            distribution_stats[('mHTT1a', 'Striatum')],
            'mHTT1a'
        )

    # Panel C: full-length mHTT distributions
    print("  Creating panel C (full-length mHTT distributions)...")
    if ('full-length mHTT', 'Cortex') in distribution_stats and ('full-length mHTT', 'Striatum') in distribution_stats:
        plot_combined_distribution(
            axes['C'],
            distribution_stats[('full-length mHTT', 'Cortex')],
            distribution_stats[('full-length mHTT', 'Striatum')],
            'full-length mHTT'
        )

    # Remove y-axis label from C (shares with B)
    axes['C'].set_ylabel('')

    # ══════════════════════════════════════════════════════════════════════════
    # FILL PANELS - ROW 2
    # ══════════════════════════════════════════════════════════════════════════

    # Panel D: Age breakdown (all channels merged)
    print("  Creating panel D (age breakdown - all channels)...")
    plot_age_breakdown_all_channels(axes['D'], age_data)

    # Panel E: Atlas breakdown (all channels merged)
    print("  Creating panel E (atlas breakdown - all channels)...")
    plot_atlas_breakdown_all_channels(axes['E'], atlas_data)

    # Remove y-axis label from E (shares with D)
    axes['E'].set_ylabel('')

    # ══════════════════════════════════════════════════════════════════════════
    # FILL PANELS - ROW 3
    # ══════════════════════════════════════════════════════════════════════════

    # Panel F: Mouse ID breakdown (all channels merged)
    print("  Creating panel F (mouse ID breakdown - all channels)...")
    mouse_id_map_q111, mouse_id_map_wt = plot_mouse_breakdown_all_channels(axes['F'], mouse_data)

    # ══════════════════════════════════════════════════════════════════════════
    # ADD PANEL LABELS
    # ══════════════════════════════════════════════════════════════════════════

    label_offset_x = -0.08
    label_offset_y = 0.02

    for label in ['A', 'B', 'C', 'D', 'E', 'F']:
        bbox = axes[label].get_position()
        fig.text(bbox.x0 + label_offset_x, bbox.y1 + label_offset_y, label,
                 fontsize=cfg.FONT_SIZE_PANEL_LABEL, fontweight=cfg.FONT_WEIGHT_PANEL_LABEL,
                 va='bottom', ha='left')

    # ══════════════════════════════════════════════════════════════════════════
    # STATISTICS
    # ══════════════════════════════════════════════════════════════════════════

    stats = {
        'distribution_stats': distribution_stats,
        'mouse_id_map_q111': mouse_id_map_q111,
        'mouse_id_map_wt': mouse_id_map_wt,
    }

    return fig, axes, stats


def generate_caption(stats):
    """Generate figure caption with FOV counts and mouse ID mapping."""
    dist_stats = stats.get('distribution_stats', {})
    mouse_id_map_q111 = stats.get('mouse_id_map_q111', {})
    mouse_id_map_wt = stats.get('mouse_id_map_wt', {})

    # Get FOV counts and statistics for each condition
    fov_counts = {}
    stat_values = {}
    for key, val in dist_stats.items():
        ch, region = key
        fov_counts[(ch, region, 'Q111')] = val.get('n_q111_total', len(val.get('q111_data', [])))
        fov_counts[(ch, region, 'WT')] = len(val.get('wt_data', []))
        stat_values[(ch, region)] = {
            'wt_p95': val.get('wt_p95', 0),
            'n_extreme': val.get('n_q111_extreme', 0),
            'frac_extreme': val.get('frac_extreme', 0),
            'q111_median': val.get('q111_median', 0),
            'wt_median': val.get('wt_median', 0),
            'p_mwu': val.get('p_mwu', 1.0)
        }

    caption = f"""Figure 4: FOV-level variance and extreme outlier analysis reveals heterogeneous mRNA accumulation in Q111 mice.

OVERVIEW:
This figure presents a comprehensive analysis of field-of-view (FOV) level heterogeneity in mRNA expression, with a focus on identifying "extreme" FOVs that exceed normal variation. The analysis defines extreme FOVs as those exceeding the 95th percentile of the wildtype distribution, providing a biologically-grounded threshold that accounts for natural variation in healthy tissue. The results reveal that Q111 transgenic mice have a significantly higher proportion of extreme FOVs compared to wildtype controls, indicating disease-associated mRNA accumulation hotspots.

PANEL DESCRIPTIONS:

(A) Placeholder for representative microscopy images showing:
- Example of a "normal" FOV with typical mRNA expression levels
- Example of an "extreme" FOV with elevated mRNA accumulation
- Scale bar and imaging parameters

(B) mHTT1a PROBE: DISTRIBUTION OF CLUSTERED mRNA PER NUCLEUS
Comparative histogram analysis of mRNA expression levels in Q111 vs WT mice.
- Data source: FOV-level measurements of clustered mRNA per nucleus
- X-axis: Clustered mRNA per nucleus (mRNA equivalents)
  * Main histogram range: 0 to 40 mRNA/nucleus
  * Number of bins: 10 (bin width = 4.0 mRNA/nucleus)
  * Tail bin (">40"): Aggregates all values exceeding cutoff to compress extreme outliers
- Y-axis: Probability density (normalized histogram)
- Color coding:
  * Green bars: Q111 transgenic mice
  * Blue bars: Wildtype (WT) control mice
- Pattern coding:
  * Solid bars: Cortex region
  * Hatched bars (///): Striatum region
- Grouping: 4 bars per bin (Q111 Cortex, Q111 Striatum, WT Cortex, WT Striatum)
- Black dashed vertical line: WT 95th percentile threshold (average of Cortex and Striatum)
  * Cortex P95 threshold: {stat_values.get(('mHTT1a', 'Cortex'), {}).get('wt_p95', 0):.2f} mRNA/nucleus
  * Striatum P95 threshold: {stat_values.get(('mHTT1a', 'Striatum'), {}).get('wt_p95', 0):.2f} mRNA/nucleus
- FOV counts:
  * Q111 Cortex: n = {fov_counts.get(('mHTT1a', 'Cortex', 'Q111'), 'N/A')} FOVs
  * Q111 Striatum: n = {fov_counts.get(('mHTT1a', 'Striatum', 'Q111'), 'N/A')} FOVs
  * WT Cortex: n = {fov_counts.get(('mHTT1a', 'Cortex', 'WT'), 'N/A')} FOVs
  * WT Striatum: n = {fov_counts.get(('mHTT1a', 'Striatum', 'WT'), 'N/A')} FOVs
- Key statistics:
  * Q111 Cortex median: {stat_values.get(('mHTT1a', 'Cortex'), {}).get('q111_median', 0):.2f} mRNA/nucleus
  * WT Cortex median: {stat_values.get(('mHTT1a', 'Cortex'), {}).get('wt_median', 0):.2f} mRNA/nucleus
  * Q111 Striatum median: {stat_values.get(('mHTT1a', 'Striatum'), {}).get('q111_median', 0):.2f} mRNA/nucleus
  * WT Striatum median: {stat_values.get(('mHTT1a', 'Striatum'), {}).get('wt_median', 0):.2f} mRNA/nucleus
- Statistical comparison (Mann-Whitney U test):
  * Cortex Q111 vs WT: p = {stat_values.get(('mHTT1a', 'Cortex'), {}).get('p_mwu', 1):.2e}
  * Striatum Q111 vs WT: p = {stat_values.get(('mHTT1a', 'Striatum'), {}).get('p_mwu', 1):.2e}

(C) FULL-LENGTH mHTT PROBE: DISTRIBUTION OF CLUSTERED mRNA PER NUCLEUS
Same analysis as panel B for the full-length mHTT probe.
- X-axis: Main histogram range: 0 to 50 mRNA/nucleus
  * Number of bins: 10 (bin width = 5.0 mRNA/nucleus)
  * Tail bin (">50"): Aggregates all values exceeding cutoff
- WT P95 thresholds:
  * Cortex: {stat_values.get(('full-length mHTT', 'Cortex'), {}).get('wt_p95', 0):.2f} mRNA/nucleus
  * Striatum: {stat_values.get(('full-length mHTT', 'Striatum'), {}).get('wt_p95', 0):.2f} mRNA/nucleus
- FOV counts:
  * Q111 Cortex: n = {fov_counts.get(('full-length mHTT', 'Cortex', 'Q111'), 'N/A')} FOVs
  * Q111 Striatum: n = {fov_counts.get(('full-length mHTT', 'Striatum', 'Q111'), 'N/A')} FOVs
  * WT Cortex: n = {fov_counts.get(('full-length mHTT', 'Cortex', 'WT'), 'N/A')} FOVs
  * WT Striatum: n = {fov_counts.get(('full-length mHTT', 'Striatum', 'WT'), 'N/A')} FOVs
- Key statistics:
  * Q111 Cortex median: {stat_values.get(('full-length mHTT', 'Cortex'), {}).get('q111_median', 0):.2f} mRNA/nucleus
  * WT Cortex median: {stat_values.get(('full-length mHTT', 'Cortex'), {}).get('wt_median', 0):.2f} mRNA/nucleus
  * Q111 Striatum median: {stat_values.get(('full-length mHTT', 'Striatum'), {}).get('q111_median', 0):.2f} mRNA/nucleus
  * WT Striatum median: {stat_values.get(('full-length mHTT', 'Striatum'), {}).get('wt_median', 0):.2f} mRNA/nucleus
- Statistical comparison:
  * Cortex Q111 vs WT: p = {stat_values.get(('full-length mHTT', 'Cortex'), {}).get('p_mwu', 1):.2e}
  * Striatum Q111 vs WT: p = {stat_values.get(('full-length mHTT', 'Striatum'), {}).get('p_mwu', 1):.2e}
- Y-axis shared with panel B for direct comparison

(D) EXTREME FOV PERCENTAGE BY AGE
Age-stratified analysis of extreme FOV prevalence.
- X-axis: Age groups (months) - Q111 ages on left, WT ages on right
- Y-axis: Percentage of FOVs classified as "extreme" (exceeding WT P95 threshold)
- Bar grouping: 4 bars per age group representing each channel-region combination
- Color and pattern coding:
  * Green solid: mHTT1a Cortex (Q111)
  * Green hatched: mHTT1a Striatum (Q111)
  * Orange solid: Full-length mHTT Cortex (Q111)
  * Orange hatched: Full-length mHTT Striatum (Q111)
  * Blue solid/hatched: WT mHTT1a (expected ~5%)
  * Purple solid/hatched: WT full-length (expected ~5%)
- Vertical dashed line separates Q111 (left) from WT (right) data
- WT bars show ACTUAL percentage of WT FOVs exceeding their own P95 threshold
  * Expected value: ~5% by definition (95th percentile)
  * Observed variation around 5% reflects sampling variability
- Interpretation:
  * Age-related increases in extreme FOV percentage may indicate disease progression
  * Comparison across channels reveals which probe is more sensitive to disease effects
  * Region-specific patterns (Cortex vs Striatum) indicate differential vulnerability

(E) EXTREME FOV PERCENTAGE BY ATLAS COORDINATE
Anatomical location analysis of extreme FOV prevalence.
- X-axis: Brain atlas coordinate in 25 μm units from Bregma
  * Left side: Q111 coordinates
  * Right side: WT coordinates
- Y-axis: Percentage of FOVs classified as "extreme" (0-100%)
- Same color/pattern coding as panel D
- Vertical dashed line separates Q111 (left) from WT (right) data
- Interpretation:
  * Anterior-posterior gradients in extreme FOV prevalence
  * Hotspot identification: coordinates with elevated extreme FOV percentage
  * Comparison with WT reveals disease-specific regional vulnerability

(F) EXTREME FOV PERCENTAGE BY MOUSE ID (GROUPED BY AGE)
Per-animal analysis of extreme FOV prevalence, organized by age brackets.
- X-axis: Individual mouse identifiers (anonymized as Q#1, Q#2, etc. for Q111; W#1, W#2, etc. for WT)
  * Data grouped by age brackets (2mo, 6mo, 12mo)
  * Within each age bracket: Q111 mice on left, WT mice on right
  * Solid vertical lines separate age brackets
  * Dashed vertical lines separate Q111 from WT within each age
- Y-axis: Percentage of FOVs classified as "extreme" (0-100%)
- Same color/pattern coding as panels D and E
- Age bracket labels displayed at top of each group
- Interpretation:
  * Inter-individual variability in disease severity within each age
  * Age-dependent trends in extreme FOV prevalence
  * Identification of "high-accumulator" animals at each age
  * Quality control: WT mice should show ~5% extreme FOVs across all conditions

================================================================================
MOUSE ID MAPPING TABLE
================================================================================
The following table maps anonymized mouse IDs used in panel F to actual mouse identifiers.
This mapping is consistent with Figure 3, panel F.

Q111 MICE:
{chr(10).join([f"  Q{label} → {mouse_id}" for mouse_id, label in sorted(mouse_id_map_q111.items(), key=lambda x: tuple(map(float, x[1].replace('#', '').split('.'))))]) if mouse_id_map_q111 else "  (No Q111 mice)"}

WILDTYPE (WT) MICE:
{chr(10).join([f"  W{label} → {mouse_id}" for mouse_id, label in sorted(mouse_id_map_wt.items(), key=lambda x: tuple(map(float, x[1].replace('#', '').split('.'))))]) if mouse_id_map_wt else "  (No WT mice)"}

EXTREME FOV DEFINITION:
"Extreme" FOVs are defined as those with clustered mRNA per nucleus EXCEEDING the 95th percentile of the wildtype distribution. This threshold is calculated SEPARATELY for each of 4 channel-region combinations:
1. mHTT1a + Cortex → threshold = {stat_values.get(('mHTT1a', 'Cortex'), {}).get('wt_p95', 0):.2f} mRNA/nucleus
2. mHTT1a + Striatum → threshold = {stat_values.get(('mHTT1a', 'Striatum'), {}).get('wt_p95', 0):.2f} mRNA/nucleus
3. Full-length mHTT + Cortex → threshold = {stat_values.get(('full-length mHTT', 'Cortex'), {}).get('wt_p95', 0):.2f} mRNA/nucleus
4. Full-length mHTT + Striatum → threshold = {stat_values.get(('full-length mHTT', 'Striatum'), {}).get('wt_p95', 0):.2f} mRNA/nucleus

This approach ensures that:
- The threshold is biologically meaningful (based on healthy tissue variation)
- Region-specific differences in baseline expression are accounted for
- Channel-specific detection efficiencies are normalized
- By definition, ~5% of WT FOVs will be "extreme" (sampling variability causes deviation from exactly 5%)

EXTREME FOV STATISTICS (Q111 vs expected 5%):
- mHTT1a Cortex: {stat_values.get(('mHTT1a', 'Cortex'), {}).get('n_extreme', 0)} extreme / {fov_counts.get(('mHTT1a', 'Cortex', 'Q111'), 0)} total = {stat_values.get(('mHTT1a', 'Cortex'), {}).get('frac_extreme', 0)*100:.1f}%
- mHTT1a Striatum: {stat_values.get(('mHTT1a', 'Striatum'), {}).get('n_extreme', 0)} extreme / {fov_counts.get(('mHTT1a', 'Striatum', 'Q111'), 0)} total = {stat_values.get(('mHTT1a', 'Striatum'), {}).get('frac_extreme', 0)*100:.1f}%
- Full-length Cortex: {stat_values.get(('full-length mHTT', 'Cortex'), {}).get('n_extreme', 0)} extreme / {fov_counts.get(('full-length mHTT', 'Cortex', 'Q111'), 0)} total = {stat_values.get(('full-length mHTT', 'Cortex'), {}).get('frac_extreme', 0)*100:.1f}%
- Full-length Striatum: {stat_values.get(('full-length mHTT', 'Striatum'), {}).get('n_extreme', 0)} extreme / {fov_counts.get(('full-length mHTT', 'Striatum', 'Q111'), 0)} total = {stat_values.get(('full-length mHTT', 'Striatum'), {}).get('frac_extreme', 0)*100:.1f}%

================================================================================
FILTERING APPLIED (consistent with Figure 1, panels E onwards)
================================================================================

FOV-LEVEL ANALYSIS:
This figure analyzes mRNA expression at the FOV (field-of-view) level, comparing Q111 vs WT distributions and identifying "extreme" FOVs.

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

FOV-LEVEL AGGREGATION:
- Clustered mRNA per nucleus: (Sum of cluster intensities) / (Number of DAPI-positive nuclei)
- Minimum nuclei threshold: FOVs with < 40 nuclei are excluded

TECHNICAL NOTES:
- Bead PSF: σ_x = {BEAD_PSF_X:.1f} nm, σ_y = {BEAD_PSF_Y:.1f} nm, σ_z = {BEAD_PSF_Z:.1f} nm
- Size lower bound: σ ≥ 80% of bead PSF ({SIGMA_X_LOWER:.1f} nm for σ_x)
- Excluded slides: {EXCLUDED_SLIDES} (technical failures - imaging artifacts or tissue damage)
- Statistical test: Mann-Whitney U test (non-parametric, two-sided)
- Histogram tail compression: Values > cutoff aggregated into single ">X" bin to improve visualization

COLOR SCHEME SUMMARY:
| Condition | mHTT1a | Full-length mHTT |
|-----------|--------|------------------|
| Q111 | Green (#2ecc71) | Orange (#f39c12) |
| WT | Blue (#3498db) | Purple (#9b59b6) |

| Region | Pattern |
|--------|---------|
| Cortex | Solid fill |
| Striatum | Hatched (///) |

KEY FINDINGS:
1. Q111 MICE HAVE ELEVATED EXTREME FOV PERCENTAGE: Across all channel-region combinations, Q111 mice show significantly higher proportion of extreme FOVs compared to the expected 5% from WT controls
2. REGIONAL DIFFERENCES: Striatum shows higher extreme FOV percentages than Cortex, consistent with striatal vulnerability in Huntington's disease
3. AGE-DEPENDENT PROGRESSION: Extreme FOV percentage tends to increase with age, suggesting progressive mRNA accumulation
4. INTER-INDIVIDUAL VARIABILITY: Some Q111 mice show substantially higher extreme FOV percentages, indicating biological heterogeneity in disease severity
5. ANATOMICAL HOTSPOTS: Certain atlas coordinates show elevated extreme FOV prevalence, potentially indicating regionally-selective vulnerability
6. PROBE CONSISTENCY: Both mHTT1a and full-length probes show similar patterns, validating the biological relevance of the findings

DATA CACHING:
Processed data is cached to {CACHE_FILE.name} for fast subsequent runs. Set FORCE_RELOAD = True to regenerate from raw data.
"""
    return caption


def main():
    """Generate and save Figure 4."""

    fig, axes, stats = create_figure4()

    print("\n" + "=" * 70)
    print("SAVING FIGURE")
    print("=" * 70)

    save_figure(fig, 'figure4', formats=['svg', 'png', 'pdf'], output_dir=OUTPUT_DIR)

    # Generate and save caption
    caption = generate_caption(stats)
    caption_file = OUTPUT_DIR / 'figure4_caption.txt'
    with open(caption_file, 'w') as f:
        f.write(caption)
    print(f"Caption saved: {caption_file}")

    plt.close(fig)

    print("\n" + "=" * 70)
    print("FIGURE 4 COMPLETE")
    print("=" * 70)
    print(f"\nTo make layout changes quickly, just re-run this script.")
    print(f"Data is cached at: {CACHE_FILE}")


if __name__ == '__main__':
    main()
