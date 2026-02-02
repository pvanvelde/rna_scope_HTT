"""
FOV Extremes Story Figure
==========================

Creates a comprehensive figure telling the story of FOV-level variance and extreme outliers:
1. Overall distributions showing Q111 vs WT variance
2. Tail analysis - FOVs exceeding WT P95 threshold
3. Source breakdown - where do extreme FOVs come from?
4. Spatial and temporal patterns

Story: "Big variance → Q111 tail → Sources of extremes"
"""

import sys
import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import mannwhitneyu, ks_2samp
import scienceplots
plt.style.use('science')
plt.rcParams['text.usetex'] = False
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import centralized configuration
from results_config import (
    FIGURE_DPI,
    OUTPUT_DIR_COMPREHENSIVE,
    EXCLUDED_SLIDES
)

# Output directory
OUTPUT_DIR = Path(__file__).parent / "output" / "fov_extremes_story"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("FOV EXTREMES STORY FIGURE")
print("=" * 70)

# Load FOV-level data
print("\nLoading FOV-level data...")
df_fov = pd.read_csv(OUTPUT_DIR_COMPREHENSIVE / 'fov_level_data.csv')

# Filter for experimental data and exclude bad slides
df_exp = df_fov[df_fov['Mouse_Model'].isin(['Q111', 'Wildtype'])].copy()
df_exp = df_exp[~df_exp['Slide'].isin(EXCLUDED_SLIDES)]

print(f"Total FOVs after exclusions: {len(df_exp)}")
print(f"Q111 FOVs: {len(df_exp[df_exp['Mouse_Model'] == 'Q111'])}")
print(f"Wildtype FOVs: {len(df_exp[df_exp['Mouse_Model'] == 'Wildtype'])}")

# Color scheme
COLOR_Q111 = '#2E8B57'  # Sea green
COLOR_Q111_EXTREME = '#8B0000'  # Dark red
COLOR_WT = '#FF7F50'  # Coral

# ============================================================================
# COMPUTE THRESHOLDS AND STATISTICS
# ============================================================================

print("\nComputing thresholds and statistics...")

thresholds = {}
distribution_stats = {}
extreme_fovs = {}

for ch in ['HTT1a', 'fl-HTT']:
    for region in ['Cortex', 'Striatum']:
        # Get data
        q111_data = df_exp[(df_exp['Mouse_Model'] == 'Q111') &
                           (df_exp['Channel'] == ch) &
                           (df_exp['Region'] == region)]['Clustered_mRNA_per_Cell'].dropna()

        wt_data = df_exp[(df_exp['Mouse_Model'] == 'Wildtype') &
                         (df_exp['Channel'] == ch) &
                         (df_exp['Region'] == region)]['Clustered_mRNA_per_Cell'].dropna()

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
        ks_stat, p_ks = ks_2samp(q111_data, wt_data)

        distribution_stats[(ch, region)] = {
            'q111_data': q111_data,
            'wt_data': wt_data,
            'wt_p95': wt_p95,
            'n_q111_extreme': n_q111_extreme,
            'n_q111_total': n_q111_total,
            'frac_extreme': frac_extreme,
            'q111_median': np.median(q111_data),
            'wt_median': np.median(wt_data),
            'q111_mean': np.mean(q111_data),
            'wt_mean': np.mean(wt_data),
            'q111_iqr': np.percentile(q111_data, 75) - np.percentile(q111_data, 25),
            'wt_iqr': np.percentile(wt_data, 75) - np.percentile(wt_data, 25),
            'q111_p95': np.percentile(q111_data, 95),
            'p_mwu': p_mwu,
            'p_ks': p_ks
        }

        # Get extreme FOV details
        extreme_df = df_exp[(df_exp['Mouse_Model'] == 'Q111') &
                            (df_exp['Channel'] == ch) &
                            (df_exp['Region'] == region) &
                            (df_exp['Clustered_mRNA_per_Cell'] > wt_p95)].copy()
        extreme_fovs[(ch, region)] = extreme_df

        print(f"  {ch} - {region}: {n_q111_extreme}/{n_q111_total} extreme ({100*frac_extreme:.1f}%)")

# ============================================================================
# CREATE COMPREHENSIVE FIGURE FOR EACH CHANNEL
# ============================================================================

for channel in ['HTT1a', 'fl-HTT']:
    print(f"\nCreating figure for {channel}...")

    # Figure layout: 4 rows x 4 columns
    # Row 1: Overall distributions (Cortex, Striatum) + Legend/Summary
    # Row 2: By Age breakdown (Cortex, Striatum)
    # Row 3: By Mouse breakdown (top mice) (Cortex, Striatum)
    # Row 4: By Atlas coordinate (Cortex, Striatum)

    fig = plt.figure(figsize=(20, 20), dpi=FIGURE_DPI)
    gs = fig.add_gridspec(4, 4, hspace=0.35, wspace=0.30)

    channel_label = 'HTT1a' if channel == 'HTT1a' else 'fl-HTT'
    fig.suptitle(f'FOV-Level Variance and Extreme Outliers - {channel_label}\nClustered mRNA per Cell Analysis',
                 fontsize=18, fontweight='bold', y=0.98)

    # Panel label counter
    panel_idx = 0
    panel_labels = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

    # ========================================================================
    # ROW 1: OVERALL DISTRIBUTIONS WITH TAIL HIGHLIGHTED
    # ========================================================================

    for reg_idx, region in enumerate(['Cortex', 'Striatum']):
        ax = fig.add_subplot(gs[0, reg_idx * 2:(reg_idx + 1) * 2])

        # Add panel label
        ax.text(-0.08, 1.05, panel_labels[panel_idx], transform=ax.transAxes,
               fontsize=16, fontweight='bold', va='bottom', ha='left')
        panel_idx += 1

        stats = distribution_stats.get((channel, region))
        if stats is None:
            continue

        q111_data = stats['q111_data']
        wt_data = stats['wt_data']
        wt_p95 = stats['wt_p95']

        # Create histogram bins
        max_val = np.percentile(q111_data, 99.5)
        bins = np.linspace(0, max_val, 50)

        # Plot WT distribution
        ax.hist(wt_data, bins=bins, alpha=0.8, color=COLOR_WT,
                label=f'Wildtype (n={len(wt_data)})',
                edgecolor='black', linewidth=1, density=True, zorder=5)

        # Plot Q111: separate into below and above WT P95
        q111_below = q111_data[q111_data <= wt_p95]
        q111_above = q111_data[q111_data > wt_p95]

        ax.hist(q111_below, bins=bins, alpha=0.6, color=COLOR_Q111,
                label=f'Q111 ≤WT P95 (n={len(q111_below)})',
                edgecolor='darkgreen', linewidth=0.5, density=True, zorder=3)

        ax.hist(q111_above, bins=bins, alpha=0.9, color=COLOR_Q111_EXTREME,
                label=f'Q111 >WT P95: {stats["n_q111_extreme"]} FOVs ({100*stats["frac_extreme"]:.1f}%)',
                edgecolor='black', linewidth=1.5, density=True, hatch='///', zorder=4)

        # Mark WT P95 threshold
        ax.axvline(wt_p95, color='black', linestyle='--', linewidth=2.5,
                   label=f'WT P95 = {wt_p95:.1f}')

        ax.set_xlabel('Clustered mRNA per Cell', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(f'{region} - Overall Distribution\n({stats["n_q111_extreme"]}/{stats["n_q111_total"]} Q111 FOVs exceed WT P95)',
                    fontsize=12, fontweight='bold')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')

        # Statistics text box
        textstr = f'Q111 median: {stats["q111_median"]:.1f}\n'
        textstr += f'WT median: {stats["wt_median"]:.1f}\n'
        textstr += f'Q111 IQR: {stats["q111_iqr"]:.1f}\n'
        textstr += f'WT IQR: {stats["wt_iqr"]:.1f}\n'
        textstr += f'MWU p: {stats["p_mwu"]:.1e}'
        ax.text(0.98, 0.65, textstr, transform=ax.transAxes,
               fontsize=8, verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # ========================================================================
    # ROW 2: BREAKDOWN BY AGE
    # ========================================================================

    for reg_idx, region in enumerate(['Cortex', 'Striatum']):
        ax = fig.add_subplot(gs[1, reg_idx * 2:(reg_idx + 1) * 2])

        # Add panel label
        ax.text(-0.08, 1.05, panel_labels[panel_idx], transform=ax.transAxes,
               fontsize=16, fontweight='bold', va='bottom', ha='left')
        panel_idx += 1

        wt_p95 = thresholds.get((channel, region))
        if wt_p95 is None:
            continue

        # Get all Q111 data for this channel/region
        df_q111 = df_exp[(df_exp['Mouse_Model'] == 'Q111') &
                         (df_exp['Channel'] == channel) &
                         (df_exp['Region'] == region)]

        # Get unique ages
        ages = sorted(df_q111['Age'].unique())

        x_pos = np.arange(len(ages))
        width = 0.35

        # Calculate counts for each age
        n_extreme_list = []
        n_total_list = []
        frac_list = []

        for age in ages:
            df_age = df_q111[df_q111['Age'] == age]
            n_total = len(df_age)
            n_extreme = np.sum(df_age['Clustered_mRNA_per_Cell'] > wt_p95)
            n_extreme_list.append(n_extreme)
            n_total_list.append(n_total)
            frac_list.append(n_extreme / n_total if n_total > 0 else 0)

        # Stacked bar: normal + extreme
        n_normal_list = [t - e for t, e in zip(n_total_list, n_extreme_list)]

        bars_normal = ax.bar(x_pos, n_normal_list, width=0.6, color=COLOR_Q111,
                            alpha=0.6, label='Normal FOVs', edgecolor='darkgreen')
        bars_extreme = ax.bar(x_pos, n_extreme_list, width=0.6, bottom=n_normal_list,
                             color=COLOR_Q111_EXTREME, alpha=0.8, label='Extreme FOVs',
                             edgecolor='black', hatch='///')

        # Add percentage labels
        for i, (ext, tot, frac) in enumerate(zip(n_extreme_list, n_total_list, frac_list)):
            ax.text(i, tot + max(n_total_list) * 0.02, f'{100*frac:.0f}%',
                   ha='center', va='bottom', fontsize=9, fontweight='bold', color=COLOR_Q111_EXTREME)
            # Show counts inside bars
            if ext > 0:
                ax.text(i, n_normal_list[i] + ext/2, f'{ext}',
                       ha='center', va='center', fontsize=8, color='white', fontweight='bold')

        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'{int(a)}mo' for a in ages], fontsize=10)
        ax.set_xlabel('Age (months)', fontsize=11)
        ax.set_ylabel('Number of FOVs', fontsize=11)
        ax.set_title(f'{region} - Extreme FOVs by Age\n(% shows fraction of extreme at each age)',
                    fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')

    # ========================================================================
    # ROW 3: BREAKDOWN BY MOUSE (TOP 8 MICE BY % EXTREME)
    # ========================================================================

    for reg_idx, region in enumerate(['Cortex', 'Striatum']):
        ax = fig.add_subplot(gs[2, reg_idx * 2:(reg_idx + 1) * 2])

        # Add panel label
        ax.text(-0.08, 1.05, panel_labels[panel_idx], transform=ax.transAxes,
               fontsize=16, fontweight='bold', va='bottom', ha='left')
        panel_idx += 1

        wt_p95 = thresholds.get((channel, region))
        if wt_p95 is None:
            continue

        # Get all Q111 data for this channel/region
        df_q111 = df_exp[(df_exp['Mouse_Model'] == 'Q111') &
                         (df_exp['Channel'] == channel) &
                         (df_exp['Region'] == region)]

        # Calculate stats per mouse
        mouse_stats = []
        for mouse in df_q111['Mouse_ID'].unique():
            df_mouse = df_q111[df_q111['Mouse_ID'] == mouse]
            n_total = len(df_mouse)
            n_extreme = np.sum(df_mouse['Clustered_mRNA_per_Cell'] > wt_p95)
            frac = n_extreme / n_total if n_total > 0 else 0
            slide = df_mouse['Slide'].iloc[0]
            age = df_mouse['Age'].iloc[0]

            mouse_stats.append({
                'mouse': mouse,
                'slide': slide,
                'age': int(age),
                'n_extreme': n_extreme,
                'n_total': n_total,
                'frac': frac
            })

        df_mouse_stats = pd.DataFrame(mouse_stats)

        # Sort by fraction and get top 8
        df_mouse_stats = df_mouse_stats.sort_values('frac', ascending=True).tail(8)

        y_pos = np.arange(len(df_mouse_stats))

        # Create labels
        labels = [f"{row['mouse']} ({row['slide']}, {row['age']}mo)"
                  for _, row in df_mouse_stats.iterrows()]

        # Bar plot
        bars = ax.barh(y_pos, df_mouse_stats['frac'] * 100, color=COLOR_Q111_EXTREME,
                      alpha=0.7, edgecolor='black', linewidth=1)

        # Add percentage and count labels
        for bar, frac, n_ext, n_tot in zip(bars, df_mouse_stats['frac'] * 100,
                                            df_mouse_stats['n_extreme'], df_mouse_stats['n_total']):
            width = bar.get_width()
            ax.text(width + 1, bar.get_y() + bar.get_height()/2,
                   f'{frac:.0f}% ({n_ext}/{n_tot})',
                   ha='left', va='center', fontsize=8, fontweight='bold')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel('Percentage of Extreme FOVs (%)', fontsize=10)
        ax.set_title(f'{region} - Top Mice by % Extreme FOVs\n(Mouse ID, Slide, Age)',
                    fontsize=11, fontweight='bold')
        ax.set_xlim([0, 110])
        ax.grid(True, alpha=0.3, axis='x')

    # ========================================================================
    # ROW 4: BREAKDOWN BY ATLAS COORDINATE
    # ========================================================================

    for reg_idx, region in enumerate(['Cortex', 'Striatum']):
        ax = fig.add_subplot(gs[3, reg_idx * 2:(reg_idx + 1) * 2])

        # Add panel label
        ax.text(-0.08, 1.05, panel_labels[panel_idx], transform=ax.transAxes,
               fontsize=16, fontweight='bold', va='bottom', ha='left')
        panel_idx += 1

        wt_p95 = thresholds.get((channel, region))
        if wt_p95 is None:
            continue

        # Get all Q111 data for this channel/region
        df_q111 = df_exp[(df_exp['Mouse_Model'] == 'Q111') &
                         (df_exp['Channel'] == channel) &
                         (df_exp['Region'] == region)]

        # Get unique atlas coordinates sorted
        atlas_coords = sorted(df_q111['Brain_Atlas_Coord'].unique())

        x_pos = np.arange(len(atlas_coords))

        # Calculate counts for each atlas coordinate
        n_extreme_list = []
        n_total_list = []
        frac_list = []

        for atlas in atlas_coords:
            df_atlas = df_q111[df_q111['Brain_Atlas_Coord'] == atlas]
            n_total = len(df_atlas)
            n_extreme = np.sum(df_atlas['Clustered_mRNA_per_Cell'] > wt_p95)
            n_extreme_list.append(n_extreme)
            n_total_list.append(n_total)
            frac_list.append(n_extreme / n_total if n_total > 0 else 0)

        # Bar plot showing extreme counts with color based on fraction
        colors = plt.cm.RdYlGn_r([f for f in frac_list])  # Red=high, Green=low
        bars = ax.bar(x_pos, n_extreme_list, color=colors, alpha=0.8,
                     edgecolor='black', linewidth=0.5)

        # Add fraction labels on top (including 0%)
        max_ext = max(n_extreme_list) if max(n_extreme_list) > 0 else 1
        for i, (ext, frac) in enumerate(zip(n_extreme_list, frac_list)):
            ax.text(i, ext + max_ext * 0.02, f'{100*frac:.0f}%',
                   ha='center', va='bottom', fontsize=7, fontweight='bold')

        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'{int(a)}' for a in atlas_coords], fontsize=8, rotation=45)
        ax.set_xlabel('Brain Atlas Coordinate (A-P from Bregma, 25μm units)', fontsize=10)
        ax.set_ylabel('Number of Extreme FOVs', fontsize=10)
        ax.set_title(f'{region} - Extreme FOVs by Atlas Coordinate\n(% shows fraction extreme at each coordinate)',
                    fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # Add colorbar-like legend
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
        sm = ScalarMappable(cmap='RdYlGn_r', norm=Normalize(vmin=0, vmax=max(frac_list)*100))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.02, pad=0.02)
        cbar.set_label('% Extreme', fontsize=8)

    # Save figure
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    channel_name = channel.replace(' ', '_').replace('-', '_')
    filepath_svg = OUTPUT_DIR / f'fov_extremes_story_{channel_name}.svg'
    filepath_pdf = OUTPUT_DIR / f'fov_extremes_story_{channel_name}.pdf'

    plt.savefig(filepath_svg, format='svg', bbox_inches='tight')
    plt.savefig(filepath_pdf, format='pdf', bbox_inches='tight')
    plt.close()

    print(f"  Saved: {filepath_svg}")
    print(f"  Saved: {filepath_pdf}")

# ============================================================================
# GENERATE ELABORATE CAPTION
# ============================================================================

print("\nGenerating elaborate captions...")

for channel in ['HTT1a', 'fl-HTT']:
    channel_label = 'HTT1a' if channel == 'HTT1a' else 'fl-HTT'
    channel_name = channel.replace(' ', '_').replace('-', '_')

    # Gather statistics for caption
    stats_cortex = distribution_stats.get((channel, 'Cortex'), {})
    stats_striatum = distribution_stats.get((channel, 'Striatum'), {})

    caption_lines = []
    caption_lines.append("=" * 80)
    caption_lines.append(f"FIGURE: FOV-Level Variance and Extreme Outliers - {channel_label}")
    caption_lines.append("=" * 80)
    caption_lines.append("")

    # Overview
    caption_lines.append("OVERVIEW:")
    caption_lines.append("-" * 80)
    caption_lines.append("This figure presents a comprehensive analysis of field-of-view (FOV) level variance")
    caption_lines.append(f"in clustered mRNA per cell for {channel_label} transcripts. The analysis reveals")
    caption_lines.append("substantial inter-FOV variability in Q111 transgenic mice and identifies 'extreme'")
    caption_lines.append("FOVs - those exceeding the 95th percentile of wildtype distributions.")
    caption_lines.append("")
    caption_lines.append("The figure tells a story in four parts:")
    caption_lines.append("  1. Overall distributions showing Q111 vs WT variance")
    caption_lines.append("  2. Temporal patterns - age-dependent accumulation of extreme FOVs")
    caption_lines.append("  3. Individual variation - which mice drive the extreme phenotype")
    caption_lines.append("  4. Spatial patterns - regional distribution along the A-P axis")
    caption_lines.append("")

    # Dataset statistics
    caption_lines.append("DATASET STATISTICS:")
    caption_lines.append("-" * 80)

    # Count FOVs
    n_q111_cortex = len(df_exp[(df_exp['Mouse_Model'] == 'Q111') & (df_exp['Channel'] == channel) & (df_exp['Region'] == 'Cortex')])
    n_wt_cortex = len(df_exp[(df_exp['Mouse_Model'] == 'Wildtype') & (df_exp['Channel'] == channel) & (df_exp['Region'] == 'Cortex')])
    n_q111_striatum = len(df_exp[(df_exp['Mouse_Model'] == 'Q111') & (df_exp['Channel'] == channel) & (df_exp['Region'] == 'Striatum')])
    n_wt_striatum = len(df_exp[(df_exp['Mouse_Model'] == 'Wildtype') & (df_exp['Channel'] == channel) & (df_exp['Region'] == 'Striatum')])

    caption_lines.append(f"Cortex: {n_q111_cortex} Q111 FOVs, {n_wt_cortex} WT FOVs")
    caption_lines.append(f"Striatum: {n_q111_striatum} Q111 FOVs, {n_wt_striatum} WT FOVs")
    caption_lines.append("")
    caption_lines.append("Excluded slides (technical failures): m1a2, m1b5")
    caption_lines.append("")

    # Panel descriptions
    caption_lines.append("PANEL DESCRIPTIONS:")
    caption_lines.append("-" * 80)
    caption_lines.append("")

    # Panels A-B
    caption_lines.append("(A-B) OVERALL DISTRIBUTIONS:")
    caption_lines.append("Probability density histograms showing the full distribution of clustered mRNA per")
    caption_lines.append("cell. Q111 data is split into 'normal' (≤WT P95) and 'extreme' (>WT P95) populations.")
    caption_lines.append("The hatched red region represents the Q111 'tail' - FOVs with abnormally high")
    caption_lines.append("clustered mRNA that exceed 95% of wildtype values. (A) Cortex, (B) Striatum.")
    caption_lines.append("")

    if stats_cortex:
        caption_lines.append(f"Cortex: WT P95 threshold = {stats_cortex['wt_p95']:.1f} mRNA/cell")
        caption_lines.append(f"  Q111: {stats_cortex['n_q111_extreme']}/{stats_cortex['n_q111_total']} FOVs extreme ({100*stats_cortex['frac_extreme']:.1f}%)")
        caption_lines.append(f"  Q111 median={stats_cortex['q111_median']:.1f}, IQR={stats_cortex['q111_iqr']:.1f}")
        caption_lines.append(f"  WT median={stats_cortex['wt_median']:.1f}, IQR={stats_cortex['wt_iqr']:.1f}")
        caption_lines.append(f"  Mann-Whitney U p={stats_cortex['p_mwu']:.2e}")

    if stats_striatum:
        caption_lines.append(f"Striatum: WT P95 threshold = {stats_striatum['wt_p95']:.1f} mRNA/cell")
        caption_lines.append(f"  Q111: {stats_striatum['n_q111_extreme']}/{stats_striatum['n_q111_total']} FOVs extreme ({100*stats_striatum['frac_extreme']:.1f}%)")
        caption_lines.append(f"  Q111 median={stats_striatum['q111_median']:.1f}, IQR={stats_striatum['q111_iqr']:.1f}")
        caption_lines.append(f"  WT median={stats_striatum['wt_median']:.1f}, IQR={stats_striatum['wt_iqr']:.1f}")
        caption_lines.append(f"  Mann-Whitney U p={stats_striatum['p_mwu']:.2e}")
    caption_lines.append("")

    # Panels C-D
    caption_lines.append("(C-D) BREAKDOWN BY AGE:")
    caption_lines.append("Stacked bar charts showing the number of normal vs extreme FOVs at each age (2, 6,")
    caption_lines.append("12 months). Percentages indicate the fraction of FOVs that are extreme at each age.")
    caption_lines.append("This reveals temporal patterns in the extreme phenotype. (C) Cortex, (D) Striatum.")
    caption_lines.append("")

    # Panels E-F
    caption_lines.append("(E-F) BREAKDOWN BY MOUSE:")
    caption_lines.append("Horizontal bar charts showing the top 8 mice ranked by percentage of extreme FOVs.")
    caption_lines.append("Each bar shows the mouse ID, slide identifier, and age in months. This reveals")
    caption_lines.append("inter-individual variability and identifies specific mice that drive the extreme")
    caption_lines.append("phenotype. (E) Cortex, (F) Striatum.")
    caption_lines.append("")

    # Panels G-H
    caption_lines.append("(G-H) BREAKDOWN BY ATLAS COORDINATE:")
    caption_lines.append("Bar charts showing the number of extreme FOVs at each brain atlas coordinate (A-P")
    caption_lines.append("position from Bregma in 25μm units). Color intensity reflects the fraction of FOVs")
    caption_lines.append("that are extreme at each coordinate (red=high, green=low). This reveals spatial")
    caption_lines.append("patterns along the anterior-posterior axis. (G) Cortex, (H) Striatum.")
    caption_lines.append("")

    # Color scheme
    caption_lines.append("COLOR SCHEME:")
    caption_lines.append("-" * 80)
    caption_lines.append("Sea Green (#2E8B57): Q111 normal FOVs (≤WT P95)")
    caption_lines.append("Dark Red (#8B0000): Q111 extreme FOVs (>WT P95)")
    caption_lines.append("Coral (#FF7F50): Wildtype")
    caption_lines.append("RdYlGn colormap: Fraction extreme (red=high, green=low)")
    caption_lines.append("")

    # Key observations
    caption_lines.append("KEY OBSERVATIONS:")
    caption_lines.append("-" * 80)

    if channel == 'HTT1a':
        caption_lines.append("1. Distribution variance:")
        caption_lines.append("   - Q111 shows substantially greater variance than wildtype")
        caption_lines.append("   - Clear rightward shift with extended tail beyond WT P95")
        caption_lines.append("   - HTT1a represents aberrant intron-1 terminated transcripts")
        caption_lines.append("")
        caption_lines.append("2. Extreme FOV prevalence:")
        caption_lines.append("   - Significant fraction of Q111 FOVs exceed wildtype P95")
        caption_lines.append("   - Indicates mosaic/heterogeneous phenotype at tissue level")
        caption_lines.append("   - Not all cells/regions equally affected")
        caption_lines.append("")
        caption_lines.append("3. Biological significance:")
        caption_lines.append("   - HTT1a encodes toxic N-terminal fragment with polyQ expansion")
        caption_lines.append("   - Extreme FOVs may represent hotspots of pathology")
        caption_lines.append("   - Clustering suggests sites of active transcription or aggregation")
    else:
        caption_lines.append("1. Distribution variance:")
        caption_lines.append("   - Q111 shows substantially greater variance than wildtype")
        caption_lines.append("   - Clear rightward shift with extended tail beyond WT P95")
        caption_lines.append("   - fl-HTT represents completely spliced transcripts")
        caption_lines.append("")
        caption_lines.append("2. Extreme FOV prevalence:")
        caption_lines.append("   - Significant fraction of Q111 FOVs exceed wildtype P95")
        caption_lines.append("   - Higher abundance than HTT1a clusters")
        caption_lines.append("   - Indicates widespread but heterogeneous elevation")
        caption_lines.append("")
        caption_lines.append("3. Biological significance:")
        caption_lines.append("   - fl-HTT encodes complete huntingtin protein (~350 kDa)")
        caption_lines.append("   - Extreme FOVs may indicate regions of high transcriptional activity")
        caption_lines.append("   - Clusters represent co-transcriptional sites or transport granules")
    caption_lines.append("")

    # Methodology
    caption_lines.append("METHODOLOGY:")
    caption_lines.append("-" * 80)
    caption_lines.append("Clustered mRNA per Cell calculation:")
    caption_lines.append("  - Cluster detection: DBSCAN algorithm (eps=0.75 µm, min_samples=3)")
    caption_lines.append("  - Intensity threshold: 2.5 × slide-specific peak intensity")
    caption_lines.append("  - Peak intensity: mode of KDE-fitted single spot intensity distribution")
    caption_lines.append("  - Clustered mRNA = Total cluster intensity / Peak intensity")
    caption_lines.append("  - Per-cell normalization: Clustered mRNA / N_nuclei in FOV")
    caption_lines.append("")
    caption_lines.append("Extreme FOV definition:")
    caption_lines.append("  - Threshold = 95th percentile of Wildtype distribution (region-specific)")
    caption_lines.append("  - FOVs with Clustered mRNA/Cell > WT P95 classified as 'extreme'")
    caption_lines.append("  - This identifies Q111 FOVs in the tail of the distribution")
    caption_lines.append("")
    caption_lines.append("Statistical tests:")
    caption_lines.append("  - Mann-Whitney U test: non-parametric comparison of medians")
    caption_lines.append("  - Tests whether Q111 and WT come from same distribution")
    caption_lines.append("")
    caption_lines.append("Quality control:")
    caption_lines.append("  - Slides m1a2 and m1b5 excluded (technical failures)")
    caption_lines.append("  - Based on UBC positive control analysis")
    caption_lines.append("")

    # Interpretation guide
    caption_lines.append("INTERPRETATION GUIDE:")
    caption_lines.append("-" * 80)
    caption_lines.append("The 'extreme FOV' analysis provides insight into the mosaic nature of the Q111")
    caption_lines.append("phenotype. Rather than a uniform elevation across all FOVs, we see a subset of")
    caption_lines.append("FOVs with dramatically elevated clustered mRNA. The breakdown by age, mouse, and")
    caption_lines.append("atlas coordinate helps identify:")
    caption_lines.append("")
    caption_lines.append("  - Temporal dynamics: Does the extreme phenotype progress with age?")
    caption_lines.append("  - Individual variation: Are certain mice predisposed to extreme FOVs?")
    caption_lines.append("  - Spatial patterns: Are specific anatomical regions more affected?")
    caption_lines.append("")
    caption_lines.append("This information is crucial for understanding whether the clustered mRNA phenotype")
    caption_lines.append("is driven by systemic factors (affecting all mice equally) or by stochastic/local")
    caption_lines.append("factors (creating mosaic patterns).")
    caption_lines.append("")

    caption_lines.append("=" * 80)
    caption_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    caption_lines.append("=" * 80)

    # Save caption
    caption_filepath = OUTPUT_DIR / f'figure_caption_{channel_name}.txt'
    with open(caption_filepath, 'w') as f:
        f.write('\n'.join(caption_lines))

    print(f"  Saved caption: {caption_filepath}")

# ============================================================================
# SAVE SUMMARY STATISTICS
# ============================================================================

print("\nSaving summary statistics...")

summary_data = []
for (ch, region), stats in distribution_stats.items():
    summary_data.append({
        'Channel': ch,
        'Region': region,
        'WT_P95_Threshold': stats['wt_p95'],
        'N_Q111_Extreme': stats['n_q111_extreme'],
        'N_Q111_Total': stats['n_q111_total'],
        'Fraction_Extreme': stats['frac_extreme'],
        'Q111_Median': stats['q111_median'],
        'WT_Median': stats['wt_median'],
        'Q111_Mean': stats['q111_mean'],
        'WT_Mean': stats['wt_mean'],
        'Q111_IQR': stats['q111_iqr'],
        'WT_IQR': stats['wt_iqr'],
        'Q111_P95': stats['q111_p95'],
        'MWU_pvalue': stats['p_mwu'],
        'KS_pvalue': stats['p_ks']
    })

df_summary = pd.DataFrame(summary_data)
df_summary.to_csv(OUTPUT_DIR / 'extreme_fov_summary_statistics.csv', index=False)
print(f"  Saved: {OUTPUT_DIR / 'extreme_fov_summary_statistics.csv'}")

print("\n" + "=" * 70)
print("FIGURE GENERATION COMPLETE")
print("=" * 70)
print(f"\nAll outputs saved to: {OUTPUT_DIR}")
