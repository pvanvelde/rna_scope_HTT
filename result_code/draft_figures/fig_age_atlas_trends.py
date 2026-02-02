"""
Age and Atlas Coordinate Trend Analysis for Q111 fl-HTT Expression
Shows how fl-HTT expression changes with age and anteroposterior brain position

Author: Generated with Claude Code
Date: 2025-11-16
"""

import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, linregress
from pathlib import Path
import seaborn as sns
import sys
import subprocess

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from results_config import (
    FIGURE_DPI,
    FIGURE_FORMAT
)

# Output directory
OUTPUT_DIR = Path(__file__).parent / "output" / "age_atlas_trends"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Figure settings
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9


def create_age_trend_figure(df_fov):
    """
    Create comprehensive figure showing age-dependent trends
    for all expression metrics, including both Q111 and Wildtype.
    """

    # Color scheme - different for Q111 vs Wildtype
    # Q111 colors (saturated)
    color_q111_mhtt1a = '#2ecc71'  # Green
    color_q111_full = '#f39c12'  # Orange

    # Wildtype colors (desaturated/lighter)
    color_wt_mhtt1a = '#a8e6cf'  # Light green
    color_wt_full = '#ffd8a8'  # Light orange

    # Create figure with 3x2 layout
    fig = plt.figure(figsize=(18, 16), dpi=FIGURE_DPI)
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.30,
                          left=0.08, right=0.95, top=0.96, bottom=0.06)

    metrics = ['Total_mRNA_per_Cell', 'Single_mRNA_per_Cell', 'Clustered_mRNA_per_Cell']
    metric_names = ['Total mRNA', 'Single mRNA', 'Clustered mRNA']

    for row_idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):

        # ══════════════════════════════════════════════════════════════════
        # LEFT COLUMN: Age scatter plots (like atlas figure)
        # ══════════════════════════════════════════════════════════════════

        ax_left = fig.add_subplot(gs[row_idx, 0])

        for mouse_model in ['Q111', 'Wildtype']:
            for region in ['Cortex', 'Striatum']:
                for channel in ['HTT1a', 'fl-HTT']:
                    subset = df_fov[(df_fov['Mouse_Model'] == mouse_model) &
                                    (df_fov['Region'] == region) &
                                    (df_fov['Channel'] == channel)]

                    if len(subset) < 10:
                        continue

                    # Determine plotting style and X-offset
                    if mouse_model == 'Q111':
                        if channel == 'HTT1a':
                            color = color_q111_mhtt1a
                        else:
                            color = color_q111_full
                        marker = 'o'
                        markersize = 35
                        alpha = 0.6 if region == 'Cortex' else 0.5
                        x_offset = -0.1  # Shift Q111 slightly left
                    else:  # Wildtype
                        if channel == 'HTT1a':
                            color = color_wt_mhtt1a
                        else:
                            color = color_wt_full
                        marker = 's'  # Square for wildtype
                        markersize = 30
                        alpha = 0.7 if region == 'Cortex' else 0.6  # More visible
                        x_offset = 0.3  # Shift wildtype significantly right

                    # Add jitter to scatter plot
                    ages = subset['Age'].values
                    values = subset[metric].values
                    np.random.seed(42)  # For reproducibility
                    age_jitter = ages + x_offset + np.random.normal(0, 0.05, len(ages))

                    # Scatter plot with transparency and jitter
                    ax_left.scatter(age_jitter, values,
                                   color=color, marker=marker, s=markersize, alpha=alpha,
                                   label=f"{channel} ({region}, {mouse_model})")

                    # Add trend line if significant correlation
                    valid = ~(np.isnan(ages) | np.isnan(values))

                    if valid.sum() > 10:
                        r, p = pearsonr(ages[valid], values[valid])
                        if p < 0.05:
                            slope, intercept, _, _, _ = linregress(ages[valid], values[valid])
                            x_fit = np.array([ages[valid].min(), ages[valid].max()]) + x_offset
                            y_fit = slope * (x_fit - x_offset) + intercept

                            linewidth = 3 if mouse_model == 'Q111' else 2.5
                            ax_left.plot(x_fit, y_fit, color=color, linestyle='-',
                                        linewidth=linewidth, alpha=0.95)

        ax_left.set_xlabel('Age (months)', fontsize=11, fontweight='bold')
        ax_left.set_ylabel(f'{metric_name} per Cell', fontsize=11, fontweight='bold')
        ax_left.set_title(f'{chr(65 + row_idx*2)}) {metric_name}: Age Trends (Q111 vs WT)',
                         fontsize=12, fontweight='bold', pad=10)
        # Simplified legend - only show Q111 entries to avoid clutter
        handles, labels = ax_left.get_legend_handles_labels()
        q111_indices = [i for i, label in enumerate(labels) if 'Q111' in label]
        ax_left.legend([handles[i] for i in q111_indices[:4]],
                      [labels[i].replace(', Q111', '') for i in q111_indices[:4]],
                      loc='upper left', fontsize=7, ncol=2, title='Q111 (WT=faint)')
        ax_left.grid(axis='both', alpha=0.3, linestyle='--', linewidth=0.5)

        # Add correlation statistics for Q111 only (to keep text manageable)
        stats_text = "Q111 Correlations:\n"
        for region in ['Cortex', 'Striatum']:
            for channel in ['HTT1a', 'fl-HTT']:
                subset = df_fov[(df_fov['Mouse_Model'] == 'Q111') &
                                (df_fov['Region'] == region) &
                                (df_fov['Channel'] == channel)]
                if len(subset) > 10:
                    ages = subset['Age'].values
                    values = subset[metric].values
                    # Remove NaN
                    valid = ~(np.isnan(ages) | np.isnan(values))
                    if valid.sum() > 10:
                        r, p = pearsonr(ages[valid], values[valid])
                        if p < 0.05:
                            stats_text += f"  {channel[:7]} {region[:3]}: r={r:.2f}, p={p:.2e}\n"

        if len(stats_text.split('\n')) > 1:
            ax_left.text(0.98, 0.02, stats_text.strip(),
                        transform=ax_left.transAxes, fontsize=6,
                        verticalalignment='bottom', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                        family='monospace')

        # ══════════════════════════════════════════════════════════════════
        # RIGHT COLUMN: Age trends by channel (binned by age)
        # ══════════════════════════════════════════════════════════════════

        ax_right = fig.add_subplot(gs[row_idx, 1])

        for mouse_model in ['Q111', 'Wildtype']:
            for channel in ['HTT1a', 'fl-HTT']:
                for region in ['Cortex', 'Striatum']:
                    # Bin by age
                    age_bins = []
                    for age in sorted(df_fov['Age'].unique()):
                        subset = df_fov[(df_fov['Mouse_Model'] == mouse_model) &
                                        (df_fov['Age'] == age) &
                                        (df_fov['Region'] == region) &
                                        (df_fov['Channel'] == channel)]
                        if len(subset) > 0:
                            age_bins.append({
                                'Age': age,
                                'Mean': subset[metric].mean(),
                                'SEM': subset[metric].sem(),
                                'N': len(subset)
                            })

                    if len(age_bins) == 0:
                        continue

                    df_age = pd.DataFrame(age_bins)

                    # Determine plotting style and X-offset
                    if mouse_model == 'Q111':
                        if channel == 'HTT1a':
                            color = color_q111_mhtt1a
                        else:
                            color = color_q111_full
                        marker = 'o'
                        markersize = 10
                        linewidth = 3
                        x_offset = -0.1
                    else:  # Wildtype
                        if channel == 'HTT1a':
                            color = color_wt_mhtt1a
                        else:
                            color = color_wt_full
                        marker = 's'  # Square for wildtype
                        markersize = 9
                        linewidth = 2.5
                        x_offset = 0.3  # Larger offset for clear separation

                    alpha = 0.95

                    # Plot with X-offset
                    ax_right.errorbar(df_age['Age'] + x_offset, df_age['Mean'],
                                     yerr=df_age['SEM'],
                                     label=f"{region} ({channel}, {mouse_model})",
                                     color=color, marker=marker, markersize=markersize,
                                     linestyle='-', linewidth=linewidth, alpha=alpha,
                                     capsize=3, capthick=1.5)

        ax_right.set_xlabel('Age (months)', fontsize=11, fontweight='bold')
        ax_right.set_ylabel(f'{metric_name} per Cell', fontsize=11, fontweight='bold')
        ax_right.set_title(f'{chr(65 + row_idx*2 + 1)}) {metric_name}: Age Trends (Binned)',
                          fontsize=12, fontweight='bold', pad=10)
        # Simplified legend
        handles, labels = ax_right.get_legend_handles_labels()
        q111_indices = [i for i, label in enumerate(labels) if 'Q111' in label]
        ax_right.legend([handles[i] for i in q111_indices[:4]],
                       [labels[i].replace(', Q111', '') for i in q111_indices[:4]],
                       loc='upper left', fontsize=7, ncol=2, title='Q111 (WT=faint)')
        ax_right.grid(axis='both', alpha=0.3, linestyle='--', linewidth=0.5)

    # ══════════════════════════════════════════════════════════════════════
    # GENERATE COMPREHENSIVE CAPTION
    # ══════════════════════════════════════════════════════════════════════

    caption_lines = []
    caption_lines.append("=" * 80)
    caption_lines.append("FIGURE: mRNA Expression Age Trends (Q111 vs Wildtype)")
    caption_lines.append("=" * 80)
    caption_lines.append("")

    caption_lines.append("OVERVIEW:")
    caption_lines.append("-" * 80)
    caption_lines.append("This figure displays longitudinal age-dependent expression patterns of HTT1a and")
    caption_lines.append("fl-HTT transcripts in Q111 transgenic mice compared to wildtype controls")
    caption_lines.append("across cortical and striatal regions. Each row presents a different quantification")
    caption_lines.append("metric: total mRNA, single mRNA, and clustered mRNA per cell.")
    caption_lines.append("")

    # Data statistics
    total_fovs = len(df_fov)
    q111_fovs = len(df_fov[df_fov['Mouse_Model'] == 'Q111'])
    wt_fovs = len(df_fov[df_fov['Mouse_Model'] == 'Wildtype'])

    caption_lines.append("DATASET STATISTICS:")
    caption_lines.append("-" * 80)
    caption_lines.append(f"Total FOVs: {total_fovs}")
    caption_lines.append(f"  Q111: {q111_fovs} FOVs ({100*q111_fovs/total_fovs:.1f}%)")
    caption_lines.append(f"  Wildtype: {wt_fovs} FOVs ({100*wt_fovs/total_fovs:.1f}%)")
    caption_lines.append("")

    ages = sorted(df_fov['Age'].unique())
    caption_lines.append(f"Age timepoints: {', '.join([f'{age:.1f} months' for age in ages])}")
    for age in ages:
        q111_count = len(df_fov[(df_fov['Age'] == age) & (df_fov['Mouse_Model'] == 'Q111')])
        wt_count = len(df_fov[(df_fov['Age'] == age) & (df_fov['Mouse_Model'] == 'Wildtype')])
        caption_lines.append(f"  {age:.1f}mo: Q111={q111_count} FOVs, WT={wt_count} FOVs")
    caption_lines.append("")

    caption_lines.append("PANEL ORGANIZATION:")
    caption_lines.append("-" * 80)
    caption_lines.append("Row 1 (Panels A-B): Total mRNA per cell")
    caption_lines.append("Row 2 (Panels C-D): Single mRNA per cell")
    caption_lines.append("Row 3 (Panels E-F): Clustered mRNA per cell")
    caption_lines.append("")
    caption_lines.append("Left column: Individual FOV scatter plots with horizontal offset")
    caption_lines.append("  - Q111: circles, offset -0.1 months")
    caption_lines.append("  - Wildtype: squares, offset +0.3 months")
    caption_lines.append("Right column: Binned age trends (mean±SEM)")
    caption_lines.append("")

    # Detailed statistics per metric
    metrics = ['Total_mRNA_per_Cell', 'Single_mRNA_per_Cell', 'Clustered_mRNA_per_Cell']
    metric_names = ['Total mRNA', 'Single mRNA', 'Clustered mRNA']
    panel_letters = [['A', 'B'], ['C', 'D'], ['E', 'F']]

    for metric_idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        caption_lines.append(f"ROW {metric_idx + 1} - {metric_name.upper()} ({', '.join(panel_letters[metric_idx])}):")
        caption_lines.append("-" * 80)

        for region in ['Cortex', 'Striatum']:
            caption_lines.append(f"\n{region}:")
            for age in ages:
                for channel in ['HTT1a', 'fl-HTT']:
                    for model in ['Q111', 'Wildtype']:
                        subset = df_fov[
                            (df_fov['Region'] == region) &
                            (df_fov['Age'] == age) &
                            (df_fov['Channel'] == channel) &
                            (df_fov['Mouse_Model'] == model)
                        ][metric]

                        if len(subset) > 0:
                            mean_val = subset.mean()
                            std_val = subset.std()
                            sem_val = subset.sem()
                            median_val = subset.median()
                            n_val = len(subset)
                            caption_lines.append(
                                f"  {age:.1f}mo {model} {channel}: "
                                f"n={n_val}, mean={mean_val:.2f}±{std_val:.2f} (SEM={sem_val:.2f}), "
                                f"median={median_val:.2f}"
                            )
        caption_lines.append("")

    caption_lines.append("COLOR SCHEME:")
    caption_lines.append("-" * 80)
    caption_lines.append("Q111 (saturated colors):")
    caption_lines.append("  Cortex - HTT1a: Green, full-length: Orange")
    caption_lines.append("  Striatum - HTT1a: Green (alpha=0.5), full-length: Orange (alpha=0.5)")
    caption_lines.append("Wildtype (distinct colors):")
    caption_lines.append("  Cortex - HTT1a: Blue, full-length: Purple")
    caption_lines.append("  Striatum - HTT1a: Blue (alpha=0.6), full-length: Purple (alpha=0.6)")
    caption_lines.append("")

    caption_lines.append("QUALITY CONTROL:")
    caption_lines.append("-" * 80)
    caption_lines.append("Excluded slides: m1a2, m1b5 (technical failures)")
    caption_lines.append("Minimum nuclei per FOV: 40")
    caption_lines.append("")

    caption_lines.append("METHODOLOGY:")
    caption_lines.append("-" * 80)
    caption_lines.append("Total mRNA = N_spots + (I_cluster / I_peak_single) / N_nuclei")
    caption_lines.append("Single mRNA = N_spots / N_nuclei")
    caption_lines.append("Clustered mRNA = Total mRNA - Single mRNA")
    caption_lines.append("Slide-specific peak intensity normalization via KDE")
    caption_lines.append("")

    caption_lines.append("=" * 80)
    caption_lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    caption_lines.append("=" * 80)

    # Save caption
    caption_path = OUTPUT_DIR / "fig_age_trends_caption.txt"
    with open(caption_path, 'w') as f:
        f.write('\n'.join(caption_lines))
    print(f"  Saved caption: {caption_path}")

    # Save figure
    for fmt in ['png', 'svg', 'pdf']:
        output_path = OUTPUT_DIR / f"fig_age_trends.{fmt}"
        fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"  Saved: {output_path}")

    plt.close(fig)


def create_atlas_trend_figure(df_fov):
    """
    Create comprehensive figure showing anteroposterior atlas coordinate trends
    for all expression metrics, including both Q111 and Wildtype.
    """

    # Remove NaN atlas coordinates
    df_fov = df_fov[~df_fov['Brain_Atlas_Coord'].isna()].copy()

    # Color scheme - different for Q111 vs Wildtype
    # Q111 colors (saturated)
    color_q111_mhtt1a = '#2ecc71'  # Green
    color_q111_full = '#f39c12'  # Orange

    # Wildtype colors (desaturated/lighter)
    color_wt_mhtt1a = '#a8e6cf'  # Light green
    color_wt_full = '#ffd8a8'  # Light orange

    # Create figure with 3x2 layout
    fig = plt.figure(figsize=(18, 16), dpi=FIGURE_DPI)
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.30,
                          left=0.08, right=0.95, top=0.96, bottom=0.06)

    metrics = ['Total_mRNA_per_Cell', 'Single_mRNA_per_Cell', 'Clustered_mRNA_per_Cell']
    metric_names = ['Total mRNA', 'Single mRNA', 'Clustered mRNA']

    for row_idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):

        # ══════════════════════════════════════════════════════════════════
        # LEFT COLUMN: Atlas trends by region (scatter plots)
        # ══════════════════════════════════════════════════════════════════

        ax_left = fig.add_subplot(gs[row_idx, 0])

        for mouse_model in ['Q111', 'Wildtype']:
            for region in ['Cortex', 'Striatum']:
                for channel in ['HTT1a', 'fl-HTT']:
                    subset = df_fov[(df_fov['Mouse_Model'] == mouse_model) &
                                    (df_fov['Region'] == region) &
                                    (df_fov['Channel'] == channel)]

                    if len(subset) < 10:
                        continue

                    # Determine plotting style and X-offset
                    if mouse_model == 'Q111':
                        if channel == 'HTT1a':
                            color = color_q111_mhtt1a
                        else:
                            color = color_q111_full
                        marker = 'o'
                        markersize = 35
                        alpha = 0.6 if region == 'Cortex' else 0.5
                        x_offset = -0.3
                    else:  # Wildtype
                        if channel == 'HTT1a':
                            color = color_wt_mhtt1a
                        else:
                            color = color_wt_full
                        marker = 's'  # Square for wildtype
                        markersize = 30
                        alpha = 0.7 if region == 'Cortex' else 0.6
                        x_offset = 0.8  # Larger offset for atlas (wider range)

                    # Add jitter to scatter plot
                    coords = subset['Brain_Atlas_Coord'].values
                    values = subset[metric].values
                    np.random.seed(42)  # For reproducibility
                    coord_jitter = coords + x_offset + np.random.normal(0, 0.15, len(coords))

                    # Scatter plot with transparency and jitter
                    ax_left.scatter(coord_jitter, values,
                                   color=color, marker=marker, s=markersize, alpha=alpha,
                                   label=f"{channel} ({region}, {mouse_model})")

                    # Add trend line if significant correlation
                    valid = ~(np.isnan(coords) | np.isnan(values))

                    if valid.sum() > 10:
                        r, p = pearsonr(coords[valid], values[valid])
                        if p < 0.05:
                            slope, intercept, _, _, _ = linregress(coords[valid], values[valid])
                            x_fit = np.array([coords[valid].min(), coords[valid].max()]) + x_offset
                            y_fit = slope * (x_fit - x_offset) + intercept

                            linewidth = 3 if mouse_model == 'Q111' else 2.5
                            ax_left.plot(x_fit, y_fit, color=color, linestyle='-',
                                        linewidth=linewidth, alpha=0.95)

        ax_left.set_xlabel('Brain Atlas Coordinate (A-P position, 25μm units)',
                          fontsize=11, fontweight='bold')
        ax_left.set_ylabel(f'{metric_name} per Cell', fontsize=11, fontweight='bold')
        ax_left.set_title(f'{chr(65 + row_idx*2)}) {metric_name}: Atlas Position Trends (Q111 vs WT)',
                         fontsize=12, fontweight='bold', pad=10)
        # Simplified legend - only show Q111 entries to avoid clutter
        handles, labels = ax_left.get_legend_handles_labels()
        q111_indices = [i for i, label in enumerate(labels) if 'Q111' in label]
        ax_left.legend([handles[i] for i in q111_indices[:4]],
                      [labels[i].replace(', Q111', '') for i in q111_indices[:4]],
                      loc='upper left', fontsize=7, ncol=2, title='Q111 (WT=faint)')
        ax_left.grid(axis='both', alpha=0.3, linestyle='--', linewidth=0.5)

        # Add correlation statistics for Q111 only (to keep text manageable)
        stats_text = "Q111 Correlations:\n"
        for region in ['Cortex', 'Striatum']:
            for channel in ['HTT1a', 'fl-HTT']:
                subset = df_fov[(df_fov['Mouse_Model'] == 'Q111') &
                                (df_fov['Region'] == region) &
                                (df_fov['Channel'] == channel)]
                if len(subset) > 10:
                    coords = subset['Brain_Atlas_Coord'].values
                    values = subset[metric].values
                    valid = ~(np.isnan(coords) | np.isnan(values))
                    if valid.sum() > 10:
                        r, p = pearsonr(coords[valid], values[valid])
                        if p < 0.05:
                            stats_text += f"  {channel[:7]} {region[:3]}: r={r:.2f}, p={p:.2e}\n"

        if len(stats_text.split('\n')) > 1:
            ax_left.text(0.98, 0.02, stats_text.strip(),
                        transform=ax_left.transAxes, fontsize=6,
                        verticalalignment='bottom', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                        family='monospace')

        # ══════════════════════════════════════════════════════════════════
        # RIGHT COLUMN: Individual atlas coordinate binning (one per coordinate)
        # ══════════════════════════════════════════════════════════════════

        ax_right = fig.add_subplot(gs[row_idx, 1])

        for mouse_model in ['Q111', 'Wildtype']:
            for region in ['Cortex', 'Striatum']:
                for channel in ['HTT1a', 'fl-HTT']:
                    subset = df_fov[(df_fov['Mouse_Model'] == mouse_model) &
                                    (df_fov['Region'] == region) &
                                    (df_fov['Channel'] == channel)]

                    if len(subset) < 10:
                        continue

                    # Group by individual atlas coordinates
                    coord_stats = []
                    unique_coords = sorted(subset['Brain_Atlas_Coord'].unique())

                    for coord in unique_coords:
                        coord_data = subset[subset['Brain_Atlas_Coord'] == coord]
                        if len(coord_data) >= 3:  # Require at least 3 FOVs per coordinate
                            coord_stats.append({
                                'Atlas_Coord': coord,
                                'Mean': coord_data[metric].mean(),
                                'SEM': coord_data[metric].sem(),
                                'N': len(coord_data)
                            })

                    if len(coord_stats) == 0:
                        continue

                    df_coords = pd.DataFrame(coord_stats)

                    # Determine plotting style and X-offset
                    if mouse_model == 'Q111':
                        if channel == 'HTT1a':
                            color = color_q111_mhtt1a
                        else:
                            color = color_q111_full
                        marker = 'o'
                        markersize = 10
                        linewidth = 3
                        x_offset = -0.3
                    else:  # Wildtype
                        if channel == 'HTT1a':
                            color = color_wt_mhtt1a
                        else:
                            color = color_wt_full
                        marker = 's'  # Square for wildtype
                        markersize = 9
                        linewidth = 2.5
                        x_offset = 0.8

                    alpha = 0.95

                    # Plot with X-offset
                    ax_right.errorbar(df_coords['Atlas_Coord'] + x_offset, df_coords['Mean'],
                                     yerr=df_coords['SEM'],
                                     label=f"{region} ({channel}, {mouse_model})",
                                     color=color, marker=marker, markersize=markersize,
                                     linestyle='-', linewidth=linewidth, alpha=alpha,
                                     capsize=3, capthick=1.5)

        ax_right.set_xlabel('Brain Atlas Coordinate (A-P position, 25μm units)',
                           fontsize=11, fontweight='bold')
        ax_right.set_ylabel(f'{metric_name} per Cell', fontsize=11, fontweight='bold')
        ax_right.set_title(f'{chr(65 + row_idx*2 + 1)}) {metric_name}: Per-Coordinate Atlas Trends',
                          fontsize=12, fontweight='bold', pad=10)
        # Simplified legend
        handles, labels = ax_right.get_legend_handles_labels()
        q111_indices = [i for i, label in enumerate(labels) if 'Q111' in label]
        ax_right.legend([handles[i] for i in q111_indices[:4]],
                       [labels[i].replace(', Q111', '') for i in q111_indices[:4]],
                       loc='upper left', fontsize=7, ncol=2, title='Q111 (WT=faint)')
        ax_right.grid(axis='both', alpha=0.3, linestyle='--', linewidth=0.5)

    # ══════════════════════════════════════════════════════════════════════
    # GENERATE COMPREHENSIVE CAPTION
    # ══════════════════════════════════════════════════════════════════════

    caption_lines = []
    caption_lines.append("=" * 80)
    caption_lines.append("FIGURE: mRNA Expression Atlas Coordinate Trends (Q111 vs Wildtype)")
    caption_lines.append("=" * 80)
    caption_lines.append("")

    caption_lines.append("OVERVIEW:")
    caption_lines.append("-" * 80)
    caption_lines.append("This figure displays spatial expression patterns of HTT1a and fl-HTT")
    caption_lines.append("transcripts along the anterior-posterior axis of the mouse brain in Q111 transgenic")
    caption_lines.append("mice compared to wildtype controls. Brain atlas coordinates are expressed in 25μm")
    caption_lines.append("units from Bregma. Each row presents a different quantification metric: total mRNA,")
    caption_lines.append("single mRNA, and clustered mRNA per cell.")
    caption_lines.append("")

    # Data statistics
    df_fov_clean = df_fov.dropna(subset=['Brain_Atlas_Coord'])
    total_fovs = len(df_fov_clean)
    q111_fovs = len(df_fov_clean[df_fov_clean['Mouse_Model'] == 'Q111'])
    wt_fovs = len(df_fov_clean[df_fov_clean['Mouse_Model'] == 'Wildtype'])

    caption_lines.append("DATASET STATISTICS:")
    caption_lines.append("-" * 80)
    caption_lines.append(f"Total FOVs (with atlas coordinates): {total_fovs}")
    caption_lines.append(f"  Q111: {q111_fovs} FOVs ({100*q111_fovs/total_fovs:.1f}%)")
    caption_lines.append(f"  Wildtype: {wt_fovs} FOVs ({100*wt_fovs/total_fovs:.1f}%)")
    caption_lines.append("")

    for region in ['Cortex', 'Striatum']:
        region_fovs = len(df_fov_clean[df_fov_clean['Region'] == region])
        coords = sorted(df_fov_clean[df_fov_clean['Region'] == region]['Brain_Atlas_Coord'].unique())
        caption_lines.append(f"{region}: {region_fovs} FOVs across {len(coords)} coordinates")
        caption_lines.append(f"  Coordinate range: {min(coords):.0f} to {max(coords):.0f} (25μm units)")
    caption_lines.append("")

    caption_lines.append("PANEL ORGANIZATION:")
    caption_lines.append("-" * 80)
    caption_lines.append("Row 1 (Panels A-B): Total mRNA per cell")
    caption_lines.append("Row 2 (Panels C-D): Single mRNA per cell")
    caption_lines.append("Row 3 (Panels E-F): Clustered mRNA per cell")
    caption_lines.append("")
    caption_lines.append("Left column: Individual FOV scatter plots with horizontal offset")
    caption_lines.append("  - Q111: circles, offset -0.3 units")
    caption_lines.append("  - Wildtype: squares, offset +0.8 units")
    caption_lines.append("Right column: Per-coordinate trends (mean±SEM)")
    caption_lines.append("")

    # Detailed statistics per metric
    metrics = ['Total_mRNA_per_Cell', 'Single_mRNA_per_Cell', 'Clustered_mRNA_per_Cell']
    metric_names = ['Total mRNA', 'Single mRNA', 'Clustered mRNA']
    panel_letters = [['A', 'B'], ['C', 'D'], ['E', 'F']]

    for metric_idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        caption_lines.append(f"ROW {metric_idx + 1} - {metric_name.upper()} ({', '.join(panel_letters[metric_idx])}):")
        caption_lines.append("-" * 80)

        for region in ['Cortex', 'Striatum']:
            caption_lines.append(f"\n{region}:")
            coords = sorted(df_fov_clean[df_fov_clean['Region'] == region]['Brain_Atlas_Coord'].unique())

            for coord in coords:
                coord_data = []
                for channel in ['HTT1a', 'fl-HTT']:
                    for model in ['Q111', 'Wildtype']:
                        subset = df_fov_clean[
                            (df_fov_clean['Region'] == region) &
                            (df_fov_clean['Brain_Atlas_Coord'] == coord) &
                            (df_fov_clean['Channel'] == channel) &
                            (df_fov_clean['Mouse_Model'] == model)
                        ][metric]

                        if len(subset) > 0:
                            mean_val = subset.mean()
                            sem_val = subset.sem()
                            n_val = len(subset)
                            coord_data.append(f"{model} {channel}: n={n_val}, mean={mean_val:.2f}±{sem_val:.2f}")

                if coord_data:
                    caption_lines.append(f"  Coord {coord:.0f}: {' | '.join(coord_data)}")
        caption_lines.append("")

    caption_lines.append("COLOR SCHEME:")
    caption_lines.append("-" * 80)
    caption_lines.append("Same as age trends figure:")
    caption_lines.append("Q111: Green (HTT1a), Orange (full-length)")
    caption_lines.append("Wildtype: Blue (HTT1a), Purple (full-length)")
    caption_lines.append("Striatum data shown with reduced alpha for visual distinction")
    caption_lines.append("")

    caption_lines.append("QUALITY CONTROL:")
    caption_lines.append("-" * 80)
    caption_lines.append("Excluded slides: m1a2, m1b5 (technical failures)")
    caption_lines.append("Minimum nuclei per FOV: 40")
    caption_lines.append("FOVs without atlas coordinates excluded from this analysis")
    caption_lines.append("")

    caption_lines.append("METHODOLOGY:")
    caption_lines.append("-" * 80)
    caption_lines.append("Atlas coordinates: Anterior-posterior position in 25μm units from Bregma")
    caption_lines.append("Total mRNA = N_spots + (I_cluster / I_peak_single) / N_nuclei")
    caption_lines.append("Single mRNA = N_spots / N_nuclei")
    caption_lines.append("Clustered mRNA = Total mRNA - Single mRNA")
    caption_lines.append("Slide-specific peak intensity normalization via KDE")
    caption_lines.append("")

    caption_lines.append("=" * 80)
    caption_lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    caption_lines.append("=" * 80)

    # Save caption
    caption_path = OUTPUT_DIR / "fig_atlas_trends_caption.txt"
    with open(caption_path, 'w') as f:
        f.write('\n'.join(caption_lines))
    print(f"  Saved caption: {caption_path}")

    # Save figure
    for fmt in ['png', 'svg', 'pdf']:
        output_path = OUTPUT_DIR / f"fig_atlas_trends.{fmt}"
        fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"  Saved: {output_path}")

    plt.close(fig)


def create_combined_age_atlas_figure(df_fov):
    """
    Create a combined figure showing both age and atlas coordinate effects
    on total mRNA expression, with 2D heatmaps.
    """

    # Filter for Q111 mice only
    df_q111 = df_fov[df_fov['Mouse_Model'] == 'Q111'].copy()

    # Remove NaN atlas coordinates
    df_q111 = df_q111[~df_q111['Brain_Atlas_Coord'].isna()].copy()

    # Create figure with 2x2 layout
    fig = plt.figure(figsize=(16, 14), dpi=FIGURE_DPI)
    gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.25,
                          left=0.08, right=0.92, top=0.94, bottom=0.08)

    panel_idx = 0

    for region_idx, region in enumerate(['Cortex', 'Striatum']):
        for channel_idx, channel in enumerate(['HTT1a', 'fl-HTT']):

            ax = fig.add_subplot(gs[region_idx, channel_idx])

            subset = df_q111[(df_q111['Region'] == region) &
                            (df_q111['Channel'] == channel)]

            if len(subset) < 50:
                continue

            # Create 2D bins for age and atlas coordinate
            age_bins = [2, 6, 12]  # Age boundaries
            atlas_bins = np.linspace(subset['Brain_Atlas_Coord'].min(),
                                    subset['Brain_Atlas_Coord'].max(), 8)

            # Build heatmap data
            heatmap_data = np.zeros((len(age_bins), len(atlas_bins)-1))
            heatmap_counts = np.zeros((len(age_bins), len(atlas_bins)-1))

            for i, age in enumerate(age_bins):
                age_mask = subset['Age'] == age

                for j in range(len(atlas_bins) - 1):
                    atlas_mask = ((subset['Brain_Atlas_Coord'] >= atlas_bins[j]) &
                                 (subset['Brain_Atlas_Coord'] < atlas_bins[j+1]))

                    bin_data = subset[age_mask & atlas_mask]

                    if len(bin_data) > 0:
                        heatmap_data[i, j] = bin_data['Total_mRNA_per_Cell'].mean()
                        heatmap_counts[i, j] = len(bin_data)
                    else:
                        heatmap_data[i, j] = np.nan

            # Plot heatmap
            im = ax.imshow(heatmap_data, aspect='auto', origin='lower',
                          cmap='viridis', interpolation='nearest')

            # Colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Total mRNA per Cell', fontsize=10)

            # Labels
            ax.set_yticks(np.arange(len(age_bins)))
            ax.set_yticklabels([f'{int(a)}mo' for a in age_bins])
            ax.set_ylabel('Age', fontsize=11, fontweight='bold')

            atlas_tick_positions = np.arange(len(atlas_bins)-1)
            atlas_tick_labels = [f'{int((atlas_bins[i] + atlas_bins[i+1])/2)}'
                                for i in range(len(atlas_bins)-1)]
            ax.set_xticks(atlas_tick_positions[::2])  # Show every other label
            ax.set_xticklabels(atlas_tick_labels[::2], rotation=45, ha='right')
            ax.set_xlabel('Brain Atlas Coordinate (A-P)', fontsize=11, fontweight='bold')

            ax.set_title(f'{chr(65 + panel_idx)}) {channel} - {region}',
                        fontsize=12, fontweight='bold', pad=10)

            # Add sample counts as text
            for i in range(len(age_bins)):
                for j in range(len(atlas_bins)-1):
                    if heatmap_counts[i, j] > 0:
                        text_color = 'white' if heatmap_data[i, j] > np.nanmean(heatmap_data) else 'black'
                        ax.text(j, i, f'{int(heatmap_counts[i, j])}',
                               ha='center', va='center', fontsize=7,
                               color=text_color, weight='bold')

            panel_idx += 1

    # ══════════════════════════════════════════════════════════════════════
    # GENERATE COMPREHENSIVE CAPTION
    # ══════════════════════════════════════════════════════════════════════

    caption_lines = []
    caption_lines.append("=" * 80)
    caption_lines.append("FIGURE: Age × Atlas Coordinate Heatmap (Q111 mice only)")
    caption_lines.append("=" * 80)
    caption_lines.append("")

    caption_lines.append("OVERVIEW:")
    caption_lines.append("-" * 80)
    caption_lines.append("This figure presents 2D heatmaps showing the combined effects of age and")
    caption_lines.append("anterior-posterior brain position on total mRNA expression in Q111 transgenic mice.")
    caption_lines.append("Each panel displays one combination of brain region (Cortex/Striatum) and")
    caption_lines.append("transcript type (HTT1a/fl-HTT). Heat intensity represents mean total")
    caption_lines.append("mRNA per cell, with numbers overlaid showing sample counts (FOVs) per bin.")
    caption_lines.append("")

    # Get data for Q111 only
    df_q111 = df_fov[df_fov['Mouse_Model'] == 'Q111'].dropna(subset=['Brain_Atlas_Coord'])

    caption_lines.append("DATASET STATISTICS:")
    caption_lines.append("-" * 80)
    caption_lines.append(f"Total Q111 FOVs (with atlas coordinates): {len(df_q111)}")
    caption_lines.append("")

    # Age and atlas binning
    age_bins = sorted(df_q111['Age'].unique())
    atlas_min = df_q111['Brain_Atlas_Coord'].min()
    atlas_max = df_q111['Brain_Atlas_Coord'].max()

    caption_lines.append("BINNING STRATEGY:")
    caption_lines.append("-" * 80)
    caption_lines.append(f"Age bins: {len(age_bins)} discrete timepoints ({', '.join([f'{a:.1f}mo' for a in age_bins])})")
    caption_lines.append(f"Atlas bins: 5 equal-width bins spanning {atlas_min:.0f} to {atlas_max:.0f} (25μm units)")
    caption_lines.append("")

    # Panel descriptions with statistics
    caption_lines.append("PANEL DESCRIPTIONS:")
    caption_lines.append("-" * 80)

    panel_labels = ['A', 'B', 'C', 'D']
    panel_idx = 0

    for region in ['Cortex', 'Striatum']:
        for channel in ['HTT1a', 'fl-HTT']:
            subset = df_q111[
                (df_q111['Region'] == region) &
                (df_q111['Channel'] == channel)
            ]

            caption_lines.append(f"\nPanel {panel_labels[panel_idx]}: {channel} - {region}")
            caption_lines.append(f"  Total FOVs: {len(subset)}")
            caption_lines.append(f"  Mean expression: {subset['Total_mRNA_per_Cell'].mean():.2f} ± {subset['Total_mRNA_per_Cell'].std():.2f} mRNA/cell")
            caption_lines.append(f"  Range: {subset['Total_mRNA_per_Cell'].min():.2f} - {subset['Total_mRNA_per_Cell'].max():.2f} mRNA/cell")

            # Age breakdown
            caption_lines.append(f"  Per-age breakdown:")
            for age in age_bins:
                age_subset = subset[subset['Age'] == age]
                if len(age_subset) > 0:
                    caption_lines.append(
                        f"    {age:.1f}mo: n={len(age_subset)}, "
                        f"mean={age_subset['Total_mRNA_per_Cell'].mean():.2f}±{age_subset['Total_mRNA_per_Cell'].std():.2f}"
                    )

            panel_idx += 1

    caption_lines.append("")
    caption_lines.append("INTERPRETATION:")
    caption_lines.append("-" * 80)
    caption_lines.append("This visualization enables identification of spatiotemporal expression patterns,")
    caption_lines.append("revealing whether fl-HTT expression varies systematically with:")
    caption_lines.append("  1. Age (vertical axis): Progressive changes over disease course")
    caption_lines.append("  2. Brain position (horizontal axis): Regional vulnerability along A-P axis")
    caption_lines.append("  3. Interaction effects: Age-dependent regional changes")
    caption_lines.append("")
    caption_lines.append("Numbers in each cell indicate FOV count (sample size) for that bin.")
    caption_lines.append("Empty/low-count bins appear due to uneven spatial sampling across ages.")
    caption_lines.append("")

    caption_lines.append("COLOR SCALE:")
    caption_lines.append("-" * 80)
    caption_lines.append("Colormap: 'viridis' (perceptually uniform)")
    caption_lines.append("Range: Automatically scaled to data range per panel")
    caption_lines.append("Interpretation: Warmer colors = higher mRNA expression")
    caption_lines.append("")

    caption_lines.append("QUALITY CONTROL:")
    caption_lines.append("-" * 80)
    caption_lines.append("Excluded slides: m1a2, m1b5 (technical failures)")
    caption_lines.append("Minimum nuclei per FOV: 40")
    caption_lines.append("Wildtype data not included (sparse sampling across age/space)")
    caption_lines.append("")

    caption_lines.append("METHODOLOGY:")
    caption_lines.append("-" * 80)
    caption_lines.append("Total mRNA = N_spots + (I_cluster / I_peak_single) / N_nuclei")
    caption_lines.append("Slide-specific peak intensity normalization via KDE")
    caption_lines.append("Atlas coordinates: Anterior-posterior position in 25μm units from Bregma")
    caption_lines.append("Heatmap values: Mean total mRNA per cell within each age×atlas bin")
    caption_lines.append("")

    caption_lines.append("=" * 80)
    caption_lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    caption_lines.append("=" * 80)

    # Save caption
    caption_path = OUTPUT_DIR / "fig_age_atlas_heatmap_caption.txt"
    with open(caption_path, 'w') as f:
        f.write('\n'.join(caption_lines))
    print(f"  Saved caption: {caption_path}")

    # Save figure
    for fmt in ['png', 'svg', 'pdf']:
        output_path = OUTPUT_DIR / f"fig_age_atlas_heatmap.{fmt}"
        fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"  Saved: {output_path}")

    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("="*70)
    print("AGE AND ATLAS COORDINATE TREND ANALYSIS")
    print("="*70)

    # Check if FOV-level data exists, if not, generate it automatically
    # FOV-level data is generated by fig_expression_analysis_q111.py
    fov_data_path = Path(__file__).parent / "output" / "expression_analysis_q111" / "fov_level_data.csv"

    if not fov_data_path.exists():
        print(f"\nFOV data not found at {fov_data_path}")
        print("Automatically running fig_expression_analysis_q111.py to generate FOV data...")
        print("-"*70)

        import subprocess
        script_path = Path(__file__).parent / "fig_expression_analysis_q111.py"
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=False,
            text=True
        )

        if result.returncode != 0:
            print(f"\nERROR: Failed to generate FOV data (exit code {result.returncode})")
            print("Please check fig_expression_analysis_q111.py for errors.")
            exit(1)

        print("-"*70)
        print("FOV data generation complete!")
        print("-"*70)

    print(f"\nLoading FOV data from: {fov_data_path}")
    df_fov = pd.read_csv(fov_data_path)
    print(f"Loaded {len(df_fov)} FOV records")

    # Check if wildtype data is present
    if 'Mouse_Model' in df_fov.columns:
        print(f"Mouse models in data: {df_fov['Mouse_Model'].unique()}")
        print(f"Record counts: {dict(df_fov['Mouse_Model'].value_counts())}")
    else:
        print("Warning: Mouse_Model column not found in FOV data")

    # ──────────────────────────────────────────────────────────────────────
    # FIGURE 1: AGE TRENDS
    # ──────────────────────────────────────────────────────────────────────

    print("\n" + "="*70)
    print("CREATING AGE TREND FIGURE")
    print("="*70)

    create_age_trend_figure(df_fov)

    # ──────────────────────────────────────────────────────────────────────
    # FIGURE 2: ATLAS COORDINATE TRENDS
    # ──────────────────────────────────────────────────────────────────────

    print("\n" + "="*70)
    print("CREATING ATLAS COORDINATE TREND FIGURE")
    print("="*70)

    create_atlas_trend_figure(df_fov)

    # ──────────────────────────────────────────────────────────────────────
    # FIGURE 3: COMBINED AGE × ATLAS HEATMAPS
    # ──────────────────────────────────────────────────────────────────────

    print("\n" + "="*70)
    print("CREATING COMBINED AGE × ATLAS HEATMAP FIGURE")
    print("="*70)

    create_combined_age_atlas_figure(df_fov)

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
