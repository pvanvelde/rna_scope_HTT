"""
Comprehensive Single mRNA Expression Figure
Upper rows: Age and atlas coordinate trends with violin plots
Bottom row: Per-mouse ID breakdown

Author: Generated with Claude Code
Date: 2025-11-17
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
from scipy import stats

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from results_config import FIGURE_DPI, FIGURE_FORMAT

# Output directory
OUTPUT_DIR = Path(__file__).parent / "output" / "single_mrna_comprehensive"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Figure settings - larger fonts for publication
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 15
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 11


def extract_mouse_info(mouse_id):
    """
    Extract age and mouse number from Mouse_ID string.
    e.g., 'Q111 2mo #3 UNT Early STR' -> (2, 3)
          'WT 12mo #2 aCSF Mid STR' -> (12, 2)
    """
    import re
    age_match = re.search(r'(\d+)mo', mouse_id)
    num_match = re.search(r'#(\d+)', mouse_id)
    age = int(age_match.group(1)) if age_match else 0
    num = int(num_match.group(1)) if num_match else 0
    return (age, num)


def build_slide_sublabels(mouse_ids, genotype_prefix, df_fov=None):
    """
    Create labels with sub-indices for each slide within a mouse.
    Labels are assigned based on age order:
    - Q#1-3: 2mo mice
    - Q#4-6: 6mo mice
    - Q#7-9: 12mo mice (for Q111)
    - W#1: 2mo WT, W#2: 12mo WT
    """
    import re
    # Extract age and original mouse number from each Mouse_ID
    id_info = {}
    for mid in mouse_ids:
        age, orig_num = extract_mouse_info(mid)
        id_info[mid] = {'age': age, 'orig_num': orig_num}

    # Sort by age, then by original mouse number, then by Mouse_ID for consistency
    sorted_ids = sorted(mouse_ids, key=lambda x: (id_info[x]['age'], id_info[x]['orig_num'], x))

    # Assign sequential mouse numbers based on age-ordered appearance
    sublabels = {}
    mouse_counter = 0
    current_key = None  # (age, orig_num) tuple to identify unique mice

    for mid in sorted_ids:
        key = (id_info[mid]['age'], id_info[mid]['orig_num'])
        if key != current_key:
            current_key = key
            mouse_counter += 1
            slide_idx = 1
        else:
            slide_idx += 1
        sublabels[mid] = f"{genotype_prefix}#{mouse_counter}.{slide_idx}"

    return sublabels

# Color scheme
COLOR_Q111_MHTT1A = '#2ecc71'  # Green
COLOR_Q111_FULL = '#f39c12'  # Orange
COLOR_WT_MHTT1A = '#3498db'  # Blue
COLOR_WT_FULL = '#9b59b6'  # Purple


def create_comprehensive_figure(df_fov):
    """
    Create comprehensive figure with:
    - Row 1: Age trends (violin plots) for Cortex and Striatum
    - Row 2: Atlas coordinate trends (violin plots) for Cortex and Striatum
    - Row 3: Per-mouse ID breakdown
    """

    # Filter for single mRNA only
    df_fov = df_fov.copy()

    # Create figure with 3 rows - smaller figure with larger fonts
    fig = plt.figure(figsize=(14, 12), dpi=FIGURE_DPI)
    gs = fig.add_gridspec(3, 4, hspace=0.40, wspace=0.35,
                          left=0.08, right=0.96, top=0.95, bottom=0.07,
                          height_ratios=[1, 1, 1.2])

    # ══════════════════════════════════════════════════════════════════════════
    # ROW 1: AGE TRENDS - VIOLIN PLOTS
    # ══════════════════════════════════════════════════════════════════════════

    for col_idx, region in enumerate(['Cortex', 'Striatum']):
        ax = fig.add_subplot(gs[0, col_idx*2:(col_idx+1)*2])

        # Prepare data for violin plots
        df_region = df_fov[df_fov['Region'] == region].copy()

        # Create age-channel-model combinations
        plot_data = []
        for age in sorted(df_region['Age'].unique()):
            for model in ['Q111', 'Wildtype']:
                for channel in ['HTT1a', 'fl-HTT']:
                    subset = df_region[
                        (df_region['Age'] == age) &
                        (df_region['Mouse_Model'] == model) &
                        (df_region['Channel'] == channel)
                    ]

                    for val in subset['Single_mRNA_per_Cell'].values:
                        if not np.isnan(val):
                            plot_data.append({
                                'Age': age,
                                'Model': model,
                                'Channel': channel,
                                'Value': val
                            })

        df_plot = pd.DataFrame(plot_data)

        if len(df_plot) == 0:
            continue

        # Create positions for violin plots
        ages = sorted(df_plot['Age'].unique())
        x_positions = []
        x_labels = []
        colors_list = []

        for i, age in enumerate(ages):
            base_x = i * 4

            # Q111 - HTT1a
            x_positions.append(base_x)
            x_labels.append(f'{age}mo\nQ111\nHTT1a')
            colors_list.append(COLOR_Q111_MHTT1A)

            # Q111 - full-length
            x_positions.append(base_x + 0.8)
            x_labels.append(f'{age}mo\nQ111\nfull')
            colors_list.append(COLOR_Q111_FULL)

            # WT - HTT1a
            x_positions.append(base_x + 1.8)
            x_labels.append(f'{age}mo\nWT\nHTT1a')
            colors_list.append(COLOR_WT_MHTT1A)

            # WT - full-length
            x_positions.append(base_x + 2.6)
            x_labels.append(f'{age}mo\nWT\nfull')
            colors_list.append(COLOR_WT_FULL)

        # Plot violins
        pos_idx = 0
        for i, age in enumerate(ages):
            for model in ['Q111', 'Wildtype']:
                for channel in ['HTT1a', 'fl-HTT']:
                    subset = df_plot[
                        (df_plot['Age'] == age) &
                        (df_plot['Model'] == model) &
                        (df_plot['Channel'] == channel)
                    ]

                    if len(subset) > 5:
                        color = colors_list[pos_idx]
                        parts = ax.violinplot(
                            subset['Value'].values,
                            positions=[x_positions[pos_idx]],
                            widths=0.7,
                            showmeans=True,
                            showextrema=True
                        )

                        # Color the violin
                        for pc in parts['bodies']:
                            pc.set_facecolor(color)
                            pc.set_alpha(0.7)
                            pc.set_edgecolor('black')
                            pc.set_linewidth(1.5)

                        # Color the lines
                        for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans'):
                            if partname in parts:
                                parts[partname].set_edgecolor('black')
                                parts[partname].set_linewidth(1.5)

                    pos_idx += 1

        ax.set_xticks([x_positions[i*4] + 1.3 for i in range(len(ages))])
        ax.set_xticklabels([f'{age}mo' for age in ages], fontsize=10, fontweight='bold')
        ax.set_ylabel('Single mRNA/nucleus', fontsize=12, fontweight='bold')
        ax.set_title(f'{chr(65 + col_idx)}) {region}: Single mRNA by Age',
                    fontsize=13, fontweight='bold', pad=15)
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=COLOR_Q111_MHTT1A, alpha=0.7, label='Q111 - HTT1a'),
            Patch(facecolor=COLOR_Q111_FULL, alpha=0.7, label='Q111 - full-length'),
            Patch(facecolor=COLOR_WT_MHTT1A, alpha=0.7, label='WT - HTT1a'),
            Patch(facecolor=COLOR_WT_FULL, alpha=0.7, label='WT - full-length')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9, ncol=2)

        # Add statistical tests (Q111 vs Wildtype comparisons)
        from scipy.stats import ttest_ind

        # Perform t-tests for each age and channel
        test_results = []
        y_max = ax.get_ylim()[1]
        text_y_pos = y_max * 0.95

        for i, age in enumerate(ages):
            for j, channel in enumerate(['HTT1a', 'fl-HTT']):
                q111_data = df_plot[
                    (df_plot['Age'] == age) &
                    (df_plot['Model'] == 'Q111') &
                    (df_plot['Channel'] == channel)
                ]['Value'].values

                wt_data = df_plot[
                    (df_plot['Age'] == age) &
                    (df_plot['Model'] == 'Wildtype') &
                    (df_plot['Channel'] == channel)
                ]['Value'].values

                if len(q111_data) >= 3 and len(wt_data) >= 3:
                    t_stat, p_val = ttest_ind(q111_data, wt_data)
                    test_results.append({
                        'region': region,
                        'age': age,
                        'channel': channel,
                        'n_q111': len(q111_data),
                        'n_wt': len(wt_data),
                        't_stat': t_stat,
                        'p_value': p_val
                    })

                    # Add p-value annotation
                    if p_val < 0.001:
                        sig_text = '***'
                    elif p_val < 0.01:
                        sig_text = '**'
                    elif p_val < 0.05:
                        sig_text = '*'
                    else:
                        sig_text = 'ns'

                    # Position text above the violin pairs
                    x_pos = i * 4 + j * 0.8 + 0.9
                    if sig_text != 'ns':
                        ax.text(x_pos, text_y_pos, sig_text,
                               ha='center', va='top', fontsize=8, fontweight='bold')

    # Store test results for caption
    if 'age_trend_tests' not in locals():
        age_trend_tests = []
    age_trend_tests.extend(test_results)

    # ══════════════════════════════════════════════════════════════════════════
    # ROW 2: ATLAS COORDINATE TRENDS - VIOLIN PLOTS (per coordinate, no binning)
    # ══════════════════════════════════════════════════════════════════════════

    for col_idx, region in enumerate(['Cortex', 'Striatum']):
        ax = fig.add_subplot(gs[1, col_idx*2:(col_idx+1)*2])

        # Prepare data for violin plots
        df_region = df_fov[(df_fov['Region'] == region) &
                           (~df_fov['Brain_Atlas_Coord'].isna())].copy()

        # Get unique atlas coordinates (sorted)
        unique_coords = sorted(df_region['Brain_Atlas_Coord'].unique())

        plot_data = []
        for coord in unique_coords:
            for model in ['Q111', 'Wildtype']:
                for channel in ['HTT1a', 'fl-HTT']:
                    subset = df_region[
                        (df_region['Brain_Atlas_Coord'] == coord) &
                        (df_region['Mouse_Model'] == model) &
                        (df_region['Channel'] == channel)
                    ]

                    for val in subset['Single_mRNA_per_Cell'].values:
                        if not np.isnan(val):
                            plot_data.append({
                                'Coord': coord,
                                'Model': model,
                                'Channel': channel,
                                'Value': val
                            })

        df_plot = pd.DataFrame(plot_data)

        if len(df_plot) == 0:
            continue

        # Create positions for violin plots
        x_positions = []
        colors_list = []

        for coord_idx, coord in enumerate(unique_coords):
            base_x = coord_idx * 4

            # Q111 - HTT1a
            x_positions.append(base_x)
            colors_list.append(COLOR_Q111_MHTT1A)

            # Q111 - full-length
            x_positions.append(base_x + 0.8)
            colors_list.append(COLOR_Q111_FULL)

            # WT - HTT1a
            x_positions.append(base_x + 1.8)
            colors_list.append(COLOR_WT_MHTT1A)

            # WT - full-length
            x_positions.append(base_x + 2.6)
            colors_list.append(COLOR_WT_FULL)

        # Plot box plots
        pos_idx = 0
        for coord in unique_coords:
            for model in ['Q111', 'Wildtype']:
                for channel in ['HTT1a', 'fl-HTT']:
                    subset = df_plot[
                        (df_plot['Coord'] == coord) &
                        (df_plot['Model'] == model) &
                        (df_plot['Channel'] == channel)
                    ]

                    if len(subset) > 2:  # Need at least 2 points for box plot
                        color = colors_list[pos_idx]

                        # Create box plot
                        bp = ax.boxplot(
                            [subset['Value'].values],
                            positions=[x_positions[pos_idx]],
                            widths=0.7,  # Bigger boxes
                            patch_artist=True,
                            showmeans=True,
                            meanprops=dict(marker='D', markerfacecolor=color,
                                         markeredgecolor='black', markersize=5),
                            medianprops=dict(linewidth=0),  # Remove median line
                            boxprops=dict(facecolor=color, edgecolor='black',
                                        linewidth=1.5, alpha=0.8),
                            whiskerprops=dict(color='black', linewidth=1.5),
                            capprops=dict(color='black', linewidth=1.5),
                            flierprops=dict(marker='o', markerfacecolor=color,
                                          markersize=3, alpha=0.5,
                                          markeredgecolor='none')
                        )

                    pos_idx += 1

        ax.set_xticks([x_positions[i*4] + 1.3 for i in range(len(unique_coords))])
        ax.set_xticklabels([f'{int(c)}' for c in unique_coords], fontsize=8)
        ax.set_xlabel('Brain Atlas Coordinate (A-P position, 25μm units)',
                     fontsize=11, fontweight='bold')
        ax.set_ylabel('Single mRNA/nucleus', fontsize=12, fontweight='bold')
        ax.set_title(f'{chr(67 + col_idx)}) {region}: Single mRNA by Atlas Position',
                    fontsize=13, fontweight='bold', pad=15)
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)

    # ══════════════════════════════════════════════════════════════════════════
    # ROW 3: PER-SLIDE BREAKDOWN (all FOVs concatenated per slide)
    # ══════════════════════════════════════════════════════════════════════════

    # Separate Q111 and WT data
    df_q111 = df_fov[df_fov['Mouse_Model'] == 'Q111'].copy()
    df_wt = df_fov[df_fov['Mouse_Model'] == 'Wildtype'].copy()

    # Build mouse labels for Q111 and WT (using Mouse_ID to get Q#X.Y format)
    q111_mouse_ids = df_q111['Mouse_ID'].unique()
    wt_mouse_ids = df_wt['Mouse_ID'].unique()
    q111_labels = build_slide_sublabels(q111_mouse_ids, 'Q')
    wt_labels = build_slide_sublabels(wt_mouse_ids, 'W')

    for col_idx, region in enumerate(['Cortex', 'Striatum']):
        ax = fig.add_subplot(gs[2, col_idx*2:(col_idx+1)*2])

        # Prepare per-mouse data (all FOVs concatenated per Mouse_ID)
        mouse_data = []

        # Process Q111 mice first (sorted by age, then original mouse number)
        for mouse_id in sorted(q111_mouse_ids, key=lambda x: (extract_mouse_info(x), x)):
            mouse_subset = df_q111[
                (df_q111['Mouse_ID'] == mouse_id) &
                (df_q111['Region'] == region)
            ]

            if len(mouse_subset) == 0:
                continue

            age = mouse_subset['Age'].iloc[0]
            mouse_label = q111_labels.get(mouse_id, mouse_id)
            n_fovs = len(mouse_subset[mouse_subset['Channel'] == 'HTT1a'])

            for channel in ['HTT1a', 'fl-HTT']:
                channel_subset = mouse_subset[mouse_subset['Channel'] == channel]

                if len(channel_subset) > 0:
                    mean_val = channel_subset['Single_mRNA_per_Cell'].mean()
                    sem_val = channel_subset['Single_mRNA_per_Cell'].sem()

                    mouse_data.append({
                        'Mouse_ID': mouse_id,
                        'Mouse_Label': mouse_label,
                        'Model': 'Q111',
                        'Age': age,
                        'Channel': channel,
                        'Mean': mean_val,
                        'SEM': sem_val,
                        'N_FOVs': n_fovs
                    })

        # Process WT mice (sorted by age, then original mouse number)
        for mouse_id in sorted(wt_mouse_ids, key=lambda x: (extract_mouse_info(x), x)):
            mouse_subset = df_wt[
                (df_wt['Mouse_ID'] == mouse_id) &
                (df_wt['Region'] == region)
            ]

            if len(mouse_subset) == 0:
                continue

            age = mouse_subset['Age'].iloc[0]
            mouse_label = wt_labels.get(mouse_id, mouse_id)
            n_fovs = len(mouse_subset[mouse_subset['Channel'] == 'HTT1a'])

            for channel in ['HTT1a', 'fl-HTT']:
                channel_subset = mouse_subset[mouse_subset['Channel'] == channel]

                if len(channel_subset) > 0:
                    mean_val = channel_subset['Single_mRNA_per_Cell'].mean()
                    sem_val = channel_subset['Single_mRNA_per_Cell'].sem()

                    mouse_data.append({
                        'Mouse_ID': mouse_id,
                        'Mouse_Label': mouse_label,
                        'Model': 'Wildtype',
                        'Age': age,
                        'Channel': channel,
                        'Mean': mean_val,
                        'SEM': sem_val,
                        'N_FOVs': n_fovs
                    })

        df_mouse = pd.DataFrame(mouse_data)

        if len(df_mouse) == 0:
            continue

        # Plot per-mouse bars
        x_pos = 0
        x_ticks = []
        x_labels = []

        # Get unique mouse labels in order
        mouse_labels_ordered = df_mouse['Mouse_Label'].unique()

        for mouse_label in mouse_labels_ordered:
            mouse_subset = df_mouse[df_mouse['Mouse_Label'] == mouse_label]

            if len(mouse_subset) == 0:
                continue

            model = mouse_subset['Model'].iloc[0]

            # HTT1a
            mhtt1a_data = mouse_subset[mouse_subset['Channel'] == 'HTT1a']
            if len(mhtt1a_data) > 0:
                color = COLOR_Q111_MHTT1A if model == 'Q111' else COLOR_WT_MHTT1A
                ax.bar(x_pos, mhtt1a_data['Mean'].iloc[0],
                      yerr=mhtt1a_data['SEM'].iloc[0],
                      color=color, alpha=0.8, width=0.4,
                      edgecolor='black', linewidth=1.5,
                      capsize=3, error_kw={'linewidth': 1.5})
            x_pos += 0.5

            # full-length
            full_data = mouse_subset[mouse_subset['Channel'] == 'fl-HTT']
            if len(full_data) > 0:
                color = COLOR_Q111_FULL if model == 'Q111' else COLOR_WT_FULL
                ax.bar(x_pos, full_data['Mean'].iloc[0],
                      yerr=full_data['SEM'].iloc[0],
                      color=color, alpha=0.8, width=0.4,
                      edgecolor='black', linewidth=1.5,
                      capsize=3, error_kw={'linewidth': 1.5})

            x_ticks.append(x_pos - 0.25)
            x_labels.append(mouse_label)

            x_pos += 1.0

        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=10, fontweight='bold')
        ax.set_ylabel('Single mRNA/nucleus', fontsize=14, fontweight='bold')
        ax.set_title(f'{chr(69 + col_idx)}) {region}: Per-Slide (all FOVs concatenated)',
                    fontsize=15, fontweight='bold', pad=10)
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)

    # ══════════════════════════════════════════════════════════════════════
    # GENERATE COMPREHENSIVE CAPTION
    # ══════════════════════════════════════════════════════════════════════

    caption_lines = []
    caption_lines.append("=" * 80)
    caption_lines.append("FIGURE: Comprehensive Single mRNA Expression Analysis")
    caption_lines.append("=" * 80)
    caption_lines.append("")

    # Overall description
    caption_lines.append("OVERVIEW:")
    caption_lines.append("-" * 80)
    caption_lines.append("This figure presents a comprehensive analysis of single mRNA expression in Q111")
    caption_lines.append("transgenic mice compared to wildtype controls, examining both HTT1a (intron-1")
    caption_lines.append("terminated, exon 1 only) and fl-HTT transcripts across cortical and")
    caption_lines.append("striatal brain regions. Single mRNA represents individual diffraction-limited")
    caption_lines.append("spots, quantified as: N_spots / N_nuclei per nucleus.")
    caption_lines.append("")

    # Data statistics
    caption_lines.append("DATASET STATISTICS:")
    caption_lines.append("-" * 80)

    total_fovs = len(df_fov)
    q111_fovs = len(df_fov[df_fov['Mouse_Model'] == 'Q111'])
    wt_fovs = len(df_fov[df_fov['Mouse_Model'] == 'Wildtype'])

    caption_lines.append(f"Total field-of-views (FOVs): {total_fovs}")
    caption_lines.append(f"  Q111 mice: {q111_fovs} FOVs ({100*q111_fovs/total_fovs:.1f}%)")
    caption_lines.append(f"  Wildtype mice: {wt_fovs} FOVs ({100*wt_fovs/total_fovs:.1f}%)")
    caption_lines.append("")

    # Per-region breakdown
    for region in ['Cortex', 'Striatum']:
        region_fovs = len(df_fov[df_fov['Region'] == region])
        q111_region = len(df_fov[(df_fov['Region'] == region) & (df_fov['Mouse_Model'] == 'Q111')])
        wt_region = len(df_fov[(df_fov['Region'] == region) & (df_fov['Mouse_Model'] == 'Wildtype')])

        caption_lines.append(f"{region}: {region_fovs} FOVs total")
        caption_lines.append(f"  Q111: {q111_region} FOVs, Wildtype: {wt_region} FOVs")
    caption_lines.append("")

    # Mouse IDs
    q111_mice = sorted(df_fov[df_fov['Mouse_Model'] == 'Q111']['Mouse_ID'].unique())
    wt_mice = sorted(df_fov[df_fov['Mouse_Model'] == 'Wildtype']['Mouse_ID'].unique())

    caption_lines.append(f"Q111 mice (n={len(q111_mice)}): {', '.join(q111_mice)}")
    caption_lines.append(f"Wildtype mice (n={len(wt_mice)}): {', '.join(wt_mice)}")
    caption_lines.append("")

    # Age distribution
    ages = sorted(df_fov['Age'].unique())
    caption_lines.append(f"Age timepoints: {', '.join([f'{age:.1f} months' for age in ages])}")
    for age in ages:
        age_subset = df_fov[df_fov['Age'] == age]
        q111_count = len(age_subset[age_subset['Mouse_Model'] == 'Q111'])
        wt_count = len(age_subset[age_subset['Mouse_Model'] == 'Wildtype'])
        caption_lines.append(f"  {age:.1f} mo: Q111={q111_count} FOVs, WT={wt_count} FOVs")
    caption_lines.append("")

    # Quality control (dynamic from data)
    caption_lines.append("QUALITY CONTROL:")
    caption_lines.append("-" * 80)
    n_unique_slides = df_fov['Slide'].nunique()
    caption_lines.append(f"Total slides analyzed: {n_unique_slides}")
    caption_lines.append("Slides with technical failures were excluded based on UBC positive control analysis.")
    caption_lines.append("(Slides with UBC expression <1 mRNA/nucleus were excluded)")
    caption_lines.append("")

    # Panel descriptions
    caption_lines.append("PANEL DESCRIPTIONS:")
    caption_lines.append("-" * 80)
    caption_lines.append("")
    caption_lines.append("ROW 1 - AGE TRENDS (Panels A-B):")
    caption_lines.append("Violin plots showing the distribution of single mRNA expression across ages.")
    caption_lines.append("Each violin represents the full distribution of FOV-level measurements at that age.")

    for region in ['Cortex', 'Striatum']:
        caption_lines.append(f"\n{region} (Panel {'A' if region == 'Cortex' else 'B'}):")
        for age in ages:
            for channel in ['HTT1a', 'fl-HTT']:
                for model in ['Q111', 'Wildtype']:
                    subset = df_fov[
                        (df_fov['Region'] == region) &
                        (df_fov['Age'] == age) &
                        (df_fov['Channel'] == channel) &
                        (df_fov['Mouse_Model'] == model)
                    ]['Single_mRNA_per_Cell']

                    if len(subset) > 0:
                        mean_val = subset.mean()
                        std_val = subset.std()
                        median_val = subset.median()
                        n_val = len(subset)
                        caption_lines.append(
                            f"  {age:.1f}mo {model} {channel}: "
                            f"n={n_val}, mean={mean_val:.2f}±{std_val:.2f}, median={median_val:.2f}"
                        )

    # Add statistical test results
    caption_lines.append("")
    caption_lines.append("STATISTICAL TESTS (Q111 vs Wildtype, Age Trends):")
    caption_lines.append("-" * 80)
    caption_lines.append("Independent t-tests comparing Q111 and Wildtype at each age/channel/region.")
    caption_lines.append("Significance: *** p<0.001, ** p<0.01, * p<0.05, ns p≥0.05")
    caption_lines.append("")

    for test in age_trend_tests:
        sig_marker = ''
        if test['p_value'] < 0.001:
            sig_marker = '***'
        elif test['p_value'] < 0.01:
            sig_marker = '**'
        elif test['p_value'] < 0.05:
            sig_marker = '*'
        else:
            sig_marker = 'ns'

        caption_lines.append(
            f"{test['region']} - {test['age']:.1f}mo - {test['channel']}: "
            f"t={test['t_stat']:.3f}, p={test['p_value']:.4f} {sig_marker} "
            f"(Q111 n={test['n_q111']}, WT n={test['n_wt']})"
        )

    caption_lines.append("")
    caption_lines.append("ROW 2 - ATLAS COORDINATE TRENDS (Panels C-D):")
    caption_lines.append("Box plots showing single mRNA expression across anterior-posterior brain atlas")
    caption_lines.append("coordinates (25μm units from Bregma). Each box represents the interquartile range")
    caption_lines.append("(IQR), whiskers extend to 1.5×IQR, and colored diamonds indicate mean values.")
    caption_lines.append("Individual coordinates are shown without binning for maximum spatial resolution.")

    for region in ['Cortex', 'Striatum']:
        caption_lines.append(f"\n{region} (Panel {'C' if region == 'Cortex' else 'D'}):")
        coords = sorted(df_fov[df_fov['Region'] == region]['Brain_Atlas_Coord'].unique())
        caption_lines.append(f"  Atlas coordinates (n={len(coords)}): {min(coords):.0f} to {max(coords):.0f} (25μm units)")

        # Summary statistics per model/channel
        for model in ['Q111', 'Wildtype']:
            for channel in ['HTT1a', 'fl-HTT']:
                subset = df_fov[
                    (df_fov['Region'] == region) &
                    (df_fov['Channel'] == channel) &
                    (df_fov['Mouse_Model'] == model)
                ]['Single_mRNA_per_Cell']

                if len(subset) > 0:
                    caption_lines.append(
                        f"  {model} {channel}: n={len(subset)} FOVs across {len(coords)} coordinates, "
                        f"mean={subset.mean():.2f}±{subset.std():.2f}"
                    )

    caption_lines.append("")
    caption_lines.append("ROW 3 - PER-SLIDE BREAKDOWN (Panels E-F):")
    caption_lines.append("Bar plots showing mean±SEM single mRNA expression for each individual slide,")
    caption_lines.append("with ALL FOVs from that slide CONCATENATED into a single value.")
    caption_lines.append("X-axis labels use mouse identifiers:")
    caption_lines.append("  - Q#X.Y = Q111 mouse #X, slide Y (e.g., Q#3.2 = Q111 mouse 3, slide 2)")
    caption_lines.append("  - W#X.Y = Wildtype mouse #X, slide Y")
    caption_lines.append("This reveals inter-individual variability within each genotype.")
    caption_lines.append("")
    caption_lines.append("IMPORTANT: DATA AGGREGATION NOTE:")
    caption_lines.append("- Rows 1-2 (Age/Atlas): Each violin/box represents all FOVs pooled for that condition")
    caption_lines.append("- Row 3 (Per-Slide): Each bar represents all FOVs concatenated for one slide")

    # Build labels for caption listing
    q111_labels_caption = build_slide_sublabels(df_fov[df_fov['Mouse_Model'] == 'Q111']['Mouse_ID'].unique(), 'Q')
    wt_labels_caption = build_slide_sublabels(df_fov[df_fov['Mouse_Model'] == 'Wildtype']['Mouse_ID'].unique(), 'W')

    for region in ['Cortex', 'Striatum']:
        caption_lines.append(f"\n{region} (Panel {'E' if region == 'Cortex' else 'F'}):")

        # Q111 mice first (sorted by age, then original mouse number)
        for mouse_id in sorted(df_fov[(df_fov['Region'] == region) & (df_fov['Mouse_Model'] == 'Q111')]['Mouse_ID'].unique(),
                               key=lambda x: (extract_mouse_info(x), x)):
            subset = df_fov[(df_fov['Region'] == region) & (df_fov['Mouse_ID'] == mouse_id)]
            if len(subset) > 0:
                age = subset['Age'].iloc[0]
                mouse_label = q111_labels_caption.get(mouse_id, mouse_id)
                n_fovs = len(subset) // 2
                caption_lines.append(f"  {mouse_label} ({age:.0f}mo): {n_fovs} FOVs concatenated")

        # WT mice (sorted by age, then original mouse number)
        for mouse_id in sorted(df_fov[(df_fov['Region'] == region) & (df_fov['Mouse_Model'] == 'Wildtype')]['Mouse_ID'].unique(),
                               key=lambda x: (extract_mouse_info(x), x)):
            subset = df_fov[(df_fov['Region'] == region) & (df_fov['Mouse_ID'] == mouse_id)]
            if len(subset) > 0:
                age = subset['Age'].iloc[0]
                mouse_label = wt_labels_caption.get(mouse_id, mouse_id)
                n_fovs = len(subset) // 2
                caption_lines.append(f"  {mouse_label} ({age:.0f}mo): {n_fovs} FOVs concatenated")

    caption_lines.append("")
    caption_lines.append("COLOR SCHEME:")
    caption_lines.append("-" * 80)
    caption_lines.append("Q111 mice:")
    caption_lines.append("  - Green: HTT1a (intron-1 terminated, exon 1 only)")
    caption_lines.append("  - Orange: fl-HTT")
    caption_lines.append("Wildtype mice:")
    caption_lines.append("  - Blue: HTT1a")
    caption_lines.append("  - Purple: fl-HTT")
    caption_lines.append("")

    caption_lines.append("METHODOLOGY:")
    caption_lines.append("-" * 80)
    caption_lines.append("mRNA quantification:")
    caption_lines.append("  - Single mRNA: Individual diffraction-limited spots (N_spots)")
    caption_lines.append("  - Clustered mRNA: Aggregated signal normalized by single-spot peak intensity")
    caption_lines.append("  - Total mRNA/nucleus = N_spots + (I_cluster_total / I_single_peak) / N_nuclei")
    caption_lines.append("  - Single mRNA/nucleus = N_spots / N_nuclei")
    caption_lines.append("")
    caption_lines.append("Normalization:")
    caption_lines.append("  - Slide-specific peak intensity determined via Kernel Density Estimation (KDE)")
    caption_lines.append("  - Accounts for slide-to-slide variation in signal intensity")
    caption_lines.append("")
    caption_lines.append("Field-of-view (FOV) filtering:")
    caption_lines.append("  - Minimum 40 nuclei per FOV (ensures adequate DAPI segmentation)")
    caption_lines.append("  - Excludes FOVs with poor tissue quality or imaging artifacts")
    caption_lines.append("")

    caption_lines.append("STATISTICAL NOTES:")
    caption_lines.append("-" * 80)
    caption_lines.append("All data points represent individual FOVs (biological replicates).")
    caption_lines.append("Error bars in panel rows E-F represent SEM (standard error of the mean).")
    caption_lines.append("Box plots show median (removed for clarity), mean (diamond), IQR (box),")
    caption_lines.append("and 1.5×IQR whiskers with outliers as individual points.")
    caption_lines.append("")

    caption_lines.append("=" * 80)
    caption_lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    caption_lines.append("=" * 80)

    # Save caption
    caption_path = OUTPUT_DIR / "figure_caption.txt"
    with open(caption_path, 'w') as f:
        f.write('\n'.join(caption_lines))
    print(f"  Saved caption: {caption_path}")

    # Save figure
    for fmt in ['png', 'svg', 'pdf']:
        output_path = OUTPUT_DIR / f"fig_single_mrna_comprehensive.{fmt}"
        fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"  Saved: {output_path}")

    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("="*70)
    print("COMPREHENSIVE SINGLE mRNA EXPRESSION FIGURE")
    print("="*70)

    # Load FOV-level data (generated by fig_expression_analysis_q111.py)
    fov_data_path = Path(__file__).parent / "output" / "expression_analysis_q111" / "fov_level_data.csv"

    if not fov_data_path.exists():
        print(f"\nERROR: FOV data not found at {fov_data_path}")
        print("Please run fig_expression_analysis_q111.py first.")
        exit(1)

    print(f"\nLoading FOV data from: {fov_data_path}")
    df_fov = pd.read_csv(fov_data_path)
    print(f"Loaded {len(df_fov)} FOV records")
    print(f"Mouse models: {df_fov['Mouse_Model'].unique()}")
    print(f"Counts: {dict(df_fov['Mouse_Model'].value_counts())}")

    # Create comprehensive figure
    create_comprehensive_figure(df_fov)

    print("\n" + "="*70)
    print("FIGURE GENERATION COMPLETE!")
    print("="*70)
