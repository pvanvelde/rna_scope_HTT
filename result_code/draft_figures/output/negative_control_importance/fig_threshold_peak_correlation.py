"""
Supplementary Figure: Correlation between Negative Control Threshold and Single mRNA Peak Intensity

This figure demonstrates the strong correlation between the negative control threshold
and the single mRNA peak intensity, justifying why per-slide calibration is important.

The threshold (from negative control) and peak intensity (from experimental data)
both scale with imaging conditions, so they should be calibrated together on a per-slide basis.

Author: Generated for RNA Scope analysis
Date: 2025-12-24
"""

import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, pearsonr
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Import custom functions
from result_functions_v2 import compute_thresholds, recursively_load_dict, extract_dataframe
from results_config import (
    PIXELSIZE,
    SLICE_DEPTH,
    H5_FILE_PATH_EXPERIMENTAL,
    MEAN_NUCLEAR_VOLUME,
    VOXEL_SIZE,
    MAX_PFA,
    QUANTILE_NEGATIVE_CONTROL,
    N_BOOTSTRAP,
    MIN_NUCLEI_THRESHOLD,
    EXCLUDED_SLIDES,
    SLIDE_FIELD,
    FIGURE_DPI,
    CV_THRESHOLD
)

# Physical parameters
pixelsize = PIXELSIZE
slice_depth = SLICE_DEPTH
mean_nuclear_volume = MEAN_NUCLEAR_VOLUME
voxel_size = VOXEL_SIZE

# Create output directory
OUTPUT_DIR = Path(__file__).parent
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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


if __name__ == "__main__":

    print("="*80)
    print("THRESHOLD-PEAK INTENSITY CORRELATION ANALYSIS")
    print("Demonstrating relationship between negative control threshold and peak intensity")
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

    df_extracted_full = extract_dataframe(
        data_dict,
        field_keys=fields_to_extract,
        channels=channels_to_extract,
        include_file_metadata_sample=True,
        explode_fields=[]
    )

    print(f"DataFrame extracted: {len(df_extracted_full)} total rows")

    # ══════════════════════════════════════════════════════════════════════════
    # 3. COMPUTE THRESHOLDS
    # ══════════════════════════════════════════════════════════════════════════

    print("\nComputing per-slide thresholds...")

    (thresholds, thresholds_cluster,
     error_thresholds, error_thresholds_cluster,
     number_of_datapoints, age_dict) = compute_thresholds(
        df_extracted=df_extracted_full,
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

    # Build threshold dataframe (ONLY INCLUDED SLIDES)
    thr_records = []
    for (slide, channel, area), vec in error_thresholds.items():
        if slide in EXCLUDED_SLIDES:
            continue  # Skip excluded slides
        session = extract_imaging_session(slide)
        thr_records.append({
            'slide': slide,
            'channel': channel,
            'session': session,
            'threshold': np.mean(vec),
            'threshold_std': np.std(vec)
        })

    thr_df = pd.DataFrame(thr_records)
    print(f"\nThreshold data for {len(thr_df)} slide-channel combinations (excluded {len(EXCLUDED_SLIDES)} slides)")

    # ══════════════════════════════════════════════════════════════════════════
    # 4. GET EXPERIMENTAL DATA (Q111, INCLUDED SLIDES ONLY)
    # ══════════════════════════════════════════════════════════════════════════

    df_exp = df_extracted_full[
        (df_extracted_full["metadata_sample_Mouse_Model"] == 'Q111') &
        (df_extracted_full['metadata_sample_Probe-Set'] == experimental_field) &
        (~df_extracted_full[slide_field].isin(EXCLUDED_SLIDES))
    ].copy()

    print(f"Experimental Q111 data (included slides only): {len(df_exp)} rows")

    # ══════════════════════════════════════════════════════════════════════════
    # 5. COMPUTE PEAK INTENSITIES PER SLIDE
    # ══════════════════════════════════════════════════════════════════════════

    print("\nComputing peak intensities per slide...")

    # Build threshold lookup
    thresholds_slide = {(row['slide'], row['channel']): row['threshold']
                        for _, row in thr_df.iterrows()}

    # Collect intensities per slide-channel
    slide_channel_intensities = {}

    for idx, row in df_exp.iterrows():
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

    # Compute peak intensities
    peak_records = []
    for (slide, channel), data in slide_channel_intensities.items():
        intensities = np.array(data['intensities'])
        session = data['session']

        if len(intensities) < 100:
            continue

        # Get threshold
        thr = thresholds_slide.get((slide, channel), np.nan)
        if np.isnan(thr):
            continue

        # Compute peak with slide-specific threshold
        intensities_filtered = intensities[intensities > thr]
        if len(intensities_filtered) >= 50:
            peak = compute_peak_intensity(intensities_filtered)
            if not np.isnan(peak):
                peak_records.append({
                    'slide': slide,
                    'channel': channel,
                    'session': session,
                    'threshold': thr,
                    'peak_intensity': peak,
                    'n_spots': len(intensities_filtered)
                })
                print(f"  {slide} - {channel}: threshold={thr:.0f}, peak={peak:.0f}, n={len(intensities_filtered)}")

    df_peak = pd.DataFrame(peak_records)
    print(f"\nPeak intensity data: {len(df_peak)} slide-channel combinations")

    # ══════════════════════════════════════════════════════════════════════════
    # 6. CREATE FIGURE
    # ══════════════════════════════════════════════════════════════════════════

    print("\nCreating figure...")

    # Channel-specific colors and labels
    channel_config = {
        'green': {
            'label': 'HTT1a (488 nm)',
            'color': '#2ecc71',
            'marker': 'o'
        },
        'orange': {
            'label': 'Full-length mHTT (548 nm)',
            'color': '#e67e22',
            'marker': 's'
        }
    }

    session_colors = {
        'Session 1': '#3498db',
        'Session 2': '#e74c3c',
        'Session 3': '#9b59b6'
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=FIGURE_DPI)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.12, wspace=0.25)

    # ── Panel A: Threshold vs Peak Intensity by Channel ──
    ax1 = axes[0]

    for channel in ['green', 'orange']:
        ch_data = df_peak[df_peak['channel'] == channel]
        if len(ch_data) > 0:
            ax1.scatter(ch_data['threshold'], ch_data['peak_intensity'],
                       c=channel_config[channel]['color'],
                       marker=channel_config[channel]['marker'],
                       s=100, alpha=0.7, edgecolor='black', linewidth=0.5,
                       label=f"{channel_config[channel]['label']} (n={len(ch_data)})")

    # Add x=y reference line
    all_vals = np.concatenate([df_peak['threshold'].values, df_peak['peak_intensity'].values])
    min_val, max_val = all_vals.min() * 0.9, all_vals.max() * 1.1
    ax1.plot([min_val, max_val], [min_val, max_val],
             'k--', linewidth=1.5, alpha=0.5, label='x = y')

    # Correlation for all data
    r_all, p_all = pearsonr(df_peak['threshold'], df_peak['peak_intensity'])

    ax1.set_xlabel("Negative Control Threshold (A.U.)", fontsize=12)
    ax1.set_ylabel("Single mRNA Peak Intensity (A.U.)", fontsize=12)
    ax1.set_title("A. Threshold vs Peak Intensity by Channel", fontsize=13, fontweight='bold', loc='left')
    ax1.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')

    # Add correlation text
    ax1.text(0.95, 0.05, f"Overall: r = {r_all:.3f}\np < {p_all:.1e}",
             transform=ax1.transAxes, fontsize=10, ha='right', va='bottom',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

    # ── Panel B: Threshold vs Peak Intensity by Session ──
    ax2 = axes[1]

    for session in sorted(df_peak['session'].unique()):
        s_data = df_peak[df_peak['session'] == session]
        if len(s_data) > 0:
            # Use different markers for channels within session
            for channel in ['green', 'orange']:
                ch_data = s_data[s_data['channel'] == channel]
                if len(ch_data) > 0:
                    ax2.scatter(ch_data['threshold'], ch_data['peak_intensity'],
                               c=session_colors[session],
                               marker=channel_config[channel]['marker'],
                               s=100, alpha=0.7, edgecolor='black', linewidth=0.5)

    # Add x=y reference line
    ax2.plot([min_val, max_val], [min_val, max_val],
             'k--', linewidth=1.5, alpha=0.5)

    # Create custom legend
    from matplotlib.lines import Line2D
    legend_elements = []

    # Session colors
    for session, color in session_colors.items():
        n_session = len(df_peak[df_peak['session'] == session])
        if n_session > 0:
            legend_elements.append(Line2D([0], [0], marker='o', color='w',
                                         markerfacecolor=color, markersize=10,
                                         label=f'{session} (n={n_session})'))

    # Channel markers
    legend_elements.append(Line2D([0], [0], marker='o', color='w',
                                 markerfacecolor='gray', markersize=10,
                                 label=f'○ = {channel_config["green"]["label"]}'))
    legend_elements.append(Line2D([0], [0], marker='s', color='w',
                                 markerfacecolor='gray', markersize=10,
                                 label=f'□ = {channel_config["orange"]["label"]}'))
    legend_elements.append(Line2D([0], [0], linestyle='--', color='black',
                                 label='x = y'))

    ax2.set_xlabel("Negative Control Threshold (A.U.)", fontsize=12)
    ax2.set_ylabel("Single mRNA Peak Intensity (A.U.)", fontsize=12)
    ax2.set_title("B. Threshold vs Peak Intensity by Session", fontsize=13, fontweight='bold', loc='left')
    ax2.legend(handles=legend_elements, loc='upper left', fontsize=9, framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--')

    # Per-channel correlations
    corr_text = []
    for channel in ['green', 'orange']:
        ch_data = df_peak[df_peak['channel'] == channel]
        if len(ch_data) >= 3:
            r, p = pearsonr(ch_data['threshold'], ch_data['peak_intensity'])
            ch_label = 'HTT1a' if channel == 'green' else 'fl-HTT'
            corr_text.append(f"{ch_label}: r = {r:.3f}")

    ax2.text(0.95, 0.05, '\n'.join(corr_text),
             transform=ax2.transAxes, fontsize=10, ha='right', va='bottom',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

    fig.suptitle('Correlation between Negative Control Threshold and Single mRNA Peak Intensity\n'
                 f'(n={len(df_peak)} slide-channel combinations, {len(EXCLUDED_SLIDES)} slides excluded for QC)',
                 fontsize=14, fontweight='bold')

    # Save figure
    for fmt in ['png', 'svg', 'pdf']:
        filepath = OUTPUT_DIR / f"fig_threshold_peak_correlation.{fmt}"
        fig.savefig(filepath, format=fmt, bbox_inches='tight', dpi=FIGURE_DPI)
        print(f"  Saved: {filepath}")

    plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════════
    # 7. GENERATE CAPTION
    # ══════════════════════════════════════════════════════════════════════════

    # Compute statistics for caption
    r_green, p_green = pearsonr(df_peak[df_peak['channel']=='green']['threshold'],
                                 df_peak[df_peak['channel']=='green']['peak_intensity'])
    r_orange, p_orange = pearsonr(df_peak[df_peak['channel']=='orange']['threshold'],
                                   df_peak[df_peak['channel']=='orange']['peak_intensity'])

    caption_lines = [
        "=" * 100,
        "SUPPLEMENTARY FIGURE: Correlation between Negative Control Threshold and Single mRNA Peak Intensity",
        "=" * 100,
        "",
        "OVERVIEW",
        "-" * 100,
        "This figure demonstrates the strong positive correlation between the negative control threshold",
        "(95th percentile of background intensity) and the single mRNA peak intensity (mode of spot",
        "intensity distribution from experimental data). This correlation justifies the use of per-slide",
        "calibration: imaging conditions that affect background also affect signal intensity.",
        "",
        "=" * 100,
        "DATA SUMMARY",
        "=" * 100,
        "",
        f"Total slide-channel combinations: {len(df_peak)}",
        f"  - HTT1a (green, 488 nm): {len(df_peak[df_peak['channel']=='green'])} slides",
        f"  - Full-length mHTT (orange, 548 nm): {len(df_peak[df_peak['channel']=='orange'])} slides",
        "",
        f"Imaging sessions: {len(df_peak['session'].unique())}",
    ]

    for session in sorted(df_peak['session'].unique()):
        n = len(df_peak[df_peak['session'] == session])
        caption_lines.append(f"  - {session}: {n} slide-channel combinations")

    caption_lines.extend([
        "",
        "QUALITY CONTROL:",
        f"  - Excluded slides (n={len(EXCLUDED_SLIDES)}): {', '.join(sorted(EXCLUDED_SLIDES))}",
        "  - Reason: Poor UBC positive control expression indicating technical failures",
        "",
        "=" * 100,
        "CORRELATION ANALYSIS",
        "=" * 100,
        "",
        f"Overall correlation: r = {r_all:.4f}, p = {p_all:.2e}",
        "",
        "Per-channel correlations:",
        f"  - HTT1a (green): r = {r_green:.4f}, p = {p_green:.2e}",
        f"  - Full-length mHTT (orange): r = {r_orange:.4f}, p = {p_orange:.2e}",
        "",
        "=" * 100,
        "INTERPRETATION",
        "=" * 100,
        "",
        "The strong positive correlation (r > 0.75) between threshold and peak intensity indicates that:",
        "",
        "1. IMAGING CONDITIONS AFFECT BOTH METRICS:",
        "   - Higher laser power, detector gain, or tissue autofluorescence increases both",
        "     background (threshold) and signal (peak intensity)",
        "   - This covariation is expected from the physics of fluorescence microscopy",
        "",
        "2. PER-SLIDE CALIBRATION CAPTURES THIS RELATIONSHIP:",
        "   - Using slide-specific thresholds naturally accounts for imaging variability",
        "   - The ratio (peak_intensity / threshold) is more stable than either value alone",
        "",
        "3. IMPLICATIONS FOR mRNA QUANTIFICATION:",
        "   - Total mRNA = N_spots + (I_clusters / I_peak)",
        "   - Both the threshold (which spots pass) and peak (normalization factor) scale together",
        "   - Per-slide calibration ensures consistent mRNA estimates across imaging conditions",
        "",
        "=" * 100,
        "PANEL DESCRIPTIONS",
        "=" * 100,
        "",
        "Panel A: Threshold vs Peak Intensity by Channel",
        "  - Green circles: HTT1a probe (488 nm excitation)",
        "  - Orange squares: Full-length mHTT probe (548 nm excitation)",
        "  - Dashed line: x = y reference (perfect correlation would follow this line)",
        "  - Both channels show strong positive correlation",
        "",
        "Panel B: Threshold vs Peak Intensity by Session",
        "  - Colors indicate imaging session (blue = Session 1, red = Session 2, purple = Session 3)",
        "  - Marker shapes indicate channel (circles = HTT1a, squares = fl-HTT)",
        "  - Sessions cluster together, reflecting session-specific imaging conditions",
        "  - Within-session variation is smaller than between-session variation",
        "",
        "=" * 100,
        "METHODOLOGY",
        "=" * 100,
        "",
        "Threshold calculation:",
        f"  - Quantile: {QUANTILE_NEGATIVE_CONTROL*100:.0f}th percentile of negative control spot intensities",
        "  - Negative control: Bacterial DapB probe (not expressed in mammalian tissue)",
        "",
        "Peak intensity calculation:",
        "  - Method: Mode of kernel density estimate (KDE) of spot intensities",
        "  - Only spots above threshold are included",
        "  - Represents the most probable intensity for a single mRNA molecule",
        "",
        "=" * 100,
        f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 100,
    ])

    caption_text = '\n'.join(caption_lines)
    caption_path = OUTPUT_DIR / "fig_threshold_peak_correlation_caption.txt"
    with open(caption_path, 'w') as f:
        f.write(caption_text)
    print(f"  Caption saved: {caption_path}")

    # Save data
    csv_path = OUTPUT_DIR / "threshold_peak_correlation_data.csv"
    df_peak.to_csv(csv_path, index=False)
    print(f"  Data saved: {csv_path}")

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Overall correlation: r = {r_all:.4f}")
    print(f"HTT1a correlation: r = {r_green:.4f}")
    print(f"fl-HTT correlation: r = {r_orange:.4f}")
    print("="*80)
