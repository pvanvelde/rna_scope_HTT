"""
Supplementary Figure: Importance of Per-Session Negative Control Thresholds

This script demonstrates why per-imaging-session negative control calibration is essential:
1. Shows threshold variability across imaging sessions
2. Quantifies the impact on mRNA/nucleus quantification if using global vs session-specific thresholds
3. Illustrates potential false positive/negative rates with global thresholds

Author: Generated for RNA Scope analysis
Date: 2025-12-23
"""

import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import scienceplots
plt.style.use('science')
plt.rcParams['text.usetex'] = False
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

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
OUTPUT_DIR = Path(__file__).parent / "output" / "negative_control_importance"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_imaging_session(slide_name):
    """
    Extract imaging session from slide name.
    Slide names are like 'm1a2', 'm2b5', 'm3a1' - the digit after 'm' is the session.
    """
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
    print("NEGATIVE CONTROL IMPORTANCE ANALYSIS")
    print("Demonstrating the need for per-session threshold calibration")
    print("="*80)

    # ══════════════════════════════════════════════════════════════════════════
    # 1. LOAD DATA
    # ══════════════════════════════════════════════════════════════════════════

    h5_file_path = H5_FILE_PATH_EXPERIMENTAL

    with h5py.File(h5_file_path, 'r') as h5_file:
        data_dict = recursively_load_dict(h5_file)

    print(f"\n{'='*80}")
    print(f"Loaded data from: {h5_file_path}")
    print(f"{'='*80}")

    # ══════════════════════════════════════════════════════════════════════════
    # 2. EXTRACT DATAFRAME WITH METADATA
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

    print(f"\nDataFrame extracted: {len(df_extracted_full)} total rows")

    # ══════════════════════════════════════════════════════════════════════════
    # 3. COMPUTE PER-SLIDE THRESHOLDS
    # ══════════════════════════════════════════════════════════════════════════

    print(f"\n{'='*80}")
    print("Computing per-slide thresholds...")
    print(f"{'='*80}")

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

    # Build threshold dataframe with session information
    thr_rows = []
    for (slide, channel, area), vec in error_thresholds.items():
        session = extract_imaging_session(slide)
        thr_rows.append({
            "slide": slide,
            "channel": channel,
            "session": session,
            "threshold": np.mean(vec),
            "threshold_std": np.std(vec) if len(vec) > 1 else 0
        })

    thr_df = pd.DataFrame(thr_rows).drop_duplicates(["slide", "channel"])

    # Add session to main dataframe
    df_extracted_full['session'] = df_extracted_full[slide_field].apply(extract_imaging_session)

    print(f"\nThreshold summary by session:")
    for session in sorted(thr_df['session'].unique()):
        session_data = thr_df[thr_df['session'] == session]
        for channel in ['green', 'orange']:
            ch_data = session_data[session_data['channel'] == channel]
            if len(ch_data) > 0:
                print(f"  {session}, {channel}: n={len(ch_data)} slides, "
                      f"mean={ch_data['threshold'].mean():.1f}, "
                      f"std={ch_data['threshold'].std():.1f}, "
                      f"range=[{ch_data['threshold'].min():.1f}, {ch_data['threshold'].max():.1f}]")

    # ══════════════════════════════════════════════════════════════════════════
    # 4. COMPUTE GLOBAL THRESHOLDS (what if we ignored session?)
    # ══════════════════════════════════════════════════════════════════════════

    print(f"\n{'='*80}")
    print("Computing global thresholds (ignoring session)...")
    print(f"{'='*80}")

    global_thresholds = {}
    for channel in ['green', 'orange']:
        ch_data = thr_df[thr_df['channel'] == channel]
        global_thresholds[channel] = {
            'mean': ch_data['threshold'].mean(),
            'median': ch_data['threshold'].median(),
            'std': ch_data['threshold'].std(),
            'cv': ch_data['threshold'].std() / ch_data['threshold'].mean() * 100
        }
        print(f"  {channel}: global mean={global_thresholds[channel]['mean']:.1f}, "
              f"global median={global_thresholds[channel]['median']:.1f}, "
              f"CV={global_thresholds[channel]['cv']:.1f}%")

    # Merge thresholds into main dataframe
    df_extracted_full = df_extracted_full.merge(
        thr_df[['slide', 'channel', 'threshold']],
        how="left",
        left_on=[slide_field, "channel"],
        right_on=["slide", "channel"]
    )
    df_extracted_full.drop(columns=["slide"], inplace=True, errors='ignore')

    # ══════════════════════════════════════════════════════════════════════════
    # 5. EXTRACT NEGATIVE CONTROL INTENSITY DISTRIBUTIONS
    # ══════════════════════════════════════════════════════════════════════════

    print(f"\n{'='*80}")
    print("Extracting negative control intensity distributions...")
    print(f"{'='*80}")

    # Get negative control data (include ALL slides for calibration analysis)
    df_neg = df_extracted_full[
        df_extracted_full['metadata_sample_Probe-Set'] == negative_control_field
    ].copy()

    # NOTE: Include ALL slides for this supplementary figure to show full threshold variability

    print(f"Negative control FOVs: {len(df_neg)}")
    print(f"Sessions: {sorted(df_neg['session'].unique())}")

    # Extract intensity distributions per session
    session_intensities = {ch: {} for ch in ['green', 'orange']}

    for idx, row in df_neg.iterrows():
        channel = row.get('channel', 'unknown')
        session = row.get('session', 'Unknown')

        if channel not in ['green', 'orange']:
            continue

        sigma_var_params = row.get('spots_sigma_var.params_raw', None)
        final_filter = row.get('spots.final_filter', None)

        if sigma_var_params is not None and final_filter is not None:
            try:
                sigma_var_params = np.asarray(sigma_var_params)
                final_filter = np.asarray(final_filter).astype(bool)

                if sigma_var_params.ndim >= 2 and sigma_var_params.shape[1] > 3:
                    if final_filter.sum() > 0:
                        intensities = sigma_var_params[final_filter, 3]

                        if session not in session_intensities[channel]:
                            session_intensities[channel][session] = []
                        session_intensities[channel][session].extend(intensities)
            except:
                pass

    for channel in ['green', 'orange']:
        print(f"\n{channel} channel negative control spots:")
        for session in sorted(session_intensities[channel].keys()):
            intensities = session_intensities[channel][session]
            print(f"  {session}: n={len(intensities):,}, "
                  f"mean={np.mean(intensities):.1f}, "
                  f"median={np.median(intensities):.1f}, "
                  f"95th percentile={np.percentile(intensities, 95):.1f}")

    # ══════════════════════════════════════════════════════════════════════════
    # 6. COMPUTE SESSION MEAN THRESHOLDS
    # ══════════════════════════════════════════════════════════════════════════

    print(f"\n{'='*80}")
    print("Computing session mean thresholds...")
    print(f"{'='*80}")

    # Compute session mean thresholds (what if we used session average instead of slide-specific?)
    session_mean_thresholds = {}
    for channel in ['green', 'orange']:
        session_mean_thresholds[channel] = {}
        for session in thr_df['session'].unique():
            session_data = thr_df[(thr_df['session'] == session) & (thr_df['channel'] == channel)]
            if len(session_data) > 0:
                session_mean_thresholds[channel][session] = session_data['threshold'].mean()
                print(f"  {channel} {session}: mean threshold = {session_mean_thresholds[channel][session]:.1f}")

    # Add session mean threshold to thr_df
    thr_df['session_mean_threshold'] = thr_df.apply(
        lambda row: session_mean_thresholds.get(row['channel'], {}).get(row['session'], np.nan),
        axis=1
    )

    # ══════════════════════════════════════════════════════════════════════════
    # 7. COMPUTE SINGLE mRNA PEAK INTENSITY WITH DIFFERENT THRESHOLDS
    # ══════════════════════════════════════════════════════════════════════════

    print(f"\n{'='*80}")
    print("Computing single mRNA peak intensity with slide-specific vs session-mean thresholds...")
    print(f"{'='*80}")

    # Get experimental Q111 data (include ALL slides for calibration analysis)
    df_exp = df_extracted_full[
        (df_extracted_full["metadata_sample_Mouse_Model"] == 'Q111') &
        (df_extracted_full['metadata_sample_Probe-Set'] == experimental_field)
    ].copy()

    # NOTE: Include ALL slides for this supplementary figure to demonstrate full variability

    print(f"Experimental FOVs: {len(df_exp)}")

    # Collect all spot intensities per slide for peak intensity calculation
    slide_spot_data = {}  # {(slide, channel): list of (threshold, session, intensities)}

    for idx, row in df_exp.iterrows():
        slide = row.get(slide_field, 'unknown')
        channel = row.get('channel', 'unknown')
        session = row.get('session', 'Unknown')

        if channel == 'blue':
            continue

        key = (slide, channel)
        if key not in slide_spot_data:
            slide_spot_data[key] = {
                'session': session,
                'threshold_slide': row.get('threshold', np.nan),
                'intensities': []
            }

        sigma_var_params = row.get('spots_sigma_var.params_raw', None)
        final_filter = row.get('spots.final_filter', None)

        if sigma_var_params is not None and final_filter is not None:
            try:
                sigma_var_params = np.asarray(sigma_var_params)
                final_filter = np.asarray(final_filter).astype(bool)
                if sigma_var_params.ndim >= 2 and sigma_var_params.shape[1] > 3:
                    photons = sigma_var_params[final_filter, 3]
                    slide_spot_data[key]['intensities'].extend(photons)
            except:
                pass

    # Compute peak intensities with both threshold methods
    peak_comparison = []

    for (slide, channel), data in slide_spot_data.items():
        session = data['session']
        threshold_slide = data['threshold_slide']
        intensities = np.array(data['intensities'])

        if len(intensities) < 100 or np.isnan(threshold_slide):
            continue

        # Get session mean threshold
        threshold_session_mean = session_mean_thresholds.get(channel, {}).get(session, np.nan)
        if np.isnan(threshold_session_mean):
            continue

        # Compute peak intensity with slide-specific threshold
        intensities_slide_thr = intensities[intensities > threshold_slide]
        if len(intensities_slide_thr) >= 50:
            peak_slide = compute_peak_intensity(intensities_slide_thr)
        else:
            peak_slide = np.nan

        # Compute peak intensity with session mean threshold
        intensities_session_thr = intensities[intensities > threshold_session_mean]
        if len(intensities_session_thr) >= 50:
            peak_session_mean = compute_peak_intensity(intensities_session_thr)
        else:
            peak_session_mean = np.nan

        if np.isnan(peak_slide) or np.isnan(peak_session_mean):
            continue

        peak_comparison.append({
            'slide': slide,
            'channel': channel,
            'session': session,
            'threshold_slide': threshold_slide,
            'threshold_session_mean': threshold_session_mean,
            'threshold_diff': threshold_slide - threshold_session_mean,
            'threshold_diff_pct': (threshold_slide - threshold_session_mean) / threshold_session_mean * 100,
            'n_spots_slide_thr': len(intensities_slide_thr),
            'n_spots_session_thr': len(intensities_session_thr),
            'peak_slide_specific': peak_slide,
            'peak_session_mean': peak_session_mean,
            'peak_diff': peak_session_mean - peak_slide,
            'peak_diff_pct': (peak_session_mean - peak_slide) / peak_slide * 100,
        })

    df_peak_comparison = pd.DataFrame(peak_comparison)
    print(f"\nComputed peak intensity comparison for {len(df_peak_comparison)} slides")

    # Summary statistics
    print("\nImpact of using session-mean vs slide-specific thresholds on single mRNA peak intensity:")
    for channel in ['green', 'orange']:
        ch_data = df_peak_comparison[df_peak_comparison['channel'] == channel]
        if len(ch_data) == 0:
            continue
        print(f"\n{channel} channel:")
        print(f"  Mean threshold difference (slide - session_mean): {ch_data['threshold_diff'].mean():.1f} "
              f"({ch_data['threshold_diff_pct'].mean():+.1f}%)")
        print(f"  Mean peak intensity (slide-specific): {ch_data['peak_slide_specific'].mean():.1f}")
        print(f"  Mean peak intensity (session-mean): {ch_data['peak_session_mean'].mean():.1f}")
        print(f"  Mean peak difference: {ch_data['peak_diff'].mean():.1f} "
              f"({ch_data['peak_diff_pct'].mean():+.1f}%)")

        # Per-session breakdown
        for session in sorted(ch_data['session'].unique()):
            s_data = ch_data[ch_data['session'] == session]
            print(f"    {session}: threshold diff={s_data['threshold_diff'].mean():+.1f}, "
                  f"peak diff={s_data['peak_diff_pct'].mean():+.1f}%")

    # ══════════════════════════════════════════════════════════════════════════
    # 7. CREATE COMPREHENSIVE FIGURE
    # ══════════════════════════════════════════════════════════════════════════

    print(f"\n{'='*80}")
    print("Creating comprehensive figure...")
    print(f"{'='*80}")

    fig = plt.figure(figsize=(16, 18), dpi=FIGURE_DPI)
    gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.30,
                         left=0.08, right=0.98, top=0.95, bottom=0.05)

    session_colors = {'Session 1': '#3498db', 'Session 2': '#e74c3c', 'Session 3': '#2ecc71'}
    channel_labels = {'green': 'HTT1a (488 nm)', 'orange': 'fl-HTT (548 nm)'}

    # ══════════════════════════════════════════════════════════════════════════
    # ROW 1: THRESHOLD VARIABILITY
    # ══════════════════════════════════════════════════════════════════════════

    # Panel A: Threshold distributions by session (green channel)
    ax_a = fig.add_subplot(gs[0, 0])

    for session in sorted(thr_df['session'].unique()):
        session_data = thr_df[(thr_df['session'] == session) & (thr_df['channel'] == 'green')]
        if len(session_data) > 0:
            thresholds = session_data['threshold'].values
            # Jitter x positions
            x = np.random.normal(list(session_colors.keys()).index(session), 0.1, len(thresholds))
            ax_a.scatter(x, thresholds, c=session_colors[session], s=80, alpha=0.7,
                        edgecolor='black', linewidth=0.5, label=f"{session} (n={len(thresholds)})")

    # Add global threshold line
    ax_a.axhline(global_thresholds['green']['mean'], color='black', linestyle='--',
                 linewidth=2, label=f"Global mean: {global_thresholds['green']['mean']:.1f}")

    ax_a.set_xticks(range(len(session_colors)))
    ax_a.set_xticklabels(session_colors.keys())
    ax_a.set_ylabel("Threshold (photons)", fontsize=11)
    ax_a.set_title(f"A. Threshold Variability - {channel_labels['green']}\n"
                   f"CV = {global_thresholds['green']['cv']:.1f}%",
                   fontsize=12, fontweight='bold', loc='left')
    ax_a.legend(loc='upper right', fontsize=8)
    ax_a.grid(True, alpha=0.3, linestyle='--', axis='y')

    # Panel B: Threshold distributions by session (orange channel)
    ax_b = fig.add_subplot(gs[0, 1])

    for session in sorted(thr_df['session'].unique()):
        session_data = thr_df[(thr_df['session'] == session) & (thr_df['channel'] == 'orange')]
        if len(session_data) > 0:
            thresholds = session_data['threshold'].values
            x = np.random.normal(list(session_colors.keys()).index(session), 0.1, len(thresholds))
            ax_b.scatter(x, thresholds, c=session_colors[session], s=80, alpha=0.7,
                        edgecolor='black', linewidth=0.5, label=f"{session} (n={len(thresholds)})")

    ax_b.axhline(global_thresholds['orange']['mean'], color='black', linestyle='--',
                 linewidth=2, label=f"Global mean: {global_thresholds['orange']['mean']:.1f}")

    ax_b.set_xticks(range(len(session_colors)))
    ax_b.set_xticklabels(session_colors.keys())
    ax_b.set_ylabel("Threshold (photons)", fontsize=11)
    ax_b.set_title(f"B. Threshold Variability - {channel_labels['orange']}\n"
                   f"CV = {global_thresholds['orange']['cv']:.1f}%",
                   fontsize=12, fontweight='bold', loc='left')
    ax_b.legend(loc='upper right', fontsize=8)
    ax_b.grid(True, alpha=0.3, linestyle='--', axis='y')

    # ══════════════════════════════════════════════════════════════════════════
    # ROW 2: NEGATIVE CONTROL INTENSITY DISTRIBUTIONS
    # ══════════════════════════════════════════════════════════════════════════

    # Panel C: Negative control intensity distributions (green)
    ax_c = fig.add_subplot(gs[1, 0])

    for session in sorted(session_intensities['green'].keys()):
        intensities = np.array(session_intensities['green'][session])
        if len(intensities) > 100:
            # KDE for smooth distribution
            kde = gaussian_kde(intensities[intensities < np.percentile(intensities, 99)])
            x_range = np.linspace(0, np.percentile(intensities, 99), 500)
            ax_c.plot(x_range, kde(x_range), color=session_colors[session],
                     linewidth=2, label=f"{session} (n={len(intensities):,})")

            # Add threshold markers
            session_thr = thr_df[(thr_df['session'] == session) & (thr_df['channel'] == 'green')]['threshold'].mean()
            ax_c.axvline(session_thr, color=session_colors[session], linestyle='--', alpha=0.7)

    ax_c.axvline(global_thresholds['green']['mean'], color='black', linestyle='-',
                 linewidth=2, alpha=0.8, label='Global threshold')

    ax_c.set_xlabel("Spot Intensity (photons)", fontsize=11)
    ax_c.set_ylabel("Probability Density", fontsize=11)
    ax_c.set_title(f"C. Negative Control Distributions - {channel_labels['green']}",
                   fontsize=12, fontweight='bold', loc='left')
    ax_c.legend(loc='upper right', fontsize=8)
    ax_c.grid(True, alpha=0.3, linestyle='--')
    ax_c.set_xlim(0, None)

    # Panel D: Negative control intensity distributions (orange)
    ax_d = fig.add_subplot(gs[1, 1])

    for session in sorted(session_intensities['orange'].keys()):
        intensities = np.array(session_intensities['orange'][session])
        if len(intensities) > 100:
            kde = gaussian_kde(intensities[intensities < np.percentile(intensities, 99)])
            x_range = np.linspace(0, np.percentile(intensities, 99), 500)
            ax_d.plot(x_range, kde(x_range), color=session_colors[session],
                     linewidth=2, label=f"{session} (n={len(intensities):,})")

            session_thr = thr_df[(thr_df['session'] == session) & (thr_df['channel'] == 'orange')]['threshold'].mean()
            ax_d.axvline(session_thr, color=session_colors[session], linestyle='--', alpha=0.7)

    ax_d.axvline(global_thresholds['orange']['mean'], color='black', linestyle='-',
                 linewidth=2, alpha=0.8, label='Global threshold')

    ax_d.set_xlabel("Spot Intensity (photons)", fontsize=11)
    ax_d.set_ylabel("Probability Density", fontsize=11)
    ax_d.set_title(f"D. Negative Control Distributions - {channel_labels['orange']}",
                   fontsize=12, fontweight='bold', loc='left')
    ax_d.legend(loc='upper right', fontsize=8)
    ax_d.grid(True, alpha=0.3, linestyle='--')
    ax_d.set_xlim(0, None)

    # ══════════════════════════════════════════════════════════════════════════
    # ROW 3: IMPACT ON SINGLE mRNA PEAK INTENSITY
    # ══════════════════════════════════════════════════════════════════════════

    # Panel E: Peak intensity comparison (green channel)
    ax_e = fig.add_subplot(gs[2, 0])

    ch_data = df_peak_comparison[df_peak_comparison['channel'] == 'green']

    if len(ch_data) > 0:
        for session in sorted(ch_data['session'].unique()):
            s_data = ch_data[ch_data['session'] == session]
            ax_e.scatter(s_data['peak_slide_specific'], s_data['peak_session_mean'],
                        c=session_colors[session], s=80, alpha=0.7,
                        edgecolor='black', linewidth=0.5, label=f"{session} (n={len(s_data)})")

        # Add diagonal (perfect agreement)
        max_val = max(ch_data['peak_slide_specific'].max(), ch_data['peak_session_mean'].max())
        min_val = min(ch_data['peak_slide_specific'].min(), ch_data['peak_session_mean'].min())
        ax_e.plot([min_val*0.9, max_val*1.1], [min_val*0.9, max_val*1.1], 'k--',
                  linewidth=1.5, alpha=0.5, label='Perfect agreement')

        # Calculate and show bias
        mean_bias = ch_data['peak_diff_pct'].mean()
        std_bias = ch_data['peak_diff_pct'].std()
        ax_e.text(0.05, 0.95, f"Mean bias: {mean_bias:+.1f}% ± {std_bias:.1f}%\n(session_mean - slide_specific)",
                 transform=ax_e.transAxes, fontsize=9, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax_e.set_xlabel("Peak intensity (slide-specific threshold)", fontsize=11)
    ax_e.set_ylabel("Peak intensity (session-mean threshold)", fontsize=11)
    ax_e.set_title(f"E. Single mRNA Peak Intensity - {channel_labels['green']}",
                   fontsize=12, fontweight='bold', loc='left')
    ax_e.legend(loc='lower right', fontsize=8)
    ax_e.grid(True, alpha=0.3, linestyle='--')

    # Panel F: Peak intensity comparison (orange channel)
    ax_f = fig.add_subplot(gs[2, 1])

    ch_data = df_peak_comparison[df_peak_comparison['channel'] == 'orange']

    if len(ch_data) > 0:
        for session in sorted(ch_data['session'].unique()):
            s_data = ch_data[ch_data['session'] == session]
            ax_f.scatter(s_data['peak_slide_specific'], s_data['peak_session_mean'],
                        c=session_colors[session], s=80, alpha=0.7,
                        edgecolor='black', linewidth=0.5, label=f"{session} (n={len(s_data)})")

        max_val = max(ch_data['peak_slide_specific'].max(), ch_data['peak_session_mean'].max())
        min_val = min(ch_data['peak_slide_specific'].min(), ch_data['peak_session_mean'].min())
        ax_f.plot([min_val*0.9, max_val*1.1], [min_val*0.9, max_val*1.1], 'k--',
                  linewidth=1.5, alpha=0.5, label='Perfect agreement')

        mean_bias = ch_data['peak_diff_pct'].mean()
        std_bias = ch_data['peak_diff_pct'].std()
        ax_f.text(0.05, 0.95, f"Mean bias: {mean_bias:+.1f}% ± {std_bias:.1f}%\n(session_mean - slide_specific)",
                 transform=ax_f.transAxes, fontsize=9, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax_f.set_xlabel("Peak intensity (slide-specific threshold)", fontsize=11)
    ax_f.set_ylabel("Peak intensity (session-mean threshold)", fontsize=11)
    ax_f.set_title(f"F. Single mRNA Peak Intensity - {channel_labels['orange']}",
                   fontsize=12, fontweight='bold', loc='left')
    ax_f.legend(loc='lower right', fontsize=8)
    ax_f.grid(True, alpha=0.3, linestyle='--')

    # ══════════════════════════════════════════════════════════════════════════
    # ROW 4: THRESHOLD VS PEAK INTENSITY RELATIONSHIP
    # ══════════════════════════════════════════════════════════════════════════

    # Panel G: Threshold vs Peak intensity (green channel)
    ax_g = fig.add_subplot(gs[3, 0])

    ch_data = df_peak_comparison[df_peak_comparison['channel'] == 'green']

    if len(ch_data) > 0:
        for session in sorted(ch_data['session'].unique()):
            s_data = ch_data[ch_data['session'] == session]
            ax_g.scatter(s_data['threshold_slide'], s_data['peak_slide_specific'],
                        c=session_colors[session], s=80, alpha=0.7,
                        edgecolor='black', linewidth=0.5, label=f"{session} (n={len(s_data)})")

        # Add x=y line
        from scipy.stats import pearsonr
        min_val = min(ch_data['threshold_slide'].min(), ch_data['peak_slide_specific'].min())
        max_val = max(ch_data['threshold_slide'].max(), ch_data['peak_slide_specific'].max())
        ax_g.plot([min_val*0.9, max_val*1.1], [min_val*0.9, max_val*1.1],
                  color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='x = y')

        # Correlation stats
        r, p = pearsonr(ch_data['threshold_slide'], ch_data['peak_slide_specific'])
        ax_g.text(0.05, 0.95, f"r = {r:.3f}\np = {p:.2e}",
                 transform=ax_g.transAxes, fontsize=9, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax_g.set_xlabel("Threshold (photons)", fontsize=11)
    ax_g.set_ylabel("Peak intensity (photons)", fontsize=11)
    ax_g.set_title(f"G. Threshold vs Peak Intensity - {channel_labels['green']}",
                   fontsize=12, fontweight='bold', loc='left')
    ax_g.legend(loc='lower right', fontsize=8)
    ax_g.grid(True, alpha=0.3, linestyle='--')

    # Panel H: Threshold vs Peak intensity (orange channel)
    ax_h = fig.add_subplot(gs[3, 1])

    ch_data = df_peak_comparison[df_peak_comparison['channel'] == 'orange']

    if len(ch_data) > 0:
        for session in sorted(ch_data['session'].unique()):
            s_data = ch_data[ch_data['session'] == session]
            ax_h.scatter(s_data['threshold_slide'], s_data['peak_slide_specific'],
                        c=session_colors[session], s=80, alpha=0.7,
                        edgecolor='black', linewidth=0.5, label=f"{session} (n={len(s_data)})")

        # Add x=y line
        min_val = min(ch_data['threshold_slide'].min(), ch_data['peak_slide_specific'].min())
        max_val = max(ch_data['threshold_slide'].max(), ch_data['peak_slide_specific'].max())
        ax_h.plot([min_val*0.9, max_val*1.1], [min_val*0.9, max_val*1.1],
                  color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='x = y')

        # Correlation stats
        r, p = pearsonr(ch_data['threshold_slide'], ch_data['peak_slide_specific'])
        ax_h.text(0.05, 0.95, f"r = {r:.3f}\np = {p:.2e}",
                 transform=ax_h.transAxes, fontsize=9, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax_h.set_xlabel("Threshold (photons)", fontsize=11)
    ax_h.set_ylabel("Peak intensity (photons)", fontsize=11)
    ax_h.set_title(f"H. Threshold vs Peak Intensity - {channel_labels['orange']}",
                   fontsize=12, fontweight='bold', loc='left')
    ax_h.legend(loc='lower right', fontsize=8)
    ax_h.grid(True, alpha=0.3, linestyle='--')

    # Title removed - included in caption instead

    # ══════════════════════════════════════════════════════════════════════════
    # 8. SAVE FIGURE AND CAPTION
    # ══════════════════════════════════════════════════════════════════════════

    print(f"\n{'='*80}")
    print("Saving figure and caption...")
    print(f"{'='*80}")

    # Save figure
    for fmt in ['png', 'svg', 'pdf']:
        filepath = OUTPUT_DIR / f"fig_negative_control_importance.{fmt}"
        plt.savefig(filepath, format=fmt, bbox_inches='tight', dpi=FIGURE_DPI)
        print(f"  Saved: {filepath}")

    plt.close(fig)

    # Save peak intensity comparison data
    csv_path = OUTPUT_DIR / "peak_intensity_comparison_data.csv"
    df_peak_comparison.to_csv(csv_path, index=False)
    print(f"  Data saved: {csv_path}")

    # Save threshold summary
    thr_summary_path = OUTPUT_DIR / "threshold_summary_by_session.csv"
    thr_df.to_csv(thr_summary_path, index=False)
    print(f"  Threshold summary saved: {thr_summary_path}")

    # ══════════════════════════════════════════════════════════════════════════
    # GENERATE COMPREHENSIVE CAPTION
    # ══════════════════════════════════════════════════════════════════════════

    # Collect statistics for caption
    n_sessions = len(thr_df['session'].unique())
    sessions_list = sorted(thr_df['session'].unique())
    n_slides_total = thr_df['slide'].nunique()
    n_slides_peak_comparison = len(df_peak_comparison)

    caption_lines = [
        "Supplementary Figure: Importance of Per-Slide Negative Control Calibration",
        "",
        "This figure demonstrates why per-slide (not just per-session) negative control calibration is essential",
        "for accurate single mRNA peak intensity determination. Even within the same imaging session, slide-to-slide",
        "variability in background levels affects the estimated peak intensity used for mRNA normalization.",
        "",
        "DATA SUMMARY:",
        f"- Imaging sessions: {n_sessions} ({', '.join(sessions_list)})",
        f"- Slides with negative control: {n_slides_total}",
        f"- Slides analyzed for peak comparison: {n_slides_peak_comparison}",
        f"- Note: ALL slides included (no exclusions) to demonstrate full variability",
        "",
        "THRESHOLD VARIABILITY:",
        "",
    ]

    for channel in ['green', 'orange']:
        ch_name = channel_labels[channel]
        cv = global_thresholds[channel]['cv']
        mean_thr = global_thresholds[channel]['mean']
        std_thr = global_thresholds[channel]['std']

        caption_lines.append(f"{ch_name}:")
        caption_lines.append(f"  Global threshold: {mean_thr:.1f} ± {std_thr:.1f} photons (CV = {cv:.1f}%)")

        for session in sorted(thr_df['session'].unique()):
            s_data = thr_df[(thr_df['session'] == session) & (thr_df['channel'] == channel)]
            if len(s_data) > 0:
                caption_lines.append(f"  {session}: {s_data['threshold'].mean():.1f} ± {s_data['threshold'].std():.1f} photons "
                                   f"(n={len(s_data)} slides)")
        caption_lines.append("")

    caption_lines.extend([
        "IMPACT ON SINGLE mRNA PEAK INTENSITY:",
        "",
        "Comparing slide-specific threshold vs session-mean threshold for peak intensity calculation:",
        "",
    ])

    for channel in ['green', 'orange']:
        ch_name = channel_labels[channel]
        ch_data = df_peak_comparison[df_peak_comparison['channel'] == channel]
        if len(ch_data) == 0:
            continue
        mean_bias = ch_data['peak_diff_pct'].mean()
        std_bias = ch_data['peak_diff_pct'].std()

        caption_lines.append(f"{ch_name}:")
        caption_lines.append(f"  Mean peak intensity (slide-specific thr): {ch_data['peak_slide_specific'].mean():.1f} photons")
        caption_lines.append(f"  Mean peak intensity (session-mean thr): {ch_data['peak_session_mean'].mean():.1f} photons")
        caption_lines.append(f"  Mean bias (session_mean - slide_specific): {mean_bias:+.1f}% ± {std_bias:.1f}%")

        for session in sorted(ch_data['session'].unique()):
            s_data = ch_data[ch_data['session'] == session]
            session_bias = s_data['peak_diff_pct'].mean()
            caption_lines.append(f"    {session}: {session_bias:+.1f}% bias (n={len(s_data)} slides)")
        caption_lines.append("")

    caption_lines.extend([
        "PANEL DESCRIPTIONS:",
        "",
        "Row 1 - Threshold Variability:",
        "A. Per-slide thresholds by session for HTT1a (green channel)",
        "B. Per-slide thresholds by session for fl-HTT (orange channel)",
        "   - Each point represents one slide's threshold (95th percentile of negative control)",
        "   - Dashed line shows global mean threshold",
        "   - CV (coefficient of variation) quantifies threshold variability across all slides",
        "",
        "Row 2 - Negative Control Intensity Distributions:",
        "C. KDE of negative control spot intensities by session (green channel)",
        "D. KDE of negative control spot intensities by session (orange channel)",
        "   - Solid line: KDE of spot intensities from negative control probe",
        "   - Dashed vertical lines: session-mean thresholds",
        "   - Black line: global threshold (mean across all sessions)",
        "",
        "Row 3 - Impact on Single mRNA Peak Intensity:",
        "E. Scatter plot comparing single mRNA peak intensity: slide-specific vs session-mean threshold (green)",
        "F. Scatter plot comparing single mRNA peak intensity: slide-specific vs session-mean threshold (orange)",
        "   - Each point represents one slide",
        "   - Diagonal line indicates perfect agreement",
        "   - Deviation from diagonal shows how session-mean threshold affects peak intensity estimation",
        "   - Peak intensity is used to normalize all mRNA counts (spots + clusters)",
        "",
        "Row 4 - Threshold vs Peak Intensity Relationship:",
        "G. Scatter plot of threshold vs peak intensity (green channel)",
        "H. Scatter plot of threshold vs peak intensity (orange channel)",
        "   - Each point represents one slide",
        "   - Linear regression line shows the correlation trend",
        "   - Pearson correlation coefficient (r) and p-value quantify the relationship",
        "   - Positive correlation indicates that higher thresholds (more background) are associated with higher peak intensities",
        "   - This relationship may reflect imaging conditions that affect both background and signal levels",
        "",
        "INTERPRETATION:",
        "- Slide-to-slide variability in thresholds exists even within the same imaging session",
        "- Using session-mean threshold instead of slide-specific threshold introduces bias in peak intensity",
        "- Peak intensity bias directly propagates to all mRNA quantification:",
        "  * If peak intensity is overestimated: mRNA counts are underestimated",
        "  * If peak intensity is underestimated: mRNA counts are overestimated",
        "- Per-slide calibration provides the most accurate single mRNA reference intensity",
        "",
        "METHODOLOGY:",
        f"- Threshold determination: {QUANTILE_NEGATIVE_CONTROL*100:.0f}th percentile of negative control intensities",
        f"- Negative control: dT probe hybridizing to non-specific background",
        "- Peak intensity: Mode of KDE (kernel density estimation) of spot intensities above threshold",
        "- Session mean threshold: Average of all slide-specific thresholds within a session",
        "",
        f"Analysis performed with scienceplots style.",
    ])

    caption_text = '\n'.join(caption_lines)

    # Save caption
    caption_path = OUTPUT_DIR / 'fig_negative_control_importance_caption.txt'
    with open(caption_path, 'w') as f:
        f.write(caption_text)
    print(f"  Caption saved: {caption_path}")

    # Save LaTeX caption
    caption_latex = caption_text.replace('_', '\\_').replace('%', '\\%').replace('μ', '$\\mu$').replace('³', '$^3$')
    caption_latex_path = OUTPUT_DIR / 'fig_negative_control_importance_caption.tex'
    with open(caption_latex_path, 'w') as f:
        f.write(caption_latex)
    print(f"  LaTeX caption saved: {caption_latex_path}")

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
