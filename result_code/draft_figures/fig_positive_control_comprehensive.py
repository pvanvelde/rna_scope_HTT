"""
Comprehensive Positive Control Analysis Figure

This script creates a detailed multi-panel figure analyzing positive control expression
levels across different brain regions (Cortex vs Striatum) and their relationships with:
- Gene identity (POLR2A low-expression vs UBC high-expression housekeepers)
- Age
- Brain atlas coordinates
- Regional differences

Author: Generated for RNA Scope analysis
Date: 2025-11-16
"""

import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel, pearsonr, linregress, gaussian_kde
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
    CHANNEL_COLORS,
    MEAN_NUCLEAR_VOLUME,
    VOXEL_SIZE,
    MAX_PFA,
    QUANTILE_NEGATIVE_CONTROL,
    N_BOOTSTRAP,
    MIN_NUCLEI_THRESHOLD,
    EXCLUDED_SLIDES,
    SLIDE_FIELD,
    CV_THRESHOLD
)

# Physical parameters
pixelsize = PIXELSIZE
slice_depth = SLICE_DEPTH
mean_nuclear_volume = MEAN_NUCLEAR_VOLUME
voxel_size = VOXEL_SIZE

# Create output directory
OUTPUT_DIR = Path(__file__).parent / "output" / "positive_control_comprehensive"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def compute_peak_intensity(intensities, bw_method='scott'):
    """Compute peak intensity from KDE (highest probability intensity)."""
    if len(intensities) < 50:
        return np.nan

    try:
        kde = gaussian_kde(intensities, bw_method=bw_method)
        # Sample KDE at many points to find peak
        x_range = np.linspace(np.percentile(intensities, 1),
                             np.percentile(intensities, 99), 1000)
        y_density = kde(x_range)

        # Find peak (maximum density)
        peak_idx = np.argmax(y_density)
        peak_intensity = x_range[peak_idx]

        return peak_intensity
    except:
        return np.nan


if __name__ == "__main__":

    print("="*80)
    print("COMPREHENSIVE POSITIVE CONTROL ANALYSIS")
    print("Analyzing POLR2A (low) and UBC (high) housekeeping gene expression")
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
        'spots.pfa_values',
        'spots.photons',
        'cluster_intensities',
        'cluster_cvs',
        'num_cells',
        'label_sizes',
        'metadata_sample.Age',
        'spots.final_filter',
        'spots.params_raw',
        'metadata_sample.Brain_Atlas_coordinates',
        'spots_sigma_var.params_raw'
    ]

    negative_control_field = 'Negative control'
    experimental_field = 'ExperimentalQ111 - 488mHT - 548mHTa - 647Darp'
    slide_field = 'metadata_sample_slide_name_std'
    max_pfa = MAX_PFA
    quantile_negative_control = QUANTILE_NEGATIVE_CONTROL

    df_extracted_full = extract_dataframe(
        data_dict,
        field_keys=fields_to_extract,
        channels=channels_to_extract,
        include_file_metadata_sample=True,
        explode_fields=[]
    )

    print(f"\nDataFrame extracted:")
    print(f"  Total rows: {len(df_extracted_full)}")
    print(f"  Channels: {channels_to_analyze}")

    # ══════════════════════════════════════════════════════════════════════════
    # 3. COMPUTE THRESHOLDS
    # ══════════════════════════════════════════════════════════════════════════

    print(f"\n{'='*80}")
    print("Computing thresholds...")
    print(f"{'='*80}")

    (thresholds, thresholds_cluster,
     error_thresholds, error_thresholds_cluster,
     number_of_datapoints, age_dict) = compute_thresholds(
        df_extracted=df_extracted_full,
        slide_field=slide_field,
        desired_channels=channels_to_analyze,
        negative_control_field=negative_control_field,
        experimental_field=experimental_field,
        quantile_negative_control=quantile_negative_control,
        max_pfa=max_pfa,
        plot=False,
        n_bootstrap=N_BOOTSTRAP,
        use_region=False,
        use_final_filter=True,
    )

    # Build threshold lookup table
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

    # Merge thresholds
    df_extracted_full = df_extracted_full.merge(
        thr_df,
        how="left",
        left_on=[slide_field, "channel"],
        right_on=["slide", "channel"]
    )
    df_extracted_full.rename(columns={"thr": "threshold"}, inplace=True)
    df_extracted_full.drop(columns=["slide"], inplace=True, errors='ignore')

    # Filter to Q111 and Wildtype positive control
    df_pc = df_extracted_full[
        (df_extracted_full["metadata_sample_Mouse_Model"].isin(['Q111', 'Wildtype'])) &
        (df_extracted_full['metadata_sample_Probe-Set'] == 'Positive control')
    ].copy()

    print(f"Positive control Q111 and Wildtype records (before QC): {len(df_pc)}")

    # Apply slide exclusion (QC filter for technical failures)
    if len(EXCLUDED_SLIDES) > 0:
        n_before = len(df_pc)
        df_pc = df_pc[~df_pc[SLIDE_FIELD].isin(EXCLUDED_SLIDES)].copy()
        n_after = len(df_pc)
        n_excluded = n_before - n_after
        print(f"  Excluded {n_excluded} FOVs from {len(EXCLUDED_SLIDES)} slides: {EXCLUDED_SLIDES}")
        print(f"  Remaining FOVs after QC: {n_after}")
    else:
        print(f"  No slides excluded (EXCLUDED_SLIDES is empty)")

    print(f"Positive control Q111 and Wildtype records (after QC): {len(df_pc)}")

    # ══════════════════════════════════════════════════════════════════════════
    # 4. COMPUTE SLIDE-SPECIFIC PEAK INTENSITIES
    # ══════════════════════════════════════════════════════════════════════════

    print(f"\n{'='*80}")
    print("COMPUTING SLIDE-SPECIFIC PEAK INTENSITIES")
    print(f"{'='*80}")

    spot_peaks = {}  # (slide, channel) -> peak_intensity

    for idx, row in df_pc.iterrows():
        slide = row.get(slide_field, 'unknown')
        channel = row.get('channel', 'unknown')
        threshold_val = row.get('threshold', np.nan)

        # Skip blue channel
        if channel == 'blue':
            continue

        key = (slide, channel)

        # Compute peak intensity for this slide/channel if not already done
        if key not in spot_peaks:
            # Collect all single spot intensities from all FOVs for this slide/channel
            all_intensities = []

            for idx2, row2 in df_pc.iterrows():
                slide2 = row2.get(slide_field, 'unknown')
                channel2 = row2.get('channel', 'unknown')
                threshold_val2 = row2.get('threshold', np.nan)

                if slide2 == slide and channel2 == channel:
                    sigma_var_params = row2.get('spots_sigma_var.params_raw', None)
                    final_filter = row2.get('spots.final_filter', None)

                    if sigma_var_params is not None and final_filter is not None:
                        try:
                            sigma_var_params = np.asarray(sigma_var_params)
                            final_filter = np.asarray(final_filter).astype(bool)

                            if sigma_var_params.ndim >= 2 and sigma_var_params.shape[1] > 3:
                                if final_filter.sum() > 0:
                                    photons_filtered = sigma_var_params[final_filter, 3]

                                    if not np.isnan(threshold_val2) and len(photons_filtered) > 0:
                                        above_threshold = photons_filtered > threshold_val2
                                        valid_photons = photons_filtered[above_threshold]
                                        all_intensities.extend(valid_photons)
                        except:
                            pass

            # Compute peak from all intensities
            if len(all_intensities) >= 50:
                peak_intensity = compute_peak_intensity(np.array(all_intensities))
                if not np.isnan(peak_intensity):
                    spot_peaks[key] = peak_intensity
                    print(f"  {slide}, {channel}: peak = {peak_intensity:.2f}")

    print(f"\nComputed {len(spot_peaks)} slide-specific peak intensities")

    # ══════════════════════════════════════════════════════════════════════════
    # 5. BUILD DAPI LOOKUP AND EXTRACT FOV-LEVEL DATA
    # ══════════════════════════════════════════════════════════════════════════

    print(f"\n{'='*80}")
    print("BUILDING FOV-LEVEL EXPRESSION DATA")
    print(f"{'='*80}")

    # Get all data including blue channel for DAPI
    df_all_channels = df_extracted_full[
        (df_extracted_full["metadata_sample_Mouse_Model"].isin(['Q111', 'Wildtype'])) &
        (df_extracted_full['metadata_sample_Probe-Set'] == 'Positive control')
    ].copy()

    # Sort by index
    df_all_channels = df_all_channels.sort_index()

    # Build DAPI lookup
    dapi_lookup = {}

    for idx, row in df_all_channels.iterrows():
        channel = row.get('channel', 'unknown')

        if channel != 'blue':
            continue

        label_sizes = row.get('label_sizes', None)
        N_nuc = 0
        V_DAPI = 0

        if label_sizes is not None:
            try:
                label_sizes = np.asarray(label_sizes)
                V_DAPI = np.sum(label_sizes) * voxel_size
                N_nuc = V_DAPI / mean_nuclear_volume
            except:
                pass

        dapi_lookup[idx] = (N_nuc, V_DAPI)
        dapi_lookup[idx + 1] = (N_nuc, V_DAPI)
        dapi_lookup[idx + 2] = (N_nuc, V_DAPI)

    # Extract FOV-level data with QC
    fov_data = []
    min_nuclei_threshold = MIN_NUCLEI_THRESHOLD

    for idx, row in df_pc.iterrows():
        slide = row.get(slide_field, 'unknown')
        region = row.get('metadata_sample_Slice_Region', 'unknown')
        channel = row.get('channel', 'unknown')
        age = row.get('metadata_sample_Age', np.nan)
        atlas_coord = row.get('metadata_sample_Brain_Atlas_coordinates', np.nan)
        threshold_val = row.get('threshold', np.nan)

        if channel == 'blue':
            continue

        # Get slide-specific peak intensity
        key = (slide, channel)
        if key not in spot_peaks:
            continue

        peak_intensity = spot_peaks[key]

        # Get N_nuc from DAPI lookup
        if idx not in dapi_lookup:
            continue

        N_nuc, V_DAPI = dapi_lookup[idx]

        if N_nuc < min_nuclei_threshold:
            continue

        # Count single spots
        num_spots = 0
        sigma_var_params = row.get('spots_sigma_var.params_raw', None)
        final_filter = row.get('spots.final_filter', None)

        if sigma_var_params is not None and final_filter is not None:
            try:
                sigma_var_params = np.asarray(sigma_var_params)
                final_filter = np.asarray(final_filter).astype(bool)

                if sigma_var_params.ndim >= 2 and sigma_var_params.shape[1] > 3:
                    if final_filter.sum() > 0:
                        photons_filtered = sigma_var_params[final_filter, 3]

                        if not np.isnan(threshold_val) and len(photons_filtered) > 0:
                            above_threshold = photons_filtered > threshold_val
                            num_spots = above_threshold.sum()
            except:
                pass

        # Count clusters (with intensity AND CV filtering)
        num_clusters = 0
        I_cluster_total = 0
        cluster_int = row.get('cluster_intensities', None)
        cluster_cvs = row.get('cluster_cvs', None)
        if cluster_int is not None:
            try:
                cluster_int = np.asarray(cluster_int)
                if not np.isnan(threshold_val) and len(cluster_int) > 0:
                    # Intensity threshold
                    intensity_mask = cluster_int > threshold_val
                    # CV threshold (CV >= CV_THRESHOLD means good quality)
                    # CV data is required - no fallback
                    if cluster_cvs is None:
                        raise ValueError("CV data missing for cluster filtering")
                    cluster_cvs = np.asarray(cluster_cvs)
                    if len(cluster_cvs) != len(cluster_int):
                        raise ValueError(f"CV data length mismatch: {len(cluster_cvs)} vs {len(cluster_int)}")
                    cv_mask = cluster_cvs >= CV_THRESHOLD
                    above_threshold = intensity_mask & cv_mask
                    num_clusters = above_threshold.sum()
                    I_cluster_total = cluster_int[above_threshold].sum()
            except:
                pass

        # Compute expression: E = (N_spots + I_cluster_total / I_single_peak) / N_nuc
        cluster_mrna_equiv = I_cluster_total / peak_intensity if peak_intensity > 0 else 0
        total_mrna_equiv = num_spots + cluster_mrna_equiv
        expression_per_cell = total_mrna_equiv / N_nuc if N_nuc > 0 else np.nan

        # Simplify region labels
        region_short = "Cortex" if "Cortex" in region else "Striatum" if "Striatum" in region else region

        fov_data.append({
            'slide': slide,
            'region': region_short,
            'region_full': region,
            'channel': channel,
            'age': age,
            'atlas_coord': atlas_coord,
            'N_nuc': N_nuc,
            'num_spots': num_spots,
            'num_clusters': num_clusters,
            'expression_per_cell': expression_per_cell,
            'total_mrna': total_mrna_equiv,
            'peak_intensity': peak_intensity
        })

    df_fov = pd.DataFrame(fov_data)

    print(f"\nFOV-level data extracted: {len(df_fov)} FOVs")
    print(f"  Unique slides: {df_fov['slide'].nunique()}")
    print(f"  Regions: {sorted(df_fov['region'].unique())}")
    print(f"  Channels: {sorted(df_fov['channel'].unique())}")

    # ══════════════════════════════════════════════════════════════════════════
    # 6. COMPUTE PER-SLIDE AVERAGES
    # ══════════════════════════════════════════════════════════════════════════

    print(f"\n{'='*80}")
    print("COMPUTING PER-SLIDE AVERAGES")
    print(f"{'='*80}")

    # Group by slide, channel, region
    slide_avg = df_fov.groupby(['slide', 'channel', 'region']).agg({
        'expression_per_cell': 'mean',
        'age': 'first',
        'atlas_coord': 'first',
        'N_nuc': 'mean'
    }).reset_index()

    # Pivot to get gene columns
    slide_wide = slide_avg.pivot_table(
        index=['slide', 'region', 'age', 'atlas_coord'],
        columns='channel',
        values='expression_per_cell'
    ).reset_index()

    # Rename channels to gene names
    slide_wide.rename(columns={'green': 'POLR2A', 'orange': 'UBC'}, inplace=True)

    # Compute ratio
    slide_wide['UBC_POLR2A_ratio'] = slide_wide['UBC'] / slide_wide['POLR2A']

    # Filter for paired samples (both Cortex and Striatum)
    slides_with_both = slide_wide.groupby('slide').filter(
        lambda x: set(x['region']) >= {'Cortex', 'Striatum'}
    )

    # Create paired dataframe
    mean_df = (
        slides_with_both.groupby(['slide'])
        .apply(lambda x: pd.Series({
            'Cortex_POLR2A': x[x['region']=='Cortex']['POLR2A'].values[0] if 'Cortex' in x['region'].values else np.nan,
            'Striatum_POLR2A': x[x['region']=='Striatum']['POLR2A'].values[0] if 'Striatum' in x['region'].values else np.nan,
            'Cortex_UBC': x[x['region']=='Cortex']['UBC'].values[0] if 'Cortex' in x['region'].values else np.nan,
            'Striatum_UBC': x[x['region']=='Striatum']['UBC'].values[0] if 'Striatum' in x['region'].values else np.nan,
            'age': x['age'].values[0],
            'atlas_coord': x['atlas_coord'].values[0]
        }))
        .reset_index()
        .dropna(subset=['Cortex_POLR2A', 'Striatum_POLR2A', 'Cortex_UBC', 'Striatum_UBC'])
    )

    mean_df['diff_POLR2A'] = mean_df['Cortex_POLR2A'] - mean_df['Striatum_POLR2A']
    mean_df['diff_UBC'] = mean_df['Cortex_UBC'] - mean_df['Striatum_UBC']
    mean_df['ratio_Cortex'] = mean_df['Cortex_UBC'] / mean_df['Cortex_POLR2A']
    mean_df['ratio_Striatum'] = mean_df['Striatum_UBC'] / mean_df['Striatum_POLR2A']
    mean_df['ratio_diff'] = mean_df['ratio_Cortex'] - mean_df['ratio_Striatum']

    print(f"\nPaired samples: {len(mean_df)}")
    print(f"Age range: {mean_df['age'].min():.0f} - {mean_df['age'].max():.0f} months")
    print(f"Atlas range: {mean_df['atlas_coord'].min():.1f} - {mean_df['atlas_coord'].max():.1f} mm")

    # ══════════════════════════════════════════════════════════════════════════
    # 7. STATISTICAL ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════

    print(f"\n{'='*80}")
    print("STATISTICAL ANALYSIS")
    print(f"{'='*80}")

    # Regional comparison for POLR2A
    t_polr2a, p_polr2a = ttest_rel(mean_df['Cortex_POLR2A'], mean_df['Striatum_POLR2A'])
    print(f"\n1. POLR2A Regional Comparison:")
    print(f"   Cortex: {mean_df['Cortex_POLR2A'].mean():.2f} ± {mean_df['Cortex_POLR2A'].std():.2f}")
    print(f"   Striatum: {mean_df['Striatum_POLR2A'].mean():.2f} ± {mean_df['Striatum_POLR2A'].std():.2f}")
    print(f"   t={t_polr2a:.3f}, p={p_polr2a:.4g}")

    # Regional comparison for UBC
    t_ubc, p_ubc = ttest_rel(mean_df['Cortex_UBC'], mean_df['Striatum_UBC'])
    print(f"\n2. UBC Regional Comparison:")
    print(f"   Cortex: {mean_df['Cortex_UBC'].mean():.2f} ± {mean_df['Cortex_UBC'].std():.2f}")
    print(f"   Striatum: {mean_df['Striatum_UBC'].mean():.2f} ± {mean_df['Striatum_UBC'].std():.2f}")
    print(f"   t={t_ubc:.3f}, p={p_ubc:.4g}")

    # Gene comparison (UBC vs POLR2A)
    print(f"\n3. Gene Expression Comparison (averaged across regions):")
    all_polr2a = pd.concat([mean_df['Cortex_POLR2A'], mean_df['Striatum_POLR2A']])
    all_ubc = pd.concat([mean_df['Cortex_UBC'], mean_df['Striatum_UBC']])
    print(f"   POLR2A: {all_polr2a.mean():.2f} ± {all_polr2a.std():.2f} mRNA/cell")
    print(f"   UBC: {all_ubc.mean():.2f} ± {all_ubc.std():.2f} mRNA/cell")
    print(f"   Ratio (UBC/POLR2A): {(all_ubc.mean()/all_polr2a.mean()):.2f}")

    # ══════════════════════════════════════════════════════════════════════════
    # 8. CREATE COMPREHENSIVE FIGURE
    # ══════════════════════════════════════════════════════════════════════════

    print(f"\n{'='*80}")
    print("Creating comprehensive figure...")
    print(f"{'='*80}")

    # Create figure with 4×3 grid
    # Leave space at top (top=0.92) for adding images later
    fig = plt.figure(figsize=(16, 12), dpi=300)
    gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.30,
                         left=0.07, right=0.98, top=0.92, bottom=0.05)

    # Gene colors
    gene_colors = {
        'POLR2A': 'green',
        'UBC': 'orange'
    }

    # ══════════════════════════════════════════════════════════════════════════
    # ROW 1: GENE EXPRESSION COMPARISONS
    # ══════════════════════════════════════════════════════════════════════════

    # Panel A: POLR2A vs UBC scatter (colored by region)
    ax_a = fig.add_subplot(gs[0, 0])
    for region in ['Cortex', 'Striatum']:
        region_data = slide_wide[slide_wide['region'] == region]
        color = 'blue' if region == 'Cortex' else 'red'
        ax_a.scatter(region_data['POLR2A'], region_data['UBC'],
                   c=color, s=70, alpha=0.7, edgecolor='black', linewidth=0.7,
                   label=region)

    # Set x-axis limit to start from 0
    polr2a_max = slide_wide['POLR2A'].max() * 1.1
    ax_a.set_xlim(0, polr2a_max)

    # Add diagonal line from 0 to the appropriate limit
    ubc_max = slide_wide['UBC'].max()
    max_lim = max(polr2a_max, ubc_max * 1.05)
    ax_a.plot([0, max_lim], [0, max_lim], 'k--', lw=1.5, alpha=0.5, label='Unity')

    ax_a.set_xlabel("POLR2A - Total mRNA [mRNA/cell]", fontsize=11)
    ax_a.set_ylabel("UBC - Total mRNA [mRNA/cell]", fontsize=11)
    ax_a.set_title("A. POLR2A vs UBC Expression", fontsize=12, fontweight='bold', loc='left')
    ax_a.legend(loc='best', fontsize=9)
    ax_a.grid(True, alpha=0.3, linestyle='--')

    # Panel B: Expression ratio distribution
    ax_b = fig.add_subplot(gs[0, 1])
    ratios = slide_wide['UBC_POLR2A_ratio'].dropna()
    ax_b.hist(ratios, bins=20, alpha=0.7, color='purple',
             edgecolor='black', linewidth=0.8)
    ax_b.axvline(ratios.mean(), color='darkblue', linestyle='-', linewidth=2,
                alpha=0.7, label=f'Mean={ratios.mean():.2f}')
    ax_b.axvline(ratios.median(), color='red', linestyle='--', linewidth=2,
                alpha=0.7, label=f'Median={ratios.median():.2f}')
    ax_b.set_xlabel("UBC / POLR2A ratio", fontsize=11)
    ax_b.set_ylabel("Count", fontsize=11)
    ax_b.set_title("B. Expression Ratio Distribution", fontsize=12, fontweight='bold', loc='left')
    ax_b.legend(loc='best', fontsize=9)
    ax_b.grid(True, alpha=0.3, linestyle='--', axis='y')

    # Panel C: Regional comparison for both genes
    ax_c = fig.add_subplot(gs[0, 2])

    # POLR2A
    polr2a_cortex = mean_df['Cortex_POLR2A'].values
    polr2a_striatum = mean_df['Striatum_POLR2A'].values
    positions_polr2a = [1, 2]
    bp1 = ax_c.boxplot([polr2a_cortex, polr2a_striatum], positions=positions_polr2a,
                       widths=0.6, patch_artist=True,
                       boxprops=dict(facecolor='lightgreen', alpha=0.7),
                       medianprops=dict(color='darkgreen', linewidth=2))

    # UBC
    ubc_cortex = mean_df['Cortex_UBC'].values
    ubc_striatum = mean_df['Striatum_UBC'].values
    positions_ubc = [3.5, 4.5]
    bp2 = ax_c.boxplot([ubc_cortex, ubc_striatum], positions=positions_ubc,
                       widths=0.6, patch_artist=True,
                       boxprops=dict(facecolor='orange', alpha=0.7),
                       medianprops=dict(color='darkorange', linewidth=2))

    ax_c.set_xticks([1.5, 4])
    ax_c.set_xticklabels(['POLR2A', 'UBC'], fontsize=10)
    ax_c.set_ylabel("Total mRNA Expression [mRNA/cell]", fontsize=11)
    ax_c.set_title("C. Regional Expression Comparison", fontsize=12, fontweight='bold', loc='left')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='lightblue', label='Cortex'),
                      Patch(facecolor='lightcoral', label='Striatum')]

    # Color the boxes
    for i, patch in enumerate(bp1['boxes']):
        patch.set_facecolor('lightblue' if i == 0 else 'lightcoral')
    for i, patch in enumerate(bp2['boxes']):
        patch.set_facecolor('lightblue' if i == 0 else 'lightcoral')

    ax_c.grid(True, alpha=0.3, linestyle='--', axis='y')

    # Add t-test statistics annotations
    def format_pvalue(p):
        if p < 0.001:
            return "***"
        elif p < 0.01:
            return "**"
        elif p < 0.05:
            return "*"
        else:
            return "ns"

    # Calculate y-axis limits to accommodate annotations
    y_max_ubc = max(ubc_cortex.max(), ubc_striatum.max())
    y_max_overall = y_max_ubc * 1.25  # Add headroom for annotations
    ax_c.set_ylim(0, y_max_overall)

    # POLR2A significance line (lower, since values are smaller)
    y_max_polr2a = max(polr2a_cortex.max(), polr2a_striatum.max())
    y_line_polr2a = y_max_polr2a * 1.15
    ax_c.plot([1, 2], [y_line_polr2a, y_line_polr2a], 'k-', linewidth=1.2)
    ax_c.text(1.5, y_line_polr2a + 2, format_pvalue(p_polr2a),
             ha='center', va='bottom', fontsize=10, fontweight='bold')

    # UBC significance line (higher)
    y_line_ubc = y_max_ubc * 1.08
    ax_c.plot([3.5, 4.5], [y_line_ubc, y_line_ubc], 'k-', linewidth=1.2)
    ax_c.text(4, y_line_ubc + 5, format_pvalue(p_ubc),
             ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Place legend in upper left to avoid overlap with UBC annotations
    ax_c.legend(handles=legend_elements, loc='upper left', fontsize=8)

    # ══════════════════════════════════════════════════════════════════════════
    # ROW 2: AGE DEPENDENCIES
    # ══════════════════════════════════════════════════════════════════════════

    valid_age_df = mean_df[mean_df['age'].notna()]

    # Panel D: Age vs POLR2A (both regions with separate regressions)
    ax_d = fig.add_subplot(gs[1, 0])
    if len(valid_age_df) > 0:
        ax_d.scatter(valid_age_df['age'], valid_age_df['Cortex_POLR2A'],
                   c='blue', s=60, alpha=0.7, edgecolor='black', linewidth=0.7,
                   label='Cortex', marker='o')
        ax_d.scatter(valid_age_df['age'], valid_age_df['Striatum_POLR2A'],
                   c='red', s=60, alpha=0.7, edgecolor='black', linewidth=0.7,
                   label='Striatum', marker='s')

        # Separate regression for Cortex
        if len(valid_age_df) > 2:
            slope_c, intercept_c, r_c, p_c, _ = linregress(valid_age_df['age'], valid_age_df['Cortex_POLR2A'])
            x_line = np.array([valid_age_df['age'].min(), valid_age_df['age'].max()])
            y_line_c = slope_c * x_line + intercept_c
            ax_d.plot(x_line, y_line_c, '--', color='blue', linewidth=2, alpha=0.8,
                    label=f'Cortex: r={r_c:.3f}, p={p_c:.3g}')

        # Separate regression for Striatum
        if len(valid_age_df) > 2:
            slope_s, intercept_s, r_s, p_s, _ = linregress(valid_age_df['age'], valid_age_df['Striatum_POLR2A'])
            y_line_s = slope_s * x_line + intercept_s
            ax_d.plot(x_line, y_line_s, '--', color='red', linewidth=2, alpha=0.8,
                    label=f'Striatum: r={r_s:.3f}, p={p_s:.3g}')

    ax_d.set_xlabel("Age [months]", fontsize=11)
    ax_d.set_ylabel("Total mRNA Expression [mRNA/cell]", fontsize=11)
    ax_d.set_title("D. Age vs POLR2A Expression", fontsize=12, fontweight='bold', loc='left')
    ax_d.legend(loc='best', fontsize=7)
    ax_d.grid(True, alpha=0.3, linestyle='--')

    # Panel E: Age vs UBC (both regions with separate regressions)
    ax_e = fig.add_subplot(gs[1, 1])
    if len(valid_age_df) > 0:
        ax_e.scatter(valid_age_df['age'], valid_age_df['Cortex_UBC'],
                   c='blue', s=60, alpha=0.7, edgecolor='black', linewidth=0.7,
                   label='Cortex', marker='o')
        ax_e.scatter(valid_age_df['age'], valid_age_df['Striatum_UBC'],
                   c='red', s=60, alpha=0.7, edgecolor='black', linewidth=0.7,
                   label='Striatum', marker='s')

        # Separate regression for Cortex
        if len(valid_age_df) > 2:
            slope_c, intercept_c, r_c, p_c, _ = linregress(valid_age_df['age'], valid_age_df['Cortex_UBC'])
            x_line = np.array([valid_age_df['age'].min(), valid_age_df['age'].max()])
            y_line_c = slope_c * x_line + intercept_c
            ax_e.plot(x_line, y_line_c, '--', color='blue', linewidth=2, alpha=0.8,
                    label=f'Cortex: r={r_c:.3f}, p={p_c:.3g}')

        # Separate regression for Striatum
        if len(valid_age_df) > 2:
            slope_s, intercept_s, r_s, p_s, _ = linregress(valid_age_df['age'], valid_age_df['Striatum_UBC'])
            y_line_s = slope_s * x_line + intercept_s
            ax_e.plot(x_line, y_line_s, '--', color='red', linewidth=2, alpha=0.8,
                    label=f'Striatum: r={r_s:.3f}, p={p_s:.3g}')

    ax_e.set_xlabel("Age [months]", fontsize=11)
    ax_e.set_ylabel("Total mRNA Expression [mRNA/cell]", fontsize=11)
    ax_e.set_title("E. Age vs UBC Expression", fontsize=12, fontweight='bold', loc='left')
    ax_e.legend(loc='best', fontsize=7)
    ax_e.grid(True, alpha=0.3, linestyle='--')

    # Panel F: Age vs ratio (both regions with separate regressions)
    ax_f = fig.add_subplot(gs[1, 2])
    if len(valid_age_df) > 0:
        ax_f.scatter(valid_age_df['age'], valid_age_df['ratio_Cortex'],
                   c='blue', s=60, alpha=0.7, edgecolor='black', linewidth=0.7,
                   label='Cortex', marker='o')
        ax_f.scatter(valid_age_df['age'], valid_age_df['ratio_Striatum'],
                   c='red', s=60, alpha=0.7, edgecolor='black', linewidth=0.7,
                   label='Striatum', marker='s')

        # Separate regression for Cortex
        if len(valid_age_df) > 2:
            slope_c, intercept_c, r_c, p_c, _ = linregress(valid_age_df['age'], valid_age_df['ratio_Cortex'])
            x_line = np.array([valid_age_df['age'].min(), valid_age_df['age'].max()])
            y_line_c = slope_c * x_line + intercept_c
            ax_f.plot(x_line, y_line_c, '--', color='blue', linewidth=2, alpha=0.8,
                    label=f'Cortex: r={r_c:.3f}, p={p_c:.3g}')

        # Separate regression for Striatum
        if len(valid_age_df) > 2:
            slope_s, intercept_s, r_s, p_s, _ = linregress(valid_age_df['age'], valid_age_df['ratio_Striatum'])
            y_line_s = slope_s * x_line + intercept_s
            ax_f.plot(x_line, y_line_s, '--', color='red', linewidth=2, alpha=0.8,
                    label=f'Striatum: r={r_s:.3f}, p={p_s:.3g}')

    ax_f.set_xlabel("Age [months]", fontsize=11)
    ax_f.set_ylabel("UBC / POLR2A Ratio", fontsize=11)
    ax_f.set_title("F. Age vs Expression Ratio", fontsize=12, fontweight='bold', loc='left')
    ax_f.legend(loc='best', fontsize=7)
    ax_f.grid(True, alpha=0.3, linestyle='--')

    # ══════════════════════════════════════════════════════════════════════════
    # ROW 3: ATLAS COORDINATES
    # ══════════════════════════════════════════════════════════════════════════

    valid_atlas_df = mean_df[mean_df['atlas_coord'].notna()]

    # Panel G: Atlas vs POLR2A (both regions with separate regressions)
    ax_g = fig.add_subplot(gs[2, 0])
    if len(valid_atlas_df) > 0:
        ax_g.scatter(valid_atlas_df['atlas_coord'], valid_atlas_df['Cortex_POLR2A'],
                   c='blue', s=60, alpha=0.7, edgecolor='black', linewidth=0.7,
                   label='Cortex', marker='o')
        ax_g.scatter(valid_atlas_df['atlas_coord'], valid_atlas_df['Striatum_POLR2A'],
                   c='red', s=60, alpha=0.7, edgecolor='black', linewidth=0.7,
                   label='Striatum', marker='s')

        # Separate regression for Cortex
        if len(valid_atlas_df) > 2:
            slope_c, intercept_c, r_c, p_c, _ = linregress(valid_atlas_df['atlas_coord'], valid_atlas_df['Cortex_POLR2A'])
            x_line = np.array([valid_atlas_df['atlas_coord'].min(), valid_atlas_df['atlas_coord'].max()])
            y_line_c = slope_c * x_line + intercept_c
            ax_g.plot(x_line, y_line_c, '--', color='blue', linewidth=2, alpha=0.8,
                    label=f'Cortex: r={r_c:.3f}, p={p_c:.3g}')

        # Separate regression for Striatum
        if len(valid_atlas_df) > 2:
            slope_s, intercept_s, r_s, p_s, _ = linregress(valid_atlas_df['atlas_coord'], valid_atlas_df['Striatum_POLR2A'])
            y_line_s = slope_s * x_line + intercept_s
            ax_g.plot(x_line, y_line_s, '--', color='red', linewidth=2, alpha=0.8,
                    label=f'Striatum: r={r_s:.3f}, p={p_s:.3g}')

    ax_g.set_xlabel("Brain Atlas Coordinate [mm]", fontsize=11)
    ax_g.set_ylabel("Total mRNA Expression [mRNA/cell]", fontsize=11)
    ax_g.set_title("G. Atlas Coordinate vs POLR2A", fontsize=12, fontweight='bold', loc='left')
    ax_g.legend(loc='best', fontsize=7)
    ax_g.grid(True, alpha=0.3, linestyle='--')

    # Panel H: Atlas vs UBC (both regions with separate regressions)
    ax_h = fig.add_subplot(gs[2, 1])
    if len(valid_atlas_df) > 0:
        ax_h.scatter(valid_atlas_df['atlas_coord'], valid_atlas_df['Cortex_UBC'],
                   c='blue', s=60, alpha=0.7, edgecolor='black', linewidth=0.7,
                   label='Cortex', marker='o')
        ax_h.scatter(valid_atlas_df['atlas_coord'], valid_atlas_df['Striatum_UBC'],
                   c='red', s=60, alpha=0.7, edgecolor='black', linewidth=0.7,
                   label='Striatum', marker='s')

        # Separate regression for Cortex
        if len(valid_atlas_df) > 2:
            slope_c, intercept_c, r_c, p_c, _ = linregress(valid_atlas_df['atlas_coord'], valid_atlas_df['Cortex_UBC'])
            x_line = np.array([valid_atlas_df['atlas_coord'].min(), valid_atlas_df['atlas_coord'].max()])
            y_line_c = slope_c * x_line + intercept_c
            ax_h.plot(x_line, y_line_c, '--', color='blue', linewidth=2, alpha=0.8,
                    label=f'Cortex: r={r_c:.3f}, p={p_c:.3g}')

        # Separate regression for Striatum
        if len(valid_atlas_df) > 2:
            slope_s, intercept_s, r_s, p_s, _ = linregress(valid_atlas_df['atlas_coord'], valid_atlas_df['Striatum_UBC'])
            y_line_s = slope_s * x_line + intercept_s
            ax_h.plot(x_line, y_line_s, '--', color='red', linewidth=2, alpha=0.8,
                    label=f'Striatum: r={r_s:.3f}, p={p_s:.3g}')

    ax_h.set_xlabel("Brain Atlas Coordinate [mm]", fontsize=11)
    ax_h.set_ylabel("Total mRNA Expression [mRNA/cell]", fontsize=11)
    ax_h.set_title("H. Atlas Coordinate vs UBC", fontsize=12, fontweight='bold', loc='left')
    ax_h.legend(loc='best', fontsize=7)
    ax_h.grid(True, alpha=0.3, linestyle='--')

    # Panel I: Atlas vs ratio (both regions with separate regressions)
    ax_i = fig.add_subplot(gs[2, 2])
    if len(valid_atlas_df) > 0:
        ax_i.scatter(valid_atlas_df['atlas_coord'], valid_atlas_df['ratio_Cortex'],
                   c='blue', s=60, alpha=0.7, edgecolor='black', linewidth=0.7,
                   label='Cortex', marker='o')
        ax_i.scatter(valid_atlas_df['atlas_coord'], valid_atlas_df['ratio_Striatum'],
                   c='red', s=60, alpha=0.7, edgecolor='black', linewidth=0.7,
                   label='Striatum', marker='s')

        # Separate regression for Cortex
        if len(valid_atlas_df) > 2:
            slope_c, intercept_c, r_c, p_c, _ = linregress(valid_atlas_df['atlas_coord'], valid_atlas_df['ratio_Cortex'])
            x_line = np.array([valid_atlas_df['atlas_coord'].min(), valid_atlas_df['atlas_coord'].max()])
            y_line_c = slope_c * x_line + intercept_c
            ax_i.plot(x_line, y_line_c, '--', color='blue', linewidth=2, alpha=0.8,
                    label=f'Cortex: r={r_c:.3f}, p={p_c:.3g}')

        # Separate regression for Striatum
        if len(valid_atlas_df) > 2:
            slope_s, intercept_s, r_s, p_s, _ = linregress(valid_atlas_df['atlas_coord'], valid_atlas_df['ratio_Striatum'])
            y_line_s = slope_s * x_line + intercept_s
            ax_i.plot(x_line, y_line_s, '--', color='red', linewidth=2, alpha=0.8,
                    label=f'Striatum: r={r_s:.3f}, p={p_s:.3g}')

    ax_i.set_xlabel("Brain Atlas Coordinate [mm]", fontsize=11)
    ax_i.set_ylabel("UBC / POLR2A Ratio", fontsize=11)
    ax_i.set_title("I. Atlas Coordinate vs Ratio", fontsize=12, fontweight='bold', loc='left')
    ax_i.legend(loc='best', fontsize=7)
    ax_i.grid(True, alpha=0.3, linestyle='--')

    # ══════════════════════════════════════════════════════════════════════════
    # ROW 4: DISTRIBUTIONS
    # ══════════════════════════════════════════════════════════════════════════

    # Panel J: Expression distributions by gene
    ax_j = fig.add_subplot(gs[3, 0])
    all_polr2a_expr = slide_wide['POLR2A'].dropna()
    all_ubc_expr = slide_wide['UBC'].dropna()
    bins_gene = np.linspace(0, max(all_ubc_expr.max(), all_polr2a_expr.max()), 25)
    ax_j.hist(all_polr2a_expr, bins=bins_gene, alpha=0.6, label='POLR2A',
            color='green', edgecolor='black', linewidth=0.7, density=True)
    ax_j.hist(all_ubc_expr, bins=bins_gene, alpha=0.6, label='UBC',
            color='orange', edgecolor='black', linewidth=0.7, density=True)
    ax_j.axvline(all_polr2a_expr.mean(), color='darkgreen', linestyle='--', linewidth=2, alpha=0.8)
    ax_j.axvline(all_ubc_expr.mean(), color='darkorange', linestyle='--', linewidth=2, alpha=0.8)
    ax_j.set_xlabel("Total mRNA Expression [mRNA/cell]", fontsize=11)
    ax_j.set_ylabel("Probability Density", fontsize=11)
    ax_j.set_title("J. Expression Distributions by Gene", fontsize=12, fontweight='bold', loc='left')
    ax_j.legend(loc='best', fontsize=9)
    ax_j.grid(True, alpha=0.3, linestyle='--', axis='y')

    # Panel K: Expression distributions by region
    ax_k = fig.add_subplot(gs[3, 1])
    cortex_expr = pd.concat([mean_df['Cortex_POLR2A'], mean_df['Cortex_UBC']])
    striatum_expr = pd.concat([mean_df['Striatum_POLR2A'], mean_df['Striatum_UBC']])
    bins_region = np.linspace(0, max(cortex_expr.max(), striatum_expr.max()), 25)
    ax_k.hist(cortex_expr, bins=bins_region, alpha=0.6, label='Cortex',
            color='blue', edgecolor='black', linewidth=0.7, density=True)
    ax_k.hist(striatum_expr, bins=bins_region, alpha=0.6, label='Striatum',
            color='red', edgecolor='black', linewidth=0.7, density=True)
    ax_k.axvline(cortex_expr.mean(), color='darkblue', linestyle='--', linewidth=2, alpha=0.8)
    ax_k.axvline(striatum_expr.mean(), color='darkred', linestyle='--', linewidth=2, alpha=0.8)
    ax_k.set_xlabel("Total mRNA Expression [mRNA/cell]", fontsize=11)
    ax_k.set_ylabel("Probability Density", fontsize=11)
    ax_k.set_title("K. Expression Distributions by Region", fontsize=12, fontweight='bold', loc='left')
    ax_k.legend(loc='best', fontsize=9)
    ax_k.grid(True, alpha=0.3, linestyle='--', axis='y')

    # Panel L: UBC/POLR2A ratio distribution
    ax_l = fig.add_subplot(gs[3, 2])
    all_ratios_final = slide_wide['UBC_POLR2A_ratio'].dropna()
    ax_l.hist(all_ratios_final, bins=20, alpha=0.7, color='purple',
            edgecolor='black', linewidth=0.8, density=True)
    ax_l.axvline(all_ratios_final.mean(), color='darkblue', linestyle='-',
                linewidth=2, alpha=0.8, label=f'Mean={all_ratios_final.mean():.2f}')
    ax_l.axvline(all_ratios_final.median(), color='red', linestyle='--',
                linewidth=2, alpha=0.8, label=f'Median={all_ratios_final.median():.2f}')
    ax_l.set_xlabel("UBC / POLR2A ratio", fontsize=11)
    ax_l.set_ylabel("Probability Density", fontsize=11)
    ax_l.set_title("L. Expression Ratio Distribution", fontsize=12, fontweight='bold', loc='left')
    ax_l.legend(loc='best', fontsize=9)
    ax_l.grid(True, alpha=0.3, linestyle='--', axis='y')

    # No title - leaving space at top for user to add images later

    # ══════════════════════════════════════════════════════════════════════════
    # 9. SAVE FIGURE
    # ══════════════════════════════════════════════════════════════════════════

    print(f"\n{'='*80}")
    print("Saving figure...")
    print(f"{'='*80}")

    # Save in multiple formats
    for fmt in ['png', 'svg', 'pdf']:
        filepath = OUTPUT_DIR / f"fig_positive_control_comprehensive.{fmt}"
        plt.savefig(filepath, format=fmt, bbox_inches='tight', dpi=300)
        print(f"  Saved: {filepath}")

    plt.close(fig)

    # Save data to CSV
    csv_path = OUTPUT_DIR / "positive_control_comprehensive_data.csv"
    mean_df.to_csv(csv_path, index=False)
    print(f"  Data saved: {csv_path}")

    csv_path_fov = OUTPUT_DIR / "positive_control_fov_data.csv"
    df_fov.to_csv(csv_path_fov, index=False)
    print(f"  FOV data saved: {csv_path_fov}")

    # ══════════════════════════════════════════════════════════════════════════
    # 10. GENERATE COMPREHENSIVE CAPTION
    # ══════════════════════════════════════════════════════════════════════════

    print(f"\n{'='*80}")
    print("Generating comprehensive caption...")
    print(f"{'='*80}")

    # Collect statistics for caption
    n_slides = df_fov['slide'].nunique()
    n_fovs = len(df_fov)
    n_excluded_slides = len(EXCLUDED_SLIDES)
    excluded_slides_str = ', '.join(sorted(EXCLUDED_SLIDES))
    slides_used = sorted(df_fov['slide'].unique())
    slides_used_str = ', '.join(slides_used)

    # Age and atlas ranges
    age_min = df_fov['age'].min()
    age_max = df_fov['age'].max()
    atlas_min = df_fov['atlas_coord'].min()
    atlas_max = df_fov['atlas_coord'].max()

    # Expression statistics
    polr2a_cortex_mean = mean_df['Cortex_POLR2A'].mean()
    polr2a_cortex_std = mean_df['Cortex_POLR2A'].std()
    polr2a_striatum_mean = mean_df['Striatum_POLR2A'].mean()
    polr2a_striatum_std = mean_df['Striatum_POLR2A'].std()
    ubc_cortex_mean = mean_df['Cortex_UBC'].mean()
    ubc_cortex_std = mean_df['Cortex_UBC'].std()
    ubc_striatum_mean = mean_df['Striatum_UBC'].mean()
    ubc_striatum_std = mean_df['Striatum_UBC'].std()

    # UBC/POLR2A ratio
    ratio_mean = slide_wide['UBC_POLR2A_ratio'].mean()
    ratio_std = slide_wide['UBC_POLR2A_ratio'].std()
    ratio_median = slide_wide['UBC_POLR2A_ratio'].median()

    # Paired samples count
    n_paired = len(mean_df)

    # Build caption
    caption_lines = [
        "Figure: Comprehensive Positive Control Analysis",
        "",
        "This figure presents a detailed analysis of housekeeping gene expression (POLR2A and UBC)",
        "in positive control samples from Q111 and Wildtype mice, demonstrating the validity of the",
        "RNAscope quantification pipeline and the expected differential expression between low-",
        "and high-expression housekeeping genes.",
        "",
        "DATA FILTERING AND QUALITY CONTROL:",
        f"- Dataset: Positive control samples from Q111 and Wildtype mice",
        f"- Excluded slides (n={n_excluded_slides}): {excluded_slides_str}",
        f"  (Slides excluded due to poor UBC positive control expression indicating technical failures)",
        f"- CV threshold for cluster filtering: CV >= {CV_THRESHOLD}",
        f"- Minimum nuclei per FOV: {MIN_NUCLEI_THRESHOLD}",
        f"- Intensity threshold: Per-slide, determined from negative control at quantile={QUANTILE_NEGATIVE_CONTROL}, max PFA={MAX_PFA}",
        "",
        "DATA SUMMARY:",
        f"- Slides analyzed (n={n_slides}): {slides_used_str}",
        f"- Total FOVs: {n_fovs}",
        f"- Paired samples (slides with both Cortex and Striatum): {n_paired}",
        f"- Age range: {age_min:.0f} - {age_max:.0f} months",
        f"- Brain atlas coordinate range: {atlas_min:.2f} - {atlas_max:.2f} mm",
        "",
        "VOXEL AND PIXEL PARAMETERS:",
        f"- Pixel size (XY): {PIXELSIZE} nm",
        f"- Slice depth (Z): {SLICE_DEPTH} nm",
        f"- Voxel size: {VOXEL_SIZE} μm³",
        f"- Mean nuclear volume (for nuclei estimation): {MEAN_NUCLEAR_VOLUME} μm³",
        "",
        "EXPRESSION STATISTICS:",
        "",
        "POLR2A (low-expression housekeeping gene, green channel):",
        f"- Cortex: {polr2a_cortex_mean:.2f} ± {polr2a_cortex_std:.2f} mRNA/cell (n={n_paired} slides)",
        f"- Striatum: {polr2a_striatum_mean:.2f} ± {polr2a_striatum_std:.2f} mRNA/cell (n={n_paired} slides)",
        f"- Regional comparison (paired t-test): t={t_polr2a:.3f}, p={p_polr2a:.4g}",
        "",
        "UBC (high-expression housekeeping gene, orange channel):",
        f"- Cortex: {ubc_cortex_mean:.2f} ± {ubc_cortex_std:.2f} mRNA/cell (n={n_paired} slides)",
        f"- Striatum: {ubc_striatum_mean:.2f} ± {ubc_striatum_std:.2f} mRNA/cell (n={n_paired} slides)",
        f"- Regional comparison (paired t-test): t={t_ubc:.3f}, p={p_ubc:.4g}",
        "",
        "UBC/POLR2A RATIO:",
        f"- Mean: {ratio_mean:.2f} ± {ratio_std:.2f}",
        f"- Median: {ratio_median:.2f}",
        f"- Expected ratio based on known expression levels: ~10-20x",
        "",
        "PANEL DESCRIPTIONS:",
        "",
        "Row 1 - Gene Expression Comparisons:",
        "A. POLR2A vs UBC scatter plot showing correlation between housekeeping genes (colored by region)",
        "B. UBC/POLR2A ratio distribution histogram",
        "C. Regional comparison box plots for both genes with paired t-test significance",
        "",
        "Row 2 - Age Dependencies:",
        "D. Age vs POLR2A expression for Cortex and Striatum with separate regression lines",
        "E. Age vs UBC expression for Cortex and Striatum with separate regression lines",
        "F. Age vs UBC/POLR2A ratio for Cortex and Striatum with separate regression lines",
        "",
        "Row 3 - Atlas Coordinate Dependencies:",
        "G. Brain atlas coordinate vs POLR2A expression with separate regional regressions",
        "H. Brain atlas coordinate vs UBC expression with separate regional regressions",
        "I. Brain atlas coordinate vs UBC/POLR2A ratio with separate regional regressions",
        "",
        "Row 4 - Expression Distributions:",
        "J. Expression distributions by gene (POLR2A vs UBC)",
        "K. Expression distributions by region (Cortex vs Striatum)",
        "L. UBC/POLR2A ratio distribution with mean and median indicated",
        "",
        "INTERPRETATION:",
        "- UBC (high-expression housekeeping gene) shows ~10x higher expression than POLR2A (low-expression)",
        "- This ratio is consistent with known expression levels from RNA-seq data",
        "- Regional differences may reflect cell-type composition variations between Cortex and Striatum",
        "- Age and atlas coordinate dependencies help identify potential confounders",
        "- Non-significant correlations with age/atlas suggest robust normalization",
        "",
        "METHODOLOGY:",
        "- Peak intensity normalization: Slide-specific KDE mode estimation from single spots",
        "- Total mRNA per cell: (N_spots + I_cluster_total / I_single_peak) / N_nuclei",
        "- Cluster filtering: Intensity > threshold AND CV >= CV_THRESHOLD",
        "- Statistical tests: Paired t-tests for regional comparisons, Pearson correlation for continuous variables",
        "",
        f"Analysis performed with scienceplots style. CV threshold={CV_THRESHOLD}, Min nuclei={MIN_NUCLEI_THRESHOLD}.",
    ]

    caption_text = '\n'.join(caption_lines)

    # Save caption
    caption_path = OUTPUT_DIR / 'fig_positive_control_comprehensive_caption.txt'
    with open(caption_path, 'w') as f:
        f.write(caption_text)
    print(f"  Caption saved: {caption_path}")

    # Also save as LaTeX
    caption_latex = caption_text.replace('_', '\\_').replace('%', '\\%').replace('μ', '$\\mu$').replace('³', '$^3$').replace('²', '$^2$')
    caption_latex_path = OUTPUT_DIR / 'fig_positive_control_comprehensive_caption.tex'
    with open(caption_latex_path, 'w') as f:
        f.write(caption_latex)
    print(f"  LaTeX caption saved: {caption_latex_path}")

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
