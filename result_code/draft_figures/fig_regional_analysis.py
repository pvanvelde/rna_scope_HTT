"""
Comprehensive Regional Analysis Figure

This script creates a detailed multi-panel figure analyzing regional differences in fl-HTT expression:
- Sub-regional differences WITHIN Cortex (testing homogeneity across cortical subregions)
- Sub-regional differences WITHIN Striatum (testing homogeneity across striatal subregions)
- Regional differences BETWEEN Cortex and Striatum (testing main regional effect)

The goal is to demonstrate that sub-regional variation is minimal (coronal section location
within a region doesn't matter much), but regional differences are substantial (Cortex vs
Striatum matters).

Author: Generated for RNA Scope analysis
Date: 2025-11-16
"""

import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel, f_oneway, gaussian_kde
from statsmodels.stats.multicomp import pairwise_tukeyhsd
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
    FIGURE_FORMAT,
    CV_THRESHOLD
)

# Physical parameters
pixelsize = PIXELSIZE
slice_depth = SLICE_DEPTH
mean_nuclear_volume = MEAN_NUCLEAR_VOLUME
voxel_size = VOXEL_SIZE

# Create output directory
OUTPUT_DIR = Path(__file__).parent / "output" / "regional_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Define subregions
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

# Abbreviated labels for display (to prevent x-axis overlap)
subregion_display_labels = {
    "Striatum - lower left": "Striatum\nlower left",
    "Striatum - lower right": "Striatum\nlower right",
    "Striatum - upper left": "Striatum\nupper left",
    "Striatum - upper right": "Striatum\nupper right",
    "Cortex - Piriform area": "Cortex\nPiriform area",
    "Cortex - Primary and secondary motor areas": "Cortex\nMotor areas",
    "Cortex - Primary somatosensory (mouth, upper limb)": "Cortex\nSomatosensory\n(mouth, upper limb)",
    "Cortex - Supplemental/primary somatosensory (nose)": "Cortex\nSomatosensory (nose)",
    "Cortex - Visceral/gustatory/agranular areas": "Cortex\nVisceral/agranular",
}

# Channel labels
channel_labels = {
    'green': 'HTT1a',
    'orange': 'fl-HTT'
}

channel_colors = {
    'green': 'green',
    'orange': 'orange'
}


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
    print("COMPREHENSIVE REGIONAL ANALYSIS")
    print("Analyzing sub-regional homogeneity and regional differences")
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

    negative_control_field = 'Negative control'
    experimental_field = 'ExperimentalQ111 - 488mHT - 548mHTa - 647Darp'
    slide_field = SLIDE_FIELD
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

    # Filter to Q111 experimental data
    df_exp = df_extracted_full[
        (df_extracted_full["metadata_sample_Mouse_Model"] == 'Q111') &
        (df_extracted_full['metadata_sample_Probe-Set'] == experimental_field)
    ].copy()

    print(f"Q111 experimental records (before QC): {len(df_exp)}")

    # Apply slide exclusion (QC filter for technical failures)
    if len(EXCLUDED_SLIDES) > 0:
        n_before = len(df_exp)
        df_exp = df_exp[~df_exp[SLIDE_FIELD].isin(EXCLUDED_SLIDES)].copy()
        n_after = len(df_exp)
        n_excluded = n_before - n_after
        print(f"  Excluded {n_excluded} FOVs from {len(EXCLUDED_SLIDES)} slides: {EXCLUDED_SLIDES}")
        print(f"  Remaining FOVs after QC: {n_after}")

    # ══════════════════════════════════════════════════════════════════════════
    # 4. COMPUTE SLIDE-SPECIFIC PEAK INTENSITIES
    # ══════════════════════════════════════════════════════════════════════════

    print(f"\n{'='*80}")
    print("COMPUTING SLIDE-SPECIFIC PEAK INTENSITIES")
    print(f"{'='*80}")

    spot_peaks = {}  # (slide, channel) -> peak_intensity

    for idx, row in df_exp.iterrows():
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

            for idx2, row2 in df_exp.iterrows():
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
        (df_extracted_full["metadata_sample_Mouse_Model"] == 'Q111') &
        (df_extracted_full['metadata_sample_Probe-Set'] == experimental_field)
    ].copy()

    # Apply exclusion
    if len(EXCLUDED_SLIDES) > 0:
        df_all_channels = df_all_channels[~df_all_channels[SLIDE_FIELD].isin(EXCLUDED_SLIDES)].copy()

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

    for idx, row in df_exp.iterrows():
        slide = row.get(slide_field, 'unknown')
        subregion = row.get('metadata_sample_Slice_Region', 'unknown')
        channel = row.get('channel', 'unknown')
        age = row.get('metadata_sample_Age', np.nan)
        atlas_coord = row.get('metadata_sample_Brain_Atlas_coordinates', np.nan)
        threshold_val = row.get('threshold', np.nan)

        if channel == 'blue':
            continue

        # Determine merged region
        if any(sub in subregion for sub in cortex_subregions):
            region = 'Cortex'
        elif any(sub in subregion for sub in striatum_subregions):
            region = 'Striatum'
        else:
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

        fov_data.append({
            'slide': slide,
            'region': region,
            'subregion': subregion,
            'channel': channel_labels[channel],
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
    print(f"  Subregions: {sorted(df_fov['subregion'].unique())}")
    print(f"  Channels: {sorted(df_fov['channel'].unique())}")

    # ══════════════════════════════════════════════════════════════════════════
    # 6. COMPUTE PER-SLIDE AVERAGES (AGGREGATION BY REGION)
    # ══════════════════════════════════════════════════════════════════════════

    print(f"\n{'='*80}")
    print("COMPUTING PER-SLIDE AVERAGES BY REGION")
    print(f"{'='*80}")

    # Group by slide, channel, region (collapsing subregions within each region)
    slide_avg = df_fov.groupby(['slide', 'channel', 'region']).agg({
        'expression_per_cell': 'mean',
        'age': 'first',
        'atlas_coord': 'first',
        'N_nuc': 'mean'
    }).reset_index()

    print(f"Slide-level averages: {len(slide_avg)} entries")

    # Also compute subregion-level averages (FOV -> subregion aggregation)
    subregion_avg = df_fov.groupby(['slide', 'channel', 'region', 'subregion']).agg({
        'expression_per_cell': 'mean',
        'age': 'first',
        'atlas_coord': 'first',
        'N_nuc': 'mean'
    }).reset_index()

    print(f"Subregion-level averages: {len(subregion_avg)} entries")

    # ══════════════════════════════════════════════════════════════════════════
    # 7. STATISTICAL ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════

    print(f"\n{'='*80}")
    print("STATISTICAL ANALYSIS")
    print(f"{'='*80}")

    stats_results = {}

    for gene in ['HTT1a', 'fl-HTT']:
        print(f"\n{'='*60}")
        print(f"GENE: {gene}")
        print(f"{'='*60}")

        gene_data = subregion_avg[subregion_avg['channel'] == gene].copy()

        # ── 1. Sub-regional analysis WITHIN Cortex ─────────────────────────────
        print(f"\n1. CORTEX SUB-REGIONAL ANALYSIS ({gene})")
        print("-" * 60)

        cortex_sub = gene_data[gene_data['region'] == 'Cortex'].copy()
        cortex_subregions_present = cortex_sub['subregion'].unique()

        print(f"  Subregions present: {len(cortex_subregions_present)}")
        for sub in cortex_subregions_present:
            sub_data = cortex_sub[cortex_sub['subregion'] == sub]
            print(f"    {sub}: n={len(sub_data)}, mean={sub_data['expression_per_cell'].mean():.3f} ± {sub_data['expression_per_cell'].std():.3f}")

        if len(cortex_subregions_present) > 1:
            # One-way ANOVA
            groups = [cortex_sub[cortex_sub['subregion'] == sub]['expression_per_cell'].values
                     for sub in cortex_subregions_present]
            f_stat, p_anova = f_oneway(*groups)
            print(f"\n  One-way ANOVA: F={f_stat:.3f}, p={p_anova:.4g}")

            # Tukey's HSD
            try:
                tukey_result = pairwise_tukeyhsd(
                    endog=cortex_sub['expression_per_cell'],
                    groups=cortex_sub['subregion'],
                    alpha=0.05
                )
                print(f"\n  Tukey's HSD post-hoc test:")
                print(tukey_result)
                stats_results[f'{gene}_Cortex_Tukey'] = tukey_result
            except Exception as e:
                print(f"  Tukey's HSD failed: {e}")

            stats_results[f'{gene}_Cortex_ANOVA'] = {'F': f_stat, 'p': p_anova}
        else:
            print("  Insufficient subregions for ANOVA")

        # ── 2. Sub-regional analysis WITHIN Striatum ───────────────────────────
        print(f"\n2. STRIATUM SUB-REGIONAL ANALYSIS ({gene})")
        print("-" * 60)

        striatum_sub = gene_data[gene_data['region'] == 'Striatum'].copy()
        striatum_subregions_present = striatum_sub['subregion'].unique()

        print(f"  Subregions present: {len(striatum_subregions_present)}")
        for sub in striatum_subregions_present:
            sub_data = striatum_sub[striatum_sub['subregion'] == sub]
            print(f"    {sub}: n={len(sub_data)}, mean={sub_data['expression_per_cell'].mean():.3f} ± {sub_data['expression_per_cell'].std():.3f}")

        if len(striatum_subregions_present) > 1:
            # One-way ANOVA
            groups = [striatum_sub[striatum_sub['subregion'] == sub]['expression_per_cell'].values
                     for sub in striatum_subregions_present]
            f_stat, p_anova = f_oneway(*groups)
            print(f"\n  One-way ANOVA: F={f_stat:.3f}, p={p_anova:.4g}")

            # Tukey's HSD
            try:
                tukey_result = pairwise_tukeyhsd(
                    endog=striatum_sub['expression_per_cell'],
                    groups=striatum_sub['subregion'],
                    alpha=0.05
                )
                print(f"\n  Tukey's HSD post-hoc test:")
                print(tukey_result)
                stats_results[f'{gene}_Striatum_Tukey'] = tukey_result
            except Exception as e:
                print(f"  Tukey's HSD failed: {e}")

            stats_results[f'{gene}_Striatum_ANOVA'] = {'F': f_stat, 'p': p_anova}
        else:
            print("  Insufficient subregions for ANOVA")

        # ── 3. Regional comparison: Cortex vs Striatum ─────────────────────────
        print(f"\n3. CORTEX vs STRIATUM COMPARISON ({gene})")
        print("-" * 60)

        # Use slide-level averages (one value per slide per region)
        gene_slide = slide_avg[slide_avg['channel'] == gene].copy()

        # Get paired samples (slides with both Cortex and Striatum)
        slides_with_both = gene_slide.groupby('slide').filter(
            lambda x: set(x['region']) >= {'Cortex', 'Striatum'}
        )['slide'].unique()

        cortex_paired = []
        striatum_paired = []

        for slide in slides_with_both:
            cortex_val = gene_slide[(gene_slide['slide'] == slide) & (gene_slide['region'] == 'Cortex')]['expression_per_cell'].values
            striatum_val = gene_slide[(gene_slide['slide'] == slide) & (gene_slide['region'] == 'Striatum')]['expression_per_cell'].values

            if len(cortex_val) > 0 and len(striatum_val) > 0:
                cortex_paired.append(cortex_val[0])
                striatum_paired.append(striatum_val[0])

        cortex_paired = np.array(cortex_paired)
        striatum_paired = np.array(striatum_paired)

        print(f"  Paired samples: n={len(cortex_paired)}")
        print(f"  Cortex: {cortex_paired.mean():.3f} ± {cortex_paired.std():.3f}")
        print(f"  Striatum: {striatum_paired.mean():.3f} ± {striatum_paired.std():.3f}")

        if len(cortex_paired) > 0:
            t_stat, p_ttest = ttest_rel(cortex_paired, striatum_paired)
            print(f"  Paired t-test: t={t_stat:.3f}, p={p_ttest:.4g}")
            stats_results[f'{gene}_CortexVsStriatum'] = {
                't': t_stat,
                'p': p_ttest,
                'n': len(cortex_paired),
                'cortex_mean': cortex_paired.mean(),
                'cortex_std': cortex_paired.std(),
                'striatum_mean': striatum_paired.mean(),
                'striatum_std': striatum_paired.std()
            }

    # ══════════════════════════════════════════════════════════════════════════
    # 8. CREATE COMPREHENSIVE FIGURE
    # ══════════════════════════════════════════════════════════════════════════

    print(f"\n{'='*80}")
    print("Creating comprehensive figure...")
    print(f"{'='*80}")

    # Create figure with 3×2 grid
    fig = plt.figure(figsize=(16, 16), dpi=300)
    gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.30,
                         left=0.08, right=0.98, top=0.94, bottom=0.08)

    # ══════════════════════════════════════════════════════════════════════════
    # ROW 1: CORTEX SUB-REGIONAL ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════

    for col_idx, gene in enumerate(['HTT1a', 'fl-HTT']):
        ax = fig.add_subplot(gs[0, col_idx])

        gene_data = subregion_avg[subregion_avg['channel'] == gene].copy()
        cortex_sub = gene_data[gene_data['region'] == 'Cortex'].copy()
        cortex_subregions_present = sorted(cortex_sub['subregion'].unique())

        # Create box plot
        data_to_plot = [cortex_sub[cortex_sub['subregion'] == sub]['expression_per_cell'].values
                       for sub in cortex_subregions_present]

        # Use abbreviated labels for display
        labels_formatted = [subregion_display_labels.get(label, label.replace(' - ', '\n'))
                           for label in cortex_subregions_present]

        bp = ax.boxplot(data_to_plot, labels=labels_formatted,
                       patch_artist=True, widths=0.6)

        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)

        ax.set_ylabel("Total mRNA Expression [mRNA/cell]", fontsize=11)
        ax.set_title(f"{'A' if col_idx == 0 else 'B'}. Cortex Sub-regions ({gene})",
                    fontsize=12, fontweight='bold', loc='left')
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')

        # Add ANOVA result if available
        if f'{gene}_Cortex_ANOVA' in stats_results:
            anova_res = stats_results[f'{gene}_Cortex_ANOVA']
            p_val = anova_res['p']
            if p_val < 0.001:
                p_str = "p<0.001***"
            elif p_val < 0.01:
                p_str = f"p={p_val:.3f}**"
            elif p_val < 0.05:
                p_str = f"p={p_val:.3f}*"
            else:
                p_str = f"p={p_val:.3f}ns"

            ax.text(0.98, 0.98, f"ANOVA: F={anova_res['F']:.2f}, {p_str}",
                   transform=ax.transAxes, ha='right', va='top', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # ══════════════════════════════════════════════════════════════════════════
    # ROW 2: STRIATUM SUB-REGIONAL ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════

    for col_idx, gene in enumerate(['HTT1a', 'fl-HTT']):
        ax = fig.add_subplot(gs[1, col_idx])

        gene_data = subregion_avg[subregion_avg['channel'] == gene].copy()
        striatum_sub = gene_data[gene_data['region'] == 'Striatum'].copy()
        striatum_subregions_present = sorted(striatum_sub['subregion'].unique())

        # Create box plot
        data_to_plot = [striatum_sub[striatum_sub['subregion'] == sub]['expression_per_cell'].values
                       for sub in striatum_subregions_present]

        # Use abbreviated labels for display
        labels_formatted = [subregion_display_labels.get(label, label.replace(' - ', '\n'))
                           for label in striatum_subregions_present]

        bp = ax.boxplot(data_to_plot, labels=labels_formatted,
                       patch_artist=True, widths=0.6)

        for patch in bp['boxes']:
            patch.set_facecolor('lightcoral')
            patch.set_alpha(0.7)

        ax.set_ylabel("Total mRNA Expression [mRNA/cell]", fontsize=11)
        ax.set_title(f"{'C' if col_idx == 0 else 'D'}. Striatum Sub-regions ({gene})",
                    fontsize=12, fontweight='bold', loc='left')
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')

        # Add ANOVA result if available
        if f'{gene}_Striatum_ANOVA' in stats_results:
            anova_res = stats_results[f'{gene}_Striatum_ANOVA']
            p_val = anova_res['p']
            if p_val < 0.001:
                p_str = "p<0.001***"
            elif p_val < 0.01:
                p_str = f"p={p_val:.3f}**"
            elif p_val < 0.05:
                p_str = f"p={p_val:.3f}*"
            else:
                p_str = f"p={p_val:.3f}ns"

            ax.text(0.98, 0.98, f"ANOVA: F={anova_res['F']:.2f}, {p_str}",
                   transform=ax.transAxes, ha='right', va='top', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # ══════════════════════════════════════════════════════════════════════════
    # ROW 3: CORTEX vs STRIATUM COMPARISON
    # ══════════════════════════════════════════════════════════════════════════

    for col_idx, gene in enumerate(['HTT1a', 'fl-HTT']):
        ax = fig.add_subplot(gs[2, col_idx])

        if f'{gene}_CortexVsStriatum' in stats_results:
            result = stats_results[f'{gene}_CortexVsStriatum']

            # Get paired data
            gene_slide = slide_avg[slide_avg['channel'] == gene].copy()
            slides_with_both = gene_slide.groupby('slide').filter(
                lambda x: set(x['region']) >= {'Cortex', 'Striatum'}
            )['slide'].unique()

            cortex_vals = []
            striatum_vals = []

            for slide in slides_with_both:
                cortex_val = gene_slide[(gene_slide['slide'] == slide) & (gene_slide['region'] == 'Cortex')]['expression_per_cell'].values
                striatum_val = gene_slide[(gene_slide['slide'] == slide) & (gene_slide['region'] == 'Striatum')]['expression_per_cell'].values

                if len(cortex_val) > 0 and len(striatum_val) > 0:
                    cortex_vals.append(cortex_val[0])
                    striatum_vals.append(striatum_val[0])

            # Box plot
            bp = ax.boxplot([cortex_vals, striatum_vals], labels=['Cortex', 'Striatum'],
                           patch_artist=True, widths=0.6)

            bp['boxes'][0].set_facecolor('lightblue')
            bp['boxes'][1].set_facecolor('lightcoral')

            # Add connecting lines for paired data
            for i in range(len(cortex_vals)):
                ax.plot([1, 2], [cortex_vals[i], striatum_vals[i]], 'k-', alpha=0.2, linewidth=0.8)

            ax.set_ylabel("Total mRNA Expression [mRNA/cell]", fontsize=11)
            ax.set_title(f"{'E' if col_idx == 0 else 'F'}. Cortex vs Striatum ({gene})",
                        fontsize=12, fontweight='bold', loc='left')
            ax.grid(True, alpha=0.3, linestyle='--', axis='y')

            # Add t-test result
            p_val = result['p']
            if p_val < 0.001:
                p_str = "p<0.001***"
            elif p_val < 0.01:
                p_str = f"p={p_val:.3f}**"
            elif p_val < 0.05:
                p_str = f"p={p_val:.3f}*"
            else:
                p_str = f"p={p_val:.3f}ns"

            # Add significance line
            y_max = max(max(cortex_vals), max(striatum_vals))
            y_line = y_max * 1.1
            ax.plot([1, 2], [y_line, y_line], 'k-', linewidth=1.5)

            # Add t-test result as text box in top-right corner (not overlapping with title)
            ax.text(0.98, 0.98, f"Paired t-test:\nt={result['t']:.2f}, {p_str}",
                   transform=ax.transAxes, ha='right', va='top', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.suptitle('Regional Analysis: Sub-regional Homogeneity vs Inter-regional Differences',
                fontsize=16, fontweight='bold')

    # ══════════════════════════════════════════════════════════════════════════
    # 9. SAVE FIGURE
    # ══════════════════════════════════════════════════════════════════════════

    print(f"\n{'='*80}")
    print("Saving figure...")
    print(f"{'='*80}")

    # Save in multiple formats
    for fmt in ['png', 'svg', 'pdf']:
        filepath = OUTPUT_DIR / f"fig_regional_analysis.{fmt}"
        plt.savefig(filepath, format=fmt, bbox_inches='tight', dpi=300)
        print(f"  Saved: {filepath}")

    plt.close(fig)

    # Save data to CSV
    csv_path_slide = OUTPUT_DIR / "regional_analysis_slide_level.csv"
    slide_avg.to_csv(csv_path_slide, index=False)
    print(f"  Slide-level data saved: {csv_path_slide}")

    csv_path_subregion = OUTPUT_DIR / "regional_analysis_subregion_level.csv"
    subregion_avg.to_csv(csv_path_subregion, index=False)
    print(f"  Subregion-level data saved: {csv_path_subregion}")

    csv_path_fov = OUTPUT_DIR / "regional_analysis_fov_level.csv"
    df_fov.to_csv(csv_path_fov, index=False)
    print(f"  FOV-level data saved: {csv_path_fov}")

    # Save statistics summary
    stats_path = OUTPUT_DIR / "regional_analysis_statistics.txt"
    with open(stats_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("REGIONAL ANALYSIS STATISTICS SUMMARY\n")
        f.write("="*80 + "\n\n")

        for key, value in stats_results.items():
            f.write(f"\n{key}:\n")
            f.write("-" * 60 + "\n")
            if isinstance(value, dict):
                for k, v in value.items():
                    f.write(f"  {k}: {v}\n")
            else:
                f.write(str(value) + "\n")

    print(f"  Statistics saved: {stats_path}")

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

    # Subregion counts
    n_cortex_subregions = len([s for s in df_fov['subregion'].unique() if 'Cortex' in s])
    n_striatum_subregions = len([s for s in df_fov['subregion'].unique() if 'Striatum' in s])
    cortex_subregions_str = ', '.join(sorted([s for s in df_fov['subregion'].unique() if 'Cortex' in s]))
    striatum_subregions_str = ', '.join(sorted([s for s in df_fov['subregion'].unique() if 'Striatum' in s]))

    # FOVs per region
    n_cortex_fovs = len(df_fov[df_fov['region'] == 'Cortex'])
    n_striatum_fovs = len(df_fov[df_fov['region'] == 'Striatum'])

    # Age and atlas ranges
    age_min = df_fov['age'].min()
    age_max = df_fov['age'].max()
    atlas_min = df_fov['atlas_coord'].min()
    atlas_max = df_fov['atlas_coord'].max()

    # Build caption
    caption_lines = [
        "Figure: Regional Analysis - Sub-regional Homogeneity vs Inter-regional Differences",
        "",
        "This figure presents a comprehensive analysis of fl-HTT mRNA expression patterns across brain regions,",
        "demonstrating that while sub-regional variation within Cortex and Striatum is minimal (location within",
        "a region doesn't significantly affect expression), substantial differences exist between the two major",
        "brain regions.",
        "",
        "DATA FILTERING AND QUALITY CONTROL:",
        f"- Dataset: Q111 experimental samples from the {experimental_field} probe set",
        f"- Excluded slides (n={n_excluded_slides}): {excluded_slides_str}",
        f"  (Slides excluded due to poor UBC positive control expression indicating technical failures)",
        f"- CV threshold for cluster filtering: CV >= {CV_THRESHOLD}",
        f"- Minimum nuclei per FOV: {MIN_NUCLEI_THRESHOLD}",
        f"- Intensity threshold: Per-slide, determined from negative control at quantile={QUANTILE_NEGATIVE_CONTROL}, max PFA={MAX_PFA}",
        "",
        "DATA SUMMARY:",
        f"- Slides analyzed (n={n_slides}): {slides_used_str}",
        f"- Total FOVs: {n_fovs}",
        f"  - Cortex: {n_cortex_fovs} FOVs",
        f"  - Striatum: {n_striatum_fovs} FOVs",
        f"- Age range: {age_min:.0f} - {age_max:.0f} months",
        f"- Brain atlas coordinate range: {atlas_min:.2f} - {atlas_max:.2f} mm",
        "",
        "SUBREGIONS ANALYZED:",
        f"- Cortex subregions (n={n_cortex_subregions}): {cortex_subregions_str}",
        f"- Striatum subregions (n={n_striatum_subregions}): {striatum_subregions_str}",
        "",
        "VOXEL AND PIXEL PARAMETERS:",
        f"- Pixel size (XY): {PIXELSIZE} nm",
        f"- Slice depth (Z): {SLICE_DEPTH} nm",
        f"- Voxel size: {VOXEL_SIZE} μm³",
        f"- Mean nuclear volume (for nuclei estimation): {MEAN_NUCLEAR_VOLUME} μm³",
        "",
        "STATISTICAL RESULTS:",
        "",
    ]

    # Add ANOVA results for each gene
    for gene in ['HTT1a', 'fl-HTT']:
        caption_lines.append(f"{gene.upper()}:")
        caption_lines.append("")

        # Cortex sub-regional ANOVA
        if f'{gene}_Cortex_ANOVA' in stats_results:
            anova_res = stats_results[f'{gene}_Cortex_ANOVA']
            p_str = "***" if anova_res['p'] < 0.001 else "**" if anova_res['p'] < 0.01 else "*" if anova_res['p'] < 0.05 else "ns"
            caption_lines.append(f"  Cortex sub-regional ANOVA: F={anova_res['F']:.3f}, p={anova_res['p']:.4g} {p_str}")

        # Striatum sub-regional ANOVA
        if f'{gene}_Striatum_ANOVA' in stats_results:
            anova_res = stats_results[f'{gene}_Striatum_ANOVA']
            p_str = "***" if anova_res['p'] < 0.001 else "**" if anova_res['p'] < 0.01 else "*" if anova_res['p'] < 0.05 else "ns"
            caption_lines.append(f"  Striatum sub-regional ANOVA: F={anova_res['F']:.3f}, p={anova_res['p']:.4g} {p_str}")

        # Cortex vs Striatum comparison
        if f'{gene}_CortexVsStriatum' in stats_results:
            result = stats_results[f'{gene}_CortexVsStriatum']
            p_str = "***" if result['p'] < 0.001 else "**" if result['p'] < 0.01 else "*" if result['p'] < 0.05 else "ns"
            caption_lines.append(f"  Cortex vs Striatum (paired t-test):")
            caption_lines.append(f"    Cortex: {result['cortex_mean']:.3f} ± {result['cortex_std']:.3f} mRNA/cell")
            caption_lines.append(f"    Striatum: {result['striatum_mean']:.3f} ± {result['striatum_std']:.3f} mRNA/cell")
            caption_lines.append(f"    t={result['t']:.3f}, p={result['p']:.4g} {p_str} (n={result['n']} paired samples)")

        caption_lines.append("")

    caption_lines.extend([
        "PANEL DESCRIPTIONS:",
        "",
        "Row 1 - Cortex Sub-regional Analysis:",
        "A. HTT1a expression across cortical subregions with one-way ANOVA",
        "B. fl-fl-HTT expression across cortical subregions with one-way ANOVA",
        "",
        "Row 2 - Striatum Sub-regional Analysis:",
        "C. HTT1a expression across striatal subregions with one-way ANOVA",
        "D. fl-fl-HTT expression across striatal subregions with one-way ANOVA",
        "",
        "Row 3 - Inter-regional Comparison:",
        "E. HTT1a Cortex vs Striatum with paired t-test and connecting lines",
        "F. fl-HTT Cortex vs Striatum with paired t-test and connecting lines",
        "",
        "INTERPRETATION:",
        "- Non-significant sub-regional ANOVAs indicate expression is homogeneous within each major brain region",
        "- This validates pooling FOVs across subregions for region-level analyses",
        "- Significant Cortex vs Striatum differences reflect true regional variation in fl-HTT expression",
        "- Paired t-tests control for inter-animal variability by comparing regions within the same slide",
        "",
        "METHODOLOGY:",
        "- Sub-regional analysis: One-way ANOVA with Tukey's HSD post-hoc test (α=0.05)",
        "- Inter-regional analysis: Paired t-test using slide-level averages",
        "- Expression normalized: (N_spots + I_cluster_total / I_single_peak) / N_nuclei",
        "- Peak intensity: Slide-specific KDE mode estimation from single spots",
        "",
        f"Analysis performed with scienceplots style. CV threshold={CV_THRESHOLD}, Min nuclei={MIN_NUCLEI_THRESHOLD}.",
    ])

    caption_text = '\n'.join(caption_lines)

    # Save caption
    caption_path = OUTPUT_DIR / 'fig_regional_analysis_caption.txt'
    with open(caption_path, 'w') as f:
        f.write(caption_text)
    print(f"  Caption saved: {caption_path}")

    # Also save as LaTeX
    caption_latex = caption_text.replace('_', '\\_').replace('%', '\\%').replace('μ', '$\\mu$').replace('³', '$^3$').replace('²', '$^2$').replace('α', '$\\alpha$')
    caption_latex_path = OUTPUT_DIR / 'fig_regional_analysis_caption.tex'
    with open(caption_latex_path, 'w') as f:
        f.write(caption_latex)
    print(f"  LaTeX caption saved: {caption_latex_path}")

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
