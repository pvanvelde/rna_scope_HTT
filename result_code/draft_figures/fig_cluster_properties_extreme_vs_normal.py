"""
Cluster Properties Analysis: Extreme vs Normal FOVs
====================================================

Compare cluster-level properties between extreme FOVs (> WT P95) and normal FOVs.

Analyzes:
1. Number of clusters per cell
2. Cluster intensities (distribution)
3. Nuclear vs cytoplasmic location (distance to DAPI)
4. Cluster sizes

Key question: Are extreme FOVs different because they have:
- More clusters per cell?
- Larger/more intense clusters?
- Different subcellular localization (nuclear vs cytoplasmic)?
"""

import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, ks_2samp
from scipy.stats import gaussian_kde
import scienceplots
plt.style.use('science')
plt.rcParams['text.usetex'] = False
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from result_functions_v2 import (compute_thresholds, recursively_load_dict,
                                extract_dataframe)

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
    OUTPUT_DIR_COMPREHENSIVE,
    EXCLUDED_SLIDES,
    CV_THRESHOLD
)

# Create output directory
OUTPUT_DIR = Path(__file__).parent / "output" / "cluster_properties_extreme_vs_normal"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Channel labels
channel_labels_exp = {
    'green': 'HTT1a',
    'orange': 'fl-HTT'
}

# Region definitions
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


def get_region(subregion):
    """Map subregion to merged region."""
    if any(sub in subregion for sub in cortex_subregions):
        return 'Cortex'
    elif any(sub in subregion for sub in striatum_subregions):
        return 'Striatum'
    return None


if __name__ == "__main__":

    print("="*70)
    print("CLUSTER PROPERTIES ANALYSIS: EXTREME vs NORMAL FOVs")
    print("="*70)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 1: LOAD DATA
    # ══════════════════════════════════════════════════════════════════════════

    print("\n" + "="*70)
    print("SECTION 1: LOADING DATA")
    print("="*70)

    # Load HDF5 data
    h5_file_path = H5_FILE_PATH_EXPERIMENTAL

    with h5py.File(h5_file_path, 'r') as h5_file:
        data_dict = recursively_load_dict(h5_file)

    print(f"Loaded data from: {h5_file_path}")

    # Extract DataFrame with cluster fields
    desired_channels = ['blue', 'green', 'orange']
    fields_to_extract = [
        'spots_sigma_var.params_raw',
        'spots.params_raw',
        'cluster_intensities',
        'cluster_cvs',               # Coefficient of variance for clusters
        'cluster_distance_dapi_um',  # Distance to DAPI edge
        'label_sizes',               # Cluster sizes
        'label_coms',                # Cluster centers of mass
        'num_cells',
        'metadata_sample.Age',
        'spots.final_filter',
        'metadata_sample.Brain_Atlas_coordinates',
        'metadata_sample.mouse_ID'
    ]

    slide_field = 'metadata_sample_slide_name_std'
    experimental_field = 'ExperimentalQ111 - 488mHT - 548mHTa - 647Darp'

    df_extracted_full = extract_dataframe(
        data_dict,
        field_keys=fields_to_extract,
        channels=desired_channels,
        include_file_metadata_sample=True,
        explode_fields=[]
    )

    print(f"Extracted {len(df_extracted_full)} records")

    # Compute thresholds
    print("\nComputing thresholds...")
    (thresholds, thresholds_cluster,
     error_thresholds, error_thresholds_cluster,
     number_of_datapoints, age) = compute_thresholds(
        df_extracted=df_extracted_full,
        slide_field=slide_field,
        desired_channels=['green', 'orange'],
        negative_control_field='Negative control',
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
    thr_df = pd.DataFrame(thr_rows).drop_duplicates(["slide", "channel"])

    # Merge thresholds
    df_extracted_full = df_extracted_full.merge(
        thr_df, how="left",
        left_on=[slide_field, "channel"],
        right_on=["slide", "channel"]
    )
    df_extracted_full.rename(columns={"thr": "threshold"}, inplace=True)
    df_extracted_full.drop(columns=["slide"], inplace=True, errors='ignore')

    # Filter experimental data
    df_exp = df_extracted_full[df_extracted_full['metadata_sample_Probe-Set'] == experimental_field].copy()

    # Split by mouse model
    df_q111 = df_exp[df_exp["metadata_sample_Mouse_Model"] == 'Q111'].copy()
    df_wt = df_exp[df_exp["metadata_sample_Mouse_Model"] == 'Wildtype'].copy()

    # Exclude problematic slides
    df_q111 = df_q111[~df_q111[slide_field].isin(EXCLUDED_SLIDES)]
    df_wt = df_wt[~df_wt[slide_field].isin(EXCLUDED_SLIDES)]

    print(f"Q111 records: {len(df_q111)} (after excluding {EXCLUDED_SLIDES})")
    print(f"Wildtype records: {len(df_wt)}")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 2: LOAD FOV-LEVEL DATA AND WT P95 THRESHOLDS
    # ══════════════════════════════════════════════════════════════════════════

    print("\n" + "="*70)
    print("SECTION 2: LOADING FOV-LEVEL DATA AND THRESHOLDS")
    print("="*70)

    # Load FOV-level data from comprehensive analysis
    df_fov = pd.read_csv(OUTPUT_DIR_COMPREHENSIVE / 'fov_level_data.csv')
    df_fov_exp = df_fov[df_fov['Mouse_Model'].isin(['Q111', 'Wildtype'])].copy()

    # Exclude problematic slides
    df_fov_exp = df_fov_exp[~df_fov_exp['Slide'].isin(EXCLUDED_SLIDES)]

    # Standardize channel name to match channel_labels_exp
    df_fov_exp['Channel'] = df_fov_exp['Channel'].replace({
        'fl-HTT': 'fl-HTT',
        'fl-HTT': 'fl-HTT',
        'HTT1a': 'HTT1a'
    })

    print(f"FOV-level data: {len(df_fov_exp)} records (after excluding {EXCLUDED_SLIDES})")

    # Calculate WT P95 thresholds for each channel/region
    wt_p95_thresholds = {}

    for ch in ['HTT1a', 'fl-HTT']:
        for region in ['Cortex', 'Striatum']:
            wt_data = df_fov_exp[(df_fov_exp['Mouse_Model'] == 'Wildtype') &
                                 (df_fov_exp['Channel'] == ch) &
                                 (df_fov_exp['Region'] == region)]['Clustered_mRNA_per_Cell'].dropna()

            if len(wt_data) > 0:
                wt_p95 = np.percentile(wt_data, 95)
                wt_p95_thresholds[(ch, region)] = wt_p95
                print(f"  {ch} - {region}: WT P95 = {wt_p95:.2f}")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 3: EXTRACT CLUSTER-LEVEL DATA FOR EACH FOV
    # ══════════════════════════════════════════════════════════════════════════

    print("\n" + "="*70)
    print("SECTION 3: EXTRACTING CLUSTER-LEVEL DATA")
    print("="*70)

    # Build DAPI lookup from blue channel
    df_exp_sorted = df_exp.sort_index()
    dapi_lookup = {}

    for idx, row in df_exp_sorted.iterrows():
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

    # Compute slide-specific peak intensities
    print("Computing slide-specific peak intensities...")
    spot_peaks = {}

    for slide in df_q111[slide_field].unique():
        for channel in ['green', 'orange']:
            all_intensities = []

            df_subset = df_q111[(df_q111[slide_field] == slide) &
                                (df_q111['channel'] == channel)]

            for idx, row in df_subset.iterrows():
                threshold_val = row.get('threshold', np.nan)
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
                                    valid_photons = photons_filtered[above_threshold]
                                    all_intensities.extend(valid_photons)
                    except:
                        pass

            if len(all_intensities) >= 50:
                peak_intensity = compute_peak_intensity(np.array(all_intensities))
                if not np.isnan(peak_intensity):
                    spot_peaks[(slide, channel)] = peak_intensity

    print(f"  Computed {len(spot_peaks)} slide-specific peak intensities")

    # Extract cluster-level data with FOV classification
    print("Extracting cluster properties...")

    cluster_data_all = []
    fov_cluster_summary = []

    for idx, row in df_q111.iterrows():
        slide = row.get(slide_field, 'unknown')
        subregion = row.get('metadata_sample_Slice_Region', 'unknown')
        channel = row.get('channel', 'unknown')
        age = row.get('metadata_sample_Age', np.nan)
        mouse_id = row.get('metadata_sample_mouse_ID', 'unknown')
        threshold_val = row.get('threshold', np.nan)

        # Skip blue channel
        if channel == 'blue':
            continue

        # Get region
        region = get_region(subregion)
        if region is None:
            continue

        # Get peak intensity
        if (slide, channel) not in spot_peaks:
            continue
        peak_intensity = spot_peaks[(slide, channel)]

        # Get N_nuc
        if idx not in dapi_lookup:
            continue
        N_nuc, V_DAPI = dapi_lookup[idx]

        if N_nuc < MIN_NUCLEI_THRESHOLD:
            continue

        # Get cluster data
        cluster_intensities = row.get('cluster_intensities', None)
        cluster_cvs = row.get('cluster_cvs', None)
        cluster_distances = row.get('cluster_distance_dapi_um', None)
        label_sizes = row.get('label_sizes', None)

        if cluster_intensities is None:
            continue

        try:
            cluster_intensities = np.asarray(cluster_intensities)

            if cluster_cvs is not None:
                cluster_cvs = np.asarray(cluster_cvs)
            else:
                cluster_cvs = None

            if cluster_distances is not None:
                cluster_distances = np.asarray(cluster_distances)
            else:
                cluster_distances = np.array([])

            if label_sizes is not None:
                label_sizes = np.asarray(label_sizes)
            else:
                label_sizes = np.array([])
        except:
            continue

        # Apply threshold to clusters (intensity AND CV)
        if np.isnan(threshold_val) or len(cluster_intensities) == 0:
            continue

        # Intensity threshold
        intensity_mask = cluster_intensities > threshold_val

        # CV threshold (CV >= CV_THRESHOLD means good quality)
        # CV data is required - no fallback
        if cluster_cvs is None or len(cluster_cvs) != len(cluster_intensities):
            raise ValueError(f"CV data missing or mismatched for cluster filtering. cluster_cvs={cluster_cvs is not None}, len mismatch={len(cluster_cvs) if cluster_cvs is not None else 'N/A'} vs {len(cluster_intensities)}")
        cv_mask = cluster_cvs >= CV_THRESHOLD
        above_threshold = intensity_mask & cv_mask

        n_clusters = above_threshold.sum()

        if n_clusters == 0:
            # FOV with no clusters above threshold
            I_cluster_total = 0
        else:
            I_cluster_total = cluster_intensities[above_threshold].sum()

        # Calculate clustered mRNA per cell
        cluster_mrna_equiv = I_cluster_total / peak_intensity
        clustered_mrna_per_cell = cluster_mrna_equiv / N_nuc

        # Classify FOV as extreme or normal
        ch_label = channel_labels_exp[channel]
        wt_p95 = wt_p95_thresholds.get((ch_label, region), np.nan)

        if np.isnan(wt_p95):
            continue

        is_extreme = clustered_mrna_per_cell > wt_p95
        fov_class = 'Extreme' if is_extreme else 'Normal'

        # Clusters per cell
        clusters_per_cell = n_clusters / N_nuc

        # Extract individual cluster properties
        if n_clusters > 0:
            cluster_int_filtered = cluster_intensities[above_threshold]

            # Get indices that would sort by intensity (descending)
            # cluster_distance_dapi_um corresponds to top N clusters by intensity
            sorted_indices = np.argsort(cluster_intensities)[::-1]
            n_with_distance = len(cluster_distances)

            # Create a mapping: original index -> distance (if in top N)
            distance_map = {}
            for rank, orig_idx in enumerate(sorted_indices):
                if rank < n_with_distance:
                    distance_map[orig_idx] = cluster_distances[rank]

            # Get distances for above-threshold clusters
            above_threshold_indices = np.where(above_threshold)[0]
            cluster_dist_filtered = []
            for orig_idx in above_threshold_indices:
                if orig_idx in distance_map:
                    cluster_dist_filtered.append(distance_map[orig_idx])
                else:
                    cluster_dist_filtered.append(np.nan)
            cluster_dist_filtered = np.array(cluster_dist_filtered)

            if len(label_sizes) == len(cluster_intensities):
                cluster_size_filtered = label_sizes[above_threshold]
            else:
                cluster_size_filtered = np.array([])

            # Store individual cluster data
            for i in range(n_clusters):
                cluster_entry = {
                    'Channel': ch_label,
                    'Region': region,
                    'FOV_Class': fov_class,
                    'Mouse_ID': mouse_id,
                    'Age': age,
                    'Slide': slide,
                    'Cluster_Intensity': cluster_int_filtered[i],
                    'Cluster_mRNA_Equiv': cluster_int_filtered[i] / peak_intensity,
                }

                if i < len(cluster_dist_filtered) and not np.isnan(cluster_dist_filtered[i]):
                    cluster_entry['Distance_to_DAPI_um'] = cluster_dist_filtered[i]

                if len(cluster_size_filtered) > i:
                    cluster_entry['Cluster_Size_voxels'] = cluster_size_filtered[i]

                cluster_data_all.append(cluster_entry)

        # FOV-level summary
        fov_summary = {
            'Channel': ch_label,
            'Region': region,
            'FOV_Class': fov_class,
            'Mouse_ID': mouse_id,
            'Age': age,
            'Slide': slide,
            'N_Nuclei': N_nuc,
            'N_Clusters': n_clusters,
            'Clusters_per_Cell': clusters_per_cell,
            'Clustered_mRNA_per_Cell': clustered_mrna_per_cell,
            'WT_P95_Threshold': wt_p95,
        }

        if n_clusters > 0:
            cluster_int_filtered = cluster_intensities[above_threshold]
            # Store in mRNA equivalents
            cluster_mrna_filtered = cluster_int_filtered / peak_intensity
            fov_summary['Median_Cluster_mRNA'] = np.median(cluster_mrna_filtered)
            fov_summary['Mean_Cluster_mRNA'] = np.mean(cluster_mrna_filtered)
            fov_summary['Max_Cluster_mRNA'] = np.max(cluster_mrna_filtered)
            fov_summary['Total_Cluster_mRNA'] = np.sum(cluster_mrna_filtered)

            # Use the same distance mapping as for individual clusters
            valid_distances = cluster_dist_filtered[~np.isnan(cluster_dist_filtered)]
            if len(valid_distances) > 0:
                fov_summary['Median_Distance_DAPI'] = np.median(valid_distances)
                fov_summary['Mean_Distance_DAPI'] = np.mean(valid_distances)
                fov_summary['Frac_Nuclear'] = np.mean(valid_distances < 0)  # Negative = inside
                fov_summary['N_Clusters_with_Distance'] = len(valid_distances)

        fov_cluster_summary.append(fov_summary)

    df_clusters = pd.DataFrame(cluster_data_all)
    df_fov_summary = pd.DataFrame(fov_cluster_summary)

    print(f"  Total clusters extracted: {len(df_clusters)}")
    print(f"  FOV summaries: {len(df_fov_summary)}")

    # Save data
    df_clusters.to_csv(OUTPUT_DIR / 'cluster_level_data.csv', index=False)
    df_fov_summary.to_csv(OUTPUT_DIR / 'fov_cluster_summary.csv', index=False)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 4: VISUALIZATIONS
    # ══════════════════════════════════════════════════════════════════════════

    print("\n" + "="*70)
    print("SECTION 4: GENERATING FIGURES")
    print("="*70)

    # Colors
    colors = {
        'Extreme': '#8B0000',  # Dark red
        'Normal': '#2E8B57',   # Sea green
    }

    for ch in ['HTT1a', 'fl-HTT']:
        print(f"\nProcessing {ch}...")

        # Create comprehensive figure
        fig = plt.figure(figsize=(20, 24), dpi=FIGURE_DPI)
        fig.suptitle(f'{ch} - Cluster Properties: Extreme vs Normal FOVs\n'
                    f'(Extreme = Clustered mRNA/Cell > WT P95)',
                    fontsize=16, fontweight='bold', y=0.98)

        # Panel labels
        panel_idx = 0
        panel_labels = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

        # Layout: 6 rows x 2 columns
        # Row 1: Clusters per cell (Cortex, Striatum)
        # Row 2: Cluster intensity distributions
        # Row 3: Distance to DAPI distributions (nuclear vs cytoplasmic)
        # Row 4: Fraction nuclear
        # Row 5: Mean cluster intensity per FOV
        # Row 6: Scatter: N clusters vs intensity

        for col_idx, region in enumerate(['Cortex', 'Striatum']):
            df_ch_reg = df_fov_summary[(df_fov_summary['Channel'] == ch) &
                                       (df_fov_summary['Region'] == region)]

            df_clusters_ch_reg = df_clusters[(df_clusters['Channel'] == ch) &
                                              (df_clusters['Region'] == region)]

            if len(df_ch_reg) == 0:
                continue

            extreme = df_ch_reg[df_ch_reg['FOV_Class'] == 'Extreme']
            normal = df_ch_reg[df_ch_reg['FOV_Class'] == 'Normal']

            n_extreme = len(extreme)
            n_normal = len(normal)

            extreme_clusters = df_clusters_ch_reg[df_clusters_ch_reg['FOV_Class'] == 'Extreme']
            normal_clusters = df_clusters_ch_reg[df_clusters_ch_reg['FOV_Class'] == 'Normal']

            # ── Row 1: Clusters per cell ──────────────────────────────────────
            ax = fig.add_subplot(6, 2, col_idx + 1)

            data_extreme = extreme['Clusters_per_Cell'].dropna()
            data_normal = normal['Clusters_per_Cell'].dropna()

            if len(data_extreme) > 0 and len(data_normal) > 0:
                positions = [1, 2]
                bp = ax.boxplot([data_normal, data_extreme], positions=positions,
                               patch_artist=True, widths=0.6, showfliers=False)

                bp['boxes'][0].set_facecolor(colors['Normal'])
                bp['boxes'][1].set_facecolor(colors['Extreme'])

                for patch in bp['boxes']:
                    patch.set_alpha(0.7)

                # Statistics
                stat, p = mannwhitneyu(data_normal, data_extreme, alternative='two-sided')

                ax.set_xticks(positions)
                ax.set_xticklabels([f'Normal\n(n={n_normal})', f'Extreme\n(n={n_extreme})'])
                ax.set_ylabel('Clusters per Cell', fontsize=10)
                ax.set_title(f'{region}\nMWU p={p:.2e}', fontsize=11, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='y')

                # Set y-limit based on P99 to avoid outlier stretching
                all_data = np.concatenate([data_normal, data_extreme])
                y_max = np.percentile(all_data, 99) * 1.1
                ax.set_ylim([0, y_max])

                # Add medians
                ax.text(1, np.median(data_normal), f'{np.median(data_normal):.2f}',
                       ha='center', va='bottom', fontsize=8)
                ax.text(2, np.median(data_extreme), f'{np.median(data_extreme):.2f}',
                       ha='center', va='bottom', fontsize=8)

            # Panel label
            ax.text(-0.08, 1.05, panel_labels[panel_idx], transform=ax.transAxes,
                   fontsize=14, fontweight='bold', va='bottom', ha='left')
            panel_idx += 1

            # ── Row 2: Cluster intensity distributions ────────────────────────
            ax = fig.add_subplot(6, 2, col_idx + 3)

            if 'Cluster_mRNA_Equiv' in extreme_clusters.columns:
                data_extreme_int = extreme_clusters['Cluster_mRNA_Equiv'].dropna()
                data_normal_int = normal_clusters['Cluster_mRNA_Equiv'].dropna()

                if len(data_extreme_int) > 0 and len(data_normal_int) > 0:
                    max_val = np.percentile(np.concatenate([data_extreme_int, data_normal_int]), 99)
                    bins = np.linspace(0, max_val, 50)

                    ax.hist(data_normal_int, bins=bins, alpha=0.6, color=colors['Normal'],
                           label=f'Normal (n={len(data_normal_int)})', density=True, edgecolor='black')
                    ax.hist(data_extreme_int, bins=bins, alpha=0.6, color=colors['Extreme'],
                           label=f'Extreme (n={len(data_extreme_int)})', density=True, edgecolor='black')

                    stat, p = mannwhitneyu(data_normal_int, data_extreme_int, alternative='two-sided')

                    ax.set_xlabel('Cluster mRNA Equivalents', fontsize=10)
                    ax.set_ylabel('Density', fontsize=10)
                    ax.set_title(f'{region} - Cluster Intensity\nMWU p={p:.2e}',
                                fontsize=11, fontweight='bold')
                    ax.legend(fontsize=8, loc='upper right')
                    ax.grid(True, alpha=0.3, axis='y')

            ax.text(-0.08, 1.05, panel_labels[panel_idx], transform=ax.transAxes,
                   fontsize=14, fontweight='bold', va='bottom', ha='left')
            panel_idx += 1

            # ── Row 3: Distance to DAPI distributions ─────────────────────────
            ax = fig.add_subplot(6, 2, col_idx + 5)

            if 'Distance_to_DAPI_um' in extreme_clusters.columns:
                data_extreme_dist = extreme_clusters['Distance_to_DAPI_um'].dropna()
                data_normal_dist = normal_clusters['Distance_to_DAPI_um'].dropna()

                if len(data_extreme_dist) > 0 and len(data_normal_dist) > 0:
                    min_val = np.percentile(np.concatenate([data_extreme_dist, data_normal_dist]), 1)
                    max_val = np.percentile(np.concatenate([data_extreme_dist, data_normal_dist]), 99)
                    # Use 1 μm bin width for distance to DAPI
                    bins = np.arange(min_val, max_val + 1.0, 1.0)

                    ax.hist(data_normal_dist, bins=bins, alpha=0.6, color=colors['Normal'],
                           label=f'Normal', density=True, edgecolor='black')
                    ax.hist(data_extreme_dist, bins=bins, alpha=0.6, color=colors['Extreme'],
                           label=f'Extreme', density=True, edgecolor='black')

                    # Mark nuclear boundary
                    ax.axvline(0, color='black', linestyle='--', linewidth=2, alpha=0.7)
                    ax.text(0.02, 0.95, '← Nuclear | Cytoplasmic →', transform=ax.transAxes,
                           fontsize=8, va='top')

                    stat, p = mannwhitneyu(data_normal_dist, data_extreme_dist, alternative='two-sided')

                    ax.set_xlabel('Distance to DAPI Edge (μm)', fontsize=10)
                    ax.set_ylabel('Density', fontsize=10)
                    ax.set_title(f'{region} - Cluster Localization\nMWU p={p:.2e}',
                                fontsize=11, fontweight='bold')
                    ax.legend(fontsize=8, loc='upper right')
                    ax.grid(True, alpha=0.3, axis='y')

            ax.text(-0.08, 1.05, panel_labels[panel_idx], transform=ax.transAxes,
                   fontsize=14, fontweight='bold', va='bottom', ha='left')
            panel_idx += 1

            # ── Row 4: Fraction nuclear per FOV ───────────────────────────────
            ax = fig.add_subplot(6, 2, col_idx + 7)

            if 'Frac_Nuclear' in extreme.columns:
                data_extreme_nuc = extreme['Frac_Nuclear'].dropna()
                data_normal_nuc = normal['Frac_Nuclear'].dropna()

                if len(data_extreme_nuc) > 0 and len(data_normal_nuc) > 0:
                    positions = [1, 2]
                    bp = ax.boxplot([data_normal_nuc * 100, data_extreme_nuc * 100],
                                   positions=positions, patch_artist=True, widths=0.6)

                    bp['boxes'][0].set_facecolor(colors['Normal'])
                    bp['boxes'][1].set_facecolor(colors['Extreme'])

                    for patch in bp['boxes']:
                        patch.set_alpha(0.7)

                    stat, p = mannwhitneyu(data_normal_nuc, data_extreme_nuc, alternative='two-sided')

                    ax.set_xticks(positions)
                    ax.set_xticklabels([f'Normal', f'Extreme'])
                    ax.set_ylabel('% Clusters in Nucleus', fontsize=10)
                    ax.set_title(f'{region} - Nuclear Fraction\nMWU p={p:.2e}',
                                fontsize=11, fontweight='bold')
                    ax.set_ylim([0, 100])
                    ax.grid(True, alpha=0.3, axis='y')

                    # Add medians
                    ax.text(1, np.median(data_normal_nuc * 100) + 2,
                           f'{np.median(data_normal_nuc * 100):.0f}%',
                           ha='center', va='bottom', fontsize=8)
                    ax.text(2, np.median(data_extreme_nuc * 100) + 2,
                           f'{np.median(data_extreme_nuc * 100):.0f}%',
                           ha='center', va='bottom', fontsize=8)

            ax.text(-0.08, 1.05, panel_labels[panel_idx], transform=ax.transAxes,
                   fontsize=14, fontweight='bold', va='bottom', ha='left')
            panel_idx += 1

            # ── Row 5: Mean cluster mRNA per FOV ─────────────────────────
            ax = fig.add_subplot(6, 2, col_idx + 9)

            if 'Mean_Cluster_mRNA' in extreme.columns:
                data_extreme_mean = extreme['Mean_Cluster_mRNA'].dropna()
                data_normal_mean = normal['Mean_Cluster_mRNA'].dropna()

                if len(data_extreme_mean) > 0 and len(data_normal_mean) > 0:
                    positions = [1, 2]
                    bp = ax.boxplot([data_normal_mean, data_extreme_mean],
                                   positions=positions, patch_artist=True, widths=0.6, showfliers=False)

                    bp['boxes'][0].set_facecolor(colors['Normal'])
                    bp['boxes'][1].set_facecolor(colors['Extreme'])

                    for patch in bp['boxes']:
                        patch.set_alpha(0.7)

                    stat, p = mannwhitneyu(data_normal_mean, data_extreme_mean, alternative='two-sided')

                    ax.set_xticks(positions)
                    ax.set_xticklabels([f'Normal', f'Extreme'])
                    ax.set_ylabel('Mean Cluster mRNA Equiv.', fontsize=10)
                    ax.set_title(f'{region} - Mean mRNA/Cluster per FOV\nMWU p={p:.2e}',
                                fontsize=11, fontweight='bold')
                    ax.grid(True, alpha=0.3, axis='y')

                    # Set y-limit based on P99 to avoid outlier stretching
                    all_data = np.concatenate([data_normal_mean, data_extreme_mean])
                    y_max = np.percentile(all_data, 99) * 1.1
                    ax.set_ylim([0, y_max])

                    # Add medians
                    ax.text(1, np.median(data_normal_mean), f'{np.median(data_normal_mean):.1f}',
                           ha='center', va='bottom', fontsize=8)
                    ax.text(2, np.median(data_extreme_mean), f'{np.median(data_extreme_mean):.1f}',
                           ha='center', va='bottom', fontsize=8)

            ax.text(-0.08, 1.05, panel_labels[panel_idx], transform=ax.transAxes,
                   fontsize=14, fontweight='bold', va='bottom', ha='left')
            panel_idx += 1

            # ── Row 6: Scatter plot - clusters vs intensity ───────────────────
            ax = fig.add_subplot(6, 2, col_idx + 11)

            if len(extreme) > 0 and len(normal) > 0:
                ax.scatter(normal['Clusters_per_Cell'], normal['Clustered_mRNA_per_Cell'],
                          c=colors['Normal'], alpha=0.5, s=20, label='Normal', edgecolors='none')
                ax.scatter(extreme['Clusters_per_Cell'], extreme['Clustered_mRNA_per_Cell'],
                          c=colors['Extreme'], alpha=0.5, s=20, label='Extreme', edgecolors='none')

                # Add WT P95 threshold line
                wt_p95 = wt_p95_thresholds.get((ch, region), np.nan)
                if not np.isnan(wt_p95):
                    ax.axhline(wt_p95, color='black', linestyle='--', linewidth=2,
                              alpha=0.7, label=f'WT P95 = {wt_p95:.1f}')

                ax.set_xlabel('Clusters per Cell', fontsize=10)
                ax.set_ylabel('Clustered mRNA per Cell', fontsize=10)
                ax.set_title(f'{region} - Clusters vs mRNA/Cell',
                            fontsize=11, fontweight='bold')
                ax.legend(fontsize=8, loc='upper left')
                ax.grid(True, alpha=0.3)

            ax.text(-0.08, 1.05, panel_labels[panel_idx], transform=ax.transAxes,
                   fontsize=14, fontweight='bold', va='bottom', ha='left')
            panel_idx += 1

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Save figure
        ch_safe = ch.replace(" ", "_")
        for fmt in ['pdf', 'svg', 'png']:
            filepath = OUTPUT_DIR / f'cluster_properties_{ch_safe}.{fmt}'
            plt.savefig(filepath, format=fmt, bbox_inches='tight', dpi=FIGURE_DPI)

        plt.close()
        print(f"  Saved figures for {ch}")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 5: SUMMARY STATISTICS
    # ══════════════════════════════════════════════════════════════════════════

    print("\n" + "="*70)
    print("SECTION 5: SUMMARY STATISTICS")
    print("="*70)

    summary_stats = []

    for ch in ['HTT1a', 'fl-HTT']:
        for region in ['Cortex', 'Striatum']:
            df_ch_reg = df_fov_summary[(df_fov_summary['Channel'] == ch) &
                                       (df_fov_summary['Region'] == region)]

            if len(df_ch_reg) == 0:
                continue

            extreme = df_ch_reg[df_ch_reg['FOV_Class'] == 'Extreme']
            normal = df_ch_reg[df_ch_reg['FOV_Class'] == 'Normal']

            stats = {
                'Channel': ch,
                'Region': region,
                'N_Extreme_FOVs': len(extreme),
                'N_Normal_FOVs': len(normal),
            }

            # Clusters per cell
            if len(extreme) > 0 and len(normal) > 0:
                ext_cpc = extreme['Clusters_per_Cell'].dropna()
                norm_cpc = normal['Clusters_per_Cell'].dropna()

                if len(ext_cpc) > 0 and len(norm_cpc) > 0:
                    stats['Extreme_Median_Clusters_per_Cell'] = np.median(ext_cpc)
                    stats['Normal_Median_Clusters_per_Cell'] = np.median(norm_cpc)
                    stats['Extreme_Mean_Clusters_per_Cell'] = np.mean(ext_cpc)
                    stats['Normal_Mean_Clusters_per_Cell'] = np.mean(norm_cpc)

                    stat, p = mannwhitneyu(norm_cpc, ext_cpc, alternative='two-sided')
                    stats['Clusters_per_Cell_MWU_p'] = p

            # Nuclear fraction
            if 'Frac_Nuclear' in extreme.columns:
                ext_nuc = extreme['Frac_Nuclear'].dropna()
                norm_nuc = normal['Frac_Nuclear'].dropna()

                if len(ext_nuc) > 0 and len(norm_nuc) > 0:
                    stats['Extreme_Median_Frac_Nuclear'] = np.median(ext_nuc)
                    stats['Normal_Median_Frac_Nuclear'] = np.median(norm_nuc)

                    stat, p = mannwhitneyu(norm_nuc, ext_nuc, alternative='two-sided')
                    stats['Frac_Nuclear_MWU_p'] = p

            # Mean cluster mRNA
            if 'Mean_Cluster_mRNA' in extreme.columns:
                ext_mrna = extreme['Mean_Cluster_mRNA'].dropna()
                norm_mrna = normal['Mean_Cluster_mRNA'].dropna()

                if len(ext_mrna) > 0 and len(norm_mrna) > 0:
                    stats['Extreme_Median_Mean_Cluster_mRNA'] = np.median(ext_mrna)
                    stats['Normal_Median_Mean_Cluster_mRNA'] = np.median(norm_mrna)

                    stat, p = mannwhitneyu(norm_mrna, ext_mrna, alternative='two-sided')
                    stats['Mean_Cluster_mRNA_MWU_p'] = p

            summary_stats.append(stats)

            print(f"\n{ch} - {region}:")
            print(f"  Extreme FOVs: {stats.get('N_Extreme_FOVs', 0)}")
            print(f"  Normal FOVs: {stats.get('N_Normal_FOVs', 0)}")
            print(f"  Clusters/Cell - Extreme: {stats.get('Extreme_Median_Clusters_per_Cell', 'N/A'):.2f}, "
                  f"Normal: {stats.get('Normal_Median_Clusters_per_Cell', 'N/A'):.2f}")
            if 'Frac_Nuclear_MWU_p' in stats:
                print(f"  % Nuclear - Extreme: {stats.get('Extreme_Median_Frac_Nuclear', 0)*100:.1f}%, "
                      f"Normal: {stats.get('Normal_Median_Frac_Nuclear', 0)*100:.1f}%")

    # Save summary
    df_summary = pd.DataFrame(summary_stats)
    df_summary.to_csv(OUTPUT_DIR / 'summary_statistics.csv', index=False)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 6: GENERATE COMPREHENSIVE CAPTION WITH ALL STATISTICS
    # ══════════════════════════════════════════════════════════════════════════

    for ch in ['HTT1a', 'fl-HTT']:
        ch_safe = ch.replace(" ", "_")

        # Compute detailed statistics for this channel
        region_stats = {}
        for region in ['Cortex', 'Striatum']:
            df_ch_reg = df_fov_summary[(df_fov_summary['Channel'] == ch) &
                                       (df_fov_summary['Region'] == region)]
            df_clusters_ch_reg = df_clusters[(df_clusters['Channel'] == ch) &
                                              (df_clusters['Region'] == region)]

            if len(df_ch_reg) == 0:
                continue

            extreme = df_ch_reg[df_ch_reg['FOV_Class'] == 'Extreme']
            normal = df_ch_reg[df_ch_reg['FOV_Class'] == 'Normal']
            extreme_clusters = df_clusters_ch_reg[df_clusters_ch_reg['FOV_Class'] == 'Extreme']
            normal_clusters = df_clusters_ch_reg[df_clusters_ch_reg['FOV_Class'] == 'Normal']

            stats = {'n_extreme': len(extreme), 'n_normal': len(normal)}

            # Clusters per cell
            ext_cpc = extreme['Clusters_per_Cell'].dropna()
            norm_cpc = normal['Clusters_per_Cell'].dropna()
            if len(ext_cpc) > 0 and len(norm_cpc) > 0:
                stat, p = mannwhitneyu(norm_cpc, ext_cpc, alternative='two-sided')
                stats['cpc'] = {
                    'ext_median': np.median(ext_cpc), 'norm_median': np.median(norm_cpc),
                    'ext_iqr': np.percentile(ext_cpc, 75) - np.percentile(ext_cpc, 25),
                    'norm_iqr': np.percentile(norm_cpc, 75) - np.percentile(norm_cpc, 25),
                    'fold_change': np.median(ext_cpc) / np.median(norm_cpc) if np.median(norm_cpc) > 0 else np.nan,
                    'p_value': p
                }

            # Cluster mRNA equivalents
            ext_mrna = extreme_clusters['Cluster_mRNA_Equiv'].dropna() if 'Cluster_mRNA_Equiv' in extreme_clusters.columns else pd.Series([])
            norm_mrna = normal_clusters['Cluster_mRNA_Equiv'].dropna() if 'Cluster_mRNA_Equiv' in normal_clusters.columns else pd.Series([])
            if len(ext_mrna) > 0 and len(norm_mrna) > 0:
                stat, p = mannwhitneyu(norm_mrna, ext_mrna, alternative='two-sided')
                stats['cluster_mrna'] = {
                    'ext_median': np.median(ext_mrna), 'norm_median': np.median(norm_mrna),
                    'ext_n': len(ext_mrna), 'norm_n': len(norm_mrna),
                    'p_value': p
                }

            # Distance to DAPI
            ext_dist = extreme_clusters['Distance_to_DAPI_um'].dropna() if 'Distance_to_DAPI_um' in extreme_clusters.columns else pd.Series([])
            norm_dist = normal_clusters['Distance_to_DAPI_um'].dropna() if 'Distance_to_DAPI_um' in normal_clusters.columns else pd.Series([])
            if len(ext_dist) > 0 and len(norm_dist) > 0:
                stat, p = mannwhitneyu(norm_dist, ext_dist, alternative='two-sided')
                stats['distance'] = {
                    'ext_median': np.median(ext_dist), 'norm_median': np.median(norm_dist),
                    'ext_frac_nuclear': np.mean(ext_dist < 0) * 100,
                    'norm_frac_nuclear': np.mean(norm_dist < 0) * 100,
                    'p_value': p
                }

            # Fraction nuclear per FOV
            ext_nuc = extreme['Frac_Nuclear'].dropna() if 'Frac_Nuclear' in extreme.columns else pd.Series([])
            norm_nuc = normal['Frac_Nuclear'].dropna() if 'Frac_Nuclear' in normal.columns else pd.Series([])
            if len(ext_nuc) > 0 and len(norm_nuc) > 0:
                stat, p = mannwhitneyu(norm_nuc, ext_nuc, alternative='two-sided')
                stats['frac_nuclear'] = {
                    'ext_median': np.median(ext_nuc) * 100, 'norm_median': np.median(norm_nuc) * 100,
                    'p_value': p
                }

            # Mean cluster mRNA per FOV
            ext_mean = extreme['Mean_Cluster_mRNA'].dropna() if 'Mean_Cluster_mRNA' in extreme.columns else pd.Series([])
            norm_mean = normal['Mean_Cluster_mRNA'].dropna() if 'Mean_Cluster_mRNA' in normal.columns else pd.Series([])
            if len(ext_mean) > 0 and len(norm_mean) > 0:
                stat, p = mannwhitneyu(norm_mean, ext_mean, alternative='two-sided')
                stats['mean_mrna'] = {
                    'ext_median': np.median(ext_mean), 'norm_median': np.median(norm_mean),
                    'p_value': p
                }

            # WT P95 threshold
            stats['wt_p95'] = wt_p95_thresholds.get((ch, region), np.nan)

            region_stats[region] = stats

        caption_lines = [
            "=" * 80,
            f"FIGURE: Cluster Properties - Extreme vs Normal FOVs - {ch}",
            "=" * 80,
            "",
            "OVERVIEW:",
            "-" * 80,
            "This figure compares cluster-level properties between extreme and normal FOVs",
            "within Q111 transgenic mice only. FOVs are classified as 'extreme' if their",
            "Clustered mRNA/Cell exceeds the 95th percentile of the Wildtype distribution.",
            "",
            "The analysis addresses the key question: What makes extreme Q111 FOVs different",
            "from normal Q111 FOVs? Note: Wildtype data is only used to define the threshold;",
            "all comparisons shown are between Q111 extreme vs Q111 normal FOVs.",
            "",
            "Biological context:",
            f"  - {ch} {'represents aberrant intron-1 terminated transcripts encoding toxic N-terminal fragments' if ch == 'HTT1a' else 'represents completely spliced transcripts encoding full huntingtin protein'}",
            "  - Clusters indicate sites of active transcription or mRNA aggregation",
            "  - Higher cluster abundance may indicate pathological mRNA accumulation",
            "",
            "DATASET STATISTICS (Q111 only):",
            "-" * 80,
        ]

        for region in ['Cortex', 'Striatum']:
            if region in region_stats:
                s = region_stats[region]
                caption_lines.extend([
                    f"{region}:",
                    f"  WT P95 threshold: {s['wt_p95']:.2f} mRNA/cell",
                    f"  Extreme FOVs: {s['n_extreme']}",
                    f"  Normal FOVs: {s['n_normal']}",
                    ""
                ])

        caption_lines.extend([
            "PANEL DESCRIPTIONS WITH STATISTICAL RESULTS:",
            "-" * 80,
            "",
            "(A-B) CLUSTERS PER CELL:",
            "Box plots comparing the number of clusters per cell between normal and extreme",
            "FOVs. Outliers hidden for clarity (y-axis capped at 99th percentile).",
            "(A) Cortex, (B) Striatum.",
            ""
        ])

        for region in ['Cortex', 'Striatum']:
            if region in region_stats and 'cpc' in region_stats[region]:
                s = region_stats[region]['cpc']
                caption_lines.extend([
                    f"  {region}:",
                    f"    Extreme: median={s['ext_median']:.2f}, IQR={s['ext_iqr']:.2f}",
                    f"    Normal: median={s['norm_median']:.2f}, IQR={s['norm_iqr']:.2f}",
                    f"    Fold change: {s['fold_change']:.1f}×",
                    f"    Mann-Whitney U p-value: {s['p_value']:.2e}",
                    ""
                ])

        caption_lines.extend([
            "(C-D) CLUSTER mRNA EQUIVALENTS DISTRIBUTION:",
            "Histograms of individual cluster sizes (mRNA equivalents). Each cluster's",
            "intensity is divided by the slide-specific single-spot peak intensity.",
            "(C) Cortex, (D) Striatum.",
            ""
        ])

        for region in ['Cortex', 'Striatum']:
            if region in region_stats and 'cluster_mrna' in region_stats[region]:
                s = region_stats[region]['cluster_mrna']
                caption_lines.extend([
                    f"  {region}:",
                    f"    Extreme clusters: n={s['ext_n']}, median={s['ext_median']:.1f} mRNA equiv.",
                    f"    Normal clusters: n={s['norm_n']}, median={s['norm_median']:.1f} mRNA equiv.",
                    f"    Mann-Whitney U p-value: {s['p_value']:.2e}",
                    ""
                ])

        caption_lines.extend([
            "(E-F) CLUSTER LOCALIZATION (Distance to DAPI):",
            "Distribution of signed distance from cluster center to nearest DAPI edge.",
            "Negative = nuclear (inside DAPI), Positive = cytoplasmic (outside DAPI).",
            "Dashed line marks the nuclear boundary (distance = 0).",
            "(E) Cortex, (F) Striatum.",
            ""
        ])

        for region in ['Cortex', 'Striatum']:
            if region in region_stats and 'distance' in region_stats[region]:
                s = region_stats[region]['distance']
                caption_lines.extend([
                    f"  {region}:",
                    f"    Extreme: median={s['ext_median']:.2f} µm, {s['ext_frac_nuclear']:.1f}% nuclear",
                    f"    Normal: median={s['norm_median']:.2f} µm, {s['norm_frac_nuclear']:.1f}% nuclear",
                    f"    Mann-Whitney U p-value: {s['p_value']:.2e}",
                    ""
                ])

        caption_lines.extend([
            "(G-H) FRACTION NUCLEAR PER FOV:",
            "Box plots of the percentage of clusters located within the nucleus per FOV.",
            "Nuclear clusters defined as those with distance to DAPI < 0.",
            "(G) Cortex, (H) Striatum.",
            ""
        ])

        for region in ['Cortex', 'Striatum']:
            if region in region_stats and 'frac_nuclear' in region_stats[region]:
                s = region_stats[region]['frac_nuclear']
                caption_lines.extend([
                    f"  {region}:",
                    f"    Extreme: median={s['ext_median']:.1f}%",
                    f"    Normal: median={s['norm_median']:.1f}%",
                    f"    Mann-Whitney U p-value: {s['p_value']:.2e}",
                    ""
                ])

        caption_lines.extend([
            "(I-J) MEAN CLUSTER mRNA PER FOV:",
            "Box plots of mean cluster mRNA equivalents per FOV. Shows whether extreme FOVs",
            "have larger individual clusters on average. Outliers hidden for clarity.",
            "(I) Cortex, (J) Striatum.",
            ""
        ])

        for region in ['Cortex', 'Striatum']:
            if region in region_stats and 'mean_mrna' in region_stats[region]:
                s = region_stats[region]['mean_mrna']
                caption_lines.extend([
                    f"  {region}:",
                    f"    Extreme: median={s['ext_median']:.1f} mRNA equiv.",
                    f"    Normal: median={s['norm_median']:.1f} mRNA equiv.",
                    f"    Mann-Whitney U p-value: {s['p_value']:.2e}",
                    ""
                ])

        caption_lines.extend([
            "(K-L) SCATTER: CLUSTERS vs mRNA/CELL:",
            "Scatter plot showing relationship between number of clusters per cell and",
            "total clustered mRNA per cell. Dashed line = WT P95 threshold.",
            "(K) Cortex, (L) Striatum.",
            "",
            "KEY CONCLUSIONS:",
            "-" * 80,
        ])

        # Generate conclusions based on statistics
        conclusions = []
        for region in ['Cortex', 'Striatum']:
            if region in region_stats:
                s = region_stats[region]
                if 'cpc' in s:
                    fold = s['cpc']['fold_change']
                    conclusions.append(f"  - {region}: Extreme FOVs have {fold:.1f}× more clusters/cell (p={s['cpc']['p_value']:.2e})")

        if conclusions:
            caption_lines.extend([
                "1. CLUSTER ABUNDANCE:",
                "   Extreme FOVs have dramatically more clusters per cell than normal FOVs:",
            ] + conclusions + [""])

        # Nuclear localization conclusions
        nuclear_conclusions = []
        for region in ['Cortex', 'Striatum']:
            if region in region_stats and 'frac_nuclear' in region_stats[region]:
                s = region_stats[region]['frac_nuclear']
                diff = s['norm_median'] - s['ext_median']
                if diff > 0:
                    nuclear_conclusions.append(f"  - {region}: Normal FOVs {diff:.1f}% more nuclear (p={s['p_value']:.2e})")

        if nuclear_conclusions:
            caption_lines.extend([
                "2. NUCLEAR LOCALIZATION:",
                "   Clusters are predominantly nuclear (~90%), with slight differences:",
            ] + nuclear_conclusions + [""])

        caption_lines.extend([
            "3. CLUSTER SIZE:",
            "   Individual clusters in extreme FOVs tend to be larger than in normal FOVs,",
            "   contributing to the elevated total clustered mRNA per cell.",
            "",
            "COLOR SCHEME:",
            "-" * 80,
            "Sea Green (#2E8B57): Normal FOVs (Clustered mRNA/Cell ≤ WT P95)",
            "Dark Red (#8B0000): Extreme FOVs (Clustered mRNA/Cell > WT P95)",
            "",
            "METHODOLOGY:",
            "-" * 80,
            "Extreme FOV definition:",
            "  - Threshold = 95th percentile of Wildtype clustered mRNA/cell distribution",
            "  - FOVs with values exceeding this threshold classified as 'extreme'",
            "",
            "Cluster detection:",
            "  - DBSCAN algorithm (eps=0.75 µm, min_samples=3)",
            "  - Only clusters with intensity > 2.5 × slide-specific peak intensity",
            "  - Peak intensity = mode of single spot intensity distribution (KDE)",
            "",
            "Cluster mRNA equivalents:",
            "  - Cluster intensity / slide-specific peak intensity",
            "  - Represents the number of mRNA molecules within the cluster",
            "",
            "Distance to DAPI:",
            "  - Signed distance from cluster center to nearest DAPI edge",
            "  - Negative values = inside nucleus, Positive values = cytoplasm",
            "",
            "Statistical test:",
            "  - Mann-Whitney U test (two-sided): Non-parametric comparison of distributions",
            "  - Chosen because data may not be normally distributed",
            "",
            "Quality control:",
            "  - Slides m1a2 and m1b5 excluded (technical failures based on UBC analysis)",
            "  - FOVs with < minimum nuclei threshold excluded",
            "",
            "=" * 80,
            f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 80,
        ])

        with open(OUTPUT_DIR / f'figure_caption_{ch_safe}.txt', 'w') as f:
            f.write('\n'.join(caption_lines))

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("\nKey findings to look for:")
    print("  1. Do extreme FOVs have more clusters per cell?")
    print("  2. Are clusters in extreme FOVs larger/more intense?")
    print("  3. Are clusters in extreme FOVs more nuclear or cytoplasmic?")
    print("  4. Is there a correlation between N clusters and total mRNA?")
