"""
Supplementary Figure: Normalization Verification and Quality Control

This script generates comprehensive supplementary figures for:
1. Peak intensity normalization consistency vs age and atlas coordinates
2. Intensity distributions for single spots and clusters
3. Cells per FOV quality control metrics
4. Comparison of single, clustered, and total mRNA across metadata variables

Created: 2025-01-16
"""

import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, pearsonr, linregress
import scienceplots
plt.style.use('science')
plt.rcParams['text.usetex'] = False
from pathlib import Path
import seaborn as sns
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from result functions
from result_functions_v2 import (
    compute_thresholds,
    recursively_load_dict,
    extract_dataframe
)

# Import centralized configuration
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
    CV_THRESHOLD,
    EXCLUDED_SLIDES
)

# Create output directory
OUTPUT_DIR = Path(__file__).parent / "output" / "normalization_and_qc"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("SUPPLEMENTARY FIGURE: NORMALIZATION VERIFICATION AND QUALITY CONTROL")
print("="*80)


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


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: LOAD DATA AND COMPUTE THRESHOLDS
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("SECTION 1: LOADING DATA")
print("="*80)

slide_field = 'metadata_sample_slide_name_std'
desired_channels = ['blue', 'green', 'orange']
fields_to_extract = [
    'spots_sigma_var.params_raw', 'spots.params_raw', 'cluster_intensities',
    'cluster_cvs', 'num_cells', 'label_sizes', 'metadata_sample.Age', 'spots.final_filter',
    'metadata_sample.Brain_Atlas_coordinates', 'metadata_sample.mouse_ID'
]
negative_control_field = 'Negative control'
experimental_field = 'ExperimentalQ111 - 488mHT - 548mHTa - 647Darp'

# Load HDF5
with h5py.File(H5_FILE_PATH_EXPERIMENTAL, 'r') as h5_file:
    data_dict = recursively_load_dict(h5_file)

print(f"Loaded data from: {H5_FILE_PATH_EXPERIMENTAL}")

# Extract DataFrame
df_extracted_full = extract_dataframe(
    data_dict,
    field_keys=fields_to_extract,
    channels=desired_channels,
    include_file_metadata_sample=True,
    explode_fields=[]
)

# Compute thresholds
print("\nComputing thresholds...")
(thresholds, thresholds_cluster,
 error_thresholds, error_thresholds_cluster,
 number_of_datapoints, age) = compute_thresholds(
    df_extracted=df_extracted_full,
    slide_field=slide_field,
    desired_channels=['green', 'orange'],
    negative_control_field=negative_control_field,
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
    thr_rows.append({"slide": slide, "channel": channel, "thr": np.mean(vec)})

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
df_exp = df_extracted_full[
    df_extracted_full['metadata_sample_Probe-Set'] == experimental_field
].copy()

# Split by mouse model
df_q111 = df_exp[df_exp["metadata_sample_Mouse_Model"] == 'Q111'].copy()
df_wt = df_exp[df_exp["metadata_sample_Mouse_Model"] == 'Wildtype'].copy()

print(f"\nQ111 records (before slide exclusion): {len(df_q111)}")
print(f"Wildtype records (before slide exclusion): {len(df_wt)}")

# Filter out excluded slides (technical failures identified via UBC positive control)
n_q111_before = len(df_q111)
n_wt_before = len(df_wt)
q111_slides_before = set(df_q111[slide_field].unique())
wt_slides_before = set(df_wt[slide_field].unique())

df_q111 = df_q111[~df_q111[slide_field].isin(EXCLUDED_SLIDES)].copy()
df_wt = df_wt[~df_wt[slide_field].isin(EXCLUDED_SLIDES)].copy()

q111_slides_after = set(df_q111[slide_field].unique())
wt_slides_after = set(df_wt[slide_field].unique())
q111_excluded = q111_slides_before - q111_slides_after
wt_excluded = wt_slides_before - wt_slides_after

print(f"\nSlide filtering (EXCLUDED_SLIDES, n={len(EXCLUDED_SLIDES)}):")
print(f"  Q111: {n_q111_before} -> {len(df_q111)} records ({len(q111_excluded)} slides excluded)")
if q111_excluded:
    print(f"    Excluded Q111 slides: {', '.join(sorted(q111_excluded))}")
print(f"  Wildtype: {n_wt_before} -> {len(df_wt)} records ({len(wt_excluded)} slides excluded)")
if wt_excluded:
    print(f"    Excluded WT slides: {', '.join(sorted(wt_excluded))}")

print(f"\nQ111 records (after slide exclusion): {len(df_q111)}")
print(f"Wildtype records (after slide exclusion): {len(df_wt)}")

# Define regions
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

channel_labels_exp = {
    'green': 'HTT1a',
    'orange': 'fl-HTT'
}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: EXTRACT FOV-LEVEL DATA
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("SECTION 2: EXTRACTING FOV-LEVEL DATA")
print("="*80)


def extract_fov_level_data(df_input, mouse_model_name):
    """Extract FOV-level data with expression metrics."""

    print(f"\nProcessing {mouse_model_name} FOV-level data...")

    # Step 1: Compute slide-specific peak intensities
    print("  Computing slide-specific single spot peak intensities...")
    spot_peaks = {}  # (slide, channel) -> peak_intensity

    for idx, row in df_input.iterrows():
        slide = row.get(slide_field, 'unknown')
        channel = row.get('channel', 'unknown')

        if channel == 'blue':
            continue

        key = (slide, channel)

        if key not in spot_peaks:
            all_intensities = []

            for idx2, row2 in df_input.iterrows():
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

            if len(all_intensities) >= 50:
                peak_intensity = compute_peak_intensity(np.array(all_intensities))
                if not np.isnan(peak_intensity):
                    spot_peaks[key] = peak_intensity
                    print(f"    {slide}, {channel_labels_exp[channel]}: peak = {peak_intensity:.2f}")

    print(f"  Computed {len(spot_peaks)} slide-specific peak intensities")

    # Step 2: Build DAPI lookup from blue channel
    print("  Building DAPI lookup from blue channel...")
    df_input_sorted = df_input.sort_index()
    dapi_lookup = {}

    for idx, row in df_input_sorted.iterrows():
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

    print(f"  Built DAPI lookup for {len(dapi_lookup)} FOV entries")

    # Step 3: Extract FOV-level expression data
    fov_data = []
    n_total = 0
    n_filtered = 0

    for idx, row in df_input.iterrows():
        n_total += 1
        slide = row.get(slide_field, 'unknown')
        subregion = row.get('metadata_sample_Slice_Region', 'unknown')
        channel = row.get('channel', 'unknown')
        age = row.get('metadata_sample_Age', np.nan)
        atlas_coord = row.get('metadata_sample_Brain_Atlas_coordinates', np.nan)
        mouse_id = row.get('metadata_sample_mouse_ID', 'unknown')
        threshold_val = row.get('threshold', np.nan)

        if channel == 'blue':
            continue

        # Determine region
        if any(sub in subregion for sub in cortex_subregions):
            region_merged = 'Cortex'
        elif any(sub in subregion for sub in striatum_subregions):
            region_merged = 'Striatum'
        else:
            continue

        # Get peak intensity
        key = (slide, channel)
        if key not in spot_peaks:
            n_filtered += 1
            continue
        peak_intensity = spot_peaks[key]

        # Get N_nuc from DAPI
        if idx not in dapi_lookup:
            n_filtered += 1
            continue
        N_nuc, V_DAPI = dapi_lookup[idx]

        # Filter low DAPI
        if N_nuc < MIN_NUCLEI_THRESHOLD:
            n_filtered += 1
            continue

        # Count single spots
        num_spots = 0
        single_spot_intensities = []
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
                            single_spot_intensities = photons_filtered[above_threshold].tolist()
            except:
                pass

        # Count clusters and sum intensities
        num_clusters = 0
        I_cluster_total = 0
        cluster_intensities = []
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
                    cluster_intensities = cluster_int[above_threshold].tolist()
            except:
                pass

        # Compute mRNA metrics
        cluster_mrna_equiv = I_cluster_total / peak_intensity
        total_mrna_equiv = num_spots + cluster_mrna_equiv

        # Per-cell metrics
        single_mrna_per_cell = num_spots / N_nuc
        clustered_mrna_per_cell = cluster_mrna_equiv / N_nuc
        total_mrna_per_cell = total_mrna_equiv / N_nuc
        clusters_per_cell = num_clusters / N_nuc

        fov_data.append({
            'Mouse_Model': mouse_model_name,
            'Mouse_ID': mouse_id,
            'Slide': slide,
            'Region': region_merged,
            'Subregion': subregion,
            'Channel': channel_labels_exp[channel],
            'Age': age,
            'Brain_Atlas_Coord': atlas_coord,
            'Threshold': threshold_val,
            'N_Nuclei': N_nuc,
            'V_DAPI': V_DAPI,
            'N_Spots': num_spots,
            'N_Clusters': num_clusters,
            'I_Cluster_Total': I_cluster_total,
            'I_Single_Peak': peak_intensity,
            'Cluster_mRNA_Equiv': cluster_mrna_equiv,
            'Total_mRNA_Equiv': total_mrna_equiv,
            'Single_mRNA_per_Cell': single_mrna_per_cell,
            'Clustered_mRNA_per_Cell': clustered_mrna_per_cell,
            'Total_mRNA_per_Cell': total_mrna_per_cell,
            'Clusters_per_Cell': clusters_per_cell,
            'Single_Spot_Intensities': single_spot_intensities,
            'Cluster_Intensities': cluster_intensities
        })

    df_fov = pd.DataFrame(fov_data)

    print(f"\n  Filtering statistics:")
    print(f"    Total FOVs processed: {n_total}")
    print(f"    Filtered FOVs: {n_filtered}")
    print(f"    Extracted FOVs: {len(df_fov)}")

    return df_fov


# Extract FOV-level data
fov_q111 = extract_fov_level_data(df_q111, 'Q111')
fov_wt = extract_fov_level_data(df_wt, 'Wildtype')
fov_all = pd.concat([fov_q111, fov_wt], ignore_index=True)

print(f"\nTotal FOV records: {len(fov_all)}")

# Save FOV-level data
fov_all.to_csv(OUTPUT_DIR / "fov_level_data.csv", index=False)
print(f"\nSaved: {OUTPUT_DIR / 'fov_level_data.csv'}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: CREATE NORMALIZATION VERIFICATION FIGURE
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("SECTION 3: NORMALIZATION VERIFICATION FIGURE")
print("="*80)

# Create slide-level summary for peak intensities
slide_summary = fov_all.groupby(
    ['Mouse_Model', 'Mouse_ID', 'Slide', 'Region', 'Channel', 'Age', 'Brain_Atlas_Coord']
).agg({
    'I_Single_Peak': 'first',
    'N_Spots': 'sum',
    'N_Clusters': 'sum'
}).reset_index()

# Focus on Q111 for normalization verification
slide_q111 = slide_summary[slide_summary['Mouse_Model'] == 'Q111'].copy()

# Create figure
fig_norm = plt.figure(figsize=(24, 16), dpi=FIGURE_DPI)
gs = fig_norm.add_gridspec(4, 4, hspace=0.4, wspace=0.35)

# Define mouse color palette
mouse_ids = sorted(slide_q111['Mouse_ID'].unique())
mouse_colors = plt.cm.tab10(np.linspace(0, 1, len(mouse_ids)))
mouse_color_map = dict(zip(mouse_ids, mouse_colors))

print("\nAnalyzing peak intensity correlations...")

# Rows 0-1: Peak intensity vs Age
for ch_idx, ch in enumerate(['HTT1a', 'fl-HTT']):
    for region_idx, region in enumerate(['Cortex', 'Striatum']):
        ax = fig_norm.add_subplot(gs[ch_idx, region_idx*2:(region_idx+1)*2])

        data = slide_q111[(slide_q111['Channel'] == ch) & (slide_q111['Region'] == region)]

        if len(data) > 0:
            # Plot by mouse ID
            for mouse_id in mouse_ids:
                mouse_data = data[data['Mouse_ID'] == mouse_id]
                if len(mouse_data) > 0:
                    ax.scatter(mouse_data['Age'], mouse_data['I_Single_Peak'],
                             c=[mouse_color_map[mouse_id]], s=100, alpha=0.7,
                             edgecolor='black', linewidth=1, label=mouse_id)

            # Correlation
            valid = data[['Age', 'I_Single_Peak']].dropna()
            if len(valid) >= 3:
                r, p = pearsonr(valid['Age'], valid['I_Single_Peak'])
                slope, intercept, _, _, _ = linregress(valid['Age'], valid['I_Single_Peak'])

                x_range = np.linspace(valid['Age'].min(), valid['Age'].max(), 50)
                y_pred = slope * x_range + intercept
                ax.plot(x_range, y_pred, 'k--', linewidth=2, alpha=0.5)

                ax.text(0.05, 0.95, f'r = {r:.3f}\np = {p:.4f}',
                       transform=ax.transAxes, fontsize=11, va='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

                print(f"  {ch} - {region} vs Age: r={r:.3f}, p={p:.4f}")

        ax.set_xlabel('Age (months)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Peak Intensity (A.U.)', fontsize=12, fontweight='bold')
        ax.set_title(f'{ch}\n{region}', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        if region_idx == 1 and ch_idx == 0:
            ax.legend(loc='upper right', fontsize=8, ncol=2, title='Mouse ID')

# Rows 2-3: Peak intensity vs Atlas Coordinate
for ch_idx, ch in enumerate(['HTT1a', 'fl-HTT']):
    for region_idx, region in enumerate(['Cortex', 'Striatum']):
        ax = fig_norm.add_subplot(gs[ch_idx+2, region_idx*2:(region_idx+1)*2])

        data = slide_q111[(slide_q111['Channel'] == ch) & (slide_q111['Region'] == region)]

        if len(data) > 0:
            # Plot by mouse ID
            for mouse_id in mouse_ids:
                mouse_data = data[data['Mouse_ID'] == mouse_id]
                if len(mouse_data) > 0:
                    ax.scatter(mouse_data['Brain_Atlas_Coord'], mouse_data['I_Single_Peak'],
                             c=[mouse_color_map[mouse_id]], s=100, alpha=0.7,
                             edgecolor='black', linewidth=1, label=mouse_id)

            # Correlation
            valid = data[['Brain_Atlas_Coord', 'I_Single_Peak']].dropna()
            if len(valid) >= 3:
                r, p = pearsonr(valid['Brain_Atlas_Coord'], valid['I_Single_Peak'])
                slope, intercept, _, _, _ = linregress(valid['Brain_Atlas_Coord'], valid['I_Single_Peak'])

                x_range = np.linspace(valid['Brain_Atlas_Coord'].min(),
                                     valid['Brain_Atlas_Coord'].max(), 50)
                y_pred = slope * x_range + intercept
                ax.plot(x_range, y_pred, 'k--', linewidth=2, alpha=0.5)

                ax.text(0.05, 0.95, f'r = {r:.3f}\np = {p:.4f}',
                       transform=ax.transAxes, fontsize=11, va='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

                print(f"  {ch} - {region} vs Atlas: r={r:.3f}, p={p:.4f}")

        ax.set_xlabel('Brain Atlas Coordinate (mm)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Peak Intensity (A.U.)', fontsize=12, fontweight='bold')
        ax.set_title(f'{ch}\n{region}', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)

fig_norm.suptitle('Peak Intensity Normalization Consistency\n' +
                 'Verification across Age and Atlas Coordinates',
                 fontsize=16, fontweight='bold', y=0.995)

plt.savefig(OUTPUT_DIR / 'fig_normalization_verification.png', dpi=FIGURE_DPI, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / 'fig_normalization_verification.svg', dpi=FIGURE_DPI, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / 'fig_normalization_verification.pdf', dpi=FIGURE_DPI, bbox_inches='tight')
print(f"\nSaved: {OUTPUT_DIR / 'fig_normalization_verification.*'}")
plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: CREATE INTENSITY DISTRIBUTION FIGURE
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("SECTION 4: INTENSITY DISTRIBUTION FIGURE")
print("="*80)

# Create figure for each channel
for ch in ['HTT1a', 'fl-HTT']:
    print(f"\nCreating intensity distributions for {ch}...")

    fig_int = plt.figure(figsize=(24, 16), dpi=FIGURE_DPI)
    gs = fig_int.add_gridspec(4, 4, hspace=0.4, wspace=0.35)

    ch_data = fov_q111[fov_q111['Channel'] == ch].copy()

    # Row 0: Single spot intensities - Cortex vs Striatum (histograms)
    for region_idx, region in enumerate(['Cortex', 'Striatum']):
        ax = fig_int.add_subplot(gs[0, region_idx*2:(region_idx+1)*2])

        region_data = ch_data[ch_data['Region'] == region]

        # Collect all single spot intensities
        all_intensities = []
        for intensities in region_data['Single_Spot_Intensities']:
            if isinstance(intensities, list):
                all_intensities.extend(intensities)

        if len(all_intensities) > 0:
            all_intensities = np.array(all_intensities)

            # Histogram with KDE
            ax.hist(all_intensities, bins=50, alpha=0.6, color='steelblue',
                   edgecolor='darkblue', linewidth=1, density=True,
                   label=f'n={len(all_intensities)} spots')

            # KDE overlay
            if len(all_intensities) > 50:
                kde = gaussian_kde(all_intensities)
                x_range = np.linspace(all_intensities.min(), all_intensities.max(), 200)
                ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')

                # Mark peak
                peak_val = x_range[np.argmax(kde(x_range))]
                ax.axvline(peak_val, color='darkred', linestyle='--',
                          linewidth=2, label=f'Peak={peak_val:.1f}')

            # Statistics
            mean_int = np.mean(all_intensities)
            median_int = np.median(all_intensities)

            ax.text(0.95, 0.95,
                   f'Mean: {mean_int:.1f}\nMedian: {median_int:.1f}',
                   transform=ax.transAxes, fontsize=10, va='top', ha='right',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

        ax.set_xlabel('Single Spot Intensity (A.U.)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Density', fontsize=12, fontweight='bold')
        ax.set_title(f'Single Spots - {region}', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # Row 1: Cluster intensities - Cortex vs Striatum (histograms)
    for region_idx, region in enumerate(['Cortex', 'Striatum']):
        ax = fig_int.add_subplot(gs[1, region_idx*2:(region_idx+1)*2])

        region_data = ch_data[ch_data['Region'] == region]

        # Collect all cluster intensities
        all_intensities = []
        for intensities in region_data['Cluster_Intensities']:
            if isinstance(intensities, list):
                all_intensities.extend(intensities)

        if len(all_intensities) > 0:
            all_intensities = np.array(all_intensities)

            # Histogram with KDE
            ax.hist(all_intensities, bins=50, alpha=0.6, color='coral',
                   edgecolor='darkred', linewidth=1, density=True,
                   label=f'n={len(all_intensities)} clusters')

            # KDE overlay
            if len(all_intensities) > 50:
                kde = gaussian_kde(all_intensities)
                x_range = np.linspace(all_intensities.min(), all_intensities.max(), 200)
                ax.plot(x_range, kde(x_range), 'darkgreen', linewidth=2, label='KDE')

            # Statistics
            mean_int = np.mean(all_intensities)
            median_int = np.median(all_intensities)

            ax.text(0.95, 0.95,
                   f'Mean: {mean_int:.1f}\nMedian: {median_int:.1f}',
                   transform=ax.transAxes, fontsize=10, va='top', ha='right',
                   bbox=dict(boxstyle='round', facecolor='lightsalmon', alpha=0.8))

        ax.set_xlabel('Cluster Intensity (A.U.)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Density', fontsize=12, fontweight='bold')
        ax.set_title(f'Clusters - {region}', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # Row 2: Spots vs Clusters comparison (box plots) - by region
    ax = fig_int.add_subplot(gs[2, :2])

    cortex_data = ch_data[ch_data['Region'] == 'Cortex']
    striatum_data = ch_data[ch_data['Region'] == 'Striatum']

    data_to_plot = [
        cortex_data['Single_mRNA_per_Cell'].dropna(),
        cortex_data['Clustered_mRNA_per_Cell'].dropna(),
        striatum_data['Single_mRNA_per_Cell'].dropna(),
        striatum_data['Clustered_mRNA_per_Cell'].dropna()
    ]

    bp = ax.boxplot(data_to_plot,
                   labels=['Cortex\nSingle', 'Cortex\nClustered',
                          'Striatum\nSingle', 'Striatum\nClustered'],
                   patch_artist=True, widths=0.6)

    colors = ['steelblue', 'coral', 'steelblue', 'coral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel('mRNA per Cell', fontsize=12, fontweight='bold')
    ax.set_title('Single vs Clustered mRNA Distribution', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Row 2, right: Total mRNA comparison
    ax = fig_int.add_subplot(gs[2, 2:])

    data_to_plot = [
        cortex_data['Total_mRNA_per_Cell'].dropna(),
        striatum_data['Total_mRNA_per_Cell'].dropna()
    ]

    bp = ax.boxplot(data_to_plot,
                   labels=['Cortex', 'Striatum'],
                   patch_artist=True, widths=0.6)

    for patch, color in zip(bp['boxes'], ['green', 'purple']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel('Total mRNA per Cell', fontsize=12, fontweight='bold')
    ax.set_title('Total mRNA Distribution', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Row 3: mRNA metrics vs Age
    ax = fig_int.add_subplot(gs[3, :2])

    for region, color, marker in [('Cortex', 'green', 'o'), ('Striatum', 'purple', 's')]:
        region_data = ch_data[ch_data['Region'] == region]
        slide_avg = region_data.groupby(['Slide', 'Age']).agg({
            'Single_mRNA_per_Cell': 'mean',
            'Clustered_mRNA_per_Cell': 'mean',
            'Total_mRNA_per_Cell': 'mean'
        }).reset_index()

        if len(slide_avg) > 0:
            ax.scatter(slide_avg['Age'], slide_avg['Total_mRNA_per_Cell'],
                     c=color, marker=marker, s=100, alpha=0.7,
                     edgecolor='black', linewidth=1, label=region)

    ax.set_xlabel('Age (months)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total mRNA per Cell', fontsize=12, fontweight='bold')
    ax.set_title('Total mRNA vs Age', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Row 3: mRNA metrics vs Atlas
    ax = fig_int.add_subplot(gs[3, 2:])

    for region, color, marker in [('Cortex', 'green', 'o'), ('Striatum', 'purple', 's')]:
        region_data = ch_data[ch_data['Region'] == region]
        slide_avg = region_data.groupby(['Slide', 'Brain_Atlas_Coord']).agg({
            'Single_mRNA_per_Cell': 'mean',
            'Clustered_mRNA_per_Cell': 'mean',
            'Total_mRNA_per_Cell': 'mean'
        }).reset_index()

        if len(slide_avg) > 0:
            ax.scatter(slide_avg['Brain_Atlas_Coord'], slide_avg['Total_mRNA_per_Cell'],
                     c=color, marker=marker, s=100, alpha=0.7,
                     edgecolor='black', linewidth=1, label=region)

    ax.set_xlabel('Brain Atlas Coordinate (mm)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total mRNA per Cell', fontsize=12, fontweight='bold')
    ax.set_title('Total mRNA vs Atlas Coordinate', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    fig_int.suptitle(f'Intensity Distributions and mRNA Metrics: {ch}\n' +
                    'Single Spots, Clusters, and Total Expression',
                    fontsize=16, fontweight='bold', y=0.995)

    ch_safe = ch.replace(' ', '_').replace('-', '_')
    plt.savefig(OUTPUT_DIR / f'fig_intensity_distributions_{ch_safe}.png',
               dpi=FIGURE_DPI, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / f'fig_intensity_distributions_{ch_safe}.svg',
               dpi=FIGURE_DPI, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / f'fig_intensity_distributions_{ch_safe}.pdf',
               dpi=FIGURE_DPI, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / f'fig_intensity_distributions_{ch_safe}.*'}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5: CREATE CELLS PER FOV QC FIGURE
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("SECTION 5: CELLS PER FOV QC FIGURE")
print("="*80)

fig_qc = plt.figure(figsize=(24, 16), dpi=FIGURE_DPI)
gs = fig_qc.add_gridspec(4, 4, hspace=0.4, wspace=0.35)

# Row 0: Overall distribution by model
ax = fig_qc.add_subplot(gs[0, :2])

q111_cells = fov_q111['N_Nuclei'].values
wt_cells = fov_wt['N_Nuclei'].values

ax.hist(q111_cells, bins=40, alpha=0.6, color='steelblue',
       label='Q111', edgecolor='darkblue', linewidth=1)
ax.hist(wt_cells, bins=40, alpha=0.6, color='coral',
       label='WT', edgecolor='darkred', linewidth=1)
ax.axvline(MIN_NUCLEI_THRESHOLD, color='red', linestyle='--', linewidth=2,
          label=f'QC Threshold ({MIN_NUCLEI_THRESHOLD})')

ax.set_xlabel('Number of Nuclei per FOV', fontsize=12, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax.set_title('Cells per FOV by Mouse Model', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Statistics
ax.text(0.95, 0.95,
       f'Q111: μ={np.mean(q111_cells):.1f}, n={len(q111_cells)}\n' +
       f'WT: μ={np.mean(wt_cells):.1f}, n={len(wt_cells)}',
       transform=ax.transAxes, fontsize=10, va='top', ha='right',
       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Row 0, right: Box plot by model
ax = fig_qc.add_subplot(gs[0, 2:])

bp = ax.boxplot([q111_cells, wt_cells],
               labels=['Q111', 'WT'],
               patch_artist=True, widths=0.6)
bp['boxes'][0].set_facecolor('steelblue')
bp['boxes'][1].set_facecolor('coral')

ax.axhline(MIN_NUCLEI_THRESHOLD, color='red', linestyle='--', linewidth=2)
ax.set_ylabel('Number of Nuclei per FOV', fontsize=12, fontweight='bold')
ax.set_title('Cells per FOV: Box Plot Comparison', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Row 1: By region (Q111 only)
ax = fig_qc.add_subplot(gs[1, :2])

cortex_cells = fov_q111[fov_q111['Region'] == 'Cortex']['N_Nuclei'].values
striatum_cells = fov_q111[fov_q111['Region'] == 'Striatum']['N_Nuclei'].values

ax.hist(cortex_cells, bins=40, alpha=0.6, color='mediumpurple',
       label='Cortex', edgecolor='purple', linewidth=1)
ax.hist(striatum_cells, bins=40, alpha=0.6, color='gold',
       label='Striatum', edgecolor='darkorange', linewidth=1)
ax.axvline(MIN_NUCLEI_THRESHOLD, color='red', linestyle='--', linewidth=2)

ax.set_xlabel('Number of Nuclei per FOV', fontsize=12, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax.set_title('Cells per FOV by Brain Region (Q111)', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Row 1, right: Box plot by region
ax = fig_qc.add_subplot(gs[1, 2:])

bp = ax.boxplot([cortex_cells, striatum_cells],
               labels=['Cortex', 'Striatum'],
               patch_artist=True, widths=0.6)
bp['boxes'][0].set_facecolor('mediumpurple')
bp['boxes'][1].set_facecolor('gold')

ax.axhline(MIN_NUCLEI_THRESHOLD, color='red', linestyle='--', linewidth=2)
ax.set_ylabel('Number of Nuclei per FOV', fontsize=12, fontweight='bold')
ax.set_title('Cells per FOV: Box Plot by Region', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Row 2: By mouse ID (Q111 only)
ax = fig_qc.add_subplot(gs[2, :])

mouse_cell_data = []
mouse_labels = []
for mouse_id in sorted(fov_q111['Mouse_ID'].unique()):
    mouse_data = fov_q111[fov_q111['Mouse_ID'] == mouse_id]['N_Nuclei'].values
    if len(mouse_data) > 0:
        mouse_cell_data.append(mouse_data)
        mouse_labels.append(mouse_id)

bp = ax.boxplot(mouse_cell_data, labels=mouse_labels,
               patch_artist=True, widths=0.6)

for patch, mouse_id in zip(bp['boxes'], mouse_labels):
    patch.set_facecolor(mouse_color_map[mouse_id])
    patch.set_alpha(0.7)

ax.axhline(MIN_NUCLEI_THRESHOLD, color='red', linestyle='--', linewidth=2,
          label=f'QC Threshold ({MIN_NUCLEI_THRESHOLD})')
ax.set_xlabel('Mouse ID', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Nuclei per FOV', fontsize=12, fontweight='bold')
ax.set_title('Cells per FOV by Mouse ID (Q111)', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Row 3: DAPI volume vs Nuclei count
ax = fig_qc.add_subplot(gs[3, :2])

ax.scatter(fov_q111['V_DAPI'], fov_q111['N_Nuclei'],
          c='steelblue', s=30, alpha=0.5, edgecolor='none')

# Fit line
valid = fov_q111[['V_DAPI', 'N_Nuclei']].dropna()
if len(valid) > 0:
    slope, intercept, r, p, _ = linregress(valid['V_DAPI'], valid['N_Nuclei'])
    x_range = np.linspace(valid['V_DAPI'].min(), valid['V_DAPI'].max(), 50)
    y_pred = slope * x_range + intercept
    ax.plot(x_range, y_pred, 'r--', linewidth=2,
           label=f'r={r:.3f}, p={p:.2e}')

ax.set_xlabel('DAPI Volume (μm³)', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Nuclei', fontsize=12, fontweight='bold')
ax.set_title('DAPI Volume vs Estimated Nuclei Count', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Row 3, right: Age vs Nuclei count
ax = fig_qc.add_subplot(gs[3, 2:])

for region, color, marker in [('Cortex', 'mediumpurple', 'o'),
                               ('Striatum', 'gold', 's')]:
    region_data = fov_q111[fov_q111['Region'] == region]
    ax.scatter(region_data['Age'], region_data['N_Nuclei'],
              c=color, marker=marker, s=50, alpha=0.6,
              edgecolor='black', linewidth=0.5, label=region)

ax.axhline(MIN_NUCLEI_THRESHOLD, color='red', linestyle='--', linewidth=2)
ax.set_xlabel('Age (months)', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Nuclei per FOV', fontsize=12, fontweight='bold')
ax.set_title('Nuclei Count vs Age', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

fig_qc.suptitle(f'Quality Control: Cells per FOV Analysis\n' +
               f'QC Threshold: N_nuclei ≥ {MIN_NUCLEI_THRESHOLD}',
               fontsize=16, fontweight='bold', y=0.995)

plt.savefig(OUTPUT_DIR / 'fig_cells_per_fov_qc.png', dpi=FIGURE_DPI, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / 'fig_cells_per_fov_qc.svg', dpi=FIGURE_DPI, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / 'fig_cells_per_fov_qc.pdf', dpi=FIGURE_DPI, bbox_inches='tight')
print(f"\nSaved: {OUTPUT_DIR / 'fig_cells_per_fov_qc.*'}")
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6: GENERATE COMPREHENSIVE CAPTION
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("SECTION 6: GENERATING CAPTION")
print("="*80)

# Collect statistics for caption
n_q111_slides = len(df_q111[slide_field].unique())
n_wt_slides = len(df_wt[slide_field].unique())
n_q111_fovs = len(fov_q111)
n_wt_fovs = len(fov_wt)
n_total_fovs = len(fov_all)
n_excluded_slides = len(EXCLUDED_SLIDES)
excluded_slides_str = ', '.join(sorted(EXCLUDED_SLIDES))

# Q111 slide list
q111_slides_list = sorted(df_q111[slide_field].unique())
q111_slides_str = ', '.join(q111_slides_list)

# WT slide list
wt_slides_list = sorted(df_wt[slide_field].unique())
wt_slides_str = ', '.join(wt_slides_list) if wt_slides_list else 'N/A'

# Get age and atlas coordinate ranges
q111_ages = sorted(fov_q111['Age'].dropna().unique())
wt_ages = sorted(fov_wt['Age'].dropna().unique()) if len(fov_wt) > 0 else []
q111_atlas_range = (fov_q111['Brain_Atlas_Coord'].min(), fov_q111['Brain_Atlas_Coord'].max())

# Get mouse IDs
q111_mouse_ids = sorted(fov_q111['Mouse_ID'].unique())
wt_mouse_ids = sorted(fov_wt['Mouse_ID'].unique()) if len(fov_wt) > 0 else []

# Peak intensity summary
spot_peaks_per_channel = {}
for ch in ['HTT1a', 'fl-HTT']:
    ch_data = fov_q111[fov_q111['Channel'] == ch]
    peak_values = ch_data['I_Single_Peak'].dropna().unique()
    spot_peaks_per_channel[ch] = {
        'n_slides': len(peak_values),
        'mean': np.mean(peak_values) if len(peak_values) > 0 else np.nan,
        'std': np.std(peak_values) if len(peak_values) > 0 else np.nan,
        'min': np.min(peak_values) if len(peak_values) > 0 else np.nan,
        'max': np.max(peak_values) if len(peak_values) > 0 else np.nan,
    }

# Intensity statistics
for ch in ['HTT1a', 'fl-HTT']:
    ch_data = fov_q111[fov_q111['Channel'] == ch]

    # Single spots
    all_single = []
    for intensities in ch_data['Single_Spot_Intensities']:
        if isinstance(intensities, list):
            all_single.extend(intensities)

    # Clusters
    all_cluster = []
    for intensities in ch_data['Cluster_Intensities']:
        if isinstance(intensities, list):
            all_cluster.extend(intensities)

    spot_peaks_per_channel[ch]['n_single_spots'] = len(all_single)
    spot_peaks_per_channel[ch]['n_clusters'] = len(all_cluster)
    if len(all_single) > 0:
        spot_peaks_per_channel[ch]['single_mean'] = np.mean(all_single)
        spot_peaks_per_channel[ch]['single_median'] = np.median(all_single)
    if len(all_cluster) > 0:
        spot_peaks_per_channel[ch]['cluster_mean'] = np.mean(all_cluster)
        spot_peaks_per_channel[ch]['cluster_median'] = np.median(all_cluster)

# Nuclei statistics
nuclei_stats = {
    'q111_mean': np.mean(fov_q111['N_Nuclei']),
    'q111_median': np.median(fov_q111['N_Nuclei']),
    'q111_std': np.std(fov_q111['N_Nuclei']),
    'q111_min': np.min(fov_q111['N_Nuclei']),
    'q111_max': np.max(fov_q111['N_Nuclei']),
}
if len(fov_wt) > 0:
    nuclei_stats['wt_mean'] = np.mean(fov_wt['N_Nuclei'])
    nuclei_stats['wt_median'] = np.median(fov_wt['N_Nuclei'])

# Build caption
caption_lines = [
    "Supplementary Figure: Normalization Verification and Quality Control",
    "",
    "This supplementary figure provides comprehensive quality control metrics and normalization verification",
    "for the RNAscope quantification of mutant fl-HTT mRNA in Q111 Huntington's disease mouse model tissue.",
    "",
    "DATA FILTERING AND QUALITY CONTROL:",
    f"- Dataset: Experimental samples from the {experimental_field} probe set",
    f"- Excluded slides (n={n_excluded_slides}): {excluded_slides_str}",
    f"  (Slides excluded due to poor UBC positive control expression indicating technical failures)",
    f"- CV threshold for cluster filtering: CV >= {CV_THRESHOLD} (to exclude uniform background artifacts)",
    f"- Minimum nuclei per FOV threshold: {MIN_NUCLEI_THRESHOLD}",
    f"- Intensity threshold: Per-slide, determined from negative control at quantile={QUANTILE_NEGATIVE_CONTROL}, max PFA={MAX_PFA}",
    "",
    "Q111 DATA SUMMARY:",
    f"- Q111 slides analyzed (n={n_q111_slides}): {q111_slides_str}",
    f"- Q111 FOVs: {n_q111_fovs}",
    f"- Q111 mice (n={len(q111_mouse_ids)}): {', '.join(q111_mouse_ids)}",
    f"- Q111 ages (months): {', '.join(str(a) for a in q111_ages)}",
    f"- Q111 atlas coordinate range: {q111_atlas_range[0]:.2f} to {q111_atlas_range[1]:.2f} mm",
    "",
    "WILDTYPE DATA SUMMARY:",
    f"- Wildtype slides analyzed (n={n_wt_slides}): {wt_slides_str}",
    f"- Wildtype FOVs: {n_wt_fovs}",
    f"- Wildtype mice (n={len(wt_mouse_ids)}): {', '.join(wt_mouse_ids) if wt_mouse_ids else 'N/A'}",
    f"- Wildtype ages (months): {', '.join(str(a) for a in wt_ages) if wt_ages else 'N/A'}",
    "",
    f"TOTAL FOVs ANALYZED: {n_total_fovs}",
    "",
    "VOXEL AND PIXEL PARAMETERS:",
    f"- Pixel size (XY): {pixelsize} nm",
    f"- Slice depth (Z): {slice_depth} nm",
    f"- Voxel size: {voxel_size} μm³",
    f"- Mean nuclear volume (for nuclei estimation): {mean_nuclear_volume} μm³",
    "",
]

# Add channel-specific statistics
for ch in ['HTT1a', 'fl-HTT']:
    s = spot_peaks_per_channel.get(ch, {})
    caption_lines.extend([
        f"{ch.upper()} CHANNEL STATISTICS:",
        f"- Single spots analyzed: {s.get('n_single_spots', 0):,}",
        f"- Clusters analyzed: {s.get('n_clusters', 0):,}",
        f"- Peak intensities from {s.get('n_slides', 0)} slides",
        f"- Peak intensity mean: {s.get('mean', np.nan):.1f} ± {s.get('std', np.nan):.1f} A.U.",
        f"- Peak intensity range: {s.get('min', np.nan):.1f} - {s.get('max', np.nan):.1f} A.U.",
        f"- Single spot intensity mean: {s.get('single_mean', np.nan):.1f} A.U.",
        f"- Single spot intensity median: {s.get('single_median', np.nan):.1f} A.U.",
        f"- Cluster intensity mean: {s.get('cluster_mean', np.nan):.1f} A.U.",
        f"- Cluster intensity median: {s.get('cluster_median', np.nan):.1f} A.U.",
        "",
    ])

caption_lines.extend([
    "NUCLEI PER FOV STATISTICS:",
    f"- Q111 mean nuclei per FOV: {nuclei_stats['q111_mean']:.1f} ± {nuclei_stats['q111_std']:.1f}",
    f"- Q111 median nuclei per FOV: {nuclei_stats['q111_median']:.1f}",
    f"- Q111 nuclei range: {nuclei_stats['q111_min']:.1f} - {nuclei_stats['q111_max']:.1f}",
])
if len(fov_wt) > 0:
    caption_lines.extend([
        f"- Wildtype mean nuclei per FOV: {nuclei_stats.get('wt_mean', np.nan):.1f}",
        f"- Wildtype median nuclei per FOV: {nuclei_stats.get('wt_median', np.nan):.1f}",
    ])

caption_lines.extend([
    "",
    "FIGURE PANELS:",
    "",
    "fig_normalization_verification:",
    "- Rows 0-1: Peak intensity vs Age for HTT1a and fl-HTT in Cortex and Striatum",
    "- Rows 2-3: Peak intensity vs Brain Atlas Coordinate for HTT1a and fl-HTT in Cortex and Striatum",
    "- Each point represents a slide, colored by mouse ID",
    "- Pearson correlation coefficients and p-values shown for each panel",
    "",
    "fig_intensity_distributions (per channel):",
    "- Row 0: Single spot intensity histograms with KDE for Cortex and Striatum",
    "- Row 1: Cluster intensity histograms with KDE for Cortex and Striatum",
    "- Row 2: Box plots comparing single vs clustered mRNA per cell, and total mRNA by region",
    "- Row 3: Total mRNA per cell vs Age and Brain Atlas Coordinate",
    "",
    "fig_cells_per_fov_qc:",
    "- Row 0: Distribution of nuclei per FOV by mouse model (Q111 vs WT) with box plots",
    "- Row 1: Distribution of nuclei per FOV by brain region (Cortex vs Striatum)",
    "- Row 2: Nuclei per FOV by individual mouse ID",
    "- Row 3: DAPI volume vs nuclei count correlation, and nuclei count vs age",
    "",
    "INTERPRETATION:",
    "- Peak intensity should be independent of age and atlas coordinate for valid normalization",
    "- Low correlation coefficients indicate consistent peak intensity across samples",
    "- Nuclei counts should be above the QC threshold for reliable per-cell quantification",
    "- Single spots represent individual mRNA molecules; clusters represent mRNA aggregates",
    "",
    f"Analysis performed with scienceplots style. CV threshold={CV_THRESHOLD}, Min nuclei={MIN_NUCLEI_THRESHOLD}.",
])

caption_text = '\n'.join(caption_lines)

# Save caption
caption_path = OUTPUT_DIR / 'fig_normalization_and_qc_caption.txt'
with open(caption_path, 'w') as f:
    f.write(caption_text)
print(f"Saved caption to: {caption_path}")

# Also save as LaTeX
caption_latex = caption_text.replace('_', '\\_').replace('%', '\\%').replace('μ', '$\\mu$').replace('³', '$^3$').replace('²', '$^2$')
caption_latex_path = OUTPUT_DIR / 'fig_normalization_and_qc_caption.tex'
with open(caption_latex_path, 'w') as f:
    f.write(caption_latex)
print(f"Saved LaTeX caption to: {caption_latex_path}")

print("\n" + "="*80)
print("ALL FIGURES COMPLETED SUCCESSFULLY")
print("="*80)
print(f"\nOutput directory: {OUTPUT_DIR}")
print("\nGenerated figures:")
print("  1. fig_normalization_verification.*")
print("  2. fig_intensity_distributions_HTT1a.*")
print("  3. fig_intensity_distributions_fl_HTT.*")
print("  4. fig_cells_per_fov_qc.*")
print("\nGenerated data:")
print("  1. fov_level_data.csv")
print("\nGenerated captions:")
print("  1. fig_normalization_and_qc_caption.txt")
print("  2. fig_normalization_and_qc_caption.tex")
