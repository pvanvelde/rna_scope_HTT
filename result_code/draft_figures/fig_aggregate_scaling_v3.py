"""
Aggregate Scaling Analysis Figure - Version 3
==============================================

Improvements over v2:
- Adaptive bin sizing (variable width bins with consistent point counts)
- Per-slide minimum requirement for bins
- Volume PDF panels
- Intensity PDF panels
- Expanded to 3x2 layout

Based on manuscript text and exact logic from size_single_spots_normalized.py
"""

import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, binned_statistic, pearsonr
import scienceplots
plt.style.use('science')
plt.rcParams['text.usetex'] = False
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import functions and config
from result_functions_v2 import (compute_thresholds, recursively_load_dict,
                                extract_dataframe, concatenate_fields,
                                concatenated_data_to_df)
from results_config import (
    PIXELSIZE,
    SLICE_DEPTH,
    VOXEL_SIZE,
    H5_FILE_PATHS_EXPERIMENTAL,
    CHANNELS_TO_ANALYZE,
    EXPERIMENTAL_FIELD,
    NEGATIVE_CONTROL_FIELD,
    SLIDE_FIELD,
    MAX_PFA,
    QUANTILE_NEGATIVE_CONTROL,
    CV_THRESHOLD,
    CHANNEL_PARAMS,
    SIGMA_Z_XLIM,
    BEAD_PSF_Z,
    EXCLUDED_SLIDES,
    SLIDE_LABEL_MAP_Q111,
    SLIDE_LABEL_MAP_WT
)

# Constants
pixelsize = PIXELSIZE  # nm
slice_depth = SLICE_DEPTH  # nm
voxel_size = VOXEL_SIZE  # μm³

# Create output directory
OUTPUT_DIR = Path(__file__).parent / "output" / "aggregate_scaling"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*70)
print("AGGREGATE SCALING ANALYSIS FIGURE V3")
print("="*70)
print(f"\nOutput directory: {OUTPUT_DIR}")

# ══════════════════════════════════════════════════════════════════════════
# LOAD DATA - SAME AS V2
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("LOADING DATA")
print("="*70)

h5_file_paths = H5_FILE_PATHS_EXPERIMENTAL

with h5py.File(h5_file_paths[0], 'r') as h5_file:
    data_dict = recursively_load_dict(h5_file)

print(f"Loaded data from: {h5_file_paths[0]}")

desired_channels = CHANNELS_TO_ANALYZE
fields_to_extract = ['spots_sigma_var.params_raw', 'spots.params_raw',
                    'cluster_intensities', 'label_sizes', 'cluster_cvs',
                    'spots.final_filter', 'spots.pfa_values', 'spots.photons']

df_extracted_full = extract_dataframe(
    data_dict,
    field_keys=fields_to_extract,
    channels=desired_channels,
    include_file_metadata_sample=True,
    explode_fields=[]
)

print(f"Extracted dataframe shape: {df_extracted_full.shape}")

# ══════════════════════════════════════════════════════════════════════════
# COMPUTE THRESHOLDS
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("COMPUTING THRESHOLDS")
print("="*70)

(thresholds, thresholds_cluster,
 error_thresholds, error_thresholds_cluster,
 number_of_datapoints, age) = compute_thresholds(
    df_extracted=df_extracted_full,
    slide_field=SLIDE_FIELD,
    desired_channels=desired_channels,
    negative_control_field=NEGATIVE_CONTROL_FIELD,
    experimental_field=EXPERIMENTAL_FIELD,
    quantile_negative_control=QUANTILE_NEGATIVE_CONTROL,
    max_pfa=MAX_PFA,
    plot=False,
    n_bootstrap=1,
    use_region=False,
    use_final_filter=True
)

print(f"Computed thresholds for {len(thresholds)} combinations")

# Build threshold lookup and merge
thr_rows = []
for (slide, channel, area), vec in error_thresholds.items():
    thr_rows.append({"slide": slide, "channel": channel, "thr": np.mean(vec)})
thr_df = pd.DataFrame(thr_rows).drop_duplicates(["slide", "channel"])

df_extracted_full = df_extracted_full.merge(
    thr_df, how="left",
    left_on=[SLIDE_FIELD, "channel"],
    right_on=["slide", "channel"]
)
df_extracted_full.rename(columns={"thr": "threshold"}, inplace=True)
df_extracted_full.drop(columns=["slide"], inplace=True, errors='ignore')

# Filter to Q111 experimental data
df_exp = df_extracted_full[df_extracted_full['metadata_sample_Probe-Set'] == EXPERIMENTAL_FIELD].copy()
df_exp = df_exp[df_exp['metadata_sample_Mouse_Model'] == 'Q111'].copy()

# Exclude problematic slides
df_exp = df_exp[~df_exp[SLIDE_FIELD].isin(EXCLUDED_SLIDES)].copy()
print(f"Excluded slides: {EXCLUDED_SLIDES}")
print(f"Experimental data (Q111, excluding {len(EXCLUDED_SLIDES)} slides): {len(df_exp)} FOVs")

# Build slide_name_std to mouse_ID mapping (Q111)
slide_to_mouse_id = df_exp[[SLIDE_FIELD, 'metadata_sample_mouse_ID']].drop_duplicates()
slide_to_mouse_id = dict(zip(slide_to_mouse_id[SLIDE_FIELD], slide_to_mouse_id['metadata_sample_mouse_ID']))
print(f"Built slide to mouse_ID mapping for {len(slide_to_mouse_id)} Q111 slides")

# Build slide_name_std to age mapping (Q111)
slide_to_age = df_exp[[SLIDE_FIELD, 'metadata_sample_Age']].drop_duplicates()
slide_to_age = dict(zip(slide_to_age[SLIDE_FIELD], slide_to_age['metadata_sample_Age']))
print(f"Built slide to age mapping for {len(slide_to_age)} Q111 slides")

# Concatenate fields
concatenated_data = concatenate_fields(
    df_extracted=df_exp,
    slide_field=SLIDE_FIELD,
    desired_channels=desired_channels,
    fields_to_extract=fields_to_extract,
    probe_set_field='metadata_sample_Probe-Set',
)
df_groups = concatenated_data_to_df(concatenated_data)

print(f"Grouped data: {len(df_groups)} groups")

# ══════════════════════════════════════════════════════════════════════════
# EXTRACT CLUSTERS
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("EXTRACTING SINGLE SPOTS AND CLUSTERS")
print("="*70)

# Compute sigma_params from CHANNEL_PARAMS break_sigma values
# break_sigma is in pixels, convert to nm using PIXELSIZE (xy) and SLICE_DEPTH (z)
sigma_params = {}
for ch in ['green', 'orange']:
    bs = CHANNEL_PARAMS[ch]['break_sigma']
    sigma_params[ch] = {
        'bp_x': bs[0] * PIXELSIZE,  # nm
        'bp_y': bs[1] * PIXELSIZE,  # nm
        'bp_z': bs[2] * SLICE_DEPTH  # nm
    }
print(f"Using sigma_params from CHANNEL_PARAMS:")
for ch, params in sigma_params.items():
    print(f"  {ch}: bp_x={params['bp_x']:.1f}nm, bp_y={params['bp_y']:.1f}nm, bp_z={params['bp_z']:.1f}nm")

max_pfa = MAX_PFA
use_sigma_final_filter = True

single_rows = []
cluster_rows = []

for idx, row in df_groups.iterrows():
    slide = row["slide"]
    channel = row["channel"]
    probe_set = row["probe_set"]
    region = row["region"]

    components = [str(slide), str(channel)]
    threshold_val = next((v for k, v in thresholds.items()
                          if all(comp in str(k) for comp in components)), None)
    threshold_val_cluster = next((v for k, v in thresholds_cluster.items()
                                  if all(comp in str(k) for comp in components)), None)

    if threshold_val is None:
        continue

    if not use_sigma_final_filter:
        mask = row["spots.pfa_values"] < max_pfa
        row["spots.final_filter"] = np.any(mask, axis=1)

    cluster_data = row["cluster_intensities"]
    cluster_cvs = row.get("cluster_cvs", None)
    final_filter = row["spots.final_filter"]

    if final_filter is None or not np.any(final_filter):
        continue

    photon_arr_sig = row["spots_sigma_var.params_raw"][final_filter, 3]
    sigma = row["spots_sigma_var.params_raw"][final_filter, 5::]
    label_sizes = row["label_sizes"]

    calib = sigma_params.get(channel.lower())
    single_mask = photon_arr_sig > threshold_val

    # Cluster filtering: intensity threshold AND CV threshold
    # CV data is required - no fallback
    intensity_mask = cluster_data > threshold_val
    if cluster_cvs is None or len(cluster_cvs) != len(cluster_data):
        raise ValueError(f"CV data missing or mismatched for cluster filtering")
    cv_mask = cluster_cvs >= CV_THRESHOLD
    cluster_mask = intensity_mask & cv_mask

    single_rows.extend(dict(
        slide=slide, channel=channel, probe_set=probe_set, region=region,
        photons=photon_arr_sig[i], sigma_x=sigma[i, 0], sigma_y=sigma[i, 1], sigma_z=sigma[i, 2]
    ) for i in np.where(single_mask)[0])

    cluster_rows.extend(dict(
        slide=slide, channel=channel, probe_set=probe_set, region=region,
        cluster_volume=label_sizes[i], cluster_intensity=cluster_data[i],
        cluster_cv=cluster_cvs[i] if cluster_cvs is not None else np.nan
    ) for i in np.where(cluster_mask)[0])

df_single = pd.DataFrame(single_rows)
df_clusters = pd.DataFrame(cluster_rows)

print(f"Single spots (Q111): {df_single.shape}")
print(f"Clusters (Q111): {df_clusters.shape}")

# ══════════════════════════════════════════════════════════════════════════
# EXTRACT WILDTYPE SINGLE SPOTS (for peak intensity computation only)
# ══════════════════════════════════════════════════════════════════════════

print("\nExtracting Wildtype single spots for peak intensity computation...")

# Get Wildtype experimental data
df_wt = df_extracted_full[df_extracted_full['metadata_sample_Probe-Set'] == EXPERIMENTAL_FIELD].copy()
df_wt = df_wt[df_wt['metadata_sample_Mouse_Model'] == 'Wildtype'].copy()

print(f"Wildtype experimental data: {len(df_wt)} FOVs")

# Build Wildtype slide mappings
if len(df_wt) > 0:
    slide_to_mouse_id_wt = df_wt[[SLIDE_FIELD, 'metadata_sample_mouse_ID']].drop_duplicates()
    slide_to_mouse_id_wt = dict(zip(slide_to_mouse_id_wt[SLIDE_FIELD], slide_to_mouse_id_wt['metadata_sample_mouse_ID']))
    slide_to_age_wt = df_wt[[SLIDE_FIELD, 'metadata_sample_Age']].drop_duplicates()
    slide_to_age_wt = dict(zip(slide_to_age_wt[SLIDE_FIELD], slide_to_age_wt['metadata_sample_Age']))
    print(f"Built slide mappings for {len(slide_to_mouse_id_wt)} Wildtype slides")
else:
    slide_to_mouse_id_wt = {}
    slide_to_age_wt = {}

if len(df_wt) > 0:
    # Concatenate fields for Wildtype
    concatenated_data_wt = concatenate_fields(
        df_extracted=df_wt,
        slide_field=SLIDE_FIELD,
        desired_channels=desired_channels,
        fields_to_extract=fields_to_extract,
        probe_set_field='metadata_sample_Probe-Set',
    )
    df_groups_wt = concatenated_data_to_df(concatenated_data_wt)

    # Extract single spots and clusters from Wildtype
    single_rows_wt = []
    cluster_rows_wt = []
    for idx, row in df_groups_wt.iterrows():
        slide = row["slide"]
        channel = row["channel"]
        probe_set = row["probe_set"]
        region = row["region"]

        components = [str(slide), str(channel)]
        threshold_val = next((v for k, v in thresholds.items()
                              if all(comp in str(k) for comp in components)), None)

        if threshold_val is None:
            continue

        final_filter = row["spots.final_filter"]
        if final_filter is None or not np.any(final_filter):
            continue

        photon_arr_sig = row["spots_sigma_var.params_raw"][final_filter, 3]
        sigma = row["spots_sigma_var.params_raw"][final_filter, 5::]
        cluster_data = row["cluster_intensities"]
        cluster_cvs = row.get("cluster_cvs", None)
        label_sizes = row["label_sizes"]

        single_mask = photon_arr_sig > threshold_val

        # Cluster filtering: intensity threshold AND CV threshold
        # CV data is required - no fallback
        intensity_mask = cluster_data > threshold_val
        if cluster_cvs is None or len(cluster_cvs) != len(cluster_data):
            raise ValueError(f"CV data missing or mismatched for cluster filtering")
        cv_mask = cluster_cvs >= CV_THRESHOLD
        cluster_mask = intensity_mask & cv_mask

        single_rows_wt.extend(dict(
            slide=slide, channel=channel, probe_set=probe_set, region=region,
            photons=photon_arr_sig[i], sigma_x=sigma[i, 0], sigma_y=sigma[i, 1], sigma_z=sigma[i, 2]
        ) for i in np.where(single_mask)[0])

        cluster_rows_wt.extend(dict(
            slide=slide, channel=channel, probe_set=probe_set, region=region,
            cluster_volume=label_sizes[i], cluster_intensity=cluster_data[i],
            cluster_cv=cluster_cvs[i] if cluster_cvs is not None else np.nan
        ) for i in np.where(cluster_mask)[0])

    df_single_wt = pd.DataFrame(single_rows_wt)
    df_clusters_wt = pd.DataFrame(cluster_rows_wt)
    print(f"Single spots (Wildtype): {df_single_wt.shape}")
    print(f"Clusters (Wildtype): {df_clusters_wt.shape}")

    # Combine Q111 and Wildtype single spots for peak intensity computation
    df_single_all = pd.concat([df_single, df_single_wt], ignore_index=True)
    print(f"Single spots (combined for peak intensity): {df_single_all.shape}")
else:
    df_single_all = df_single
    df_clusters_wt = pd.DataFrame()
    print("No Wildtype data found, using Q111 only for peak intensities")

# ══════════════════════════════════════════════════════════════════════════
# COMPUTE PEAK INTENSITIES
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("COMPUTING SINGLE SPOT PEAK INTENSITIES")
print("="*70)

def compute_single_spot_peak_intensities(df_single, sigma_col, int_col,
                                        n_bins=50, xlim=(100, 600), scaling=pixelsize):
    peak_intensities = {}
    for (slide, p_set, ch), sub in df_single.groupby(["slide", "probe_set", "channel"]):
        width = sub[sigma_col].to_numpy() * scaling
        intens = sub[int_col].to_numpy()

        lo, hi = xlim
        mask = (width >= lo) & (width <= hi)
        width_filt, intens_filt = width[mask], intens[mask]

        if width_filt.size < 50:
            continue

        try:
            kde_intensity = gaussian_kde(intens_filt, bw_method='scott')
            intensity_range = np.linspace(np.percentile(intens_filt, 1),
                                         np.percentile(intens_filt, 99), 500)
            pdf_values = kde_intensity(intensity_range)
            peak_idx = np.argmax(pdf_values)
            peak_intensity = intensity_range[peak_idx]

            if peak_intensity > 0 and np.isfinite(peak_intensity):
                peak_intensities[(slide, p_set, ch)] = peak_intensity
        except:
            continue

    return peak_intensities

peak_intensities_for_clusters = compute_single_spot_peak_intensities(
    df_single_all, sigma_col="sigma_z", int_col="photons",
    n_bins=50, xlim=SIGMA_Z_XLIM, scaling=slice_depth
)
print(f"Using sigma_z xlim from config: {SIGMA_Z_XLIM}")

print(f"Computed peak intensities for {len(peak_intensities_for_clusters)} combinations")

# ══════════════════════════════════════════════════════════════════════════
# SAVE PEAK INTENSITIES (N_1mRNA) PER SLIDE TO CSV
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("SAVING PEAK INTENSITIES (N_1mRNA) PER SLIDE")
print("="*70)

# Convert peak intensities dict to DataFrame
peak_rows = []
for (slide, probe_set, channel), peak_intensity in peak_intensities_for_clusters.items():
    peak_rows.append({
        'slide': slide,
        'probe_set': probe_set,
        'channel': channel,
        'peak_intensity_photons': peak_intensity,
        'description': f'Single mRNA peak intensity (KDE mode) from sigma_z filtered spots ({SIGMA_Z_XLIM[0]:.1f}-{SIGMA_Z_XLIM[1]:.1f}nm width)'
    })

df_peak_intensities = pd.DataFrame(peak_rows)

# Sort by slide, channel
df_peak_intensities = df_peak_intensities.sort_values(['slide', 'channel']).reset_index(drop=True)

# Save to CSV
peak_csv_path = OUTPUT_DIR / 'peak_intensities_per_slide.csv'
df_peak_intensities.to_csv(peak_csv_path, index=False)
print(f"Saved peak intensities to: {peak_csv_path}")

# Print summary
print(f"\nPeak intensities per slide (N_1mRNA in photons):")
print("-" * 60)
for channel in ['green', 'orange']:
    ch_data = df_peak_intensities[df_peak_intensities['channel'] == channel]
    if len(ch_data) > 0:
        print(f"\n{channel.upper()} channel ({len(ch_data)} slides):")
        print(f"  Mean:   {ch_data['peak_intensity_photons'].mean():.1f} photons")
        print(f"  Median: {ch_data['peak_intensity_photons'].median():.1f} photons")
        print(f"  Std:    {ch_data['peak_intensity_photons'].std():.1f} photons")
        print(f"  Range:  {ch_data['peak_intensity_photons'].min():.1f} - {ch_data['peak_intensity_photons'].max():.1f} photons")

# ══════════════════════════════════════════════════════════════════════════
# PROCESS CLUSTERS WITH ADAPTIVE BINNING
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("PROCESSING CLUSTERS WITH ADAPTIVE BINNING")
print("="*70)

scaling_vol = voxel_size  # μm³ per voxel

# Parameters
MIN_FRACTION_OF_TOTAL = 0.02  # Minimum fraction of total counts per bin (2%)
MIN_SLIDES_PER_BIN = 3  # Minimum number of slides contributing to bin
MIN_BIN_WIDTH = 0.5  # Minimum bin width in μm³
xlim_vol = (0, 30)  # Volume range in μm³
ylim_int = (0, 80)  # Intensity range in mRNA equivalents

def create_adaptive_bins_growing(data, min_points, min_width, xlim):
    """
    Create adaptive bins by growing from left to right.
    Each bin satisfies: (width >= min_width) OR (count >= min_points)
    Bins are wider where data is sparse, narrower where data is dense.
    """
    # Filter and sort data
    data_sorted = np.sort(data[(data >= xlim[0]) & (data <= xlim[1])])

    if len(data_sorted) < min_points:
        # Not enough data for even one bin
        return np.array([xlim[0], xlim[1]])

    bins = [xlim[0]]
    current_idx = 0

    while current_idx < len(data_sorted):
        # Grow bin until we satisfy: (width >= min_width) OR (count >= min_points)
        next_idx = current_idx + min_points

        if next_idx >= len(data_sorted):
            # Last bin - just extend to the end
            break

        # Get candidate bin edge
        bin_edge = data_sorted[next_idx]
        bin_width = bin_edge - bins[-1]

        # Check if bin is wide enough OR has enough points
        if bin_width >= min_width:
            # Bin is wide enough, accept it even if we have fewer points
            # Find the actual edge that gives us min_width
            target_edge = bins[-1] + min_width
            # Find closest data point >= target_edge
            idx = np.searchsorted(data_sorted[current_idx:], target_edge, side='left')
            if idx < len(data_sorted[current_idx:]):
                actual_idx = current_idx + idx
                bin_edge = data_sorted[actual_idx]
                if bin_edge > bins[-1]:
                    bins.append(bin_edge)
                    current_idx = actual_idx
                else:
                    current_idx += 1
            else:
                break
        elif next_idx < len(data_sorted) and bin_edge > bins[-1]:
            # Bin has enough points, accept it
            bins.append(bin_edge)
            current_idx = next_idx
        else:
            # Skip ahead to find a different value
            current_idx = next_idx + 1

    # Final edge
    bins.append(xlim[1])

    # Ensure strictly increasing and remove any remaining duplicates
    bins = np.unique(bins)

    # Make sure we have at least 2 bins (start and end)
    if len(bins) < 2:
        bins = np.array([xlim[0], xlim[1]])

    return bins

channel_results = {}

for ch in ['green', 'orange']:
    print(f"\nProcessing {ch} channel...")

    df_ch = df_clusters[df_clusters['channel'] == ch].copy()

    if len(df_ch) == 0:
        print(f"No data for {ch} channel")
        continue

    for p_set, sub in df_ch.groupby("probe_set"):
        slide_data = {}
        slide_correlations = {}
        slide_slopes = {}

        # Collect all data first
        all_volumes_raw = []
        all_intensities_raw = []
        all_slides = []
        all_regions = []

        for slide, slide_sub in sub.groupby("slide"):
            volumes_voxels = slide_sub['cluster_volume'].to_numpy()
            intensities_raw = slide_sub['cluster_intensity'].to_numpy()
            regions = slide_sub['region'].to_numpy()
            volumes = volumes_voxels * scaling_vol  # μm³

            peak_intensity = peak_intensities_for_clusters.get((slide, p_set, ch), None)
            if peak_intensity is None:
                continue

            intensities_norm = intensities_raw / peak_intensity

            # Filter to volume range
            lo, hi = xlim_vol
            mask = (volumes >= lo) & (volumes <= hi)
            volumes_filt = volumes[mask]
            intensities_filt = intensities_norm[mask]
            regions_filt = regions[mask]

            if len(volumes_filt) < 10:
                continue

            all_volumes_raw.extend(volumes_filt)
            all_intensities_raw.extend(intensities_filt)
            all_slides.extend([slide] * len(volumes_filt))
            all_regions.extend(regions_filt)

        all_volumes = np.array(all_volumes_raw)
        all_intensities = np.array(all_intensities_raw)
        all_slides_arr = np.array(all_slides)
        all_regions_arr = np.array(all_regions)

        if len(all_volumes) == 0:
            continue

        # Create adaptive bins based on volume distribution
        # Bins grow wider where data is sparse to ensure each has enough points
        min_points_per_bin = int(np.ceil(len(all_volumes) * MIN_FRACTION_OF_TOTAL))
        bins = create_adaptive_bins_growing(all_volumes, min_points=min_points_per_bin,
                                           min_width=MIN_BIN_WIDTH, xlim=xlim_vol)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_widths = bins[1:] - bins[:-1]

        print(f"  Created {len(bins)-1} adaptive bins (widths: {bin_widths.min():.2f}-{bin_widths.max():.2f} μm³)")

        # Compute combined statistics
        mean_I_combined, _, _ = binned_statistic(all_volumes, all_intensities,
                                                 statistic="mean", bins=bins)
        counts_combined, _, _ = binned_statistic(all_volumes, all_intensities,
                                                statistic="count", bins=bins)

        # Count slides per bin
        slides_per_bin = []
        for i in range(len(bins) - 1):
            bin_mask = (all_volumes >= bins[i]) & (all_volumes < bins[i+1])
            unique_slides = np.unique(all_slides_arr[bin_mask])
            slides_per_bin.append(len(unique_slides))
        slides_per_bin = np.array(slides_per_bin)

        # Per-slide binning and std calculation
        slide_bin_means = []
        for slide, slide_sub in sub.groupby("slide"):
            volumes_voxels = slide_sub['cluster_volume'].to_numpy()
            intensities_raw = slide_sub['cluster_intensity'].to_numpy()
            volumes = volumes_voxels * scaling_vol

            peak_intensity = peak_intensities_for_clusters.get((slide, p_set, ch), None)
            if peak_intensity is None:
                continue

            intensities_norm = intensities_raw / peak_intensity

            lo, hi = xlim_vol
            mask = (volumes >= lo) & (volumes <= hi)
            volumes_filt = volumes[mask]
            intensities_filt = intensities_norm[mask]

            if len(volumes_filt) < 10:
                continue

            # Bin this slide's data
            slide_bin_mean, _, _ = binned_statistic(volumes_filt, intensities_filt,
                                                    statistic='mean', bins=bins)
            slide_bin_means.append(slide_bin_mean)

            # Store slide data
            slide_data[slide] = {
                'volumes_filt': volumes_filt,
                'intensities_norm': intensities_filt
            }

            # Compute linear fit for this slide
            if len(volumes_filt) >= 3:
                coeffs = np.polyfit(volumes_filt, intensities_filt, 1)
                slope = coeffs[0]
                r, p = pearsonr(volumes_filt, intensities_filt)
                r2 = r**2
                slide_slopes[slide] = slope
                slide_correlations[slide] = (r, p, r2)

        slide_bin_means = np.array(slide_bin_means)
        std_across_slides = np.nanstd(slide_bin_means, axis=0)

        # Filter bins for fitting (require minimum slides and finite values)
        # Note: bins already guaranteed to have enough total counts by construction
        valid_for_fit = (np.isfinite(mean_I_combined) &
                        (slides_per_bin >= MIN_SLIDES_PER_BIN))

        # Linear fit on ALL INDIVIDUAL CLUSTERS (not binned means)
        # This aggregates over all clusters directly for the regression
        if len(all_volumes) >= 10:
            coeffs = np.polyfit(all_volumes, all_intensities, 1)
            slope = coeffs[0]
            intercept = coeffs[1]

            # Compute R² on all individual data points
            y_pred = slope * all_volumes + intercept
            ss_res = np.sum((all_intensities - y_pred)**2)
            ss_tot = np.sum((all_intensities - np.mean(all_intensities))**2)
            r2_all = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        else:
            slope = 0
            intercept = 0
            r2_all = 0

        print(f"  {ch} channel:")
        print(f"    Slope (all clusters fit): β = {slope:.2f} molecules/μm³")
        print(f"    R² (all clusters): {r2_all:.3f}")
        print(f"    Bin criteria: (width ≥ {MIN_BIN_WIDTH} μm³) OR (count ≥ {MIN_FRACTION_OF_TOTAL*100:.0f}% = {min_points_per_bin:,})")
        print(f"    Number of adaptive bins: {len(bins)-1}")
        print(f"    Bins used in fit: {np.sum(valid_for_fit)}/{len(bins)-1} (≥{MIN_SLIDES_PER_BIN} slides)")
        print(f"    N slides: {len(slide_data)}")
        print(f"    N clusters: {len(all_volumes)}")

        # Store results
        channel_results[ch] = {
            'bins': bins,
            'bin_centers': bin_centers,
            'bin_widths': bin_widths,
            'mean_I_combined': mean_I_combined,
            'std_across_slides': std_across_slides,
            'counts_combined': counts_combined,
            'slides_per_bin': slides_per_bin,
            'slide_data': slide_data,
            'slope': slope,
            'intercept': intercept,
            'r2': r2_all,
            'min_points_per_bin': min_points_per_bin,
            'n_slides': len(slide_data),
            'n_clusters': len(all_volumes),
            'all_volumes': all_volumes,
            'all_intensities': all_intensities,
            'all_slides': all_slides_arr,
            'all_regions': all_regions_arr
        }

# ══════════════════════════════════════════════════════════════════════════
# CREATE FIGURE - 3x2 LAYOUT
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("CREATING FIGURE")
print("="*70)

fig = plt.figure(figsize=(20, 24), dpi=300)
gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)

channel_labels = {
    'green': 'HTT1a (488 nm)',
    'orange': 'fl-HTT (548 nm)'
}
# Short probe names for filenames
channel_probe_names = {
    'green': 'HTT1a',
    'orange': 'fl-HTT'
}
channel_colors = {'green': 'green', 'orange': 'orange'}

# Panels A & B: Volume vs Intensity for Green and Orange
for panel_idx, channel in enumerate(['green', 'orange']):
    ax = fig.add_subplot(gs[0, panel_idx])

    if channel not in channel_results:
        ax.text(0.5, 0.5, f'No data for {channel} channel',
               ha='center', va='center', fontsize=12)
        continue

    results = channel_results[channel]
    bin_centers = results['bin_centers']
    mean_I = results['mean_I_combined']
    std_slides = results['std_across_slides']
    counts = results['counts_combined']
    slides_per_bin = results['slides_per_bin']
    slide_data = results['slide_data']

    # Plot all bins (already sized to have enough data by construction)
    # Only filter for finite values and minimum slides
    valid_plot = (np.isfinite(mean_I) &
                  (slides_per_bin >= MIN_SLIDES_PER_BIN))
    valid_std = np.isfinite(std_slides)
    valid_combined = valid_plot & valid_std

    # Plot std across slides
    ax.fill_between(bin_centers[valid_combined],
                    mean_I[valid_combined] - std_slides[valid_combined],
                    mean_I[valid_combined] + std_slides[valid_combined],
                    alpha=0.25, color=channel_colors[channel],
                    label='±1 std (across slides)', zorder=5)

    # Plot mean curve
    ax.plot(bin_centers[valid_plot], mean_I[valid_plot],
           'o-', color=channel_colors[channel], linewidth=2.5, markersize=6,
           label='Mean', zorder=10)

    # Linear fit line (through mean binned curve)
    slope = results['slope']
    intercept = results['intercept']
    vol_ref = np.array([xlim_vol[0], xlim_vol[1]])
    intens_ref = slope * vol_ref + intercept
    ax.plot(vol_ref, intens_ref, 'k--', linewidth=2, alpha=0.7,
           label=f'Linear fit: β={slope:.2f} mRNA/μm³', zorder=8)

    # Formatting
    ax.set_xlabel('Aggregate Volume (μm³)', fontsize=12)
    ax.set_ylabel('Aggregate Intensity (mRNA equiv.)', fontsize=12)
    ax.set_title(f'({chr(65 + panel_idx)}) {channel_labels[channel]}: Intensity vs Volume',
                fontsize=14, fontweight='bold', loc='left')
    ax.set_xlim(xlim_vol)
    ax.set_ylim(ylim_int)
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Statistics box
    textstr = f'β = {results["slope"]:.2f} mRNA/μm³ (all clusters)\n'
    textstr += f'R² = {results["r2"]:.3f}\n'
    textstr += f'{results["n_slides"]} slides, {results["n_clusters"]:,} clusters\n'
    textstr += f'{len(bins)-1} adaptive bins'

    ax.text(0.98, 0.02, textstr, transform=ax.transAxes,
           fontsize=10, verticalalignment='bottom', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black'))

# Panels C & D: Volume PDFs
ax_vol_pdf = fig.add_subplot(gs[1, :])

for channel in ['green', 'orange']:
    if channel not in channel_results:
        continue

    results = channel_results[channel]
    volumes = results['all_volumes']

    # Compute KDE
    try:
        kde = gaussian_kde(volumes, bw_method='scott')
        x_range = np.linspace(0, 30, 300)
        y_density = kde(x_range)

        ax_vol_pdf.plot(x_range, y_density, '-', linewidth=2.5,
                       color=channel_colors[channel],
                       label=f'{channel_labels[channel]} ({len(volumes):,} clusters)',
                       alpha=0.8)
        ax_vol_pdf.fill_between(x_range, y_density, alpha=0.2,
                                color=channel_colors[channel])
    except:
        print(f"  Could not compute KDE for {channel} volume")

ax_vol_pdf.set_xlabel('Aggregate Volume (μm³)', fontsize=12)
ax_vol_pdf.set_ylabel('Probability Density', fontsize=12)
ax_vol_pdf.set_title('(C) Volume Distribution of Aggregates',
                     fontsize=14, fontweight='bold', loc='left')
ax_vol_pdf.set_xlim(0, 30)
ax_vol_pdf.legend(loc='upper right', fontsize=11, framealpha=0.9)
ax_vol_pdf.grid(True, alpha=0.3, linestyle='--')

# Panels E & F: Intensity PDFs
ax_int_pdf = fig.add_subplot(gs[2, :])

for channel in ['green', 'orange']:
    if channel not in channel_results:
        continue

    results = channel_results[channel]
    intensities = results['all_intensities']

    # Filter to reasonable range
    intensities_filt = intensities[(intensities >= 0) & (intensities <= 100)]

    # Compute KDE
    try:
        kde = gaussian_kde(intensities_filt, bw_method='scott')
        x_range = np.linspace(0, 100, 300)
        y_density = kde(x_range)

        ax_int_pdf.plot(x_range, y_density, '-', linewidth=2.5,
                       color=channel_colors[channel],
                       label=f'{channel_labels[channel]} (median={np.median(intensities_filt):.1f})',
                       alpha=0.8)
        ax_int_pdf.fill_between(x_range, y_density, alpha=0.2,
                                color=channel_colors[channel])
    except:
        print(f"  Could not compute KDE for {channel} intensity")

ax_int_pdf.set_xlabel('Aggregate Intensity (mRNA equivalents)', fontsize=12)
ax_int_pdf.set_ylabel('Probability Density', fontsize=12)
ax_int_pdf.set_title('(D) Intensity Distribution of Aggregates',
                     fontsize=14, fontweight='bold', loc='left')
ax_int_pdf.set_xlim(0, 100)
ax_int_pdf.legend(loc='upper right', fontsize=11, framealpha=0.9)
ax_int_pdf.grid(True, alpha=0.3, linestyle='--')

# Overall title
fig.suptitle('Aggregate Intensity-Volume Scaling with Adaptive Binning and Distribution Analysis',
            fontsize=18, fontweight='bold', y=0.995)

# Save figure
output_files = []
for ext in ['pdf', 'svg', 'png']:
    filepath = OUTPUT_DIR / f'fig_aggregate_scaling_v3.{ext}'
    plt.savefig(filepath, format=ext, bbox_inches='tight', dpi=300)
    output_files.append(filepath)
    print(f"Saved: {filepath}")

plt.close()

# ══════════════════════════════════════════════════════════════════════════
# CREATE CLUSTER METRICS VIOLIN PLOTS (4 panels per channel)
# With anonymized mouse IDs (Q#1, Q#2, W#1, W#2)
# Metrics: Clusters per nucleus, mRNA per Cluster, Density, Total mRNA per nucleus
# Using FOV-level data from fov_level_data.csv (same as figure3.py and figure4.py)
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("CREATING CLUSTER METRICS FIGURE")
print("="*70)

# Load FOV-level data (pre-computed with actual DAPI nucleus counts)
FOV_DATA_PATH = OUTPUT_DIR.parent / 'expression_analysis_q111' / 'fov_level_data.csv'
print(f"Loading FOV-level data from: {FOV_DATA_PATH}")
df_fov = pd.read_csv(FOV_DATA_PATH)

# Filter and exclude problematic slides
df_fov = df_fov[~df_fov['Slide'].isin(EXCLUDED_SLIDES)].copy()

# Separate Q111 and WT
df_fov_q111 = df_fov[df_fov['Mouse_Model'] == 'Q111'].copy()
df_fov_wt = df_fov[df_fov['Mouse_Model'] == 'Wildtype'].copy()

print(f"  Q111 FOVs after exclusion: {len(df_fov_q111)}")
print(f"  WT FOVs after exclusion: {len(df_fov_wt)}")

# Channel mapping: CSV uses 'mHTT1a' and 'full-length mHTT', we use 'green' and 'orange'
CHANNEL_MAP = {'green': 'mHTT1a', 'orange': 'full-length mHTT'}

# Axis limits for violin plots
DENSITY_YLIM = (0, 6)
MRNA_YLIM = (0, 80)

# Create 4-panel figure for each channel
for channel in ['green', 'orange']:
    print(f"\nProcessing {channel} channel...")

    if channel not in channel_results:
        print(f"  No data for {channel} channel, skipping")
        continue

    # Get channel name for FOV CSV
    channel_csv = CHANNEL_MAP[channel]

    # Filter FOV data by channel
    df_fov_q111_ch = df_fov_q111[df_fov_q111['Channel'] == channel_csv].copy()
    df_fov_wt_ch = df_fov_wt[df_fov_wt['Channel'] == channel_csv].copy()

    print(f"  Q111 FOVs for {channel} ({channel_csv}): {len(df_fov_q111_ch)}")
    print(f"  WT FOVs for {channel} ({channel_csv}): {len(df_fov_wt_ch)}")

    # Get Q111 cluster data for mRNA per cluster and density
    results = channel_results[channel]
    all_volumes_q111 = results['all_volumes']
    all_intensities_q111 = results['all_intensities']
    all_slides_q111 = results['all_slides']

    # Filter out excluded slides for cluster data
    exclude_mask_q111 = np.isin(all_slides_q111, EXCLUDED_SLIDES)
    keep_mask_q111 = ~exclude_mask_q111
    volumes_q111 = all_volumes_q111[keep_mask_q111]
    intensities_q111 = all_intensities_q111[keep_mask_q111]
    slides_q111 = all_slides_q111[keep_mask_q111]
    density_q111 = intensities_q111 / volumes_q111

    print(f"  Q111: {len(intensities_q111):,} clusters after excluding {EXCLUDED_SLIDES}")

    # Get Wildtype cluster data for mRNA per cluster and density
    if len(df_clusters_wt) > 0:
        df_wt_ch = df_clusters_wt[df_clusters_wt['channel'] == channel].copy()

        wt_volumes_list = []
        wt_intensities_list = []
        wt_slides_list = []

        for slide in df_wt_ch['slide'].unique():
            if slide in EXCLUDED_SLIDES:
                continue

            slide_data = df_wt_ch[df_wt_ch['slide'] == slide]
            volumes_voxels = slide_data['cluster_volume'].values
            intensities_raw = slide_data['cluster_intensity'].values
            volumes = volumes_voxels * scaling_vol

            p_set = EXPERIMENTAL_FIELD
            peak_intensity = peak_intensities_for_clusters.get((slide, p_set, channel), None)
            if peak_intensity is None:
                continue

            intensities_norm = intensities_raw / peak_intensity

            wt_volumes_list.extend(volumes)
            wt_intensities_list.extend(intensities_norm)
            wt_slides_list.extend([slide] * len(volumes))

        volumes_wt = np.array(wt_volumes_list)
        intensities_wt = np.array(wt_intensities_list)
        slides_wt = np.array(wt_slides_list)
        density_wt = intensities_wt / volumes_wt if len(volumes_wt) > 0 else np.array([])
        print(f"  WT: {len(intensities_wt):,} clusters")
    else:
        volumes_wt = np.array([])
        intensities_wt = np.array([])
        slides_wt = np.array([])
        density_wt = np.array([])

    # Get unique slides from FOV data (which has actual nucleus counts)
    q111_unique_slides = sorted(df_fov_q111_ch['Slide'].unique())
    wt_unique_slides = sorted(df_fov_wt_ch['Slide'].unique()) if len(df_fov_wt_ch) > 0 else []

    # Build slide to age and mouse_id mapping from FOV data
    slide_to_age_fov = dict(zip(df_fov_q111_ch['Slide'], df_fov_q111_ch['Age']))
    slide_to_age_wt_fov = dict(zip(df_fov_wt_ch['Slide'], df_fov_wt_ch['Age'])) if len(df_fov_wt_ch) > 0 else {}
    slide_to_mouse_id_fov = dict(zip(df_fov_q111_ch['Slide'], df_fov_q111_ch['Mouse_ID']))
    slide_to_mouse_id_wt_fov = dict(zip(df_fov_wt_ch['Slide'], df_fov_wt_ch['Mouse_ID'])) if len(df_fov_wt_ch) > 0 else {}

    # Helper to extract numeric label for sorting
    def get_label_num(slide, label_map):
        label = label_map.get(slide, '#999')
        return int(label.replace('#', ''))

    # Build sub-index mapping: Q#1.1, Q#1.2, etc. for slides within same mouse
    def build_slide_sublabels(slides, label_map, prefix):
        """Create labels with sub-indices for each slide within a mouse."""
        sorted_slides = sorted(slides, key=lambda s: (get_label_num(s, label_map), s))
        sublabels = {}
        current_mouse = None
        sub_idx = 0
        for slide in sorted_slides:
            mouse_num = label_map.get(slide, '#?')
            if mouse_num != current_mouse:
                current_mouse = mouse_num
                sub_idx = 1
            else:
                sub_idx += 1
            sublabels[slide] = f"{prefix}{mouse_num}.{sub_idx}"
        return sublabels

    # Sort by numeric label from fixed mapping for consistent ordering
    q111_unique_slides_sorted = sorted(q111_unique_slides,
                                        key=lambda s: (get_label_num(s, SLIDE_LABEL_MAP_Q111), s))
    wt_unique_slides_sorted = sorted(wt_unique_slides,
                                      key=lambda s: (get_label_num(s, SLIDE_LABEL_MAP_WT), s))

    # Use sublabel mappings (matching Figure 3 format: Q#1.1, Q#1.2, etc.)
    mouse_id_map_q111 = build_slide_sublabels(q111_unique_slides, SLIDE_LABEL_MAP_Q111, 'Q')
    mouse_id_map_wt = build_slide_sublabels(wt_unique_slides, SLIDE_LABEL_MAP_WT, 'W')

    # Get per-FOV metrics from CSV (using actual DAPI nucleus counts)
    # Each row in df_fov is one FOV with pre-computed Clustered_mRNA_per_Cell and Clusters_per_Cell
    q111_fov_clusters_per_nucleus = {}  # {slide: array of Clusters_per_Cell values}
    q111_fov_mrna_per_nucleus = {}  # {slide: array of Clustered_mRNA_per_Cell values}

    for slide in q111_unique_slides_sorted:
        slide_fovs = df_fov_q111_ch[df_fov_q111_ch['Slide'] == slide]
        q111_fov_clusters_per_nucleus[slide] = slide_fovs['Clusters_per_Cell'].values
        q111_fov_mrna_per_nucleus[slide] = slide_fovs['Clustered_mRNA_per_Cell'].values

    # Get per-FOV metrics for WT from CSV
    wt_fov_clusters_per_nucleus = {}
    wt_fov_mrna_per_nucleus = {}

    for slide in wt_unique_slides_sorted:
        slide_fovs = df_fov_wt_ch[df_fov_wt_ch['Slide'] == slide]
        wt_fov_clusters_per_nucleus[slide] = slide_fovs['Clusters_per_Cell'].values
        wt_fov_mrna_per_nucleus[slide] = slide_fovs['Clustered_mRNA_per_Cell'].values

    # Compute per-mouse cluster counts for reference
    q111_cluster_counts = {}
    for slide in q111_unique_slides_sorted:
        mask = slides_q111 == slide
        q111_cluster_counts[slide] = np.sum(mask)

    wt_cluster_counts = {}
    for slide in wt_unique_slides_sorted:
        mask = slides_wt == slide
        wt_cluster_counts[slide] = np.sum(mask)

    # Create figure with 4 subplots (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(20, 14), dpi=300)

    panel_labels = ['A', 'B', 'C', 'D']

    # Helper function to create violin/bar plots with age grouping
    def create_grouped_plot(ax, panel_idx, metric_name, ylabel, ylim, plot_type,
                           q111_data_dict, wt_data_dict, is_distribution=True, min_points=10):
        """Create violin or bar plot with age grouping and Q111/WT separation.

        min_points: minimum data points required per slide for violin plots.
                    Use 10 for cluster-level data, 1 for FOV-level data.
        """

        positions = []
        labels = []
        colors = []
        data_or_heights = []

        pos = 0
        age_group_info = []
        age_separator_positions = []

        # Q111 mice grouped by age (using slide_to_age_fov from FOV CSV data)
        q111_ages = sorted(set(slide_to_age_fov.get(s) for s in q111_unique_slides_sorted if s in slide_to_age_fov))

        for age in q111_ages:
            group_start = pos
            slides_for_age = [s for s in q111_unique_slides_sorted if slide_to_age_fov.get(s) == age]

            for slide in slides_for_age:
                if is_distribution:
                    vals = q111_data_dict.get(slide, np.array([]))
                    if len(vals) < min_points:
                        raise ValueError(f"Slide {slide} has only {len(vals)} data points, need at least {min_points}")
                    if ylim is not None:
                        vals = vals[vals <= ylim[1]]
                    if len(vals) < min_points:
                        raise ValueError(f"Slide {slide} has only {len(vals)} data points after ylim filter, need at least {min_points}")
                    data_or_heights.append(vals)
                else:
                    data_or_heights.append(q111_data_dict.get(slide, 0))

                positions.append(pos)
                labels.append(mouse_id_map_q111[slide])
                colors.append(channel_colors[channel])
                pos += 1

            if pos > group_start:
                age_group_info.append((group_start, pos - 1, f"Q111\n{age}mo"))
                age_separator_positions.append(pos - 0.5)
                pos += 1

        if age_separator_positions:
            age_separator_positions.pop()

        q111_wt_separator = pos - 1.5

        # Wildtype mice grouped by age (using slide_to_age_wt_fov from FOV CSV data)
        if len(wt_unique_slides_sorted) > 0:
            wt_ages = sorted(set(slide_to_age_wt_fov.get(s) for s in wt_unique_slides_sorted if s in slide_to_age_wt_fov))

            for age in wt_ages:
                group_start = pos
                slides_for_age = [s for s in wt_unique_slides_sorted if slide_to_age_wt_fov.get(s) == age]

                for slide in slides_for_age:
                    if is_distribution:
                        vals = wt_data_dict.get(slide, np.array([]))
                        if len(vals) < min_points:
                            raise ValueError(f"WT Slide {slide} has only {len(vals)} data points, need at least {min_points}")
                        if ylim is not None:
                            vals = vals[vals <= ylim[1]]
                        if len(vals) < min_points:
                            raise ValueError(f"WT Slide {slide} has only {len(vals)} data points after ylim filter, need at least {min_points}")
                        data_or_heights.append(vals)
                    else:
                        data_or_heights.append(wt_data_dict.get(slide, 0))

                    positions.append(pos)
                    labels.append(mouse_id_map_wt[slide])
                    colors.append('gray')
                    pos += 1

                if pos > group_start:
                    age_group_info.append((group_start, pos - 1, f"WT\n{age}mo"))
                    age_separator_positions.append(pos - 0.5)
                    pos += 1

            if age_separator_positions and age_separator_positions[-1] > q111_wt_separator:
                age_separator_positions.pop()

        if len(positions) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=12)
            return

        # Plot
        if plot_type == 'violin':
            parts = ax.violinplot(data_or_heights, positions=positions, showmeans=True,
                                  showmedians=True, widths=0.8)
            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(colors[i])
                pc.set_edgecolor('black')
                pc.set_alpha(0.7)
            for partname in ['cmeans', 'cmedians', 'cbars', 'cmins', 'cmaxes']:
                if partname in parts:
                    parts[partname].set_edgecolor('black')
                    parts[partname].set_linewidth(1)
        else:  # bar
            ax.bar(positions, data_or_heights, color=colors, edgecolor='black', linewidth=0.5, alpha=0.8)

        # Separators
        for sep_pos in age_separator_positions:
            ax.axvline(sep_pos, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)

        if q111_wt_separator > 0 and len(wt_unique_slides_sorted) > 0:
            ax.axvline(q111_wt_separator, color='black', linestyle='--', linewidth=2.5, alpha=0.8)

        # Age group labels
        if plot_type == 'violin' and ylim is not None:
            y_label_pos = ylim[1] * 0.97
        else:
            max_val = max(data_or_heights) if not is_distribution else ylim[1] if ylim else 1
            if not is_distribution:
                max_val = max(data_or_heights) if data_or_heights else 1
            y_label_pos = max_val * 1.05 if plot_type == 'bar' else (ylim[1] * 0.97 if ylim else max_val * 0.97)

        for start_pos, end_pos, label in age_group_info:
            center = (start_pos + end_pos) / 2
            va = 'bottom' if plot_type == 'bar' else 'top'
            ax.text(center, y_label_pos, label,
                   ha='center', va=va, fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='gray'))

        # Formatting
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=45, fontsize=8, ha='right')
        ax.set_xlabel('Mouse', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f'({panel_labels[panel_idx]}) {metric_name}',
                    fontsize=12, fontweight='bold', loc='left')
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')

        if ylim is not None:
            ax.set_ylim(ylim)
        elif plot_type == 'bar':
            max_h = max(data_or_heights) if data_or_heights else 1
            ax.set_ylim(0, max_h * 1.25)

    # Prepare per-slide distribution data for cluster-level metrics
    q111_intensity_by_slide = {s: intensities_q111[slides_q111 == s] for s in q111_unique_slides_sorted}
    q111_density_by_slide = {s: density_q111[slides_q111 == s] for s in q111_unique_slides_sorted}

    wt_intensity_by_slide = {s: intensities_wt[slides_wt == s] for s in wt_unique_slides_sorted} if len(slides_wt) > 0 else {}
    wt_density_by_slide = {s: density_wt[slides_wt == s] for s in wt_unique_slides_sorted} if len(slides_wt) > 0 else {}

    # Panel A: Clusters per Nucleus per FOV (violin - distribution of FOV values)
    # min_points=1 because each slide only has a handful of FOVs
    create_grouped_plot(axes[0, 0], 0, 'Clusters per Nucleus (per FOV)', 'Clusters/nucleus', None, 'violin',
                       q111_fov_clusters_per_nucleus, wt_fov_clusters_per_nucleus, is_distribution=True, min_points=1)

    # Panel B: mRNA per Cluster (violin - distribution of cluster values)
    # min_points=10 because each slide has thousands of clusters
    create_grouped_plot(axes[0, 1], 1, 'mRNA per Cluster', 'mRNA equivalents', MRNA_YLIM, 'violin',
                       q111_intensity_by_slide, wt_intensity_by_slide, is_distribution=True, min_points=10)

    # Panel C: Density per Cluster (violin - distribution of cluster values)
    # min_points=10 because each slide has thousands of clusters
    create_grouped_plot(axes[1, 0], 2, 'Density (mRNA/μm³)', 'mRNA/μm³', DENSITY_YLIM, 'violin',
                       q111_density_by_slide, wt_density_by_slide, is_distribution=True, min_points=10)

    # Panel D: Total mRNA per Nucleus per FOV (violin - distribution of FOV values)
    # min_points=1 because each slide only has a handful of FOVs
    create_grouped_plot(axes[1, 1], 3, 'Total Clustered mRNA per Nucleus (per FOV)', 'mRNA/nucleus', None, 'violin',
                       q111_fov_mrna_per_nucleus, wt_fov_mrna_per_nucleus, is_distribution=True, min_points=1)

    fig.suptitle(f'{channel_labels[channel]}: Cluster Metrics by Mouse (Grouped by Age)',
                 fontsize=16, fontweight='bold', y=1.01)

    plt.tight_layout()

    # Save figure as SVG with probe name
    probe_name = channel_probe_names[channel]
    filepath = OUTPUT_DIR / f'fig_cluster_metrics_{probe_name}.svg'
    fig.savefig(filepath, format='svg', bbox_inches='tight')
    print(f"Saved: {filepath}")

    plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════════
    # CREATE SECOND VERSION WITH SLIDE NAMES
    # ══════════════════════════════════════════════════════════════════════════

    # Create figure with 4 subplots (2x2) - slide names version
    fig_slides, axes_slides = plt.subplots(2, 2, figsize=(20, 14), dpi=300)

    # Helper function for slide names version
    def create_grouped_plot_slides(ax, panel_idx, metric_name, ylabel, ylim, plot_type,
                                   q111_data_dict, wt_data_dict, is_distribution=True, min_points=10):
        """Create violin or bar plot with age grouping using slide names."""

        positions = []
        labels = []
        colors = []
        data_or_heights = []

        pos = 0
        age_group_info = []
        age_separator_positions = []

        # Q111 mice grouped by age
        q111_ages = sorted(set(slide_to_age_fov.get(s) for s in q111_unique_slides_sorted if s in slide_to_age_fov))

        for age in q111_ages:
            group_start = pos
            slides_for_age = [s for s in q111_unique_slides_sorted if slide_to_age_fov.get(s) == age]

            for slide in slides_for_age:
                if is_distribution:
                    vals = q111_data_dict.get(slide, np.array([]))
                    if len(vals) < min_points:
                        continue
                    if ylim is not None:
                        vals = vals[vals <= ylim[1]]
                    if len(vals) < min_points:
                        continue
                    data_or_heights.append(vals)
                else:
                    data_or_heights.append(q111_data_dict.get(slide, 0))

                positions.append(pos)
                labels.append(slide)  # Use actual slide name
                colors.append(channel_colors[channel])
                pos += 1

            if pos > group_start:
                age_group_info.append((group_start, pos - 1, f"Q111\n{age}mo"))
                age_separator_positions.append(pos - 0.5)
                pos += 1

        if age_separator_positions:
            age_separator_positions.pop()

        q111_wt_separator = pos - 1.5

        # Wildtype mice grouped by age
        if len(wt_unique_slides_sorted) > 0:
            wt_ages = sorted(set(slide_to_age_wt_fov.get(s) for s in wt_unique_slides_sorted if s in slide_to_age_wt_fov))

            for age in wt_ages:
                group_start = pos
                slides_for_age = [s for s in wt_unique_slides_sorted if slide_to_age_wt_fov.get(s) == age]

                for slide in slides_for_age:
                    if is_distribution:
                        vals = wt_data_dict.get(slide, np.array([]))
                        if len(vals) < min_points:
                            continue
                        if ylim is not None:
                            vals = vals[vals <= ylim[1]]
                        if len(vals) < min_points:
                            continue
                        data_or_heights.append(vals)
                    else:
                        data_or_heights.append(wt_data_dict.get(slide, 0))

                    positions.append(pos)
                    labels.append(slide)  # Use actual slide name
                    colors.append('gray')
                    pos += 1

                if pos > group_start:
                    age_group_info.append((group_start, pos - 1, f"WT\n{age}mo"))
                    age_separator_positions.append(pos - 0.5)
                    pos += 1

            if age_separator_positions and age_separator_positions[-1] > q111_wt_separator:
                age_separator_positions.pop()

        if len(positions) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=12)
            return

        # Plot
        if plot_type == 'violin':
            parts = ax.violinplot(data_or_heights, positions=positions, showmeans=True,
                                  showmedians=True, widths=0.8)
            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(colors[i])
                pc.set_edgecolor('black')
                pc.set_alpha(0.7)
            for partname in ['cmeans', 'cmedians', 'cbars', 'cmins', 'cmaxes']:
                if partname in parts:
                    parts[partname].set_edgecolor('black')
                    parts[partname].set_linewidth(1)
        else:  # bar
            ax.bar(positions, data_or_heights, color=colors, edgecolor='black', linewidth=0.5, alpha=0.8)

        # Separators
        for sep_pos in age_separator_positions:
            ax.axvline(sep_pos, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)

        if q111_wt_separator > 0 and len(wt_unique_slides_sorted) > 0:
            ax.axvline(q111_wt_separator, color='black', linestyle='--', linewidth=2.5, alpha=0.8)

        # Age group labels
        if plot_type == 'violin' and ylim is not None:
            y_label_pos = ylim[1] * 0.97
        else:
            max_val = max(data_or_heights) if not is_distribution else ylim[1] if ylim else 1
            if not is_distribution:
                max_val = max(data_or_heights) if data_or_heights else 1
            y_label_pos = max_val * 1.05 if plot_type == 'bar' else (ylim[1] * 0.97 if ylim else max_val * 0.97)

        for start_pos, end_pos, label in age_group_info:
            center = (start_pos + end_pos) / 2
            va = 'bottom' if plot_type == 'bar' else 'top'
            ax.text(center, y_label_pos, label,
                   ha='center', va=va, fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='gray'))

        # Formatting
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=45, fontsize=7, ha='right')
        ax.set_xlabel('Slide', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f'({panel_labels[panel_idx]}) {metric_name}',
                    fontsize=12, fontweight='bold', loc='left')
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')

        if ylim is not None:
            ax.set_ylim(ylim)
        elif plot_type == 'bar':
            max_h = max(data_or_heights) if data_or_heights else 1
            ax.set_ylim(0, max_h * 1.25)

    # Panel A: Clusters per Nucleus per FOV (violin)
    create_grouped_plot_slides(axes_slides[0, 0], 0, 'Clusters per Nucleus (per FOV)', 'Clusters/nucleus', None, 'violin',
                               q111_fov_clusters_per_nucleus, wt_fov_clusters_per_nucleus, is_distribution=True, min_points=1)

    # Panel B: mRNA per Cluster (violin)
    create_grouped_plot_slides(axes_slides[0, 1], 1, 'mRNA per Cluster', 'mRNA equivalents', MRNA_YLIM, 'violin',
                               q111_intensity_by_slide, wt_intensity_by_slide, is_distribution=True, min_points=10)

    # Panel C: Density per Cluster (violin)
    create_grouped_plot_slides(axes_slides[1, 0], 2, 'Density (mRNA/μm³)', 'mRNA/μm³', DENSITY_YLIM, 'violin',
                               q111_density_by_slide, wt_density_by_slide, is_distribution=True, min_points=10)

    # Panel D: Total mRNA per Nucleus per FOV (violin)
    create_grouped_plot_slides(axes_slides[1, 1], 3, 'Total Clustered mRNA per Nucleus (per FOV)', 'mRNA/nucleus', None, 'violin',
                               q111_fov_mrna_per_nucleus, wt_fov_mrna_per_nucleus, is_distribution=True, min_points=1)

    fig_slides.suptitle(f'{channel_labels[channel]}: Cluster Metrics by Slide (Grouped by Age)',
                        fontsize=16, fontweight='bold', y=1.01)

    plt.tight_layout()

    # Save figure with slide names
    filepath_slides = OUTPUT_DIR / f'fig_cluster_metrics_{probe_name}_slides.svg'
    fig_slides.savefig(filepath_slides, format='svg', bbox_inches='tight')
    print(f"Saved: {filepath_slides}")

    plt.close(fig_slides)

    # Print mouse ID mapping
    print(f"\n  Mouse ID mapping for {channel} channel:")
    print(f"  Q111 mice ({len(mouse_id_map_q111)}):")
    for slide, anon_id in mouse_id_map_q111.items():
        mouse_id = slide_to_mouse_id_fov.get(slide, slide)
        age = slide_to_age_fov.get(slide, '?')
        n_clusters = q111_cluster_counts.get(slide, 0)
        n_fovs = len(q111_fov_clusters_per_nucleus.get(slide, []))
        print(f"    {anon_id} -> {mouse_id} ({age}mo, {n_clusters:,} clusters, {n_fovs} FOVs)")
    if mouse_id_map_wt:
        print(f"  WT mice ({len(mouse_id_map_wt)}):")
        for slide, anon_id in mouse_id_map_wt.items():
            mouse_id = slide_to_mouse_id_wt_fov.get(slide, slide)
            age = slide_to_age_wt_fov.get(slide, '?')
            n_clusters = wt_cluster_counts.get(slide, 0)
            n_fovs = len(wt_fov_clusters_per_nucleus.get(slide, []))
            print(f"    {anon_id} -> {mouse_id} ({age}mo, {n_clusters:,} clusters, {n_fovs} FOVs)")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
print(f"\nAll outputs saved to: {OUTPUT_DIR}")
