"""
Comprehensive Cluster Size Distribution Analysis
===============================================

Creates PDF (probability density function) visualizations showing how individual
cluster sizes (mRNA equivalents per cluster) vary across:
- Genotypes (Q111 vs Wildtype)
- Ages (2, 6, 12 months)
- Individual mouse IDs

MAIN FIGURE: Overall distributions, age trends, and per-mouse breakdown
SUPPLEMENTARY FIGURE: All brain atlas coordinates (complete spatial coverage)

For each condition, shows distribution characteristics:
- Mean
- Median
- IQR (interquartile range)
- 95th percentile
- Number of clusters
- Clusters per cell

Methodology:
-----------
- Cluster size = Total cluster intensity / Single spot peak intensity
- This gives "mRNA equivalents per cluster"
- Single spot peak determined via KDE per slide
- Clusters detected via DBSCAN (eps=0.75 µm, min_samples=3)
- Threshold = 2.5 × peak_intensity per slide
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

# Import centralized configuration
import sys
sys.path.append(str(Path(__file__).parent.parent))
from results_config import (
    EXCLUDED_SLIDES,
    H5_FILE_PATH_EXPERIMENTAL,
    PIXELSIZE,
    SLICE_DEPTH,
    CV_THRESHOLD,
    CHANNEL_COLORS,
    FIGURE_DPI as CONFIG_DPI,
    FIGURE_FORMAT as CONFIG_FORMAT,
)
from result_functions_v2 import (
    compute_thresholds,
    recursively_load_dict,
    extract_dataframe
)

# Color scheme matching other figures
COLOR_Q111_MHTT1A = CHANNEL_COLORS.get('green', '#2ecc71')  # Green for Q111 HTT1a
COLOR_Q111_FULL = CHANNEL_COLORS.get('orange', '#f39c12')   # Orange for Q111 full-length
COLOR_WT_MHTT1A = '#3498db'   # Blue for WT HTT1a
COLOR_WT_FULL = '#9b59b6'     # Purple for WT full-length


def extract_mouse_info(mouse_id):
    """Extract age and mouse number from Mouse_ID string."""
    import re
    if mouse_id is None or pd.isna(mouse_id):
        return (0, 0)
    age_match = re.search(r'(\d+)mo', str(mouse_id))
    num_match = re.search(r'#(\d+)', str(mouse_id))
    age = int(age_match.group(1)) if age_match else 0
    num = int(num_match.group(1)) if num_match else 0
    return (age, num)


def build_slide_sublabels(mouse_ids, genotype_prefix):
    """Create labels like Q#1.1, Q#2.3, W#1.1 based on age-sorted order."""
    import re
    # Filter out None values
    mouse_ids = [mid for mid in mouse_ids if mid is not None and not pd.isna(mid)]
    id_info = {}
    for mid in mouse_ids:
        age, orig_num = extract_mouse_info(mid)
        id_info[mid] = {'age': age, 'orig_num': orig_num}

    sorted_ids = sorted(mouse_ids, key=lambda x: (id_info[x]['age'], id_info[x]['orig_num'], x))

    sublabels = {}
    mouse_counter = 0
    current_key = None

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

# Output directory
OUTPUT_DIR = Path(__file__).parent / "output" / "cluster_size_distributions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Constants
FIGURE_DPI = 300
FIGURE_FORMAT = 'svg'

print("="*80)
print("COMPREHENSIVE CLUSTER SIZE DISTRIBUTION ANALYSIS")
print("="*80)

# ══════════════════════════════════════════════════════════════════════════
# SECTION 1: LOAD AND PROCESS DATA
# ══════════════════════════════════════════════════════════════════════════

print("\nLoading HDF5 data...")
with h5py.File(H5_FILE_PATH_EXPERIMENTAL, 'r') as h5_file:
    data_dict = recursively_load_dict(h5_file)

# Extract DataFrame
print("Extracting dataframe...")
desired_channels = ['blue', 'green', 'orange']
fields_to_extract = [
    'spots_sigma_var.params_raw',
    'spots.params_raw',
    'cluster_intensities',
    'cluster_cvs',
    'num_cells',
    'metadata_sample.Age',
    'metadata_sample.Brain_Atlas_coordinates',
    'spots.final_filter',
    'metadata_sample_Mouse_ID'
]

slide_field = 'metadata_sample_slide_name_std'
negative_control_field = 'Negative control'
experimental_field = 'ExperimentalQ111 - 488mHT - 548mHTa - 647Darp'

df_extracted_full = extract_dataframe(
    data_dict,
    field_keys=fields_to_extract,
    channels=desired_channels,
    include_file_metadata_sample=True,
    explode_fields=[]
)

print(f"Total records extracted: {len(df_extracted_full)}")

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
    quantile_negative_control=0.95,
    max_pfa=0.05,
    plot=False,
    n_bootstrap=20,
    use_region=False,
    use_final_filter=True
)

# Build threshold lookup table
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
    thr_df,
    how="left",
    left_on=[slide_field, "channel"],
    right_on=["slide", "channel"]
)
df_extracted_full.rename(columns={"thr": "threshold"}, inplace=True)
df_extracted_full.drop(columns=["slide"], inplace=True, errors='ignore')

# Filter for experimental data only
df_exp = df_extracted_full[
    df_extracted_full['metadata_sample_Probe-Set'] == experimental_field
].copy()

# Exclude failed slides
print(f"\nExcluding slides: {EXCLUDED_SLIDES}")
df_exp = df_exp[~df_exp[slide_field].isin(EXCLUDED_SLIDES)].copy()

print(f"Records after exclusion: {len(df_exp)}")
print(f"Q111 records: {len(df_exp[df_exp['metadata_sample_Mouse_Model'] == 'Q111'])}")
print(f"Wildtype records: {len(df_exp[df_exp['metadata_sample_Mouse_Model'] == 'Wildtype'])}")

# Channel labels
channel_labels = {
    'green': 'HTT1a',
    'orange': 'fl-HTT'
}


# ══════════════════════════════════════════════════════════════════════════
# SECTION 2: EXTRACT CLUSTER SIZE DISTRIBUTIONS
# ══════════════════════════════════════════════════════════════════════════

def compute_peak_intensity(intensities, bw_method='scott'):
    """Compute peak intensity from KDE."""
    if len(intensities) < 50:
        return np.nan

    try:
        kde = gaussian_kde(intensities, bw_method=bw_method)
        x_range = np.linspace(
            np.percentile(intensities, 1),
            np.percentile(intensities, 99),
            1000
        )
        y_density = kde(x_range)
        peak_idx = np.argmax(y_density)
        peak_intensity = x_range[peak_idx]
        return peak_intensity
    except:
        return np.nan


def extract_cluster_distributions(df_input):
    """
    Extract cluster size distributions with metadata.
    Returns a list of dicts, each containing:
    - cluster_sizes: array of mRNA equivalents per cluster
    - metadata: slide, region, channel, age, atlas_coord, mouse_id, n_cells
    """

    print(f"\n{'='*60}")
    print("Extracting cluster size distributions...")
    print(f"{'='*60}")

    cluster_data = []

    # First pass: compute spot peaks per slide/region/channel
    spot_peaks = {}

    for idx, row in df_input.iterrows():
        slide = row.get(slide_field, 'unknown')
        region = row.get('metadata_sample_Slice_Region', 'unknown')
        channel = row.get('channel', 'unknown')
        threshold_val = row.get('threshold', np.nan)

        if channel == 'blue':
            continue

        # Merge regions
        if 'Cortex' in region:
            region_merged = 'Cortex'
        elif 'Striatum' in region:
            region_merged = 'Striatum'
        else:
            continue

        key = (slide, region_merged, channel)

        if key not in spot_peaks:
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

                                if len(valid_photons) >= 50:
                                    peak_spot = compute_peak_intensity(valid_photons)
                                    if not np.isnan(peak_spot):
                                        spot_peaks[key] = peak_spot
                except:
                    pass

    print(f"Computed {len(spot_peaks)} spot peaks")

    # Second pass: extract and normalize clusters
    for idx, row in df_input.iterrows():
        slide = row.get(slide_field, 'unknown')
        region = row.get('metadata_sample_Slice_Region', 'unknown')
        channel = row.get('channel', 'unknown')
        age = row.get('metadata_sample_Age', np.nan)
        atlas_coord = row.get('metadata_sample_Brain_Atlas_coordinates', np.nan)
        mouse_id = row.get('metadata_sample_Mouse_ID', 'unknown')
        mouse_model = row.get('metadata_sample_Mouse_Model', 'unknown')
        n_cells = row.get('num_cells', 0)
        threshold_val = row.get('threshold', np.nan)

        if channel == 'blue':
            continue

        # Merge regions
        if 'Cortex' in region:
            region_merged = 'Cortex'
        elif 'Striatum' in region:
            region_merged = 'Striatum'
        else:
            continue

        key = (slide, region_merged, channel)

        if key not in spot_peaks:
            continue

        peak_spot = spot_peaks[key]

        # Extract clusters (with intensity AND CV filtering)
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
                    valid_clusters = cluster_int[above_threshold]

                    if len(valid_clusters) >= 5:  # Require at least 5 clusters
                        # Normalize by peak to get mRNA equivalents
                        mrna_per_cluster = valid_clusters / peak_spot

                        cluster_data.append({
                            'cluster_sizes': mrna_per_cluster,
                            'slide': slide,
                            'region': region_merged,
                            'channel': channel_labels.get(channel, channel),
                            'age': age,
                            'atlas_coord': atlas_coord,
                            'mouse_id': mouse_id,
                            'mouse_model': mouse_model,
                            'n_cells': n_cells,
                            'n_clusters': len(mrna_per_cluster)
                        })
            except:
                pass

    print(f"Extracted {len(cluster_data)} FOV-level cluster distributions")

    return cluster_data


# Extract distributions
cluster_distributions = extract_cluster_distributions(df_exp)

# Convert to DataFrame for easier manipulation
distribution_records = []
for dist in cluster_distributions:
    # Store full distribution but also summary stats
    sizes = dist['cluster_sizes']

    distribution_records.append({
        'Slide': dist['slide'],
        'Region': dist['region'],
        'Channel': dist['channel'],
        'Age': dist['age'],
        'Brain_Atlas_Coord': dist['atlas_coord'],
        'Mouse_ID': dist['mouse_id'],
        'Mouse_Model': dist['mouse_model'],
        'N_Cells': dist['n_cells'],
        'N_Clusters': dist['n_clusters'],
        'Clusters_per_Cell': dist['n_clusters'] / dist['n_cells'] if dist['n_cells'] > 0 else 0,
        'Mean_Cluster_Size': np.mean(sizes),
        'Median_Cluster_Size': np.median(sizes),
        'IQR_Cluster_Size': np.percentile(sizes, 75) - np.percentile(sizes, 25),
        'P95_Cluster_Size': np.percentile(sizes, 95),
        'Cluster_Sizes_Array': sizes  # Store full distribution
    })

df_distributions = pd.DataFrame(distribution_records)

# Save summary statistics
df_summary = df_distributions.drop(columns=['Cluster_Sizes_Array'])
df_summary.to_csv(OUTPUT_DIR / 'cluster_size_summary_statistics.csv', index=False)
print(f"\nSaved summary statistics to: {OUTPUT_DIR / 'cluster_size_summary_statistics.csv'}")

print(f"\nDistribution breakdown:")
print(f"  Total FOVs with clusters: {len(df_distributions)}")
print(f"  Q111 FOVs: {len(df_distributions[df_distributions['Mouse_Model'] == 'Q111'])}")
print(f"  Wildtype FOVs: {len(df_distributions[df_distributions['Mouse_Model'] == 'Wildtype'])}")


# ══════════════════════════════════════════════════════════════════════════
# SECTION 3: PLOTTING HELPER FUNCTION
# ══════════════════════════════════════════════════════════════════════════

def plot_distribution_with_stats(ax, sizes_list, labels, colors, title, x_label="mRNA/cluster",
                                  show_stats_box=True, font_scale=1.0):
    """
    Plot overlapping distributions with statistics boxes.

    Args:
        ax: matplotlib axis
        sizes_list: list of arrays of cluster sizes
        labels: list of labels for each distribution
        colors: list of colors for each distribution
        title: plot title
        show_stats_box: whether to show statistics box
        font_scale: scale factor for fonts
    """

    if len(sizes_list) == 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes,
                fontsize=12 * font_scale)
        ax.set_title(title, fontsize=13 * font_scale, fontweight='bold')
        return

    # Determine x-range from all data
    all_sizes = np.concatenate(sizes_list)
    x_max = min(np.percentile(all_sizes, 99.5), 30)  # Cap at 30 mRNA or P99.5
    bins = np.linspace(0, x_max, 35)

    # Plot histograms
    for sizes, label, color in zip(sizes_list, labels, colors):
        if len(sizes) == 0:
            continue

        ax.hist(sizes, bins=bins, alpha=0.6, color=color, label=label,
               edgecolor='black', linewidth=0.8, density=True)

    # Add statistics text boxes - stacked vertically with clear separation
    if show_stats_box:
        # Calculate stats for all distributions
        stats_entries = []
        for sizes, label, color in zip(sizes_list, labels, colors):
            if len(sizes) == 0:
                continue
            n_clusters = len(sizes)
            mean_val = np.mean(sizes)
            median_val = np.median(sizes)
            p95_val = np.percentile(sizes, 95)
            stats_entries.append({
                'label': label, 'color': color, 'n': n_clusters,
                'mean': mean_val, 'median': median_val, 'p95': p95_val
            })

        # Stack boxes vertically at top right with sufficient spacing
        y_positions = [0.97, 0.62]  # Two positions with more separation
        for i, entry in enumerate(stats_entries):
            if i >= len(y_positions):
                break
            # Clear format: genotype label, then n, then key stats
            stats_text = f"{entry['label']}: n={entry['n']:,}\n"
            stats_text += f"mean={entry['mean']:.1f}, P95={entry['p95']:.1f}"
            ax.text(0.98, y_positions[i], stats_text, transform=ax.transAxes,
                   fontsize=7 * font_scale, ha='right', va='top',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=entry['color'],
                            alpha=0.4, edgecolor='black', linewidth=0.5))

    ax.set_xlabel(x_label, fontsize=12 * font_scale, fontweight='bold')
    ax.set_ylabel('Density', fontsize=12 * font_scale, fontweight='bold')
    ax.set_title(title, fontsize=13 * font_scale, fontweight='bold')
    ax.set_xlim(0, x_max)
    # Legend removed - stats boxes already show color-coded labels
    ax.tick_params(axis='both', labelsize=10 * font_scale)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')


# ══════════════════════════════════════════════════════════════════════════
# SECTION 4: CREATE SEPARATE MAIN FIGURES (ONE PER CHANNEL)
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("CREATING SEPARATE MAIN FIGURES")
print("="*80)

# Build Q/W label mappings for all mice
all_q111_ids = df_distributions[df_distributions['Mouse_Model'] == 'Q111']['Mouse_ID'].unique().tolist()
all_wt_ids = df_distributions[df_distributions['Mouse_Model'] == 'Wildtype']['Mouse_ID'].unique().tolist()
q111_labels = build_slide_sublabels(all_q111_ids, 'Q')
wt_labels = build_slide_sublabels(all_wt_ids, 'W')
all_labels = {**q111_labels, **wt_labels}

for channel in ['HTT1a', 'fl-HTT']:
    print(f"\nProcessing {channel}...")

    # Select colors based on channel
    if channel == 'HTT1a':
        color_q111 = COLOR_Q111_MHTT1A
        color_wt = COLOR_WT_MHTT1A
    else:
        color_q111 = COLOR_Q111_FULL
        color_wt = COLOR_WT_FULL

    # Create figure for this channel - SMALLER with bigger fonts
    fig = plt.figure(figsize=(12, 12), dpi=FIGURE_DPI)
    gs = fig.add_gridspec(5, 4, hspace=0.60, wspace=0.35,
                          left=0.08, right=0.97, top=0.94, bottom=0.05)

    # Panel label counter
    panel_idx = 0
    panel_labels = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

    # ROW 1: Overall Q111 vs WT by region
    print("  Row 1: Overall distributions by region...")
    for reg_idx, region in enumerate(['Cortex', 'Striatum']):
        ax = fig.add_subplot(gs[0, reg_idx])

        # Add panel label
        ax.text(-0.12, 1.08, panel_labels[panel_idx], transform=ax.transAxes,
               fontsize=16, fontweight='bold', va='bottom', ha='left')
        panel_idx += 1

        # Get data
        q111_data = df_distributions[
            (df_distributions['Channel'] == channel) &
            (df_distributions['Region'] == region) &
            (df_distributions['Mouse_Model'] == 'Q111')
        ]

        wt_data = df_distributions[
            (df_distributions['Channel'] == channel) &
            (df_distributions['Region'] == region) &
            (df_distributions['Mouse_Model'] == 'Wildtype')
        ]

        # Combine all cluster sizes
        q111_sizes = np.concatenate(q111_data['Cluster_Sizes_Array'].values) if len(q111_data) > 0 else np.array([])
        wt_sizes = np.concatenate(wt_data['Cluster_Sizes_Array'].values) if len(wt_data) > 0 else np.array([])

        # Calculate clusters per nucleus
        q111_cpc = q111_data['Clusters_per_Cell'].mean() if len(q111_data) > 0 else 0
        wt_cpc = wt_data['Clusters_per_Cell'].mean() if len(wt_data) > 0 else 0

        sizes_list = [q111_sizes, wt_sizes] if len(wt_sizes) > 0 else [q111_sizes]
        labels = [f'Q111 ({q111_cpc:.1f}/nuc)', f'WT ({wt_cpc:.1f}/nuc)'] if len(wt_sizes) > 0 else [f'Q111 ({q111_cpc:.1f}/nuc)']
        colors = [color_q111, color_wt] if len(wt_sizes) > 0 else [color_q111]

        plot_distribution_with_stats(
            ax, sizes_list, labels, colors,
            f"{region} - Overall"
        )

    # ROW 2-3: By Age (one row per region)
    print("  Rows 2-3: Distributions by age...")
    for reg_idx, region in enumerate(['Cortex', 'Striatum']):
        ages = sorted(df_distributions[
            (df_distributions['Channel'] == channel) &
            (df_distributions['Region'] == region)
        ]['Age'].unique())

        for age_idx, age in enumerate(ages):
            if age_idx >= 4:  # Max 4 ages per row
                break

            ax = fig.add_subplot(gs[1 + reg_idx, age_idx])

            # Add panel label
            ax.text(-0.12, 1.08, panel_labels[panel_idx], transform=ax.transAxes,
                   fontsize=16, fontweight='bold', va='bottom', ha='left')
            panel_idx += 1

            # Get data for this age
            q111_data = df_distributions[
                (df_distributions['Channel'] == channel) &
                (df_distributions['Region'] == region) &
                (df_distributions['Mouse_Model'] == 'Q111') &
                (df_distributions['Age'] == age)
            ]

            wt_data = df_distributions[
                (df_distributions['Channel'] == channel) &
                (df_distributions['Region'] == region) &
                (df_distributions['Mouse_Model'] == 'Wildtype') &
                (df_distributions['Age'] == age)
            ]

            q111_sizes = np.concatenate(q111_data['Cluster_Sizes_Array'].values) if len(q111_data) > 0 else np.array([])
            wt_sizes = np.concatenate(wt_data['Cluster_Sizes_Array'].values) if len(wt_data) > 0 else np.array([])

            q111_cpc = q111_data['Clusters_per_Cell'].mean() if len(q111_data) > 0 else 0
            wt_cpc = wt_data['Clusters_per_Cell'].mean() if len(wt_data) > 0 else 0

            sizes_list = []
            labels = []
            colors = []

            if len(q111_sizes) > 0:
                sizes_list.append(q111_sizes)
                labels.append(f'Q111 ({q111_cpc:.1f}/nuc)')
                colors.append(color_q111)

            if len(wt_sizes) > 0:
                sizes_list.append(wt_sizes)
                labels.append(f'WT ({wt_cpc:.1f}/nuc)')
                colors.append(color_wt)

            plot_distribution_with_stats(
                ax, sizes_list, labels, colors,
                f"{region} - {age:.0f}mo"
            )

    # ROW 4-5: By Mouse ID (one row per region, top 4 mice with Q/W labels)
    print("  Rows 4-5: Distributions by mouse ID...")
    for reg_idx, region in enumerate(['Cortex', 'Striatum']):
        # Get top 2 mice by number of FOVs for Q111
        top_q111_mice = df_distributions[
            (df_distributions['Channel'] == channel) &
            (df_distributions['Region'] == region) &
            (df_distributions['Mouse_Model'] == 'Q111')
        ]['Mouse_ID'].value_counts().head(2).index.tolist()

        # Get top 2 WT mice
        top_wt_mice = df_distributions[
            (df_distributions['Channel'] == channel) &
            (df_distributions['Region'] == region) &
            (df_distributions['Mouse_Model'] == 'Wildtype')
        ]['Mouse_ID'].value_counts().head(2).index.tolist()

        all_mice = top_q111_mice + top_wt_mice

        for mouse_idx, mouse_id in enumerate(all_mice[:4]):  # Max 4 mice per row
            ax = fig.add_subplot(gs[3 + reg_idx, mouse_idx])

            # Add panel label
            ax.text(-0.12, 1.08, panel_labels[panel_idx], transform=ax.transAxes,
                   fontsize=16, fontweight='bold', va='bottom', ha='left')
            panel_idx += 1

            # Get data for this mouse
            mouse_data = df_distributions[
                (df_distributions['Channel'] == channel) &
                (df_distributions['Region'] == region) &
                (df_distributions['Mouse_ID'] == mouse_id)
            ]

            if len(mouse_data) == 0:
                continue

            mouse_model = mouse_data.iloc[0]['Mouse_Model']
            mouse_age = mouse_data.iloc[0]['Age']
            mouse_sizes = np.concatenate(mouse_data['Cluster_Sizes_Array'].values)
            mouse_cpc = mouse_data['Clusters_per_Cell'].mean()

            # Use Q/W label
            mouse_label = all_labels.get(mouse_id, mouse_id)
            color = color_q111 if mouse_model == 'Q111' else color_wt

            plot_distribution_with_stats(
                ax,
                [mouse_sizes],
                [f'{mouse_label} ({mouse_cpc:.1f}/nuc)'],
                [color],
                f"{region} - {mouse_label} ({mouse_age:.0f}mo)"
            )

    # Overall title - cleaner
    channel_display = 'HTT1a' if channel == 'HTT1a' else 'fl-HTT'
    fig.suptitle(
        f"Cluster Size Distributions - {channel_display}",
        fontsize=16, fontweight='bold', y=1.01
    )

    # Save figure in multiple formats
    channel_name = channel.replace(' ', '_').replace('-', '_')
    # Use S13/S14 naming convention
    fig_num = 'S13' if channel == 'HTT1a' else 'S14'
    filepath_svg = OUTPUT_DIR / f"fig_{fig_num}_cluster_size_{channel_name}.svg"
    filepath_pdf = OUTPUT_DIR / f"fig_{fig_num}_cluster_size_{channel_name}.pdf"
    filepath_png = OUTPUT_DIR / f"fig_{fig_num}_cluster_size_{channel_name}.png"
    plt.savefig(filepath_svg, format='svg', bbox_inches='tight')
    plt.savefig(filepath_pdf, format='pdf', bbox_inches='tight')
    plt.savefig(filepath_png, format='png', bbox_inches='tight', dpi=FIGURE_DPI)
    plt.close()
    print(f"  Saved: {filepath_svg}")
    print(f"  Saved: {filepath_pdf}")
    print(f"  Saved: {filepath_png}")


# ══════════════════════════════════════════════════════════════════════════
# SECTION 5: CREATE SUPPLEMENTARY FIGURE (ALL ATLAS COORDINATES)
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("CREATING SUPPLEMENTARY FIGURES (ALL ATLAS COORDINATES)")
print("="*80)

for channel in ['HTT1a', 'fl-HTT']:
    print(f"\nProcessing {channel} supplementary figure...")

    # Select colors based on channel
    if channel == 'HTT1a':
        color_q111_supp = COLOR_Q111_MHTT1A
        color_wt_supp = COLOR_WT_MHTT1A
    else:
        color_q111_supp = COLOR_Q111_FULL
        color_wt_supp = COLOR_WT_FULL

    # Get all atlas coordinates for Q111 in each region
    for region in ['Cortex', 'Striatum']:
        # Get all unique atlas coordinates (sorted)
        all_atlas_coords = sorted(df_distributions[
            (df_distributions['Channel'] == channel) &
            (df_distributions['Region'] == region) &
            (df_distributions['Mouse_Model'] == 'Q111')
        ]['Brain_Atlas_Coord'].unique())

        print(f"  {region}: {len(all_atlas_coords)} atlas coordinates")

        if len(all_atlas_coords) == 0:
            continue

        # Determine grid layout (max 6 columns)
        n_coords = len(all_atlas_coords)
        n_cols = min(6, n_coords)
        n_rows = int(np.ceil(n_coords / n_cols))

        # Create figure - smaller with bigger fonts
        fig_height = max(10, n_rows * 3)
        fig = plt.figure(figsize=(18, fig_height), dpi=FIGURE_DPI)
        gs = fig.add_gridspec(n_rows, n_cols, hspace=0.5, wspace=0.4,
                              left=0.06, right=0.98, top=0.92, bottom=0.06)

        for coord_idx, atlas_coord in enumerate(all_atlas_coords):
            row = coord_idx // n_cols
            col = coord_idx % n_cols
            ax = fig.add_subplot(gs[row, col])

            # Get data for this atlas coordinate
            q111_data = df_distributions[
                (df_distributions['Channel'] == channel) &
                (df_distributions['Region'] == region) &
                (df_distributions['Mouse_Model'] == 'Q111') &
                (df_distributions['Brain_Atlas_Coord'] == atlas_coord)
            ]

            wt_data = df_distributions[
                (df_distributions['Channel'] == channel) &
                (df_distributions['Region'] == region) &
                (df_distributions['Mouse_Model'] == 'Wildtype') &
                (df_distributions['Brain_Atlas_Coord'] == atlas_coord)
            ]

            q111_sizes = np.concatenate(q111_data['Cluster_Sizes_Array'].values) if len(q111_data) > 0 else np.array([])
            wt_sizes = np.concatenate(wt_data['Cluster_Sizes_Array'].values) if len(wt_data) > 0 else np.array([])

            q111_cpc = q111_data['Clusters_per_Cell'].mean() if len(q111_data) > 0 else 0
            wt_cpc = wt_data['Clusters_per_Cell'].mean() if len(wt_data) > 0 else 0

            sizes_list = []
            labels = []
            colors = []

            if len(q111_sizes) > 0:
                sizes_list.append(q111_sizes)
                labels.append(f'Q111 ({q111_cpc:.1f}/nuc)')
                colors.append(color_q111_supp)

            if len(wt_sizes) > 0:
                sizes_list.append(wt_sizes)
                labels.append(f'WT ({wt_cpc:.1f}/nuc)')
                colors.append(color_wt_supp)

            plot_distribution_with_stats(
                ax, sizes_list, labels, colors,
                f"Atlas {atlas_coord:.0f}"
            )

        # Overall title
        channel_display = 'HTT1a' if channel == 'HTT1a' else 'fl-HTT'
        fig.suptitle(
            f"Cluster Size Distributions by Atlas Coordinate - {channel_display} ({region})",
            fontsize=16, fontweight='bold', y=0.98
        )

        # Save supplementary figure
        channel_name = channel.replace(' ', '_').replace('-', '_')
        filepath = OUTPUT_DIR / f"fig_SUPP_cluster_size_{channel_name}_{region}.{FIGURE_FORMAT}"
        plt.savefig(filepath, format=FIGURE_FORMAT, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {filepath}")


# ══════════════════════════════════════════════════════════════════════════
# SECTION 6: CREATE CAPTIONS (FULLY DYNAMIC)
# ══════════════════════════════════════════════════════════════════════════

print("\nCreating comprehensive captions...")

# Create separate captions for each channel
for channel in ['HTT1a', 'fl-HTT']:
    caption_lines = []
    fig_num = 'S13' if channel == 'HTT1a' else 'S14'
    channel_display = 'HTT1a' if channel == 'HTT1a' else 'fl-HTT'

    caption_lines.append("=" * 80)
    caption_lines.append(f"FIGURE {fig_num}: Cluster Size Distributions - {channel_display}")
    caption_lines.append("=" * 80)
    caption_lines.append("")

    caption_lines.append("OVERVIEW:")
    caption_lines.append("-" * 80)
    if channel == 'HTT1a':
        caption_lines.append("This figure presents probability density functions (PDFs) showing the distribution")
        caption_lines.append("of cluster sizes (mRNA equivalents per cluster) for HTT1a transcripts in Q111")
        caption_lines.append("transgenic mice compared to wildtype controls. HTT1a represents intron-1 terminated")
        caption_lines.append("transcripts containing only exon 1.")
    else:
        caption_lines.append("This figure presents probability density functions (PDFs) showing the distribution")
        caption_lines.append("of cluster sizes (mRNA equivalents per cluster) for fl-HTT transcripts in")
        caption_lines.append("Q111 transgenic mice compared to wildtype controls.")
    caption_lines.append("")

    caption_lines.append("DATA SOURCE:")
    caption_lines.append("-" * 80)
    caption_lines.append(f"HDF5 file: {H5_FILE_PATH_EXPERIMENTAL}")
    caption_lines.append("")

    # Dataset statistics (DYNAMIC)
    caption_lines.append("DATASET STATISTICS:")
    caption_lines.append("-" * 80)
    channel_data = df_distributions[df_distributions['Channel'] == channel]

    total_fovs = len(channel_data)
    q111_fovs = len(channel_data[channel_data['Mouse_Model'] == 'Q111'])
    wt_fovs = len(channel_data[channel_data['Mouse_Model'] == 'Wildtype'])

    total_clusters = int(channel_data['N_Clusters'].sum())
    q111_clusters = int(channel_data[channel_data['Mouse_Model'] == 'Q111']['N_Clusters'].sum())
    wt_clusters = int(channel_data[channel_data['Mouse_Model'] == 'Wildtype']['N_Clusters'].sum())

    caption_lines.append(f"Total FOVs analyzed: {total_fovs}")
    caption_lines.append(f"  Q111: {q111_fovs} FOVs")
    caption_lines.append(f"  Wildtype: {wt_fovs} FOVs")
    caption_lines.append(f"Total clusters analyzed: {total_clusters:,}")
    caption_lines.append(f"  Q111: {q111_clusters:,} clusters")
    caption_lines.append(f"  Wildtype: {wt_clusters:,} clusters")
    caption_lines.append("")

    # Breakdown by region
    for region in ['Cortex', 'Striatum']:
        region_data = channel_data[channel_data['Region'] == region]
        q111_region = region_data[region_data['Mouse_Model'] == 'Q111']
        wt_region = region_data[region_data['Mouse_Model'] == 'Wildtype']

        caption_lines.append(f"{region}:")
        caption_lines.append(f"  Q111: {len(q111_region)} FOVs, {int(q111_region['N_Clusters'].sum()):,} clusters")
        if len(wt_region) > 0:
            caption_lines.append(f"  WT: {len(wt_region)} FOVs, {int(wt_region['N_Clusters'].sum()):,} clusters")

    caption_lines.append("")

    # COMPREHENSIVE STATISTICS SUMMARY
    caption_lines.append("STATISTICAL SUMMARY:")
    caption_lines.append("-" * 80)
    caption_lines.append("Cluster size is measured in mRNA equivalents (cluster intensity / single spot peak).")
    caption_lines.append("Higher values indicate larger mRNA aggregates containing more transcripts.")
    caption_lines.append("")

    # Compute comprehensive statistics for each genotype
    for genotype, genotype_name in [('Q111', 'Q111'), ('Wildtype', 'Wildtype')]:
        geno_data = channel_data[channel_data['Mouse_Model'] == genotype]
        if len(geno_data) > 0:
            # Concatenate all cluster sizes
            all_sizes = np.concatenate(geno_data['Cluster_Sizes_Array'].values)

            # Compute statistics
            n_clusters = len(all_sizes)
            mean_size = np.mean(all_sizes)
            std_size = np.std(all_sizes)
            median_size = np.median(all_sizes)
            q25 = np.percentile(all_sizes, 25)
            q75 = np.percentile(all_sizes, 75)
            iqr = q75 - q25
            p5 = np.percentile(all_sizes, 5)
            p95 = np.percentile(all_sizes, 95)
            min_size = np.min(all_sizes)
            max_size = np.max(all_sizes)

            caption_lines.append(f"{genotype_name} ({n_clusters:,} clusters):")
            caption_lines.append(f"  Mean ± SD: {mean_size:.2f} ± {std_size:.2f} mRNA/cluster")
            caption_lines.append(f"  Median [IQR]: {median_size:.2f} [{q25:.2f} - {q75:.2f}]")
            caption_lines.append(f"  Range (5th-95th percentile): {p5:.2f} - {p95:.2f}")
            caption_lines.append(f"  Full range: {min_size:.2f} - {max_size:.2f}")
            caption_lines.append("")

    # Add genotype comparison statistics
    q111_all = channel_data[channel_data['Mouse_Model'] == 'Q111']
    wt_all = channel_data[channel_data['Mouse_Model'] == 'Wildtype']

    if len(q111_all) > 0 and len(wt_all) > 0:
        q111_sizes = np.concatenate(q111_all['Cluster_Sizes_Array'].values)
        wt_sizes = np.concatenate(wt_all['Cluster_Sizes_Array'].values)

        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.std(q111_sizes)**2 + np.std(wt_sizes)**2) / 2)
        cohens_d = (np.mean(q111_sizes) - np.mean(wt_sizes)) / pooled_std if pooled_std > 0 else 0

        # Fold change
        fold_change = np.mean(q111_sizes) / np.mean(wt_sizes) if np.mean(wt_sizes) > 0 else np.nan

        caption_lines.append("Genotype Comparison (Q111 vs Wildtype):")
        caption_lines.append(f"  Mean difference: {np.mean(q111_sizes) - np.mean(wt_sizes):.2f} mRNA/cluster")
        caption_lines.append(f"  Fold change (Q111/WT): {fold_change:.2f}x")
        caption_lines.append(f"  Effect size (Cohen's d): {cohens_d:.2f}")
        caption_lines.append("")

    caption_lines.append("Statistics Definitions:")
    caption_lines.append("  - Mean: Average cluster size across all clusters")
    caption_lines.append("  - SD: Standard deviation, measures spread of cluster sizes")
    caption_lines.append("  - Median: Middle value; 50% of clusters are smaller, 50% are larger")
    caption_lines.append("  - IQR: Interquartile range (25th-75th percentile), captures central 50%")
    caption_lines.append("  - P95: 95th percentile; only 5% of clusters are larger than this value")
    caption_lines.append("  - Cohen's d: Effect size; |d|>0.8 is large, |d|>0.5 is medium, |d|>0.2 is small")
    caption_lines.append("")

    # FILTERING APPLIED (DYNAMIC)
    caption_lines.append("FILTERING APPLIED:")
    caption_lines.append("-" * 80)
    caption_lines.append(f"1. Excluded slides (technical failures): {', '.join(EXCLUDED_SLIDES)}")
    caption_lines.append(f"2. Cluster CV threshold: CV >= {CV_THRESHOLD}")
    caption_lines.append("   (Clusters with low coefficient of variation are excluded as likely noise)")
    caption_lines.append("3. Intensity threshold: Cluster intensity > 95th percentile of negative control")
    caption_lines.append("4. Minimum clusters per FOV: 5")
    caption_lines.append("5. Minimum spots for peak calculation: 50")
    caption_lines.append("")

    # MOUSE ID MAPPING (DYNAMIC with Q/W labels)
    caption_lines.append("MOUSE ID MAPPING:")
    caption_lines.append("-" * 80)
    caption_lines.append("Labels use Q#X.Y format for Q111 mice and W#X.Y for Wildtype mice,")
    caption_lines.append("where X = mouse number (sorted by age: 1-3=2mo, 4-6=6mo, 7-9=12mo), Y = slide number.")
    caption_lines.append("")

    # List Q111 mice (filter out None/unknown values)
    q111_mice_in_channel = channel_data[channel_data['Mouse_Model'] == 'Q111']['Mouse_ID'].unique()
    q111_mice_valid = [m for m in q111_mice_in_channel if m is not None and not pd.isna(m) and m != 'unknown']
    if len(q111_mice_valid) > 0:
        caption_lines.append("Q111 mice:")
        for mouse_id in sorted(q111_mice_valid, key=lambda x: (extract_mouse_info(x), str(x))):
            label = q111_labels.get(mouse_id, mouse_id)
            age, num = extract_mouse_info(mouse_id)
            caption_lines.append(f"  {label} -> {mouse_id} ({age}mo)")
    else:
        caption_lines.append("Q111 mice: (Mouse IDs not available in metadata)")

    # List WT mice (filter out None/unknown values)
    wt_mice_in_channel = channel_data[channel_data['Mouse_Model'] == 'Wildtype']['Mouse_ID'].unique()
    wt_mice_valid = [m for m in wt_mice_in_channel if m is not None and not pd.isna(m) and m != 'unknown']
    if len(wt_mice_valid) > 0:
        caption_lines.append("Wildtype mice:")
        for mouse_id in sorted(wt_mice_valid, key=lambda x: (extract_mouse_info(x), str(x))):
            label = wt_labels.get(mouse_id, mouse_id)
            age, num = extract_mouse_info(mouse_id)
            caption_lines.append(f"  {label} -> {mouse_id} ({age}mo)")
    else:
        caption_lines.append("Wildtype mice: (Mouse IDs not available in metadata)")
    caption_lines.append("")

    # PANEL DESCRIPTIONS with DYNAMIC statistics
    caption_lines.append("PANEL DESCRIPTIONS:")
    caption_lines.append("-" * 80)
    caption_lines.append("")
    caption_lines.append("ROW 1 (A-B): OVERALL DISTRIBUTIONS BY REGION")
    caption_lines.append("Probability density functions showing the full distribution of cluster sizes.")
    caption_lines.append("")

    for region in ['Cortex', 'Striatum']:
        q111_data = df_distributions[
            (df_distributions['Channel'] == channel) &
            (df_distributions['Region'] == region) &
            (df_distributions['Mouse_Model'] == 'Q111')
        ]
        wt_data = df_distributions[
            (df_distributions['Channel'] == channel) &
            (df_distributions['Region'] == region) &
            (df_distributions['Mouse_Model'] == 'Wildtype')
        ]

        if len(q111_data) > 0:
            q111_all = np.concatenate(q111_data['Cluster_Sizes_Array'].values)
            q111_cpc = q111_data['Clusters_per_Cell'].mean()
            caption_lines.append(f"{region} Q111: n={len(q111_all):,} clusters, "
                               f"mean={np.mean(q111_all):.2f}, median={np.median(q111_all):.2f}, "
                               f"P95={np.percentile(q111_all, 95):.2f}, {q111_cpc:.2f} clusters/nucleus")

        if len(wt_data) > 0:
            wt_all = np.concatenate(wt_data['Cluster_Sizes_Array'].values)
            wt_cpc = wt_data['Clusters_per_Cell'].mean()
            caption_lines.append(f"{region} WT: n={len(wt_all):,} clusters, "
                               f"mean={np.mean(wt_all):.2f}, median={np.median(wt_all):.2f}, "
                               f"P95={np.percentile(wt_all, 95):.2f}, {wt_cpc:.2f} clusters/nucleus")

    caption_lines.append("")
    caption_lines.append("ROWS 2-3 (C-H): DISTRIBUTIONS BY AGE")
    caption_lines.append("Cluster size distributions separated by age (2, 6, 12 months).")
    caption_lines.append("")

    for region in ['Cortex', 'Striatum']:
        caption_lines.append(f"{region}:")
        ages = sorted(df_distributions[
            (df_distributions['Channel'] == channel) &
            (df_distributions['Region'] == region)
        ]['Age'].unique())

        for age in ages:
            q111_data = df_distributions[
                (df_distributions['Channel'] == channel) &
                (df_distributions['Region'] == region) &
                (df_distributions['Mouse_Model'] == 'Q111') &
                (df_distributions['Age'] == age)
            ]

            if len(q111_data) > 0:
                q111_all = np.concatenate(q111_data['Cluster_Sizes_Array'].values)
                q111_cpc = q111_data['Clusters_per_Cell'].mean()
                caption_lines.append(f"  {age:.0f}mo Q111: n={len(q111_all):,}, "
                                   f"mean={np.mean(q111_all):.2f}, med={np.median(q111_all):.2f}, "
                                   f"{q111_cpc:.2f}/nuc")

    caption_lines.append("")
    caption_lines.append("ROWS 4-5 (I-P): DISTRIBUTIONS BY MOUSE ID")
    caption_lines.append("Cluster size distributions for individual mice (top 2 Q111 and top 2 WT per region).")
    caption_lines.append("Labels use Q#/W# format (see MOUSE ID MAPPING above).")
    caption_lines.append("")

    # COLOR SCHEME
    caption_lines.append("COLOR SCHEME:")
    caption_lines.append("-" * 80)
    if channel == 'HTT1a':
        caption_lines.append(f"Q111: Green ({COLOR_Q111_MHTT1A})")
        caption_lines.append(f"Wildtype: Blue ({COLOR_WT_MHTT1A})")
    else:
        caption_lines.append(f"Q111: Orange ({COLOR_Q111_FULL})")
        caption_lines.append(f"Wildtype: Purple ({COLOR_WT_FULL})")
    caption_lines.append("")

    # METHODOLOGY
    caption_lines.append("METHODOLOGY:")
    caption_lines.append("-" * 80)
    caption_lines.append("Cluster size calculation:")
    caption_lines.append("  - Cluster intensity = sum of all voxel intensities in a cluster")
    caption_lines.append("  - Single spot peak intensity = mode of KDE-fitted intensity distribution")
    caption_lines.append("  - Cluster size (mRNA equivalents) = Cluster intensity / Single spot peak")
    caption_lines.append("  - Normalization is slide-specific to account for batch effects")
    caption_lines.append("")
    caption_lines.append("Cluster filtering:")
    caption_lines.append(f"  - CV threshold: CV >= {CV_THRESHOLD} (removes low-variance noise)")
    caption_lines.append("  - Intensity threshold: > 95th percentile of negative control")
    caption_lines.append("  - Minimum 5 valid clusters per FOV for inclusion")
    caption_lines.append("")

    caption_lines.append("=" * 80)
    caption_lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    caption_lines.append("=" * 80)

    # Save caption
    channel_name = channel.replace(' ', '_').replace('-', '_')
    caption_path = OUTPUT_DIR / f"fig_{fig_num}_caption_{channel_name}.txt"
    with open(caption_path, 'w') as f:
        f.write('\n'.join(caption_lines))

    print(f"  Saved caption: {caption_path}")

# SUPPLEMENTARY FIGURE CAPTIONS (DYNAMIC)
for channel in ['HTT1a', 'fl-HTT']:
    channel_display = 'HTT1a' if channel == 'HTT1a' else 'fl-HTT'
    channel_name = channel.replace(' ', '_').replace('-', '_')

    supp_caption_lines = []
    supp_caption_lines.append("=" * 80)
    supp_caption_lines.append(f"SUPPLEMENTARY FIGURE: Cluster Size by Atlas Coordinate - {channel_display}")
    supp_caption_lines.append("=" * 80)
    supp_caption_lines.append("")
    supp_caption_lines.append("OVERVIEW:")
    supp_caption_lines.append("-" * 80)
    supp_caption_lines.append("This supplementary figure shows cluster size distributions for ALL brain atlas")
    supp_caption_lines.append("coordinates sampled in the study. Each panel represents one atlas coordinate")
    supp_caption_lines.append("(A-P position in 25μm units from Bregma).")
    supp_caption_lines.append("")

    supp_caption_lines.append("DATA SOURCE:")
    supp_caption_lines.append("-" * 80)
    supp_caption_lines.append(f"HDF5 file: {H5_FILE_PATH_EXPERIMENTAL}")
    supp_caption_lines.append("")

    supp_caption_lines.append("FILTERING APPLIED:")
    supp_caption_lines.append("-" * 80)
    supp_caption_lines.append(f"1. Excluded slides: {', '.join(EXCLUDED_SLIDES)}")
    supp_caption_lines.append(f"2. Cluster CV threshold: CV >= {CV_THRESHOLD}")
    supp_caption_lines.append("3. Intensity threshold: > 95th percentile of negative control")
    supp_caption_lines.append("")

    # List all atlas coordinates with counts
    for region in ['Cortex', 'Striatum']:
        region_data = df_distributions[
            (df_distributions['Channel'] == channel) &
            (df_distributions['Region'] == region)
        ]
        all_coords = sorted(region_data[region_data['Mouse_Model'] == 'Q111']['Brain_Atlas_Coord'].unique())

        total_clusters = int(region_data['N_Clusters'].sum())
        supp_caption_lines.append(f"{region}:")
        supp_caption_lines.append(f"  Atlas coordinates ({len(all_coords)} total): {', '.join([f'{c:.0f}' for c in all_coords])}")
        supp_caption_lines.append(f"  Total clusters: {total_clusters:,}")
        supp_caption_lines.append("")

    supp_caption_lines.append("COLOR SCHEME:")
    supp_caption_lines.append("-" * 80)
    if channel == 'HTT1a':
        supp_caption_lines.append(f"Q111: Green ({COLOR_Q111_MHTT1A})")
        supp_caption_lines.append(f"Wildtype: Blue ({COLOR_WT_MHTT1A})")
    else:
        supp_caption_lines.append(f"Q111: Orange ({COLOR_Q111_FULL})")
        supp_caption_lines.append(f"Wildtype: Purple ({COLOR_WT_FULL})")
    supp_caption_lines.append("")

    supp_caption_lines.append("=" * 80)
    supp_caption_lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    supp_caption_lines.append("=" * 80)

    # Save supplementary caption
    supp_caption_path = OUTPUT_DIR / f"fig_SUPP_caption_{channel_name}.txt"
    with open(supp_caption_path, 'w') as f:
        f.write('\n'.join(supp_caption_lines))

    print(f"  Saved supplementary caption: {supp_caption_path}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print(f"\nAll outputs saved to: {OUTPUT_DIR}")
print("\nMAIN FIGURES:")
print("  - fig_S13_cluster_size_HTT1a.pdf / .svg / .png")
print("  - fig_S14_cluster_size_fl_HTT.pdf / .svg / .png")
print("\nSUPPLEMENTARY FIGURES (by atlas coordinate):")
print("  - fig_SUPP_cluster_size_HTT1a_Cortex.svg")
print("  - fig_SUPP_cluster_size_HTT1a_Striatum.svg")
print("  - fig_SUPP_cluster_size_fl_HTT_Cortex.svg")
print("  - fig_SUPP_cluster_size_fl_HTT_Striatum.svg")
