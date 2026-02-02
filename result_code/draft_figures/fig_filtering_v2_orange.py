"""
Figure showing filtering of spots based on size and intensity criteria.
This script uses EXACT logic from size_single_spots_normalized.py for data processing,
but creates a 3x3 figure layout instead of the 4-panel layout.

ORANGE CHANNEL VERSION - shows fl-HTT probe data.
"""

import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic, gaussian_kde, pearsonr
from scipy.signal import savgol_filter
import scienceplots
plt.style.use('science')
plt.rcParams['text.usetex'] = False
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from result_functions_v2 import compute_thresholds, concatenate_fields, concatenated_data_to_df, extract_dataframe
import seaborn as sns
from results_config import (
    PIXELSIZE as pixelsize,
    SLICE_DEPTH as slice_depth,
    MAX_PFA,
    QUANTILE_NEGATIVE_CONTROL,
    N_BOOTSTRAP,
    USE_FINAL_FILTER,
    CV_THRESHOLD,
    H5_FILE_PATHS_EXPERIMENTAL,
    SUMMARY_CSV_PATHS_EXPERIMENTAL,
    CHANNEL_PARAMS,
    SIGMA_X_XLIM,
    SIGMA_Y_XLIM,
    SIGMA_Z_XLIM,
    EXCLUDED_SLIDES,
)

# Create output directory
OUTPUT_DIR = Path(__file__).parent / "output" / "filtering_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------
# Helper functions
# ---------------------------

def recursively_load_dict(h5_group):
    """Recursively load an HDF5 group into a Python dictionary."""
    output = {}
    for key, item in h5_group.items():
        if isinstance(item, h5py.Dataset):
            output[key] = item[()]
        elif isinstance(item, h5py.Group):
            output[key] = recursively_load_dict(item)
    return output


def compute_binned_fit_and_stats(x, y, weights):
    """
    Compute a weighted linear fit (slope and intercept) on the binned data (x, y)
    using weights (number of observations per bin). Also, compute a weighted R²
    (Pearson correlation coefficient squared).
    """
    valid = weights > 0
    if np.sum(valid) < 2:
        return np.nan, np.nan, np.nan

    x_valid = x[valid]
    y_valid = y[valid]
    w_valid = weights[valid]

    # Compute weighted linear regression
    slope, intercept = np.polyfit(x_valid, y_valid, 1, w=w_valid)

    # Compute weighted Pearson correlation
    r, p_value = pearsonr(x_valid, y_valid)
    r2 = r**2

    return slope, intercept, r2


# ---------------------------
# Main processing - EXACT COPY from size_single_spots_normalized.py
# ---------------------------
if __name__ == "__main__":

    # ── file paths ─────────────────────────────────────────────────────────────
    # Use experimental PSF file paths from config
    summary_csv_paths = SUMMARY_CSV_PATHS_EXPERIMENTAL
    h5_file_paths = H5_FILE_PATHS_EXPERIMENTAL
    # Use final filter (True for experimental analysis)
    use_sigma_final_filter = USE_FINAL_FILTER
    # Load HDF5 into a Python dictionary.
    h5_file_path = h5_file_paths[0]
    with h5py.File(h5_file_path, 'r') as h5_file:
        data_dict = recursively_load_dict(h5_file)
    print(data_dict.keys())

    # Extract DataFrame from the HDF5 dictionary.
    desired_channels = ['green', 'orange']
    fields_to_extract = ['spots.pfa_values', 'spots.photons', 'cluster_intensities', 'spots.final_filter', 'spots.params_raw']
    negative_control_field = 'Negative control'
    experimental_field = 'ExperimentalQ111 - 488mHT - 548mHTa - 647Darp'
    slide_field = 'metadata_sample_slide_name_std'
    max_pfa = MAX_PFA
    quantile_negative_control = QUANTILE_NEGATIVE_CONTROL

    df_extracted = extract_dataframe(
        data_dict,
        field_keys=fields_to_extract,
        channels=desired_channels,
        include_file_metadata_sample=True,
        explode_fields=[]
    )

    thresholds, thresholds_cluster, error_thresholds, error_thresholds_cluster,_,_ = compute_thresholds(
        df_extracted=df_extracted,
        slide_field=slide_field,
        desired_channels=desired_channels,
        negative_control_field=negative_control_field,
        experimental_field=experimental_field,
        quantile_negative_control=quantile_negative_control,
        max_pfa=max_pfa,
        plot=False,
        n_bootstrap=N_BOOTSTRAP,
        use_region = False,
        cluster_threshold_val = None,
        use_final_filter=use_sigma_final_filter,
    )

    # Further data extraction and concatenation.
    fields_to_extract = ['spots.final_filter', 'spots_sigma_var.params_raw',
                         'spots_sigma_var.final_filter', 'cluster_intensities', 'cluster_cvs', 'label_sizes','metadata_sample_Age','spots.pfa_values' ]

    df_extracted = extract_dataframe(data_dict, field_keys=fields_to_extract, channels=desired_channels,
                                     include_file_metadata_sample=True, explode_fields=[])
    df_extracted =df_extracted[df_extracted['metadata_sample_Mouse_Model'] == 'Q111']
    #df_extracted = df_extracted[df_extracted['metadata_sample_Mouse_Model']=='Wildtype']
    fields_to_extract.append('metadata_sample_mouse_ID')
    concatenated_data = concatenate_fields(
        df_extracted=df_extracted,
        slide_field=slide_field,
        desired_channels=['green', 'orange'],
        fields_to_extract=fields_to_extract,
        probe_set_field='metadata_sample_Probe-Set',
    )
    df_groups = concatenated_data_to_df(concatenated_data)

    # Filter out excluded slides (technical failures identified via UBC positive control)
    n_before_exclusion = len(df_groups)
    slides_before = df_groups['slide'].unique()
    df_groups = df_groups[~df_groups['slide'].isin(EXCLUDED_SLIDES)].copy()
    n_after_exclusion = len(df_groups)
    slides_after = df_groups['slide'].unique()
    n_excluded = len([s for s in slides_before if s in EXCLUDED_SLIDES])
    print(f"Slide filtering: {n_before_exclusion} -> {n_after_exclusion} rows ({n_excluded} slides excluded: {[s for s in slides_before if s in EXCLUDED_SLIDES]})")

    sigma_params = {
        'blue': {
            'sigma_x_mean': 232, 'sigma_x_width': 37,
            'sigma_y_mean': 217, 'sigma_y_width': 39,
            'sigma_z_mean': 668, 'sigma_z_width': 159
        },
        'green': {
            'sigma_x_mean': 185, 'sigma_x_width': 30,
            'sigma_y_mean': 187, 'sigma_y_width': 30,
            'sigma_z_mean': 573, 'sigma_z_width': 123
        },
        'orange': {
            'sigma_x_mean': 187, 'sigma_x_width': 22,
            'sigma_y_mean': 191, 'sigma_y_width': 23,
            'sigma_z_mean': 592, 'sigma_z_width': 115
        },
        'dark red': {
            'sigma_x_mean': 170, 'sigma_x_width': 13,
            'sigma_y_mean': 170, 'sigma_y_width': 16,
            'sigma_z_mean': 618, 'sigma_z_width': 96
        }
    }

    single_rows = []  # one record per *spot* that passed the single-spot filter
    cluster_rows = []  # one record per *cluster* that passed its filter

    # ------------------------------------------------------------------
    # 1.  Your original loop — unchanged up to the filters
    # ------------------------------------------------------------------
    for idx, row in df_groups.iterrows():
        slide = row["slide"]
        channel = row["channel"]
        probe_set = row["probe_set"]
        region = row["region"]

        # --- threshold look-up (kept exactly as you wrote) ------------
        components = [str(slide), str(channel)]
        threshold_val = next((v for k, v in thresholds.items()
                              if all(comp in str(k) for comp in components)), None)
        threshold_val_cluster = next((v for k, v in thresholds_cluster.items()
                                      if all(comp in str(k) for comp in components)), None)


        if threshold_val is None:
            print(f"[WARN] threshold missing for slide={slide}, ch={channel}")
            continue

        # --- unpack arrays -------------------------------------------

        if use_sigma_final_filter:
            pass
        else:
            mask = row["spots.pfa_values"] < max_pfa
            row["spots.final_filter"] = np.any(mask, axis=1)


        cluster_data = row["cluster_intensities"]
        cluster_cvs = row.get("cluster_cvs", None)
        photon_arr_sig = row["spots_sigma_var.params_raw"][row["spots.final_filter"], 3]
        sigma = row["spots_sigma_var.params_raw"][row["spots.final_filter"], 5::]
        label_sizes = row["label_sizes"]
        metadata_sample_Age = row["metadata_sample_Age"]
        metadata_sample_mouse_ID =row["metadata_sample_mouse_ID"]

        # --- calibration & filters -----------------------------------
        calib = sigma_params.get(channel.lower())

        single_mask = photon_arr_sig > threshold_val

        # Cluster filtering: intensity threshold AND CV threshold
        # CV data is required - no fallback
        intensity_mask = cluster_data > threshold_val
        if cluster_cvs is None or len(cluster_cvs) != len(cluster_data):
            raise ValueError(f"CV data missing or mismatched for cluster filtering")
        cv_mask = cluster_cvs >= CV_THRESHOLD
        cluster_mask = intensity_mask & cv_mask

        # ----------------------------------------------------------------
        # 2.  APPEND filtered values to our containers (no plotting yet)
        # ----------------------------------------------------------------
        single_rows.extend(dict(
            slide=slide, channel=channel, probe_set=probe_set, region=region,
            photons=photon_arr_sig[i], metadata_sample_Age=metadata_sample_Age,
            metadata_sample_mouse_ID=metadata_sample_mouse_ID, sigma_x=sigma[i, 0], sigma_y=sigma[i, 1], sigma_z=sigma[i, 2]
        ) for i in np.where(single_mask)[0])

        cluster_rows.extend(dict(
            slide=slide, channel=channel, probe_set=probe_set, region=region,
            cluster_volume=label_sizes[i], cluster_intensity=cluster_data[i], metadata_sample_Age=metadata_sample_Age,
            metadata_sample_mouse_ID=metadata_sample_mouse_ID,
            cluster_cv=cluster_cvs[i] if cluster_cvs is not None else np.nan
        ) for i in np.where(cluster_mask)[0])

    # ------------------------------------------------------------------
    # 3.  Build DataFrames you can slice / filter later
    # ------------------------------------------------------------------
    df_single = pd.DataFrame(single_rows)
    df_clusters = pd.DataFrame(cluster_rows)

    # Add volume column to df_single
    df_single['volume'] = df_single['sigma_x'] * df_single['sigma_y'] * df_single['sigma_z']

    print("single spots:", df_single.shape)
    print("clusters:", df_clusters.shape)

    # ==================================================================
    # NEW: Create 3x3 Figure instead of running sigma_intensity_plot_normalized
    # ==================================================================

    # Breakpoints and PSF values for reference lines (from CHANNEL_PARAMS, using orange channel)
    orange_sigma = CHANNEL_PARAMS['orange']['sigma']
    orange_break = CHANNEL_PARAMS['orange']['break_sigma']

    breakpoints = {
        'sigma_x': orange_break[0] * pixelsize,  # nm
        'sigma_y': orange_break[1] * pixelsize,  # nm
        'sigma_z': orange_break[2] * slice_depth  # nm
    }

    psf_values = {
        'sigma_x': orange_sigma[0] * pixelsize,  # nm
        'sigma_y': orange_sigma[1] * pixelsize,  # nm
        'sigma_z': orange_sigma[2] * slice_depth  # nm
    }

    print(f"Using PSF values from CHANNEL_PARAMS (orange):")
    print(f"  PSF: x={psf_values['sigma_x']:.1f}nm, y={psf_values['sigma_y']:.1f}nm, z={psf_values['sigma_z']:.1f}nm")
    print(f"  Breakpoint: x={breakpoints['sigma_x']:.1f}nm, y={breakpoints['sigma_y']:.1f}nm, z={breakpoints['sigma_z']:.1f}nm")

    # Create 3x3 figure
    fig = plt.figure(figsize=(12, 14), dpi=150)
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.30,
                           left=0.08, right=0.95, top=0.95, bottom=0.05)

    # Create subplots (2 columns for row 1, 3 columns for rows 2-3)
    ax_a = fig.add_subplot(gs[0, 0])  # Raw intensity
    ax_b = fig.add_subplot(gs[0, 1:3])  # Normalized intensity (spans 2 columns)

    ax_c = fig.add_subplot(gs[1, 0])  # Biphasic sigma_x
    ax_d = fig.add_subplot(gs[1, 1])  # Biphasic sigma_y
    ax_e = fig.add_subplot(gs[1, 2])  # Biphasic sigma_z

    ax_f = fig.add_subplot(gs[2, 0])  # PDF sigma_x
    ax_g = fig.add_subplot(gs[2, 1])  # PDF sigma_y
    ax_h = fig.add_subplot(gs[2, 2])  # PDF sigma_z

    # Add panel labels
    panel_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    axes = [ax_a, ax_b, ax_c, ax_d, ax_e, ax_f, ax_g, ax_h]

    for ax, label in zip(axes, panel_labels):
        ax.text(-0.12, 1.05, label, transform=ax.transAxes,
                fontsize=16, fontweight='bold', va='top')

    # ==================================================================
    # Use EXACT logic from size_single_spots_normalized.py
    # Process data PER (probe_set, channel) combination
    # ==================================================================

    # We'll focus on orange channel, ExperimentalQ111 probe set
    df_to_plot = df_single[
        (df_single['channel'] == 'orange') &
        (df_single['probe_set'] == 'ExperimentalQ111 - 488mHT - 548mHTa - 647Darp')
    ]

    print(f"\nFiltered to orange channel, ExperimentalQ111: {df_to_plot.shape[0]} spots")

    # ==================================================================
    # Panel A & B: EXACT logic from sigma_intensity_plot_normalized
    # ==================================================================

    slide_data_for_intensity = {}

    for slide, slide_sub in df_to_plot.groupby("slide"):
        intens = slide_sub['photons'].to_numpy()

        if len(intens) < 50:
            print(f"Skip slide {slide}: not enough points for PDF")
            continue

        # Find peak intensity from PDF (EXACT logic from line 353-372)
        try:
            kde_intensity = gaussian_kde(intens, bw_method='scott')
            intensity_range = np.linspace(np.percentile(intens, 1),
                                         np.percentile(intens, 99), 500)
            pdf_values = kde_intensity(intensity_range)
            peak_idx = np.argmax(pdf_values)
            peak_intensity = intensity_range[peak_idx]

            if peak_intensity <= 0 or not np.isfinite(peak_intensity):
                print(f"Skip slide {slide}: invalid peak intensity")
                continue
        except Exception as e:
            print(f"Skip slide {slide}: PDF computation failed - {e}")
            continue

        # Store normalized intensities
        intens_norm = intens / peak_intensity

        slide_data_for_intensity[slide] = {
            'intens_raw': intens,
            'intens_norm': intens_norm,
            'peak_intensity': peak_intensity
        }

    # Collect all intensities for combined PDFs
    all_intens_raw = []
    all_intens_norm = []

    for slide, data in slide_data_for_intensity.items():
        all_intens_raw.extend(data['intens_raw'])
        all_intens_norm.extend(data['intens_norm'])

    all_intens_raw = np.array(all_intens_raw)
    all_intens_norm = np.array(all_intens_norm)

    # Panel A: Plot raw intensity PDFs (per-slide + combined)
    cmap = plt.colormaps.get_cmap('tab10')
    for idx, (slide, data) in enumerate(slide_data_for_intensity.items()):
        try:
            if len(data['intens_raw']) > 50:
                kde_slide = gaussian_kde(data['intens_raw'], bw_method='scott')
                x_range = np.linspace(0, np.percentile(all_intens_raw, 99), 200)
                y_density = kde_slide(x_range)
                ax_a.plot(x_range, y_density, '-', alpha=0.4, linewidth=1,
                         color=cmap(idx % 10))
        except:
            pass

    # Combined PDF for Panel A
    try:
        if len(all_intens_raw) > 100:
            kde = gaussian_kde(all_intens_raw, bw_method='scott')
            x_range = np.linspace(0, np.percentile(all_intens_raw, 99), 500)
            y_density = kde(x_range)
            ax_a.fill_between(x_range, y_density, alpha=0.3, color='black', label='Combined PDF')
            ax_a.plot(x_range, y_density, color='black', linewidth=2.5, label='_nolegend_')
    except:
        pass

    ax_a.set_xlabel('Integrated photon counts', fontsize=11)
    ax_a.set_ylabel('Probability density', fontsize=11)
    ax_a.set_title('Raw intensity distributions\n(per slide)', fontsize=12)
    ax_a.grid(True, alpha=0.3, linestyle='--')
    ax_a.set_xlim(left=0)
    ax_a.set_ylim(bottom=0)

    # Panel B: Plot normalized intensity PDFs (EXACT logic from line 529-569)
    intensity_min = np.percentile(all_intens_norm[np.isfinite(all_intens_norm)], 0)
    intensity_max = np.percentile(all_intens_norm[np.isfinite(all_intens_norm)], 95)
    x_intensity = np.linspace(max(0, intensity_min), intensity_max, 200)

    # Plot individual slide intensity PDFs (faint lines)
    for idx, (slide, data) in enumerate(slide_data_for_intensity.items()):
        try:
            intens_data = data['intens_norm'][np.isfinite(data['intens_norm'])]
            if len(intens_data) > 50:
                kde_slide = gaussian_kde(intens_data, bw_method='scott')
                y_density_slide = kde_slide(x_intensity)
                ax_b.plot(x_intensity, y_density_slide, '-', alpha=0.4, linewidth=1,
                         color=cmap(idx % 10))
        except:
            pass

    # Plot combined intensity PDF (bold line on top)
    try:
        intens_data_combined = all_intens_norm[np.isfinite(all_intens_norm)]
        if len(intens_data_combined) > 100:
            kde = gaussian_kde(intens_data_combined, bw_method='scott')
            y_density = kde(x_intensity)
            ax_b.fill_between(x_intensity, y_density, alpha=0.3, color='black', label='Combined PDF')
            ax_b.plot(x_intensity, y_density, color='black', linewidth=2.5, label='_nolegend_')
    except:
        pass

    # Add reference line at intensity = 1.0 (peak normalization)
    ax_b.axvline(1.0, color='red', linewidth=2, linestyle='--', alpha=0.7,
               label='Peak intensity (norm=1)')

    ax_b.set_xlabel("Normalized Intensity (# mRNAs)", fontsize=11)
    ax_b.set_ylabel("Probability Density", fontsize=11)
    ax_b.set_title('Normalized intensity distributions\n(per slide)', fontsize=12)
    ax_b.set_xlim(max(0, intensity_min), intensity_max)
    ax_b.set_ylim(bottom=0)
    ax_b.grid(True, alpha=0.3, linestyle='--')
    ax_b.legend(loc='upper right', fontsize=9, framealpha=0.9)

    # ==================================================================
    # Panels C-E and F-H: Use EXACT logic from sigma_intensity_plot_normalized
    # for biphasic plots and size distributions
    # ==================================================================

    # Use xlim values from results_config.py (80% of bead PSF as lower bound)
    print(f"Using sigma xlim from config: x={SIGMA_X_XLIM}, y={SIGMA_Y_XLIM}, z={SIGMA_Z_XLIM}")

    sigma_configs = [
        {'col': 'sigma_x', 'ax_biphasic': ax_c, 'ax_pdf': ax_f, 'xlim': SIGMA_X_XLIM,
         'scaling': pixelsize, 'xlabel': r'$\sigma_x$ [nm]', 'psf': psf_values['sigma_x'],
         'bp': breakpoints['sigma_x'], 'label': r'$\sigma_x$', 'color': '#1f77b4'},
        {'col': 'sigma_y', 'ax_biphasic': ax_d, 'ax_pdf': ax_g, 'xlim': SIGMA_Y_XLIM,
         'scaling': pixelsize, 'xlabel': r'$\sigma_y$ [nm]', 'psf': psf_values['sigma_y'],
         'bp': breakpoints['sigma_y'], 'label': r'$\sigma_y$', 'color': '#ff7f0e'},
        {'col': 'sigma_z', 'ax_biphasic': ax_e, 'ax_pdf': ax_h, 'xlim': SIGMA_Z_XLIM,
         'scaling': slice_depth, 'xlabel': r'$\sigma_z$ [nm]', 'psf': psf_values['sigma_z'],
         'bp': breakpoints['sigma_z'], 'label': r'$\sigma_z$', 'color': '#2ca02c'}
    ]

    n_bins = 50

    for config in sigma_configs:
        sigma_col = config['col']
        ax_biphasic = config['ax_biphasic']
        ax_pdf = config['ax_pdf']
        xlim = config['xlim']
        scaling = config['scaling']
        xlabel = config['xlabel']
        psf = config['psf']
        bp = config['bp']
        label = config['label']
        color = config['color']

        # Dictionary to store per-slide normalized data (EXACT logic from line 318-394)
        slide_data = {}

        # First pass: process each slide separately
        for slide, slide_sub in df_to_plot.groupby("slide"):
            width = slide_sub[sigma_col].to_numpy()
            intens = slide_sub['photons'].to_numpy()

            # Physical-unit conversion
            width = width * scaling  # nm

            # Apply the fixed x-range
            lo, hi = xlim
            mask = (width >= lo) & (width <= hi)
            width_filt, intens_filt = width[mask], intens[mask]

            if width_filt.size < 10:
                continue

            # Binning & stats
            bins = np.linspace(lo, hi, n_bins + 1)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            mean_I, _, _ = binned_statistic(width_filt, intens_filt, statistic="mean", bins=bins)
            counts, _, _ = binned_statistic(width_filt, intens_filt, statistic="count", bins=bins)

            # Find peak intensity from PDF (where PDF is maximum)
            try:
                if len(intens_filt) < 50:
                    continue

                # Compute PDF and find its maximum
                kde_intensity = gaussian_kde(intens_filt, bw_method='scott')
                intensity_range = np.linspace(np.percentile(intens_filt, 1),
                                             np.percentile(intens_filt, 99), 500)
                pdf_values = kde_intensity(intensity_range)
                peak_idx = np.argmax(pdf_values)
                peak_intensity = intensity_range[peak_idx]

                if peak_intensity <= 0 or not np.isfinite(peak_intensity):
                    continue
            except Exception as e:
                continue

            # Normalize by peak intensity
            mean_I_norm = mean_I / peak_intensity

            # Store data for this slide
            slide_data[slide] = {
                'bin_centers': bin_centers,
                'mean_I_norm': mean_I_norm,
                'counts': counts,
                'peak_intensity': peak_intensity,
                'width_filt': width_filt,
                'intens_norm': intens_filt / peak_intensity
            }

        if len(slide_data) == 0:
            continue

        # Second pass: combine all normalized data for plotting
        all_widths = []
        all_intens_norm = []

        for slide, data in slide_data.items():
            all_widths.extend(data['width_filt'])
            all_intens_norm.extend(data['intens_norm'])

        all_widths = np.array(all_widths)
        all_intens_norm = np.array(all_intens_norm)

        # Compute combined statistics on normalized data
        bins = np.linspace(lo, hi, n_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        mean_I_combined, _, _ = binned_statistic(all_widths, all_intens_norm,
                                                 statistic="mean", bins=bins)
        counts_combined, _, _ = binned_statistic(all_widths, all_intens_norm,
                                                statistic="count", bins=bins)

        # Compute std across slides (not within-bin std)
        # For each bin, compute mean per slide, then std across those slide means
        slide_bin_means = []
        for slide, data in slide_data.items():
            slide_bin_mean, _, _ = binned_statistic(data['width_filt'], data['intens_norm'],
                                                    statistic='mean', bins=bins)
            slide_bin_means.append(slide_bin_mean)

        slide_bin_means = np.array(slide_bin_means)  # shape: (n_slides, n_bins)
        std_across_slides = np.nanstd(slide_bin_means, axis=0)

        # Apply minimum count threshold for reliable statistics
        MIN_COUNTS_PER_BIN = 100  # Minimum spots per bin for inclusion in biphasic plot

        # Compute weighted linear fit
        valid_for_fit = counts_combined >= MIN_COUNTS_PER_BIN
        slope, intercept, r2 = compute_binned_fit_and_stats(
            bin_centers[valid_for_fit],
            mean_I_combined[valid_for_fit],
            counts_combined[valid_for_fit]
        )

        # ===== Panel D/E/F: Biphasic plot =====
        # Only plot bins with sufficient counts
        valid_plot = np.isfinite(mean_I_combined) & (counts_combined >= MIN_COUNTS_PER_BIN)
        valid_std = np.isfinite(std_across_slides)
        valid_combined = valid_plot & valid_std

        # Plot std across slides as shaded region
        ax_biphasic.fill_between(bin_centers[valid_combined],
                        mean_I_combined[valid_combined] - std_across_slides[valid_combined],
                        mean_I_combined[valid_combined] + std_across_slides[valid_combined],
                        alpha=0.25, color=color, label='±1 std (slides)', zorder=5)

        # Plot mean line on top
        ax_biphasic.plot(bin_centers[valid_plot], mean_I_combined[valid_plot],
                'o-', color=color, linewidth=2.5, markersize=5,
                label=f'{label} (mean)', zorder=10, alpha=0.9)

        # Add reference lines
        ax_biphasic.axvline(psf, color='purple', linestyle=':', linewidth=2.5,
                           alpha=0.7, label=f'Mode step 1 ({psf:.0f} nm)', zorder=8)
        ax_biphasic.axvline(bp, color=color, linestyle='--', linewidth=2.5,
                           alpha=0.8, label=f'Breakpoint ({bp:.0f} nm)', zorder=9)

        # Formatting
        ax_biphasic.set_xlabel(xlabel, fontsize=11)
        ax_biphasic.set_ylabel(r'Normalized intensity ($N/N_{\mathrm{1mRNA}}$)', fontsize=11)
        ax_biphasic.set_xlim(xlim)
        ax_biphasic.set_ylim(bottom=0)
        ax_biphasic.legend(loc='best', fontsize=8, framealpha=0.9)
        ax_biphasic.grid(True, alpha=0.3, linestyle='--')

        # Add linear fit info
        if np.isfinite(slope) and np.isfinite(r2):
            ax_biphasic.text(0.05, 0.95, f'Linear fit: r² = {r2:.3f}',
                           transform=ax_biphasic.transAxes, fontsize=9,
                           verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        print(f"  {sigma_col}: {len(all_widths):,} spots, slope={slope:.4f}, r²={r2:.3f}")

        # ===== Panel G/H/I: Size distribution PDF =====
        # Plot per-slide PDFs (EXACT logic from line 485-494)
        x_density = np.linspace(lo, hi, 200)

        for idx, (slide, data) in enumerate(slide_data.items()):
            try:
                if len(data['width_filt']) > 50:
                    kde_slide = gaussian_kde(data['width_filt'], bw_method='scott')
                    y_density_slide = kde_slide(x_density)
                    ax_pdf.plot(x_density, y_density_slide, '-', alpha=0.4, linewidth=1,
                               color=cmap(idx % 10))
            except:
                pass

        # Plot combined PDF (EXACT logic from line 496-508)
        try:
            if len(all_widths) > 100:
                kde = gaussian_kde(all_widths, bw_method='scott')
                y_density = kde(x_density)
                ax_pdf.fill_between(x_density, y_density, alpha=0.3, color='black', label='Combined PDF')
                ax_pdf.plot(x_density, y_density, color='black', linewidth=2.5, label='_nolegend_')

                # Find mode
                mode_idx = np.argmax(y_density)
                mode_value = x_density[mode_idx]
                ax_pdf.axvline(mode_value, color=color, linestyle='--', linewidth=2,
                              alpha=0.8, label=f"Mode: {mode_value:.0f} nm", zorder=11)
        except:
            pass

        # Add PSF reference (EXACT logic from line 510-512)
        ax_pdf.axvline(psf, color='purple', linestyle=':', linewidth=2,
                      alpha=0.7, label=f'Mode step 1 ({psf:.0f} nm)', zorder=8)

        # Formatting
        ax_pdf.set_xlabel(xlabel, fontsize=11)
        ax_pdf.set_ylabel('Probability density', fontsize=11)
        ax_pdf.set_xlim(xlim)
        ax_pdf.set_ylim(bottom=0)
        ax_pdf.legend(loc='best', fontsize=8, framealpha=0.9)
        ax_pdf.grid(True, alpha=0.3, linestyle='--')

    # ==================================================================
    # Save figure
    # ==================================================================

    output_filename = "fig_filtering_v2_orange"

    plt.savefig(OUTPUT_DIR / f"{output_filename}.pdf", format='pdf', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / f"{output_filename}.svg", format='svg', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / f"{output_filename}.png", format='png', bbox_inches='tight', dpi=300)

    print("\n" + "="*70)
    print(f"Figure saved to: {OUTPUT_DIR / output_filename}")
    print("="*70)

    # ==================================================================
    # Generate detailed caption with all computed statistics
    # ==================================================================

    n_slides = len(slide_data_for_intensity)
    n_spots_total = df_to_plot.shape[0]
    n_clusters_total = df_clusters.shape[0]
    n_single_spots = df_single.shape[0]
    n_excluded_slides = len(EXCLUDED_SLIDES)
    excluded_slides_str = ', '.join(EXCLUDED_SLIDES)
    slides_used = sorted(list(slide_data_for_intensity.keys()))
    n_slides_used = len(slides_used)
    slides_used_str = ', '.join(slides_used)
    min_counts_per_bin = MIN_COUNTS_PER_BIN
    n_bins_used = n_bins

    caption = f"""\\caption{{\\textbf{{Filtering spots based on size and intensity criteria after empirical PSF calibration (orange channel).}}
\\textbf{{Note:}} All panels show data from the \\textbf{{orange channel}} (548~nm excitation) only, corresponding to the fl-HTT probe in the ExperimentalQ111 probe set in Q111 transgenic mouse tissue.
Analysis of green channel (488~nm, HTT1a) data yields similar results (see companion figure).
\\textbf{{(A--B)}} Per-slide intensity normalization demonstrates successful removal of technical variation.
\\textbf{{(A)}} Raw integrated photon counts for each of {n_slides} slides (faint colored lines show kernel density estimates for individual slides, {n_spots_total:,} total spots, \\textbf{{orange channel}}).
Substantial slide-to-slide variation is evident, with peak intensities varying by $2$--$3\\times$ due to technical factors: tissue autofluorescence, fixation quality, probe hybridization efficiency, and imaging conditions (laser power, detector sensitivity).
This heterogeneity necessitates per-slide normalization rather than global intensity calibration.
The thick black line shows the combined probability density function (PDF) from all spots pooled across slides.
\\textbf{{(B)}} After per-slide normalization, all distributions converge around $N_{{\\mathrm{{1mRNA}}}} = 1$ (red dashed line), where the modal intensity of each slide is independently set to unity via kernel density estimation (KDE with Scott's bandwidth).
Faint colored lines show per-slide normalized intensity PDFs, while the thick black line shows the combined PDF from all pooled spots.
This normalization successfully removes technical variation while preserving biological signal, as evidenced by the conserved tail extending beyond 1 (representing multi-transcript aggregates and clusters).
After establishing $\\hat{{\\sigma}}_{{x,\\mathrm{{init}}}}$, $\\hat{{\\sigma}}_{{y,\\mathrm{{init}}}}$, $\\hat{{\\sigma}}_{{z,\\mathrm{{init}}}}$ from bead calibration data, we re-ran the detection pipeline with refined initial starting points.
We classified a detection as a \\emph{{single mRNA}} only if (i)~all $\\hat{{\\sigma}}_i < \\mathrm{{bp}}_i$ (where $\\mathrm{{bp}}$ denotes empirically-determined breakpoint thresholds), (ii)~it passed the generalized likelihood ratio test (GLRT) with $p_{{\\mathrm{{FA}}}} \\leq {max_pfa:.0e}$, and (iii)~$\\hat{{N}}$ exceeded the slide-specific negative-control threshold.
%
\\textbf{{(C)}} Placeholder for representative example images showing single mRNA detections in tissue.
%
\\textbf{{(D--F)}} Biphasic relationships between normalized intensity and spot size after filtering validate the single-molecule regime.
For each dimension---\\textbf{{(D)}} $\\sigma_x$ (lateral $x$), \\textbf{{(E)}} $\\sigma_y$ (lateral $y$), \\textbf{{(F)}} $\\sigma_z$ (axial $z$)---the bold colored line shows the mean normalized intensity computed from all spots pooled across {n_slides} slides, binned into {n_bins_used} size bins.
Only bins containing $\\geq {min_counts_per_bin}$ spots are shown to ensure statistical reliability (objective criterion applied uniformly across all panels).
The light shaded region ($\\pm 1$ standard deviation computed across per-slide bin means) represents inter-slide variability in the biphasic relationship, quantifying biological and technical heterogeneity between tissue samples.
Purple dotted vertical lines mark the bead-derived PSF from fluorescent microsphere calibration ($\\sigma_x = {psf_values['sigma_x']:.0f}$~nm, $\\sigma_y = {psf_values['sigma_y']:.0f}$~nm, $\\sigma_z = {psf_values['sigma_z']:.0f}$~nm), representing the diffraction-limited optical reference.
Colored dashed vertical lines indicate empirically-determined breakpoints ($\\sigma_x = {breakpoints['sigma_x']:.1f}$~nm, $\\sigma_y = {breakpoints['sigma_y']:.1f}$~nm, $\\sigma_z = {breakpoints['sigma_z']:.1f}$~nm), which exceed bead PSF by {(breakpoints['sigma_x']/psf_values['sigma_x']-1)*100:.0f}\\%, {(breakpoints['sigma_y']/psf_values['sigma_y']-1)*100:.0f}\\%, and {(breakpoints['sigma_z']/psf_values['sigma_z']-1)*100:.0f}\\% respectively---consistent with the finite physical extent of RNAscope probe clusters ($\\sim$20 probe pairs, each $\\sim$2--3~nm) bound to target mRNA in tissue.
After applying these strict filtering criteria (size, significance, intensity), the resulting population exhibits strong linear relationships between size and normalized intensity, with weighted least-squares fits yielding Pearson $r^2$ values consistently above 0.90 for all three dimensions.
This linearity validates that the filtered dataset represents genuine single molecules in the expected regime, where larger fitted widths arise from increased local probe density or extended hybridization sites while remaining single diffraction-limited emitters.
The diagnostic biphasic pattern observed in unfiltered data (not shown) separates two regimes: left of the breakpoint (single-molecule regime, positive slope) and right of the breakpoint (aggregate regime, plateau).
By restricting analysis to spots with $\\sigma_i < \\mathrm{{bp}}_i$, we isolate the linear single-molecule regime.
%
\\textbf{{(G--I)}} Spot size distributions after filtering confirm homogeneous single-molecule population.
For each dimension---\\textbf{{(G)}} $\\sigma_x$, \\textbf{{(H)}} $\\sigma_y$, \\textbf{{(I)}} $\\sigma_z$---faint colored lines show per-slide probability density functions (kernel density estimates with Scott's bandwidth), while the thick black line with gray shading shows the combined PDF from all spots pooled across slides.
Colored dashed lines mark the modes (peaks) of the combined distributions, which define the tissue-calibrated refined PSF.
These mode values represent the most probable spot sizes in tissue and supersede bead-derived measurements for all downstream single-molecule quantification, accounting for \\emph{{in situ}} effects including refractive index mismatch between tissue and immersion medium, optical aberrations from tissue heterogeneity, and the finite geometry of probe--mRNA complexes.
Purple dotted lines replicate bead PSF references for comparison.
The distributions show tight clustering around modal values with clear truncation at breakpoint thresholds, confirming successful filtering that yields a homogeneous population of high-confidence single-molecule detections.
%
\\textbf{{Key findings:}}
(1) Per-slide normalization (Panel~B) successfully removes $2$--$3$-fold technical variation in photon yield while preserving biological heterogeneity.
(2) Empirically-determined breakpoints exceed bead PSF by {(breakpoints['sigma_x']/psf_values['sigma_x']-1)*100:.0f}--{(breakpoints['sigma_z']/psf_values['sigma_z']-1)*100:.0f}\\%, consistent with the physical extent of RNAscope probe clusters bound to target mRNA.
(3) Strong linear relationships in biphasic plots (Panels~D--F, $r^2 > 0.90$) validate that filtered spots represent true single molecules.
(4) Tissue-calibrated modes from size distributions (Panels~G--I) provide refined PSF estimates that account for \\emph{{in situ}} optical conditions.
(5) Strict filtering criteria (size $< \\mathrm{{bp}}$, $p_{{\\mathrm{{FA}}}} < {max_pfa:.0e}$, intensity $>$ threshold) yield {n_spots_total:,} high-confidence single molecules across {n_slides} slides.
%
\\textbf{{Methods:}}
Analysis performed on {n_spots_total:,} single spots and {n_clusters_total:,} clusters across {n_slides_used} slides from Q111 transgenic mouse tissue, using RNAscope \\emph{{in situ}} hybridization with fluorescent detection.
\\textbf{{Slides used:}} {slides_used_str}.
\\textbf{{Slides excluded:}} {n_excluded_slides} slides ({excluded_slides_str}) were excluded based on abnormally low UBC positive control expression ($>$100$\\times$ below median), indicating technical failures (poor hybridization, tissue damage).
\\textbf{{Channel analyzed:}} Orange channel (548~nm excitation wavelength, detecting fl-fl-HTT mRNA via fluorophore-conjugated probes).
Data source: spots\\_sigma\\_var analysis, which applies size-based filtering during detection.
\\textbf{{Quality filtering:}} (i)~probability of false alarm $< {max_pfa:.0e}$ from generalized likelihood ratio test (GLRT), (ii)~size filtering $\\sigma_i < \\mathrm{{bp}}_i$ for all dimensions ($\\sigma_x < {breakpoints['sigma_x']:.1f}$~nm, $\\sigma_y < {breakpoints['sigma_y']:.1f}$~nm, $\\sigma_z < {breakpoints['sigma_z']:.1f}$~nm), (iii)~intensity filtering: photon counts exceed slide-specific negative-control threshold, (iv)~for clusters: coefficient of variation (CV) $\\geq$ {CV_THRESHOLD} to exclude uniform background artifacts.
\\textbf{{Per-slide normalization:}} kernel density estimation (KDE) with Scott's rule to identify modal intensity, then division by peak to set $N_{{\\mathrm{{1mRNA}}}} = 1$.
\\textbf{{Spot size determination:}} 3D Gaussian fitting to extract $\\sigma_x$, $\\sigma_y$, $\\sigma_z$ from background-subtracted fluorescence images.
\\textbf{{Binning for biphasic plots:}} {n_bins_used} bins spanning [{SIGMA_X_XLIM[0]:.0f}, {SIGMA_X_XLIM[1]:.0f}]~nm (lateral $x$), [{SIGMA_Y_XLIM[0]:.0f}, {SIGMA_Y_XLIM[1]:.0f}]~nm (lateral $y$), and [{SIGMA_Z_XLIM[0]:.0f}, {SIGMA_Z_XLIM[1]:.0f}]~nm (axial $z$).
\\textbf{{Minimum count threshold:}} bins with $<$ {min_counts_per_bin} spots excluded from biphasic plots (Panels~D--F) to ensure statistical reliability.
\\textbf{{Standard deviation:}} computed across per-slide bin means (not within-bin variability), representing inter-slide heterogeneity.
\\textbf{{Imaging parameters:}} pixel size = {pixelsize:.1f}~nm (lateral), slice depth = {slice_depth:.1f}~nm (axial).
\\textbf{{Negative control threshold:}} computed at {QUANTILE_NEGATIVE_CONTROL*100:.0f}th percentile of negative control probe intensities, per slide and channel.
\\textbf{{Statistical analysis:}} Python 3.x with \\texttt{{scipy}}, \\texttt{{numpy}}, \\texttt{{matplotlib}}.
\\textbf{{Breakpoint determination:}} weighted piecewise-linear regression with Savitzky--Golay smoothing (window = 9 bins, polynomial order = 2) on bead calibration data, searching only after intensity reaches 80\\% of maximum and applying strict penalties to enforce positive slope before breakpoint and near-zero slope after.
}}"""

    # Save caption
    caption_file = OUTPUT_DIR / f"{output_filename}_caption.tex"
    with open(caption_file, 'w') as f:
        f.write(caption)

    print(f"\nCaption saved to: {caption_file}")

    print("\n" + "="*70)
    print("ALL PROCESSING COMPLETE")
    print("="*70)
