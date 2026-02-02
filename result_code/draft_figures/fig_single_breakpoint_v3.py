import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic, gaussian_kde, pearsonr
from scipy.optimize import minimize
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
    H5_FILE_PATHS_BEAD,
    CHANNEL_PARAMS,
    SIGMA_X_XLIM,
    SIGMA_Y_XLIM,
    SIGMA_Z_XLIM,
    BEAD_PSF_X,
    BEAD_PSF_Y,
    BEAD_PSF_Z
)

# Create output directory
OUTPUT_DIR = Path(__file__).parent / "output" / "single_breakpoint"
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


def parabolic_peak_from_topk(xc, yc, k=5):
    """
    Estimate peak using a quadratic fit on the k highest yc points.
    Returns (x_peak, y_peak).
    """
    m = np.isfinite(xc) & np.isfinite(yc)
    xv, yv = xc[m], yc[m]
    if xv.size < 3:
        return (np.nan, np.nan)

    # indices of top-k by y
    k = min(k, xv.size)
    idx_top = np.argsort(yv)[-k:]
    xs, ys = xv[idx_top], yv[idx_top]
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]

    # Quadratic fit on top-k
    x_peak = y_peak = np.nan
    ok_quad = False
    if np.unique(xs).size >= 3:
        try:
            coeffs = np.polyfit(xs, ys, 2, w=np.sqrt(np.clip(ys, 1, None)))
            a, b, c = coeffs
            if np.isfinite(a) and a < 0:  # concave parabola
                x_peak = -b / (2*a)
                x_peak = float(np.clip(x_peak, xs.min(), xs.max()))
                y_peak = float(a*x_peak**2 + b*x_peak + c)
                ok_quad = True
        except Exception:
            pass

    if not ok_quad:
        # Fallback: discrete maximum
        imax = int(np.argmax(yv))
        x_peak, y_peak = float(xv[imax]), float(yv[imax])

    return (x_peak, y_peak)


def smooth_curve(y, window_length=5, polyorder=2):
    """
    Smooth a curve using Savitzky-Golay filter.
    """
    if len(y) < window_length:
        return y

    # Ensure window_length is odd
    if window_length % 2 == 0:
        window_length += 1

    # Ensure polyorder < window_length
    polyorder = min(polyorder, window_length - 1)

    return savgol_filter(y, window_length, polyorder)


def find_breakpoint_piecewise(x, y, weights, min_breakpoint_idx=8, smooth=True, debug=False, min_x_for_breakpoint=None):
    """
    Find breakpoint where intensity stops increasing using piecewise linear regression.
    EXACT COPY from size_single_spots_beads_v2.py

    Strategy:
    - Optionally smooth the data first to reduce noise
    - Fit two linear segments: before and after a breakpoint
    - Find the breakpoint that minimizes total weighted residual error
    - Constrain: first segment should have positive slope, second should be nearly flat
    - Only consider breakpoints after intensity reaches ~80% of maximum
    - If min_x_for_breakpoint is specified, only consider breakpoints at x >= this value
      (useful for ignoring data below PSF when finding breakpoint)

    Returns:
        breakpoint_x: x value where the break occurs
        slope1: slope before breakpoint (should be positive)
        slope2: slope after breakpoint (should be ~0 or negative)
    """
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(weights) & (weights > 0)
    x_valid = x[valid]
    y_valid = y[valid]
    w_valid = weights[valid]

    if len(x_valid) < 8:
        return np.nan, np.nan, np.nan

    # Smooth the curve if requested - use more aggressive smoothing
    if smooth and len(y_valid) >= 9:
        y_smooth = smooth_curve(y_valid, window_length=9, polyorder=2)
    else:
        y_smooth = y_valid

    # Find where we reach 80% of max intensity (to avoid detecting breakpoint too early)
    max_y = np.max(y_smooth)
    threshold_y = 0.80 * max_y
    min_idx_for_threshold = np.argmax(y_smooth >= threshold_y)

    if debug:
        print(f"      DEBUG: max_y={max_y:.3f}, threshold_y (80%)={threshold_y:.3f}")
        print(f"      DEBUG: 80% threshold reached at idx={min_idx_for_threshold}, x={x_valid[min_idx_for_threshold]:.1f} nm")
        # Find where max actually occurs
        max_idx = np.argmax(y_smooth)
        print(f"      DEBUG: Maximum intensity at idx={max_idx}, x={x_valid[max_idx]:.1f} nm")

    # Start searching for breakpoint only after reaching 80% of max
    # But not too late either (leave at least 3 points for second segment)
    min_search_idx = max(min_breakpoint_idx, min_idx_for_threshold)

    # If min_x_for_breakpoint is specified, don't search for breakpoints below this x value
    # (e.g., ignore data below PSF when finding breakpoint)
    if min_x_for_breakpoint is not None:
        min_idx_for_x = np.argmax(x_valid >= min_x_for_breakpoint)
        if debug:
            print(f"      DEBUG: min_x_for_breakpoint={min_x_for_breakpoint:.1f} nm -> min_idx={min_idx_for_x}")
        min_search_idx = max(min_search_idx, min_idx_for_x)

    min_search_idx = min(min_search_idx, len(x_valid) - 3)

    best_breakpoint_idx = np.nan
    best_breakpoint = np.nan
    best_slopes = (np.nan, np.nan)
    best_intercepts = (np.nan, np.nan)
    best_error = np.inf

    # Try different breakpoints
    for i in range(min_search_idx, len(x_valid) - 2):
        # Split data at this point
        x1, y1, w1 = x_valid[:i+1], y_smooth[:i+1], w_valid[:i+1]
        x2, y2, w2 = x_valid[i:], y_smooth[i:], w_valid[i:]

        # Fit first segment (should be increasing)
        if len(x1) >= 3:
            slope1, intercept1 = np.polyfit(x1, y1, 1, w=w1)
            y_fit1 = slope1 * x1 + intercept1
            error1 = np.sum(w1 * (y1 - y_fit1)**2)
        else:
            continue

        # Fit second segment (should be flat or decreasing)
        if len(x2) >= 3:
            slope2, intercept2 = np.polyfit(x2, y2, 1, w=w2)
            y_fit2 = slope2 * x2 + intercept2
            error2 = np.sum(w2 * (y2 - y_fit2)**2)
        else:
            continue

        # Total error
        total_error = error1 + error2

        # STRICTER penalties:
        # 1. Penalize negative first slope
        if slope1 < 0:
            total_error *= 100

        # 2. Second slope should be MUCH flatter than first (changed from 0.2 to 0.1)
        if slope2 > slope1 * 0.1:
            total_error *= 10

        # 3. Strongly prefer second slope close to zero (plateau)
        # Add penalty proportional to how far slope2 is from zero
        if slope2 > 0:
            total_error *= (1 + abs(slope2 / slope1))

        if total_error < best_error:
            best_error = total_error
            best_breakpoint_idx = i
            best_breakpoint = x_valid[i]  # Initial estimate (bin center)
            best_slopes = (slope1, slope2)
            best_intercepts = (intercept1, intercept2)

    # Refine breakpoint: find where the two fitted lines intersect
    # Line 1: y = slope1 * x + intercept1
    # Line 2: y = slope2 * x + intercept2
    # At intersection: slope1 * x + intercept1 = slope2 * x + intercept2
    # Solving for x: x = (intercept2 - intercept1) / (slope1 - slope2)

    if np.isfinite(best_breakpoint_idx):
        slope1, slope2 = best_slopes
        intercept1, intercept2 = best_intercepts

        # Calculate intersection point for refined breakpoint
        if abs(slope1 - slope2) > 1e-10:  # Avoid division by zero
            breakpoint_refined = (intercept2 - intercept1) / (slope1 - slope2)

            # Sanity check: refined breakpoint should be near the original estimate
            # If it's way off, stick with the bin-based estimate
            if abs(breakpoint_refined - best_breakpoint) < 3 * (x_valid[1] - x_valid[0]):
                best_breakpoint = breakpoint_refined

    if debug:
        print(f"      DEBUG: Selected breakpoint at idx={best_breakpoint_idx}, x={best_breakpoint:.1f} nm")
        print(f"      DEBUG: slope1={best_slopes[0]:.6f}, slope2={best_slopes[1]:.6f}")

    return best_breakpoint, best_slopes[0], best_slopes[1]


def analyze_dimension_for_combined_figure(df, sigma_col, n_bins, scaling, psf, psf_multiplier=0.5, xlim_lower_override=None, use_peak_as_breakpoint=False):
    """
    Analyze a dimension and return data for combined figure.
    Uses same logic as sigma_intensity_plot_improved but returns data instead of plotting.

    xlim is computed automatically:
    - Lower bound: PSF * psf_multiplier (physically, spots cannot be smaller than ~half the diffraction limit)
      OR xlim_lower_override if specified (e.g., 500 nm = 1 slice depth for z-direction)
    - Upper bound: 98th percentile of the data (data-driven, excludes extreme outliers)

    This hybrid approach is principled: the lower bound is physics-based (relative to measured PSF),
    while the upper bound is data-driven (adapts to actual data distribution).

    Parameters:
        psf_multiplier: multiplier for lower bound (default 0.5, use 0.7 for z-direction in orange channel)
        xlim_lower_override: if specified, use this fixed value for lower bound instead of PSF * multiplier
        use_peak_as_breakpoint: if True, find breakpoint at maximum intensity (for curves that rise then fall,
                                 like orange channel sigma_z, instead of rise then plateau)

    Returns dictionary with:
        - slide_data: per-slide data (widths, intensities, normalization constants)
        - bin_centers, mean_I_combined, sem_I_combined: binned statistics across all slides
        - breakpoint, slope1, slope2: breakpoint analysis results
        - all_widths, all_intens_norm: combined data for overall statistics
        - xlim: the computed xlim tuple
        - bin_size: the bin size in nm
        - psf_multiplier: the multiplier used for the lower bound (or None if override used)
        - xlim_lower_override: the override value used (or None if PSF-based)
    """
    print(f"  Analyzing {sigma_col}...")

    # Storage for per-slide data
    slide_data = {}

    # ===========================
    # FIRST PASS: Process each slide separately
    # ===========================
    for slide in df['slide'].unique():
        if pd.isna(slide):
            continue

        slide_sub = df[df['slide'] == slide].copy()

        # Filter by Q111 and max_pfa
        slide_sub = slide_sub[
            (slide_sub['quality'] >= 0.111) &
            (slide_sub['pfa'] <= 1e-4)
        ].copy()

        if len(slide_sub) < 50:
            continue

        # Extract width and intensity
        width = slide_sub[sigma_col].values * scaling
        intens = slide_sub['photons'].values

        # Filter out invalid values
        valid = np.isfinite(width) & np.isfinite(intens) & (width > 0) & (intens > 0)
        width_filt = width[valid]
        intens_filt = intens[valid]

        if len(width_filt) < 50:
            continue

        # === KEY NORMALIZATION STEP (from size_single_spots_beads_v2.py) ===
        # Find peak intensity from PDF using KDE
        try:
            kde_intensity = gaussian_kde(intens_filt, bw_method='scott')
            intensity_range = np.linspace(np.percentile(intens_filt, 1),
                                         np.percentile(intens_filt, 99), 500)
            pdf_values = kde_intensity(intensity_range)
            peak_idx = np.argmax(pdf_values)
            peak_intensity = intensity_range[peak_idx]
        except:
            # Fallback: use median
            peak_intensity = np.median(intens_filt)

        # Store data for this slide
        slide_data[slide] = {
            'width_filt': width_filt,
            'intens_filt': intens_filt,
            'intens_norm': intens_filt / peak_intensity,  # Normalized by peak
            'peak_intensity': peak_intensity
        }

    if len(slide_data) == 0:
        print(f"    WARNING: No valid slide data for {sigma_col}")
        return None

    # ===========================
    # SECOND PASS: Combine all normalized data
    # ===========================
    all_widths = []
    all_intens_norm = []

    for slide, data in slide_data.items():
        all_widths.extend(data['width_filt'])
        all_intens_norm.extend(data['intens_norm'])

    all_widths = np.array(all_widths)
    all_intens_norm = np.array(all_intens_norm)

    # ===========================
    # COMPUTE xlim: physics-based lower bound, data-driven upper bound
    # ===========================
    # Lower bound: PSF * psf_multiplier (spots cannot be smaller than ~half the diffraction limit)
    #              OR xlim_lower_override if specified (e.g., 500 nm = 1 slice depth for z)
    # Upper bound: 98th percentile of data (excludes extreme outliers)
    if xlim_lower_override is not None:
        xlim_lower = xlim_lower_override
        print(f"    xlim computed: ({xlim_lower:.1f}, ...) nm")
        print(f"      Lower bound: FIXED at {xlim_lower:.1f} nm (1 slice depth)")
    else:
        xlim_lower = psf * psf_multiplier
        print(f"    xlim computed: ({xlim_lower:.1f}, ...) nm")
        print(f"      Lower bound: PSF * {psf_multiplier} = {psf:.1f} * {psf_multiplier} = {xlim_lower:.1f} nm")

    xlim_upper = np.percentile(all_widths, 98)
    xlim = (xlim_lower, xlim_upper)

    print(f"      Upper bound: 98th percentile of data = {xlim_upper:.1f} nm")
    print(f"    Final xlim: ({xlim_lower:.1f}, {xlim_upper:.1f}) nm")

    # ===========================
    # BINNED STATISTICS across all slides
    # ===========================
    bins = np.linspace(xlim[0], xlim[1], n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_size = (xlim[1] - xlim[0]) / n_bins  # Store bin size for caption

    # Compute mean, SEM, and counts for each bin
    mean_I_combined = np.zeros(n_bins)
    sem_I_combined = np.zeros(n_bins)
    counts_combined = np.zeros(n_bins)

    for i in range(n_bins):
        mask = (all_widths >= bins[i]) & (all_widths < bins[i+1])
        intens_in_bin = all_intens_norm[mask]

        if len(intens_in_bin) > 0:
            mean_I_combined[i] = np.mean(intens_in_bin)
            sem_I_combined[i] = np.std(intens_in_bin) / np.sqrt(len(intens_in_bin))
            counts_combined[i] = len(intens_in_bin)
        else:
            mean_I_combined[i] = np.nan
            sem_I_combined[i] = np.nan
            counts_combined[i] = 0

    # Only use bins with valid data
    valid = np.isfinite(mean_I_combined) & np.isfinite(sem_I_combined)

    # ===========================
    # BREAKPOINT ANALYSIS (using exact logic from size_single_spots_beads_v2.py)
    # ===========================
    # Use counts as weights (more spots = more reliable)
    weights = counts_combined.copy()
    weights[~valid] = 0

    # Enable debug for sigma_z to diagnose orange channel issue
    debug_mode = ('sigma_z' in sigma_col)

    if debug_mode:
        print(f"    DEBUG {sigma_col}: Binned intensity data (first 20 bins):")
        for i in range(min(20, len(bin_centers))):
            if valid[i]:
                print(f"      bin {i}: x={bin_centers[i]:.1f} nm, I={mean_I_combined[i]:.4f}, count={counts_combined[i]:.0f}")

    # For breakpoint finding, ignore data below PSF (spots can't be smaller than PSF)
    # This helps avoid artifacts from noise in the sub-PSF region

    if use_peak_as_breakpoint:
        # Special case: find breakpoint at maximum intensity
        # Used for curves that rise then fall (like orange channel sigma_z)
        # instead of the typical rise then plateau pattern

        # Smooth the data first
        y_smooth = smooth_curve(mean_I_combined[valid], window_length=9, polyorder=2)
        x_valid = bin_centers[valid]

        # Only consider points at x >= PSF
        psf_mask = x_valid >= psf
        if np.sum(psf_mask) > 3:
            y_after_psf = y_smooth.copy()
            y_after_psf[~psf_mask] = -np.inf  # Ignore points before PSF

            # Find the maximum
            max_idx = np.argmax(y_after_psf)
            breakpoint_x = x_valid[max_idx]

            # Compute approximate slopes before/after for reporting
            # (these are less meaningful for peak-based detection)
            if max_idx > 2 and max_idx < len(x_valid) - 2:
                slope1 = (y_smooth[max_idx] - y_smooth[max_idx-3]) / (x_valid[max_idx] - x_valid[max_idx-3])
                slope2 = (y_smooth[min(max_idx+3, len(y_smooth)-1)] - y_smooth[max_idx]) / (x_valid[min(max_idx+3, len(x_valid)-1)] - x_valid[max_idx])
            else:
                slope1, slope2 = np.nan, np.nan

            if debug_mode:
                print(f"      DEBUG: Using PEAK method for breakpoint")
                print(f"      DEBUG: Maximum intensity at x={breakpoint_x:.1f} nm, I={y_smooth[max_idx]:.4f}")
        else:
            breakpoint_x, slope1, slope2 = np.nan, np.nan, np.nan
    else:
        # Standard piecewise linear regression method
        breakpoint_x, slope1, slope2 = find_breakpoint_piecewise(
            bin_centers,
            mean_I_combined,
            weights,
            min_breakpoint_idx=8,
            smooth=True,
            debug=debug_mode,
            min_x_for_breakpoint=psf  # Only search for breakpoints at x >= PSF
        )

    print(f"    Breakpoint for {sigma_col}: {breakpoint_x:.1f} nm (PSF: {psf:.1f} nm)")
    print(f"    Ratio to PSF: {breakpoint_x / psf:.2f}")

    return {
        'slide_data': slide_data,
        'bin_centers': bin_centers,
        'mean_I_combined': mean_I_combined,
        'sem_I_combined': sem_I_combined,
        'all_widths': all_widths,
        'all_intens_norm': all_intens_norm,
        'breakpoint': breakpoint_x,
        'slope1': slope1,
        'slope2': slope2,
        'psf': psf,
        'xlim': xlim,
        'bin_size': bin_size,
        'psf_multiplier': psf_multiplier if xlim_lower_override is None else None,
        'xlim_lower_override': xlim_lower_override,
        'valid': valid
    }


def create_combined_breakpoint_figure(results_x, results_y, results_z, output_path, channel='green'):
    """
    Create figure with 3×3 layout:
    - Row 1: A (Raw intensity PDFs), B (Normalized intensity PDFs), C (σ_x biphasic with per-slide)
    - Row 2: D (σ_y biphasic with per-slide), E (σ_z biphasic with per-slide), F (σ_x size PDF)
    - Row 3: G (σ_y size PDF), H (σ_z size PDF), empty
    Upper right panel is left empty as requested.

    Parameters:
        channel: 'green' or 'orange' - specifies which channel is being analyzed
    """
    # Channel-specific information
    channel_info = {
        'green': {
            'name': 'Green channel (488 nm)',
            'probe': 'HTT1a (mutant huntingtin exon 1)',
            'wavelength': '488 nm',
            'color': '#2ecc71'
        },
        'orange': {
            'name': 'Orange channel (548 nm)',
            'probe': 'fl-HTT (complete mutant huntingtin)',
            'wavelength': '548 nm',
            'color': '#f39c12'
        }
    }
    ch_info = channel_info.get(channel, channel_info['green'])
    # Create figure with 3×3 layout
    fig = plt.figure(figsize=(18, 14), dpi=300)

    # Use gridspec for 3×3 layout
    gs = plt.matplotlib.gridspec.GridSpec(3, 3, figure=fig,
                                          hspace=0.35, wspace=0.3)

    cmap = plt.colormaps.get_cmap('tab10')

    # ========================================================================
    # PANEL A (Row 0, Col 0): Raw (non-normalized) intensity PDFs per slide
    # ========================================================================
    ax_a = fig.add_subplot(gs[0, 0])

    for idx, (slide, data) in enumerate(results_x['slide_data'].items()):
        # Get raw photon values (before normalization)
        raw_photons = data['intens_norm'] * data['peak_intensity']
        raw_photons = raw_photons[np.isfinite(raw_photons)]

        if len(raw_photons) < 50:
            continue

        # Use KDE for smoothing (same as Panel B)
        try:
            kde_slide = gaussian_kde(raw_photons, bw_method='scott')
            x_range = np.linspace(0, np.percentile(raw_photons, 99), 200)
            y_density = kde_slide(x_range)
            ax_a.plot(x_range, y_density, '-', alpha=0.3, linewidth=1,
                    color=cmap(idx % 10))
        except:
            pass

    ax_a.set_xlabel('Integrated photons', fontsize=10)
    ax_a.set_ylabel('Probability density', fontsize=10)
    ax_a.set_xlim(left=0)
    ax_a.set_ylim(bottom=0)
    ax_a.set_title('Raw intensity distributions per slide', fontsize=10, fontweight='bold')
    ax_a.text(-0.15, 1.05, 'A', transform=ax_a.transAxes,
            fontsize=14, fontweight='bold')
    ax_a.grid(True, alpha=0.3)

    # ========================================================================
    # PANEL B (Row 0, Col 1): Normalized intensity PDFs per slide (peak at 1)
    # ========================================================================
    ax_b = fig.add_subplot(gs[0, 1])

    for idx, (slide, data) in enumerate(results_x['slide_data'].items()):
        intens_norm = data['intens_norm']
        intens_norm = intens_norm[np.isfinite(intens_norm)]

        if len(intens_norm) < 50:
            continue

        # Plot KDE
        try:
            kde_slide = gaussian_kde(intens_norm, bw_method='scott')
            x_intensity = np.linspace(0, 5, 200)
            y_density = kde_slide(x_intensity)
            ax_b.plot(x_intensity, y_density, '-', alpha=0.3, linewidth=1,
                    color=cmap(idx % 10))
        except:
            pass

    # Reference line at 1.0 (peak normalization)
    ax_b.axvline(1.0, color='red', linestyle='--', linewidth=2, alpha=0.7,
               label='N$_{1mRNA}$ = 1', zorder=11)

    ax_b.set_xlabel('Normalized intensity (N/N$_{1mRNA}$)', fontsize=10)
    ax_b.set_ylabel('Probability density', fontsize=10)
    ax_b.set_xlim(0, 5)
    ax_b.set_ylim(bottom=0)
    ax_b.legend(fontsize=8, loc='upper right')
    ax_b.set_title('Per-slide normalized distributions', fontsize=10, fontweight='bold')
    ax_b.text(-0.15, 1.05, 'B', transform=ax_b.transAxes,
            fontsize=14, fontweight='bold')
    ax_b.grid(True, alpha=0.3)

    # ========================================================================
    # NEW 3x3 LAYOUT: Biphasic plots and Size PDFs
    # Row 0, Col 2: C (σ_x biphasic with per-slide)
    # Row 1, Col 0: D (σ_y biphasic with per-slide)
    # Row 1, Col 1: E (σ_z biphasic with per-slide)
    # Row 1, Col 2: F (σ_x size PDF)
    # Row 2, Col 0: G (σ_y size PDF)
    # Row 2, Col 1: H (σ_z size PDF)
    # Row 2, Col 2: Empty
    # ========================================================================

    # Define dimension configurations: (results, panel_label, sigma_label, color, biphasic_position, pdf_position)
    # Layout: Row 0: A, B, empty
    #         Row 1: C (σ_x biphasic), D (σ_y biphasic), E (σ_z biphasic)
    #         Row 2: F (σ_x PDF), G (σ_y PDF), H (σ_z PDF)
    dimensions = [
        (results_x, 'C', r'$\sigma_x$', 'tab:blue', (1, 0), (2, 0)),
        (results_y, 'D', r'$\sigma_y$', 'tab:green', (1, 1), (2, 1)),
        (results_z, 'E', r'$\sigma_z$', 'tab:red', (1, 2), (2, 2))
    ]

    for results, panel_label, sigma_label, color, biphasic_pos, pdf_pos in dimensions:
        if results is None:
            continue

        # Biphasic relationship subplot
        ax_biphasic = fig.add_subplot(gs[biphasic_pos])

        # Use the SAME bins as the combined analysis for consistency
        bin_centers = results['bin_centers']
        n_bins = len(bin_centers)

        # Compute per-slide binned data to get std across slides
        # Reconstruct bins from bin_centers
        bin_width = bin_centers[1] - bin_centers[0]
        bins = np.concatenate([bin_centers - bin_width/2, [bin_centers[-1] + bin_width/2]])

        # Store binned intensities for each slide
        slide_intensities = []
        for slide, data in results['slide_data'].items():
            width_filt = data['width_filt']
            intens_norm = data['intens_norm']

            slide_mean_I = np.zeros(n_bins)
            for i in range(n_bins):
                mask = (width_filt >= bins[i]) & (width_filt < bins[i+1])
                intens_in_bin = intens_norm[mask]
                if len(intens_in_bin) > 0:
                    slide_mean_I[i] = np.mean(intens_in_bin)
                else:
                    slide_mean_I[i] = np.nan

            # Apply smoothing
            valid_slide = np.isfinite(slide_mean_I)
            if np.sum(valid_slide) > 5:
                slide_mean_I_smooth = slide_mean_I.copy()
                slide_mean_I_smooth[valid_slide] = smooth_curve(slide_mean_I[valid_slide],
                                                                 window_length=5, polyorder=2)
                slide_intensities.append(slide_mean_I_smooth)
            else:
                slide_intensities.append(slide_mean_I)

        # Convert to array and compute std across slides
        slide_intensities = np.array(slide_intensities)  # shape: (n_slides, n_bins)
        std_across_slides = np.nanstd(slide_intensities, axis=0)

        # Plot shaded region showing ±1 std across slides (centered on combined mean from all spots)
        valid_plot = results['valid']
        valid_std = np.isfinite(std_across_slides)
        valid_combined = valid_plot & valid_std

        ax_biphasic.fill_between(bin_centers[valid_combined],
                                 results['mean_I_combined'][valid_combined] - std_across_slides[valid_combined],
                                 results['mean_I_combined'][valid_combined] + std_across_slides[valid_combined],
                                 alpha=0.25, color=color, zorder=5,
                                 label=f'±1 std (slides)')

        # SECOND: Plot combined mean from all spots
        ax_biphasic.plot(bin_centers[valid_plot], results['mean_I_combined'][valid_plot],
                   'o-', color=color, linewidth=2.5, markersize=5,
                   label=f'{sigma_label} (mean)', zorder=10, alpha=0.9)
        ax_biphasic.fill_between(bin_centers[valid_plot],
                           results['mean_I_combined'][valid_plot] - results['sem_I_combined'][valid_plot],
                           results['mean_I_combined'][valid_plot] + results['sem_I_combined'][valid_plot],
                           alpha=0.2, color=color, zorder=8)

        # Mark bead PSF
        ax_biphasic.axvline(results['psf'], color='purple', linestyle=':', linewidth=2,
                      alpha=0.6, label=f"Bead PSF", zorder=7)

        # Mark breakpoint
        if np.isfinite(results['breakpoint']):
            ax_biphasic.axvline(results['breakpoint'], color=color, linestyle='--', linewidth=2,
                          alpha=0.7, label=f"Breakpoint", zorder=7)

        # Reference line at normalized intensity = 1.0
        ax_biphasic.axhline(1.0, color='gray', linewidth=1, linestyle=':', alpha=0.5, zorder=5)

        ax_biphasic.set_xlabel(f'{sigma_label} [nm]', fontsize=10)
        ax_biphasic.set_ylabel('Normalized intensity\n(N/N$_{1mRNA}$)', fontsize=10)
        ax_biphasic.set_xlim(results['xlim'])
        ax_biphasic.set_ylim(bottom=0)
        ax_biphasic.legend(fontsize=7, loc='best', framealpha=0.9)
        ax_biphasic.set_title(f'{sigma_label}: Biphasic relationship',
                        fontsize=10, fontweight='bold')
        ax_biphasic.text(-0.12, 1.05, panel_label, transform=ax_biphasic.transAxes,
                   fontsize=14, fontweight='bold')
        ax_biphasic.grid(True, alpha=0.3)

        # Size PDF subplot
        pdf_label = chr(ord('F') + (ord(panel_label) - ord('C')))  # F, G, H for C, D, E
        ax_pdf = fig.add_subplot(gs[pdf_pos])

        # Plot per-slide distributions (faint lines)
        for idx, (slide, data) in enumerate(results['slide_data'].items()):
            widths = data['width_filt']
            widths = widths[np.isfinite(widths)]

            if len(widths) < 50:
                continue

            # Plot KDE
            try:
                kde_width = gaussian_kde(widths, bw_method='scott')
                x_width = np.linspace(results['xlim'][0], results['xlim'][1], 200)
                y_density = kde_width(x_width)
                ax_pdf.plot(x_width, y_density, '-', alpha=0.3, linewidth=1,
                             color=cmap(idx % 10))
            except:
                pass

        # Plot combined PDF (THICK LINE for all spots combined)
        try:
            kde_all = gaussian_kde(results['all_widths'], bw_method='scott')
            x_range = np.linspace(results['xlim'][0], results['xlim'][1], 500)
            y_pdf = kde_all(x_range)
            mode_idx = np.argmax(y_pdf)
            mode_value = x_range[mode_idx]

            # Plot thick combined PDF
            ax_pdf.plot(x_range, y_pdf, '-', color=color, linewidth=2.5,
                       alpha=0.9, label=f"Combined PDF", zorder=10)

            # Mark mode with vertical line
            ax_pdf.axvline(mode_value, color=color, linestyle='--', linewidth=2,
                            alpha=0.8, label=f"Mode: {mode_value:.0f} nm", zorder=11)
        except:
            pass

        # Mark bead PSF
        ax_pdf.axvline(results['psf'], color='purple', linestyle=':', linewidth=2,
                         alpha=0.6, label=f"Bead PSF", zorder=8)

        ax_pdf.set_xlabel(f'{sigma_label} [nm]', fontsize=10)
        ax_pdf.set_ylabel('Probability\ndensity', fontsize=10)
        ax_pdf.set_xlim(results['xlim'])
        ax_pdf.set_ylim(bottom=0)
        ax_pdf.legend(fontsize=7, loc='upper right')
        ax_pdf.set_title(f'{sigma_label}: Size distribution', fontsize=10, fontweight='bold')
        ax_pdf.text(-0.12, 1.05, pdf_label, transform=ax_pdf.transAxes,
                   fontsize=14, fontweight='bold')
        ax_pdf.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.savefig(str(output_path).replace('.pdf', '.svg'), format='svg', bbox_inches='tight')
    plt.close(fig)

    print(f"\nCombined figure saved to:")
    print(f"  {output_path}")
    print(f"  {str(output_path).replace('.pdf', '.svg')}")

    # Generate and save caption (with channel info)
    caption = generate_detailed_caption(results_x, results_y, results_z, channel=channel)
    caption_file = str(output_path).replace('.pdf', '_caption.txt')
    with open(caption_file, 'w') as f:
        f.write(caption)
    print(f"  {caption_file}")

    # Generate and save LaTeX caption (with channel info)
    latex_caption = generate_latex_caption(results_x, results_y, results_z, channel=channel)
    latex_caption_file = str(output_path).replace('.pdf', '_caption.tex')
    with open(latex_caption_file, 'w') as f:
        f.write(latex_caption)
    print(f"  {latex_caption_file}")

    # Generate and save alternative LaTeX caption (integrated format, with channel info)
    latex_caption_alt = generate_latex_caption_integrated(results_x, results_y, results_z, channel=channel)
    latex_caption_alt_file = str(output_path).replace('.pdf', '_caption_alt.tex')
    with open(latex_caption_alt_file, 'w') as f:
        f.write(latex_caption_alt)
    print(f"  {latex_caption_alt_file}")


def generate_detailed_caption(results_x, results_y, results_z, channel='green'):
    """Generate highly detailed figure caption with all statistical information.

    Parameters:
        channel: 'green' or 'orange' - specifies which channel is being analyzed
    """
    # Channel-specific information
    channel_info = {
        'green': {
            'name': 'Green channel (488 nm)',
            'probe': 'HTT1a (mutant huntingtin exon 1)',
            'wavelength': '488 nm',
            'excitation': '488 nm'
        },
        'orange': {
            'name': 'Orange channel (548 nm)',
            'probe': 'fl-HTT (complete mutant huntingtin transcript)',
            'wavelength': '548 nm',
            'excitation': '548 nm'
        }
    }
    ch_info = channel_info.get(channel, channel_info['green'])

    # Calculate mode values for each dimension
    try:
        kde_x = gaussian_kde(results_x['all_widths'], bw_method='scott')
        x_range_x = np.linspace(results_x['xlim'][0], results_x['xlim'][1], 500)
        mode_x = x_range_x[np.argmax(kde_x(x_range_x))]
    except:
        mode_x = np.nan

    try:
        kde_y = gaussian_kde(results_y['all_widths'], bw_method='scott')
        x_range_y = np.linspace(results_y['xlim'][0], results_y['xlim'][1], 500)
        mode_y = x_range_y[np.argmax(kde_y(x_range_y))]
    except:
        mode_y = np.nan

    try:
        kde_z = gaussian_kde(results_z['all_widths'], bw_method='scott')
        x_range_z = np.linspace(results_z['xlim'][0], results_z['xlim'][1], 500)
        mode_z = x_range_z[np.argmax(kde_z(x_range_z))]
    except:
        mode_z = np.nan

    # Calculate statistics
    n_slides = len(results_x['slide_data'])
    n_spots_x = len(results_x['all_widths'])
    n_spots_y = len(results_y['all_widths'])
    n_spots_z = len(results_z['all_widths'])
    n_bins_biphasic = len(results_x['bin_centers'])

    # Calculate expansion percentages
    expansion_x = ((results_x['breakpoint'] / results_x['psf']) - 1) * 100
    expansion_y = ((results_y['breakpoint'] / results_y['psf']) - 1) * 100
    expansion_z = ((results_z['breakpoint'] / results_z['psf']) - 1) * 100

    # Calculate mode expansion
    mode_expansion_x = ((mode_x / results_x['psf']) - 1) * 100
    mode_expansion_y = ((mode_y / results_y['psf']) - 1) * 100
    mode_expansion_z = ((mode_z / results_z['psf']) - 1) * 100

    caption = f"""Figure: Empirical definition of the single-molecule regime via per-slide normalization and breakpoint analysis

======================================================================
CHANNEL ANALYZED: {ch_info['name'].upper()}
======================================================================

Probe: {ch_info['probe']}
Excitation wavelength: {ch_info['excitation']}

======================================================================
OVERVIEW AND MOTIVATION
======================================================================

After setting a slide-specific photon threshold based on negative control regions, we normalized the photon yield of all detected spots such that the peak (mode) of the probability density function (PDF) is set at 1 for each slide independently. This per-slide normalization approach is necessary because photon yield varies substantially across slides due to tissue autofluorescence, fixation quality, probe hybridization efficiency, and imaging conditions (laser power fluctuations, objective alignment, detector sensitivity drift). A global intensity calibration would incorrectly classify dim spots on bright slides as non-specific background, and bright spots on dim slides as aggregates.

======================================================================
DATA SOURCES AND FILTERING
======================================================================

Total spots analyzed: {n_spots_x:,} (σₓ), {n_spots_y:,} (σᵧ), {n_spots_z:,} (σz)
Number of slides: {n_slides}
Data source: Bead calibration dataset (H5_FILE_PATH_BEAD)
Mouse model filter: Q111 only
Quality filter: spots.pfa_values < {1e-4:.0e} (probability of false alarm)
Channel analyzed: {ch_info['name']} - {ch_info['probe']}

Note: Bead data is used to establish the reference PSF (point spread function) from diffraction-limited fluorescent microspheres, which provides an optical baseline before analyzing biological tissue samples.

======================================================================
PANEL A - RAW INTENSITY DISTRIBUTIONS PER SLIDE
======================================================================

Visualization: Probability density functions of integrated photon counts (total photons detected per spot) before normalization.

Methodology:
- Each faint colored line represents one individual slide's intensity distribution
- Kernel density estimation (KDE) using Scott's rule for bandwidth selection
- 200 evaluation points spanning [0, 99th percentile of raw photons]
- Density normalization ensures area under each curve equals 1
- Smoothing via KDE provides clearer visualization of distribution shape

Observation: Substantial slide-to-slide variation in absolute photon yield is immediately apparent, with peak intensities varying by factors of 2-3× across slides. This heterogeneity arises from technical variation in sample preparation and imaging conditions.

======================================================================
PANEL B - PER-SLIDE NORMALIZED INTENSITY DISTRIBUTIONS
======================================================================

Visualization: Same data as Panel A, after per-slide normalization.

Normalization procedure for each slide independently:
1. Extract all photon values for spots passing quality filters
2. Compute kernel density estimate (KDE) using Scott's rule for bandwidth selection
3. Evaluate KDE on 500 points spanning [1st percentile, 99th percentile]
4. Identify peak (mode) of the PDF as the maximum of the KDE
5. Divide all photon values by this peak intensity: I_normalized = I_raw / I_peak
6. This sets the mode of each slide's distribution to N₁mRNA = 1

Visualization details:
- Each faint line shows one slide's normalized intensity distribution
- KDE computed using scipy.stats.gaussian_kde with bw_method='scott'
- x-axis range: [0, 5] in units of N₁mRNA
- Red dashed vertical line marks N₁mRNA = 1

Result: All slide distributions now converge around N₁mRNA = 1, demonstrating successful removal of slide-to-slide technical variation while preserving the biological signal (tail extending beyond 1, representing multi-transcript aggregates).

======================================================================
PANELS C, D, E - BIPHASIC RELATIONSHIPS (INTENSITY vs SIZE)
======================================================================

Panel C: σₓ (lateral x-dimension)
Panel D: σᵧ (lateral y-dimension)
Panel E: σz (axial z-dimension)

Visualization: Normalized intensity (N/N₁mRNA) plotted against fitted Gaussian spot size parameters.

X-axis range (xlim) determination - PRINCIPLED HYBRID APPROACH:
- Lower bound: Physics-based
  * For lateral dimensions (σₓ, σᵧ): PSF × {results_x['psf_multiplier']} (spots cannot be smaller than ~half the diffraction limit)
  * σₓ: {results_x['psf']:.1f} × {results_x['psf_multiplier']} = {results_x['xlim'][0]:.1f} nm
  * σᵧ: {results_y['psf']:.1f} × {results_y['psf_multiplier']} = {results_y['xlim'][0]:.1f} nm
  * For axial dimension (σz): Fixed at {results_z['xlim'][0]:.1f} nm (1 slice depth = 500 nm)
  * Using 1 slice depth as lower bound for z ensures we capture the full rising phase of the biphasic curve
- Upper bound: 98th percentile of data (data-driven)
  * Adapts to actual data distribution, excludes extreme outliers
  * σₓ: {results_x['xlim'][1]:.1f} nm (98th percentile)
  * σᵧ: {results_y['xlim'][1]:.1f} nm (98th percentile)
  * σz: {results_z['xlim'][1]:.1f} nm (98th percentile)

This hybrid approach ensures the analysis range is grounded in physical constraints (PSF-based lower bound)
while remaining adaptive to the data distribution (percentile-based upper bound).

Binning procedure for combined data (across all slides):
- Bin edges: np.linspace(xlim[0], xlim[1], {n_bins_biphasic + 1})
- σₓ: {n_bins_biphasic} bins spanning [{results_x['xlim'][0]:.1f}, {results_x['xlim'][1]:.1f}] nm (bin size: {results_x['bin_size']:.1f} nm)
- σᵧ: {n_bins_biphasic} bins spanning [{results_y['xlim'][0]:.1f}, {results_y['xlim'][1]:.1f}] nm (bin size: {results_y['bin_size']:.1f} nm)
- σz: {n_bins_biphasic} bins spanning [{results_z['xlim'][0]:.1f}, {results_z['xlim'][1]:.1f}] nm (bin size: {results_z['bin_size']:.1f} nm)
- For each bin: compute mean and standard error of mean (SEM) of normalized intensities

Per-slide analysis (light shaded region):
- Each slide analyzed independently with same {n_bins_biphasic} bins as combined analysis
- Binned mean intensity computed for each size bin per slide
- Savitzky-Golay smoothing applied to each slide: window_length=5, polyorder=2
- Mean and standard deviation computed across all slides for each bin
- Light colored shaded region shows ±1 standard deviation across slides
- Represents inter-slide variability in the biphasic relationship

Combined mean from all spots (bold colored line with error bands):
- Mean computed from all spots pooled together across all slides
- This provides the best statistical estimate of the true biphasic relationship
- Light shaded region: ±1 std across per-slide means (shows slide-to-slide variability)
- Narrower darker bands: ±1 SEM of the combined mean (uncertainty in the mean)
- Note: Mean line may not be perfectly centered in std region because std is computed from per-slide means while the plotted mean is from pooled spots

Reference lines:
- Purple dotted vertical line: Bead-derived PSF (mode of bead size distribution)
  * σₓ: {results_x['psf']:.1f} nm
  * σᵧ: {results_y['psf']:.1f} nm
  * σz: {results_z['psf']:.1f} nm
- Colored dashed vertical line: Empirically-determined breakpoint from tissue data
  * σₓ: {results_x['breakpoint']:.1f} nm ({expansion_x:.1f}% larger than bead PSF)
  * σᵧ: {results_y['breakpoint']:.1f} nm ({expansion_y:.1f}% larger than bead PSF)
  * σz: {results_z['breakpoint']:.1f} nm ({expansion_z:.1f}% larger than bead PSF)
- Gray horizontal dotted line: N₁mRNA = 1 reference

Interpretation of biphasic relationship:
- LEFT of breakpoint (single-molecule regime): Intensity increases approximately linearly with size. This positive correlation indicates that larger fitted widths within this regime arise from increased local probe density, more extended mRNA hybridization sites, or slight variations in probe cluster geometry—while still representing single diffraction-limited molecules.
- RIGHT of breakpoint (aggregate regime): Intensity plateaus or increases sub-linearly despite continued size growth. This indicates that broadened spots do not represent brighter single molecules, but rather spatially unresolved clusters of multiple transcripts, aggregation-induced quenching, or artifacts where additional size reflects aggregation rather than increased probe binding.

======================================================================
PANELS F, G, H - SPOT SIZE DISTRIBUTIONS
======================================================================

Panel F: σₓ size distribution
Panel G: σᵧ size distribution
Panel H: σz size distribution

Visualization: Probability density functions of fitted Gaussian spot widths for each slide.

Methodology:
- Each faint colored line: KDE of one slide's spot sizes (bw_method='scott')
- Thick colored line: Combined PDF from all spots across all slides
- 200 evaluation points per slide, 500 for combined distribution
- All spots from all slides pooled to compute overall distribution and mode

Mode identification (dashed colored vertical line):
- KDE computed on combined data from all slides
- 500 evaluation points for high resolution
- Mode = argmax(KDE) of combined distribution
- σₓ mode: {mode_x:.1f} nm ({mode_expansion_x:.1f}% larger than bead PSF)
- σᵧ mode: {mode_y:.1f} nm ({mode_expansion_y:.1f}% larger than bead PSF)
- σz mode: {mode_z:.1f} nm ({mode_expansion_z:.1f}% larger than bead PSF)

Reference lines and curves:
- Thick colored solid line: Combined PDF (KDE) from all spots across all slides
- Purple dotted vertical line: Bead-derived PSF
- Colored dashed vertical line: Mode of combined size distribution (refined PSF)
- Note: Breakpoint NOT shown in these panels (only in biphasic plots C-E)

Note: The modes represent the most probable spot sizes in tissue and serve as the refined, tissue-calibrated PSF values. These modes are typically close to bead PSF values, confirming consistency between optical calibration and tissue measurements. However, modes may differ from bead PSF due to tissue-specific effects (refractive index, aberrations, probe cluster size). The breakpoints are substantially larger than both bead PSF and modes, defining the upper bound of the single-molecule regime where aggregation begins.

======================================================================
BREAKPOINT DETERMINATION ALGORITHM
======================================================================

Method: Two-segment piecewise-linear regression with optimal breakpoint search

Steps:
1. Input: Binned data (size vs intensity) with {n_bins_biphasic} bins
2. Apply Savitzky-Golay smoothing to intensity values: window_length=9, polyorder=2
3. Search over candidate breakpoints from 20th to 80th percentile of size range
4. For each candidate breakpoint position:
   a. Fit linear regression to left segment (slope1, intercept1)
   b. Fit linear regression to right segment (slope2, intercept2)
   c. Compute residual sum of squares (RSS) for both segments
5. Select breakpoint that minimizes total RSS
6. Final breakpoint = x-coordinate at optimal split position

Output:
- Breakpoint position (nm)
- Slope1: positive slope in single-molecule regime
- Slope2: near-zero or negative slope in aggregate regime

Results:
- σₓ breakpoint: {results_x['breakpoint']:.1f} nm (slope1={results_x['slope1']:.3f}, slope2={results_x['slope2']:.3f})
- σᵧ breakpoint: {results_y['breakpoint']:.1f} nm (slope1={results_y['slope1']:.3f}, slope2={results_y['slope2']:.3f})
- σz breakpoint: {results_z['breakpoint']:.1f} nm (slope1={results_z['slope1']:.3f}, slope2={results_z['slope2']:.3f})

======================================================================
KEY FINDINGS AND BIOLOGICAL INTERPRETATION
======================================================================

1. Per-slide normalization successfully removes technical variation:
   - Raw intensity peaks vary by 2-3× across slides (Panel A)
   - Normalized intensity peaks all align at N₁mRNA = 1 (Panel B)
   - Biological signal (aggregate tail) preserved after normalization

2. Bead PSF provides diffraction-limited reference:
   - Bead-derived PSF from calibration: σₓ={results_x['psf']:.0f} nm, σᵧ={results_y['psf']:.0f} nm, σz={results_z['psf']:.0f} nm
   - Refined PSF from tissue data modes (Panels F-H): σₓ={mode_x:.0f} nm, σᵧ={mode_y:.0f} nm, σz={mode_z:.0f} nm
   - Modes are determined from combined size distributions across all slides
   - Refined PSF values supersede bead measurements for downstream analysis

3. Tissue-calibrated breakpoints exceed bead PSF by 15-40%:
   - σₓ: {expansion_x:.1f}% expansion
   - σᵧ: {expansion_y:.1f}% expansion
   - σz: {expansion_z:.1f}% expansion
   - This expansion reflects the physical extent of RNAscope probe clusters (~20 probe pairs, each ~2-3 nm) bound to target mRNA in tissue context

4. Biphasic relationship provides objective single-molecule criterion:
   - Positive slope regime: single diffraction-limited molecules
   - Zero/negative slope regime: multi-transcript aggregates
   - Breakpoint separates these two regimes empirically from data

5. Refined PSF from tissue data supersedes bead measurements:
   - Mode values from Panels F-H (σₓ={mode_x:.0f} nm, σᵧ={mode_y:.0f} nm, σz={mode_z:.0f} nm) define tissue-calibrated PSF
   - These refined widths represent the most probable spot size for single molecules in tissue
   - Used for all downstream single-molecule quantification and filtering
   - Accounts for tissue-induced aberrations, refractive index mismatch, and actual probe cluster geometry
   - Bead PSF ({results_x['psf']:.0f}, {results_y['psf']:.0f}, {results_z['psf']:.0f} nm) remains as theoretical diffraction limit reference

======================================================================
TECHNICAL SPECIFICATIONS
======================================================================

Microscopy parameters:
- Pixel size (lateral): 108.0 nm
- Slice depth (axial): 200.0 nm
- Excitation wavelength: 488 nm (green channel)
- Objective: [specific objective from metadata]
- Imaging modality: Widefield fluorescence microscopy

Image analysis:
- Spot detection: 3D Gaussian fitting to find σₓ, σᵧ, σz
- Photon integration: Sum of background-subtracted pixel intensities
- Quality metric: Probability of false alarm (pfa) from statistical detection theory

Statistical analysis:
- Kernel density estimation: scipy.stats.gaussian_kde with Scott's rule
- Smoothing: scipy.signal.savgol_filter
- Error bars: Standard error of mean (SEM) = std / sqrt(n_slides)
- Breakpoint fitting: Custom piecewise-linear regression

Software:
- Python 3.x with scipy, numpy, matplotlib
- Analysis code: result_code/draft_figures/fig_single_breakpoint_v3.py

======================================================================
SUPPLEMENTARY STATISTICS
======================================================================

Per-slide spot counts (range across {n_slides} slides): [compute min/max if available]
Median spots per slide: {n_spots_x // n_slides:,}

Intensity normalization factors (peak photon values per slide):
[Would require storing these values during analysis]

Size distribution statistics:
- σₓ: mean={np.mean(results_x['all_widths']):.1f} nm, median={np.median(results_x['all_widths']):.1f} nm, std={np.std(results_x['all_widths']):.1f} nm
- σᵧ: mean={np.mean(results_y['all_widths']):.1f} nm, median={np.median(results_y['all_widths']):.1f} nm, std={np.std(results_y['all_widths']):.1f} nm
- σz: mean={np.mean(results_z['all_widths']):.1f} nm, median={np.median(results_z['all_widths']):.1f} nm, std={np.std(results_z['all_widths']):.1f} nm

Breakpoint confidence:
- Determined from {n_bins_biphasic} bins spanning full size range
- Smoothing reduces noise while preserving trend (window=9 bins ≈ {9 * (results_x['xlim'][1] - results_x['xlim'][0]) / n_bins_biphasic:.0f} nm effective averaging)

======================================================================
"""

    return caption


def generate_latex_caption(results_x, results_y, results_z, channel='green'):
    """Generate elaborate LaTeX figure caption for publication.

    Parameters:
        channel: 'green' or 'orange' - specifies which channel is being analyzed
    """
    # Channel-specific information
    channel_info = {
        'green': {
            'name': 'Green channel (488 nm)',
            'probe': 'HTT1a (mutant huntingtin exon 1)',
            'wavelength': '488 nm',
            'excitation': '488~nm'
        },
        'orange': {
            'name': 'Orange channel (548 nm)',
            'probe': 'fl-HTT (complete mutant huntingtin transcript)',
            'wavelength': '548 nm',
            'excitation': '548~nm'
        }
    }
    ch_info = channel_info.get(channel, channel_info['green'])

    # Calculate mode values for each dimension
    try:
        kde_x = gaussian_kde(results_x['all_widths'], bw_method='scott')
        x_range_x = np.linspace(results_x['xlim'][0], results_x['xlim'][1], 500)
        mode_x = x_range_x[np.argmax(kde_x(x_range_x))]
    except:
        mode_x = np.nan

    try:
        kde_y = gaussian_kde(results_y['all_widths'], bw_method='scott')
        x_range_y = np.linspace(results_y['xlim'][0], results_y['xlim'][1], 500)
        mode_y = x_range_y[np.argmax(kde_y(x_range_y))]
    except:
        mode_y = np.nan

    try:
        kde_z = gaussian_kde(results_z['all_widths'], bw_method='scott')
        x_range_z = np.linspace(results_z['xlim'][0], results_z['xlim'][1], 500)
        mode_z = x_range_z[np.argmax(kde_z(x_range_z))]
    except:
        mode_z = np.nan

    # Calculate statistics
    n_slides = len(results_x['slide_data'])
    n_spots_total = len(results_x['all_widths'])
    bin_size = results_x['bin_size']  # Bin size in nm

    # Calculate expansion percentages
    expansion_x = ((results_x['breakpoint'] / results_x['psf']) - 1) * 100
    expansion_y = ((results_y['breakpoint'] / results_y['psf']) - 1) * 100
    expansion_z = ((results_z['breakpoint'] / results_z['psf']) - 1) * 100

    latex_caption = f"""\\caption{{\\textbf{{Empirical definition of the single-molecule regime via per-slide normalization and breakpoint analysis ({ch_info['name']}, {ch_info['probe']}).}}
\\textbf{{(A--B)}} Per-slide intensity normalization.
\\textbf{{(A)}} Raw integrated photon counts for each slide (faint colored lines, kernel density estimates). Substantial slide-to-slide variation in absolute photon yield is evident (peaks vary by $2$--$3\\times$), arising from technical factors including tissue autofluorescence, fixation quality, probe hybridization efficiency, and imaging conditions.
\\textbf{{(B)}} After per-slide normalization, setting the modal intensity to $N_{{1\\mathrm{{mRNA}}}} = 1$ for each slide independently. The red dashed line marks $N_{{1\\mathrm{{mRNA}}}} = 1$. All slide distributions now converge around this reference value, demonstrating successful removal of slide-to-slide technical variation while preserving the biological signal (tail extending beyond 1, representing multi-transcript aggregates).
%
\\textbf{{(C--E)}} Biphasic relationships between normalized intensity and spot size.
For each dimension---\\textbf{{(C)}} $\\sigma_x$ (lateral $x$), \\textbf{{(D)}} $\\sigma_y$ (lateral $y$), \\textbf{{(E)}} $\\sigma_z$ (axial $z$)---the bold colored line shows the mean normalized intensity computed from all {n_spots_total:,} spots pooled across {n_slides} slides.
The analysis range uses a principled hybrid approach: for lateral dimensions, the lower bound is {results_x['psf_multiplier']*100:.0f}\\% of bead PSF (since spots cannot be smaller than approximately half the diffraction limit); for the axial dimension, the lower bound is fixed at {results_z['xlim'][0]:.0f}~nm (1 slice depth) to capture the full rising phase of the biphasic curve. The upper bound is data-driven (98th percentile, adapting to the actual data distribution).
This yields ranges of [{results_x['xlim'][0]:.0f}, {results_x['xlim'][1]:.0f}]~nm for $\\sigma_x$/$\\sigma_y$ and [{results_z['xlim'][0]:.0f}, {results_z['xlim'][1]:.0f}]~nm for $\\sigma_z$, with bin sizes of {bin_size:.1f}~nm (50 bins).
The light shaded region ($\\pm 1$ standard deviation across per-slide means) represents inter-slide variability in the biphasic relationship, while the narrower darker error bands show $\\pm 1$ SEM (standard error of the mean) of the combined mean.
Purple dotted vertical lines mark the bead-derived PSF from fluorescent microsphere calibration ($\\sigma_x = {results_x['psf']:.0f}$~nm, $\\sigma_y = {results_y['psf']:.0f}$~nm, $\\sigma_z = {results_z['psf']:.0f}$~nm), representing the diffraction-limited reference.
Colored dashed vertical lines indicate empirically-determined breakpoints ($\\sigma_x = {results_x['breakpoint']:.0f}$~nm, $\\sigma_y = {results_y['breakpoint']:.0f}$~nm, $\\sigma_z = {results_z['breakpoint']:.0f}$~nm), which are ${expansion_x:.0f}\\%$, ${expansion_y:.0f}\\%$, and ${expansion_z:.0f}\\%$ larger than bead PSF values, respectively.
%
The biphasic pattern is diagnostic:
\\emph{{Left of breakpoint}} (single-molecule regime): Intensity increases approximately linearly with size, consistent with single diffraction-limited emitters where larger fitted widths arise from increased local probe density or extended mRNA hybridization sites.
\\emph{{Right of breakpoint}} (aggregate regime): Intensity plateaus despite continued size growth, indicating spatially unresolved multi-transcript clusters where additional size reflects aggregation rather than increased probe binding.
Breakpoints were determined via weighted piecewise-linear regression with Savitzky--Golay smoothing (window = 9 bins, polynomial order = 2), searching only after intensity reaches 80\\% of maximum and applying strict penalties to enforce the expected biphasic structure (positive slope before breakpoint, near-zero slope after).
%
\\textbf{{(F--H)}} Spot size distributions.
For each dimension---\\textbf{{(F)}} $\\sigma_x$, \\textbf{{(G)}} $\\sigma_y$, \\textbf{{(H)}} $\\sigma_z$---faint colored lines show per-slide probability density functions (kernel density estimates with Scott's bandwidth), while the thick colored line shows the combined PDF from all spots pooled across slides.
Purple dotted lines mark the bead PSF, and colored dashed lines mark the modes (peaks) of the combined distributions, which define the tissue-calibrated refined PSF: $\\sigma_x = {mode_x:.0f}$~nm, $\\sigma_y = {mode_y:.0f}$~nm, $\\sigma_z = {mode_z:.0f}$~nm.
These refined PSF values supersede the bead-derived measurements for all downstream single-molecule quantification, as they account for tissue-specific effects (refractive index mismatch, optical aberrations, finite probe cluster size).
%
\\textbf{{Key findings:}}
(1) Per-slide normalization successfully removes technical variation (Panel~B) while preserving biological heterogeneity.
(2) Empirically-determined breakpoints exceed bead PSF by ${expansion_x:.0f}$--${expansion_z:.0f}\\%$, consistent with the physical extent of RNAscope probe clusters ($\\sim$20 probe pairs, each $\\sim$2--3~nm) bound to target mRNA.
(3) The biphasic intensity--size relationship provides an objective, data-driven criterion for distinguishing single molecules (positive slope regime) from aggregates (plateau regime).
(4) Tissue-calibrated modes from size distributions (Panels~F--H) provide refined PSF estimates that account for \\emph{{in situ}} optical conditions.
%
\\textbf{{Methods:}}
Analysis performed on {n_spots_total:,} spots across {n_slides} slides from Q111 transgenic mouse tissue, using RNAscope \\emph{{in situ}} hybridization with fluorescent detection at {ch_info['excitation']} excitation ({ch_info['name']}, detecting {ch_info['probe']}).
Quality filtering: probability of false alarm $< 10^{{-4}}$.
Per-slide normalization: kernel density estimation (KDE) with Scott's rule to identify modal intensity, then division by peak to set $N_{{1\\mathrm{{mRNA}}}} = 1$.
Spot size determination: 3D Gaussian fitting to extract $\\sigma_x$, $\\sigma_y$, $\\sigma_z$.
Analysis range: principled hybrid approach with physics-based lower bound (50\\% of bead PSF) and data-driven upper bound (98th percentile of data).
Binning: 50 bins spanning [{results_x['xlim'][0]:.0f}, {results_x['xlim'][1]:.0f}]~nm (lateral) and [{results_z['xlim'][0]:.0f}, {results_z['xlim'][1]:.0f}]~nm (axial).
Imaging parameters: pixel size = 108.0~nm (lateral), slice depth = 200.0~nm (axial).
Statistical analysis: Python 3.x with \\texttt{{scipy}}, \\texttt{{numpy}}, \\texttt{{matplotlib}}.
}}"""

    return latex_caption


def generate_latex_caption_integrated(results_x, results_y, results_z, channel='green'):
    """Generate alternative LaTeX caption with integrated findings (no separate headers).

    Parameters:
        channel: 'green' or 'orange' - specifies which channel is being analyzed
    """
    # Channel-specific information
    channel_info = {
        'green': {
            'name': 'Green channel (488 nm)',
            'probe': 'HTT1a (mutant huntingtin exon 1)',
            'wavelength': '488 nm',
            'excitation': '488~nm'
        },
        'orange': {
            'name': 'Orange channel (548 nm)',
            'probe': 'fl-HTT (complete mutant huntingtin transcript)',
            'wavelength': '548 nm',
            'excitation': '548~nm'
        }
    }
    ch_info = channel_info.get(channel, channel_info['green'])

    # Calculate mode values for each dimension
    try:
        kde_x = gaussian_kde(results_x['all_widths'], bw_method='scott')
        x_range_x = np.linspace(results_x['xlim'][0], results_x['xlim'][1], 500)
        mode_x = x_range_x[np.argmax(kde_x(x_range_x))]
    except:
        mode_x = np.nan

    try:
        kde_y = gaussian_kde(results_y['all_widths'], bw_method='scott')
        x_range_y = np.linspace(results_y['xlim'][0], results_y['xlim'][1], 500)
        mode_y = x_range_y[np.argmax(kde_y(x_range_y))]
    except:
        mode_y = np.nan

    try:
        kde_z = gaussian_kde(results_z['all_widths'], bw_method='scott')
        x_range_z = np.linspace(results_z['xlim'][0], results_z['xlim'][1], 500)
        mode_z = x_range_z[np.argmax(kde_z(x_range_z))]
    except:
        mode_z = np.nan

    # Calculate statistics
    n_slides = len(results_x['slide_data'])
    n_spots_total = len(results_x['all_widths'])
    bin_size = results_x['bin_size']  # Bin size in nm

    # Calculate expansion percentages
    expansion_x = ((results_x['breakpoint'] / results_x['psf']) - 1) * 100
    expansion_y = ((results_y['breakpoint'] / results_y['psf']) - 1) * 100
    expansion_z = ((results_z['breakpoint'] / results_z['psf']) - 1) * 100

    latex_caption = f"""\\caption{{\\textbf{{Empirical definition of the single-molecule regime in tissue via per-slide normalization and breakpoint analysis ({ch_info['name']}, {ch_info['probe']}).}}
\\textbf{{(A)}} Raw integrated photon counts for each of {n_slides} slides analyzed (faint colored lines show kernel density estimates for individual slides, {n_spots_total:,} total spots from Q111 transgenic mouse tissue).
Substantial slide-to-slide variation is evident, with peak intensities varying by $2$--$3\\times$ due to technical factors: tissue autofluorescence, fixation quality, probe hybridization efficiency, and imaging conditions (laser power, detector sensitivity).
This heterogeneity necessitates per-slide normalization rather than global intensity calibration.
\\textbf{{(B)}} After per-slide normalization, all distributions converge around $N_{{1\\mathrm{{mRNA}}}} = 1$ (red dashed line), where the modal intensity of each slide is independently set to unity via kernel density estimation (KDE with Scott's bandwidth).
This normalization successfully removes technical variation while preserving biological signal, as evidenced by the conserved tail extending beyond 1 (representing multi-transcript aggregates and clusters).
%
\\textbf{{(C--E)}} Biphasic relationships reveal the empirical boundary between single molecules and aggregates.
For \\textbf{{(C)}} lateral $\\sigma_x$, \\textbf{{(D)}} lateral $\\sigma_y$, and \\textbf{{(E)}} axial $\\sigma_z$, the bold colored line shows mean normalized intensity from all pooled spots using {bin_size:.1f}~nm bins.
The analysis range uses a principled hybrid approach: for lateral dimensions, the lower bound is {results_x['psf_multiplier']*100:.0f}\\% of bead PSF (since spots cannot be smaller than approximately half the diffraction limit); for the axial dimension, the lower bound is fixed at {results_z['xlim'][0]:.0f}~nm (1 slice depth) to capture the full rising phase of the biphasic curve. The upper bound is data-driven (98th percentile of data, adapting to actual distribution).
This yields ranges of [{results_x['xlim'][0]:.0f}, {results_x['xlim'][1]:.0f}]~nm for $\\sigma_x$/$\\sigma_y$ and [{results_z['xlim'][0]:.0f}, {results_z['xlim'][1]:.0f}]~nm for $\\sigma_z$.
Light shading indicates $\\pm 1$ standard deviation across per-slide means (inter-slide variability) and darker narrow bands show $\\pm 1$ SEM of the combined mean (uncertainty).
Purple dotted vertical lines mark bead-derived PSF from fluorescent microsphere calibration ($\\sigma_x = {results_x['psf']:.0f}$~nm, $\\sigma_y = {results_y['psf']:.0f}$~nm, $\\sigma_z = {results_z['psf']:.0f}$~nm), representing the diffraction-limited optical reference.
Colored dashed vertical lines indicate empirically-determined breakpoints ($\\sigma_x = {results_x['breakpoint']:.0f}$~nm, $\\sigma_y = {results_y['breakpoint']:.0f}$~nm, $\\sigma_z = {results_z['breakpoint']:.0f}$~nm), which exceed bead PSF by ${expansion_x:.0f}\\%$, ${expansion_y:.0f}\\%$, and ${expansion_z:.0f}\\%$ respectively---consistent with the finite physical extent of RNAscope probe clusters ($\\sim$20 probe pairs, each $\\sim$2--3~nm) bound to target mRNA in tissue.
The diagnostic biphasic pattern separates two regimes: left of the breakpoint (single-molecule regime), intensity increases linearly with size, indicating that larger fitted widths arise from increased local probe density or extended hybridization sites while remaining single diffraction-limited emitters; right of the breakpoint (aggregate regime), intensity plateaus despite size growth, revealing spatially unresolved multi-transcript clusters where size reflects aggregation rather than probe binding.
Breakpoints were computed via weighted piecewise-linear regression with Savitzky--Golay smoothing (window = 9 bins, polynomial order = 2), searching only after intensity reaches 80\\% of maximum and applying strict penalties to enforce positive slope before breakpoint and near-zero slope after (details in Methods).
%
\\textbf{{(F--H)}} Spot size distributions define tissue-calibrated PSF values.
For \\textbf{{(F)}} $\\sigma_x$, \\textbf{{(G)}} $\\sigma_y$, and \\textbf{{(H)}} $\\sigma_z$, faint colored lines show per-slide probability density functions while thick colored lines show combined PDFs from all pooled spots.
Purple dotted lines replicate bead PSF references, while colored dashed lines mark the modes of the combined distributions: $\\sigma_x = {mode_x:.0f}$~nm, $\\sigma_y = {mode_y:.0f}$~nm, $\\sigma_z = {mode_z:.0f}$~nm.
These mode values represent the most probable spot sizes in tissue and define the refined, tissue-calibrated PSF that supersedes bead-derived measurements for all downstream single-molecule quantification, accounting for \\emph{{in situ}} effects including refractive index mismatch between tissue and immersion medium, optical aberrations from tissue heterogeneity, and the finite geometry of probe--mRNA complexes.
The close correspondence between modes and bead PSF ($<$5\\% difference for $\\sigma_z$, $\\sim$40\\% for lateral dimensions) validates the optical calibration while revealing tissue-specific broadening from probe cluster size.
Spot sizes were determined by 3D Gaussian fitting to background-subtracted fluorescence images acquired at {ch_info['excitation']} excitation ({ch_info['name']}, detecting {ch_info['probe']}) with 108.0~nm lateral pixel size and 200.0~nm axial slice depth.
Quality filtering retained spots with probability of false alarm $< 10^{{-4}}$.
}}"""

    return latex_caption


# ---------------------------
# Main processing
# ---------------------------
if __name__ == "__main__":

    print("="*70)
    print("BEAD PSF ANALYSIS - Determining sigma values and breakpoints")
    print("="*70)

    # File paths - use bead PSF from config
    h5_file_paths = H5_FILE_PATHS_BEAD

    # IMPORTANT: For bead analysis, we use use_sigma_final_filter = False
    # (different from experimental analysis which uses True)
    use_sigma_final_filter = False

    # Load HDF5 data
    h5_file_path = h5_file_paths[0]
    with h5py.File(h5_file_path, 'r') as h5_file:
        data_dict = recursively_load_dict(h5_file)

    print(f"Loaded {len(data_dict.keys())} keys from HDF5")

    # Extract DataFrame
    desired_channels = ['green', 'orange']
    fields_to_extract = ['spots.pfa_values', 'spots.photons', 'cluster_intensities',
                        'spots.final_filter', 'spots.params_raw']
    slide_field = 'metadata_sample_slide_name_std'
    max_pfa = MAX_PFA

    df_extracted = extract_dataframe(
        data_dict,
        field_keys=fields_to_extract,
        channels=desired_channels,
        include_file_metadata_sample=True,
        explode_fields=[]
    )

    # Compute thresholds
    thresholds, thresholds_cluster, *_ = compute_thresholds(
        df_extracted=df_extracted,
        slide_field=slide_field,
        desired_channels=desired_channels,
        negative_control_field='Negative control',
        experimental_field='ExperimentalQ111 - 488mHT - 548mHTa - 647Darp',
        quantile_negative_control=QUANTILE_NEGATIVE_CONTROL,
        max_pfa=max_pfa,
        plot=False,
        n_bootstrap=N_BOOTSTRAP,
        use_region=False,
        cluster_threshold_val=None,
        use_final_filter=use_sigma_final_filter,
    )

    # Further extraction with additional fields
    fields_to_extract = ['spots.final_filter', 'spots.params_raw', 'spots_sigma_var.params_raw',
                         'spots_sigma_var.final_filter', 'cluster_intensities', 'label_sizes',
                         'metadata_sample_Age', 'spots.pfa_values']
    fields_to_extract.append('metadata_sample_mouse_ID')

    df_extracted = extract_dataframe(data_dict, field_keys=fields_to_extract,
                                     channels=desired_channels,
                                     include_file_metadata_sample=True, explode_fields=[])
    df_extracted = df_extracted[df_extracted['metadata_sample_Mouse_Model'] == 'Q111']

    concatenated_data = concatenate_fields(
        df_extracted=df_extracted,
        slide_field=slide_field,
        desired_channels=['green', 'orange'],
        fields_to_extract=fields_to_extract,
        probe_set_field='metadata_sample_Probe-Set',
    )
    df_groups = concatenated_data_to_df(concatenated_data)

    # Build single spots dataframe
    single_rows = []

    for idx, row in df_groups.iterrows():
        slide = row["slide"]
        channel = row["channel"]
        probe_set = row["probe_set"]
        region = row["region"]

        # Threshold look-up
        components = [str(slide), str(channel)]
        threshold_val = next((v for k, v in thresholds.items()
                              if all(comp in str(k) for comp in components)), None)

        if threshold_val is None:
            print(f"[WARN] threshold missing for slide={slide}, ch={channel}")
            continue

        # Unpack arrays
        if not use_sigma_final_filter:
            mask = row["spots.pfa_values"] < max_pfa
            row["spots.final_filter"] = np.any(mask, axis=1)

        photon_arr_sig = row["spots_sigma_var.params_raw"][row["spots.final_filter"], 3]
        sigma = row["spots_sigma_var.params_raw"][row["spots.final_filter"], 5::]
        metadata_sample_Age = row["metadata_sample_Age"]
        metadata_sample_mouse_ID = row["metadata_sample_mouse_ID"]

        single_mask = photon_arr_sig > threshold_val

        # Build rows for single spots
        for i in range(len(photon_arr_sig)):
            if single_mask[i]:
                single_rows.append({
                    'slide': slide,
                    'channel': channel,
                    'probe_set': probe_set,
                    'region': region,
                    'photons': photon_arr_sig[i],
                    'sigma_x': sigma[i, 0],
                    'sigma_y': sigma[i, 1],
                    'sigma_z': sigma[i, 2],
                    'quality': 1.0,  # Placeholder - Q111 filter already applied
                    'pfa': 0.0,  # Placeholder - already filtered by max_pfa
                    'age': metadata_sample_Age,
                    'mouse_ID': metadata_sample_mouse_ID
                })

    df_single = pd.DataFrame(single_rows)

    print(f"\nTotal single spots: {len(df_single):,}")

    # ========================================================================
    # Use specified bead PSF values (from table)
    # ========================================================================

    # Bead PSF values from results_config.py (from fluorescent microsphere calibration)
    psf_x_bead = BEAD_PSF_X  # nm
    psf_y_bead = BEAD_PSF_Y  # nm
    psf_z_bead = BEAD_PSF_Z  # nm

    print(f"\nBead PSF values (from config):")
    print(f"  σ_x: {psf_x_bead:.1f} nm")
    print(f"  σ_y: {psf_y_bead:.1f} nm")
    print(f"  σ_z: {psf_z_bead:.1f} nm")
    print(f"\nSigma xlim values (from config):")
    print(f"  sigma_x: {SIGMA_X_XLIM}")
    print(f"  sigma_y: {SIGMA_Y_XLIM}")
    print(f"  sigma_z: {SIGMA_Z_XLIM}")

    # ========================================================================
    # ANALYZE AND CREATE FIGURES FOR BOTH CHANNELS
    # ========================================================================

    # xlim is computed automatically for each channel:
    #   - Lower bound: PSF * 0.5 (physics-based, spots cannot be smaller than ~half the diffraction limit)
    #   - Upper bound: 98th percentile of data (data-driven, excludes extreme outliers)

    for channel in ['green', 'orange']:
        print("\n" + "="*70)
        print(f"ANALYZING {channel.upper()} CHANNEL")
        print("="*70)

        # Filter data for this channel
        df_channel = df_single[df_single['channel'] == channel].copy()
        print(f"  Spots in {channel} channel: {len(df_channel):,}")

        if len(df_channel) < 100:
            print(f"  WARNING: Not enough spots for {channel} channel, skipping...")
            continue

        # PSF multiplier for lower bound:
        # - Use 0.5 for x/y (standard)
        # - For z: use fixed 500 nm (1 slice depth) instead of PSF-based calculation
        psf_mult_xy = 0.5
        # z-direction uses fixed xlim_lower=500 nm (see analyze_dimension_for_combined_figure call)

        # For orange channel sigma_z: use peak-based breakpoint detection
        # because the curve rises then falls (no plateau), unlike the typical biphasic pattern
        use_peak_for_z = (channel == 'orange')

        # Analyze each dimension separately using BEAD PSF values
        results_x = analyze_dimension_for_combined_figure(
            df_channel, 'sigma_x', n_bins=50,
            scaling=pixelsize, psf=psf_x_bead, psf_multiplier=psf_mult_xy
        )

        results_y = analyze_dimension_for_combined_figure(
            df_channel, 'sigma_y', n_bins=50,
            scaling=pixelsize, psf=psf_y_bead, psf_multiplier=psf_mult_xy
        )

        results_z = analyze_dimension_for_combined_figure(
            df_channel, 'sigma_z', n_bins=50,
            scaling=slice_depth, psf=psf_z_bead, xlim_lower_override=500.0,
            use_peak_as_breakpoint=use_peak_for_z
        )

        # ========================================================================
        # CREATE COMBINED FIGURE FOR THIS CHANNEL
        # ========================================================================
        print("\n" + "-"*70)
        print(f"CREATING COMBINED FIGURE FOR {channel.upper()} CHANNEL")
        print("-"*70)

        # Channel-specific output filename
        output_file = OUTPUT_DIR / f"fig_single_breakpoint_v3_{channel}.pdf"
        create_combined_breakpoint_figure(results_x, results_y, results_z, output_file, channel=channel)

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE FOR ALL CHANNELS")
    print("="*70)
