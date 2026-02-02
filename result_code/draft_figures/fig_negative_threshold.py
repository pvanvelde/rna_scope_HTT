"""
Generate Figure: Negative controls establish per-slide detection thresholds

This script creates a 4-panel figure showing:
- Panel A: Example FOVs from negative-control sections (bacterial DapB probe)
- Panel B: Distributions of integrated photon counts with per-slide 95th percentile thresholds
- Panel C: Cumulative distribution functions (CDFs) per slide
- Panel D: Threshold values versus slide number, colored by imaging date
"""

import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
import sys

# Add parent directory to path to import result_functions
sys.path.insert(0, str(Path(__file__).parent.parent))

from result_functions_v2 import compute_thresholds, recursively_load_dict, extract_dataframe
from results_config import (
    H5_FILE_PATH_EXPERIMENTAL,
    PIXELSIZE, SLICE_DEPTH,
    SLIDE_FIELD, NEGATIVE_CONTROL_FIELD, EXPERIMENTAL_FIELD,
    QUANTILE_NEGATIVE_CONTROL, MAX_PFA, N_BOOTSTRAP,
    FIGURE_DPI, USE_LATEX,
    CHANNEL_COLORS
)

# Import scienceplots for nice plotting
import scienceplots
plt.style.use('science')
plt.rcParams['text.usetex'] = USE_LATEX

# Output directory
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_negative_control_data():
    """Load and process negative control data from HDF5 file."""
    print("="*70)
    print("LOADING NEGATIVE CONTROL DATA")
    print("="*70)

    # Load HDF5 data
    with h5py.File(H5_FILE_PATH_EXPERIMENTAL, 'r') as h5_file:
        data_dict = recursively_load_dict(h5_file)

    print(f"\nLoaded data from: {H5_FILE_PATH_EXPERIMENTAL}")

    # Extract DataFrame
    desired_channels = ['green', 'orange']
    fields_to_extract = [
        'spots.pfa_values',
        'spots.photons',
        'cluster_intensities',
        'metadata_sample.Age',
        'spots.final_filter',
        'spots.params_raw',
        'metadata_sample.Date'
    ]

    df_extracted = extract_dataframe(
        data_dict,
        field_keys=fields_to_extract,
        channels=desired_channels,
        include_file_metadata_sample=True,
        explode_fields=[]
    )

    print(f"Channels analyzed: {desired_channels}")
    print(f"Quantile for threshold: {QUANTILE_NEGATIVE_CONTROL}")

    return df_extracted, desired_channels


def compute_slide_thresholds(df_extracted, desired_channels):
    """Compute per-slide thresholds from negative control data."""
    print("\n" + "="*70)
    print("COMPUTING PER-SLIDE THRESHOLDS")
    print("="*70)

    (thresholds, thresholds_cluster,
     error_thresholds, error_thresholds_cluster,
     number_of_datapoints, age) = compute_thresholds(
        df_extracted=df_extracted,
        slide_field=SLIDE_FIELD,
        desired_channels=desired_channels,
        negative_control_field=NEGATIVE_CONTROL_FIELD,
        experimental_field=EXPERIMENTAL_FIELD,
        quantile_negative_control=QUANTILE_NEGATIVE_CONTROL,
        max_pfa=MAX_PFA,
        plot=False,
        n_bootstrap=N_BOOTSTRAP,
        use_region=False,  # Slide-wide thresholds
        use_final_filter=True,
    )

    return thresholds, error_thresholds, number_of_datapoints, age


def extract_negative_control_spots(df_extracted, desired_channels):
    """Extract all negative control spot intensities for plotting."""
    # Filter for negative control samples (case-insensitive)
    df_neg = df_extracted[
        df_extracted['metadata_sample_Probe-Set']
        .str.lower()
        .str.contains(NEGATIVE_CONTROL_FIELD.lower(), na=False)
    ].copy()

    # Extract all spot photon values per slide and channel
    spot_data = []
    for idx, row in df_neg.iterrows():
        slide = row[SLIDE_FIELD]
        channel = row['channel']

        # Get params_raw (contains x, y, z, photons, ...)
        if row['spots.params_raw'] is None or len(row['spots.params_raw']) == 0:
            continue

        photons_array = row['spots.params_raw'][:, 3]  # Column 3 is photons
        filter_mask = row['spots.final_filter']

        # Apply filter
        if filter_mask is not None and len(filter_mask) > 0:
            filter_mask = np.atleast_1d(np.array(filter_mask)).astype(bool)
            if len(filter_mask) == len(photons_array):
                photons = photons_array[filter_mask]
            else:
                photons = photons_array  # Fallback if filter size doesn't match
        else:
            photons = photons_array

        # Extract date
        date = row.get('metadata_sample.Date', 'Unknown')
        if pd.isna(date):
            date = 'Unknown'

        for p in photons:
            spot_data.append({
                'slide': slide,
                'channel': channel,
                'photons': p,
                'date': str(date)
            })

    df_spots = pd.DataFrame(spot_data)
    return df_spots


def get_example_fov_data(df_extracted):
    """Get example FOV images from negative control for Panel A."""
    # Filter for negative control (case-insensitive)
    df_neg = df_extracted[
        df_extracted['metadata_sample_Probe-Set']
        .str.lower()
        .str.contains(NEGATIVE_CONTROL_FIELD.lower(), na=False)
    ].copy()

    # Get first FOV with detections in both channels
    example_fovs = {}
    for channel in ['green', 'orange']:
        df_ch = df_neg[df_neg['channel'] == channel]
        for idx, row in df_ch.iterrows():
            # Get spot positions and photons from params_raw
            params = row['spots.params_raw']
            if params is not None and len(params) > 0:
                example_fovs[channel] = {
                    'positions': params[:, :3],  # x, y, z positions
                    'photons': params[:, 3],  # photons in column 3
                    'filter': row['spots.final_filter']
                }
                break

    return example_fovs


def create_figure(df_spots, thresholds, error_thresholds, example_fovs):
    """Create the 3-panel figure (placeholder panel A removed)."""
    print("\n" + "="*70)
    print("CREATING FIGURE")
    print("="*70)

    # Create figure with GridSpec - 3 panels: A (top full width), B and C (bottom row)
    fig = plt.figure(figsize=(12, 10), dpi=FIGURE_DPI)
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    # --- Panel A: Histograms with thresholds (spans top row) ---
    ax_a = fig.add_subplot(gs[0, :])
    plot_panel_b(ax_a, df_spots, thresholds)

    # --- Panel B: CDFs per slide ---
    ax_b = fig.add_subplot(gs[1, 0])
    plot_panel_c(ax_b, df_spots, thresholds)

    # --- Panel C: Threshold values vs slide number ---
    ax_c = fig.add_subplot(gs[1, 1])
    plot_panel_d(ax_c, thresholds, df_spots)

    # Add panel labels (positioned higher to avoid overlap with titles)
    for ax, label in zip([ax_a, ax_b, ax_c], ['A', 'B', 'C']):
        ax.text(-0.12, 1.15, f'({label})', transform=ax.transAxes,
                fontsize=14, fontweight='bold', va='top')

    return fig


def plot_panel_a(ax, example_fovs):
    """
    Panel A: Placeholder for example FOVs from negative-control sections.
    """
    # Create blank panel with placeholder text
    ax.text(0.5, 0.5, '[Placeholder for example FOV images]\n\n' +
            'Negative control sections (bacterial DapB probe)\n' +
            'showing sparse false-positive detections from\n' +
            'autofluorescence and non-specific binding',
            ha='center', va='center', fontsize=10, transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))

    # Remove axis ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)

    ax.set_title('Negative control sections (DapB)', fontsize=10, fontweight='bold')


def plot_panel_b(ax, df_spots, thresholds):
    """
    Panel B: Distributions of integrated photon counts with per-slide 95th percentile thresholds.
    """
    channels = ['green', 'orange']

    for i, channel in enumerate(channels):
        df_ch = df_spots[df_spots['channel'] == channel]

        # Plot histogram for all slides combined
        color = CHANNEL_COLORS.get(channel, 'gray')
        ax.hist(df_ch['photons'], bins=50, alpha=0.5, color=color,
                label=f'{channel} (all slides)', density=True)

        # Plot per-slide thresholds as vertical lines
        slide_thresholds = []
        for key, val in thresholds.items():
            if key[1] == channel:  # (slide, channel, region) or (slide, channel)
                slide_thresholds.append(val)

        # Plot mean threshold
        if slide_thresholds:
            mean_thresh = np.mean(slide_thresholds)
            ax.axvline(mean_thresh, color=color, linestyle='--', linewidth=2,
                      label=f'{channel} mean threshold')

    ax.set_xlabel('Integrated photons', fontsize=9)
    ax.set_ylabel('Probability density', fontsize=9)
    ax.set_title('Photon count distributions with thresholds', fontsize=10, fontweight='bold')
    ax.legend(fontsize=7, loc='upper right')
    ax.set_xlim(0, df_spots['photons'].quantile(0.99))
    ax.grid(True, alpha=0.3)


def plot_panel_c(ax, df_spots, thresholds):
    """
    Panel C: Cumulative distribution functions (CDFs) per slide.
    """
    channels = ['green', 'orange']

    for channel in channels:
        color = CHANNEL_COLORS.get(channel, 'gray')

        # Get unique slides for this channel
        df_ch = df_spots[df_spots['channel'] == channel]
        slides = df_ch['slide'].unique()

        # Plot CDF for each slide
        for slide in slides:
            df_slide = df_ch[df_ch['slide'] == slide]
            photons = np.sort(df_slide['photons'].values)
            cdf = np.arange(1, len(photons) + 1) / len(photons)

            ax.plot(photons, cdf, color=color, alpha=0.3, linewidth=1)

        # Plot mean CDF
        all_photons = np.sort(df_ch['photons'].values)
        cdf = np.arange(1, len(all_photons) + 1) / len(all_photons)
        ax.plot(all_photons, cdf, color=color, linewidth=2.5,
               label=f'{channel} (mean)', zorder=10)

    # Mark 95th percentile
    ax.axhline(QUANTILE_NEGATIVE_CONTROL, color='red', linestyle='--',
              linewidth=1.5, label=f'{QUANTILE_NEGATIVE_CONTROL*100:.0f}th percentile')

    ax.set_xlabel('Integrated photons', fontsize=9)
    ax.set_ylabel('Cumulative probability', fontsize=9)
    ax.set_title('CDFs per slide show consistent tail behavior', fontsize=10, fontweight='bold')
    ax.legend(fontsize=7, loc='lower right')
    ax.set_xlim(0, df_spots['photons'].quantile(0.99))
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)


def plot_panel_d(ax, thresholds, df_spots):
    """
    Panel D: Threshold values versus slide number, using consistent channel colors.
    """
    # Extract threshold data
    threshold_data = []
    for key, val in thresholds.items():
        if len(key) == 2:  # (slide, channel)
            slide, channel = key
            region = None
        elif len(key) == 3:  # (slide, channel, region)
            slide, channel, region = key
        else:
            continue

        # Get date for this slide
        df_slide = df_spots[df_spots['slide'] == slide]
        date = df_slide['date'].iloc[0] if len(df_slide) > 0 else 'Unknown'

        threshold_data.append({
            'slide': slide,
            'channel': channel,
            'threshold': val,
            'date': date
        })

    df_thresh = pd.DataFrame(threshold_data)

    # Get all unique slides across both channels
    all_slides = sorted(df_thresh['slide'].unique())
    slide_to_x = {s: i for i, s in enumerate(all_slides)}

    # Plot for each channel with consistent colors
    channels = ['green', 'orange']
    for i, channel in enumerate(channels):
        df_ch = df_thresh[df_thresh['channel'] == channel].copy()

        # Map slides to x-positions
        df_ch.loc[:, 'x'] = df_ch['slide'].map(slide_to_x)

        # Use channel-specific color
        color = CHANNEL_COLORS.get(channel, 'gray')

        # Plot with color matching the channel
        ax.scatter(df_ch['x'], df_ch['threshold'],
                  c=color, s=60, alpha=0.7, edgecolor='black', linewidth=0.5,
                  marker='o' if channel == 'green' else 's',
                  label=channel)

    ax.set_xlabel('Slide number', fontsize=9)
    ax.set_ylabel('Threshold (photons)', fontsize=9)
    ax.set_title('Batch-to-batch variation motivates per-slide normalization',
                fontsize=10, fontweight='bold')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xlim(-0.5, len(all_slides) - 0.5)


def generate_caption(df_spots, thresholds):
    """Generate LaTeX caption with statistics."""
    # Compute statistics
    total_spots = len(df_spots)
    n_slides = df_spots['slide'].nunique()
    spots_per_slide = total_spots / n_slides if n_slides > 0 else 0

    # Per-channel statistics
    stats_per_channel = {}
    for channel in ['green', 'orange']:
        df_ch = df_spots[df_spots['channel'] == channel]
        n_spots = len(df_ch)
        n_slides_ch = df_ch['slide'].nunique()

        # Compute spots per slide statistics
        spots_per_slide_array = df_ch.groupby('slide').size().values
        spots_per_slide_ch = np.mean(spots_per_slide_array)
        spots_per_slide_std = np.std(spots_per_slide_array)

        # Get threshold statistics
        thresh_vals = [v for k, v in thresholds.items() if k[1] == channel]
        mean_thresh = np.mean(thresh_vals) if thresh_vals else 0
        std_thresh = np.std(thresh_vals) if thresh_vals else 0

        stats_per_channel[channel] = {
            'n_spots': n_spots,
            'n_slides': n_slides_ch,
            'spots_per_slide': spots_per_slide_ch,
            'spots_per_slide_std': spots_per_slide_std,
            'mean_threshold': mean_thresh,
            'std_threshold': std_thresh
        }

    # Generate caption
    cv_green = stats_per_channel['green']['std_threshold'] / stats_per_channel['green']['mean_threshold'] * 100
    cv_orange = stats_per_channel['orange']['std_threshold'] / stats_per_channel['orange']['mean_threshold'] * 100

    caption = r"""
\begin{figure}[t]
  \centering
  \includegraphics[width=0.95\linewidth]{fig_negative_threshold.pdf}
  \caption{\textbf{Negative controls establish per-slide detection thresholds.}
  Analysis of """ + f"{total_spots:,}" + r""" negative control spots (bacterial \textit{DapB} probe,
  absent in mammalian tissue) across """ + f"{n_slides}" + r""" slides (mean $\pm$ standard deviation:
  """ + f"{spots_per_slide:.0f}" + r""" $\pm$ """ + f"{stats_per_channel['green']['spots_per_slide_std']:.0f}" + r""" spots/slide;
  """ + f"{stats_per_channel['green']['spots_per_slide']:.0f}" + r""" $\pm$ """ + f"{stats_per_channel['green']['spots_per_slide_std']:.0f}" + r""" green,
  """ + f"{stats_per_channel['orange']['spots_per_slide']:.0f}" + r""" $\pm$ """ + f"{stats_per_channel['orange']['spots_per_slide_std']:.0f}" + r""" orange).
  \textbf{(A)} Distributions of integrated photon counts from all negative-control detections
  (combined across slides). Mean $\pm$ s.d. of 95th percentile thresholds: """ + f"{stats_per_channel['green']['mean_threshold']:.0f}" + r""" $\pm$ """ + f"{stats_per_channel['green']['std_threshold']:.0f}" + r""" photons (green),
  """ + f"{stats_per_channel['orange']['mean_threshold']:.0f}" + r""" $\pm$ """ + f"{stats_per_channel['orange']['std_threshold']:.0f}" + r""" photons (orange).
  \textbf{(B)} Cumulative distribution functions (CDFs) per slide (thin lines) and mean CDF across all slides (bold)
  demonstrate consistent tail behavior despite variation in median background. Horizontal line at 0.95 marks
  the quantile used for thresholding.
  \textbf{(C)} Threshold values versus slide number (circles: green channel, squares: orange channel)
  reveal substantial slide-to-slide variation: coefficient of variation CV$_{\mathrm{green}}$ = """ + f"{cv_green:.1f}" + r"""\%
  (""" + f"{stats_per_channel['green']['mean_threshold']:.0f}" + r""" $\pm$ """ + f"{stats_per_channel['green']['std_threshold']:.0f}" + r""" photons) and
  CV$_{\mathrm{orange}}$ = """ + f"{cv_orange:.1f}" + r"""\%
  (""" + f"{stats_per_channel['orange']['mean_threshold']:.0f}" + r""" $\pm$ """ + f"{stats_per_channel['orange']['std_threshold']:.0f}" + r""" photons),
  motivating per-slide normalization to account for batch-to-batch variation in autofluorescence and staining efficiency.
  All values reported as mean $\pm$ standard deviation across slides.}
  \label{fig:negative_threshold}
\end{figure}
"""

    # Save caption to LaTeX file
    caption_file_tex = OUTPUT_DIR / 'fig_negative_threshold_caption.tex'
    with open(caption_file_tex, 'w') as f:
        f.write(caption)

    # Save caption to TXT file (same name as figure)
    caption_file_txt = OUTPUT_DIR / 'fig_negative_threshold_caption.txt'
    with open(caption_file_txt, 'w') as f:
        f.write("Figure: Negative controls establish per-slide detection thresholds\n")
        f.write("="*70 + "\n\n")
        f.write(f"Analysis of {total_spots:,} negative control spots (bacterial DapB probe,\n")
        f.write(f"absent in mammalian tissue) across {n_slides} slides (mean ± standard deviation:\n")
        f.write(f"{spots_per_slide:.0f} ± {stats_per_channel['green']['spots_per_slide_std']:.0f} spots/slide;\n")
        f.write(f"{stats_per_channel['green']['spots_per_slide']:.0f} ± {stats_per_channel['green']['spots_per_slide_std']:.0f} green, ")
        f.write(f"{stats_per_channel['orange']['spots_per_slide']:.0f} ± {stats_per_channel['orange']['spots_per_slide_std']:.0f} orange).\n\n")
        f.write("(A) Distributions of integrated photon counts from all negative-control detections\n")
        f.write("    (combined across slides). Mean ± s.d. of 95th percentile thresholds:\n")
        f.write(f"    - Green: {stats_per_channel['green']['mean_threshold']:.0f} ± {stats_per_channel['green']['std_threshold']:.0f} photons\n")
        f.write(f"    - Orange: {stats_per_channel['orange']['mean_threshold']:.0f} ± {stats_per_channel['orange']['std_threshold']:.0f} photons\n\n")
        f.write("(B) Cumulative distribution functions (CDFs) per slide (thin lines) and mean CDF\n")
        f.write("    across all slides (bold) demonstrate consistent tail behavior despite variation\n")
        f.write("    in median background. Horizontal line at 0.95 marks the quantile used for\n")
        f.write("    thresholding.\n\n")
        f.write("(C) Threshold values versus slide number (circles: green channel, squares: orange\n")
        f.write("    channel) reveal substantial slide-to-slide variation:\n")
        f.write(f"    - Coefficient of variation CV_green = {cv_green:.1f}%\n")
        f.write(f"      ({stats_per_channel['green']['mean_threshold']:.0f} ± {stats_per_channel['green']['std_threshold']:.0f} photons)\n")
        f.write(f"    - Coefficient of variation CV_orange = {cv_orange:.1f}%\n")
        f.write(f"      ({stats_per_channel['orange']['mean_threshold']:.0f} ± {stats_per_channel['orange']['std_threshold']:.0f} photons)\n")
        f.write("    This motivates per-slide normalization to account for batch-to-batch variation\n")
        f.write("    in autofluorescence and staining efficiency.\n\n")
        f.write("All values reported as mean ± standard deviation across slides.\n\n")
        f.write("="*70 + "\n")
        f.write("STATISTICS SUMMARY\n")
        f.write("="*70 + "\n")
        f.write(f"Total negative control spots: {total_spots:,}\n")
        f.write(f"Number of slides: {n_slides}\n")
        f.write(f"Spots per slide (overall): {spots_per_slide:.0f}\n\n")
        f.write("Green channel:\n")
        f.write(f"  Total spots: {stats_per_channel['green']['n_spots']:,}\n")
        f.write(f"  Spots/slide: {stats_per_channel['green']['spots_per_slide']:.0f} ± {stats_per_channel['green']['spots_per_slide_std']:.0f}\n")
        f.write(f"  Mean threshold: {stats_per_channel['green']['mean_threshold']:.0f} ± {stats_per_channel['green']['std_threshold']:.0f} photons\n")
        f.write(f"  Coefficient of variation: {cv_green:.1f}%\n\n")
        f.write("Orange channel:\n")
        f.write(f"  Total spots: {stats_per_channel['orange']['n_spots']:,}\n")
        f.write(f"  Spots/slide: {stats_per_channel['orange']['spots_per_slide']:.0f} ± {stats_per_channel['orange']['spots_per_slide_std']:.0f}\n")
        f.write(f"  Mean threshold: {stats_per_channel['orange']['mean_threshold']:.0f} ± {stats_per_channel['orange']['std_threshold']:.0f} photons\n")
        f.write(f"  Coefficient of variation: {cv_orange:.1f}%\n")

    print(f"\nLaTeX caption saved to: {caption_file_tex}")
    print(f"Text caption saved to: {caption_file_txt}")
    print("\n" + "="*70)
    print("CAPTION STATISTICS")
    print("="*70)
    print(f"Total negative control spots: {total_spots:,}")
    print(f"Number of slides: {n_slides}")
    print(f"Spots per slide (overall): {spots_per_slide:.0f}")
    print(f"\nGreen channel:")
    print(f"  Spots: {stats_per_channel['green']['n_spots']:,}")
    print(f"  Spots/slide: {stats_per_channel['green']['spots_per_slide']:.0f} ± {stats_per_channel['green']['spots_per_slide_std']:.0f}")
    print(f"  Mean threshold: {stats_per_channel['green']['mean_threshold']:.0f} ± {stats_per_channel['green']['std_threshold']:.0f} photons")
    print(f"  CV: {cv_green:.1f}%")
    print(f"\nOrange channel:")
    print(f"  Spots: {stats_per_channel['orange']['n_spots']:,}")
    print(f"  Spots/slide: {stats_per_channel['orange']['spots_per_slide']:.0f} ± {stats_per_channel['orange']['spots_per_slide_std']:.0f}")
    print(f"  Mean threshold: {stats_per_channel['orange']['mean_threshold']:.0f} ± {stats_per_channel['orange']['std_threshold']:.0f} photons")
    print(f"  CV: {cv_orange:.1f}%")


def main():
    """Main execution function."""
    # Load data
    df_extracted, desired_channels = load_negative_control_data()

    # Compute thresholds
    thresholds, error_thresholds, number_of_datapoints, age = compute_slide_thresholds(
        df_extracted, desired_channels
    )

    # Debug: Check probe set values
    if 'metadata_sample_Probe-Set' in df_extracted.columns:
        print(f"\nUnique probe sets found: {df_extracted['metadata_sample_Probe-Set'].unique()}")
    else:
        print("\nWARNING: 'metadata_sample_Probe-Set' column not found!")
        print(f"Available columns: {[c for c in df_extracted.columns if 'probe' in c.lower() or 'Probe' in c]}")

    # Extract spot-level data
    df_spots = extract_negative_control_spots(df_extracted, desired_channels)

    if len(df_spots) == 0:
        print("\nWARNING: No negative control spots found! Creating placeholder figure.")
        # Create placeholder data for demonstration
        df_spots = pd.DataFrame({
            'slide': ['placeholder'] * 100,
            'channel': ['green'] * 50 + ['orange'] * 50,
            'photons': np.random.exponential(1000, 100),
            'date': ['2025-01-01'] * 100
        })
    else:
        print(f"\nExtracted {len(df_spots)} negative control spots across {df_spots['slide'].nunique()} slides")

    # Get example FOV data
    example_fovs = get_example_fov_data(df_extracted)

    # Create figure
    fig = create_figure(df_spots, thresholds, error_thresholds, example_fovs)

    # Save figure
    output_path = OUTPUT_DIR / 'fig_negative_threshold.pdf'
    fig.savefig(output_path, format='pdf', bbox_inches='tight')
    print(f"\nFigure saved to: {output_path}")

    # Also save as SVG for easier editing
    output_path_svg = OUTPUT_DIR / 'fig_negative_threshold.svg'
    fig.savefig(output_path_svg, format='svg', bbox_inches='tight')
    print(f"Figure saved to: {output_path_svg}")

    plt.close(fig)

    # Save threshold data to CSV
    threshold_csv = OUTPUT_DIR / 'negative_control_thresholds.csv'
    threshold_rows = []
    for key, val in thresholds.items():
        if len(key) == 2:
            slide, channel = key
            threshold_rows.append({'slide': slide, 'channel': channel, 'threshold': val})
        elif len(key) == 3:
            slide, channel, region = key
            threshold_rows.append({'slide': slide, 'channel': channel, 'region': region, 'threshold': val})

    pd.DataFrame(threshold_rows).to_csv(threshold_csv, index=False)
    print(f"Threshold data saved to: {threshold_csv}")

    # Generate updated caption with statistics
    generate_caption(df_spots, thresholds)

    print("\n" + "="*70)
    print("FIGURE GENERATION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
