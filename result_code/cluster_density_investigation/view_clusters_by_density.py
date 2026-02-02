#!/usr/bin/env python3
"""
View Clusters by Density

Load saved cluster images from cluster_images/ and filter by various criteria.
Uses cluster_data.csv for metadata and matches to saved PNG files.

Usage:
    python view_clusters_by_density.py --data_dir ./output
    python view_clusters_by_density.py --data_dir ./output --min_density 0.5 --max_density 1.0
    python view_clusters_by_density.py --data_dir ./output --channel green --min_volume 10
    python view_clusters_by_density.py --data_dir ./output --max_density 0.3 --save_montage low_density.png

Options:
    --data_dir       Directory containing cluster_data.csv and cluster_images/
    --min_density    Minimum density threshold (mRNA/µm³)
    --max_density    Maximum density threshold (mRNA/µm³)
    --min_volume     Minimum volume threshold (µm³)
    --max_volume     Maximum volume threshold (µm³)
    --min_intensity  Minimum raw intensity
    --max_intensity  Maximum raw intensity
    --channel        Filter by channel (green/orange)
    --slide          Filter by slide (e.g., m1a1)
    --n_clusters     Max number of clusters to display (default: 100)
    --sort_by        Sort by: density, volume, mrna_equiv, raw_intensity (default: density)
    --ascending      Sort in ascending order (default: descending)
    --save_montage   Save montage to file instead of displaying
    --n_cols         Number of columns in montage (default: 10)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import re


def parse_filename(filename: str) -> dict:
    """
    Parse a cluster image filename to extract metadata.

    Format: {density:07.2f}_{channel}_{slide}_{region}_{label_id}.png
    Example: 0025.30_green_m1a1_r003_0042.png

    Returns dict with density, channel, slide, region, label_id or None if parsing fails.
    """
    match = re.match(r'^(\d+\.\d+)_(\w+)_(\w+)_(r\d+)_(\d+)\.png$', filename)
    if match:
        return {
            'density': float(match.group(1)),
            'channel': match.group(2),
            'slide': match.group(3),
            'region': match.group(4),
            'label_id': int(match.group(5)),
        }
    return None


def load_cluster_images(image_dir: Path) -> list:
    """
    Load all cluster images from directory.

    Returns list of dicts with 'path', 'image', and parsed metadata.
    """
    images = []

    if not image_dir.exists():
        return images

    for img_path in sorted(image_dir.glob('*.png')):
        parsed = parse_filename(img_path.name)
        if parsed is None:
            continue

        try:
            img = np.array(Image.open(img_path))
            images.append({
                'path': img_path,
                'image': img,
                **parsed
            })
        except Exception as e:
            print(f"WARNING: Could not load {img_path.name}: {e}")

    return images


def show_distribution_plots(df: pd.DataFrame, filter_desc: list, reference_factor: float = 0.5,
                            outlier_mask: np.ndarray = None, outlier_method: str = None):
    """
    Show distribution plots for the filtered clusters.

    Creates a 4x3 figure with:
    - Row 1: Density, Volume, mRNA histograms
    - Row 2: Density vs Volume scatter (per channel)
    - Row 3: Intensity vs Volume scatter (per channel)
    - Row 4: Log-log Intensity vs Volume (helps identify background)

    Args:
        df: DataFrame with cluster data
        filter_desc: List of filter descriptions for title
        reference_factor: Factor for reference lines (default 0.5 shows 0.5x and 2x median)
        outlier_mask: Boolean array indicating outliers (True = outlier)
        outlier_method: Name of outlier detection method for labeling
    """
    fig, axes = plt.subplots(4, 3, figsize=(15, 18))

    # If outlier mask provided, create normal/outlier subsets
    has_outliers = outlier_mask is not None and outlier_mask.any()

    channels = list(df['channel'].unique())
    colors = {'green': 'green', 'orange': 'orange'}

    # Row 0: Density histograms
    ax = axes[0, 0]
    for channel in channels:
        ch_data = df[df['channel'] == channel]['density']
        ax.hist(ch_data, bins=50, alpha=0.6, color=colors.get(channel, 'gray'),
                label=f'{channel} (n={len(ch_data)})')
    ax.set_xlabel('Density (mRNA/µm³)')
    ax.set_ylabel('Count')
    ax.set_title('Density Distribution')
    ax.legend()

    # Row 0: Volume histograms
    ax = axes[0, 1]
    for channel in channels:
        ch_data = df[df['channel'] == channel]['volume_um3']
        ax.hist(ch_data, bins=50, alpha=0.6, color=colors.get(channel, 'gray'),
                label=f'{channel} (n={len(ch_data)})')
    ax.set_xlabel('Volume (µm³)')
    ax.set_ylabel('Count')
    ax.set_title('Volume Distribution')
    ax.legend()

    # Row 0: mRNA equivalent histograms
    ax = axes[0, 2]
    for channel in channels:
        ch_data = df[df['channel'] == channel]['mrna_equiv']
        ax.hist(ch_data, bins=50, alpha=0.6, color=colors.get(channel, 'gray'),
                label=f'{channel} (n={len(ch_data)})')
    ax.set_xlabel('mRNA equivalent')
    ax.set_ylabel('Count')
    ax.set_title('mRNA Equivalent Distribution')
    ax.legend()

    # Row 1: Density vs Volume scatter plots (per channel)
    for i, channel in enumerate(channels):
        if i >= 3:
            break
        ax = axes[1, i]
        ch_data = df[df['channel'] == channel]
        ax.scatter(ch_data['volume_um3'], ch_data['density'],
                   alpha=0.3, s=3, c=colors.get(channel, 'gray'))
        ax.set_xlabel('Volume (µm³)')
        ax.set_ylabel('Density (mRNA/µm³)')
        ax.set_title(f'{channel.capitalize()} - Density vs Volume')
        # Data-driven limits
        if len(ch_data) > 0:
            ax.set_xlim(0, ch_data['volume_um3'].quantile(0.99) * 1.1)
            ax.set_ylim(0, ch_data['density'].quantile(0.99) * 1.1)

    # Hide unused axes in row 1
    for i in range(len(channels), 3):
        axes[1, i].axis('off')

    # Row 2: Intensity vs Volume scatter plots (per channel)
    for i, channel in enumerate(channels):
        if i >= 3:
            break
        ax = axes[2, i]
        ch_data = df[df['channel'] == channel]
        ax.scatter(ch_data['volume_um3'], ch_data['raw_intensity'],
                   alpha=0.3, s=3, c=colors.get(channel, 'gray'))
        ax.set_xlabel('Volume (µm³)')
        ax.set_ylabel('Raw Intensity (photons)')
        ax.set_title(f'{channel.capitalize()} - Intensity vs Volume')
        # Data-driven limits
        if len(ch_data) > 0:
            ax.set_xlim(0, ch_data['volume_um3'].quantile(0.99) * 1.1)
            ax.set_ylim(0, ch_data['raw_intensity'].quantile(0.99) * 1.1)

    # Hide unused axes in row 2
    for i in range(len(channels), 3):
        axes[2, i].axis('off')

    # Row 3: Log-log Intensity vs Volume (helps identify background clusters)
    # Real clusters should follow a line (intensity ~ volume), background deviates
    for i, channel in enumerate(channels):
        if i >= 3:
            break
        ax = axes[3, i]
        ch_mask = df['channel'] == channel
        ch_data = df[ch_mask]

        # Filter out zeros for log scale
        valid = (ch_data['volume_um3'] > 0) & (ch_data['raw_intensity'] > 0)
        ch_valid = ch_data[valid]

        if len(ch_valid) > 0:
            if has_outliers:
                # Get outlier mask for this channel's valid data
                ch_outlier_mask = outlier_mask[ch_mask.values][valid.values]
                normal_data = ch_valid[~ch_outlier_mask]
                outlier_data = ch_valid[ch_outlier_mask]

                # Plot normal points in blue
                ax.scatter(normal_data['volume_um3'], normal_data['raw_intensity'],
                           alpha=0.3, s=3, c='blue', label=f'Normal (n={len(normal_data)})')
                # Plot outliers in red
                ax.scatter(outlier_data['volume_um3'], outlier_data['raw_intensity'],
                           alpha=0.6, s=8, c='red', label=f'Outlier (n={len(outlier_data)})')
                ax.set_title(f'{channel.capitalize()} - Log-Log ({outlier_method or "outliers"})')
            else:
                # Color by density to show which points are "diffuse"
                scatter = ax.scatter(ch_valid['volume_um3'], ch_valid['raw_intensity'],
                           alpha=0.4, s=3, c=ch_valid['density'], cmap='viridis')
                ax.set_title(f'{channel.capitalize()} - Log-Log (color=density)')
                plt.colorbar(scatter, ax=ax, label='Density')

            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('Volume (µm³) [log]')
            ax.set_ylabel('Raw Intensity [log]')

            # Add reference line for constant density (median)
            vol_range = np.logspace(np.log10(ch_valid['volume_um3'].min()),
                                    np.log10(ch_valid['volume_um3'].max()), 100)
            median_density = ch_valid['density'].median()
            # intensity = density * volume * peak_intensity
            # For reference, just show slope=1 line through median
            median_int = ch_valid['raw_intensity'].median()
            median_vol = ch_valid['volume_um3'].median()
            ref_intensity = median_int * (vol_range / median_vol)
            ax.plot(vol_range, ref_intensity, 'k--', alpha=0.7, linewidth=2, label='1x median')

            # Add reference factor lines (e.g., 0.5x and 2x median density)
            # Low factor line: reference_factor * median (below this = potential background)
            low_factor = reference_factor
            low_intensity = ref_intensity * low_factor
            ax.plot(vol_range, low_intensity, 'gray', linestyle='--', alpha=0.5, linewidth=1.5,
                    label=f'{low_factor}x median')

            # High factor line: 1/reference_factor * median (above this = high density)
            high_factor = 1.0 / reference_factor
            high_intensity = ref_intensity * high_factor
            ax.plot(vol_range, high_intensity, 'gray', linestyle=':', alpha=0.5, linewidth=1.5,
                    label=f'{high_factor:.1f}x median')

            ax.legend(fontsize=7, loc='lower right')

    # Hide unused axes in row 3
    for i in range(len(channels), 3):
        axes[3, i].axis('off')

    # Title with filter info
    filter_str = ', '.join(filter_desc) if filter_desc else 'all clusters'
    fig.suptitle(f'Distribution of filtered clusters: {filter_str}\n(n={len(df)})', fontsize=12)

    plt.tight_layout()
    plt.show()


def create_montage(images: list, n_cols: int = 10,
                   padding: int = 2, bg_color: tuple = (30, 30, 30)) -> np.ndarray:
    """
    Create a montage of images.

    Args:
        images: List of numpy arrays (RGB images)
        n_cols: Number of columns
        padding: Padding between images
        bg_color: Background color (RGB tuple)

    Returns:
        Montage as numpy array
    """
    if len(images) == 0:
        return np.zeros((100, 100, 3), dtype=np.uint8)

    # Get image size from first image
    img_h, img_w = images[0].shape[:2]
    n_rows = (len(images) + n_cols - 1) // n_cols

    montage_h = n_rows * (img_h + padding) + padding
    montage_w = n_cols * (img_w + padding) + padding

    montage = np.full((montage_h, montage_w, 3), bg_color, dtype=np.uint8)

    for i, img in enumerate(images):
        row = i // n_cols
        col = i % n_cols

        y = row * (img_h + padding) + padding
        x = col * (img_w + padding) + padding

        # Handle size mismatches
        ih, iw = img.shape[:2]
        h = min(ih, img_h)
        w = min(iw, img_w)

        # Handle grayscale images
        if len(img.shape) == 2:
            img = np.stack([img] * 3, axis=-1)

        montage[y:y+h, x:x+w] = img[:h, :w, :3]

    return montage


def main():
    parser = argparse.ArgumentParser(description='View clusters filtered by various criteria')
    parser.add_argument('--data_dir', type=str, default='./output',
                        help='Directory containing cluster_data.csv and cluster_images/')
    parser.add_argument('--min_density', type=float, default=None,
                        help='Minimum density threshold (mRNA/µm³)')
    parser.add_argument('--max_density', type=float, default=None,
                        help='Maximum density threshold (mRNA/µm³)')
    parser.add_argument('--min_volume', type=float, default=None,
                        help='Minimum volume threshold (µm³)')
    parser.add_argument('--max_volume', type=float, default=None,
                        help='Maximum volume threshold (µm³)')
    parser.add_argument('--min_intensity', type=float, default=None,
                        help='Minimum raw intensity')
    parser.add_argument('--max_intensity', type=float, default=None,
                        help='Maximum raw intensity')
    parser.add_argument('--min_intensity_per_volume', type=float, default=None,
                        help='Minimum intensity per volume (photons/µm³) - filters out diffuse background')
    parser.add_argument('--below_reference', action='store_true',
                        help='Show only clusters BELOW the reference line (low density, potential background)')
    parser.add_argument('--above_reference', action='store_true',
                        help='Show only clusters ABOVE the reference line (high density)')
    parser.add_argument('--reference_factor', type=float, default=0.5,
                        help='Factor for reference line filtering (default 0.5 = below half median density)')
    parser.add_argument('--channel', type=str, default=None,
                        help='Filter by channel (green/orange)')
    parser.add_argument('--slide', type=str, default=None,
                        help='Filter by slide (e.g., m1a1)')
    parser.add_argument('--no_threshold_filter', action='store_true',
                        help='Include clusters that do NOT pass the negative control threshold (filtered by default)')
    parser.add_argument('--n_clusters', type=int, default=None,
                        help='Max number of clusters to display (default: 100 for montage, unlimited for napari)')
    parser.add_argument('--sort_by', type=str, default='density',
                        choices=['density', 'volume_um3', 'mrna_equiv', 'raw_intensity'],
                        help='Sort clusters by this field')
    parser.add_argument('--ascending', action='store_true',
                        help='Sort in ascending order (default: descending)')
    parser.add_argument('--save_montage', type=str, default=None,
                        help='Save montage to this path instead of displaying')
    parser.add_argument('--n_cols', type=int, default=10,
                        help='Number of columns in montage')
    parser.add_argument('--napari', action='store_true',
                        help='View in napari (scrollable stack)')
    parser.add_argument('--no_distributions', action='store_true',
                        help='Skip showing distribution plots')
    parser.add_argument('--show_outliers', action='store_true',
                        help='Load and show outliers from outlier_detection results')
    parser.add_argument('--outlier_method', type=str, default='Threshold',
                        choices=['Threshold', 'IsolationForest', 'LOF'],
                        help='Which outlier detection method to use (default: Threshold)')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    cluster_csv = data_dir / 'cluster_data.csv'
    image_dir = data_dir / 'cluster_images'

    # Check for required files
    if not cluster_csv.exists():
        print(f"ERROR: cluster_data.csv not found at {cluster_csv}")
        print("Please run investigate_cluster_density.py first.")
        sys.exit(1)

    if not image_dir.exists():
        print(f"ERROR: cluster_images/ directory not found at {image_dir}")
        print("Please run investigate_cluster_density.py with --save_images flag.")
        sys.exit(1)

    # Print defaults
    print("\nDefaults:")
    print("  - passes_threshold filter: ON (use --no_threshold_filter to disable)")
    print("  - distribution plots: ON (use --no_distributions to disable)")
    print()

    # Load cluster metadata
    print(f"Loading cluster data from {cluster_csv}...")
    df = pd.read_csv(cluster_csv)
    print(f"Loaded {len(df)} cluster records")

    # Apply filters to metadata
    mask = pd.Series([True] * len(df))
    filter_desc = []

    if args.min_density is not None:
        mask &= df['density'] >= args.min_density
        filter_desc.append(f"density >= {args.min_density}")

    if args.max_density is not None:
        mask &= df['density'] <= args.max_density
        filter_desc.append(f"density <= {args.max_density}")

    if args.min_volume is not None:
        mask &= df['volume_um3'] >= args.min_volume
        filter_desc.append(f"volume >= {args.min_volume}")

    if args.max_volume is not None:
        mask &= df['volume_um3'] <= args.max_volume
        filter_desc.append(f"volume <= {args.max_volume}")

    if args.min_intensity is not None:
        mask &= df['raw_intensity'] >= args.min_intensity
        filter_desc.append(f"intensity >= {args.min_intensity}")

    if args.max_intensity is not None:
        mask &= df['raw_intensity'] <= args.max_intensity
        filter_desc.append(f"intensity <= {args.max_intensity}")

    if args.min_intensity_per_volume is not None:
        intensity_per_vol = df['raw_intensity'] / df['volume_um3']
        mask &= intensity_per_vol >= args.min_intensity_per_volume
        filter_desc.append(f"intensity/volume >= {args.min_intensity_per_volume}")

    # Filter by position relative to reference line (median density)
    if args.below_reference or args.above_reference:
        # Calculate median density per channel and filter
        for channel in df['channel'].unique():
            ch_mask = df['channel'] == channel
            ch_data = df[ch_mask]
            median_density = ch_data['density'].median()
            threshold_density = median_density * args.reference_factor

            if args.below_reference:
                # Keep only clusters with density below threshold
                mask &= ~ch_mask | (df['density'] < threshold_density)
            elif args.above_reference:
                # Keep only clusters with density above threshold (inverse factor)
                high_threshold = median_density * (1 / args.reference_factor)
                mask &= ~ch_mask | (df['density'] > high_threshold)

        if args.below_reference:
            filter_desc.append(f"density < {args.reference_factor}x median (below ref)")
        else:
            filter_desc.append(f"density > {1/args.reference_factor:.1f}x median (above ref)")

    if args.channel is not None:
        mask &= df['channel'] == args.channel
        filter_desc.append(f"channel = {args.channel}")

    if args.slide is not None:
        mask &= df['slide'] == args.slide
        filter_desc.append(f"slide = {args.slide}")

    # Filter by threshold (default ON, use --no_threshold_filter to disable)
    if not args.no_threshold_filter:
        if 'passes_threshold' in df.columns:
            mask &= df['passes_threshold'] == True
            filter_desc.append("passes_threshold")
        else:
            print("WARNING: 'passes_threshold' column not found in cluster_data.csv")
            print("  This column is only present if investigate_cluster_density.py was run with photon_thresholds.csv")
            print("  Use --no_threshold_filter to skip this filter")

    df_filtered = df[mask].copy()

    # Load outlier data if requested
    outlier_mask = None
    outlier_method = None
    if args.show_outliers:
        outlier_dir = data_dir / 'outlier_detection'
        outlier_method = args.outlier_method
        outlier_col = f'outlier_{outlier_method}'

        # Load outlier labels for each channel and merge
        outlier_masks = {}
        for channel in df_filtered['channel'].unique():
            outlier_csv = outlier_dir / f'outliers_{channel}.csv'
            if outlier_csv.exists():
                outlier_df = pd.read_csv(outlier_csv)
                if outlier_col in outlier_df.columns:
                    # Create lookup by (slide, label_id)
                    for _, row in outlier_df.iterrows():
                        key = (channel, row['slide'], row['label_id'])
                        outlier_masks[key] = row[outlier_col]
                else:
                    print(f"WARNING: Column '{outlier_col}' not found in {outlier_csv}")
            else:
                print(f"WARNING: Outlier file not found: {outlier_csv}")
                print(f"  Run outlier_detection.py first")

        # Map outliers to filtered dataframe
        if outlier_masks:
            outlier_mask = np.array([
                outlier_masks.get((row['channel'], row['slide'], row['label_id']), False)
                for _, row in df_filtered.iterrows()
            ])
            n_outliers = outlier_mask.sum()
            print(f"Loaded outlier labels ({outlier_method}): {n_outliers} outliers ({n_outliers/len(df_filtered)*100:.1f}%)")
            filter_desc.append(f"showing {outlier_method} outliers")

    if filter_desc:
        print(f"Filters applied: {', '.join(filter_desc)}")
    print(f"After filtering: {len(df_filtered)} clusters")

    if len(df_filtered) == 0:
        print("No clusters match the criteria. Exiting.")
        sys.exit(0)

    # Sort
    df_filtered = df_filtered.sort_values(args.sort_by, ascending=args.ascending)

    # Limit (only if explicitly specified)
    if args.n_clusters is not None and len(df_filtered) > args.n_clusters:
        df_filtered = df_filtered.head(args.n_clusters)
        print(f"Limiting to {args.n_clusters} clusters")

    # Print summary
    print(f"\nFiltered cluster summary:")
    print(f"  Density range: {df_filtered['density'].min():.2f} - {df_filtered['density'].max():.2f} mRNA/µm³")
    print(f"  Volume range: {df_filtered['volume_um3'].min():.2f} - {df_filtered['volume_um3'].max():.2f} µm³")
    print(f"  mRNA range: {df_filtered['mrna_equiv'].min():.1f} - {df_filtered['mrna_equiv'].max():.1f}")
    print(f"  Slides: {sorted(df_filtered['slide'].unique())}")
    print(f"  Channels: {sorted(df_filtered['channel'].unique())}")

    # Show distribution plots (default on, use --no_distributions to skip)
    if not args.no_distributions:
        show_distribution_plots(df_filtered, filter_desc, reference_factor=args.reference_factor,
                                outlier_mask=outlier_mask, outlier_method=outlier_method)

    # Load images
    print(f"\nLoading images from {image_dir}...")
    all_images = load_cluster_images(image_dir)
    print(f"Found {len(all_images)} images in cluster_images/")

    if len(all_images) == 0:
        print("No images found. Exiting.")
        sys.exit(0)

    # Match filtered clusters to images
    # Create a lookup key for each image
    image_lookup = {}
    for img_data in all_images:
        # Key: (channel, slide, label_id) - density might have rounding differences
        key = (img_data['channel'], img_data['slide'], img_data['label_id'])
        image_lookup[key] = img_data

    matched_images = []
    for _, row in df_filtered.iterrows():
        key = (row['channel'], row['slide'], int(row['label_id']))
        if key in image_lookup:
            matched_images.append({
                'image': image_lookup[key]['image'],
                'density': row['density'],
                'volume': row['volume_um3'],
                'mrna': row['mrna_equiv'],
                'channel': row['channel'],
                'slide': row['slide'],
            })

    print(f"Matched {len(matched_images)} images to filtered clusters")

    if len(matched_images) == 0:
        print("No images matched. This may happen if images were saved with different parameters.")
        print("Try running investigate_cluster_density.py with --save_images again.")
        sys.exit(0)

    # Create montage
    images_array = [m['image'] for m in matched_images]

    # Napari stack output
    if args.napari:
        # Pad images to same size and stack
        max_h = max(img.shape[0] for img in images_array)
        max_w = max(img.shape[1] for img in images_array)

        padded = []
        for img in images_array:
            h, w = img.shape[:2]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, axis=-1)
            pad_h = max_h - h
            pad_w = max_w - w
            if pad_h > 0 or pad_w > 0:
                img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
            padded.append(img[:, :, :3])  # Ensure RGB only

        stack = np.stack(padded, axis=0)  # Shape: (N, H, W, 3)

        # Launch napari directly (no file saving)
        print(f"\nLaunching napari with {len(matched_images)} clusters...")
        print(f"  Sorted by: {args.sort_by} ({'ascending' if args.ascending else 'descending'})")
        import napari
        viewer = napari.view_image(stack, rgb=True, name='clusters')
        viewer.title = f"Clusters ({len(matched_images)}) - sorted by {args.sort_by}"
        napari.run()
        sys.exit(0)

    montage = create_montage(images_array, n_cols=args.n_cols)

    if args.save_montage:
        # Save montage
        output_path = Path(args.save_montage)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(montage).save(output_path)
        print(f"\nSaved montage to {output_path}")
        print(f"  Size: {montage.shape[1]}x{montage.shape[0]} pixels")
        print(f"  Contains: {len(matched_images)} clusters")
    else:
        # Interactive display
        fig, ax = plt.subplots(figsize=(16, 12))
        ax.imshow(montage)
        ax.axis('off')

        # Title with filter info
        title = f"Clusters: {', '.join(filter_desc) if filter_desc else 'all'}"
        title += f"\n{len(matched_images)} clusters, sorted by {args.sort_by} ({'asc' if args.ascending else 'desc'})"
        ax.set_title(title, fontsize=12)

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    main()
