#!/usr/bin/env python3
"""
Create mRNA Size Montage

Generate a montage showing clusters organized by mRNA size for each channel.
4 rows x 6 columns layout, with columns representing different mRNA size ranges.

Usage:
    python create_mrna_size_montage.py --data_dir ./output
    python create_mrna_size_montage.py --data_dir ./output --output_dir ./output/montages

    # With contrast adjustment
    python create_mrna_size_montage.py --data_dir ./output --auto_contrast --save
    python create_mrna_size_montage.py --data_dir ./output --contrast_low 2 --contrast_high 98 --save
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import argparse
import re
import json
from datetime import datetime


# Define mRNA size bins (column categories)
MRNA_BINS = [
    ("~1 mRNA", 0.5,1),      # Single spots
    ("2-5 mRNA", 2, 5.0),     # Small clusters
    ("5-10 mRNA", 5.0, 10.0),   # Medium-small clusters
    ("10-15 mRNA", 10.0, 15.0), # Medium clusters
    ("15-20 mRNA", 15.0, 20.0), # Medium-large clusters
    (">20 mRNA", 20.0, None),   # Large clusters
]

N_ROWS = 4
N_COLS = 6


def adjust_gamma(img: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """
    Apply gamma correction to an image.

    Args:
        img: Input image (uint8)
        gamma: Gamma value. <1 brightens, >1 darkens. Default 1.0 (no change)

    Returns:
        Gamma-corrected image
    """
    if gamma == 1.0:
        return img

    # Build lookup table for efficiency
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)

    return table[img]


def adjust_contrast(img: np.ndarray, low_pct: float = 1, high_pct: float = 99,
                    channel_idx: int = None) -> np.ndarray:
    """
    Adjust image contrast using percentile-based normalization.

    Args:
        img: Input image (RGB or grayscale)
        low_pct: Lower percentile for black point (0-100)
        high_pct: Upper percentile for white point (0-100)
        channel_idx: If specified, only adjust contrast based on this channel (0=R, 1=G, 2=B)
                     but apply to all channels proportionally

    Returns:
        Contrast-adjusted image
    """
    img = img.astype(np.float32)

    if len(img.shape) == 2:
        # Grayscale
        low_val = np.percentile(img, low_pct)
        high_val = np.percentile(img, high_pct)
        if high_val > low_val:
            img = (img - low_val) / (high_val - low_val) * 255
        img = np.clip(img, 0, 255).astype(np.uint8)
    else:
        # RGB - adjust based on specified channel or luminance
        if channel_idx is not None and 0 <= channel_idx < img.shape[2]:
            # Use specific channel for percentile calculation
            ref_channel = img[:, :, channel_idx]
            low_val = np.percentile(ref_channel, low_pct)
            high_val = np.percentile(ref_channel, high_pct)
        else:
            # Use luminance (weighted average)
            luminance = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
            low_val = np.percentile(luminance, low_pct)
            high_val = np.percentile(luminance, high_pct)

        if high_val > low_val:
            # Apply same scaling to all channels
            img = (img - low_val) / (high_val - low_val) * 255
        img = np.clip(img, 0, 255).astype(np.uint8)

    return img


def parse_filename(filename: str) -> dict:
    """Parse cluster image filename to extract metadata."""
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


def load_cluster_images(image_dir: Path) -> dict:
    """
    Load all cluster images and create lookup by (channel, slide, label_id).

    Returns dict mapping (channel, slide, label_id) -> image array
    """
    image_lookup = {}

    if not image_dir.exists():
        return image_lookup

    for img_path in sorted(image_dir.glob('*.png')):
        parsed = parse_filename(img_path.name)
        if parsed is None:
            continue

        try:
            img = np.array(Image.open(img_path))
            key = (parsed['channel'], parsed['slide'], parsed['label_id'])
            image_lookup[key] = img
        except Exception as e:
            print(f"WARNING: Could not load {img_path.name}: {e}")

    return image_lookup


def get_clusters_by_mrna_bin(df: pd.DataFrame, min_mrna: float, max_mrna: float) -> pd.DataFrame:
    """Filter clusters by mRNA range."""
    if max_mrna is None:
        return df[df['mrna_equiv'] >= min_mrna]
    else:
        return df[(df['mrna_equiv'] >= min_mrna) & (df['mrna_equiv'] < max_mrna)]


def create_placeholder_image(size: tuple, text: str = "No data") -> np.ndarray:
    """Create a placeholder image with text."""
    img = Image.new('RGB', size, color=(50, 50, 50))
    draw = ImageDraw.Draw(img)

    # Try to use a default font, fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except:
        font = ImageFont.load_default()

    # Get text bounding box
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    # Center text
    x = (size[0] - text_w) // 2
    y = (size[1] - text_h) // 2
    draw.text((x, y), text, fill=(150, 150, 150), font=font)

    return np.array(img)


def create_montage_grid(images_grid: list, padding: int = 4,
                        bg_color: tuple = (30, 30, 30),
                        column_labels: list = None,
                        title: str = None) -> np.ndarray:
    """
    Create a montage from a 2D grid of images.

    Args:
        images_grid: List of lists, each inner list is a column of images
        padding: Padding between images
        bg_color: Background color
        column_labels: Labels for each column
        title: Title for the montage

    Returns:
        Montage as numpy array
    """
    if len(images_grid) == 0:
        return np.zeros((100, 100, 3), dtype=np.uint8)

    # Get max image dimensions
    max_h = 0
    max_w = 0
    for col in images_grid:
        for img in col:
            if img is not None:
                max_h = max(max_h, img.shape[0])
                max_w = max(max_w, img.shape[1])

    if max_h == 0 or max_w == 0:
        max_h, max_w = 100, 100

    n_cols = len(images_grid)
    n_rows = max(len(col) for col in images_grid) if images_grid else 0

    # Calculate dimensions
    header_height = 30 if column_labels else 0
    title_height = 40 if title else 0

    montage_h = title_height + header_height + n_rows * (max_h + padding) + padding
    montage_w = n_cols * (max_w + padding) + padding

    montage = np.full((montage_h, montage_w, 3), bg_color, dtype=np.uint8)

    # Add title
    if title:
        pil_montage = Image.fromarray(montage)
        draw = ImageDraw.Draw(pil_montage)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except:
            font = ImageFont.load_default()

        bbox = draw.textbbox((0, 0), title, font=font)
        text_w = bbox[2] - bbox[0]
        x = (montage_w - text_w) // 2
        draw.text((x, 10), title, fill=(255, 255, 255), font=font)
        montage = np.array(pil_montage)

    # Add column labels
    if column_labels:
        pil_montage = Image.fromarray(montage)
        draw = ImageDraw.Draw(pil_montage)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
        except:
            font = ImageFont.load_default()

        for col_idx, label in enumerate(column_labels):
            x = col_idx * (max_w + padding) + padding + max_w // 2
            bbox = draw.textbbox((0, 0), label, font=font)
            text_w = bbox[2] - bbox[0]
            draw.text((x - text_w // 2, title_height + 8), label, fill=(200, 200, 200), font=font)

        montage = np.array(pil_montage)

    # Place images
    for col_idx, col_images in enumerate(images_grid):
        for row_idx, img in enumerate(col_images):
            if img is None:
                img = create_placeholder_image((max_w, max_h), "No data")

            y = title_height + header_height + row_idx * (max_h + padding) + padding
            x = col_idx * (max_w + padding) + padding

            # Handle size mismatches
            ih, iw = img.shape[:2]
            h = min(ih, max_h)
            w = min(iw, max_w)

            # Handle grayscale
            if len(img.shape) == 2:
                img = np.stack([img] * 3, axis=-1)

            # Center smaller images
            y_offset = (max_h - h) // 2
            x_offset = (max_w - w) // 2

            montage[y + y_offset:y + y_offset + h, x + x_offset:x + x_offset + w] = img[:h, :w, :3]

    return montage


def main():
    parser = argparse.ArgumentParser(description='Create mRNA size montage')
    parser.add_argument('--data_dir', type=str, default='./output',
                        help='Directory containing cluster_data.csv and cluster_images/')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for montages (default: data_dir)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--n_examples', type=int, default=1,
                        help='Number of example montages to generate (each with different seed)')
    parser.add_argument('--save', action='store_true',
                        help='Save montages to files instead of displaying')

    # Contrast and gamma adjustment options
    parser.add_argument('--auto_contrast', action='store_true',
                        help='Apply automatic per-image contrast adjustment (default: 1-99 percentile)')
    parser.add_argument('--contrast_low', type=float, default=1,
                        help='Lower percentile for contrast adjustment (default: 1)')
    parser.add_argument('--contrast_high', type=float, default=99,
                        help='Upper percentile for contrast adjustment (default: 99)')
    parser.add_argument('--contrast_per_column', action='store_true',
                        help='Apply same contrast to all images in a column (based on column stats)')
    parser.add_argument('--gamma', type=float, default=1.0,
                        help='Gamma correction value. <1 brightens, >1 darkens (default: 1.0)')
    parser.add_argument('--svg', action='store_true',
                        help='Also save as SVG format (in addition to PNG)')

    # Filtering options
    parser.add_argument('--max_cv', type=float, default=None,
                        help='Maximum CV (coefficient of variation) filter (e.g., 0.5)')
    parser.add_argument('--min_cv', type=float, default=None,
                        help='Minimum CV filter')
    parser.add_argument('--exclude_slides', type=str, nargs='+', default=None,
                        help='Slides to exclude (e.g., --exclude_slides m1a1 m2a3)')
    parser.add_argument('--include_slides', type=str, nargs='+', default=None,
                        help='Only include these slides (e.g., --include_slides m1a1 m1a2)')
    parser.add_argument('--one_slide_per_example', action='store_true',
                        help='Each example uses clusters from only one slide (cycles through slides)')
    parser.add_argument('--min_clusters_per_slide', type=int, default=None,
                        help='Minimum clusters required per slide (only with --one_slide_per_example)')
    args = parser.parse_args()

    np.random.seed(args.seed)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else data_dir
    cluster_csv = data_dir / 'cluster_data.csv'
    image_dir = data_dir / 'cluster_images'

    # Check for required files
    if not cluster_csv.exists():
        print(f"ERROR: cluster_data.csv not found at {cluster_csv}")
        sys.exit(1)

    if not image_dir.exists():
        print(f"ERROR: cluster_images/ directory not found at {image_dir}")
        sys.exit(1)

    # Load data
    print(f"Loading cluster data from {cluster_csv}...")
    df = pd.read_csv(cluster_csv)
    print(f"Loaded {len(df)} cluster records")

    # Filter by threshold (keep only passing)
    if 'passes_threshold' in df.columns:
        df = df[df['passes_threshold'] == True].copy()
        print(f"After threshold filter: {len(df)} clusters")

    # Apply CV filter if requested
    if args.max_cv is not None:
        if 'cv_photons' in df.columns:
            df = df[df['cv_photons'] <= args.max_cv].copy()
            print(f"After CV <= {args.max_cv} filter: {len(df)} clusters")
        else:
            print("WARNING: cv_photons column not found, skipping CV filter")

    if args.min_cv is not None:
        if 'cv_photons' in df.columns:
            df = df[df['cv_photons'] >= args.min_cv].copy()
            print(f"After CV >= {args.min_cv} filter: {len(df)} clusters")
        else:
            print("WARNING: cv_photons column not found, skipping CV filter")

    # Apply slide filters
    if args.exclude_slides is not None:
        excluded = [s.lower() for s in args.exclude_slides]
        df = df[~df['slide'].str.lower().isin(excluded)].copy()
        print(f"After excluding slides {args.exclude_slides}: {len(df)} clusters")

    if args.include_slides is not None:
        included = [s.lower() for s in args.include_slides]
        df = df[df['slide'].str.lower().isin(included)].copy()
        print(f"After including only slides {args.include_slides}: {len(df)} clusters")

    # Show available slides
    print(f"Slides in data: {sorted(df['slide'].unique())}")

    # Load images
    print(f"Loading images from {image_dir}...")
    image_lookup = load_cluster_images(image_dir)
    print(f"Loaded {len(image_lookup)} images")

    # Get channels
    channels = sorted(df['channel'].unique())
    print(f"Channels: {channels}")

    # Get available slides for one_slide_per_example mode
    available_slides = sorted(df['slide'].unique())
    if args.one_slide_per_example:
        # Filter slides by minimum cluster count if specified
        if args.min_clusters_per_slide:
            slide_counts = df.groupby('slide').size()
            available_slides = [s for s in available_slides
                               if slide_counts.get(s, 0) >= args.min_clusters_per_slide]
            print(f"One-slide-per-example mode: {len(available_slides)} slides with >= {args.min_clusters_per_slide} clusters")
            if len(available_slides) == 0:
                print("ERROR: No slides meet the minimum cluster requirement")
                sys.exit(1)
        else:
            print(f"One-slide-per-example mode: {len(available_slides)} slides available")

        # Print cluster counts per slide for reference
        slide_counts = df.groupby('slide').size().sort_values(ascending=False)
        print(f"  Top slides by cluster count:")
        for slide, count in slide_counts.head(10).items():
            marker = " *" if slide in available_slides else ""
            print(f"    {slide}: {count} clusters{marker}")

        if args.n_examples > len(available_slides):
            print(f"  Note: n_examples ({args.n_examples}) > slides ({len(available_slides)}), will cycle")

    # Loop for multiple examples
    for example_idx in range(args.n_examples):
        current_seed = args.seed + example_idx

        # Select slide for this example if one_slide_per_example mode
        current_slide = None
        if args.one_slide_per_example:
            slide_idx = example_idx % len(available_slides)
            current_slide = available_slides[slide_idx]

        if args.n_examples > 1:
            print(f"\n{'='*60}")
            if current_slide:
                print(f"Generating example {example_idx + 1}/{args.n_examples} (seed={current_seed}, slide={current_slide})")
            else:
                print(f"Generating example {example_idx + 1}/{args.n_examples} (seed={current_seed})")
            print(f"{'='*60}")

        # Create montage for each channel
        for channel in channels:
            print(f"\nCreating montage for {channel} channel...")
            ch_df = df[df['channel'] == channel].copy()

            # Filter to single slide if one_slide_per_example mode
            if current_slide:
                ch_df = ch_df[ch_df['slide'] == current_slide].copy()
                print(f"  Filtered to slide {current_slide}: {len(ch_df)} clusters")

            # Build grid: each column is a mRNA bin, each has N_ROWS images
            images_grid = []
            column_labels = []

            for bin_label, min_mrna, max_mrna in MRNA_BINS:
                column_labels.append(bin_label)

                # Filter clusters in this bin
                bin_df = get_clusters_by_mrna_bin(ch_df, min_mrna, max_mrna)

                if example_idx == 0:  # Only print counts on first example
                    print(f"  {bin_label}: {len(bin_df)} clusters available")

                # Sample up to N_ROWS clusters
                column_images = []
                if len(bin_df) > 0:
                    # Sample randomly
                    n_sample = min(N_ROWS, len(bin_df))
                    sampled = bin_df.sample(n=n_sample, random_state=current_seed)

                    for _, row in sampled.iterrows():
                        key = (row['channel'], row['slide'], int(row['label_id']))
                        if key in image_lookup:
                            img = image_lookup[key].copy()
                            # Apply per-image contrast if requested
                            if args.auto_contrast and not args.contrast_per_column:
                                # Use the dominant channel for this fluorophore
                                ch_idx = 1 if channel == 'green' else 0  # G for green, R for orange
                                img = adjust_contrast(img, args.contrast_low, args.contrast_high, channel_idx=ch_idx)
                            # Apply gamma correction
                            if args.gamma != 1.0:
                                img = adjust_gamma(img, args.gamma)
                            column_images.append(img)
                        else:
                            column_images.append(None)

                    # Pad with None if not enough
                    while len(column_images) < N_ROWS:
                        column_images.append(None)
                else:
                    # No clusters in this bin
                    column_images = [None] * N_ROWS

                # Apply per-column contrast if requested
                if args.auto_contrast and args.contrast_per_column:
                    # Collect all pixel values from this column
                    all_pixels = []
                    ch_idx = 1 if channel == 'green' else 0
                    for img in column_images:
                        if img is not None:
                            all_pixels.append(img[:, :, ch_idx].flatten())

                    if all_pixels:
                        all_pixels = np.concatenate(all_pixels)
                        low_val = np.percentile(all_pixels, args.contrast_low)
                        high_val = np.percentile(all_pixels, args.contrast_high)

                        # Apply same contrast to all images in column
                        for i, img in enumerate(column_images):
                            if img is not None:
                                img_float = img.astype(np.float32)
                                if high_val > low_val:
                                    img_float = (img_float - low_val) / (high_val - low_val) * 255
                                column_images[i] = np.clip(img_float, 0, 255).astype(np.uint8)

                images_grid.append(column_images)

            # Create montage
            channel_name = "mHTT1a" if channel == "green" else "full-length mHTT"
            title = f"{channel.capitalize()} Channel ({channel_name}) - Clusters by mRNA Size"

            montage = create_montage_grid(
                images_grid,
                padding=4,
                column_labels=column_labels,
                title=title
            )

            if args.save:
                # Save to file
                output_dir.mkdir(parents=True, exist_ok=True)

                # Build filename suffix
                if args.one_slide_per_example and current_slide:
                    suffix = f"_{current_slide}"
                elif args.n_examples > 1:
                    suffix = f"_ex{example_idx + 1:02d}"
                else:
                    suffix = ""

                output_path = output_dir / f"mrna_size_montage_{channel}{suffix}.png"
                Image.fromarray(montage).save(output_path)
                print(f"Saved: {output_path}")

                # Also save as SVG if requested
                if args.svg:
                    svg_path = output_dir / f"mrna_size_montage_{channel}{suffix}.svg"
                    fig, ax = plt.subplots(figsize=(14, 10))
                    ax.imshow(montage)
                    ax.axis('off')
                    ax.set_title(title, fontsize=12)
                    plt.tight_layout()
                    fig.savefig(svg_path, format='svg', bbox_inches='tight', pad_inches=0.1)
                    plt.close(fig)
                    print(f"Saved: {svg_path}")

                # Save metadata
                metadata = {
                    "command": "python " + " ".join(sys.argv),
                    "created": datetime.now().isoformat(),
                    "example": example_idx + 1 if args.n_examples > 1 else None,
                    "n_examples": args.n_examples if args.n_examples > 1 else None,
                    "seed_used": current_seed,
                    "slide": current_slide,
                    "one_slide_per_example": args.one_slide_per_example,
                    "channel": channel,
                    "channel_name": channel_name,
                    "filters": {
                        "passes_threshold": True,
                        "min_cv": args.min_cv,
                        "max_cv": args.max_cv,
                        "exclude_slides": args.exclude_slides,
                        "include_slides": args.include_slides,
                    },
                    "adjustments": {
                        "auto_contrast": args.auto_contrast,
                        "contrast_low": args.contrast_low if args.auto_contrast else None,
                        "contrast_high": args.contrast_high if args.auto_contrast else None,
                        "contrast_per_column": args.contrast_per_column,
                        "gamma": args.gamma,
                    },
                    "layout": {
                        "n_rows": N_ROWS,
                        "n_cols": N_COLS,
                        "mrna_bins": [(label, min_v, max_v) for label, min_v, max_v in MRNA_BINS],
                    },
                    "data": {
                        "base_seed": args.seed,
                        "total_clusters_after_filters": len(ch_df),
                        "clusters_per_bin": {label: len(get_clusters_by_mrna_bin(ch_df, min_v, max_v))
                                             for label, min_v, max_v in MRNA_BINS},
                        "slides_included": sorted(ch_df['slide'].unique().tolist()),
                    },
                }
                metadata_path = output_dir / f"mrna_size_montage_{channel}{suffix}_metadata.json"
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                print(f"Saved: {metadata_path}")
            else:
                # Display
                fig, ax = plt.subplots(figsize=(14, 10))
                ax.imshow(montage)
                ax.axis('off')
                ax.set_title(title)
                plt.tight_layout()
                plt.show()

    print("\nDone!")


if __name__ == '__main__':
    main()
