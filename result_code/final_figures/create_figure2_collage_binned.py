#!/usr/bin/env python3
"""
Create SVG collages with columns representing mRNA bins (dim to bright).

Layout: 3 rows x 7 columns
Each column = one mRNA bin (e.g., 1-3, 3-5, 5-8, etc.)
Each row = different example from that bin

Each comparison image shows LEFT=single mRNA reference, RIGHT=cluster.
Creates multiple collages with different samples from each bin.

Dimensions: 140mm width (max), height scaled proportionally.
"""

import os
import re
import sys
import json
from pathlib import Path
import base64
from io import BytesIO
from collections import defaultdict
from PIL import Image

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from figure_config import FigureConfig
from results_config import EXCLUDED_SLIDES

# Input/output directories
INPUT_DIR = Path(__file__).parent / 'output' / 'example_images' / 'figure2'
OUTPUT_DIR = Path(__file__).parent / 'output' / 'figure2_collages_binned'

# ══════════════════════════════════════════════════════════════════════════════
# LAYOUT DIMENSIONS (in mm)
# ══════════════════════════════════════════════════════════════════════════════
TOTAL_WIDTH_MM = 140.0        # Max width
GAP_MM = 0.5                  # Gap between cells
HEADER_HEIGHT_MM = 4.0        # Height for column headers
N_ROWS = 3                    # Number of example rows

# mRNA bins for columns - each image shows [single mRNA | cluster] side-by-side
# Bins represent the cluster mRNA count (right side of each image)
MRNA_BINS = [
    (1, 3),      # Column 1: very small clusters
    (3, 5),      # Column 2
    (5, 8),      # Column 3
    (8, 12),     # Column 4
    (12, 18),    # Column 5
    (18, 30),    # Column 6
    (30, 50),    # Column 7: up to 50 mRNA
]

# Calculate cell dimensions based on layout
N_COLS = len(MRNA_BINS)
CELL_WIDTH_MM = (TOTAL_WIDTH_MM - (N_COLS - 1) * GAP_MM) / N_COLS
# Maintain aspect ratio ~1.75:1 (original 140:80 pixels)
CELL_HEIGHT_MM = CELL_WIDTH_MM / 1.75
TOTAL_HEIGHT_MM = HEADER_HEIGHT_MM + N_ROWS * CELL_HEIGHT_MM + (N_ROWS - 1) * GAP_MM


def extract_mrna_from_filename(filename: str) -> float:
    """Extract mRNA count from filename like 'comparison_01_small_6mrna.png'."""
    match = re.search(r'(\d+)mrna', filename)
    if match:
        return float(match.group(1))
    return 0


def image_to_base64(img_path: str) -> str:
    """Convert image (TIFF or PNG) to base64-encoded PNG string."""
    with Image.open(img_path) as img:
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode('utf-8')


def is_excluded_slide(fov_dir: Path) -> bool:
    """Check if FOV is from an excluded slide."""
    fov_name = fov_dir.name.lower()
    for slide in EXCLUDED_SLIDES:
        if slide.lower() in fov_name:
            return True
    return False


def collect_and_bin_images(channel: str) -> dict:
    """
    Collect all comparison images for a channel and bin by mRNA.
    Each image shows [single mRNA | cluster] side-by-side.
    Excludes images from EXCLUDED_SLIDES.
    Prefers TIFF files (raw, no annotations) over PNG.
    Returns dict: {bin_index: [list of (path, mrna, contour_path) tuples]}
    """
    binned = defaultdict(list)

    for fov_dir in INPUT_DIR.glob('fov_*'):
        # Skip excluded slides
        if is_excluded_slide(fov_dir):
            continue

        channel_dir = fov_dir / channel / 'spot_vs_cluster_comparisons'
        if not channel_dir.exists():
            continue

        # Look for TIFF files first (raw images)
        for img_path in channel_dir.glob('comparison_*.tif'):
            mrna = extract_mrna_from_filename(img_path.name)
            # Find corresponding contour JSON
            contour_path = img_path.with_name(img_path.stem + '_contour.json')

            # Find which bin this cluster belongs to
            for bin_idx, (low, high) in enumerate(MRNA_BINS):
                if low <= mrna < high:
                    binned[bin_idx].append((img_path, mrna, contour_path if contour_path.exists() else None))
                    break

    # Sort each bin by mrna
    for bin_idx in binned:
        binned[bin_idx].sort(key=lambda x: x[1])

    return binned


def create_binned_collage_svg(binned_images: dict, output_path: Path,
                               collage_index: int = 0):
    """
    Create an SVG collage with columns representing mRNA bins.

    Uses dimensions from module constants (TOTAL_WIDTH_MM, CELL_WIDTH_MM, etc.)

    Args:
        binned_images: Dict {bin_index: [(path, mrna, contour_path), ...]}
        output_path: Where to save the SVG
        collage_index: Which collage (0, 1, 2...) - determines which samples to pick
    """
    # Get styling from FigureConfig
    cfg = FigureConfig
    font_family = ', '.join(cfg.FONT_SANS_SERIF)

    # Start SVG with mm units
    # Font size conversion: 1pt ≈ 0.35mm, but we want smaller so use 0.25
    header_font_size_mm = cfg.FONT_SIZE_ANNOTATION * 0.30  # Smaller header font

    svg_parts = [f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg"
     xmlns:xlink="http://www.w3.org/1999/xlink"
     width="{TOTAL_WIDTH_MM}mm" height="{TOTAL_HEIGHT_MM:.2f}mm"
     viewBox="0 0 {TOTAL_WIDTH_MM} {TOTAL_HEIGHT_MM:.2f}">

  <defs>
    <style>
      .header-label {{ font-family: {font_family}; font-size: {header_font_size_mm:.2f}mm; font-weight: bold; fill: {cfg.COLOR_WHITE}; text-anchor: middle; }}
    </style>
  </defs>

  <!-- Background -->
  <rect width="100%" height="100%" fill="black"/>

  <!-- Column Headers (mRNA bin ranges) -->
''']

    # Add column headers (mRNA bin ranges for clusters)
    for col, (low, high) in enumerate(MRNA_BINS):
        x = col * (CELL_WIDTH_MM + GAP_MM) + CELL_WIDTH_MM / 2
        y = HEADER_HEIGHT_MM * 0.75  # Center vertically in header
        if high < 2000:
            label = f"{low}-{high}"
        else:
            label = f"{low}+"
        svg_parts.append(f'  <text x="{x:.2f}" y="{y:.2f}" class="header-label">{label}</text>\n')

    svg_parts.append('\n  <!-- Image Grid (columns = mRNA bins, rows = examples) -->\n')

    # For each column (mRNA bin)
    for col, (low, high) in enumerate(MRNA_BINS):
        bin_images = binned_images.get(col, [])

        # Sample N_ROWS images from this bin, offset by collage_index
        if len(bin_images) >= N_ROWS:
            # Pick different samples for each collage
            start_offset = (collage_index * N_ROWS) % len(bin_images)
            samples = []
            for i in range(N_ROWS):
                idx = (start_offset + i) % len(bin_images)
                samples.append(bin_images[idx])
        elif len(bin_images) > 0:
            # Use what we have, repeat if needed
            samples = bin_images[:N_ROWS]
            while len(samples) < N_ROWS:
                samples.append(samples[-1])
        else:
            samples = [(None, 0, None)] * N_ROWS

        # Add images for this column
        x = col * (CELL_WIDTH_MM + GAP_MM)
        for row, item in enumerate(samples):
            # Unpack - handle both old (path, mrna) and new (path, mrna, contour_path) formats
            if len(item) == 3:
                img_path, mrna, contour_path = item
            else:
                img_path, mrna = item
                contour_path = None

            y = HEADER_HEIGHT_MM + row * (CELL_HEIGHT_MM + GAP_MM)

            if img_path is not None and img_path.exists():
                img_b64 = image_to_base64(str(img_path))
                svg_parts.append(f'''  <image x="{x:.2f}" y="{y:.2f}" width="{CELL_WIDTH_MM:.2f}" height="{CELL_HEIGHT_MM:.2f}"
         xlink:href="data:image/png;base64,{img_b64}"
         preserveAspectRatio="xMidYMid meet"/>
''')

                # Load and draw annotations (spot circle + cluster contours) if available
                if contour_path is not None and contour_path.exists():
                    with open(contour_path, 'r') as f:
                        annotation_data = json.load(f)

                    # Calculate scale factor from original image (pixels) to cell (mm)
                    orig_w = annotation_data['image_width']
                    orig_h = annotation_data['image_height']
                    scale_x = CELL_WIDTH_MM / orig_w
                    scale_y = CELL_HEIGHT_MM / orig_h

                    # Draw spot circle (on left side of cell)
                    if 'spot_circle' in annotation_data:
                        sc = annotation_data['spot_circle']
                        cx = x + sc['cx'] * scale_x
                        cy = y + sc['cy'] * scale_y
                        r = sc['radius'] * min(scale_x, scale_y)
                        svg_parts.append(
                            f'  <circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r:.2f}" fill="none" stroke="white" stroke-width="0.15" opacity="0.9"/>\n'
                        )

                    # Draw each cluster contour as an SVG path
                    for contour in annotation_data.get('contours', []):
                        if len(contour) < 2:
                            continue
                        path_d = f"M {x + contour[0]['x'] * scale_x:.2f} {y + contour[0]['y'] * scale_y:.2f}"
                        for point in contour[1:]:
                            path_d += f" L {x + point['x'] * scale_x:.2f} {y + point['y'] * scale_y:.2f}"
                        path_d += " Z"
                        svg_parts.append(
                            f'  <path d="{path_d}" fill="none" stroke="white" stroke-width="0.15" opacity="0.9"/>\n'
                        )

    svg_parts.append('</svg>\n')

    # Write SVG
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(''.join(svg_parts))


def main():
    """Create binned collages for both channels."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("FIGURE 2 BINNED COLLAGE CREATION")
    print("Columns = mRNA bins (dim to bright), Rows = examples")
    print("="*70)

    print(f"\nDimensions: {TOTAL_WIDTH_MM:.1f}mm x {TOTAL_HEIGHT_MM:.1f}mm")
    print(f"Cell size: {CELL_WIDTH_MM:.2f}mm x {CELL_HEIGHT_MM:.2f}mm")
    print(f"Layout: {N_ROWS} rows x {N_COLS} columns")
    print(f"Excluded slides: {EXCLUDED_SLIDES}")

    print("\nmRNA bins:")
    for i, (low, high) in enumerate(MRNA_BINS):
        print(f"  Column {i+1}: {low}-{high} mRNA")

    for channel in ['green', 'orange']:
        print(f"\n--- Processing {channel.upper()} channel ---")

        # Collect and bin images
        binned = collect_and_bin_images(channel)

        # Print bin statistics
        print("  Images per bin:")
        for i, (low, high) in enumerate(MRNA_BINS):
            count = len(binned.get(i, []))
            print(f"    {low}-{high} mRNA: {count} images")

        # Create multiple collages with different samples
        n_collages = 20
        for i in range(n_collages):
            output_path = OUTPUT_DIR / f"{channel}_binned_collage_{i+1:02d}.svg"
            print(f"  Creating collage {i+1}/{n_collages}...")
            create_binned_collage_svg(binned, output_path, collage_index=i)
            print(f"    -> Saved: {output_path.name}")

    print("\n" + "="*70)
    print("DONE!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*70)


if __name__ == '__main__':
    main()
