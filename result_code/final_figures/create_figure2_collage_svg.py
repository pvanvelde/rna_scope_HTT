#!/usr/bin/env python3
"""
Create SVG collages from spot_vs_cluster_comparisons images for Figure 2.

Layout: 3 rows x 10 columns = 30 images per collage
Images sorted by mRNA count (small clusters on left, large on right)
Creates separate collages for green and orange channels.

Scale bar is added only to the SVG output, not embedded in the raw images.
"""

import os
import re
import sys
import json
from pathlib import Path
import base64
from io import BytesIO
from PIL import Image
import numpy as np

# Input/output directories
INPUT_DIR = Path(__file__).parent / 'output' / 'example_images' / 'figure2'
OUTPUT_DIR = Path(__file__).parent / 'output' / 'figure2_collages'

# Physical parameters for scale bar
PIXEL_SIZE_XY = 0.16  # µm per pixel
SCALE_BAR_UM = 5  # Scale bar length in µm


def extract_mrna_from_filename(filename: str) -> float:
    """Extract mRNA count from filename like 'comparison_01_small_6mrna.png'."""
    match = re.search(r'(\d+)mrna', filename)
    if match:
        return float(match.group(1))
    return 0


def image_to_base64(img_path: str) -> str:
    """Convert image (TIFF or PNG) to base64-encoded PNG string."""
    # Open with PIL and convert to PNG bytes
    with Image.open(img_path) as img:
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode('utf-8')


def collect_comparison_images(channel: str) -> list:
    """
    Collect all comparison images for a channel.
    Returns list of (path, mrna_count, contour_path) tuples sorted by mrna_count.
    Prefers TIFF files (raw, no annotations) over PNG.
    """
    images = []

    for fov_dir in INPUT_DIR.glob('fov_*'):
        channel_dir = fov_dir / channel / 'spot_vs_cluster_comparisons'
        if not channel_dir.exists():
            continue

        # Look for TIFF files first (raw images without annotations)
        for img_path in channel_dir.glob('comparison_*.tif'):
            mrna = extract_mrna_from_filename(img_path.name)
            # Find corresponding contour JSON
            contour_path = img_path.with_suffix('').with_name(img_path.stem + '_contour.json')
            images.append((img_path, mrna, contour_path if contour_path.exists() else None))

        # Fallback to PNG if no TIFFs found
        if len(images) == 0:
            for img_path in channel_dir.glob('comparison_*.png'):
                mrna = extract_mrna_from_filename(img_path.name)
                images.append((img_path, mrna, None))

    # Sort by mRNA count (small to large)
    images.sort(key=lambda x: x[1])

    return images


def create_collage_svg(images: list, output_path: Path,
                       n_rows: int = 3, n_cols: int = 10,
                       cell_width: int = 140, cell_height: int = 80,
                       gap: int = 4, line_thickness: int = 2):
    """
    Create an SVG collage from a list of image paths.

    Args:
        images: List of (path, mrna, contour_path) tuples
        output_path: Where to save the SVG
        n_rows: Number of rows in grid
        n_cols: Number of columns in grid
        cell_width: Width of each cell
        cell_height: Height of each cell
        gap: Gap between cells (thicker for better separation)
        line_thickness: Thickness of divider lines
    """
    n_images = n_rows * n_cols
    if len(images) < n_images:
        print(f"  Warning: Only {len(images)} images, need {n_images}")
        # Pad with None
        images = images + [(None, 0, None)] * (n_images - len(images))

    # Take first n_images
    images = images[:n_images]

    # Calculate total dimensions
    total_width = n_cols * cell_width + (n_cols - 1) * gap
    total_height = n_rows * cell_height + (n_rows - 1) * gap

    # Calculate scale bar dimensions (for last cell)
    # Each cell contains spot (left) + cluster (right), each ~32px wide with 4px gap
    # So cluster panel starts at about cell_width/2
    scale_bar_px = int(SCALE_BAR_UM / PIXEL_SIZE_XY)  # ~31 pixels for 5µm

    # Start SVG
    svg_parts = [f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg"
     xmlns:xlink="http://www.w3.org/1999/xlink"
     width="{total_width}" height="{total_height}"
     viewBox="0 0 {total_width} {total_height}">

  <!-- Background -->
  <rect width="100%" height="100%" fill="black"/>

  <!-- Image Grid -->
''']

    # Add images and contours
    contour_paths_svg = []  # Collect contour SVG elements to add after images

    for idx, item in enumerate(images):
        # Unpack - handle both old (path, mrna) and new (path, mrna, contour_path) formats
        if len(item) == 3:
            img_path, mrna, contour_path = item
        else:
            img_path, mrna = item
            contour_path = None

        row = idx // n_cols
        col = idx % n_cols

        cell_x = col * (cell_width + gap)
        cell_y = row * (cell_height + gap)

        if img_path is not None and img_path.exists():
            img_b64 = image_to_base64(str(img_path))
            svg_parts.append(f'''  <image x="{cell_x}" y="{cell_y}" width="{cell_width}" height="{cell_height}"
         xlink:href="data:image/png;base64,{img_b64}"
         preserveAspectRatio="xMidYMid meet"/>
''')

            # Load and draw annotations (spot circle + cluster contours) if available
            if contour_path is not None and contour_path.exists():
                with open(contour_path, 'r') as f:
                    annotation_data = json.load(f)

                # Calculate scale factor from original image to cell
                orig_w = annotation_data['image_width']
                orig_h = annotation_data['image_height']
                scale_x = cell_width / orig_w
                scale_y = cell_height / orig_h

                # Draw spot circle (on left side of cell)
                if 'spot_circle' in annotation_data:
                    sc = annotation_data['spot_circle']
                    cx = cell_x + sc['cx'] * scale_x
                    cy = cell_y + sc['cy'] * scale_y
                    r = sc['radius'] * min(scale_x, scale_y)  # Scale radius
                    contour_paths_svg.append(
                        f'  <circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="none" stroke="white" stroke-width="1.5" opacity="0.9"/>\n'
                    )

                # Draw each cluster contour as an SVG path (on right side of cell)
                for contour in annotation_data.get('contours', []):
                    if len(contour) < 2:
                        continue
                    # Build SVG path
                    path_d = f"M {cell_x + contour[0]['x'] * scale_x:.1f} {cell_y + contour[0]['y'] * scale_y:.1f}"
                    for point in contour[1:]:
                        path_d += f" L {cell_x + point['x'] * scale_x:.1f} {cell_y + point['y'] * scale_y:.1f}"
                    path_d += " Z"  # Close path

                    contour_paths_svg.append(
                        f'  <path d="{path_d}" fill="none" stroke="white" stroke-width="1.5" opacity="0.9"/>\n'
                    )

    # Add thicker divider lines between cells (vertical)
    svg_parts.append('\n  <!-- Divider lines -->\n')
    for col in range(1, n_cols):
        x = col * (cell_width + gap) - gap / 2
        svg_parts.append(f'''  <line x1="{x}" y1="0" x2="{x}" y2="{total_height}"
        stroke="#333" stroke-width="{line_thickness}"/>
''')

    # Add thicker divider lines between cells (horizontal)
    for row in range(1, n_rows):
        y = row * (cell_height + gap) - gap / 2
        svg_parts.append(f'''  <line x1="0" y1="{y}" x2="{total_width}" y2="{y}"
        stroke="#333" stroke-width="{line_thickness}"/>
''')

    # Add scale bar to the last cell (bottom-right corner of collage)
    last_cell_x = (n_cols - 1) * (cell_width + gap)
    last_cell_y = (n_rows - 1) * (cell_height + gap)

    # Position scale bar at bottom-right of last cell
    scale_bar_x2 = last_cell_x + cell_width - 5
    scale_bar_x1 = scale_bar_x2 - scale_bar_px
    scale_bar_y = last_cell_y + cell_height - 8

    svg_parts.append(f'''
  <!-- Scale bar ({SCALE_BAR_UM} µm) -->
  <line x1="{scale_bar_x1}" y1="{scale_bar_y}" x2="{scale_bar_x2}" y2="{scale_bar_y}"
        stroke="white" stroke-width="3"/>
  <text x="{(scale_bar_x1 + scale_bar_x2) / 2}" y="{scale_bar_y - 4}"
        fill="white" font-family="Arial, sans-serif" font-size="10"
        text-anchor="middle">{SCALE_BAR_UM} µm</text>
''')

    # Add contour paths (drawn on top of images)
    if contour_paths_svg:
        svg_parts.append('\n  <!-- Cluster contours -->\n')
        svg_parts.extend(contour_paths_svg)

    svg_parts.append('</svg>\n')

    # Write SVG
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(''.join(svg_parts))


def main():
    """Create collages for both channels."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("FIGURE 2 COLLAGE CREATION")
    print("3 rows x 10 columns, sorted by cluster size (small to large)")
    print("="*70)

    for channel in ['green', 'orange']:
        print(f"\n--- Processing {channel.upper()} channel ---")

        # Collect all images
        images = collect_comparison_images(channel)
        print(f"  Found {len(images)} comparison images")

        if len(images) == 0:
            continue

        # Print mRNA range
        mrna_values = [m for _, m, _ in images]
        print(f"  mRNA range: {min(mrna_values):.0f} - {max(mrna_values):.0f}")

        # Create multiple collages (30 images each)
        n_per_collage = 30
        n_collages = (len(images) + n_per_collage - 1) // n_per_collage

        for i in range(n_collages):
            start = i * n_per_collage
            end = min(start + n_per_collage, len(images))
            collage_images = images[start:end]

            # Get mRNA range for this collage
            mrna_min = collage_images[0][1]
            mrna_max = collage_images[-1][1]

            output_path = OUTPUT_DIR / f"{channel}_collage_{i+1:02d}_mrna{int(mrna_min)}-{int(mrna_max)}.svg"

            print(f"  Creating collage {i+1}/{n_collages}: {len(collage_images)} images, mRNA {mrna_min:.0f}-{mrna_max:.0f}")
            create_collage_svg(collage_images, output_path)
            print(f"    -> Saved: {output_path.name}")

    print("\n" + "="*70)
    print("DONE!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*70)


if __name__ == '__main__':
    main()
