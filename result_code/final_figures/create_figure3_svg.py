#!/usr/bin/env python3
"""
Create SVG figures for Figure 3 panel selection.

Layout: 1 FOV overview on top, 4 zoom panels below (2x2 grid)
Each FOV generates one SVG file that can be edited in Inkscape/Illustrator.

Images are embedded as base64 PNG for portability.
Includes mRNA/nuc labels for green and orange channels.
"""

import os
import sys
from pathlib import Path
import base64
from io import BytesIO
import numpy as np
import pandas as pd
from PIL import Image
from tifffile import imread
import h5py

# Add parent directories for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from figure2_panels import (
    find_h5_file,
    get_slide_peak_intensity,
    extract_slide_from_fov_key,
    VOXEL_VOLUME_UM3,
)
from results_config import VOXEL_SIZE, EXCLUDED_SLIDES, MEAN_NUCLEAR_VOLUME, EXPERIMENTAL_FIELD
from figure_config import FigureConfig

# Figure dimensions from figure_config
TARGET_HEIGHT_MM = 69.406  # Target height in mm

# Output directory for SVGs
INPUT_DIR = Path(__file__).parent / 'output' / 'example_images' / 'figure3'
OUTPUT_DIR = Path(__file__).parent / 'output' / 'figure3_svg_panels'


def get_fov_expression_data_from_json(fov_dir: Path, fov_name: str) -> dict:
    """Get mRNA/nucleus values, dynamic range, and zoom coordinates from saved JSON files."""
    result = {
        'green': None, 'orange': None, 'n_nuclei': None, 'metadata': {},
        'dynamic_range': None, 'image_width': None, 'image_height': None,
        'zooms': None
    }

    # Look for metadata JSON file
    json_path = fov_dir / f"{fov_name}_metadata.json"
    if json_path.exists():
        import json
        with open(json_path, 'r') as f:
            data = json.load(f)
        result['green'] = data.get('green_mrna_per_nucleus')
        result['orange'] = data.get('orange_mrna_per_nucleus')
        result['n_nuclei'] = data.get('n_nuclei')
        result['dynamic_range'] = data.get('dynamic_range')
        result['image_width'] = data.get('image_width')
        result['image_height'] = data.get('image_height')
        result['metadata'] = {
            'mouse_model': data.get('mouse_model'),
            'age': data.get('age'),
            'region': data.get('region'),
        }

    # Look for zoom coordinates JSON file
    zoom_json_path = fov_dir / f"{fov_name}_zoom_coords.json"
    if zoom_json_path.exists():
        import json
        with open(zoom_json_path, 'r') as f:
            zoom_data = json.load(f)
        result['zooms'] = zoom_data.get('zooms', [])

    return result


def tiff_to_png_base64(tiff_path: str, max_size: int = None) -> str:
    """Convert TIFF to base64-encoded PNG string."""
    img_array = imread(tiff_path)

    # Handle different array shapes
    if img_array.ndim == 3 and img_array.shape[2] in [3, 4]:
        img = Image.fromarray(img_array.astype(np.uint8))
    elif img_array.ndim == 2:
        img = Image.fromarray(img_array.astype(np.uint8))
    else:
        img = Image.fromarray(img_array.astype(np.uint8))

    # Resize if needed
    if max_size and max(img.size) > max_size:
        ratio = max_size / max(img.size)
        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
        img = img.resize(new_size, Image.LANCZOS)

    # Convert to PNG bytes
    buffer = BytesIO()
    img.save(buffer, format='PNG', optimize=True)
    buffer.seek(0)

    return base64.b64encode(buffer.read()).decode('utf-8')


def fov_name_to_key(fov_name: str) -> str:
    """Convert FOV filename back to H5 key format."""
    # Reverse the transformations done when saving:
    # Replace _ with -- except for the slide name separator
    parts = fov_name.split('_')
    if len(parts) >= 4:
        # Format: {experiment}_{slide}_{region}_{coords}[_n000X]
        # Original: {experiment}--{slide}--{region}--{coords}[--n000X]
        experiment = parts[0]
        slide = parts[1]
        region = parts[2]
        coords = parts[3]

        # Check if there's a sequence number
        if len(parts) > 4 and parts[4].startswith('n'):
            seq = parts[4]
            return f"{experiment}--{slide}--{region}--{coords}--{seq}"
        else:
            return f"{experiment}--{slide}--{region}--{coords}"
    return fov_name.replace('_', '--')


def create_svg_panel(fov_name: str, fov_dir: Path, output_path: Path,
                     channel: str = 'all', n_zooms: int = 4,
                     expression_data: dict = None, use_gamma: bool = False):
    """
    Create an SVG with 1 FOV overview + n zoom panels.

    Args:
        fov_name: Base name of the FOV
        fov_dir: Directory containing the FOV images
        output_path: Output SVG path
        channel: 'all', 'green', or 'orange'
        n_zooms: Number of zoom panels to include (1-5)
        expression_data: Dict with 'green' and 'orange' mRNA/nuc values, dynamic range, zoom coords
        use_gamma: If True, use gamma-corrected images (*_gamma.tif)
    """
    cfg = FigureConfig

    # Physical scale bar parameters (from figure2_panels)
    PIXEL_SIZE_XY = 0.16  # µm per pixel
    SCALE_BAR_OVERVIEW_UM = 20  # µm for overview
    SCALE_BAR_ZOOM_UM = 10  # µm for zooms

    # Suffix for gamma images
    gamma_suffix = "_gamma" if use_gamma else ""

    # Find the FOV image
    fov_image = fov_dir / f"{fov_name}_{channel}{gamma_suffix}.tif"
    if not fov_image.exists():
        print(f"  WARNING: FOV image not found: {fov_image}")
        return False

    # Find zoom images
    zoom_dir = fov_dir / f"{fov_name}_zooms"
    if not zoom_dir.exists():
        print(f"  WARNING: Zoom directory not found: {zoom_dir}")
        return False

    zoom_images = []
    for i in range(1, n_zooms + 1):
        zoom_path = zoom_dir / f"zoom{i}_{channel}{gamma_suffix}.tif"
        if zoom_path.exists():
            zoom_images.append(zoom_path)

    if len(zoom_images) == 0:
        return False

    # Layout parameters (in viewBox units - internal coordinate system)
    fov_width = 600
    fov_height = 600
    zoom_size = 280
    gap = 15
    margin = 20
    label_height = 60  # Space for mRNA labels

    # Calculate total dimensions in viewBox units
    total_width_vb = margin * 2 + fov_width
    zoom_row_width = 2 * zoom_size + gap
    if zoom_row_width > fov_width:
        total_width_vb = margin * 2 + zoom_row_width

    n_zoom_rows = (len(zoom_images) + 1) // 2
    total_height_vb = margin * 2 + label_height + fov_height + gap + n_zoom_rows * zoom_size + (n_zoom_rows - 1) * gap

    # Calculate physical dimensions in mm (height fixed, width scaled proportionally)
    height_mm = TARGET_HEIGHT_MM
    width_mm = height_mm * (total_width_vb / total_height_vb)

    # Font sizes - scale from figure_config (points to viewBox units)
    scale_factor = total_height_vb / TARGET_HEIGHT_MM  # viewBox units per mm
    font_size_label = cfg.FONT_SIZE_AXIS_LABEL * scale_factor * 0.35 * 1.5
    font_size_small = cfg.FONT_SIZE_ANNOTATION * scale_factor * 0.35 * 1.2
    font_size_scalebar = cfg.FONT_SIZE_ANNOTATION * scale_factor * 0.35 * 0.9

    # Colors from figure_config
    color_green = cfg.COLOR_Q111_MHTT1A
    color_orange = cfg.COLOR_Q111_FULL
    color_white = cfg.COLOR_WHITE

    # Font family from figure_config
    font_family = ', '.join(cfg.FONT_SANS_SERIF)

    # Convert images to base64
    print(f"  Converting images...")
    fov_b64 = tiff_to_png_base64(str(fov_image), max_size=1200)

    zoom_b64_list = []
    for zp in zoom_images:
        zoom_b64_list.append(tiff_to_png_base64(str(zp), max_size=600))

    # Prepare mRNA labels
    green_mrna = expression_data.get('green') if expression_data else None
    orange_mrna = expression_data.get('orange') if expression_data else None

    green_text = f"{green_mrna:.1f} mRNA/nuc" if green_mrna is not None else ""
    orange_text = f"{orange_mrna:.1f} mRNA/nuc" if orange_mrna is not None else ""

    # Get zoom coordinates for drawing boxes on overview
    zoom_coords = expression_data.get('zooms', []) if expression_data else []
    img_width = expression_data.get('image_width') if expression_data else None
    img_height = expression_data.get('image_height') if expression_data else None

    # Calculate scale from image pixels to FOV viewBox
    if img_width and img_height:
        scale_x = fov_width / img_width
        scale_y = fov_height / img_height
    else:
        scale_x = scale_y = 1.0

    # Scale bar length in viewBox units (overview)
    overview_scalebar_px = SCALE_BAR_OVERVIEW_UM / PIXEL_SIZE_XY  # pixels
    overview_scalebar_vb = overview_scalebar_px * scale_x  # viewBox units

    # Create SVG with mm dimensions
    svg_parts = []
    svg_parts.append(f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg"
     xmlns:xlink="http://www.w3.org/1999/xlink"
     width="{width_mm:.3f}mm" height="{height_mm:.3f}mm"
     viewBox="0 0 {total_width_vb} {total_height_vb}">

  <defs>
    <style>
      .label-green {{ font-family: {font_family}; font-size: {font_size_label:.1f}px; font-weight: bold; fill: {color_green}; }}
      .label-orange {{ font-family: {font_family}; font-size: {font_size_label:.1f}px; font-weight: bold; fill: {color_orange}; }}
      .scale-text {{ font-family: {font_family}; font-size: {font_size_scalebar:.1f}px; fill: {color_white}; text-anchor: middle; }}
      .range-text {{ font-family: {font_family}; font-size: {font_size_small:.1f}px; fill: {color_white}; }}
    </style>
  </defs>

  <!-- Background -->
  <rect width="100%" height="100%" fill="black"/>

  <!-- mRNA Labels -->
  <g id="mrna_labels">
    <text x="{margin + 10}" y="{margin + 28}" class="label-green">{green_text}</text>
    <text x="{margin + 10}" y="{margin + 55}" class="label-orange">{orange_text}</text>
  </g>

  <!-- FOV Overview -->
  <g id="fov_overview">
    <image x="{margin}" y="{margin + label_height}" width="{fov_width}" height="{fov_height}"
           xlink:href="data:image/png;base64,{fov_b64}"
           preserveAspectRatio="xMidYMid meet"/>
    <rect x="{margin}" y="{margin + label_height}" width="{fov_width}" height="{fov_height}"
          fill="none" stroke="white" stroke-width="2"/>
''')

    # Add zoom boxes on overview image
    if zoom_coords and len(zoom_coords) > 0:
        svg_parts.append('    <!-- Zoom region boxes -->\n')
        # Define distinct colors for each zoom box
        box_colors = ['#FFFFFF', '#FFFF00', '#00FFFF', '#FF00FF', '#FFA500']
        for i, zoom in enumerate(zoom_coords[:len(zoom_images)]):
            # Transform coordinates from image pixels to viewBox
            box_x1 = margin + zoom['x1'] * scale_x
            box_y1 = margin + label_height + zoom['y1'] * scale_y
            box_w = (zoom['x2'] - zoom['x1']) * scale_x
            box_h = (zoom['y2'] - zoom['y1']) * scale_y
            box_color = box_colors[i % len(box_colors)]
            svg_parts.append(f'''    <rect x="{box_x1:.1f}" y="{box_y1:.1f}" width="{box_w:.1f}" height="{box_h:.1f}"
          fill="none" stroke="{box_color}" stroke-width="2" stroke-dasharray="4,2"/>
''')

    # Add scale bar to overview (bottom-right)
    sb_x2 = margin + fov_width - 10
    sb_x1 = sb_x2 - overview_scalebar_vb
    sb_y = margin + label_height + fov_height - 15
    svg_parts.append(f'''    <!-- Scale bar ({SCALE_BAR_OVERVIEW_UM} µm) -->
    <line x1="{sb_x1:.1f}" y1="{sb_y:.1f}" x2="{sb_x2:.1f}" y2="{sb_y:.1f}"
          stroke="white" stroke-width="3"/>
    <text x="{(sb_x1 + sb_x2) / 2:.1f}" y="{sb_y - 5:.1f}" class="scale-text">{SCALE_BAR_OVERVIEW_UM} µm</text>
''')

    svg_parts.append('  </g>\n\n')

    # Add zoom panels in 2x2 grid
    svg_parts.append('  <!-- Zoom Panels -->\n  <g id="zoom_panels">\n')
    zoom_start_y = margin + label_height + fov_height + gap

    # Scale bar for zooms (calculate based on zoom size in pixels)
    zoom_px_size = 256  # ZOOM_SIZE from figure3_panels
    zoom_scale = zoom_size / zoom_px_size
    zoom_scalebar_px = SCALE_BAR_ZOOM_UM / PIXEL_SIZE_XY
    zoom_scalebar_vb = zoom_scalebar_px * zoom_scale

    box_colors = ['#FFFFFF', '#FFFF00', '#00FFFF', '#FF00FF', '#FFA500']
    for i, zb64 in enumerate(zoom_b64_list):
        row = i // 2
        col = i % 2
        x = margin + col * (zoom_size + gap)
        y = zoom_start_y + row * (zoom_size + gap)
        box_color = box_colors[i % len(box_colors)]

        # Scale bar position in this zoom panel
        zs_x2 = x + zoom_size - 8
        zs_x1 = zs_x2 - zoom_scalebar_vb
        zs_y = y + zoom_size - 12

        svg_parts.append(f'''    <g id="zoom_{i+1}">
      <image x="{x}" y="{y}" width="{zoom_size}" height="{zoom_size}"
             xlink:href="data:image/png;base64,{zb64}"
             preserveAspectRatio="xMidYMid meet"/>
      <rect x="{x}" y="{y}" width="{zoom_size}" height="{zoom_size}"
            fill="none" stroke="{box_color}" stroke-width="2"/>
      <!-- Scale bar ({SCALE_BAR_ZOOM_UM} µm) -->
      <line x1="{zs_x1:.1f}" y1="{zs_y:.1f}" x2="{zs_x2:.1f}" y2="{zs_y:.1f}"
            stroke="white" stroke-width="2"/>
      <text x="{(zs_x1 + zs_x2) / 2:.1f}" y="{zs_y - 4:.1f}" class="scale-text">{SCALE_BAR_ZOOM_UM} µm</text>
    </g>
''')

    svg_parts.append('  </g>\n')

    # Add dynamic range text (bottom of SVG)
    dynamic_range = expression_data.get('dynamic_range') if expression_data else None
    if dynamic_range:
        dr_y = total_height_vb - margin / 2
        dapi_range = dynamic_range.get('dapi_range', (0, 0))
        green_range = dynamic_range.get('green_range', (0, 0))
        orange_range = dynamic_range.get('orange_range', (0, 0))
        gamma_value = dynamic_range.get('gamma', 1.0)

        # Include gamma info if using gamma correction
        if use_gamma:
            gamma_text = f" | Gamma: {gamma_value:.1f}"
        else:
            gamma_text = " | Linear"

        svg_parts.append(f'''
  <!-- Dynamic Range Info -->
  <g id="dynamic_range">
    <text x="{margin}" y="{dr_y:.1f}" class="range-text">
      DAPI [{dapi_range[0]:.0f}-{dapi_range[1]:.0f}], Green [{green_range[0]:.0f}-{green_range[1]:.0f}], Orange [{orange_range[0]:.0f}-{orange_range[1]:.0f}]{gamma_text}
    </text>
  </g>
''')

    svg_parts.append('</svg>\n')

    # Write SVG
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(''.join(svg_parts))

    print(f"  Dimensions: {width_mm:.2f} x {height_mm:.2f} mm")
    return True


def check_fov_probe_set(h5_file, fov_key: str) -> bool:
    """Check if FOV belongs to experimental probe set."""
    if fov_key not in h5_file:
        return False

    fov = h5_file[fov_key]
    if 'metadata_sample' not in fov or 'Probe-Set' not in fov['metadata_sample']:
        return False

    ps = fov['metadata_sample']['Probe-Set'][()]
    if isinstance(ps, np.ndarray) and len(ps) > 0:
        ps = ps[0]
    if isinstance(ps, bytes):
        ps = ps.decode()

    return ps == EXPERIMENTAL_FIELD


def process_all_fovs(channel: str = 'all'):
    """Process all FOVs and create SVG panels."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load H5 file path
    h5_path = find_h5_file()
    print(f"Using H5 file: {h5_path}")

    # Open H5 file for probe set checking
    with h5py.File(h5_path, 'r') as h5_file:
        # Process each expression level
        for level in ['low', 'medium', 'high']:
            level_dir = INPUT_DIR / level
            if not level_dir.exists():
                print(f"Skipping {level} - directory not found")
                continue

            print(f"\n{'='*60}")
            print(f"Processing {level.upper()} expression FOVs")
            print('='*60)

            # Find all FOV names
            fov_names = set()
            for f in level_dir.glob('*_all.tif'):
                name = f.stem.replace('_all', '')
                fov_names.add(name)

            print(f"Found {len(fov_names)} FOVs")

            # Create output subdirectory
            level_output = OUTPUT_DIR / level
            level_output.mkdir(parents=True, exist_ok=True)

            for fov_name in sorted(fov_names):
                # Check if slide is excluded
                fov_key = fov_name_to_key(fov_name)
                slide = extract_slide_from_fov_key(fov_key)
                if slide in EXCLUDED_SLIDES:
                    print(f"\n{fov_name[:50]}... SKIPPED (excluded slide: {slide})")
                    continue

                # Check probe set - only process experimental FOVs
                if not check_fov_probe_set(h5_file, fov_key):
                    print(f"\n{fov_name[:50]}... SKIPPED (not experimental probe set)")
                    continue

                print(f"\n{fov_name[:50]}...")

                # Get expression data from JSON metadata file
                expression_data = get_fov_expression_data_from_json(level_dir, fov_name)

                if expression_data.get('green') is not None:
                    print(f"  mHTT1a: {expression_data['green']:.2f}, fl-mHTT: {expression_data.get('orange', 0):.2f} mRNA/nuc")
                else:
                    print(f"  WARNING: No metadata JSON found for {fov_name}")

                # Create LINEAR SVG
                output_path = level_output / f"{fov_name}_{channel}_panel.svg"
                success = create_svg_panel(fov_name, level_dir, output_path,
                                           channel=channel, n_zooms=4,
                                           expression_data=expression_data,
                                           use_gamma=False)
                if success:
                    print(f"  -> Saved: {output_path.name}")

                # Create GAMMA SVG
                output_path_gamma = level_output / f"{fov_name}_{channel}_gamma_panel.svg"
                success_gamma = create_svg_panel(fov_name, level_dir, output_path_gamma,
                                                  channel=channel, n_zooms=4,
                                                  expression_data=expression_data,
                                                  use_gamma=True)
                if success_gamma:
                    print(f"  -> Saved: {output_path_gamma.name}")


def main():
    """Main entry point."""
    import argparse
    parser = argparse.ArgumentParser(description='Create SVG panels for Figure 3')
    parser.add_argument('--channel', choices=['all', 'green', 'orange'],
                        default='all', help='Channel to use for images')
    args = parser.parse_args()

    print("="*70)
    print("FIGURE 3 SVG PANEL CREATION")
    print("Layout: 1 FOV overview + 4 zoom panels + mRNA labels")
    print("="*70)

    process_all_fovs(channel=args.channel)

    print("\n" + "="*70)
    print("DONE!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*70)


if __name__ == '__main__':
    main()
