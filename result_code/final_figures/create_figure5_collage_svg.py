#!/usr/bin/env python3
"""
Create SVG collages for Figure 5 - Extreme vs Normal Q111 FOVs.

Layout: 2 rows x 3 columns (3 animals)
- Row 1: Extreme FOVs (high expression)
- Row 2: Normal FOVs (low expression)
Each cell: FOV overview + 2 zoom panels

Width: 130mm, height scaled accordingly.
"""

import os
import re
import sys
from pathlib import Path
import base64
import json
import numpy as np
from collections import defaultdict
from io import BytesIO
from PIL import Image

# Add parent directories for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from figure_config import FigureConfig
from results_config import (
    EXCLUDED_SLIDES,
    SLIDE_LABEL_MAP_Q111,
)

# Import from figure2_panels (reuse visualization functions)
from figure2_panels import (
    find_h5_file,
    get_npz_path_from_h5,
    load_npz_image,
    create_mip,
    normalize_to_8bit,
    create_colored_image,
    draw_scale_bar,
    extract_slide_from_fov_key,
    get_slide_peak_intensity,
    CHANNEL_MAP,
    PIXEL_SIZE_XY,
    SCALE_BAR_OVERVIEW,
    SCALE_BAR_ZOOM,
    ZOOM_SIZE,
)

# Import from figure3_panels
from figure3_panels import (
    load_fov_data_from_h5,
    select_zoom_regions,
    create_composite_three_channel,
)

# Import from figure5_panels
from figure5_panels import (
    get_all_fov_expression_data,
    select_paired_fovs,
)

# Output directory
OUTPUT_DIR = Path(__file__).parent / 'output' / 'figure5_collages'

# Gamma correction value for enhanced visibility of dim spots
GAMMA_VALUE = 2.2

# ══════════════════════════════════════════════════════════════════════════════
# LAYOUT DIMENSIONS (in mm)
# ══════════════════════════════════════════════════════════════════════════════
TOTAL_WIDTH_MM = 130.0        # Max width
N_ANIMALS = 3                 # Number of animals (columns)
N_ROWS = 2                    # Extreme and Normal rows
GAP_MM = 1.0                  # Gap between cells
HEADER_HEIGHT_MM = 5.0        # Height for column headers (Animal #1, etc.)
ROW_LABEL_WIDTH_MM = 8.0      # Width for row labels (Extreme, Normal)

# Each cell contains: 1 FOV + 2 zooms stacked vertically next to it
# Calculate cell dimensions
CONTENT_WIDTH_MM = TOTAL_WIDTH_MM - ROW_LABEL_WIDTH_MM - (N_ANIMALS - 1) * GAP_MM
CELL_WIDTH_MM = CONTENT_WIDTH_MM / N_ANIMALS
# FOV takes ~55% of cell width, zooms take ~45%
FOV_WIDTH_MM = CELL_WIDTH_MM * 0.54
ZOOM_WIDTH_MM = CELL_WIDTH_MM - FOV_WIDTH_MM - GAP_MM * 0.2  # Bigger zooms
ZOOM_GAP_MM = 0.3             # Small gap between zooms
# Height = FOV height (square aspect ratio)
CELL_HEIGHT_MM = FOV_WIDTH_MM
TOTAL_HEIGHT_MM = HEADER_HEIGHT_MM + N_ROWS * CELL_HEIGHT_MM + (N_ROWS - 1) * GAP_MM


def get_animal_label(slide: str) -> str:
    """Get animal label from slide name using SLIDE_LABEL_MAP_Q111."""
    # Return ID directly (e.g., "#1", "#2") without "Animal" prefix
    return SLIDE_LABEL_MAP_Q111.get(slide, slide)


def load_fov_mips(h5_path: str, fov_key: str) -> tuple:
    """
    Load raw MIP data for a FOV without normalization.

    Returns:
        Tuple of (dapi_mip, green_mip, orange_mip, cluster_coms) or (None, None, None, None) on failure
    """
    npz_path = get_npz_path_from_h5(fov_key, h5_path)
    if npz_path is None:
        return None, None, None, None

    try:
        image_4d, _ = load_npz_image(npz_path)
    except Exception as e:
        print(f"  Could not load NPZ: {e}")
        return None, None, None, None

    # Extract channels and create MIPs
    dapi_mip = create_mip(image_4d[CHANNEL_MAP['blue']])
    green_mip = create_mip(image_4d[CHANNEL_MAP['green']])
    orange_mip = create_mip(image_4d[CHANNEL_MAP['orange']])

    # Load cluster data for zoom selection
    green_data = load_fov_data_from_h5(h5_path, fov_key, 'green')
    orange_data = load_fov_data_from_h5(h5_path, fov_key, 'orange')

    green_coms = green_data.get('cluster_coms', np.array([]))
    orange_coms = orange_data.get('cluster_coms', np.array([]))

    if len(green_coms) > 0 and len(orange_coms) > 0:
        all_coms = np.vstack([green_coms, orange_coms])
    elif len(green_coms) > 0:
        all_coms = green_coms
    elif len(orange_coms) > 0:
        all_coms = orange_coms
    else:
        all_coms = np.array([])

    return dapi_mip, green_mip, orange_mip, all_coms


def create_composite_with_shared_range(dapi_mip: np.ndarray, green_mip: np.ndarray, orange_mip: np.ndarray,
                                        green_range: tuple, orange_range: tuple,
                                        dapi_pmin: float = 10, dapi_pmax: float = 99,
                                        gamma: float = 1.0) -> dict:
    """
    Create composite image using shared dynamic range for green/orange channels.

    Args:
        dapi_mip: DAPI channel MIP
        green_mip: Green channel MIP
        orange_mip: Orange channel MIP
        green_range: (min, max) for green channel normalization
        orange_range: (min, max) for orange channel normalization
        dapi_pmin/pmax: Percentiles for DAPI (computed per-image)
        gamma: Gamma correction value (1.0 = linear, >1 brightens dim values)

    Returns:
        dict with 'composite', 'dynamic_range', 'image_width', 'image_height'
    """
    # Normalize DAPI per-image (background varies)
    dapi_low = np.percentile(dapi_mip, dapi_pmin)
    dapi_high = np.percentile(dapi_mip, dapi_pmax)
    dapi_norm = np.clip((dapi_mip.astype(np.float32) - dapi_low) / (dapi_high - dapi_low + 1e-10), 0, 1)

    # Normalize green/orange with shared range
    green_low, green_high = green_range
    orange_low, orange_high = orange_range

    green_norm = np.clip((green_mip.astype(np.float32) - green_low) / (green_high - green_low + 1e-10), 0, 1)
    orange_norm = np.clip((orange_mip.astype(np.float32) - orange_low) / (orange_high - orange_low + 1e-10), 0, 1)

    # Apply gamma correction to signal channels (not DAPI)
    if gamma != 1.0:
        inv_gamma = 1.0 / gamma
        green_norm = np.power(green_norm, inv_gamma)
        orange_norm = np.power(orange_norm, inv_gamma)

    # Create RGB composite in float first
    # Orange color #f39c12 = RGB(243, 156, 18) ≈ (0.95, 0.61, 0.07)
    # Use 0.65 for green component to get proper orange (not red)
    composite = np.zeros((*dapi_mip.shape, 3), dtype=np.float32)
    composite[:, :, 0] = orange_norm                      # Red component of orange
    composite[:, :, 1] = green_norm + orange_norm * 0.65  # Green + orange tint (0.65 matches #f39c12)
    composite[:, :, 2] = dapi_norm                        # Blue = DAPI

    # Clip to prevent overflow and convert to 8-bit
    composite = np.clip(composite, 0, 1)
    composite_8bit = (composite * 255).astype(np.uint8)

    return {
        'composite': composite_8bit,
        'dynamic_range': {
            'dapi_range': (float(dapi_low), float(dapi_high)),
            'green_range': (float(green_low), float(green_high)),
            'orange_range': (float(orange_low), float(orange_high)),
            'gamma': gamma,
        },
        'image_width': dapi_mip.shape[1],
        'image_height': dapi_mip.shape[0],
    }


def create_fov_image_with_zooms(h5_path: str, fov_key: str, n_zooms: int = 2,
                                 green_range: tuple = None, orange_range: tuple = None,
                                 gamma: float = 1.0) -> dict:
    """
    Create FOV composite image and zoom images.

    Args:
        h5_path: Path to H5 file
        fov_key: FOV key
        n_zooms: Number of zoom regions
        green_range: Optional (min, max) for green channel normalization (shared across FOVs)
        orange_range: Optional (min, max) for orange channel normalization (shared across FOVs)
        gamma: Gamma correction value (1.0 = linear)

    Returns:
        dict with 'fov_array', 'zoom_arrays', 'zoom_coords', 'dynamic_range', 'image_width', 'image_height'
        or None on failure
    """
    # Load raw MIPs
    dapi_mip, green_mip, orange_mip, all_coms = load_fov_mips(h5_path, fov_key)
    if dapi_mip is None:
        return None

    # Calculate per-image range if not provided (fallback to percentile-based)
    if green_range is None:
        green_range = (np.percentile(green_mip, 50), np.percentile(green_mip, 99.9))
    if orange_range is None:
        orange_range = (np.percentile(orange_mip, 50), np.percentile(orange_mip, 99.9))

    # Create composite image with shared range (no scale bar - will add in SVG)
    composite_result = create_composite_with_shared_range(
        dapi_mip, green_mip, orange_mip, green_range, orange_range, gamma=gamma
    )
    fov_array = composite_result['composite']  # No scale bar - added in SVG
    dynamic_range = composite_result['dynamic_range']
    img_width = composite_result['image_width']
    img_height = composite_result['image_height']

    # Select zoom regions
    zoom_centers = select_zoom_regions(
        green_mip.shape, all_coms, n_zooms=n_zooms,
        zoom_size=ZOOM_SIZE, min_distance=300
    )

    zoom_arrays = []
    zoom_coords = []
    half = ZOOM_SIZE // 2
    h, w = green_mip.shape

    for cy, cx in zoom_centers:
        y1 = max(0, cy - half)
        y2 = min(h, cy + half)
        x1 = max(0, cx - half)
        x2 = min(w, cx + half)

        # Store zoom coordinates for drawing boxes
        zoom_coords.append({
            'x1': x1,
            'y1': y1,
            'x2': x2,
            'y2': y2,
        })

        # Extract zoom regions
        dapi_zoom = dapi_mip[y1:y2, x1:x2]
        green_zoom = green_mip[y1:y2, x1:x2]
        orange_zoom = orange_mip[y1:y2, x1:x2]

        # Create composite zoom with shared range (no scale bar)
        zoom_result = create_composite_with_shared_range(
            dapi_zoom, green_zoom, orange_zoom, green_range, orange_range, gamma=gamma
        )
        zoom_composite = zoom_result['composite']

        # Pad if necessary
        if zoom_composite.shape[0] < ZOOM_SIZE or zoom_composite.shape[1] < ZOOM_SIZE:
            padded = np.zeros((ZOOM_SIZE, ZOOM_SIZE, 3), dtype=zoom_composite.dtype)
            padded[:zoom_composite.shape[0], :zoom_composite.shape[1]] = zoom_composite
            zoom_composite = padded

        zoom_arrays.append(zoom_composite)

    return {
        'fov_array': fov_array,
        'zoom_arrays': zoom_arrays,
        'zoom_coords': zoom_coords,
        'dynamic_range': dynamic_range,
        'image_width': img_width,
        'image_height': img_height,
    }


def array_to_base64_png(arr: np.ndarray, max_size: int = None) -> str:
    """
    Convert numpy array to base64-encoded PNG.
    Matches figure4 approach for consistency.

    Args:
        arr: Numpy array (RGB image)
        max_size: Maximum dimension (width or height)

    Returns:
        Base64 encoded PNG string
    """
    if arr is None:
        return ""

    img = Image.fromarray(arr.astype(np.uint8))

    if max_size and max(img.size) > max_size:
        ratio = max_size / max(img.size)
        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
        img = img.resize(new_size, Image.LANCZOS)

    buffer = BytesIO()
    img.save(buffer, format='PNG', optimize=True)
    buffer.seek(0)

    return base64.b64encode(buffer.read()).decode('utf-8')


def create_figure5_collage_svg(paired_data: list, output_path: Path, use_gamma: bool = False):
    """
    Create SVG collage for Figure 5.

    Args:
        paired_data: List of dicts with keys:
            - animal_label: str
            - extreme: dict with fov_array, zoom_arrays, zoom_coords, dynamic_range, image_width, image_height, mrna_green, mrna_orange
            - normal: dict with same keys as extreme
        output_path: Where to save the SVG
        use_gamma: If True, show gamma value in dynamic range text
    """
    cfg = FigureConfig
    font_family = ', '.join(cfg.FONT_SANS_SERIF)

    # Physical scale parameters
    SCALE_BAR_FOV_UM = 20  # µm for FOV overview
    SCALE_BAR_ZOOM_UM = 10  # µm for zooms

    # Zoom box colors
    box_colors = ['#FFFFFF', '#FFFF00']  # White for zoom1, yellow for zoom2

    # Add space for dynamic range text at bottom
    dr_height_mm = 3.0
    total_height_with_dr = TOTAL_HEIGHT_MM + dr_height_mm

    # Font sizes in mm (smaller, matching figure2 style)
    header_font_mm = cfg.FONT_SIZE_ANNOTATION * 0.28
    label_font_mm = cfg.FONT_SIZE_ANNOTATION * 0.26
    mrna_font_mm = cfg.FONT_SIZE_ANNOTATION * 0.22
    scale_font_mm = cfg.FONT_SIZE_ANNOTATION * 0.18
    dr_font_mm = cfg.FONT_SIZE_ANNOTATION * 0.20

    svg_parts = [f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg"
     xmlns:xlink="http://www.w3.org/1999/xlink"
     width="{TOTAL_WIDTH_MM}mm" height="{total_height_with_dr:.2f}mm"
     viewBox="0 0 {TOTAL_WIDTH_MM} {total_height_with_dr:.2f}">

  <defs>
    <style>
      .header-label {{ font-family: {font_family}; font-size: {header_font_mm:.2f}mm; fill: {cfg.COLOR_WHITE}; text-anchor: middle; }}
      .row-label {{ font-family: {font_family}; font-size: {label_font_mm:.2f}mm; fill: {cfg.COLOR_WHITE}; text-anchor: middle; }}
      .mrna-green {{ font-family: {font_family}; font-size: {mrna_font_mm:.2f}mm; fill: {cfg.COLOR_Q111_MHTT1A}; text-anchor: start; }}
      .mrna-orange {{ font-family: {font_family}; font-size: {mrna_font_mm:.2f}mm; fill: {cfg.COLOR_Q111_FULL}; text-anchor: start; }}
      .scale-text {{ font-family: {font_family}; font-size: {scale_font_mm:.2f}mm; fill: {cfg.COLOR_WHITE}; text-anchor: middle; }}
      .range-text {{ font-family: {font_family}; font-size: {dr_font_mm:.2f}mm; fill: {cfg.COLOR_WHITE}; }}
      .range-text-small {{ font-family: {font_family}; font-size: {scale_font_mm:.2f}mm; text-anchor: start; }}
    </style>
  </defs>

  <!-- Background -->
  <rect width="100%" height="100%" fill="black"/>

  <!-- Column Headers (Animal labels) -->
''']

    # Add column headers
    for col, data in enumerate(paired_data):
        x = ROW_LABEL_WIDTH_MM + col * (CELL_WIDTH_MM + GAP_MM) + CELL_WIDTH_MM / 2
        y = HEADER_HEIGHT_MM * 0.7
        svg_parts.append(f'  <text x="{x:.2f}" y="{y:.2f}" class="header-label">{data["animal_label"]}</text>\n')

    # Add row labels
    row_labels = ['Extreme', 'Normal']
    for row, label in enumerate(row_labels):
        x = ROW_LABEL_WIDTH_MM / 2
        y = HEADER_HEIGHT_MM + row * (CELL_HEIGHT_MM + GAP_MM) + CELL_HEIGHT_MM / 2
        # Rotate text for vertical row label
        svg_parts.append(f'  <text x="{x:.2f}" y="{y:.2f}" class="row-label" transform="rotate(-90 {x:.2f} {y:.2f})">{label}</text>\n')

    svg_parts.append('\n  <!-- Image Grid -->\n')

    # Track first FOV's dynamic range for bottom text
    first_dynamic_range = None

    # Add images
    for col, data in enumerate(paired_data):
        cell_x = ROW_LABEL_WIDTH_MM + col * (CELL_WIDTH_MM + GAP_MM)

        for row, category in enumerate(['extreme', 'normal']):
            cell_y = HEADER_HEIGHT_MM + row * (CELL_HEIGHT_MM + GAP_MM)

            cat_data = data[category]
            if cat_data is None:
                continue

            # Store first dynamic range
            if first_dynamic_range is None and cat_data.get('dynamic_range'):
                first_dynamic_range = cat_data['dynamic_range']

            # Get image dimensions for scaling zoom boxes
            img_width = cat_data.get('image_width', 512)
            img_height = cat_data.get('image_height', 512)
            scale_x = FOV_WIDTH_MM / img_width
            scale_y = CELL_HEIGHT_MM / img_height

            # FOV image (high quality PNG, same as figure4)
            if cat_data.get('fov_array') is not None:
                fov_b64 = array_to_base64_png(cat_data['fov_array'], max_size=800)
                svg_parts.append(f'''  <image x="{cell_x:.2f}" y="{cell_y:.2f}" width="{FOV_WIDTH_MM:.2f}" height="{CELL_HEIGHT_MM:.2f}"
         xlink:href="data:image/png;base64,{fov_b64}"
         preserveAspectRatio="xMidYMid meet"/>
  <rect x="{cell_x:.2f}" y="{cell_y:.2f}" width="{FOV_WIDTH_MM:.2f}" height="{CELL_HEIGHT_MM:.2f}"
        fill="none" stroke="white" stroke-width="0.3"/>
''')

                # Draw zoom boxes on FOV
                zoom_coords = cat_data.get('zoom_coords', [])
                for z_idx, zc in enumerate(zoom_coords[:2]):
                    box_x1 = cell_x + zc['x1'] * scale_x
                    box_y1 = cell_y + zc['y1'] * scale_y
                    box_w = (zc['x2'] - zc['x1']) * scale_x
                    box_h = (zc['y2'] - zc['y1']) * scale_y
                    box_color = box_colors[z_idx % len(box_colors)]
                    svg_parts.append(f'''  <rect x="{box_x1:.2f}" y="{box_y1:.2f}" width="{box_w:.2f}" height="{box_h:.2f}"
        fill="none" stroke="{box_color}" stroke-width="0.3" stroke-dasharray="1,0.5"/>
''')

                # Scale bar on FOV (bottom-right)
                scalebar_mm = (SCALE_BAR_FOV_UM / PIXEL_SIZE_XY) * scale_x
                sb_x2 = cell_x + FOV_WIDTH_MM - 0.5
                sb_x1 = sb_x2 - scalebar_mm
                sb_y = cell_y + CELL_HEIGHT_MM - 1.0
                svg_parts.append(f'''  <line x1="{sb_x1:.2f}" y1="{sb_y:.2f}" x2="{sb_x2:.2f}" y2="{sb_y:.2f}"
        stroke="white" stroke-width="0.3"/>
  <text x="{(sb_x1 + sb_x2) / 2:.2f}" y="{sb_y - 0.3:.2f}" class="scale-text">{SCALE_BAR_FOV_UM} µm</text>
''')

                # mRNA labels on FOV (top-left)
                mrna_x = cell_x + 0.5
                mrna_y_green = cell_y + 2.0
                mrna_y_orange = cell_y + 3.5
                mrna_green = cat_data.get('mrna_green', 0)
                mrna_orange = cat_data.get('mrna_orange', 0)
                svg_parts.append(f'  <text x="{mrna_x:.2f}" y="{mrna_y_green:.2f}" class="mrna-green">{mrna_green:.1f} mRNA/nuc</text>\n')
                svg_parts.append(f'  <text x="{mrna_x:.2f}" y="{mrna_y_orange:.2f}" class="mrna-orange">{mrna_orange:.1f} mRNA/nuc</text>\n')

                # Dynamic range labels on FOV (bottom-left, above scale bar)
                dr = cat_data.get('dynamic_range', {})
                dapi_range = dr.get('dapi_range', (0, 0))
                green_range_val = dr.get('green_range', (0, 0))
                orange_range_val = dr.get('orange_range', (0, 0))
                dr_x = cell_x + 0.5
                dr_y_base = cell_y + CELL_HEIGHT_MM - 3.5
                svg_parts.append(f'  <text x="{dr_x:.2f}" y="{dr_y_base:.2f}" class="range-text-small" fill="#6699FF">DAPI [{dapi_range[0]:.0f}-{dapi_range[1]:.0f}]</text>\n')
                svg_parts.append(f'  <text x="{dr_x:.2f}" y="{dr_y_base + 1.2:.2f}" class="range-text-small" fill="{cfg.COLOR_Q111_MHTT1A}">Green [{green_range_val[0]:.0f}-{green_range_val[1]:.0f}]</text>\n')
                svg_parts.append(f'  <text x="{dr_x:.2f}" y="{dr_y_base + 2.4:.2f}" class="range-text-small" fill="{cfg.COLOR_Q111_FULL}">Orange [{orange_range_val[0]:.0f}-{orange_range_val[1]:.0f}]</text>\n')

            # Zoom images (stacked vertically next to FOV, high quality PNG)
            zoom_arrays = cat_data.get('zoom_arrays', [])
            # Calculate zoom size (square) to fit 2 zooms in cell height
            zoom_size_mm = (CELL_HEIGHT_MM - ZOOM_GAP_MM) / 2
            # Center zooms horizontally in the remaining space
            zoom_x = cell_x + FOV_WIDTH_MM + GAP_MM * 0.2 + (ZOOM_WIDTH_MM - zoom_size_mm) / 2

            # Scale bar for zooms
            zoom_scale_x = zoom_size_mm / ZOOM_SIZE
            zoom_sb_mm = (SCALE_BAR_ZOOM_UM / PIXEL_SIZE_XY) * zoom_scale_x

            for i, zoom_arr in enumerate(zoom_arrays[:2]):  # Max 2 zooms
                zoom_y = cell_y + i * (zoom_size_mm + ZOOM_GAP_MM)
                zoom_b64 = array_to_base64_png(zoom_arr, max_size=400)
                box_color = box_colors[i % len(box_colors)]

                # Scale bar position in zoom
                zs_x2 = zoom_x + zoom_size_mm - 0.3
                zs_x1 = zs_x2 - zoom_sb_mm
                zs_y = zoom_y + zoom_size_mm - 0.5

                svg_parts.append(f'''  <image x="{zoom_x:.2f}" y="{zoom_y:.2f}" width="{zoom_size_mm:.2f}" height="{zoom_size_mm:.2f}"
         xlink:href="data:image/png;base64,{zoom_b64}"
         preserveAspectRatio="xMidYMid meet"/>
  <rect x="{zoom_x:.2f}" y="{zoom_y:.2f}" width="{zoom_size_mm:.2f}" height="{zoom_size_mm:.2f}"
        fill="none" stroke="{box_color}" stroke-width="0.3"/>
  <line x1="{zs_x1:.2f}" y1="{zs_y:.2f}" x2="{zs_x2:.2f}" y2="{zs_y:.2f}"
        stroke="white" stroke-width="0.25"/>
  <text x="{(zs_x1 + zs_x2) / 2:.2f}" y="{zs_y - 0.2:.2f}" class="scale-text">{SCALE_BAR_ZOOM_UM} µm</text>
''')

    # Add gamma info text at bottom (dynamic ranges are shown per-pair on each image)
    if first_dynamic_range:
        gamma_val = first_dynamic_range.get('gamma', 1.0)
        gamma_text = f"Gamma: {gamma_val:.1f}" if use_gamma else "Linear scaling"
        dr_y = TOTAL_HEIGHT_MM + dr_height_mm * 0.6

        svg_parts.append(f'''
  <!-- Scaling Info (per-pair dynamic ranges shown on each FOV) -->
  <text x="{ROW_LABEL_WIDTH_MM:.2f}" y="{dr_y:.2f}" class="range-text">
    {gamma_text} | Dynamic ranges calculated per animal pair (shown on each image)
  </text>
''')

    svg_parts.append('</svg>\n')

    # Write SVG
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(''.join(svg_parts))

    print(f"Saved: {output_path}")


def main():
    """Generate Figure 5 collages - Extreme vs Normal Q111 FOVs."""

    print("=" * 70)
    print("FIGURE 5 COLLAGE GENERATION")
    print("Extreme vs Normal Q111 FOVs (2x3 grid)")
    print("=" * 70)

    print(f"\nDimensions: {TOTAL_WIDTH_MM:.1f}mm x {TOTAL_HEIGHT_MM:.1f}mm")
    print(f"Layout: {N_ROWS} rows x {N_ANIMALS} columns")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Find H5 file
    h5_path = find_h5_file()
    print(f"\nUsing H5 file: {h5_path}")

    # Get expression data for all FOVs
    print("\nLoading expression data...")
    df = get_all_fov_expression_data(h5_path)

    if len(df) == 0:
        print("ERROR: No FOV data found")
        return

    # Select paired FOVs - get more than we need to create multiple collages
    print("\nSelecting paired Normal vs Extreme Q111 FOVs...")
    normal_fovs, extreme_fovs = select_paired_fovs(df, h5_path, channel='green', n_pairs=30)

    if len(normal_fovs) < N_ANIMALS:
        print(f"ERROR: Need at least {N_ANIMALS} pairs, found {len(normal_fovs)}")
        return

    # Group pairs by animal (using SLIDE_LABEL_MAP_Q111)
    # Multiple slides can belong to the same animal
    animal_to_pairs = defaultdict(list)
    for idx, (normal_info, extreme_info) in enumerate(zip(normal_fovs, extreme_fovs)):
        slide = normal_info['slide']
        animal_label = get_animal_label(slide)
        animal_to_pairs[animal_label].append(idx)

    # Get list of unique animals
    unique_animals = sorted(animal_to_pairs.keys())
    print(f"  Found {len(unique_animals)} unique animals: {unique_animals}")

    if len(unique_animals) < N_ANIMALS:
        print(f"ERROR: Need at least {N_ANIMALS} different animals, found {len(unique_animals)}")
        return

    # Generate multiple collages with different animal combinations
    max_collages = 20  # Maximum number of collages to generate

    # Pre-generate all shuffled animal orders for reproducibility
    import random
    random.seed(42)  # Fixed seed for reproducibility

    # Track used FOV keys globally to avoid ANY FOV repetition across collages
    used_fov_keys = set()

    # Shuffle pairs within each animal for variety
    animal_pair_order = {}
    for animal_label, pair_indices in animal_to_pairs.items():
        shuffled_pairs = pair_indices.copy()
        random.shuffle(shuffled_pairs)
        animal_pair_order[animal_label] = shuffled_pairs

    # Track which pairs are still available per animal
    animal_available_pairs = {
        animal: list(pairs) for animal, pairs in animal_pair_order.items()
    }

    all_animal_shuffles = []
    for i in range(max_collages):
        animals = unique_animals.copy()
        random.shuffle(animals)
        all_animal_shuffles.append(animals)

    collage_idx = 0
    for attempt_idx in range(max_collages):
        print(f"\n--- Generating collage {collage_idx + 1} (attempt {attempt_idx + 1}/{max_collages}) ---")

        # Use pre-generated shuffle for this collage
        shuffled_animals = all_animal_shuffles[attempt_idx]

        # ══════════════════════════════════════════════════════════════════════
        # PASS 1: Collect all FOV info and MIPs, calculate PER-PAIR dynamic range
        # ══════════════════════════════════════════════════════════════════════
        print("\n  Pass 1: Loading MIPs and calculating per-pair dynamic ranges...")

        collage_fov_info = []  # Store info for all FOVs in this collage
        pairs_selected_this_collage = []  # Track pairs selected for this collage

        for animal_idx in range(N_ANIMALS):
            # Select a different animal for each column
            animal_label = shuffled_animals[animal_idx % len(shuffled_animals)]

            # Get the available pairs for this animal (not yet used in ANY collage)
            available_pairs = animal_available_pairs[animal_label]

            # Find a pair where BOTH FOVs are unused
            pair_idx = None
            for candidate_idx in available_pairs:
                normal_fov_key = normal_fovs[candidate_idx]['fov_key']
                extreme_fov_key = extreme_fovs[candidate_idx]['fov_key']
                if normal_fov_key not in used_fov_keys and extreme_fov_key not in used_fov_keys:
                    pair_idx = candidate_idx
                    break

            # If no unused pair found, skip this animal
            if pair_idx is None:
                print(f"    Animal {animal_idx + 1} ({animal_label}): No unused pairs available - skipping")
                continue

            # Remember this pair for later marking as used
            pairs_selected_this_collage.append((animal_label, pair_idx))

            normal_info = normal_fovs[pair_idx]
            extreme_info = extreme_fovs[pair_idx]
            slide = normal_info['slide']

            # Get animal label (re-derive from slide to be safe)
            animal_label = get_animal_label(slide)

            # Get mRNA values for both channels
            normal_fov_key = normal_info['fov_key']
            extreme_fov_key = extreme_info['fov_key']

            # Get green and orange mRNA for normal FOV - NO FALLBACKS
            normal_green_df = df[(df['fov_key'] == normal_fov_key) & (df['channel'] == 'green')]
            normal_orange_df = df[(df['fov_key'] == normal_fov_key) & (df['channel'] == 'orange')]

            if len(normal_green_df) == 0 or len(normal_orange_df) == 0:
                print(f"    Animal {animal_idx + 1}: Skipping - missing normal FOV data")
                continue

            normal_mrna_green = normal_green_df['mrna_per_nucleus'].values[0]
            normal_mrna_orange = normal_orange_df['mrna_per_nucleus'].values[0]

            # Skip if values are None/NaN (missing peak intensity data)
            if normal_mrna_green is None or normal_mrna_orange is None or np.isnan(normal_mrna_green) or np.isnan(normal_mrna_orange):
                print(f"    Animal {animal_idx + 1}: Skipping - normal FOV has NaN values")
                continue

            # Get green and orange mRNA for extreme FOV - NO FALLBACKS
            extreme_green_df = df[(df['fov_key'] == extreme_fov_key) & (df['channel'] == 'green')]
            extreme_orange_df = df[(df['fov_key'] == extreme_fov_key) & (df['channel'] == 'orange')]

            if len(extreme_green_df) == 0 or len(extreme_orange_df) == 0:
                print(f"    Animal {animal_idx + 1}: Skipping - missing extreme FOV data")
                continue

            extreme_mrna_green = extreme_green_df['mrna_per_nucleus'].values[0]
            extreme_mrna_orange = extreme_orange_df['mrna_per_nucleus'].values[0]

            # Skip if values are None/NaN (missing peak intensity data)
            if extreme_mrna_green is None or extreme_mrna_orange is None or np.isnan(extreme_mrna_green) or np.isnan(extreme_mrna_orange):
                print(f"    Animal {animal_idx + 1}: Skipping - extreme FOV has NaN values")
                continue

            # Load raw MIPs for both FOVs
            normal_dapi, normal_green, normal_orange, normal_coms = load_fov_mips(h5_path, normal_fov_key)
            extreme_dapi, extreme_green, extreme_orange, extreme_coms = load_fov_mips(h5_path, extreme_fov_key)

            if normal_dapi is None or extreme_dapi is None:
                print(f"    Animal {animal_idx + 1}: Skipping - could not load MIPs")
                continue

            # Calculate PER-PAIR dynamic range (from just this normal + extreme FOV)
            pair_green_pixels = np.concatenate([normal_green.ravel(), extreme_green.ravel()])
            pair_orange_pixels = np.concatenate([normal_orange.ravel(), extreme_orange.ravel()])

            pair_green_range = (np.percentile(pair_green_pixels, 50), np.percentile(pair_green_pixels, 99.9))
            pair_orange_range = (np.percentile(pair_orange_pixels, 50), np.percentile(pair_orange_pixels, 99.9))

            # Store info for pass 2 (including per-pair dynamic range)
            collage_fov_info.append({
                'animal_idx': animal_idx,
                'animal_label': animal_label,
                'slide': slide,
                'normal_fov_key': normal_fov_key,
                'extreme_fov_key': extreme_fov_key,
                'normal_mrna_green': normal_mrna_green,
                'normal_mrna_orange': normal_mrna_orange,
                'extreme_mrna_green': extreme_mrna_green,
                'extreme_mrna_orange': extreme_mrna_orange,
                'green_range': pair_green_range,
                'orange_range': pair_orange_range,
            })

            print(f"    Animal {animal_idx + 1} ({slide}): Green [{pair_green_range[0]:.0f}-{pair_green_range[1]:.0f}], Orange [{pair_orange_range[0]:.0f}-{pair_orange_range[1]:.0f}]")

        if len(collage_fov_info) < N_ANIMALS:
            print(f"  WARNING: Only {len(collage_fov_info)} valid animals, need {N_ANIMALS}")
            if len(collage_fov_info) == 0:
                print("  Skipping this collage attempt - no valid animals")
                continue
            # Allow collages with fewer animals if we have at least 1

        # Mark these FOVs as used ONLY if we're generating a collage
        for fov_info in collage_fov_info:
            used_fov_keys.add(fov_info['normal_fov_key'])
            used_fov_keys.add(fov_info['extreme_fov_key'])
        print(f"  Marked {len(collage_fov_info) * 2} FOVs as used (total used: {len(used_fov_keys)})")

        # ══════════════════════════════════════════════════════════════════════
        # PASS 2: Generate all images using PER-PAIR dynamic range
        # ══════════════════════════════════════════════════════════════════════
        print("\n  Pass 2: Generating images with per-pair dynamic range...")

        paired_data_linear = []
        paired_data_gamma = []

        for fov_info in collage_fov_info:
            animal_idx = fov_info['animal_idx']
            animal_label = fov_info['animal_label']
            slide = fov_info['slide']
            normal_fov_key = fov_info['normal_fov_key']
            extreme_fov_key = fov_info['extreme_fov_key']

            # Use PER-PAIR dynamic range
            green_range = fov_info['green_range']
            orange_range = fov_info['orange_range']

            print(f"\n    Animal {animal_idx + 1}: {slide}")
            print(f"      Normal: green={fov_info['normal_mrna_green']:.1f}, orange={fov_info['normal_mrna_orange']:.1f} mRNA/nuc")
            print(f"      Extreme: green={fov_info['extreme_mrna_green']:.1f}, orange={fov_info['extreme_mrna_orange']:.1f} mRNA/nuc")
            print(f"      Range: Green [{green_range[0]:.0f}-{green_range[1]:.0f}], Orange [{orange_range[0]:.0f}-{orange_range[1]:.0f}]")

            # Generate LINEAR images with PER-PAIR range
            print(f"      Generating normal FOV images (linear)...")
            normal_result_linear = create_fov_image_with_zooms(
                h5_path, normal_fov_key, n_zooms=2,
                green_range=green_range, orange_range=orange_range, gamma=1.0
            )

            print(f"      Generating extreme FOV images (linear)...")
            extreme_result_linear = create_fov_image_with_zooms(
                h5_path, extreme_fov_key, n_zooms=2,
                green_range=green_range, orange_range=orange_range, gamma=1.0
            )

            # Generate GAMMA images with PER-PAIR range
            print(f"      Generating normal FOV images (gamma)...")
            normal_result_gamma = create_fov_image_with_zooms(
                h5_path, normal_fov_key, n_zooms=2,
                green_range=green_range, orange_range=orange_range, gamma=GAMMA_VALUE
            )

            print(f"      Generating extreme FOV images (gamma)...")
            extreme_result_gamma = create_fov_image_with_zooms(
                h5_path, extreme_fov_key, n_zooms=2,
                green_range=green_range, orange_range=orange_range, gamma=GAMMA_VALUE
            )

            if normal_result_linear is None or extreme_result_linear is None:
                print(f"      Skipping - could not generate images")
                continue

            # Build LINEAR paired data entry
            paired_data_linear.append({
                'animal_label': animal_label,
                'slide': slide,
                'normal': {
                    **normal_result_linear,
                    'mrna_green': fov_info['normal_mrna_green'],
                    'mrna_orange': fov_info['normal_mrna_orange'],
                },
                'extreme': {
                    **extreme_result_linear,
                    'mrna_green': fov_info['extreme_mrna_green'],
                    'mrna_orange': fov_info['extreme_mrna_orange'],
                },
            })

            # Build GAMMA paired data entry
            paired_data_gamma.append({
                'animal_label': animal_label,
                'slide': slide,
                'normal': {
                    **normal_result_gamma,
                    'mrna_green': fov_info['normal_mrna_green'],
                    'mrna_orange': fov_info['normal_mrna_orange'],
                },
                'extreme': {
                    **extreme_result_gamma,
                    'mrna_green': fov_info['extreme_mrna_green'],
                    'mrna_orange': fov_info['extreme_mrna_orange'],
                },
            })

        # Create LINEAR SVG collage
        output_path = OUTPUT_DIR / f"figure5_collage_{collage_idx + 1:02d}.svg"
        print(f"\n  Creating SVG collage (linear)...")
        create_figure5_collage_svg(paired_data_linear, output_path, use_gamma=False)

        # Create GAMMA SVG collage
        output_path_gamma = OUTPUT_DIR / f"figure5_collage_{collage_idx + 1:02d}_gamma.svg"
        print(f"  Creating SVG collage (gamma)...")
        create_figure5_collage_svg(paired_data_gamma, output_path_gamma, use_gamma=True)

        # Increment collage counter only on success
        collage_idx += 1

        # Check if we have enough unused FOVs for another complete collage
        remaining_pairs_per_animal = {}
        for animal in unique_animals:
            count = 0
            for pair_idx in animal_pair_order[animal]:
                normal_fov = normal_fovs[pair_idx]['fov_key']
                extreme_fov = extreme_fovs[pair_idx]['fov_key']
                if normal_fov not in used_fov_keys and extreme_fov not in used_fov_keys:
                    count += 1
            remaining_pairs_per_animal[animal] = count

        total_remaining = sum(remaining_pairs_per_animal.values())
        print(f"\n  Remaining unused pairs: {total_remaining} (per animal: {remaining_pairs_per_animal})")

        if total_remaining < N_ANIMALS:
            print(f"\n  Not enough unused FOVs for another complete collage - stopping")
            break

    print("\n" + "=" * 70)
    print(f"DONE! Generated {collage_idx} unique collages")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == '__main__':
    main()
