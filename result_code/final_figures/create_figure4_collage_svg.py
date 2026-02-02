#!/usr/bin/env python3
"""
Create SVG collages for Figure 4 - WT vs Q111 comparison.

Layout: 3 FOVs stacked vertically, each with 2 zoom panels on the right
Height: 120mm (width scaled proportionally)

Creates separate collages for:
- Wildtype (WT) - should show minimal/no mHTT signal
- Q111 - shows mHTT expression

Uses figure_config.py for consistent styling.
"""

import os
import sys
from pathlib import Path
import base64
from io import BytesIO
import numpy as np
import pandas as pd
from PIL import Image
from tifffile import imread, imwrite
import h5py
import json

# Add parent directories for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from figure2_panels import (
    find_h5_file,
    get_npz_path_from_h5,
    load_npz_image,
    create_mip,
    normalize_to_8bit,
    get_slide_peak_intensity,
    extract_slide_from_fov_key,
    VOXEL_VOLUME_UM3,
    PIXEL_SIZE_XY,
)
from figure3_panels import create_composite_three_channel
from results_config import (
    VOXEL_SIZE,
    EXCLUDED_SLIDES,
    MEAN_NUCLEAR_VOLUME,
    EXPERIMENTAL_FIELD,
    CHANNEL_PARAMS,
    SLIDE_LABEL_MAP_Q111,
    SLIDE_LABEL_MAP_WT,
    CV_THRESHOLD,
)
from figure_config import FigureConfig

# Figure dimensions
TARGET_HEIGHT_MM = 120.0  # Target height in mm
N_FOVS_PER_COLLAGE = 3

# Gamma correction value for enhanced visibility of dim spots
GAMMA_VALUE = 2.2

# Output directory
OUTPUT_DIR = Path(__file__).parent / 'output' / 'figure4_collages'

# Path to pre-computed thresholds
THRESHOLD_CSV = Path(__file__).parent / 'output' / 'photon_thresholds.csv'


def load_thresholds():
    """Load pre-computed thresholds from negative control analysis."""
    if not THRESHOLD_CSV.exists():
        print(f"Warning: Threshold file not found: {THRESHOLD_CSV}")
        return {}

    thresholds = {}
    df = pd.read_csv(THRESHOLD_CSV)
    for _, row in df.iterrows():
        key_str = row['key']
        parts = key_str.strip("()").replace("'", "").split(", ")
        slide = parts[0]
        channel = parts[1]
        thresholds[(slide, channel)] = row['threshold']
    return thresholds


def get_fov_expression_data(h5_file, fov_key: str, thresholds: dict) -> dict:
    """
    Get clustered mRNA per nucleus for a FOV.

    Returns dict with green/orange mRNA values and metadata.
    """
    result = {
        'green': None,
        'orange': None,
        'n_nuclei': None,
        'mouse_model': None,
        'age': None,
        'region': None,
        'slide': None,
    }

    if fov_key not in h5_file:
        return result

    fov = h5_file[fov_key]
    slide = extract_slide_from_fov_key(fov_key)
    result['slide'] = slide

    # Calculate nuclei count
    if 'blue' in fov and 'label_sizes' in fov['blue']:
        label_sizes = fov['blue']['label_sizes'][:]
        if len(label_sizes) > 0:
            V_DAPI = np.sum(label_sizes) * VOXEL_SIZE
            result['n_nuclei'] = V_DAPI / MEAN_NUCLEAR_VOLUME

    # Get metadata
    if 'metadata_sample' in fov:
        sample_meta = fov['metadata_sample']

        if 'Mouse Model' in sample_meta:
            mm = sample_meta['Mouse Model'][()]
            if isinstance(mm, np.ndarray) and len(mm) > 0:
                mm = mm[0]
            if isinstance(mm, bytes):
                mm = mm.decode()
            result['mouse_model'] = mm

        if 'Age' in sample_meta:
            age_val = sample_meta['Age'][()]
            if isinstance(age_val, np.ndarray) and len(age_val) > 0:
                result['age'] = f"{int(age_val[0])}mo"
            else:
                result['age'] = f"{int(age_val)}mo"

        if 'Slice Region' in sample_meta:
            reg = sample_meta['Slice Region'][()]
            if isinstance(reg, np.ndarray) and len(reg) > 0:
                reg = reg[0]
            if isinstance(reg, bytes):
                reg = reg.decode()
            if 'Cortex' in reg:
                result['region'] = 'Cortex'
            elif 'Striatum' in reg:
                result['region'] = 'Striatum'
            else:
                result['region'] = reg

    # Calculate clustered mRNA per nucleus for each channel
    for channel in ['green', 'orange']:
        if channel not in fov:
            continue

        ch_data = fov[channel]
        if 'cluster_intensities' not in ch_data:
            continue

        intensities = ch_data['cluster_intensities'][:]
        cluster_cvs = ch_data['cluster_cvs'][:] if 'cluster_cvs' in ch_data else None
        if len(intensities) == 0:
            result[channel] = 0.0
            continue

        # Get threshold - skip if missing
        threshold_key = (slide, channel)
        if threshold_key not in thresholds:
            result[channel] = None
            continue
        threshold = thresholds[threshold_key]

        # Get peak intensity for normalization - skip if missing
        try:
            peak_intensity = get_slide_peak_intensity(slide, channel)
        except ValueError:
            result[channel] = None
            continue

        if np.isnan(peak_intensity):
            result[channel] = None
            continue

        # Filter by threshold AND CV (CV >= CV_THRESHOLD means good quality)
        intensity_mask = intensities > threshold
        if cluster_cvs is not None and len(cluster_cvs) == len(intensities):
            cv_mask = cluster_cvs >= CV_THRESHOLD
            valid_mask = intensity_mask & cv_mask
        else:
            # If no CV data, only use intensity threshold
            valid_mask = intensity_mask
        valid_intensities = intensities[valid_mask]
        total_mrna = np.sum(valid_intensities) / peak_intensity if len(valid_intensities) > 0 else 0

        if result['n_nuclei'] and result['n_nuclei'] > 0:
            result[channel] = total_mrna / result['n_nuclei']
        else:
            result[channel] = 0.0

    return result


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


def create_composite_image(h5_file, fov_key: str, h5_path: str, is_wildtype: bool = False,
                           gamma: float = 1.0) -> dict:
    """Create RGB composite from FOV (blue=DAPI, green=mHTT1a, orange=full-length).

    Args:
        h5_file: Open H5 file handle
        fov_key: FOV key in HDF5
        h5_path: Path to H5 file
        is_wildtype: If True, use higher contrast for green channel (background reduction)
        gamma: Gamma correction value (1.0 = linear, >1 brightens dim values)

    Returns:
        dict with 'composite', 'dynamic_range', 'image_width', 'image_height'
        or None if failed
    """
    npz_path = get_npz_path_from_h5(fov_key, h5_path)
    if npz_path is None or not Path(npz_path).exists():
        return None

    result = load_npz_image(npz_path)
    if result is None:
        return None

    # load_npz_image returns (arr, metadata) tuple
    img_data, metadata = result

    # Create MIPs for each channel
    blue_mip = create_mip(img_data[0]) if img_data.shape[0] > 0 else None
    green_mip = create_mip(img_data[1]) if img_data.shape[0] > 1 else None
    orange_mip = create_mip(img_data[2]) if img_data.shape[0] > 2 else None

    # Handle missing channels
    if blue_mip is None:
        blue_mip = np.zeros((512, 512), dtype=np.float32)
    if green_mip is None:
        green_mip = np.zeros_like(blue_mip)
    if orange_mip is None:
        orange_mip = np.zeros_like(blue_mip)

    # Use higher pmin for wildtype to reduce background in green channel
    if is_wildtype:
        signal_pmin = 20  # Higher floor to cut background
        signal_pmax = 99.9
    else:
        signal_pmin = 1
        signal_pmax = 99.9

    # Use create_composite_three_channel for proper orange coloring
    composite_result = create_composite_three_channel(
        blue_mip, green_mip, orange_mip,
        dapi_pmin=1, dapi_pmax=99.5,
        signal_pmin=signal_pmin, signal_pmax=signal_pmax,
        gamma=gamma
    )

    return {
        'composite': composite_result['composite'],
        'dynamic_range': {
            'dapi_range': composite_result['dapi_range'],
            'green_range': composite_result['green_range'],
            'orange_range': composite_result['orange_range'],
            'gamma': gamma,
        },
        'image_width': blue_mip.shape[1],
        'image_height': blue_mip.shape[0],
    }


def select_zoom_regions(composite: np.ndarray, n_zooms: int = 2, zoom_size: int = 128) -> list:
    """Select interesting zoom regions based on green channel intensity."""
    if composite is None:
        return []

    h, w = composite.shape[:2]
    green_channel = composite[:, :, 1]  # Green channel

    # Find regions with high green intensity
    regions = []
    step = zoom_size // 2

    for y in range(0, h - zoom_size, step):
        for x in range(0, w - zoom_size, step):
            region = green_channel[y:y+zoom_size, x:x+zoom_size]
            intensity = np.mean(region)
            regions.append((x, y, intensity))

    # Sort by intensity and pick top regions that don't overlap too much
    regions.sort(key=lambda r: r[2], reverse=True)

    selected = []
    for x, y, intensity in regions:
        # Check overlap with already selected
        overlap = False
        for sx, sy in selected:
            if abs(x - sx) < zoom_size * 0.7 and abs(y - sy) < zoom_size * 0.7:
                overlap = True
                break

        if not overlap:
            selected.append((x, y))
            if len(selected) >= n_zooms:
                break

    # If not enough regions found, add some defaults
    while len(selected) < n_zooms:
        x = np.random.randint(0, max(1, w - zoom_size))
        y = np.random.randint(0, max(1, h - zoom_size))
        selected.append((x, y))

    return selected[:n_zooms]


def array_to_base64_png(arr: np.ndarray, max_size: int = None) -> str:
    """Convert numpy array to base64-encoded PNG."""
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


def create_collage_svg(fov_data_list: list, output_path: Path, title: str = "",
                       use_gamma: bool = False):
    """
    Create an SVG collage with 3 FOVs.

    Each FOV row: [overview image with labels + zoom boxes] [zoom1 / zoom2]

    Args:
        fov_data_list: List of dicts with keys: 'overview_b64', 'zoom1_b64', 'zoom2_b64',
                       'green_mrna', 'orange_mrna', 'metadata', 'zoom_coords', 'dynamic_range',
                       'image_width', 'image_height'
        output_path: Output SVG path
        title: Title for the collage (e.g., "Wildtype" or "Q111")
        use_gamma: If True, show gamma value in dynamic range text
    """
    cfg = FigureConfig

    n_fovs = len(fov_data_list)
    if n_fovs == 0:
        return False

    # Physical scale parameters
    SCALE_BAR_OVERVIEW_UM = 20  # µm for overview
    SCALE_BAR_ZOOM_UM = 10  # µm for zooms

    # Layout parameters (in viewBox units)
    margin = 20
    title_height = 40
    row_gap = 15
    col_gap = 10
    dr_height = 25  # Space for dynamic range text at bottom

    # FOV overview size
    fov_width = 400
    fov_height = 400

    # Zoom panel size (2 stacked = same height as FOV)
    zoom_width = 180
    zoom_height = (fov_height - col_gap) // 2

    # Calculate total dimensions
    total_width_vb = margin * 2 + fov_width + col_gap + zoom_width
    total_height_vb = margin * 2 + title_height + n_fovs * fov_height + (n_fovs - 1) * row_gap + dr_height

    # Physical dimensions in mm
    height_mm = TARGET_HEIGHT_MM
    width_mm = height_mm * (total_width_vb / total_height_vb)

    # Font sizes scaled to viewBox
    scale_factor = total_height_vb / TARGET_HEIGHT_MM
    font_size_title = cfg.FONT_SIZE_TITLE * scale_factor * 0.35 * 2
    font_size_label = cfg.FONT_SIZE_AXIS_LABEL * scale_factor * 0.35 * 1.8
    font_size_small = cfg.FONT_SIZE_ANNOTATION * scale_factor * 0.35 * 1.0
    font_size_scalebar = cfg.FONT_SIZE_ANNOTATION * scale_factor * 0.35 * 0.8

    # Colors
    color_green = cfg.COLOR_Q111_MHTT1A
    color_orange = cfg.COLOR_Q111_FULL
    color_white = cfg.COLOR_WHITE
    font_family = ', '.join(cfg.FONT_SANS_SERIF)

    # Zoom box colors (match zoom panel border colors)
    box_colors = ['#FFFFFF', '#FFFF00']  # White for zoom1, yellow for zoom2

    # Build SVG
    svg_parts = []
    svg_parts.append(f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg"
     xmlns:xlink="http://www.w3.org/1999/xlink"
     width="{width_mm:.3f}mm" height="{height_mm:.3f}mm"
     viewBox="0 0 {total_width_vb} {total_height_vb}">

  <defs>
    <style>
      .title {{ font-family: {font_family}; font-size: {font_size_title:.1f}px; font-weight: bold; fill: white; text-anchor: middle; }}
      .label-green {{ font-family: {font_family}; font-size: {font_size_label:.1f}px; font-weight: bold; fill: {color_green}; }}
      .label-orange {{ font-family: {font_family}; font-size: {font_size_label:.1f}px; font-weight: bold; fill: {color_orange}; }}
      .range-text {{ font-family: {font_family}; font-size: {font_size_small:.1f}px; fill: {color_white}; }}
      .scale-text {{ font-family: {font_family}; font-size: {font_size_scalebar:.1f}px; fill: {color_white}; text-anchor: middle; }}
    </style>
  </defs>

  <!-- Black Background -->
  <rect width="100%" height="100%" fill="black"/>

  <!-- Title -->
  <text x="{total_width_vb / 2}" y="{margin + font_size_title * 0.8}" class="title">{title}</text>
''')

    # Add each FOV row
    for i, fov_data in enumerate(fov_data_list):
        row_y = margin + title_height + i * (fov_height + row_gap)

        # FOV overview
        fov_x = margin
        overview_b64 = fov_data.get('overview_b64', '')

        # Get image dimensions for scaling zoom boxes
        img_width = fov_data.get('image_width', 512)
        img_height = fov_data.get('image_height', 512)
        scale_x = fov_width / img_width
        scale_y = fov_height / img_height

        svg_parts.append(f'''
  <!-- FOV {i+1} -->
  <g id="fov_{i+1}">
    <!-- Overview -->
    <image x="{fov_x}" y="{row_y}" width="{fov_width}" height="{fov_height}"
           xlink:href="data:image/png;base64,{overview_b64}"
           preserveAspectRatio="xMidYMid meet"/>
''')

        # Draw zoom boxes on overview
        zoom_coords = fov_data.get('zoom_coords', [])
        for z_idx, zc in enumerate(zoom_coords[:2]):
            box_x1 = fov_x + zc['x1'] * scale_x
            box_y1 = row_y + zc['y1'] * scale_y
            box_w = (zc['x2'] - zc['x1']) * scale_x
            box_h = (zc['y2'] - zc['y1']) * scale_y
            box_color = box_colors[z_idx % len(box_colors)]
            svg_parts.append(f'''    <rect x="{box_x1:.1f}" y="{box_y1:.1f}" width="{box_w:.1f}" height="{box_h:.1f}"
          fill="none" stroke="{box_color}" stroke-width="2" stroke-dasharray="4,2"/>
''')

        # Scale bar on overview (bottom-right)
        scalebar_px = SCALE_BAR_OVERVIEW_UM / PIXEL_SIZE_XY
        scalebar_vb = scalebar_px * scale_x
        sb_x2 = fov_x + fov_width - 10
        sb_x1 = sb_x2 - scalebar_vb
        sb_y = row_y + fov_height - 15
        svg_parts.append(f'''    <line x1="{sb_x1:.1f}" y1="{sb_y:.1f}" x2="{sb_x2:.1f}" y2="{sb_y:.1f}"
          stroke="white" stroke-width="2"/>
    <text x="{(sb_x1 + sb_x2) / 2:.1f}" y="{sb_y - 4:.1f}" class="scale-text">{SCALE_BAR_OVERVIEW_UM} µm</text>
''')

        # mRNA labels (inside FOV, top-left)
        green_mrna = fov_data.get('green_mrna', 0)
        orange_mrna = fov_data.get('orange_mrna', 0)

        if green_mrna is not None:
            svg_parts.append(f'''    <text x="{fov_x + 10}" y="{row_y + font_size_label + 5}" class="label-green">{green_mrna:.1f} mRNA/nuc</text>
''')
        if orange_mrna is not None:
            svg_parts.append(f'''    <text x="{fov_x + 10}" y="{row_y + font_size_label * 2 + 10}" class="label-orange">{orange_mrna:.2f} mRNA/nuc</text>
''')

        # Zoom panels
        zoom_x = margin + fov_width + col_gap

        zoom1_b64 = fov_data.get('zoom1_b64', '')
        zoom2_b64 = fov_data.get('zoom2_b64', '')

        # Scale bar for zooms (based on 128px zoom size upscaled 2x to 256px displayed)
        zoom_size_px = 128
        zoom_scale = zoom_width / (zoom_size_px * 2)  # Account for 2x upscale
        zoom_sb_px = SCALE_BAR_ZOOM_UM / PIXEL_SIZE_XY
        zoom_sb_vb = zoom_sb_px * zoom_scale * 2  # Scale for upscaled zoom

        # Zoom 1 (top) - white border
        zs1_x2 = zoom_x + zoom_width - 8
        zs1_x1 = zs1_x2 - zoom_sb_vb
        zs1_y = row_y + zoom_height - 10
        svg_parts.append(f'''    <!-- Zoom 1 -->
    <image x="{zoom_x}" y="{row_y}" width="{zoom_width}" height="{zoom_height}"
           xlink:href="data:image/png;base64,{zoom1_b64}"
           preserveAspectRatio="xMidYMid meet"/>
    <rect x="{zoom_x}" y="{row_y}" width="{zoom_width}" height="{zoom_height}"
          fill="none" stroke="{box_colors[0]}" stroke-width="2"/>
    <line x1="{zs1_x1:.1f}" y1="{zs1_y:.1f}" x2="{zs1_x2:.1f}" y2="{zs1_y:.1f}"
          stroke="white" stroke-width="2"/>
    <text x="{(zs1_x1 + zs1_x2) / 2:.1f}" y="{zs1_y - 3:.1f}" class="scale-text">{SCALE_BAR_ZOOM_UM} µm</text>
''')

        # Zoom 2 (bottom) - yellow border
        zoom2_y = row_y + zoom_height + col_gap
        zs2_x2 = zoom_x + zoom_width - 8
        zs2_x1 = zs2_x2 - zoom_sb_vb
        zs2_y = zoom2_y + zoom_height - 10
        svg_parts.append(f'''    <!-- Zoom 2 -->
    <image x="{zoom_x}" y="{zoom2_y}" width="{zoom_width}" height="{zoom_height}"
           xlink:href="data:image/png;base64,{zoom2_b64}"
           preserveAspectRatio="xMidYMid meet"/>
    <rect x="{zoom_x}" y="{zoom2_y}" width="{zoom_width}" height="{zoom_height}"
          fill="none" stroke="{box_colors[1]}" stroke-width="2"/>
    <line x1="{zs2_x1:.1f}" y1="{zs2_y:.1f}" x2="{zs2_x2:.1f}" y2="{zs2_y:.1f}"
          stroke="white" stroke-width="2"/>
    <text x="{(zs2_x1 + zs2_x2) / 2:.1f}" y="{zs2_y - 3:.1f}" class="scale-text">{SCALE_BAR_ZOOM_UM} µm</text>
  </g>
''')

    # Add dynamic range text at bottom (use first FOV's range as representative)
    if fov_data_list and fov_data_list[0].get('dynamic_range'):
        dr = fov_data_list[0]['dynamic_range']
        dapi_range = dr.get('dapi_range', (0, 0))
        green_range = dr.get('green_range', (0, 0))
        orange_range = dr.get('orange_range', (0, 0))
        gamma_val = dr.get('gamma', 1.0)

        gamma_text = f" | Gamma: {gamma_val:.1f}" if use_gamma else " | Linear"
        dr_y = total_height_vb - margin / 2

        svg_parts.append(f'''
  <!-- Dynamic Range Info -->
  <text x="{margin}" y="{dr_y:.1f}" class="range-text">
    DAPI [{dapi_range[0]:.0f}-{dapi_range[1]:.0f}], Green [{green_range[0]:.0f}-{green_range[1]:.0f}], Orange [{orange_range[0]:.0f}-{orange_range[1]:.0f}]{gamma_text}
  </text>
''')

    svg_parts.append('</svg>\n')

    # Write SVG
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(''.join(svg_parts))

    print(f"  Saved: {output_path.name} ({width_mm:.1f} x {height_mm:.1f} mm)")
    return True


def get_candidate_fovs(h5_file, h5_path: str, thresholds: dict, mouse_model: str) -> list:
    """Get list of candidate FOVs for a given mouse model."""
    candidates = []

    fov_keys = list(h5_file.keys())

    for fov_key in fov_keys:
        # Check probe set
        if not check_fov_probe_set(h5_file, fov_key):
            continue

        # Check excluded slides
        slide = extract_slide_from_fov_key(fov_key)
        if slide in EXCLUDED_SLIDES:
            continue

        # Get expression data
        expr_data = get_fov_expression_data(h5_file, fov_key, thresholds)

        # Check mouse model
        if expr_data['mouse_model'] != mouse_model:
            continue

        # Check minimum nuclei
        if expr_data['n_nuclei'] is None or expr_data['n_nuclei'] < 40:
            continue

        # Skip if missing channel data (e.g., m1a1 missing orange peak intensity)
        if expr_data['green'] is None or expr_data['orange'] is None:
            continue

        # Check if NPZ file exists
        npz_path = get_npz_path_from_h5(fov_key, h5_path)
        if npz_path is None or not Path(npz_path).exists():
            continue

        candidates.append({
            'fov_key': fov_key,
            'green_mrna': expr_data['green'],
            'orange_mrna': expr_data['orange'],
            'n_nuclei': expr_data['n_nuclei'],
            'region': expr_data['region'],
            'age': expr_data['age'],
            'slide': slide,
        })

    return candidates


def create_fov_images(h5_file, fov_key: str, h5_path: str, zoom_size: int = 128,
                      is_wildtype: bool = False, gamma: float = 1.0) -> dict:
    """Create overview and zoom images for a FOV.

    Args:
        h5_file: Open H5 file handle
        fov_key: FOV key in HDF5
        h5_path: Path to H5 file
        zoom_size: Size of zoom regions in pixels
        is_wildtype: If True, use higher contrast for green channel
        gamma: Gamma correction value (1.0 = linear)

    Returns:
        dict with overview_b64, zoom1_b64, zoom2_b64, zoom_coords, dynamic_range, image dimensions
    """
    composite_result = create_composite_image(h5_file, fov_key, h5_path,
                                               is_wildtype=is_wildtype, gamma=gamma)
    if composite_result is None:
        return None

    composite = composite_result['composite']
    dynamic_range = composite_result['dynamic_range']
    img_width = composite_result['image_width']
    img_height = composite_result['image_height']

    # Get zoom regions
    zoom_regions = select_zoom_regions(composite, n_zooms=2, zoom_size=zoom_size)

    # Create zoom crops and track coordinates
    zooms = []
    zoom_coords = []
    for x, y in zoom_regions:
        # Store zoom coordinates for drawing boxes
        zoom_coords.append({
            'x1': x,
            'y1': y,
            'x2': x + zoom_size,
            'y2': y + zoom_size,
        })

        zoom_crop = composite[y:y+zoom_size, x:x+zoom_size]
        # Upscale for better visibility
        zoom_img = Image.fromarray(zoom_crop)
        zoom_img = zoom_img.resize((zoom_size * 2, zoom_size * 2), Image.LANCZOS)
        zooms.append(np.array(zoom_img))

    return {
        'overview_b64': array_to_base64_png(composite, max_size=800),
        'zoom1_b64': array_to_base64_png(zooms[0] if len(zooms) > 0 else None, max_size=400),
        'zoom2_b64': array_to_base64_png(zooms[1] if len(zooms) > 1 else None, max_size=400),
        'zoom_coords': zoom_coords,
        'dynamic_range': dynamic_range,
        'image_width': img_width,
        'image_height': img_height,
    }


def get_animal_label(slide: str, mouse_model: str) -> str:
    """Get animal label from slide name using results_config mappings."""
    if mouse_model == 'Wildtype':
        return SLIDE_LABEL_MAP_WT.get(slide, slide)
    else:
        return SLIDE_LABEL_MAP_Q111.get(slide, slide)


def build_slide_label_map(mouse_model: str) -> dict:
    """
    Build a mapping from slide to full label like '#7.1', '#7.2' for different slides
    from the same animal.

    Returns dict: {slide_name: 'animal.slide_idx'} e.g. {'m2a3': '7.1', 'm2a8': '7.2'}
    """
    if mouse_model == 'Wildtype':
        base_map = SLIDE_LABEL_MAP_WT
    else:
        base_map = SLIDE_LABEL_MAP_Q111

    # Group slides by animal label
    from collections import defaultdict
    animal_slides = defaultdict(list)
    for slide, label in base_map.items():
        animal_num = label.replace('#', '')
        animal_slides[animal_num].append(slide)

    # Sort slides within each animal for consistent ordering
    for animal_num in animal_slides:
        animal_slides[animal_num].sort()

    # Build the full label map
    full_label_map = {}
    for animal_num, slides in animal_slides.items():
        for idx, slide in enumerate(slides, start=1):
            full_label_map[slide] = f"{animal_num}.{idx}"

    return full_label_map


def main():
    """Main entry point."""
    print("=" * 70)
    print("FIGURE 4 COLLAGE GENERATION")
    print(f"Layout: {N_FOVS_PER_COLLAGE} FOVs per collage, each with 2 zooms")
    print(f"Height: {TARGET_HEIGHT_MM} mm")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load thresholds
    thresholds = load_thresholds()
    print(f"Loaded {len(thresholds)} thresholds")

    # Load H5 file
    h5_path = find_h5_file()
    print(f"Using H5 file: {h5_path}")

    with h5py.File(h5_path, 'r') as h5_file:
        # Process each mouse model
        for mouse_model, base_title in [('Wildtype', 'WT'), ('Q111', 'Q111')]:
            print(f"\n{'=' * 60}")
            print(f"Processing {mouse_model}")
            print('=' * 60)

            # Get candidate FOVs
            candidates = get_candidate_fovs(h5_file, h5_path, thresholds, mouse_model)
            print(f"Found {len(candidates)} candidate FOVs")

            if len(candidates) < N_FOVS_PER_COLLAGE:
                print(f"  Not enough FOVs for collage (need {N_FOVS_PER_COLLAGE})")
                continue

            # Build slide label map (e.g., 'm2a3' -> '7.1', 'm2a8' -> '7.2')
            slide_label_map = build_slide_label_map(mouse_model)

            # Group candidates by slide
            from collections import defaultdict
            slides = defaultdict(list)
            for cand in candidates:
                slides[cand['slide']].append(cand)

            # Sort slide labels for display
            slide_labels = {s: slide_label_map.get(s, s) for s in slides.keys()}
            sorted_slides = sorted(slides.keys(), key=lambda s: (
                float(slide_labels[s].split('.')[0]) if '.' in slide_labels[s] else float(slide_labels[s]),
                float(slide_labels[s].split('.')[1]) if '.' in slide_labels[s] else 0
            ))

            print(f"Slides found: {len(sorted_slides)}")
            for slide in sorted_slides:
                label = slide_label_map.get(slide, slide)
                print(f"  #{label} ({slide}): {len(slides[slide])} FOVs")

            # Create collages for each slide
            for slide in sorted_slides:
                slide_label = slide_label_map.get(slide, slide)
                slide_candidates = slides[slide]

                if len(slide_candidates) < N_FOVS_PER_COLLAGE:
                    print(f"\n  #{slide_label}: Not enough FOVs ({len(slide_candidates)} < {N_FOVS_PER_COLLAGE})")
                    continue

                # Sort by green mRNA (different criteria for WT vs Q111)
                if mouse_model == 'Wildtype':
                    slide_candidates.sort(key=lambda x: x['green_mrna'] if x['green_mrna'] else 0)
                else:
                    slide_candidates.sort(key=lambda x: x['green_mrna'] if x['green_mrna'] else 0, reverse=True)

                # Create up to 2 collages per slide
                n_collages = min(2, len(slide_candidates) // N_FOVS_PER_COLLAGE)

                for collage_idx in range(n_collages):
                    print(f"\n  #{slide_label} - Collage {collage_idx + 1}/{n_collages}")

                    start_idx = collage_idx * N_FOVS_PER_COLLAGE
                    selected_candidates = slide_candidates[start_idx:start_idx + N_FOVS_PER_COLLAGE]

                    if len(selected_candidates) < N_FOVS_PER_COLLAGE:
                        break

                    # Create images for each FOV (both linear and gamma versions)
                    fov_data_list_linear = []
                    fov_data_list_gamma = []
                    is_wt = (mouse_model == 'Wildtype')
                    for cand in selected_candidates:
                        fov_key = cand['fov_key']
                        print(f"    Processing: {fov_key[:50]}...")
                        print(f"      mHTT1a: {cand['green_mrna']:.2f}, fl-mHTT: {cand['orange_mrna']:.2f} mRNA/nuc")

                        # Create linear images
                        images_linear = create_fov_images(h5_file, fov_key, h5_path,
                                                           is_wildtype=is_wt, gamma=1.0)
                        # Create gamma-corrected images
                        images_gamma = create_fov_images(h5_file, fov_key, h5_path,
                                                          is_wildtype=is_wt, gamma=GAMMA_VALUE)

                        if images_linear is None or images_gamma is None:
                            print(f"      WARNING: Could not create images")
                            continue

                        fov_data_list_linear.append({
                            **images_linear,
                            'green_mrna': cand['green_mrna'],
                            'orange_mrna': cand['orange_mrna'],
                            'metadata': cand,
                        })
                        fov_data_list_gamma.append({
                            **images_gamma,
                            'green_mrna': cand['green_mrna'],
                            'orange_mrna': cand['orange_mrna'],
                            'metadata': cand,
                        })

                    if len(fov_data_list_linear) < N_FOVS_PER_COLLAGE:
                        print(f"    Not enough valid FOVs for collage")
                        continue

                    # Create SVG with slide label in title (e.g., "WT #1.1" or "Q111 #7.2")
                    title = f"{base_title} #{slide_label}"
                    # Filename uses slide label with dots replaced by underscores
                    label_clean = slide_label.replace('.', '_')

                    # Create LINEAR SVG
                    output_path = OUTPUT_DIR / f"{mouse_model.lower()}_{label_clean}_collage_{collage_idx + 1:02d}.svg"
                    create_collage_svg(fov_data_list_linear[:N_FOVS_PER_COLLAGE], output_path,
                                       title=title, use_gamma=False)

                    # Create GAMMA SVG
                    output_path_gamma = OUTPUT_DIR / f"{mouse_model.lower()}_{label_clean}_collage_{collage_idx + 1:02d}_gamma.svg"
                    create_collage_svg(fov_data_list_gamma[:N_FOVS_PER_COLLAGE], output_path_gamma,
                                       title=title, use_gamma=True)

    print("\n" + "=" * 70)
    print("DONE!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == '__main__':
    main()
