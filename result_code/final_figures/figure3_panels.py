#!/usr/bin/env python3
"""
Figure 3 Panel Generation Script - Simple Version

Generates example FOV images for Figure 3 showing:
- FOVs with different total mRNA per nucleus levels (low, medium, high)
- Condition information (genotype, age, region) as text overlay

This version loads data directly from H5 files - NO re-running of analysis.
Only visualization of existing results.

Output:
    example_images/figure3/
        low_expression/
        medium_expression/
        high_expression/
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
from tifffile import imwrite
from PIL import Image, ImageDraw, ImageFont
import blosc

# Add parent directories for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'user_code'))

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
    get_photon_threshold,
    CHANNEL_MAP,
    PIXEL_SIZE_XY,
    VOXEL_VOLUME_UM3,
    SCALE_BAR_OVERVIEW,
    SCALE_BAR_ZOOM,
    ZOOM_SIZE,
)

# Import figure styling
from figure_config import FigureConfig, apply_figure_style
apply_figure_style()

# Import global config
sys.path.insert(0, str(Path(__file__).parent.parent))
from results_config import (
    CHANNEL_PARAMS,
    EXCLUDED_SLIDES,
    VOXEL_SIZE,
    MEAN_NUCLEAR_VOLUME,
    EXPERIMENTAL_FIELD,
    CV_THRESHOLD,
)

# Output directory
OUTPUT_DIR = Path(__file__).parent / 'output' / 'example_images' / 'figure3'

# Gamma correction value for enhanced visibility of dim spots
# gamma > 1 brightens dim values: 2.0-2.5 good for seeing all spots
GAMMA_VALUE = 2.2


def load_fov_data_from_h5(h5_path: str, fov_key: str, channel: str):
    """
    Load cluster and spot data for a FOV/channel directly from H5.

    Returns:
        dict with keys: intensities, sizes, coms, spot_coords, n_nuclei, metadata
    """
    with h5py.File(h5_path, 'r') as f:
        if fov_key not in f:
            return None

        fov = f[fov_key]
        data = {}

        # Load channel data
        if channel in fov:
            ch = fov[channel]
            data['intensities'] = ch['cluster_intensities'][:] if 'cluster_intensities' in ch else np.array([])
            data['cvs'] = ch['cluster_cvs'][:] if 'cluster_cvs' in ch else np.array([])
            data['sizes'] = ch['label_sizes'][:] if 'label_sizes' in ch else np.array([])
            data['coms'] = ch['label_coms'][:] if 'label_coms' in ch else np.array([])

            # Load spot coordinates (use final_filter for proper filtering)
            if 'spots' in ch:
                spots = ch['spots']
                if 'filtered_coords' in spots and 'final_filter' in spots:
                    coords = spots['filtered_coords'][:]
                    final_filter = spots['final_filter'][:]
                    # Only include spots that passed all filters (sigma bounds + break_sigma)
                    data['spot_coords'] = coords[final_filter] if len(final_filter) > 0 else np.array([])
                elif 'filtered_coords' in spots and 'filter_indices' in spots:
                    # Fallback to filter_indices if final_filter not available
                    coords = spots['filtered_coords'][:]
                    filter_idx = spots['filter_indices'][:]
                    data['spot_coords'] = coords[filter_idx] if len(filter_idx) > 0 else np.array([])
                else:
                    data['spot_coords'] = np.array([])
            else:
                data['spot_coords'] = np.array([])

        # Calculate nuclei count from DAPI volume (same as comprehensive_cortex_striatum_analysis_v2.py)
        if 'blue' in fov and 'label_sizes' in fov['blue']:
            label_sizes = fov['blue']['label_sizes'][:]
            if len(label_sizes) > 0:
                V_DAPI = np.sum(label_sizes) * VOXEL_SIZE  # μm³
                data['n_nuclei'] = V_DAPI / MEAN_NUCLEAR_VOLUME
            else:
                data['n_nuclei'] = 0
        else:
            data['n_nuclei'] = 0

        # Load metadata
        meta = {}
        if 'metadata_sample' in fov:
            sample_meta = fov['metadata_sample']

            # Mouse model
            if 'Mouse Model' in sample_meta:
                mm = sample_meta['Mouse Model'][()]
                if isinstance(mm, np.ndarray) and len(mm) > 0:
                    mm = mm[0]
                if isinstance(mm, bytes):
                    mm = mm.decode()
                meta['mouse_model'] = mm

            # Age
            if 'Age' in sample_meta:
                age_val = sample_meta['Age'][()]
                if isinstance(age_val, np.ndarray) and len(age_val) > 0:
                    meta['age'] = f"{int(age_val[0])}mo"
                else:
                    meta['age'] = f"{int(age_val)}mo"

            # Region - use 'Slice Region' which has the actual brain region (Cortex/Striatum)
            if 'Slice Region' in sample_meta:
                reg = sample_meta['Slice Region'][()]
                if isinstance(reg, np.ndarray) and len(reg) > 0:
                    reg = reg[0]
                if isinstance(reg, bytes):
                    reg = reg.decode()
                # Shorten region name
                if 'Cortex' in reg:
                    meta['region'] = 'Cortex'
                elif 'Striatum' in reg:
                    meta['region'] = 'Striatum'
                else:
                    meta['region'] = reg
            elif 'Region' in sample_meta:
                # Fallback to Region field
                reg = sample_meta['Region'][()]
                if isinstance(reg, np.ndarray) and len(reg) > 0:
                    reg = reg[0]
                if isinstance(reg, bytes):
                    reg = reg.decode()
                meta['region'] = reg

        data['metadata'] = meta

        # Get file path for NPZ
        if 'general_metadata/file_path' in fov:
            fp = fov['general_metadata/file_path'][()]
            if isinstance(fp, bytes):
                fp = fp.decode('utf-8')
            data['npz_path'] = fp

    return data


def get_all_fov_expression_data(h5_path: str):
    """
    Get expression levels for all FOVs from H5 file.

    Returns DataFrame with fov_key, slide, mouse_model, age, region, channel,
    total_mrna, mrna_per_nucleus, n_clusters, n_nuclei
    """
    rows = []

    with h5py.File(h5_path, 'r') as f:
        fov_keys = list(f.keys())
        print(f"Scanning {len(fov_keys)} FOVs for expression levels...")

        for fov_key in fov_keys:
            try:
                fov = f[fov_key]

                # Extract slide
                slide = extract_slide_from_fov_key(fov_key)
                if slide in EXCLUDED_SLIDES:
                    continue

                # Filter for experimental probe set only (both Q111 and WT)
                if 'metadata_sample' in fov and 'Probe-Set' in fov['metadata_sample']:
                    ps = fov['metadata_sample']['Probe-Set'][()]
                    if isinstance(ps, np.ndarray) and len(ps) > 0:
                        ps = ps[0]
                    if isinstance(ps, bytes):
                        ps = ps.decode()
                    if ps != EXPERIMENTAL_FIELD:
                        continue
                else:
                    continue  # Skip FOVs without probe set metadata

                # Calculate nuclei count from DAPI volume (same as comprehensive_cortex_striatum_analysis_v2.py)
                n_nuclei = None
                if 'blue' in fov and 'label_sizes' in fov['blue']:
                    label_sizes = fov['blue']['label_sizes'][:]
                    if len(label_sizes) > 0:
                        V_DAPI = np.sum(label_sizes) * VOXEL_SIZE  # μm³
                        n_nuclei = V_DAPI / MEAN_NUCLEAR_VOLUME

                if n_nuclei is None or n_nuclei < 40:
                    continue  # Skip FOVs with too few nuclei

                # Get metadata
                meta = {}
                if 'metadata_sample' in fov:
                    sample_meta = fov['metadata_sample']

                    if 'Mouse Model' in sample_meta:
                        mm = sample_meta['Mouse Model'][()]
                        if isinstance(mm, np.ndarray) and len(mm) > 0:
                            mm = mm[0]
                        if isinstance(mm, bytes):
                            mm = mm.decode()
                        meta['mouse_model'] = mm

                    if 'Age' in sample_meta:
                        age_val = sample_meta['Age'][()]
                        if isinstance(age_val, np.ndarray) and len(age_val) > 0:
                            meta['age'] = int(age_val[0])
                        else:
                            meta['age'] = int(age_val)

                    # Region - use 'Slice Region' which has the actual brain region (Cortex/Striatum)
                    if 'Slice Region' in sample_meta:
                        reg = sample_meta['Slice Region'][()]
                        if isinstance(reg, np.ndarray) and len(reg) > 0:
                            reg = reg[0]
                        if isinstance(reg, bytes):
                            reg = reg.decode()
                        # Shorten region name to Cortex/Striatum
                        if 'Cortex' in reg:
                            meta['region'] = 'Cortex'
                        elif 'Striatum' in reg:
                            meta['region'] = 'Striatum'
                        else:
                            meta['region'] = reg
                    elif 'Region' in sample_meta:
                        # Fallback to Region field
                        reg = sample_meta['Region'][()]
                        if isinstance(reg, np.ndarray) and len(reg) > 0:
                            reg = reg[0]
                        if isinstance(reg, bytes):
                            reg = reg.decode()
                        meta['region'] = reg

                # Process each channel
                for channel in ['green', 'orange']:
                    if channel not in fov:
                        continue

                    ch_data = fov[channel]
                    if 'cluster_intensities' not in ch_data:
                        continue

                    intensities = ch_data['cluster_intensities'][:]
                    cluster_cvs = ch_data['cluster_cvs'][:] if 'cluster_cvs' in ch_data else None

                    # Get peak intensity for this slide/channel
                    peak_intensity = get_slide_peak_intensity(slide, channel)
                    if np.isnan(peak_intensity):
                        peak_intensity = 1000  # Fallback

                    # Get photon threshold for this slide/channel
                    threshold = get_photon_threshold(slide, channel)

                    # Apply intensity threshold AND CV filtering
                    n_total = len(intensities)
                    if len(intensities) > 0:
                        # Intensity threshold filter
                        if not np.isnan(threshold):
                            intensity_mask = intensities > threshold
                        else:
                            intensity_mask = np.ones(len(intensities), dtype=bool)

                        # CV filter (CV >= CV_THRESHOLD means good quality)
                        if cluster_cvs is not None and len(cluster_cvs) == len(intensities):
                            cv_mask = cluster_cvs >= CV_THRESHOLD
                        else:
                            raise ValueError(f"CV data missing for {fov_key}/{channel}")

                        # Combined filter
                        valid_mask = intensity_mask & cv_mask
                        filtered_intensities = intensities[valid_mask]
                        n_filtered = np.sum(valid_mask)
                        n_cv_filtered = np.sum(~cv_mask)
                    else:
                        filtered_intensities = intensities
                        n_filtered = 0
                        n_cv_filtered = 0

                    # Calculate total mRNA from filtered clusters
                    total_mrna = np.sum(filtered_intensities) / peak_intensity if len(filtered_intensities) > 0 else 0
                    mrna_per_nucleus = total_mrna / n_nuclei if n_nuclei > 0 else 0

                    rows.append({
                        'fov_key': fov_key,
                        'slide': slide,
                        'mouse_model': meta.get('mouse_model'),
                        'age': meta.get('age'),
                        'region': meta.get('region'),
                        'channel': channel,
                        'total_mrna': total_mrna,
                        'mrna_per_nucleus': mrna_per_nucleus,
                        'n_clusters': n_filtered,  # Count of clusters passing filters
                        'n_clusters_total': n_total,  # Total before filtering
                        'n_cv_filtered': n_cv_filtered,  # Count excluded by CV
                        'n_nuclei': n_nuclei,
                        'peak_intensity': peak_intensity,
                    })

            except Exception as e:
                continue

    df = pd.DataFrame(rows)
    print(f"  Found {len(df)} FOV-channel combinations")
    return df


def select_representative_fovs(df: pd.DataFrame, h5_path: str, channel: str = 'green', n_per_level: int = 3):
    """
    Select FOVs representing low, medium, and high expression levels.
    Only selects FOVs that have accessible NPZ files.

    Returns dict: {'low': [...], 'medium': [...], 'high': [...]}
    """
    selected = {'low': [], 'medium': [], 'high': []}

    # Filter to Q111 and specified channel
    df_filt = df[(df['mouse_model'] == 'Q111') & (df['channel'] == channel)].copy()

    if len(df_filt) == 0:
        print(f"Warning: No Q111 FOVs found for {channel}")
        return selected

    # Calculate percentiles
    p25 = df_filt['mrna_per_nucleus'].quantile(0.25)
    p50 = df_filt['mrna_per_nucleus'].quantile(0.50)
    p75 = df_filt['mrna_per_nucleus'].quantile(0.75)

    print(f"\nExpression percentiles for {channel} (mRNA/nucleus):")
    print(f"  P25: {p25:.2f}, P50: {p50:.2f}, P75: {p75:.2f}")

    # Define bins
    df_filt['level'] = pd.cut(
        df_filt['mrna_per_nucleus'],
        bins=[-np.inf, p25, p75, np.inf],
        labels=['low', 'medium', 'high']
    )

    # Select diverse FOVs for each level (only if NPZ exists)
    for level in ['low', 'medium', 'high']:
        level_df = df_filt[df_filt['level'] == level].copy()

        if len(level_df) == 0:
            continue

        # Sort by distance from level center
        if level == 'low':
            target = p25 / 2
        elif level == 'medium':
            target = p50
        else:
            target = (p75 + df_filt['mrna_per_nucleus'].max()) / 2

        level_df['dist_from_target'] = np.abs(level_df['mrna_per_nucleus'] - target)
        level_df = level_df.sort_values('dist_from_target')

        # Select from different slides, only if NPZ exists
        used_slides = set()
        for _, row in level_df.iterrows():
            slide = row['slide']
            fov_key = row['fov_key']

            # Check if NPZ file exists
            npz_path = get_npz_path_from_h5(fov_key, h5_path)
            if npz_path is None:
                continue

            if slide not in used_slides or len(selected[level]) < n_per_level:
                selected[level].append({
                    'fov_key': fov_key,
                    'mrna_per_nucleus': row['mrna_per_nucleus'],
                    'n_clusters': row['n_clusters'],
                    'n_nuclei': row['n_nuclei'],
                    'slide': slide,
                })
                used_slides.add(slide)
                if len(selected[level]) >= n_per_level:
                    break

    print(f"\nSelected FOVs (with accessible NPZ):")
    for level, fovs in selected.items():
        print(f"  {level}: {len(fovs)} FOVs")

    return selected


def add_text_overlay(image_rgb: np.ndarray, lines: list, position: str = 'top_left',
                     font_size: int = 16, padding: int = 10, bg_alpha: int = 180):
    """
    Add text overlay to image.

    Args:
        image_rgb: RGB image array (H, W, 3)
        lines: List of text lines
        position: 'top_left', 'top_right', 'bottom_left', 'bottom_right'
        font_size: Font size
        padding: Padding from edge
        bg_alpha: Background rectangle alpha (0-255)

    Returns:
        Image with text overlay
    """
    if not lines:
        return image_rgb

    # Convert to PIL Image
    img = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(img)

    # Try to load a nice font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except:
        font = ImageFont.load_default()

    text = '\n'.join(lines)

    # Get text bounding box
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Calculate position
    h, w = image_rgb.shape[:2]
    if position == 'top_left':
        x, y = padding, padding
    elif position == 'top_right':
        x, y = w - text_width - padding, padding
    elif position == 'bottom_left':
        x, y = padding, h - text_height - padding
    else:  # bottom_right
        x, y = w - text_width - padding, h - text_height - padding

    # Draw semi-transparent background
    bg_padding = 5
    draw.rectangle(
        [x - bg_padding, y - bg_padding,
         x + text_width + bg_padding, y + text_height + bg_padding],
        fill=(0, 0, 0, bg_alpha)
    )

    # Draw text
    draw.text((x, y), text, fill=(255, 255, 255), font=font)

    return np.array(img)


def draw_cluster_markers(image_rgb: np.ndarray, coms: np.ndarray,
                         color: tuple = (0, 255, 0), radius: int = 3):
    """
    Draw cluster center-of-mass markers on image.

    Args:
        image_rgb: RGB image (H, W, 3)
        coms: Cluster centers of mass, shape (N, 3) as (z, y, x)
        color: RGB color tuple
        radius: Marker radius

    Returns:
        Image with markers
    """
    if len(coms) == 0:
        return image_rgb

    img = image_rgb.copy()
    h, w = img.shape[:2]

    for com in coms:
        # COM is (z, y, x) - project to 2D (y, x)
        y, x = int(com[1]), int(com[2])

        if 0 <= y < h and 0 <= x < w:
            # Draw a small cross or circle
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if dy*dy + dx*dx <= radius*radius:  # Circle
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w:
                            img[ny, nx] = color

    return img


def draw_spot_markers(image_rgb: np.ndarray, spot_coords: np.ndarray,
                      color: tuple = (255, 255, 255), radius: int = 1):
    """
    Draw spot markers on image.

    Args:
        image_rgb: RGB image (H, W, 3)
        spot_coords: Spot coordinates, shape (N, 2) as (y, x)
        color: RGB color tuple
        radius: Marker radius

    Returns:
        Image with markers
    """
    if len(spot_coords) == 0:
        return image_rgb

    img = image_rgb.copy()
    h, w = img.shape[:2]

    for coord in spot_coords:
        y, x = int(coord[0]), int(coord[1])

        if 0 <= y < h and 0 <= x < w:
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if dy*dy + dx*dx <= radius*radius:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w:
                            img[ny, nx] = color

    return img


def select_zoom_regions(image_shape: tuple, coms: np.ndarray, n_zooms: int = 3,
                        zoom_size: int = 256, min_distance: int = 200, seed: int = 42):
    """
    Select zoom regions centered on areas with clusters.

    Args:
        image_shape: (H, W) of the image
        coms: Cluster centers of mass, shape (N, 3) as (z, y, x)
        n_zooms: Number of zoom regions to select
        zoom_size: Size of zoom region in pixels
        min_distance: Minimum distance between zoom centers
        seed: Random seed for reproducibility

    Returns:
        List of (cy, cx) tuples for zoom centers
    """
    np.random.seed(seed)  # Set seed for reproducible selection
    h, w = image_shape
    half = zoom_size // 2
    centers = []

    if len(coms) == 0:
        # No clusters - select evenly spaced regions
        for i in range(n_zooms):
            cy = h // 2
            cx = int(w * (i + 1) / (n_zooms + 1))
            centers.append((cy, cx))
        return centers

    # Sort clusters by density (prefer areas with more clusters nearby)
    coms_2d = coms[:, 1:3]  # (y, x) only

    # Try to select diverse regions with clusters
    used_coms = set()
    for _ in range(n_zooms):
        best_com = None
        best_score = -1

        for i, (y, x) in enumerate(coms_2d):
            if i in used_coms:
                continue

            # Check bounds
            if y < half or y >= h - half or x < half or x >= w - half:
                continue

            # Check distance from existing centers
            too_close = False
            for cy, cx in centers:
                if np.sqrt((y - cy)**2 + (x - cx)**2) < min_distance:
                    too_close = True
                    break

            if too_close:
                continue

            # Score by number of nearby clusters
            nearby = np.sum(np.sqrt((coms_2d[:, 0] - y)**2 + (coms_2d[:, 1] - x)**2) < zoom_size)
            if nearby > best_score:
                best_score = nearby
                best_com = (int(y), int(x), i)

        if best_com is not None:
            centers.append((best_com[0], best_com[1]))
            used_coms.add(best_com[2])

    # Fill remaining with random valid positions if needed
    while len(centers) < n_zooms:
        cy = np.random.randint(half, h - half)
        cx = np.random.randint(half, w - half)
        centers.append((cy, cx))

    return centers


def extract_zoom(image_rgb: np.ndarray, center: tuple, zoom_size: int = 256):
    """Extract a zoom region from an image."""
    cy, cx = center
    half = zoom_size // 2
    h, w = image_rgb.shape[:2]

    # Calculate bounds with clamping
    y1 = max(0, cy - half)
    y2 = min(h, cy + half)
    x1 = max(0, cx - half)
    x2 = min(w, cx + half)

    zoom = image_rgb[y1:y2, x1:x2].copy()

    # Pad if necessary to get exact size
    if zoom.shape[0] < zoom_size or zoom.shape[1] < zoom_size:
        padded = np.zeros((zoom_size, zoom_size, 3), dtype=zoom.dtype)
        padded[:zoom.shape[0], :zoom.shape[1]] = zoom
        zoom = padded

    return zoom


def apply_gamma_to_colored_image(image_rgb: np.ndarray, gamma: float = 2.2) -> np.ndarray:
    """
    Apply gamma correction to a colored RGB image.

    Args:
        image_rgb: RGB image (H, W, 3), uint8
        gamma: Gamma value. >1 brightens dim values, <1 darkens them.
               1.0 = no change, 2.0-2.5 good for seeing dim spots.

    Returns:
        Gamma-corrected RGB image (uint8)
    """
    if gamma == 1.0:
        return image_rgb

    # Convert to float, apply gamma, convert back
    img_float = image_rgb.astype(np.float32) / 255.0
    inv_gamma = 1.0 / gamma
    img_gamma = np.power(img_float, inv_gamma)
    return (img_gamma * 255).astype(np.uint8)


def create_composite_three_channel(dapi_mip: np.ndarray, green_mip: np.ndarray, orange_mip: np.ndarray,
                                    dapi_pmin: float = 10, dapi_pmax: float = 99,
                                    signal_pmin: float = 50, signal_pmax: float = 99.9,
                                    gamma: float = 1.0):
    """
    Create a colored RGB composite image from DAPI (blue), green, and orange channels.

    Args:
        dapi_mip: DAPI MIP (2D grayscale)
        green_mip: Green channel MIP (2D grayscale)
        orange_mip: Orange channel MIP (2D grayscale)
        dapi_pmin: Lower percentile for DAPI contrast
        dapi_pmax: Upper percentile for DAPI contrast
        signal_pmin: Lower percentile for signal contrast
        signal_pmax: Upper percentile for signal contrast
        gamma: Gamma correction value. >1 brightens dim spots, <1 darkens them.
               1.0 = linear (no correction), 2.0-2.5 good for seeing dim spots.

    Returns:
        dict with:
            'composite': Colored RGB composite image (8-bit)
            'dapi_range': (low, high) raw intensity values for DAPI
            'green_range': (low, high) raw intensity values for green
            'orange_range': (low, high) raw intensity values for orange
            'gamma': gamma value used
    """
    # Scale DAPI to 0-1 range
    dapi_low = np.percentile(dapi_mip, dapi_pmin)
    dapi_high = np.percentile(dapi_mip, dapi_pmax)
    dapi_scaled = np.clip((dapi_mip.astype(np.float32) - dapi_low) / (dapi_high - dapi_low + 1e-6), 0, 1)

    # Scale green to 0-1 range
    green_low = np.percentile(green_mip, signal_pmin)
    green_high = np.percentile(green_mip, signal_pmax)
    green_scaled = np.clip((green_mip.astype(np.float32) - green_low) / (green_high - green_low + 1e-6), 0, 1)

    # Scale orange to 0-1 range
    orange_low = np.percentile(orange_mip, signal_pmin)
    orange_high = np.percentile(orange_mip, signal_pmax)
    orange_scaled = np.clip((orange_mip.astype(np.float32) - orange_low) / (orange_high - orange_low + 1e-6), 0, 1)

    # Apply gamma correction to signal channels (not DAPI)
    # Gamma > 1 brightens dim values: output = input^(1/gamma)
    if gamma != 1.0:
        inv_gamma = 1.0 / gamma
        green_scaled = np.power(green_scaled, inv_gamma)
        orange_scaled = np.power(orange_scaled, inv_gamma)

    # Create colored RGB composite: Blue=DAPI, Green=green channel, Red+0.65*Green=orange
    composite = np.zeros((*dapi_mip.shape, 3), dtype=np.float32)

    # Blue channel = DAPI
    composite[:, :, 2] = dapi_scaled

    # Green channel = green signal
    composite[:, :, 1] = green_scaled

    # Orange (R + 0.65*G) added to red and green
    composite[:, :, 0] = np.clip(composite[:, :, 0] + orange_scaled, 0, 1)
    composite[:, :, 1] = np.clip(composite[:, :, 1] + orange_scaled * 0.65, 0, 1)

    # Convert to 8-bit
    composite_8bit = (composite * 255).astype(np.uint8)

    return {
        'composite': composite_8bit,
        'dapi_range': (float(dapi_low), float(dapi_high)),
        'green_range': (float(green_low), float(green_high)),
        'orange_range': (float(orange_low), float(orange_high)),
        'gamma': gamma,
    }


def create_composite_image(dapi_mip: np.ndarray, signal_mip: np.ndarray, channel: str,
                           dapi_pmin: float = 5, dapi_pmax: float = 95,
                           signal_pmin: float = 1, signal_pmax: float = 99.5):
    """
    Create a composite RGB image with DAPI (blue) and one signal channel (green or orange).

    Keeps 16-bit precision internally, only converts to 8-bit at the end for display.

    Args:
        dapi_mip: DAPI MIP (2D grayscale)
        signal_mip: Signal channel MIP (2D grayscale)
        channel: 'green' or 'orange' - determines color mapping
        dapi_pmin: Lower percentile for DAPI contrast
        dapi_pmax: Upper percentile for DAPI contrast
        signal_pmin: Lower percentile for signal contrast
        signal_pmax: Upper percentile for signal contrast

    Returns:
        RGB composite image (8-bit for display)
    """
    # Scale to 0-1 range using percentiles (keeps full precision)
    dapi_low = np.percentile(dapi_mip, dapi_pmin)
    dapi_high = np.percentile(dapi_mip, dapi_pmax)
    dapi_scaled = np.clip((dapi_mip.astype(np.float32) - dapi_low) / (dapi_high - dapi_low + 1e-6), 0, 1)

    signal_low = np.percentile(signal_mip, signal_pmin)
    signal_high = np.percentile(signal_mip, signal_pmax)
    signal_scaled = np.clip((signal_mip.astype(np.float32) - signal_low) / (signal_high - signal_low + 1e-6), 0, 1)

    # Create RGB composite in float
    composite = np.zeros((*dapi_mip.shape, 3), dtype=np.float32)
    composite[:, :, 2] = dapi_scaled  # Blue = DAPI

    if channel == 'green':
        composite[:, :, 1] = signal_scaled  # Green channel
    else:  # orange
        # Orange = Red + some Green
        composite[:, :, 0] = signal_scaled  # Red
        composite[:, :, 1] = signal_scaled * 0.5  # Half green for orange tint

    # Convert to 8-bit only at the end
    composite_8bit = (composite * 255).astype(np.uint8)

    return composite_8bit


def generate_fov_images(h5_path: str, fov_key: str,
                        expression_level: str,
                        mrna_per_nucleus_green: float, mrna_per_nucleus_orange: float,
                        n_clusters_green: int, n_clusters_orange: int,
                        n_nuclei: int, metadata: dict,
                        output_dir: Path, n_zooms: int = 3):
    """
    Generate all visualizations for a single FOV.

    Creates:
    1. Per-channel images (clean + annotated) for green and orange
    2. Composite image (DAPI + green + orange merged)
    3. Zoom images with scale bars
    """
    # Get NPZ path
    npz_path = get_npz_path_from_h5(fov_key, h5_path)
    if npz_path is None:
        print(f"  Could not find NPZ for {fov_key}")
        return None

    # Load raw image (all channels)
    try:
        image_4d, _ = load_npz_image(npz_path)
    except Exception as e:
        print(f"  Could not load NPZ: {e}")
        return None

    # Extract individual channels
    dapi_3d = image_4d[CHANNEL_MAP['blue']]
    green_3d = image_4d[CHANNEL_MAP['green']]
    orange_3d = image_4d[CHANNEL_MAP['orange']]

    # Create MIPs
    dapi_mip = create_mip(dapi_3d)
    green_mip = create_mip(green_3d)
    orange_mip = create_mip(orange_3d)

    # Load spot/cluster data from H5
    green_data = load_fov_data_from_h5(h5_path, fov_key, 'green')
    orange_data = load_fov_data_from_h5(h5_path, fov_key, 'orange')

    # Build condition text overlay
    lines = []
    if 'mouse_model' in metadata:
        lines.append(metadata['mouse_model'])
    if 'age' in metadata:
        lines.append(f"{metadata['age']}mo")
    if 'region' in metadata:
        lines.append(metadata['region'])

    # Create output directory
    fov_dir = output_dir / expression_level
    fov_dir.mkdir(parents=True, exist_ok=True)
    safe_name = fov_key.replace('/', '_').replace('--', '_')[:60]

    # =========================================================================
    # 1. IMAGES: Green only, Orange only, and DAPI+Green+Orange composite
    #    Generate both LINEAR and GAMMA-corrected versions
    # =========================================================================
    # Channel-specific expression text
    expr_lines_green = [f"mHTT1a: {mrna_per_nucleus_green:.1f} mRNA/nuc"]
    expr_lines_orange = [f"fl-mHTT: {mrna_per_nucleus_orange:.1f} mRNA/nuc"]
    expr_lines_all = [f"mHTT1a: {mrna_per_nucleus_green:.1f}", f"fl-mHTT: {mrna_per_nucleus_orange:.1f}"]

    # Green channel only (no DAPI) - raw TIFF without overlays (added in SVG)
    green_8bit = normalize_to_8bit(green_mip)
    green_colored = create_colored_image(green_8bit, 'green')
    green_path = fov_dir / f"{safe_name}_green.tif"
    imwrite(green_path, green_colored)
    print(f"  Saved: {green_path.name}")

    # Green channel - GAMMA corrected version
    green_gamma = apply_gamma_to_colored_image(green_colored, gamma=GAMMA_VALUE)
    green_gamma_path = fov_dir / f"{safe_name}_green_gamma.tif"
    imwrite(green_gamma_path, green_gamma)
    print(f"  Saved: {green_gamma_path.name}")

    # Orange channel only (no DAPI) - raw TIFF without overlays (added in SVG)
    orange_8bit = normalize_to_8bit(orange_mip)
    orange_colored = create_colored_image(orange_8bit, 'orange')
    orange_path = fov_dir / f"{safe_name}_orange.tif"
    imwrite(orange_path, orange_colored)
    print(f"  Saved: {orange_path.name}")

    # Orange channel - GAMMA corrected version
    orange_gamma = apply_gamma_to_colored_image(orange_colored, gamma=GAMMA_VALUE)
    orange_gamma_path = fov_dir / f"{safe_name}_orange_gamma.tif"
    imwrite(orange_gamma_path, orange_gamma)
    print(f"  Saved: {orange_gamma_path.name}")

    # DAPI + Green + Orange composite (all three channels) - LINEAR version
    composite_result = create_composite_three_channel(dapi_mip, green_mip, orange_mip,
                                                       dapi_pmin=10, dapi_pmax=99,
                                                       signal_pmin=50, signal_pmax=99.9,
                                                       gamma=1.0)
    composite_all = composite_result['composite']
    composite_all_path = fov_dir / f"{safe_name}_all.tif"
    imwrite(composite_all_path, composite_all)
    print(f"  Saved: {composite_all_path.name}")

    # DAPI + Green + Orange composite - GAMMA corrected version
    composite_result_gamma = create_composite_three_channel(dapi_mip, green_mip, orange_mip,
                                                             dapi_pmin=10, dapi_pmax=99,
                                                             signal_pmin=50, signal_pmax=99.9,
                                                             gamma=GAMMA_VALUE)
    composite_all_gamma = composite_result_gamma['composite']
    composite_all_gamma_path = fov_dir / f"{safe_name}_all_gamma.tif"
    imwrite(composite_all_gamma_path, composite_all_gamma)
    print(f"  Saved: {composite_all_gamma_path.name}")

    # Store dynamic range info for SVG annotations (from linear version)
    dynamic_range_info = {
        'dapi_range': composite_result['dapi_range'],
        'green_range': composite_result['green_range'],
        'orange_range': composite_result['orange_range'],
        'gamma': GAMMA_VALUE,
    }

    # Save metadata JSON with mRNA values and dynamic range info
    slide_name = extract_slide_from_fov_key(fov_key)
    fov_metadata = {
        'fov_key': fov_key,
        'green_mrna_per_nucleus': float(mrna_per_nucleus_green),
        'orange_mrna_per_nucleus': float(mrna_per_nucleus_orange),
        'n_nuclei': float(n_nuclei),
        'slide': slide_name,
        'mouse_model': metadata.get('mouse_model'),
        'age': int(metadata.get('age')) if metadata.get('age') is not None else None,
        'region': metadata.get('region'),
        'image_height': int(dapi_mip.shape[0]),
        'image_width': int(dapi_mip.shape[1]),
        'dynamic_range': dynamic_range_info,
    }
    metadata_path = fov_dir / f"{safe_name}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(fov_metadata, f, indent=2)

    # Save human-readable metadata.txt
    txt_path = fov_dir / f"{safe_name}_metadata.txt"
    with open(txt_path, 'w') as f:
        f.write(f"FOV Key: {fov_key}\n")
        f.write(f"Slide: {slide_name}\n")
        f.write(f"Mouse Model: {metadata.get('mouse_model', 'N/A')}\n")
        f.write(f"Age: {metadata.get('age', 'N/A')} months\n")
        f.write(f"Region: {metadata.get('region', 'N/A')}\n")
        f.write(f"Number of Nuclei: {n_nuclei:.1f}\n")
        f.write(f"\nmHTT1a (green): {mrna_per_nucleus_green:.2f} mRNA/nucleus\n")
        f.write(f"Full-length mHTT (orange): {mrna_per_nucleus_orange:.2f} mRNA/nucleus\n")

    # =========================================================================
    # 2. ZOOM IMAGES (green only, orange only, and composite all)
    # =========================================================================
    # Select zoom regions based on green channel clusters
    zoom_centers = select_zoom_regions(
        green_mip.shape, green_data['coms'] if green_data else np.array([]),
        n_zooms=n_zooms, zoom_size=ZOOM_SIZE, min_distance=300
    )

    zoom_dir = fov_dir / f"{safe_name}_zooms"
    zoom_dir.mkdir(exist_ok=True)

    half = ZOOM_SIZE // 2
    h, w = green_mip.shape

    # Collect zoom info for JSON export (for SVG zoom boxes)
    zoom_info_list = []

    for i, (cy, cx) in enumerate(zoom_centers):
        # Calculate zoom bounds
        y1 = max(0, cy - half)
        y2 = min(h, cy + half)
        x1 = max(0, cx - half)
        x2 = min(w, cx + half)

        # Store zoom coordinates for SVG zoom boxes
        zoom_info_list.append({
            'zoom_id': i + 1,
            'center_y': int(cy),
            'center_x': int(cx),
            'y1': int(y1),
            'y2': int(y2),
            'x1': int(x1),
            'x2': int(x2),
            'size': ZOOM_SIZE,
        })

        # Extract raw zoom regions from MIPs
        green_zoom_raw = green_mip[y1:y2, x1:x2]
        orange_zoom_raw = orange_mip[y1:y2, x1:x2]
        dapi_zoom_raw = dapi_mip[y1:y2, x1:x2]

        # Normalize each zoom locally for better contrast
        green_zoom_8bit = normalize_to_8bit(green_zoom_raw)
        orange_zoom_8bit = normalize_to_8bit(orange_zoom_raw)

        # Create colored images from locally normalized zooms - LINEAR
        green_zoom = create_colored_image(green_zoom_8bit, 'green')
        orange_zoom = create_colored_image(orange_zoom_8bit, 'orange')

        # Create GAMMA corrected versions
        green_zoom_gamma = apply_gamma_to_colored_image(green_zoom, gamma=GAMMA_VALUE)
        orange_zoom_gamma = apply_gamma_to_colored_image(orange_zoom, gamma=GAMMA_VALUE)

        # Composite zoom - LINEAR version
        composite_zoom_result = create_composite_three_channel(
            dapi_zoom_raw, green_zoom_raw, orange_zoom_raw,
            dapi_pmin=10, dapi_pmax=99,
            signal_pmin=50, signal_pmax=99.9,
            gamma=1.0
        )
        composite_all_zoom = composite_zoom_result['composite']

        # Composite zoom - GAMMA version
        composite_zoom_gamma_result = create_composite_three_channel(
            dapi_zoom_raw, green_zoom_raw, orange_zoom_raw,
            dapi_pmin=10, dapi_pmax=99,
            signal_pmin=50, signal_pmax=99.9,
            gamma=GAMMA_VALUE
        )
        composite_all_zoom_gamma = composite_zoom_gamma_result['composite']

        # Pad if necessary to get exact size
        if green_zoom.shape[0] < ZOOM_SIZE or green_zoom.shape[1] < ZOOM_SIZE:
            padded = np.zeros((ZOOM_SIZE, ZOOM_SIZE, 3), dtype=green_zoom.dtype)
            padded[:green_zoom.shape[0], :green_zoom.shape[1]] = green_zoom
            green_zoom = padded

            padded = np.zeros((ZOOM_SIZE, ZOOM_SIZE, 3), dtype=green_zoom_gamma.dtype)
            padded[:green_zoom_gamma.shape[0], :green_zoom_gamma.shape[1]] = green_zoom_gamma
            green_zoom_gamma = padded

            padded = np.zeros((ZOOM_SIZE, ZOOM_SIZE, 3), dtype=orange_zoom.dtype)
            padded[:orange_zoom.shape[0], :orange_zoom.shape[1]] = orange_zoom
            orange_zoom = padded

            padded = np.zeros((ZOOM_SIZE, ZOOM_SIZE, 3), dtype=orange_zoom_gamma.dtype)
            padded[:orange_zoom_gamma.shape[0], :orange_zoom_gamma.shape[1]] = orange_zoom_gamma
            orange_zoom_gamma = padded

            padded = np.zeros((ZOOM_SIZE, ZOOM_SIZE, 3), dtype=composite_all_zoom.dtype)
            padded[:composite_all_zoom.shape[0], :composite_all_zoom.shape[1]] = composite_all_zoom
            composite_all_zoom = padded

            padded = np.zeros((ZOOM_SIZE, ZOOM_SIZE, 3), dtype=composite_all_zoom_gamma.dtype)
            padded[:composite_all_zoom_gamma.shape[0], :composite_all_zoom_gamma.shape[1]] = composite_all_zoom_gamma
            composite_all_zoom_gamma = padded

        # Save zooms WITHOUT scale bars (scale bars added in SVG only)
        # LINEAR versions
        imwrite(zoom_dir / f"zoom{i+1}_green.tif", green_zoom)
        imwrite(zoom_dir / f"zoom{i+1}_orange.tif", orange_zoom)
        imwrite(zoom_dir / f"zoom{i+1}_all.tif", composite_all_zoom)
        # GAMMA versions
        imwrite(zoom_dir / f"zoom{i+1}_green_gamma.tif", green_zoom_gamma)
        imwrite(zoom_dir / f"zoom{i+1}_orange_gamma.tif", orange_zoom_gamma)
        imwrite(zoom_dir / f"zoom{i+1}_all_gamma.tif", composite_all_zoom_gamma)

    # Save zoom coordinates JSON (for SVG zoom box drawing)
    zoom_json_path = fov_dir / f"{safe_name}_zoom_coords.json"
    with open(zoom_json_path, 'w') as f:
        json.dump({
            'image_height': int(h),
            'image_width': int(w),
            'zoom_size': ZOOM_SIZE,
            'zooms': zoom_info_list,
        }, f, indent=2)

    print(f"  Saved: {n_zooms} zooms to {zoom_dir.name}/")

    return green_path


def generate_fov_image(h5_path: str, fov_key: str, channel: str,
                       expression_level: str, mrna_per_nucleus: float,
                       n_clusters: int, n_nuclei: int, metadata: dict,
                       output_dir: Path):
    """
    Generate visualization for a single FOV (legacy single-channel function).

    Creates two images:
    1. Clean image with just MIP + text overlays
    2. Annotated image with spots + cluster markers

    Uses proper filtering from H5 data (final_filter for spots).
    """
    # Get NPZ path
    npz_path = get_npz_path_from_h5(fov_key, h5_path)
    if npz_path is None:
        print(f"  Could not find NPZ for {fov_key}")
        return None

    # Load raw image
    try:
        image_4d, _ = load_npz_image(npz_path)
        ch_idx = CHANNEL_MAP[channel]
        image_3d = image_4d[ch_idx]
    except Exception as e:
        print(f"  Could not load NPZ: {e}")
        return None

    # Load spot/cluster data from H5
    data = load_fov_data_from_h5(h5_path, fov_key, channel)

    # Create MIP and colorize
    mip = create_mip(image_3d)
    mip_8bit = normalize_to_8bit(mip)
    colored_img = create_colored_image(mip_8bit, channel)

    # Build condition text overlay (top left)
    lines = []
    if 'mouse_model' in metadata:
        lines.append(metadata['mouse_model'])
    if 'age' in metadata:
        lines.append(f"{metadata['age']}mo")
    if 'region' in metadata:
        lines.append(metadata['region'])

    # Expression info (bottom right)
    expr_lines = [
        f"{mrna_per_nucleus:.1f} mRNA/nucleus",
        f"{n_clusters} clusters | {n_nuclei} nuclei"
    ]

    # Create output directory
    fov_dir = output_dir / expression_level
    fov_dir.mkdir(parents=True, exist_ok=True)
    safe_name = fov_key.replace('/', '_').replace('--', '_')[:60]

    # ----- 1. Clean image (no markers) -----
    clean_img = colored_img.copy()
    clean_img = draw_scale_bar(clean_img, scale_um=SCALE_BAR_OVERVIEW)
    clean_img = add_text_overlay(clean_img, lines, position='top_left', font_size=18)
    clean_img = add_text_overlay(clean_img, expr_lines, position='bottom_right',
                                 font_size=14, padding=60)

    clean_path = fov_dir / f"{safe_name}_{channel}_clean.tif"
    imwrite(clean_path, clean_img)
    print(f"  Saved: {clean_path.name}")

    # ----- 2. Annotated image (with spots + clusters) -----
    if data is not None:
        annotated_img = colored_img.copy()

        # Draw cluster markers (center of mass)
        cluster_color = (0, 255, 0) if channel == 'green' else (255, 165, 0)
        if len(data['coms']) > 0:
            annotated_img = draw_cluster_markers(annotated_img, data['coms'],
                                                  color=cluster_color, radius=4)

        # Draw spot markers (filtered spots only)
        if len(data['spot_coords']) > 0:
            annotated_img = draw_spot_markers(annotated_img, data['spot_coords'],
                                              color=(255, 255, 255), radius=1)

        annotated_img = draw_scale_bar(annotated_img, scale_um=SCALE_BAR_OVERVIEW)
        annotated_img = add_text_overlay(annotated_img, lines, position='top_left', font_size=18)
        annotated_img = add_text_overlay(annotated_img, expr_lines, position='bottom_right',
                                         font_size=14, padding=60)

        annotated_path = fov_dir / f"{safe_name}_{channel}_annotated.tif"
        imwrite(annotated_path, annotated_img)
        print(f"  Saved: {annotated_path.name}")

    return clean_path


def main():
    """Generate Figure 3 panel images."""

    print("=" * 70)
    print("FIGURE 3 PANEL GENERATION")
    print("FOVs with different total mRNA per nucleus levels")
    print("=" * 70)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Find H5 file
    h5_path = find_h5_file()
    print(f"\nUsing H5 file: {h5_path}")

    # Get expression data for all FOVs
    print("\nLoading expression data from H5...")
    df = get_all_fov_expression_data(h5_path)

    if len(df) == 0:
        print("ERROR: No FOV data found")
        return

    # Select representative FOVs for each expression level
    print("\nSelecting representative FOVs...")
    selected = select_representative_fovs(df, h5_path, channel='green', n_per_level=15)

    # Generate images
    print("\n" + "=" * 70)
    print("GENERATING FOV IMAGES")
    print("=" * 70)

    for level, fovs in selected.items():
        print(f"\n--- {level.upper()} expression ---")

        for fov_info in fovs:
            fov_key = fov_info['fov_key']
            mrna = fov_info['mrna_per_nucleus']
            n_clusters = fov_info['n_clusters']
            n_nuclei = fov_info['n_nuclei']
            slide = fov_info['slide']

            print(f"\n{fov_key[:50]}...")

            # Get metadata for this FOV from the full dataframe
            fov_df = df[df['fov_key'] == fov_key].iloc[0]
            metadata = {
                'mouse_model': fov_df['mouse_model'],
                'age': fov_df['age'],
                'region': fov_df['region'],
            }

            # Get cluster counts and mRNA/nucleus for each channel
            green_df = df[(df['fov_key'] == fov_key) & (df['channel'] == 'green')]
            orange_df = df[(df['fov_key'] == fov_key) & (df['channel'] == 'orange')]
            n_clusters_green = green_df['n_clusters'].values[0] if len(green_df) > 0 else 0
            n_clusters_orange = orange_df['n_clusters'].values[0] if len(orange_df) > 0 else 0
            mrna_green = green_df['mrna_per_nucleus'].values[0] if len(green_df) > 0 else 0
            mrna_orange = orange_df['mrna_per_nucleus'].values[0] if len(orange_df) > 0 else 0

            print(f"  mHTT1a: {mrna_green:.2f}, fl-mHTT: {mrna_orange:.2f} mRNA/nuc")

            # Generate all images (composite, per-channel, zooms)
            generate_fov_images(
                h5_path, fov_key, level,
                mrna_green, mrna_orange,
                n_clusters_green, n_clusters_orange, n_nuclei,
                metadata, OUTPUT_DIR, n_zooms=5
            )

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for level, fovs in selected.items():
        print(f"\n{level.upper()} ({len(fovs)} FOVs):")
        for fov_info in fovs:
            print(f"  {fov_info['fov_key'][:40]}... | {fov_info['mrna_per_nucleus']:.1f} mRNA/nuc")

    print(f"\nOutput saved to: {OUTPUT_DIR}")
    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
