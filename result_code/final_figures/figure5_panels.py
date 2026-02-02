#!/usr/bin/env python3
"""
Figure 5 Panel Generation Script

Generates example FOV images for Figure 5 showing:
- Normal Q111 FOVs (low/median expression within the same animal)
- Extreme Q111 FOVs (high clustered mRNA expression)
- Paired comparison from the SAME animals/slides
- Condition information (genotype, age, region, mouse/slide)
- Clustered mRNA per nucleus values

This allows direct comparison of normal vs extreme FOVs within Q111 mice.

Output:
    example_images/figure5/
        normal/
        extreme/
"""

import os
import sys
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
    CHANNEL_MAP,
    PIXEL_SIZE_XY,
    VOXEL_VOLUME_UM3,
    SCALE_BAR_OVERVIEW,
    SCALE_BAR_ZOOM,
    ZOOM_SIZE,
)

# Import from figure3_panels (reuse helper functions)
from figure3_panels import (
    load_fov_data_from_h5,
    add_text_overlay,
    select_zoom_regions,
    create_composite_three_channel,
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
OUTPUT_DIR = Path(__file__).parent / 'output' / 'example_images' / 'figure5'

# Path to pre-computed thresholds from negative control
THRESHOLD_CSV = Path(__file__).parent / 'output' / 'photon_thresholds.csv'


def load_thresholds():
    """
    Load pre-computed thresholds from negative control analysis.

    Returns dict mapping (slide, channel) -> threshold value
    """
    if not THRESHOLD_CSV.exists():
        raise FileNotFoundError(f"Threshold file not found: {THRESHOLD_CSV}")

    thresholds = {}
    df = pd.read_csv(THRESHOLD_CSV)
    for _, row in df.iterrows():
        # Parse the key tuple string, e.g., "('m3a1', 'green', None)"
        key_str = row['key']
        # Extract slide and channel using string parsing
        parts = key_str.strip("()").replace("'", "").split(", ")
        slide = parts[0]
        channel = parts[1]
        thresholds[(slide, channel)] = row['threshold']
    return thresholds


def get_all_fov_expression_data(h5_path: str):
    """
    Get expression levels for all FOVs from H5 file.

    Uses negative control thresholds to filter clusters.
    Only clusters with intensity > threshold are counted.
    Same logic as comprehensive_cortex_striatum_analysis_v2.py.

    Returns DataFrame with fov_key, slide, mouse_model, age, region, channel,
    total_mrna, mrna_per_nucleus, n_clusters, n_nuclei
    """
    # Load thresholds from negative control
    thresholds = load_thresholds()
    print(f"Loaded thresholds for {len(thresholds)} slide-channel combinations")

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

                    # Get mouse ID for pairing
                    if 'mouse ID ' in sample_meta:
                        mouse_id = sample_meta['mouse ID '][()]
                        if isinstance(mouse_id, np.ndarray) and len(mouse_id) > 0:
                            mouse_id = mouse_id[0]
                        if isinstance(mouse_id, bytes):
                            mouse_id = mouse_id.decode()
                        meta['mouse_id'] = mouse_id

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
                    # Skip if no peak intensity (same as comprehensive_cortex_striatum_analysis_v2.py)
                    try:
                        peak_intensity = get_slide_peak_intensity(slide, channel)
                    except ValueError:
                        continue  # Skip FOVs without peak intensity data

                    if np.isnan(peak_intensity):
                        continue  # Skip FOVs without peak intensity

                    # Get threshold for this slide/channel from negative control
                    threshold_key = (slide, channel)
                    if threshold_key not in thresholds:
                        continue  # Skip if no threshold available
                    threshold = thresholds[threshold_key]

                    # Only count clusters above threshold (from negative control) AND with CV >= CV_THRESHOLD
                    intensity_mask = intensities > threshold
                    # CV data is required - no fallback
                    if cluster_cvs is None or len(cluster_cvs) != len(intensities):
                        raise ValueError(f"CV data missing or mismatched for cluster filtering")
                    cv_mask = cluster_cvs >= CV_THRESHOLD
                    above_threshold = intensity_mask & cv_mask
                    n_clusters = above_threshold.sum()
                    intensities_above = intensities[above_threshold]

                    # Calculate clustered mRNA: sum intensities above threshold / peak
                    total_mrna = np.sum(intensities_above) / peak_intensity if n_clusters > 0 else 0
                    mrna_per_nucleus = total_mrna / n_nuclei if n_nuclei > 0 else 0

                    rows.append({
                        'fov_key': fov_key,
                        'slide': slide,
                        'mouse_model': meta.get('mouse_model'),
                        'mouse_id': meta.get('mouse_id'),
                        'age': meta.get('age'),
                        'region': meta.get('region'),
                        'channel': channel,
                        'total_mrna': total_mrna,
                        'mrna_per_nucleus': mrna_per_nucleus,
                        'n_clusters': n_clusters,
                        'n_nuclei': n_nuclei,
                        'peak_intensity': peak_intensity,
                        'threshold': threshold,
                    })

            except Exception as e:
                continue

    df = pd.DataFrame(rows)
    print(f"  Found {len(df)} FOV-channel combinations")
    return df


def select_paired_fovs(df: pd.DataFrame, h5_path: str, channel: str = 'green',
                       n_pairs: int = 4, seed: int = 42):
    """
    Select paired normal and extreme Q111 FOVs from the SAME animals/slides.

    For each selected slide, picks MULTIPLE pairs from different regions/subregions:
    - One extreme FOV (high expression, top quartile within slide)
    - One normal FOV (low/median expression, bottom half within slide)

    Pairs can be from different subregions within the same animal (e.g., striatum
    lower left and striatum upper right) to maximize the number of unique pairs.

    Args:
        df: DataFrame with all FOV expression data
        h5_path: Path to H5 file
        channel: Channel to use for selection
        n_pairs: Number of pairs to select
        seed: Random seed for reproducibility

    Returns:
        Tuple of (normal_fovs, extreme_fovs) - lists of dicts with FOV info
    """
    np.random.seed(seed)

    # Filter to Q111 and specified channel
    df_q111 = df[(df['mouse_model'] == 'Q111') & (df['channel'] == channel)].copy()

    if len(df_q111) == 0:
        print(f"Warning: No Q111 FOVs found for {channel}")
        return [], []

    # Sort by fov_key for deterministic order
    df_q111 = df_q111.sort_values('fov_key')

    # Get unique slides with enough FOVs for comparison
    slides_with_counts = df_q111.groupby('slide').size()
    valid_slides = slides_with_counts[slides_with_counts >= 10].index.tolist()  # Need at least 10 FOVs per slide

    print(f"  Q111 slides with >=10 FOVs: {len(valid_slides)}")

    normal_fovs = []
    extreme_fovs = []
    used_fov_keys = set()  # Track used FOVs to avoid duplicates

    # Sort slides for deterministic order
    valid_slides = sorted(valid_slides)

    # First pass: get one pair per slide (highest contrast)
    for slide in valid_slides:
        if len(normal_fovs) >= n_pairs:
            break

        # Get all FOVs from this slide
        slide_df = df_q111[df_q111['slide'] == slide].copy()
        slide_df = slide_df.sort_values('mrna_per_nucleus')

        # Calculate quartiles within this slide
        q25 = slide_df['mrna_per_nucleus'].quantile(0.25)
        q75 = slide_df['mrna_per_nucleus'].quantile(0.75)

        # Select normal FOV (bottom quartile)
        normal_candidates = slide_df[slide_df['mrna_per_nucleus'] <= q25]
        # Select extreme FOV (top quartile)
        extreme_candidates = slide_df[slide_df['mrna_per_nucleus'] >= q75]

        if len(normal_candidates) == 0 or len(extreme_candidates) == 0:
            continue

        # Find valid normal FOV (with NPZ file)
        normal_selected = None
        for _, row in normal_candidates.iterrows():
            fov_key = row['fov_key']
            if fov_key in used_fov_keys:
                continue
            npz_path = get_npz_path_from_h5(fov_key, h5_path)
            if npz_path is not None:
                normal_selected = {
                    'fov_key': fov_key,
                    'mrna_per_nucleus': row['mrna_per_nucleus'],
                    'n_clusters': row['n_clusters'],
                    'n_nuclei': row['n_nuclei'],
                    'slide': slide,
                    'region': row['region'],
                    'mouse_id': row.get('mouse_id'),
                }
                break

        # Find valid extreme FOV (with NPZ file) - pick highest expression
        extreme_candidates = extreme_candidates.sort_values('mrna_per_nucleus', ascending=False)
        extreme_selected = None
        for _, row in extreme_candidates.iterrows():
            fov_key = row['fov_key']
            if fov_key in used_fov_keys:
                continue
            npz_path = get_npz_path_from_h5(fov_key, h5_path)
            if npz_path is not None:
                extreme_selected = {
                    'fov_key': fov_key,
                    'mrna_per_nucleus': row['mrna_per_nucleus'],
                    'n_clusters': row['n_clusters'],
                    'n_nuclei': row['n_nuclei'],
                    'slide': slide,
                    'region': row['region'],
                    'mouse_id': row.get('mouse_id'),
                }
                break

        # Only add if we found both
        if normal_selected and extreme_selected:
            normal_fovs.append(normal_selected)
            extreme_fovs.append(extreme_selected)
            used_fov_keys.add(normal_selected['fov_key'])
            used_fov_keys.add(extreme_selected['fov_key'])
            print(f"    {slide}: normal={normal_selected['mrna_per_nucleus']:.2f}, "
                  f"extreme={extreme_selected['mrna_per_nucleus']:.2f} mRNA/nuc")

    # Second pass: get additional pairs from slides (different FOVs, can be from different regions)
    # This allows multiple pairs per slide from different subregions
    for slide in valid_slides:
        if len(normal_fovs) >= n_pairs:
            break

        # Get all FOVs from this slide that haven't been used
        slide_df = df_q111[df_q111['slide'] == slide].copy()
        slide_df = slide_df[~slide_df['fov_key'].isin(used_fov_keys)]

        if len(slide_df) < 4:  # Need at least a few FOVs to form another pair
            continue

        slide_df = slide_df.sort_values('mrna_per_nucleus')

        # Calculate quartiles within remaining FOVs
        q25 = slide_df['mrna_per_nucleus'].quantile(0.25)
        q75 = slide_df['mrna_per_nucleus'].quantile(0.75)

        # Select normal FOV (bottom quartile)
        normal_candidates = slide_df[slide_df['mrna_per_nucleus'] <= q25]
        # Select extreme FOV (top quartile)
        extreme_candidates = slide_df[slide_df['mrna_per_nucleus'] >= q75]

        if len(normal_candidates) == 0 or len(extreme_candidates) == 0:
            continue

        # Find valid normal FOV (with NPZ file)
        normal_selected = None
        for _, row in normal_candidates.iterrows():
            fov_key = row['fov_key']
            if fov_key in used_fov_keys:
                continue
            npz_path = get_npz_path_from_h5(fov_key, h5_path)
            if npz_path is not None:
                normal_selected = {
                    'fov_key': fov_key,
                    'mrna_per_nucleus': row['mrna_per_nucleus'],
                    'n_clusters': row['n_clusters'],
                    'n_nuclei': row['n_nuclei'],
                    'slide': slide,
                    'region': row['region'],
                    'mouse_id': row.get('mouse_id'),
                }
                break

        # Find valid extreme FOV (with NPZ file)
        extreme_candidates = extreme_candidates.sort_values('mrna_per_nucleus', ascending=False)
        extreme_selected = None
        for _, row in extreme_candidates.iterrows():
            fov_key = row['fov_key']
            if fov_key in used_fov_keys:
                continue
            npz_path = get_npz_path_from_h5(fov_key, h5_path)
            if npz_path is not None:
                extreme_selected = {
                    'fov_key': fov_key,
                    'mrna_per_nucleus': row['mrna_per_nucleus'],
                    'n_clusters': row['n_clusters'],
                    'n_nuclei': row['n_nuclei'],
                    'slide': slide,
                    'region': row['region'],
                    'mouse_id': row.get('mouse_id'),
                }
                break

        # Only add if we found both
        if normal_selected and extreme_selected:
            normal_fovs.append(normal_selected)
            extreme_fovs.append(extreme_selected)
            used_fov_keys.add(normal_selected['fov_key'])
            used_fov_keys.add(extreme_selected['fov_key'])
            print(f"    {slide}: normal={normal_selected['mrna_per_nucleus']:.2f}, "
                  f"extreme={extreme_selected['mrna_per_nucleus']:.2f} mRNA/nuc (additional pair)")

    print(f"\nSelected {len(normal_fovs)} pairs from {len(set(nf['slide'] for nf in normal_fovs))} slides")
    return normal_fovs, extreme_fovs


def generate_fov_images(h5_path: str, fov_key: str,
                        category: str, slide: str,
                        mrna_per_nucleus_green: float, mrna_per_nucleus_orange: float,
                        n_clusters_green: int, n_clusters_orange: int,
                        n_nuclei: int, metadata: dict,
                        output_dir: Path, n_zooms: int = 5):
    """
    Generate all visualizations for a single FOV.

    Creates:
    1. Green channel only (no DAPI)
    2. Orange channel only (no DAPI)
    3. Composite image (DAPI + green + orange merged)
    4. Zoom images with scale bars

    Output organized by slide/animal with category subfolder.
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

    # Build condition text overlay (include slide/mouse info)
    lines = []
    if 'mouse_model' in metadata:
        lines.append(metadata['mouse_model'])
    if 'age' in metadata:
        lines.append(f"{metadata['age']}mo")
    if 'region' in metadata:
        # Shorten region name
        region = metadata['region']
        if 'Cortex' in region:
            lines.append('Cortex')
        elif 'Striatum' in region:
            lines.append('Striatum')
        else:
            lines.append(region)
    if 'slide' in metadata:
        lines.append(metadata['slide'])

    condition_text = '\n'.join(lines)

    # Create output directories organized by slide/animal
    # Structure: output_dir / slide / category / files
    slide_dir = output_dir / slide
    category_dir = slide_dir / category
    category_dir.mkdir(parents=True, exist_ok=True)

    # Create safe filename from fov_key
    safe_name = fov_key.replace('/', '_').replace('--', '_')[:60]

    # Create colored images for each channel
    green_colored = create_colored_image(normalize_to_8bit(green_mip), 'green')
    orange_colored = create_colored_image(normalize_to_8bit(orange_mip), 'orange')
    dapi_colored = create_colored_image(normalize_to_8bit(dapi_mip), 'blue')

    # Create composite
    composite_all = create_composite_three_channel(dapi_mip, green_mip, orange_mip)

    # Add scale bars (50 µm for overview)
    green_colored = draw_scale_bar(green_colored, scale_um=SCALE_BAR_OVERVIEW)
    orange_colored = draw_scale_bar(orange_colored, scale_um=SCALE_BAR_OVERVIEW)
    composite_all = draw_scale_bar(composite_all, scale_um=SCALE_BAR_OVERVIEW)

    # Add text overlays with mRNA/nucleus info
    green_text = f"{condition_text}\nmHTT1a: {mrna_per_nucleus_green:.2f}"
    orange_text = f"{condition_text}\nfl-mHTT: {mrna_per_nucleus_orange:.2f}"
    composite_text = f"{condition_text}\nmHTT1a: {mrna_per_nucleus_green:.2f}\nfl-mHTT: {mrna_per_nucleus_orange:.2f}"

    green_colored = add_text_overlay(green_colored, green_text, position='top-left')
    orange_colored = add_text_overlay(orange_colored, orange_text, position='top-left')
    composite_all = add_text_overlay(composite_all, composite_text, position='top-left')

    # Save overview images
    green_path = category_dir / f"{safe_name}_green.tif"
    orange_path = category_dir / f"{safe_name}_orange.tif"
    composite_path = category_dir / f"{safe_name}_composite.tif"

    imwrite(green_path, green_colored)
    imwrite(orange_path, orange_colored)
    imwrite(composite_path, composite_all)

    print(f"  Saved: {green_path.name}")

    # Generate zoom images
    zoom_dir = category_dir / f"{safe_name}_zooms"
    zoom_dir.mkdir(parents=True, exist_ok=True)

    # Get cluster centers of mass for zoom selection
    green_coms = green_data.get('cluster_coms', np.array([]))
    orange_coms = orange_data.get('cluster_coms', np.array([]))

    # Combine COMs for zoom selection (prefer areas with clusters)
    if len(green_coms) > 0 and len(orange_coms) > 0:
        all_coms = np.vstack([green_coms, orange_coms])
    elif len(green_coms) > 0:
        all_coms = green_coms
    elif len(orange_coms) > 0:
        all_coms = orange_coms
    else:
        all_coms = np.array([])

    # Select zoom regions
    zoom_centers = select_zoom_regions(
        green_mip.shape, all_coms, n_zooms=n_zooms,
        zoom_size=ZOOM_SIZE, min_distance=300
    )

    half = ZOOM_SIZE // 2
    h, w = green_mip.shape

    for i, (cy, cx) in enumerate(zoom_centers):
        # Calculate zoom bounds
        y1 = max(0, cy - half)
        y2 = min(h, cy + half)
        x1 = max(0, cx - half)
        x2 = min(w, cx + half)

        # Extract raw zoom regions from MIPs
        green_zoom_raw = green_mip[y1:y2, x1:x2]
        orange_zoom_raw = orange_mip[y1:y2, x1:x2]
        dapi_zoom_raw = dapi_mip[y1:y2, x1:x2]

        # Normalize each zoom locally for better contrast
        green_zoom_8bit = normalize_to_8bit(green_zoom_raw)
        orange_zoom_8bit = normalize_to_8bit(orange_zoom_raw)

        # Create colored images from locally normalized zooms
        green_zoom = create_colored_image(green_zoom_8bit, 'green')
        orange_zoom = create_colored_image(orange_zoom_8bit, 'orange')

        # Composite zoom - also normalize locally
        composite_all_zoom = create_composite_three_channel(
            dapi_zoom_raw, green_zoom_raw, orange_zoom_raw,
            dapi_pmin=10, dapi_pmax=99,
            signal_pmin=50, signal_pmax=99.9
        )

        # Pad if necessary to get exact size
        if green_zoom.shape[0] < ZOOM_SIZE or green_zoom.shape[1] < ZOOM_SIZE:
            padded = np.zeros((ZOOM_SIZE, ZOOM_SIZE, 3), dtype=green_zoom.dtype)
            padded[:green_zoom.shape[0], :green_zoom.shape[1]] = green_zoom
            green_zoom = padded

            padded = np.zeros((ZOOM_SIZE, ZOOM_SIZE, 3), dtype=orange_zoom.dtype)
            padded[:orange_zoom.shape[0], :orange_zoom.shape[1]] = orange_zoom
            orange_zoom = padded

            padded = np.zeros((ZOOM_SIZE, ZOOM_SIZE, 3), dtype=composite_all_zoom.dtype)
            padded[:composite_all_zoom.shape[0], :composite_all_zoom.shape[1]] = composite_all_zoom
            composite_all_zoom = padded

        # Add scale bars (10 µm for zooms)
        green_zoom = draw_scale_bar(green_zoom, scale_um=SCALE_BAR_ZOOM)
        orange_zoom = draw_scale_bar(orange_zoom, scale_um=SCALE_BAR_ZOOM)
        composite_all_zoom = draw_scale_bar(composite_all_zoom, scale_um=SCALE_BAR_ZOOM)

        # Save zooms
        imwrite(zoom_dir / f"zoom{i+1}_green.tif", green_zoom)
        imwrite(zoom_dir / f"zoom{i+1}_orange.tif", orange_zoom)
        imwrite(zoom_dir / f"zoom{i+1}_all.tif", composite_all_zoom)

    print(f"  Saved: {n_zooms} zooms to {zoom_dir.name}/")

    return green_path


def save_fov_metadata(h5_path: str, fov_key: str, category: str, slide: str,
                      mrna_per_nucleus_green: float, mrna_per_nucleus_orange: float,
                      n_clusters_green: int, n_clusters_orange: int,
                      n_nuclei: int, peak_intensity_green: float, peak_intensity_orange: float,
                      threshold_green: float, threshold_orange: float,
                      output_dir: Path):
    """
    Save detailed metadata for a FOV to a text file.

    Includes all relevant sample metadata and analysis parameters.
    Output organized by slide/animal with category subfolder.
    """
    # Create safe filename from fov_key
    safe_name = fov_key.replace('/', '_').replace('--', '_')[:60]

    # Output directory organized by slide/animal
    # Structure: output_dir / slide / category / files
    slide_dir = output_dir / slide
    category_dir = slide_dir / category
    category_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = category_dir / f"{safe_name}_metadata.txt"

    # Load full metadata from H5
    with h5py.File(h5_path, 'r') as f:
        if fov_key not in f:
            print(f"  Warning: FOV {fov_key} not found in H5")
            return

        fov = f[fov_key]

        lines = []
        lines.append("=" * 70)
        lines.append(f"FOV METADATA: {fov_key}")
        lines.append("=" * 70)
        lines.append("")

        # Sample metadata
        lines.append("SAMPLE INFORMATION")
        lines.append("-" * 40)

        if 'metadata_sample' in fov:
            sample_meta = fov['metadata_sample']

            # Key fields to extract
            key_fields = [
                ('Mouse Model', 'Mouse Model'),
                ('mouse ID ', 'Mouse ID'),
                ('Age', 'Age (months)'),
                ('Sex', 'Sex'),
                ('Condition', 'Condition'),
                ('Level', 'Disease Level'),
                ('Slice Region', 'Brain Region'),
                ('Region', 'Slide Region'),
                ('Slide name', 'Slide Name'),
                ('Brain half', 'Brain Half'),
                ('Brain_Atlas_coordinates', 'Atlas Coordinates'),
                ('Probe-Set', 'Probe Set'),
                ('Date', 'Imaging Date'),
                ('Person', 'Imaged By'),
            ]

            for h5_key, display_name in key_fields:
                if h5_key in sample_meta:
                    val = sample_meta[h5_key][()]
                    if isinstance(val, np.ndarray) and len(val) > 0:
                        val = val[0]
                    if isinstance(val, bytes):
                        val = val.decode()
                    lines.append(f"{display_name}: {val}")

        lines.append("")
        lines.append("IMAGING PARAMETERS")
        lines.append("-" * 40)

        if 'metadata_sample' in fov:
            sample_meta = fov['metadata_sample']
            imaging_fields = [
                ('Exposure time FITC', 'Exposure FITC (ms)'),
                ('Exposure time CY3', 'Exposure CY3 (ms)'),
                ('Exposure time CY5', 'Exposure CY5 (ms)'),
                ('Exposure time dapi', 'Exposure DAPI (ms)'),
                ('intensity FITC', 'Intensity FITC'),
                ('intensity CY3', 'Intensity CY3'),
                ('intensity CY5', 'Intensity CY5'),
                ('intensity dapi', 'Intensity DAPI'),
            ]

            for h5_key, display_name in imaging_fields:
                if h5_key in sample_meta:
                    val = sample_meta[h5_key][()]
                    if isinstance(val, np.ndarray) and len(val) > 0:
                        val = val[0]
                    lines.append(f"{display_name}: {val}")

        lines.append("")
        lines.append("ANALYSIS RESULTS")
        lines.append("-" * 40)
        lines.append(f"Number of nuclei: {n_nuclei}")
        lines.append("")
        lines.append("mHTT1a (green channel):")
        lines.append(f"  Clusters above threshold: {n_clusters_green}")
        lines.append(f"  mRNA per nucleus: {mrna_per_nucleus_green:.4f}")
        lines.append(f"  Single mRNA peak intensity: {peak_intensity_green:.2f} photons")
        lines.append(f"  Threshold (95th %ile neg ctrl): {threshold_green:.2f} photons")
        lines.append("")
        lines.append("Full-length mHTT (orange channel):")
        lines.append(f"  Clusters above threshold: {n_clusters_orange}")
        lines.append(f"  mRNA per nucleus: {mrna_per_nucleus_orange:.4f}")
        lines.append(f"  Single mRNA peak intensity: {peak_intensity_orange:.2f} photons")
        lines.append(f"  Threshold (95th %ile neg ctrl): {threshold_orange:.2f} photons")

        lines.append("")
        lines.append("=" * 70)

    # Write to file
    with open(metadata_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"  Saved metadata: {metadata_path.name}")


def main():
    """Generate Figure 5 panel images - Extreme vs Normal Q111 FOVs from same animals."""

    print("=" * 70)
    print("FIGURE 5 PANEL GENERATION")
    print("Extreme vs Normal Q111 FOVs (Same Animals)")
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

    # Print summary statistics for Q111
    print("\nQ111 Expression summary:")
    q111_df = df[(df['mouse_model'] == 'Q111') & (df['channel'] == 'green')]
    if len(q111_df) > 0:
        print(f"  Total Q111 FOVs: {len(q111_df)}")
        print(f"  mRNA/nuc range: {q111_df['mrna_per_nucleus'].min():.2f} - {q111_df['mrna_per_nucleus'].max():.2f}")
        print(f"  Unique slides: {q111_df['slide'].nunique()}")

    # Select paired FOVs from same animals
    print("\n" + "-" * 40)
    print("Selecting paired Normal vs Extreme Q111 FOVs...")
    normal_fovs, extreme_fovs = select_paired_fovs(df, h5_path, channel='green', n_pairs=8)

    if len(normal_fovs) == 0:
        print("ERROR: Could not find valid pairs")
        return

    # Generate images
    print("\n" + "=" * 70)
    print("GENERATING FOV IMAGES")
    print("=" * 70)

    # Process each pair
    for i, (normal_info, extreme_info) in enumerate(zip(normal_fovs, extreme_fovs)):
        slide = normal_info['slide']
        print(f"\n--- PAIR {i+1}: {slide} ---")

        # Process normal FOV
        print(f"\n  NORMAL: {normal_info['fov_key'][:50]}...")
        fov_key = normal_info['fov_key']

        # Get metadata for this FOV from the full dataframe
        fov_df = df[df['fov_key'] == fov_key].iloc[0]
        metadata = {
            'mouse_model': fov_df['mouse_model'],
            'age': fov_df['age'],
            'region': fov_df['region'],
            'slide': slide,
        }

        # Get cluster counts and mRNA/nucleus for each channel
        green_df = df[(df['fov_key'] == fov_key) & (df['channel'] == 'green')]
        orange_df = df[(df['fov_key'] == fov_key) & (df['channel'] == 'orange')]
        n_clusters_green = green_df['n_clusters'].values[0] if len(green_df) > 0 else 0
        n_clusters_orange = orange_df['n_clusters'].values[0] if len(orange_df) > 0 else 0
        mrna_green = green_df['mrna_per_nucleus'].values[0] if len(green_df) > 0 else 0
        mrna_orange = orange_df['mrna_per_nucleus'].values[0] if len(orange_df) > 0 else 0
        peak_intensity_green = green_df['peak_intensity'].values[0] if len(green_df) > 0 else 0
        peak_intensity_orange = orange_df['peak_intensity'].values[0] if len(orange_df) > 0 else 0
        threshold_green = green_df['threshold'].values[0] if len(green_df) > 0 else 0
        threshold_orange = orange_df['threshold'].values[0] if len(orange_df) > 0 else 0
        n_nuclei = normal_info['n_nuclei']

        print(f"    mHTT1a: {mrna_green:.2f}, fl-mHTT: {mrna_orange:.2f} mRNA/nuc, {n_nuclei} nuclei")

        # Generate images
        generate_fov_images(
            h5_path, fov_key, 'normal', slide,
            mrna_green, mrna_orange,
            n_clusters_green, n_clusters_orange, n_nuclei,
            metadata, OUTPUT_DIR, n_zooms=5
        )

        # Save metadata
        save_fov_metadata(
            h5_path, fov_key, 'normal', slide,
            mrna_green, mrna_orange,
            n_clusters_green, n_clusters_orange, n_nuclei,
            peak_intensity_green, peak_intensity_orange,
            threshold_green, threshold_orange,
            OUTPUT_DIR
        )

        # Process extreme FOV
        print(f"\n  EXTREME: {extreme_info['fov_key'][:50]}...")
        fov_key = extreme_info['fov_key']

        # Get metadata for this FOV from the full dataframe
        fov_df = df[df['fov_key'] == fov_key].iloc[0]
        metadata = {
            'mouse_model': fov_df['mouse_model'],
            'age': fov_df['age'],
            'region': fov_df['region'],
            'slide': slide,
        }

        # Get cluster counts and mRNA/nucleus for each channel
        green_df = df[(df['fov_key'] == fov_key) & (df['channel'] == 'green')]
        orange_df = df[(df['fov_key'] == fov_key) & (df['channel'] == 'orange')]
        n_clusters_green = green_df['n_clusters'].values[0] if len(green_df) > 0 else 0
        n_clusters_orange = orange_df['n_clusters'].values[0] if len(orange_df) > 0 else 0
        mrna_green = green_df['mrna_per_nucleus'].values[0] if len(green_df) > 0 else 0
        mrna_orange = orange_df['mrna_per_nucleus'].values[0] if len(orange_df) > 0 else 0
        peak_intensity_green = green_df['peak_intensity'].values[0] if len(green_df) > 0 else 0
        peak_intensity_orange = orange_df['peak_intensity'].values[0] if len(orange_df) > 0 else 0
        threshold_green = green_df['threshold'].values[0] if len(green_df) > 0 else 0
        threshold_orange = orange_df['threshold'].values[0] if len(orange_df) > 0 else 0
        n_nuclei = extreme_info['n_nuclei']

        print(f"    mHTT1a: {mrna_green:.2f}, fl-mHTT: {mrna_orange:.2f} mRNA/nuc, {n_nuclei} nuclei")

        # Generate images
        generate_fov_images(
            h5_path, fov_key, 'extreme', slide,
            mrna_green, mrna_orange,
            n_clusters_green, n_clusters_orange, n_nuclei,
            metadata, OUTPUT_DIR, n_zooms=5
        )

        # Save metadata
        save_fov_metadata(
            h5_path, fov_key, 'extreme', slide,
            mrna_green, mrna_orange,
            n_clusters_green, n_clusters_orange, n_nuclei,
            peak_intensity_green, peak_intensity_orange,
            threshold_green, threshold_orange,
            OUTPUT_DIR
        )

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY - PAIRED COMPARISONS")
    print("=" * 70)

    print("\n{:<12} {:<40} {:>12} {:>12}".format("Slide", "Category", "mRNA/nuc", "Nuclei"))
    print("-" * 80)

    for i, (normal_info, extreme_info) in enumerate(zip(normal_fovs, extreme_fovs)):
        slide = normal_info['slide']
        print(f"{slide:<12} {'Normal':<40} {normal_info['mrna_per_nucleus']:>12.2f} {normal_info['n_nuclei']:>12}")
        print(f"{'':<12} {'Extreme':<40} {extreme_info['mrna_per_nucleus']:>12.2f} {extreme_info['n_nuclei']:>12}")
        fold_change = extreme_info['mrna_per_nucleus'] / normal_info['mrna_per_nucleus'] if normal_info['mrna_per_nucleus'] > 0 else float('inf')
        print(f"{'':<12} {'Fold change:':<40} {fold_change:>12.1f}x")
        print()

    print(f"\nOutput saved to: {OUTPUT_DIR}")
    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
