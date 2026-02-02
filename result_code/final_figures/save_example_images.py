#!/usr/bin/env python3
"""
Save example images for each figure.

This script uses the image browser to find appropriate FOVs and saves
max-projection images for use in figures.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Add parent directories for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # For view_npz_stack

from image_browser import (
    load_fov_metadata, load_fov_level_data, merge_h5_with_fov_data,
    filter_fovs, find_npz_path, find_h5_file
)
from view_npz_stack import load_npz_stack

# Output directory (centralized under output/)
OUTPUT_DIR = Path(__file__).parent / 'output' / 'example_images'

# Custom colormaps for channels
def create_channel_colormaps():
    """Create colormaps for each channel."""
    # Blue for DAPI
    blue_cmap = LinearSegmentedColormap.from_list('blue', ['black', 'blue'])
    # Green for mHTT1a
    green_cmap = LinearSegmentedColormap.from_list('green', ['black', 'lime'])
    # Orange/Red for full-length mHTT
    orange_cmap = LinearSegmentedColormap.from_list('orange', ['black', 'orange'])

    return {'blue': blue_cmap, 'green': green_cmap, 'orange': orange_cmap}


def load_and_project_npz(npz_path: str) -> dict:
    """Load NPZ file and create max projections for each channel."""
    # Use the loader from view_npz_stack.py
    arr, metadata = load_npz_stack(npz_path)

    # Channel mapping: ['mDAPI', 'sFITC', 'sCY3', 'sCY5']
    # mDAPI -> blue, sFITC -> green, sCY3 -> orange, sCY5 -> red
    channels = metadata.get('channels', ['mDAPI', 'sFITC', 'sCY3', 'sCY5'])
    channel_map = {
        'mDAPI': 'blue',
        'sFITC': 'green',
        'sCY3': 'orange',
        'sCY5': 'red'
    }

    projections = {}
    for i, ch_name in enumerate(channels):
        color_key = channel_map.get(ch_name, ch_name.lower())
        stack = arr[i]  # Shape: (z, y, x)
        projections[color_key] = np.max(stack, axis=0)  # Max projection

    return projections


def save_fov_image(npz_path: str, output_dir: str, base_name: str,
                   channels: list = None):
    """
    Save each channel as a separate TIFF file, plus a merged RGB TIFF.

    Args:
        npz_path: Path to NPZ file
        output_dir: Output directory path
        base_name: Base name for output files (e.g., 'experimental_tissue_1')
        channels: List of channels to save ['blue', 'green', 'orange']

    Saves:
        - {base_name}_DAPI.tif (blue channel, grayscale)
        - {base_name}_mHTT1a.tif (green channel, grayscale)
        - {base_name}_full_length.tif (orange channel, grayscale)
        - {base_name}_merged.tif (RGB merge: blue=DAPI, green=mHTT1a, red=full-length)
    """
    from tifffile import imwrite

    if channels is None:
        channels = ['blue', 'green', 'orange']

    projections = load_and_project_npz(npz_path)
    output_dir = Path(output_dir)

    # Channel name mapping for filenames
    channel_names = {
        'blue': 'DAPI',
        'green': 'mHTT1a',
        'orange': 'full_length'
    }

    # Store normalized images for merging
    normalized = {}

    # Save individual channels as grayscale TIFFs
    for ch in channels:
        if ch in projections:
            img = projections[ch]

            # Normalize to 0-1 range using percentiles
            vmin, vmax = np.percentile(img, [1, 99.5])
            img_norm = np.clip((img - vmin) / (vmax - vmin + 1e-10), 0, 1)
            normalized[ch] = img_norm

            # Convert to 16-bit for TIFF (preserves dynamic range)
            img_16bit = (img_norm * 65535).astype(np.uint16)

            # Save grayscale TIFF
            ch_name = channel_names.get(ch, ch)
            tiff_path = output_dir / f"{base_name}_{ch_name}.tif"
            imwrite(str(tiff_path), img_16bit)
            print(f"  Saved: {tiff_path.name}")

    # Create and save merged RGB TIFF
    # Color coding: blue=DAPI, green=mHTT1a, red=full-length (orange)
    shape = None
    for ch in channels:
        if ch in normalized:
            shape = normalized[ch].shape
            break

    if shape is not None:
        # Create RGB array (8-bit for compatibility)
        rgb = np.zeros((*shape, 3), dtype=np.uint8)

        for ch in channels:
            if ch in normalized:
                img_8bit = (normalized[ch] * 255).astype(np.uint8)

                if ch == 'blue':
                    rgb[:, :, 2] = img_8bit  # Blue channel (R=0, G=0, B=value)
                elif ch == 'green':
                    rgb[:, :, 1] = img_8bit  # Green channel (R=0, G=value, B=0)
                elif ch == 'orange':
                    rgb[:, :, 0] = img_8bit  # Red channel (R=value, G=0, B=0)

        # Save merged RGB TIFF
        merged_path = output_dir / f"{base_name}_merged.tif"
        imwrite(str(merged_path), rgb)
        print(f"  Saved: {merged_path.name}")


def main():
    """Generate example images for each figure."""

    print("="*80)
    print("LOADING DATA")
    print("="*80)

    # Load data
    h5_path = find_h5_file()
    df = load_fov_metadata(h5_path)
    df_fov, thresholds = load_fov_level_data()
    if df_fov is not None:
        df = merge_h5_with_fov_data(df, df_fov)

    print(f"\nTotal FOVs: {len(df)}")

    # Create output directories
    for i in range(1, 6):
        (OUTPUT_DIR / f'figure{i}').mkdir(parents=True, exist_ok=True)

    # ══════════════════════════════════════════════════════════════════════════
    # FIGURE 1: Methodology - need representative experimental tissue
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*80)
    print("FIGURE 1: Representative experimental tissue")
    print("="*80)

    df_fig1 = filter_fovs(df, genotype='Q111', probe_type='experimental',
                          region_type='striatum', min_nuclei=200)
    df_fig1 = df_fig1.sort_values('total_spots', ascending=False).head(3)

    for idx, (_, row) in enumerate(df_fig1.iterrows()):
        original_fov = row.get('original_fov_num', None)
        npz_path = find_npz_path(row['slide_raw'], row['region_num'], row['fov_num'], original_fov)
        if npz_path:
            output_dir = OUTPUT_DIR / 'figure1'
            base_name = f'experimental_tissue_{idx+1}'
            print(f"\n  Processing: {row['slide_std']} - {row.get('brain_region', 'Striatum')}")
            save_fov_image(npz_path, output_dir, base_name)

    # ══════════════════════════════════════════════════════════════════════════
    # FIGURE 2: Cluster characterization - need high cluster examples
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*80)
    print("FIGURE 2: High cluster examples")
    print("="*80)

    df_fig2 = filter_fovs(df, genotype='Q111', probe_type='experimental', min_nuclei=100)
    if 'green_clusters_per_cell' in df_fig2.columns:
        df_fig2 = df_fig2.sort_values('green_clusters_per_cell', ascending=False, na_position='last').head(3)
    else:
        df_fig2 = df_fig2.sort_values('total_spots', ascending=False).head(3)

    for idx, (_, row) in enumerate(df_fig2.iterrows()):
        original_fov = row.get('original_fov_num', None)
        npz_path = find_npz_path(row['slide_raw'], row['region_num'], row['fov_num'], original_fov)
        if npz_path:
            output_dir = OUTPUT_DIR / 'figure2'
            base_name = f'high_clusters_{idx+1}'
            print(f"\n  Processing: {row['slide_std']} - {row.get('brain_region', '')}")
            save_fov_image(npz_path, output_dir, base_name)

    # ══════════════════════════════════════════════════════════════════════════
    # FIGURE 3: Regional analysis - Cortex vs Striatum
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*80)
    print("FIGURE 3: Cortex vs Striatum examples")
    print("="*80)

    # Cortex example
    df_cortex = filter_fovs(df, genotype='Q111', probe_type='experimental',
                            region_type='cortex', min_nuclei=200)
    df_cortex = df_cortex.sort_values('total_spots', ascending=False).head(2)

    for idx, (_, row) in enumerate(df_cortex.iterrows()):
        original_fov = row.get('original_fov_num', None)
        npz_path = find_npz_path(row['slide_raw'], row['region_num'], row['fov_num'], original_fov)
        if npz_path:
            output_dir = OUTPUT_DIR / 'figure3'
            base_name = f'cortex_{idx+1}'
            print(f"\n  Processing: Cortex - {row['slide_std']}")
            save_fov_image(npz_path, output_dir, base_name)

    # Striatum example
    df_striatum = filter_fovs(df, genotype='Q111', probe_type='experimental',
                              region_type='striatum', min_nuclei=200)
    df_striatum = df_striatum.sort_values('total_spots', ascending=False).head(2)

    for idx, (_, row) in enumerate(df_striatum.iterrows()):
        original_fov = row.get('original_fov_num', None)
        npz_path = find_npz_path(row['slide_raw'], row['region_num'], row['fov_num'], original_fov)
        if npz_path:
            output_dir = OUTPUT_DIR / 'figure3'
            base_name = f'striatum_{idx+1}'
            print(f"\n  Processing: Striatum - {row['slide_std']}")
            save_fov_image(npz_path, output_dir, base_name)

    # ══════════════════════════════════════════════════════════════════════════
    # FIGURE 4: FOV heterogeneity - Q111 vs WT
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*80)
    print("FIGURE 4: Q111 vs WT examples")
    print("="*80)

    # Q111 high expression
    df_q111 = filter_fovs(df, genotype='Q111', probe_type='experimental', min_nuclei=200)
    df_q111 = df_q111.sort_values('total_spots', ascending=False).head(2)

    for idx, (_, row) in enumerate(df_q111.iterrows()):
        original_fov = row.get('original_fov_num', None)
        npz_path = find_npz_path(row['slide_raw'], row['region_num'], row['fov_num'], original_fov)
        if npz_path:
            output_dir = OUTPUT_DIR / 'figure4'
            base_name = f'q111_high_{idx+1}'
            print(f"\n  Processing: Q111 - {row['slide_std']} - {row.get('brain_region', '')}")
            save_fov_image(npz_path, output_dir, base_name)

    # WT (should be low)
    df_wt = filter_fovs(df, genotype='Wildtype', probe_type='experimental', min_nuclei=200)
    df_wt = df_wt.sort_values('total_spots', ascending=True).head(2)

    for idx, (_, row) in enumerate(df_wt.iterrows()):
        original_fov = row.get('original_fov_num', None)
        npz_path = find_npz_path(row['slide_raw'], row['region_num'], row['fov_num'], original_fov)
        if npz_path:
            output_dir = OUTPUT_DIR / 'figure4'
            base_name = f'wt_low_{idx+1}'
            print(f"\n  Processing: WT - {row['slide_std']} - {row.get('brain_region', '')}")
            save_fov_image(npz_path, output_dir, base_name)

    # ══════════════════════════════════════════════════════════════════════════
    # FIGURE 5: Extreme vs Normal FOVs
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*80)
    print("FIGURE 5: Extreme vs Normal examples")
    print("="*80)

    # Extreme FOVs (mHTT1a)
    df_extreme = filter_fovs(df, genotype='Q111', probe_type='experimental',
                             fov_class='extreme', channel='mHTT1a', min_nuclei=100)
    if 'green_clustered_mrna_per_cell' in df_extreme.columns:
        df_extreme = df_extreme.sort_values('green_clustered_mrna_per_cell', ascending=False, na_position='last').head(2)

    for idx, (_, row) in enumerate(df_extreme.iterrows()):
        original_fov = row.get('original_fov_num', None)
        npz_path = find_npz_path(row['slide_raw'], row['region_num'], row['fov_num'], original_fov)
        if npz_path:
            output_dir = OUTPUT_DIR / 'figure5'
            mrna = row.get('green_clustered_mrna_per_cell', 0)
            base_name = f'extreme_mhtt1a_{idx+1}'
            print(f"\n  Processing: EXTREME (mHTT1a: {mrna:.1f} mRNA/cell) - {row['slide_std']}")
            save_fov_image(npz_path, output_dir, base_name)

    # Normal FOVs (mHTT1a)
    df_normal = filter_fovs(df, genotype='Q111', probe_type='experimental',
                            fov_class='normal', channel='mHTT1a', min_nuclei=100)
    if 'green_clustered_mrna_per_cell' in df_normal.columns:
        df_normal = df_normal.sort_values('green_clustered_mrna_per_cell', ascending=False, na_position='last').head(2)

    for idx, (_, row) in enumerate(df_normal.iterrows()):
        original_fov = row.get('original_fov_num', None)
        npz_path = find_npz_path(row['slide_raw'], row['region_num'], row['fov_num'], original_fov)
        if npz_path:
            output_dir = OUTPUT_DIR / 'figure5'
            mrna = row.get('green_clustered_mrna_per_cell', 0)
            base_name = f'normal_mhtt1a_{idx+1}'
            print(f"\n  Processing: NORMAL (mHTT1a: {mrna:.1f} mRNA/cell) - {row['slide_std']}")
            save_fov_image(npz_path, output_dir, base_name)

    print("\n" + "="*80)
    print("DONE! Example images saved to:")
    print(f"  {OUTPUT_DIR}")
    print("="*80)


if __name__ == '__main__':
    main()
