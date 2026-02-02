#!/usr/bin/env python3
"""
Add Variance Features to Cluster Data

This script adds intensity variance metrics to existing cluster_data.csv:
- mean_photons: Mean photon count per voxel within cluster
- std_photons: Standard deviation of photon counts within cluster
- cv_photons: Coefficient of variation (std/mean) - normalized variance
- iqr_photons: Interquartile range (75th - 25th percentile)
- pct_range: 95th - 5th percentile range

These features help distinguish real clusters (punctate, high variance)
from diffuse background artifacts (uniform, low variance).

Usage:
    python add_variance_features.py --data_dir ./output
    python add_variance_features.py --data_dir ./output --n_fovs 50  # Limit FOVs for testing
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
import blosc
import argparse
from collections import defaultdict

# Add parent directories for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'final_figures'))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'user_code'))

# Import from results_config
sys.path.insert(0, str(Path(__file__).parent.parent))
from results_config import (
    CHANNEL_PARAMS, SLICE_DEPTH, PIXELSIZE, VOXEL_SIZE, SIGMA_Z_XLIM,
    H5_FILE_PATH_EXPERIMENTAL, EXCLUDED_SLIDES
)

# Channel map
CHANNEL_MAP = {
    'blue': 0,
    'green': 1,
    'orange': 2,
    'red': 3,
}


def convert_to_photons(image: np.ndarray, channel_params: dict) -> np.ndarray:
    """Convert raw image to photon counts using channel parameters."""
    gain = channel_params['gain']
    offset = channel_params['offset']
    # photons = (raw - offset) / gain
    return (image.astype(np.float32) - offset) / gain


def compute_cluster_variance_stats(label_mask: np.ndarray, image: np.ndarray) -> dict:
    """
    Compute variance statistics for each cluster.

    Args:
        label_mask: 3D label array where each cluster has unique integer ID
        image: 3D intensity image (photon counts)

    Returns:
        dict mapping label_id -> {mean, std, cv, iqr, pct_range}
    """
    unique_labels = np.unique(label_mask)
    unique_labels = unique_labels[unique_labels > 0]

    stats = {}
    for label_id in unique_labels:
        mask = label_mask == label_id
        voxel_values = image[mask]

        if len(voxel_values) == 0:
            continue

        mean_val = np.mean(voxel_values)
        std_val = np.std(voxel_values)
        cv = std_val / mean_val if mean_val > 0 else 0

        # Percentile-based metrics
        p5, p25, p75, p95 = np.percentile(voxel_values, [5, 25, 75, 95])
        iqr = p75 - p25
        pct_range = p95 - p5

        # Max intensity within cluster
        max_val = np.max(voxel_values)

        stats[int(label_id)] = {
            'mean_photons': float(mean_val),
            'std_photons': float(std_val),
            'cv_photons': float(cv),
            'iqr_photons': float(iqr),
            'pct_range_photons': float(pct_range),
            'max_photons': float(max_val),
        }

    return stats


def main():
    parser = argparse.ArgumentParser(description='Add variance features to cluster data')
    parser.add_argument('--data_dir', type=str, default='./output',
                        help='Directory containing cluster_data.csv')
    parser.add_argument('--n_fovs', type=int, default=None,
                        help='Limit number of FOVs to process (for testing)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV path (default: cluster_data_with_variance.csv)')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    cluster_csv = data_dir / 'cluster_data.csv'

    if not cluster_csv.exists():
        print(f"ERROR: cluster_data.csv not found at {cluster_csv}")
        sys.exit(1)

    # Load existing cluster data
    print(f"Loading cluster data from {cluster_csv}...")
    df = pd.read_csv(cluster_csv)
    print(f"Loaded {len(df)} clusters from {df['fov_key'].nunique()} FOVs")

    # Check if variance features already exist
    if 'mean_photons' in df.columns:
        print("Variance features already exist. Overwriting...")

    # Group by FOV
    fov_groups = df.groupby('fov_key')
    fov_keys = list(fov_groups.groups.keys())

    if args.n_fovs:
        fov_keys = fov_keys[:args.n_fovs]
        print(f"Limiting to {len(fov_keys)} FOVs")

    print("Starting processing...")

    # Initialize new columns with NaN
    for col in ['mean_photons', 'std_photons', 'cv_photons', 'iqr_photons', 'pct_range_photons', 'max_photons']:
        df[col] = np.nan

    # Process each FOV
    print(f"\nProcessing {len(fov_keys)} FOVs...")

    with h5py.File(H5_FILE_PATH_EXPERIMENTAL, 'r') as h5file:
        for i, fov_key in enumerate(fov_keys):
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{len(fov_keys)} FOVs...")

            if fov_key not in h5file:
                print(f"  WARNING: FOV {fov_key} not found in H5 file")
                continue

            fov = h5file[fov_key]
            fov_df = fov_groups.get_group(fov_key)

            # Process each channel
            for channel in ['green', 'orange']:
                channel_df = fov_df[fov_df['channel'] == channel]
                if len(channel_df) == 0:
                    continue

                # Load image data
                if 'image_blosc' not in fov:
                    continue

                compressed_data = fov['image_blosc'][:]
                image_data = blosc.decompress(compressed_data.tobytes())
                image_shape = tuple(fov['image_blosc'].attrs['original_shape'])
                image_dtype = np.dtype(fov['image_blosc'].attrs['original_dtype'])
                image_data = np.frombuffer(image_data, dtype=image_dtype).reshape(image_shape)

                # Get channel slice
                ch_idx = CHANNEL_MAP[channel]
                channel_image = image_data[:, ch_idx, :, :]

                # Convert to photons
                converted_image = convert_to_photons(channel_image, CHANNEL_PARAMS[channel])

                # Load label mask
                label_key = f'{channel}/label_mask'
                if label_key not in fov:
                    continue

                label_mask = fov[label_key][:]

                # Compute variance stats
                stats = compute_cluster_variance_stats(label_mask, converted_image)

                # Update dataframe
                for idx in channel_df.index:
                    label_id = int(df.loc[idx, 'label_id'])
                    if label_id in stats:
                        for col, val in stats[label_id].items():
                            df.loc[idx, col] = val

    # Count how many clusters got variance data
    n_with_variance = df['mean_photons'].notna().sum()
    print(f"\nAdded variance features to {n_with_variance}/{len(df)} clusters ({n_with_variance/len(df)*100:.1f}%)")

    # Save
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = data_dir / 'cluster_data_with_variance.csv'

    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

    # Print summary stats
    print("\nVariance feature summary:")
    for col in ['mean_photons', 'std_photons', 'cv_photons', 'iqr_photons', 'pct_range_photons']:
        valid = df[col].dropna()
        if len(valid) > 0:
            print(f"  {col}: median={valid.median():.2f}, range=[{valid.min():.2f}, {valid.max():.2f}]")


if __name__ == '__main__':
    main()
