#!/usr/bin/env python3
"""
Visualize Outliers in Cluster Data

Creates clean visualizations showing outliers detected by the threshold method
(large volume + low density) highlighted in the log-log intensity vs volume plots.

Usage:
    python visualize_outliers.py
    python visualize_outliers.py --method Threshold
    python visualize_outliers.py --method IsolationForest
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse


def load_data_with_outliers(data_dir: Path, outlier_method: str = 'Threshold') -> pd.DataFrame:
    """Load cluster data and merge with outlier labels."""
    # Load main cluster data
    cluster_csv = data_dir / 'cluster_data.csv'
    if not cluster_csv.exists():
        print(f"ERROR: cluster_data.csv not found at {cluster_csv}")
        sys.exit(1)

    df = pd.read_csv(cluster_csv)

    # Filter to only clusters that pass threshold
    if 'passes_threshold' in df.columns:
        df = df[df['passes_threshold'] == True].copy()

    # Load outlier labels for each channel
    outlier_col = f'outlier_{outlier_method}'
    df['is_outlier'] = False

    outlier_dir = data_dir / 'outlier_detection'
    for channel in df['channel'].unique():
        outlier_csv = outlier_dir / f'outliers_{channel}.csv'
        if outlier_csv.exists():
            outlier_df = pd.read_csv(outlier_csv)
            if outlier_col in outlier_df.columns:
                # Create lookup by (slide, label_id)
                outlier_lookup = {}
                for _, row in outlier_df.iterrows():
                    key = (row['slide'], row['label_id'])
                    outlier_lookup[key] = row[outlier_col]

                # Apply to main dataframe
                ch_mask = df['channel'] == channel
                for idx in df[ch_mask].index:
                    key = (df.loc[idx, 'slide'], df.loc[idx, 'label_id'])
                    if key in outlier_lookup:
                        df.loc[idx, 'is_outlier'] = outlier_lookup[key]
        else:
            print(f"WARNING: Outlier file not found: {outlier_csv}")

    return df


def plot_outliers(df: pd.DataFrame, outlier_method: str, output_dir: Path):
    """Create visualization of outliers in log-log space."""
    channels = df['channel'].unique()

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    for i, channel in enumerate(channels):
        ch_data = df[df['channel'] == channel].copy()

        # Filter valid data
        valid = (ch_data['volume_um3'] > 0) & (ch_data['raw_intensity'] > 0)
        ch_valid = ch_data[valid]

        normal = ch_valid[~ch_valid['is_outlier']]
        outliers = ch_valid[ch_valid['is_outlier']]

        n_normal = len(normal)
        n_outliers = len(outliers)
        pct_outliers = n_outliers / len(ch_valid) * 100 if len(ch_valid) > 0 else 0

        # Top row: log-log scatter
        ax = axes[0, i]
        ax.scatter(normal['volume_um3'], normal['raw_intensity'],
                   alpha=0.3, s=3, c='blue', label=f'Normal (n={n_normal})')
        ax.scatter(outliers['volume_um3'], outliers['raw_intensity'],
                   alpha=0.6, s=8, c='red', label=f'Outlier (n={n_outliers}, {pct_outliers:.1f}%)')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Volume (µm³)')
        ax.set_ylabel('Raw Intensity (photons)')
        ax.set_title(f'{channel.capitalize()} - Log-Log Intensity vs Volume')
        ax.legend(loc='lower right', fontsize=9)

        # Add reference lines
        if len(ch_valid) > 0:
            vol_range = np.logspace(np.log10(ch_valid['volume_um3'].min()),
                                    np.log10(ch_valid['volume_um3'].max()), 100)
            median_int = ch_valid['raw_intensity'].median()
            median_vol = ch_valid['volume_um3'].median()
            ref_intensity = median_int * (vol_range / median_vol)
            ax.plot(vol_range, ref_intensity, 'k--', alpha=0.5, linewidth=1.5, label='median density')

        # Bottom row: density histograms
        ax = axes[1, i]
        bins = np.linspace(0, ch_valid['density'].quantile(0.99), 50)
        ax.hist(normal['density'], bins=bins, alpha=0.6, color='blue',
                label=f'Normal (median={normal["density"].median():.2f})', density=True)
        ax.hist(outliers['density'], bins=bins, alpha=0.6, color='red',
                label=f'Outlier (median={outliers["density"].median():.2f})', density=True)
        ax.set_xlabel('Density (mRNA/µm³)')
        ax.set_ylabel('Density (normalized)')
        ax.set_title(f'{channel.capitalize()} - Density Distribution')
        ax.legend(loc='upper right', fontsize=9)

    plt.suptitle(f'Outlier Detection: {outlier_method} method\n'
                 f'Outliers = large volume + low density (background artifacts)',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    output_path = output_dir / f'outlier_visualization_{outlier_method}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_outliers_mRNA_view(df: pd.DataFrame, outlier_method: str, output_dir: Path):
    """Create visualization using mRNA equivalent (slide-normalized)."""
    channels = df['channel'].unique()

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    for i, channel in enumerate(channels):
        ch_data = df[df['channel'] == channel].copy()

        # Filter valid data
        valid = (ch_data['volume_um3'] > 0) & (ch_data['mrna_equiv'] > 0)
        ch_valid = ch_data[valid]

        normal = ch_valid[~ch_valid['is_outlier']]
        outliers = ch_valid[ch_valid['is_outlier']]

        n_normal = len(normal)
        n_outliers = len(outliers)
        pct_outliers = n_outliers / len(ch_valid) * 100 if len(ch_valid) > 0 else 0

        # Top row: log-log scatter with mRNA
        ax = axes[0, i]
        ax.scatter(normal['volume_um3'], normal['mrna_equiv'],
                   alpha=0.3, s=3, c='blue', label=f'Normal (n={n_normal})')
        ax.scatter(outliers['volume_um3'], outliers['mrna_equiv'],
                   alpha=0.6, s=8, c='red', label=f'Outlier (n={n_outliers}, {pct_outliers:.1f}%)')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Volume (µm³)')
        ax.set_ylabel('mRNA equivalent')
        ax.set_title(f'{channel.capitalize()} - Log-Log mRNA vs Volume')
        ax.legend(loc='lower right', fontsize=9)

        # Add reference lines
        if len(ch_valid) > 0:
            vol_range = np.logspace(np.log10(ch_valid['volume_um3'].min()),
                                    np.log10(ch_valid['volume_um3'].max()), 100)
            median_mrna = ch_valid['mrna_equiv'].median()
            median_vol = ch_valid['volume_um3'].median()
            ref_mrna = median_mrna * (vol_range / median_vol)
            ax.plot(vol_range, ref_mrna, 'k--', alpha=0.5, linewidth=1.5, label='median density')

        # Bottom row: volume histograms
        ax = axes[1, i]
        bins = np.linspace(0, ch_valid['volume_um3'].quantile(0.99), 50)
        ax.hist(normal['volume_um3'], bins=bins, alpha=0.6, color='blue',
                label=f'Normal (median={normal["volume_um3"].median():.1f} µm³)', density=True)
        ax.hist(outliers['volume_um3'], bins=bins, alpha=0.6, color='red',
                label=f'Outlier (median={outliers["volume_um3"].median():.1f} µm³)', density=True)
        ax.set_xlabel('Volume (µm³)')
        ax.set_ylabel('Density (normalized)')
        ax.set_title(f'{channel.capitalize()} - Volume Distribution')
        ax.legend(loc='upper right', fontsize=9)

    plt.suptitle(f'Outlier Detection: {outlier_method} method (mRNA view)\n'
                 f'Outliers have larger volumes with lower mRNA density',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    output_path = output_dir / f'outlier_visualization_mRNA_{outlier_method}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize outliers in cluster data')
    parser.add_argument('--data_dir', type=str, default='./output',
                        help='Directory containing cluster_data.csv and outlier_detection/')
    parser.add_argument('--method', type=str, default='Threshold',
                        choices=['Threshold', 'IsolationForest', 'LOF'],
                        help='Outlier detection method to visualize (default: Threshold)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for figures')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else data_dir / 'outlier_detection'
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {data_dir}...")
    df = load_data_with_outliers(data_dir, args.method)

    n_total = len(df)
    n_outliers = df['is_outlier'].sum()
    print(f"Loaded {n_total} clusters, {n_outliers} outliers ({n_outliers/n_total*100:.1f}%)")

    # Summary by channel
    for channel in df['channel'].unique():
        ch_data = df[df['channel'] == channel]
        ch_outliers = ch_data['is_outlier'].sum()
        print(f"  {channel}: {len(ch_data)} clusters, {ch_outliers} outliers ({ch_outliers/len(ch_data)*100:.1f}%)")

    print(f"\nGenerating visualizations...")
    plot_outliers(df, args.method, output_dir)
    plot_outliers_mRNA_view(df, args.method, output_dir)

    print(f"\nDone! Figures saved to {output_dir}")


if __name__ == '__main__':
    main()
