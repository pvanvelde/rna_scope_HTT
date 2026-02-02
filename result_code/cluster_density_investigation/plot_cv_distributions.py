#!/usr/bin/env python3
"""
Plot CV (Coefficient of Variation) distributions for cluster data.

Usage:
    python plot_cv_distributions.py
    python plot_cv_distributions.py --data_dir ./output
    python plot_cv_distributions.py --save_path ./output/cv_distributions.png
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_cv_distributions(df: pd.DataFrame, output_path: Path = None):
    """Create CV distribution plots for both channels."""

    # Create figure
    fig, axes = plt.subplots(3, 3, figsize=(15, 14))

    for i, channel in enumerate(['green', 'orange']):
        ch_data = df[df['channel'] == channel]
        color = 'green' if channel == 'green' else 'orange'

        # Row 0: CV histogram
        ax = axes[0, i]
        cv_data = ch_data['cv_photons'].dropna()
        ax.hist(cv_data, bins=50, alpha=0.7, color=color)
        ax.axvline(cv_data.median(), color='red', linestyle='--', linewidth=2,
                   label=f'median={cv_data.median():.2f}')
        ax.axvline(cv_data.mean(), color='blue', linestyle=':', linewidth=2,
                   label=f'mean={cv_data.mean():.2f}')
        ax.set_xlabel('CV (photons)')
        ax.set_ylabel('Count')
        ax.set_title(f'{channel.capitalize()} - CV Distribution (n={len(cv_data)})')
        ax.legend()

        # Row 1: CV vs Density scatter
        ax = axes[1, i]
        ax.scatter(ch_data['density'], ch_data['cv_photons'], alpha=0.2, s=3, c=color)
        ax.set_xlabel('Density (mRNA/µm³)')
        ax.set_ylabel('CV (photons)')
        ax.set_title(f'{channel.capitalize()} - CV vs Density')
        ax.set_xlim(0, ch_data['density'].quantile(0.99) * 1.1)
        ax.set_ylim(0, ch_data['cv_photons'].quantile(0.99) * 1.1)

        # Row 2: CV vs Volume scatter (per channel)
        ax = axes[2, i]
        ax.scatter(ch_data['volume_um3'], ch_data['cv_photons'], alpha=0.2, s=3, c=color)
        ax.set_xlabel('Volume (µm³)')
        ax.set_ylabel('CV (photons)')
        ax.set_title(f'{channel.capitalize()} - CV vs Volume')
        ax.set_xlim(0, ch_data['volume_um3'].quantile(0.99) * 1.1)
        ax.set_ylim(0, ch_data['cv_photons'].quantile(0.99) * 1.1)

        # Print stats
        print(f"\n{channel.upper()}:")
        print(f"  CV mean: {cv_data.mean():.3f}")
        print(f"  CV median: {cv_data.median():.3f}")
        print(f"  CV std: {cv_data.std():.3f}")
        print(f"  CV min: {cv_data.min():.3f}")
        print(f"  CV max: {cv_data.max():.3f}")

    # Row 0, Panel 3: Combined histogram comparison
    ax = axes[0, 2]
    for channel in ['green', 'orange']:
        ch_data = df[df['channel'] == channel]['cv_photons'].dropna()
        color = 'green' if channel == 'green' else 'orange'
        ax.hist(ch_data, bins=50, alpha=0.5, color=color, label=channel, density=True)
    ax.set_xlabel('CV (photons)')
    ax.set_ylabel('Density')
    ax.set_title('CV Distribution Comparison')
    ax.legend()

    # Row 1, Panel 3: CV vs Volume (combined)
    ax = axes[1, 2]
    for channel in ['green', 'orange']:
        ch_data = df[df['channel'] == channel]
        color = 'green' if channel == 'green' else 'orange'
        ax.scatter(ch_data['volume_um3'], ch_data['cv_photons'], alpha=0.2, s=3, c=color, label=channel)
    ax.set_xlabel('Volume (µm³)')
    ax.set_ylabel('CV (photons)')
    ax.set_title('CV vs Volume (both channels)')
    ax.set_xlim(0, df['volume_um3'].quantile(0.99) * 1.1)
    ax.set_ylim(0, df['cv_photons'].quantile(0.99) * 1.1)
    ax.legend()

    # Row 2, Panel 3: CV vs mRNA equivalent
    ax = axes[2, 2]
    for channel in ['green', 'orange']:
        ch_data = df[df['channel'] == channel]
        color = 'green' if channel == 'green' else 'orange'
        ax.scatter(ch_data['mrna_equiv'], ch_data['cv_photons'], alpha=0.2, s=3, c=color, label=channel)
    ax.set_xlabel('mRNA equivalent')
    ax.set_ylabel('CV (photons)')
    ax.set_title('CV vs mRNA equivalent')
    ax.set_xlim(0, df['mrna_equiv'].quantile(0.99) * 1.1)
    ax.set_ylim(0, df['cv_photons'].quantile(0.99) * 1.1)
    ax.legend()

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nSaved: {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Plot CV distributions for cluster data')
    parser.add_argument('--data_dir', type=str, default='./output',
                        help='Directory containing cluster_data.csv')
    parser.add_argument('--save_path', type=str, default=None,
                        help='Path to save figure (default: show interactively)')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    cluster_csv = data_dir / 'cluster_data.csv'

    if not cluster_csv.exists():
        print(f"ERROR: cluster_data.csv not found at {cluster_csv}")
        print("Please run investigate_cluster_density.py first.")
        return

    # Load data
    print(f"Loading cluster data from {cluster_csv}...")
    df = pd.read_csv(cluster_csv)
    print(f"Total clusters: {len(df)}")

    # Check if CV column exists
    if 'cv_photons' not in df.columns:
        print("ERROR: cv_photons column not found in cluster_data.csv")
        print(f"Available columns: {df.columns.tolist()}")
        return

    # Determine output path
    if args.save_path:
        output_path = Path(args.save_path)
    else:
        output_path = data_dir / 'cv_distributions.png'

    # Create plots
    plot_cv_distributions(df, output_path)


if __name__ == '__main__':
    main()
