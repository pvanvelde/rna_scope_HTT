#!/usr/bin/env python3
"""
Cluster Analysis for RNA Scope Cluster Data

Performs k-means and other clustering algorithms on cluster properties
to identify distinct populations (e.g., real signal vs background).

Usage:
    python cluster_analysis.py --data_dir ./output
    python cluster_analysis.py --data_dir ./output --n_clusters 4
    python cluster_analysis.py --data_dir ./output --channel green
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import argparse


def load_and_prepare_data(csv_path: Path, channel: str = None,
                          passes_threshold: bool = True) -> pd.DataFrame:
    """
    Load cluster data and prepare for clustering.

    Args:
        csv_path: Path to cluster_data.csv
        channel: Filter by channel (None = all)
        passes_threshold: Only include clusters passing threshold

    Returns:
        DataFrame with cluster data
    """
    df = pd.read_csv(csv_path)

    # Filter by threshold
    if passes_threshold and 'passes_threshold' in df.columns:
        df = df[df['passes_threshold'] == True].copy()

    # Filter by channel
    if channel is not None:
        df = df[df['channel'] == channel].copy()

    # Filter out invalid values
    df = df[(df['volume_um3'] > 0) & (df['raw_intensity'] > 0)].copy()

    return df


def compute_features(df: pd.DataFrame, feature_set: str = 'density_only') -> tuple:
    """
    Compute features for clustering.

    Uses mRNA equivalent instead of raw intensity since it's normalized
    per slide (accounts for slide-to-slide variation in photon thresholds).

    Args:
        df: DataFrame with cluster data
        feature_set: Which features to use for clustering:
            - 'density_only': Just log_density (best for separating high/low density populations)
            - 'volume_density': log_volume and log_density
            - 'all': log_volume, log_mrna, log_density

    Returns:
        (feature_matrix, feature_names, df_with_features)
    """
    df = df.copy()

    # Log-transform volume and mRNA equivalent (they're log-normally distributed)
    # Use mRNA equivalent (not raw intensity) because it's slide-normalized
    df['log_volume'] = np.log10(df['volume_um3'])
    df['log_mrna'] = np.log10(df['mrna_equiv'].clip(lower=0.1))  # Clip to avoid log(0)
    df['log_density'] = np.log10(df['density'].clip(lower=0.01))

    # mRNA per volume (on log scale this is log_mrna - log_volume)
    df['log_mrna_per_volume'] = df['log_mrna'] - df['log_volume']

    # Select features based on feature_set
    if feature_set == 'density_only':
        # Best for separating the two density populations visible in log-log plot
        feature_cols = ['log_density']
    elif feature_set == 'volume_density':
        # Volume and density (independent dimensions)
        feature_cols = ['log_volume', 'log_density']
    else:  # 'all'
        # All features (note: density is redundant with volume+mrna)
        feature_cols = ['log_volume', 'log_mrna', 'log_density']

    X = df[feature_cols].values

    return X, feature_cols, df


def find_optimal_k(X: np.ndarray, k_range: range = range(2, 8), max_samples_silhouette: int = 10000) -> dict:
    """
    Find optimal number of clusters using silhouette score and elbow method.

    Returns:
        dict with 'silhouette_scores', 'inertias', 'best_k_silhouette'
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    silhouette_scores = []
    inertias = []

    # Subsample for silhouette score (it's O(n²))
    if len(X_scaled) > max_samples_silhouette:
        np.random.seed(42)
        sample_idx = np.random.choice(len(X_scaled), max_samples_silhouette, replace=False)
        X_sample = X_scaled[sample_idx]
    else:
        X_sample = X_scaled
        sample_idx = np.arange(len(X_scaled))

    for k in k_range:
        print(f"  Testing k={k}...")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)

        # Silhouette on subsample
        labels_sample = labels[sample_idx]
        silhouette_scores.append(silhouette_score(X_sample, labels_sample))
        inertias.append(kmeans.inertia_)

    best_k = k_range[np.argmax(silhouette_scores)]

    return {
        'k_range': list(k_range),
        'silhouette_scores': silhouette_scores,
        'inertias': inertias,
        'best_k_silhouette': best_k
    }


def perform_kmeans(X: np.ndarray, n_clusters: int) -> tuple:
    """
    Perform k-means clustering.

    Returns:
        (labels, kmeans_model, scaler)
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    return labels, kmeans, scaler


def plot_clustering_results(df: pd.DataFrame, labels: np.ndarray,
                            feature_cols: list, n_clusters: int,
                            optimal_k_info: dict, channel_name: str,
                            output_dir: Path = None, max_plot_points: int = 20000):
    """
    Create comprehensive visualization of clustering results.
    """
    print(f"Creating visualization...")
    fig = plt.figure(figsize=(18, 14))

    # Color palette for clusters
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    cluster_colors = [colors[l] for l in labels]

    # Subsample for scatter plots if needed
    if len(df) > max_plot_points:
        np.random.seed(42)
        plot_idx = np.random.choice(len(df), max_plot_points, replace=False)
        df_plot = df.iloc[plot_idx].reset_index(drop=True)
        labels_plot = labels[plot_idx]
    else:
        df_plot = df
        labels_plot = labels

    # === Row 1: Optimal k analysis ===
    # Silhouette scores
    ax1 = fig.add_subplot(3, 4, 1)
    ax1.plot(optimal_k_info['k_range'], optimal_k_info['silhouette_scores'], 'bo-')
    ax1.axvline(x=n_clusters, color='r', linestyle='--', alpha=0.5, label=f'k={n_clusters}')
    ax1.set_xlabel('Number of clusters (k)')
    ax1.set_ylabel('Silhouette Score')
    ax1.set_title('Silhouette Score vs k')
    ax1.legend()

    # Elbow plot (inertia)
    ax2 = fig.add_subplot(3, 4, 2)
    ax2.plot(optimal_k_info['k_range'], optimal_k_info['inertias'], 'go-')
    ax2.axvline(x=n_clusters, color='r', linestyle='--', alpha=0.5, label=f'k={n_clusters}')
    ax2.set_xlabel('Number of clusters (k)')
    ax2.set_ylabel('Inertia (within-cluster sum of squares)')
    ax2.set_title('Elbow Method')
    ax2.legend()

    # Cluster size distribution
    ax3 = fig.add_subplot(3, 4, 3)
    unique, counts = np.unique(labels, return_counts=True)
    bars = ax3.bar(unique, counts, color=[colors[i] for i in unique])
    ax3.set_xlabel('Cluster')
    ax3.set_ylabel('Count')
    ax3.set_title('Cluster Sizes')
    for bar, count in zip(bars, counts):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                 f'{count}', ha='center', va='bottom', fontsize=9)

    # PCA visualization (using subsampled data) - or 1D histogram if single feature
    ax4 = fig.add_subplot(3, 4, 4)
    if len(feature_cols) >= 2:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_plot[feature_cols].values)
        n_comp = min(2, len(feature_cols))
        pca = PCA(n_components=n_comp)
        X_pca = pca.fit_transform(X_scaled)
        scatter = ax4.scatter(X_pca[:, 0], X_pca[:, 1] if n_comp > 1 else np.zeros(len(X_pca)),
                             c=labels_plot, cmap='tab10', alpha=0.5, s=5)
        ax4.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        ax4.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)' if n_comp > 1 else '')
        ax4.set_title('PCA of Clusters')
    else:
        # Single feature - show distribution per cluster
        for i in range(n_clusters):
            mask = labels == i
            ax4.hist(df.loc[mask, feature_cols[0]], bins=50, alpha=0.5,
                    color=colors[i], label=f'Cluster {i}')
        ax4.set_xlabel(feature_cols[0])
        ax4.set_ylabel('Count')
        ax4.set_title('Feature Distribution')
        ax4.legend(fontsize=7)

    # === Row 2: Feature space visualizations (using subsampled data) ===
    # Log Volume vs Log mRNA equivalent (colored by cluster)
    ax5 = fig.add_subplot(3, 4, 5)
    for i in range(n_clusters):
        mask = labels_plot == i
        # Get full counts for legend
        full_count = (labels == i).sum()
        ax5.scatter(df_plot.loc[mask, 'log_volume'], df_plot.loc[mask, 'log_mrna'],
                   alpha=0.4, s=5, c=[colors[i]], label=f'Cluster {i} (n={full_count})')
    ax5.set_xlabel('Log10(Volume µm³)')
    ax5.set_ylabel('Log10(mRNA equivalent)')
    ax5.set_title('Log-Log: mRNA vs Volume')
    ax5.legend(fontsize=7, loc='lower right')

    # Log Volume vs Density (colored by cluster)
    ax6 = fig.add_subplot(3, 4, 6)
    for i in range(n_clusters):
        mask = labels_plot == i
        ax6.scatter(df_plot.loc[mask, 'log_volume'], df_plot.loc[mask, 'density'],
                   alpha=0.4, s=5, c=[colors[i]], label=f'Cluster {i}')
    ax6.set_xlabel('Log10(Volume µm³)')
    ax6.set_ylabel('Density (mRNA/µm³)')
    ax6.set_title('Density vs Log Volume')

    # Density histogram per cluster
    ax7 = fig.add_subplot(3, 4, 7)
    for i in range(n_clusters):
        mask = labels == i
        ax7.hist(df.loc[mask, 'density'], bins=50, alpha=0.5,
                color=colors[i], label=f'Cluster {i}')
    ax7.set_xlabel('Density (mRNA/µm³)')
    ax7.set_ylabel('Count')
    ax7.set_title('Density Distribution per Cluster')
    ax7.legend(fontsize=7)

    # Volume histogram per cluster
    ax8 = fig.add_subplot(3, 4, 8)
    for i in range(n_clusters):
        mask = labels == i
        ax8.hist(df.loc[mask, 'volume_um3'], bins=50, alpha=0.5,
                color=colors[i], label=f'Cluster {i}')
    ax8.set_xlabel('Volume (µm³)')
    ax8.set_ylabel('Count')
    ax8.set_title('Volume Distribution per Cluster')
    ax8.legend(fontsize=7)

    # === Row 3: Original scale plots ===
    # Volume vs mRNA equivalent (linear, colored by cluster) - using subsampled data
    ax9 = fig.add_subplot(3, 4, 9)
    for i in range(n_clusters):
        mask = labels_plot == i
        ax9.scatter(df_plot.loc[mask, 'volume_um3'], df_plot.loc[mask, 'mrna_equiv'],
                   alpha=0.4, s=5, c=[colors[i]], label=f'Cluster {i}')
    ax9.set_xlabel('Volume (µm³)')
    ax9.set_ylabel('mRNA equivalent')
    ax9.set_title('mRNA vs Volume (linear)')
    ax9.set_xlim(0, df['volume_um3'].quantile(0.99) * 1.1)
    ax9.set_ylim(0, df['mrna_equiv'].quantile(0.99) * 1.1)

    # Density per cluster (boxplot) - this IS the mRNA per volume
    ax10 = fig.add_subplot(3, 4, 10)
    density_data = [df.loc[labels == i, 'density'].values for i in range(n_clusters)]
    bp = ax10.boxplot(density_data, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors[:n_clusters]):
        patch.set_facecolor(color)
    ax10.set_xlabel('Cluster')
    ax10.set_ylabel('Density (mRNA/µm³)')
    ax10.set_title('Density by Cluster')
    ax10.set_ylim(0, np.percentile(df['density'], 99) * 1.2)

    # Cluster statistics table
    ax11 = fig.add_subplot(3, 4, 11)
    ax11.axis('off')

    stats_text = "Cluster Statistics:\n" + "=" * 40 + "\n\n"
    for i in range(n_clusters):
        mask = labels == i
        cluster_df = df[mask]
        stats_text += f"Cluster {i} (n={mask.sum()}):\n"
        stats_text += f"  Density: {cluster_df['density'].median():.2f} (median)\n"
        stats_text += f"  Volume: {cluster_df['volume_um3'].median():.1f} µm³ (median)\n"
        stats_text += f"  mRNA: {cluster_df['mrna_equiv'].median():.1f} (median)\n\n"

    ax11.text(0.05, 0.95, stats_text, transform=ax11.transAxes, fontsize=9,
              verticalalignment='top', fontfamily='monospace')

    # mRNA equivalent per cluster (boxplot)
    ax12 = fig.add_subplot(3, 4, 12)
    mrna_data = [df.loc[labels == i, 'mrna_equiv'].values for i in range(n_clusters)]
    bp2 = ax12.boxplot(mrna_data, patch_artist=True)
    for patch, color in zip(bp2['boxes'], colors[:n_clusters]):
        patch.set_facecolor(color)
    ax12.set_xlabel('Cluster')
    ax12.set_ylabel('mRNA equivalent')
    ax12.set_title('mRNA Equivalent by Cluster')
    ax12.set_ylim(0, np.percentile(df['mrna_equiv'], 99) * 1.2)

    plt.suptitle(f'K-Means Clustering Analysis (k={n_clusters}) - {channel_name}\n'
                 f'Best k by silhouette: {optimal_k_info["best_k_silhouette"]}',
                 fontsize=14, y=1.02)

    plt.tight_layout()

    output_path = output_dir / f'cluster_analysis_k{n_clusters}_{channel_name}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def print_cluster_summary(df: pd.DataFrame, labels: np.ndarray, n_clusters: int):
    """Print detailed summary statistics for each cluster."""
    print("\n" + "=" * 70)
    print("CLUSTER SUMMARY")
    print("=" * 70)

    for i in range(n_clusters):
        mask = labels == i
        cluster_df = df[mask]

        print(f"\nCluster {i}: {mask.sum()} clusters ({mask.sum()/len(df)*100:.1f}%)")
        print("-" * 50)

        # Density stats
        print(f"  Density (mRNA/µm³):")
        print(f"    Mean: {cluster_df['density'].mean():.3f}")
        print(f"    Median: {cluster_df['density'].median():.3f}")
        print(f"    Std: {cluster_df['density'].std():.3f}")
        print(f"    Range: [{cluster_df['density'].min():.3f}, {cluster_df['density'].max():.3f}]")

        # Volume stats
        print(f"  Volume (µm³):")
        print(f"    Mean: {cluster_df['volume_um3'].mean():.2f}")
        print(f"    Median: {cluster_df['volume_um3'].median():.2f}")
        print(f"    Range: [{cluster_df['volume_um3'].min():.2f}, {cluster_df['volume_um3'].max():.2f}]")

        # mRNA stats
        print(f"  mRNA equivalent:")
        print(f"    Mean: {cluster_df['mrna_equiv'].mean():.1f}")
        print(f"    Median: {cluster_df['mrna_equiv'].median():.1f}")
        print(f"    Range: [{cluster_df['mrna_equiv'].min():.1f}, {cluster_df['mrna_equiv'].max():.1f}]")

        # Intensity per volume
        int_per_vol = cluster_df['raw_intensity'] / cluster_df['volume_um3']
        print(f"  Intensity/Volume:")
        print(f"    Median: {int_per_vol.median():.0f}")


def analyze_channel(df_channel: pd.DataFrame, channel_name: str, output_dir: Path,
                    n_clusters: int = None, compare_k: bool = False, feature_set: str = 'density_only'):
    """Analyze a single channel."""
    print(f"\n{'#'*70}")
    print(f"# CHANNEL: {channel_name.upper()} ({len(df_channel)} clusters)")
    print(f"{'#'*70}")

    # Compute features
    X, feature_cols, df_features = compute_features(df_channel, feature_set=feature_set)
    print(f"Features: {feature_cols}")

    # Find optimal k
    print("\nFinding optimal number of clusters...")
    optimal_k_info = find_optimal_k(X, k_range=range(2, 8))
    print(f"Silhouette scores: {dict(zip(optimal_k_info['k_range'],
                                          [f'{s:.3f}' for s in optimal_k_info['silhouette_scores']]))}")
    print(f"Best k by silhouette: {optimal_k_info['best_k_silhouette']}")

    # Determine n_clusters
    if n_clusters is not None:
        k_to_use = n_clusters
        print(f"\nUsing specified k={k_to_use}")
    else:
        k_to_use = optimal_k_info['best_k_silhouette']
        print(f"\nUsing auto-detected k={k_to_use}")

    # Compare k=2, k=3 and k=4 if requested
    if compare_k:
        for k in [2, 3, 4]:
            print(f"\n{'='*70}")
            print(f"K-MEANS WITH k={k}")
            print('='*70)

            labels, kmeans, scaler = perform_kmeans(X, k)
            df_features['cluster'] = labels

            print_cluster_summary(df_features, labels, k)
            plot_clustering_results(df_features, labels, feature_cols, k,
                                   optimal_k_info, channel_name, output_dir)
    else:
        # Single k analysis
        print(f"\n{'='*70}")
        print(f"K-MEANS WITH k={k_to_use}")
        print('='*70)

        labels, kmeans, scaler = perform_kmeans(X, k_to_use)
        df_features['cluster'] = labels

        print_cluster_summary(df_features, labels, k_to_use)
        plot_clustering_results(df_features, labels, feature_cols, k_to_use,
                               optimal_k_info, channel_name, output_dir)

        # Save cluster assignments
        output_csv = output_dir / f'cluster_assignments_k{k_to_use}_{channel_name}.csv'
        df_features.to_csv(output_csv, index=False)
        print(f"\nSaved cluster assignments to {output_csv}")


def main():
    parser = argparse.ArgumentParser(description='Cluster analysis of RNA scope cluster data')
    parser.add_argument('--data_dir', type=str, default='./output',
                        help='Directory containing cluster_data.csv')
    parser.add_argument('--n_clusters', type=int, default=None,
                        help='Number of clusters (default: auto-detect via silhouette)')
    parser.add_argument('--channel', type=str, default=None,
                        help='Filter by channel (green/orange), default: analyze each channel separately')
    parser.add_argument('--no_threshold_filter', action='store_true',
                        help='Include clusters that do NOT pass threshold')
    parser.add_argument('--compare_k', action='store_true',
                        help='Compare k=2, k=3 and k=4 side by side')
    parser.add_argument('--feature_set', type=str, default='density_only',
                        choices=['density_only', 'volume_density', 'all'],
                        help='Features for clustering: density_only (default), volume_density, or all')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save output figures')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    csv_path = data_dir / 'cluster_data.csv'

    if not csv_path.exists():
        print(f"ERROR: cluster_data.csv not found at {csv_path}")
        sys.exit(1)

    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = data_dir / 'clustering_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all data first
    print(f"Loading data from {csv_path}...")
    df_all = pd.read_csv(csv_path)

    # Filter by threshold
    if not args.no_threshold_filter and 'passes_threshold' in df_all.columns:
        df_all = df_all[df_all['passes_threshold'] == True].copy()

    # Filter out invalid values
    df_all = df_all[(df_all['volume_um3'] > 0) & (df_all['raw_intensity'] > 0)].copy()

    print(f"Loaded {len(df_all)} clusters total")

    # Determine which channels to analyze
    if args.channel:
        channels = [args.channel]
    else:
        channels = df_all['channel'].unique().tolist()
        print(f"Will analyze channels: {channels}")

    # Analyze each channel
    for channel in channels:
        df_channel = df_all[df_all['channel'] == channel].copy()
        analyze_channel(df_channel, channel, output_dir,
                       n_clusters=args.n_clusters, compare_k=args.compare_k,
                       feature_set=args.feature_set)

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print(f"Results saved to: {output_dir}")
    print("="*70)


if __name__ == '__main__':
    main()
