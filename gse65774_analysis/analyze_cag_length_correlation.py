#!/usr/bin/env python3
"""
Cross-CAG repeat analysis: Test if HTT-UBC-POLR2 correlations emerge
when comparing across different CAG repeat lengths (Q20-Q175)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

# Target genes
GENES_OF_INTEREST = {
    'mHTT': 'ENSMUSG00000029189',
    'UBC': 'ENSMUSG00000019505',
    'POLR2A': 'ENSMUSG00000005198',
}

def load_data():
    """Load FPKM data and metadata"""
    print("Loading data...")
    fpkm = pd.read_csv('data/GSE65774_Striatum_mRNA_FPKM_processedData.txt',
                       sep='\t', index_col=0)
    metadata = pd.read_csv('data/metadata_complete.csv')
    metadata = metadata.set_index('sample_id')
    metadata = metadata.loc[fpkm.columns]

    print(f"✓ Loaded {fpkm.shape[0]} genes across {fpkm.shape[1]} samples")
    return fpkm, metadata

def analyze_by_cag_repeats(fpkm, metadata):
    """Analyze gene expression across CAG repeat lengths"""
    print("\n" + "="*80)
    print("EXPRESSION ACROSS CAG REPEAT LENGTHS")
    print("="*80)

    # Extract gene expression
    gene_data = {}
    for gene_name, ensembl_id in GENES_OF_INTEREST.items():
        if ensembl_id in fpkm.index:
            gene_data[gene_name] = fpkm.loc[ensembl_id]

    expr_df = pd.DataFrame(gene_data)
    expr_df = expr_df.merge(metadata, left_index=True, right_index=True, how='left')

    # Summary by CAG repeat length
    print("\nExpression levels by CAG repeat length:")
    print("\n" + "-"*80)

    for cag in sorted(expr_df['cag_repeats'].unique()):
        subset = expr_df[expr_df['cag_repeats'] == cag]
        genotype = 'WT' if cag == 0 else f'Q{cag}'

        print(f"\n{genotype} (n={len(subset)}):")
        for gene in GENES_OF_INTEREST.keys():
            if gene in subset.columns:
                mean_val = subset[gene].mean()
                std_val = subset[gene].std()
                print(f"  {gene}: {mean_val:.2f} ± {std_val:.2f}")

    return expr_df

def correlation_with_cag_length(expr_df):
    """Test correlation between CAG repeat length and gene expression"""
    print("\n" + "="*80)
    print("CORRELATION WITH CAG REPEAT LENGTH")
    print("="*80)

    results = []

    # Only use HET samples (exclude WT)
    het_data = expr_df[expr_df['genotype'] != 'WT'].copy()

    print(f"\nAnalyzing {len(het_data)} heterozygous samples (excluding WT)")
    print(f"CAG repeat range: {het_data['cag_repeats'].min()}-{het_data['cag_repeats'].max()}")

    for gene in GENES_OF_INTEREST.keys():
        if gene in het_data.columns:
            # Remove missing values
            data = het_data[['cag_repeats', gene]].dropna()

            # Pearson correlation
            pearson_r, pearson_p = stats.pearsonr(data['cag_repeats'], data[gene])

            # Spearman correlation
            spearman_r, spearman_p = stats.spearmanr(data['cag_repeats'], data[gene])

            print(f"\n{gene} vs CAG repeat length:")
            print(f"  Pearson r = {pearson_r:.4f}, p = {pearson_p:.2e}")
            print(f"  Spearman ρ = {spearman_r:.4f}, p = {spearman_p:.2e}")

            if pearson_p < 0.05:
                direction = "positive" if pearson_r > 0 else "negative"
                print(f"  ✓ Significant {direction} correlation")
            else:
                print(f"  ✗ Not significant")

            results.append({
                'gene': gene,
                'pearson_r': pearson_r,
                'pearson_p': pearson_p,
                'spearman_r': spearman_r,
                'spearman_p': spearman_p,
                'n_samples': len(data)
            })

    results_df = pd.DataFrame(results)
    return results_df, het_data

def cross_gene_correlation_by_cag(expr_df):
    """Test if gene-gene correlations vary with CAG repeat groups"""
    print("\n" + "="*80)
    print("GENE-GENE CORRELATIONS BY CAG REPEAT GROUP")
    print("="*80)

    # Define CAG groups
    cag_groups = {
        'Low (Q20-Q80)': [20, 80],
        'Medium (Q92-Q111)': [92, 111],
        'High (Q140-Q175)': [140, 175]
    }

    gene_pairs = [('mHTT', 'UBC'), ('mHTT', 'POLR2A'), ('UBC', 'POLR2A')]
    results = []

    for group_name, cag_values in cag_groups.items():
        group_data = expr_df[expr_df['cag_repeats'].isin(cag_values)]

        if len(group_data) < 5:
            continue

        print(f"\n{group_name} (n={len(group_data)}):")

        for gene1, gene2 in gene_pairs:
            if gene1 in group_data.columns and gene2 in group_data.columns:
                data = group_data[[gene1, gene2]].dropna()

                if len(data) < 5:
                    continue

                pearson_r, pearson_p = stats.pearsonr(data[gene1], data[gene2])

                print(f"  {gene1} vs {gene2}: r = {pearson_r:.3f}, p = {pearson_p:.2e}")

                results.append({
                    'cag_group': group_name,
                    'gene1': gene1,
                    'gene2': gene2,
                    'pearson_r': pearson_r,
                    'pearson_p': pearson_p,
                    'n_samples': len(data)
                })

    results_df = pd.DataFrame(results)
    return results_df

def plot_cag_correlations(expr_df, het_data):
    """Create visualizations for CAG repeat correlations"""

    # Figure 1: Gene expression vs CAG repeat length
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, gene in enumerate(['mHTT', 'UBC', 'POLR2A']):
        ax = axes[idx]

        if gene not in expr_df.columns:
            continue

        # Plot by CAG repeat as categorical
        cag_order = [0, 20, 80, 92, 111, 140, 175]
        cag_present = [c for c in cag_order if c in expr_df['cag_repeats'].values]

        data_by_cag = []
        labels = []

        for cag in cag_present:
            subset = expr_df[expr_df['cag_repeats'] == cag][gene]
            if len(subset) > 0:
                data_by_cag.append(subset)
                labels.append('WT' if cag == 0 else f'Q{cag}')

        # Box plot
        bp = ax.boxplot(data_by_cag, labels=labels, patch_artist=True,
                       showmeans=True, meanline=True)

        # Color by severity
        colors = ['lightgreen'] + ['lightblue', 'skyblue', 'orange',
                                    'darkorange', 'red', 'darkred'][:len(data_by_cag)-1]
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_xlabel('CAG Repeat Length', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'{gene} Expression (FPKM)', fontsize=12, fontweight='bold')
        ax.set_title(f'{gene} Expression by CAG Repeats', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    plt.savefig('figures/cag_repeat_expression.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ CAG repeat expression plot saved to figures/cag_repeat_expression.png")
    plt.close()

    # Figure 2: Scatter plots with CAG repeat as continuous variable (HET only)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    gene_pairs = [('mHTT', 'UBC'), ('mHTT', 'POLR2A'), ('UBC', 'POLR2A')]

    for idx, (gene1, gene2) in enumerate(gene_pairs):
        ax = axes[idx]

        if gene1 not in het_data.columns or gene2 not in het_data.columns:
            continue

        # Color by CAG repeat length
        scatter = ax.scatter(het_data[gene1], het_data[gene2],
                           c=het_data['cag_repeats'], cmap='YlOrRd',
                           s=80, alpha=0.7, edgecolors='black', linewidth=0.5)

        # Add regression line
        data = het_data[[gene1, gene2]].dropna()
        if len(data) > 2:
            z = np.polyfit(data[gene1], data[gene2], 1)
            p = np.poly1d(z)
            x_line = np.linspace(data[gene1].min(), data[gene1].max(), 100)
            ax.plot(x_line, p(x_line), 'r--', linewidth=2, alpha=0.8)

        # Calculate correlation
        pearson_r, pearson_p = stats.pearsonr(data[gene1], data[gene2])

        ax.set_xlabel(f'{gene1} Expression (FPKM)', fontsize=11, fontweight='bold')
        ax.set_ylabel(f'{gene2} Expression (FPKM)', fontsize=11, fontweight='bold')
        ax.set_title(f'{gene1} vs {gene2}\n(HET samples only)\nr = {pearson_r:.3f}, p = {pearson_p:.2e}',
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('CAG Repeats', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig('figures/cag_gene_correlations.png', dpi=300, bbox_inches='tight')
    print(f"✓ CAG gene correlation plot saved to figures/cag_gene_correlations.png")
    plt.close()

    # Figure 3: Gene expression vs CAG repeats (scatter with trend)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, gene in enumerate(['mHTT', 'UBC', 'POLR2A']):
        ax = axes[idx]

        if gene not in het_data.columns:
            continue

        # Scatter plot
        ax.scatter(het_data['cag_repeats'], het_data[gene],
                  c=het_data['cag_repeats'], cmap='YlOrRd',
                  s=80, alpha=0.7, edgecolors='black', linewidth=0.5)

        # Add regression line
        data = het_data[['cag_repeats', gene]].dropna()
        z = np.polyfit(data['cag_repeats'], data[gene], 1)
        p = np.poly1d(z)
        x_line = np.linspace(data['cag_repeats'].min(), data['cag_repeats'].max(), 100)
        ax.plot(x_line, p(x_line), 'r--', linewidth=2, alpha=0.8, label='Linear fit')

        # Calculate correlation
        pearson_r, pearson_p = stats.pearsonr(data['cag_repeats'], data[gene])

        ax.set_xlabel('CAG Repeat Length', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'{gene} Expression (FPKM)', fontsize=12, fontweight='bold')
        ax.set_title(f'{gene} vs CAG Repeats\nr = {pearson_r:.3f}, p = {pearson_p:.2e}',
                    fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.savefig('figures/gene_vs_cag_scatter.png', dpi=300, bbox_inches='tight')
    print(f"✓ Gene vs CAG scatter plots saved to figures/gene_vs_cag_scatter.png")
    plt.close()

def main():
    """Main analysis workflow"""

    print("="*80)
    print("CROSS-CAG REPEAT LENGTH CORRELATION ANALYSIS")
    print("="*80)

    # Load data
    fpkm, metadata = load_data()

    # Analyze by CAG repeats
    expr_df = analyze_by_cag_repeats(fpkm, metadata)

    # Correlation with CAG length
    cag_corr_results, het_data = correlation_with_cag_length(expr_df)
    cag_corr_results.to_csv('results/gene_vs_cag_correlations.csv', index=False)
    print(f"\n✓ CAG correlation results saved to results/gene_vs_cag_correlations.csv")

    # Cross-gene correlations by CAG group
    cross_corr_results = cross_gene_correlation_by_cag(expr_df)
    cross_corr_results.to_csv('results/gene_correlations_by_cag_group.csv', index=False)
    print(f"\n✓ Cross-gene correlations saved to results/gene_correlations_by_cag_group.csv")

    # Create visualizations
    plot_cag_correlations(expr_df, het_data)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nKey findings:")
    print("-" * 80)

    # Summarize key findings
    for _, row in cag_corr_results.iterrows():
        sig = "✓" if row['pearson_p'] < 0.05 else "✗"
        print(f"{sig} {row['gene']} vs CAG repeats: r = {row['pearson_r']:.3f}, p = {row['pearson_p']:.2e}")

    print("\nOutput files:")
    print("  - results/gene_vs_cag_correlations.csv")
    print("  - results/gene_correlations_by_cag_group.csv")
    print("  - figures/cag_repeat_expression.png")
    print("  - figures/cag_gene_correlations.png")
    print("  - figures/gene_vs_cag_scatter.png")

if __name__ == '__main__':
    main()
