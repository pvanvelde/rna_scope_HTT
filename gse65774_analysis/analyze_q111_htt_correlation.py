#!/usr/bin/env python3
"""
Focused analysis of Q111 mice: HTT, UBC, and POLR2 gene expression correlations
Testing hypothesis: Higher mHTT expression correlates with higher UBC and POLR2 expression
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
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['figure.dpi'] = 100

# Target genes
GENES_OF_INTEREST = {
    'mHTT': 'ENSMUSG00000029189',      # Huntingtin
    'UBC': 'ENSMUSG00000019505',        # Ubiquitin C
    'POLR2A': 'ENSMUSG00000005198',     # RNA Polymerase II subunit A
    'POLR2B': 'ENSMUSG00000005640',     # RNA Polymerase II subunit B
}

def load_data():
    """Load FPKM data and metadata"""
    print("Loading data...")
    fpkm = pd.read_csv('data/GSE65774_Striatum_mRNA_FPKM_processedData.txt',
                       sep='\t', index_col=0)
    metadata = pd.read_csv('data/metadata_complete.csv')

    # Match metadata to data columns
    metadata = metadata.set_index('sample_id')
    metadata = metadata.loc[fpkm.columns]

    print(f"✓ Loaded {fpkm.shape[0]} genes across {fpkm.shape[1]} samples")
    return fpkm, metadata

def filter_q111_samples(fpkm, metadata):
    """Filter to Q111 mice only"""
    print("\n" + "="*80)
    print("FILTERING TO Q111 MICE")
    print("="*80)

    # Q111 samples (both HET and any controls)
    q111_mask = metadata['genotype'] == 'Q111'
    q111_samples = metadata[q111_mask].index

    # Also get WT samples for comparison
    wt_mask = metadata['genotype'] == 'WT'
    wt_samples = metadata[wt_mask].index

    print(f"\nQ111 samples: {len(q111_samples)}")
    print(f"  - 2 months: {((metadata.loc[q111_samples, 'timepoint'] == '2M').sum())}")
    print(f"  - 6 months: {((metadata.loc[q111_samples, 'timepoint'] == '6M').sum())}")
    print(f"  - 10 months: {((metadata.loc[q111_samples, 'timepoint'] == '10M').sum())}")

    print(f"\nWT samples (for comparison): {len(wt_samples)}")

    # Extract expression data
    q111_fpkm = fpkm[q111_samples]
    q111_metadata = metadata.loc[q111_samples]

    wt_fpkm = fpkm[wt_samples]
    wt_metadata = metadata.loc[wt_samples]

    return q111_fpkm, q111_metadata, wt_fpkm, wt_metadata

def extract_target_genes(fpkm, metadata, gene_dict):
    """Extract expression data for target genes"""
    print("\n" + "="*80)
    print("TARGET GENE EXPRESSION")
    print("="*80)

    gene_expr = {}
    for gene_name, ensembl_id in gene_dict.items():
        if ensembl_id in fpkm.index:
            expr = fpkm.loc[ensembl_id]
            gene_expr[gene_name] = expr
            print(f"\n{gene_name} ({ensembl_id}):")
            print(f"  Mean FPKM: {expr.mean():.2f}")
            print(f"  Median FPKM: {expr.median():.2f}")
            print(f"  Range: [{expr.min():.2f}, {expr.max():.2f}]")
        else:
            print(f"\n✗ {gene_name} ({ensembl_id}): NOT FOUND in dataset")

    # Create dataframe
    expr_df = pd.DataFrame(gene_expr)
    expr_df = expr_df.merge(metadata, left_index=True, right_index=True, how='left')

    return expr_df

def correlation_analysis(expr_df, gene_pairs):
    """Analyze correlations between gene pairs"""
    print("\n" + "="*80)
    print("CORRELATION ANALYSIS")
    print("="*80)

    results = []

    for gene1, gene2 in gene_pairs:
        if gene1 in expr_df.columns and gene2 in expr_df.columns:
            # Remove any missing values
            data = expr_df[[gene1, gene2]].dropna()

            # Pearson correlation
            pearson_r, pearson_p = stats.pearsonr(data[gene1], data[gene2])

            # Spearman correlation (rank-based, more robust)
            spearman_r, spearman_p = stats.spearmanr(data[gene1], data[gene2])

            print(f"\n{gene1} vs {gene2}:")
            print(f"  Pearson r = {pearson_r:.4f}, p-value = {pearson_p:.2e}")
            print(f"  Spearman ρ = {spearman_r:.4f}, p-value = {spearman_p:.2e}")

            if pearson_p < 0.05:
                print(f"  ✓ Significant correlation (p < 0.05)")
            else:
                print(f"  ✗ Not significant (p >= 0.05)")

            results.append({
                'gene1': gene1,
                'gene2': gene2,
                'pearson_r': pearson_r,
                'pearson_p': pearson_p,
                'spearman_r': spearman_r,
                'spearman_p': spearman_p,
                'n_samples': len(data)
            })

    results_df = pd.DataFrame(results)
    return results_df

def plot_correlations(expr_df, gene_pairs, output_prefix):
    """Create scatter plots showing gene correlations"""
    n_pairs = len(gene_pairs)
    n_cols = min(3, n_pairs)
    n_rows = (n_pairs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    if n_pairs == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_rows > 1 else axes

    for idx, (gene1, gene2) in enumerate(gene_pairs):
        ax = axes[idx]

        if gene1 not in expr_df.columns or gene2 not in expr_df.columns:
            ax.text(0.5, 0.5, f'{gene1} or {gene2}\nnot found',
                   ha='center', va='center', transform=ax.transAxes)
            continue

        # Color by timepoint
        timepoint_colors = {'2M': 'lightblue', '6M': 'orange', '10M': 'darkred'}

        for tp in ['2M', '6M', '10M']:
            mask = expr_df['timepoint'] == tp
            if mask.sum() > 0:
                ax.scatter(expr_df.loc[mask, gene1],
                          expr_df.loc[mask, gene2],
                          c=timepoint_colors[tp],
                          label=tp,
                          s=80,
                          alpha=0.7,
                          edgecolors='black',
                          linewidth=0.5)

        # Add regression line
        data = expr_df[[gene1, gene2]].dropna()
        if len(data) > 2:
            z = np.polyfit(data[gene1], data[gene2], 1)
            p = np.poly1d(z)
            x_line = np.linspace(data[gene1].min(), data[gene1].max(), 100)
            ax.plot(x_line, p(x_line), 'r--', linewidth=2, alpha=0.8, label='Linear fit')

        # Calculate correlation
        pearson_r, pearson_p = stats.pearsonr(data[gene1], data[gene2])

        ax.set_xlabel(f'{gene1} Expression (FPKM)', fontsize=11, fontweight='bold')
        ax.set_ylabel(f'{gene2} Expression (FPKM)', fontsize=11, fontweight='bold')
        ax.set_title(f'{gene1} vs {gene2}\nPearson r = {pearson_r:.3f}, p = {pearson_p:.2e}',
                    fontsize=12, fontweight='bold')
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)

    # Hide empty subplots
    for idx in range(len(gene_pairs), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(f'figures/{output_prefix}_correlations.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Correlation plots saved to figures/{output_prefix}_correlations.png")
    plt.close()

def plot_expression_by_timepoint(expr_df, genes, output_prefix):
    """Plot gene expression across timepoints"""
    n_genes = len(genes)
    fig, axes = plt.subplots(1, n_genes, figsize=(5*n_genes, 5))
    if n_genes == 1:
        axes = [axes]

    for idx, gene in enumerate(genes):
        ax = axes[idx]

        if gene not in expr_df.columns:
            ax.text(0.5, 0.5, f'{gene}\nnot found',
                   ha='center', va='center', transform=ax.transAxes)
            continue

        # Prepare data
        timepoints = ['2M', '6M', '10M']
        data_to_plot = []

        for tp in timepoints:
            tp_data = expr_df[expr_df['timepoint'] == tp][gene].dropna()
            data_to_plot.append(tp_data)

        # Box plot
        bp = ax.boxplot(data_to_plot, labels=timepoints, patch_artist=True,
                       showmeans=True, meanline=True)

        # Color boxes
        colors = ['lightblue', 'orange', 'darkred']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        # Overlay individual points
        for i, tp_data in enumerate(data_to_plot):
            x = np.random.normal(i+1, 0.04, size=len(tp_data))
            ax.scatter(x, tp_data, alpha=0.5, s=40, c='black')

        ax.set_xlabel('Timepoint', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'{gene} Expression (FPKM)', fontsize=12, fontweight='bold')
        ax.set_title(f'{gene} Expression Over Time\n(Q111 mice)',
                    fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # Add sample sizes
        for i, tp_data in enumerate(data_to_plot):
            ax.text(i+1, ax.get_ylim()[0], f'n={len(tp_data)}',
                   ha='center', va='top', fontsize=9)

    plt.tight_layout()
    plt.savefig(f'figures/{output_prefix}_timepoint_expression.png', dpi=300, bbox_inches='tight')
    print(f"✓ Timepoint expression plots saved to figures/{output_prefix}_timepoint_expression.png")
    plt.close()

def compare_q111_vs_wt(q111_expr, wt_expr, genes):
    """Compare gene expression between Q111 and WT"""
    print("\n" + "="*80)
    print("Q111 vs WT COMPARISON")
    print("="*80)

    results = []

    for gene in genes:
        if gene not in q111_expr.columns or gene not in wt_expr.columns:
            print(f"\n✗ {gene}: not found in dataset")
            continue

        q111_vals = q111_expr[gene].dropna()
        wt_vals = wt_expr[gene].dropna()

        # T-test
        t_stat, t_pval = stats.ttest_ind(q111_vals, wt_vals)

        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(q111_vals)-1)*q111_vals.std()**2 +
                              (len(wt_vals)-1)*wt_vals.std()**2) /
                             (len(q111_vals) + len(wt_vals) - 2))
        cohens_d = (q111_vals.mean() - wt_vals.mean()) / pooled_std

        print(f"\n{gene}:")
        print(f"  Q111: mean = {q111_vals.mean():.2f}, SD = {q111_vals.std():.2f} (n={len(q111_vals)})")
        print(f"  WT:   mean = {wt_vals.mean():.2f}, SD = {wt_vals.std():.2f} (n={len(wt_vals)})")
        print(f"  Fold change: {q111_vals.mean() / wt_vals.mean():.3f}x")
        print(f"  t-test: t = {t_stat:.3f}, p-value = {t_pval:.2e}")
        print(f"  Cohen's d = {cohens_d:.3f}")

        if t_pval < 0.05:
            direction = "higher" if q111_vals.mean() > wt_vals.mean() else "lower"
            print(f"  ✓ Q111 has significantly {direction} expression (p < 0.05)")
        else:
            print(f"  ✗ No significant difference (p >= 0.05)")

        results.append({
            'gene': gene,
            'q111_mean': q111_vals.mean(),
            'q111_sd': q111_vals.std(),
            'wt_mean': wt_vals.mean(),
            'wt_sd': wt_vals.std(),
            'fold_change': q111_vals.mean() / wt_vals.mean(),
            't_statistic': t_stat,
            'p_value': t_pval,
            'cohens_d': cohens_d
        })

    results_df = pd.DataFrame(results)
    return results_df

def plot_q111_vs_wt_comparison(q111_expr, wt_expr, genes):
    """Create comparison plot between Q111 and WT"""
    n_genes = len(genes)
    fig, axes = plt.subplots(1, n_genes, figsize=(5*n_genes, 6))
    if n_genes == 1:
        axes = [axes]

    for idx, gene in enumerate(genes):
        ax = axes[idx]

        if gene not in q111_expr.columns or gene not in wt_expr.columns:
            continue

        q111_vals = q111_expr[gene].dropna()
        wt_vals = wt_expr[gene].dropna()

        # Box plot
        data_to_plot = [wt_vals, q111_vals]
        bp = ax.boxplot(data_to_plot, labels=['WT', 'Q111'], patch_artist=True,
                       showmeans=True, meanline=True, widths=0.6)

        # Color boxes
        bp['boxes'][0].set_facecolor('lightgreen')
        bp['boxes'][1].set_facecolor('salmon')
        for box in bp['boxes']:
            box.set_alpha(0.6)

        # Overlay individual points
        x1 = np.random.normal(1, 0.04, size=len(wt_vals))
        x2 = np.random.normal(2, 0.04, size=len(q111_vals))
        ax.scatter(x1, wt_vals, alpha=0.4, s=40, c='darkgreen')
        ax.scatter(x2, q111_vals, alpha=0.4, s=40, c='darkred')

        # Statistics
        t_stat, p_val = stats.ttest_ind(q111_vals, wt_vals)

        ax.set_ylabel(f'{gene} Expression (FPKM)', fontsize=12, fontweight='bold')
        ax.set_title(f'{gene}\nQ111 vs WT\np = {p_val:.2e}',
                    fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # Add significance indicator
        if p_val < 0.001:
            sig = '***'
        elif p_val < 0.01:
            sig = '**'
        elif p_val < 0.05:
            sig = '*'
        else:
            sig = 'ns'

        y_max = max(q111_vals.max(), wt_vals.max())
        y_pos = y_max * 1.1
        ax.plot([1, 2], [y_pos, y_pos], 'k-', linewidth=1.5)
        ax.text(1.5, y_pos*1.05, sig, ha='center', fontsize=14, fontweight='bold')

        # Add sample sizes
        ax.text(1, ax.get_ylim()[0], f'n={len(wt_vals)}',
               ha='center', va='top', fontsize=9)
        ax.text(2, ax.get_ylim()[0], f'n={len(q111_vals)}',
               ha='center', va='top', fontsize=9)

    plt.tight_layout()
    plt.savefig('figures/q111_vs_wt_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Q111 vs WT comparison saved to figures/q111_vs_wt_comparison.png")
    plt.close()

def main():
    """Main analysis workflow"""

    print("="*80)
    print("Q111 HTT-UBC-POLR2 CORRELATION ANALYSIS")
    print("="*80)

    # Load data
    fpkm, metadata = load_data()

    # Filter to Q111 and WT samples
    q111_fpkm, q111_metadata, wt_fpkm, wt_metadata = filter_q111_samples(fpkm, metadata)

    # Extract target gene expression for Q111
    q111_expr = extract_target_genes(q111_fpkm, q111_metadata, GENES_OF_INTEREST)

    # Extract target gene expression for WT
    wt_expr = extract_target_genes(wt_fpkm, wt_metadata, GENES_OF_INTEREST)

    # Define gene pairs to test
    gene_pairs = [
        ('mHTT', 'UBC'),
        ('mHTT', 'POLR2A'),
        ('UBC', 'POLR2A'),
    ]

    # Correlation analysis
    corr_results = correlation_analysis(q111_expr, gene_pairs)
    corr_results.to_csv('results/q111_gene_correlations.csv', index=False)
    print(f"\n✓ Correlation results saved to results/q111_gene_correlations.csv")

    # Plot correlations
    plot_correlations(q111_expr, gene_pairs, 'q111')

    # Plot expression by timepoint
    genes_to_plot = ['mHTT', 'UBC', 'POLR2A']
    plot_expression_by_timepoint(q111_expr, genes_to_plot, 'q111')

    # Compare Q111 vs WT
    comparison_results = compare_q111_vs_wt(q111_expr, wt_expr, genes_to_plot)
    comparison_results.to_csv('results/q111_vs_wt_comparison.csv', index=False)
    print(f"\n✓ Q111 vs WT comparison saved to results/q111_vs_wt_comparison.csv")

    # Plot Q111 vs WT comparison
    plot_q111_vs_wt_comparison(q111_expr, wt_expr, genes_to_plot)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nOutput files:")
    print("  - results/q111_gene_correlations.csv")
    print("  - results/q111_vs_wt_comparison.csv")
    print("  - figures/q111_correlations.png")
    print("  - figures/q111_timepoint_expression.png")
    print("  - figures/q111_vs_wt_comparison.png")

if __name__ == '__main__':
    main()
