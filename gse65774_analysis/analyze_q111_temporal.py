#!/usr/bin/env python3
"""
Focused Q111 temporal analysis:
- Only Q111 and WT mice
- Analyze by age group (2M, 6M, 10M)
- Test correlations within each age
- Track temporal progression within genotypes
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

def load_and_filter_data():
    """Load data and filter to Q111 and WT only"""
    print("="*80)
    print("Q111 TEMPORAL ANALYSIS: Q111 vs WT OVER TIME")
    print("="*80)

    print("\nLoading data...")
    fpkm = pd.read_csv('data/GSE65774_Striatum_mRNA_FPKM_processedData.txt',
                       sep='\t', index_col=0)
    metadata = pd.read_csv('data/metadata_complete.csv')
    metadata = metadata.set_index('sample_id')
    metadata = metadata.loc[fpkm.columns]

    print(f"✓ Loaded {fpkm.shape[0]} genes across {fpkm.shape[1]} samples")

    # Filter to Q111 and WT only
    q111_wt_mask = metadata['genotype'].isin(['Q111', 'WT'])
    fpkm_filtered = fpkm.loc[:, q111_wt_mask]
    metadata_filtered = metadata[q111_wt_mask]

    print(f"\n" + "-"*80)
    print("FILTERED TO Q111 AND WT ONLY")
    print("-"*80)
    print(f"Total samples: {len(metadata_filtered)}")
    print(f"\nBy genotype:")
    print(metadata_filtered['genotype'].value_counts())
    print(f"\nBy timepoint:")
    print(metadata_filtered['timepoint'].value_counts())
    print(f"\nBreakdown by genotype and timepoint:")
    print(pd.crosstab(metadata_filtered['genotype'], metadata_filtered['timepoint']))

    return fpkm_filtered, metadata_filtered

def extract_gene_expression(fpkm, metadata):
    """Extract target gene expression"""
    print("\n" + "="*80)
    print("TARGET GENE EXPRESSION")
    print("="*80)

    gene_expr = {}
    for gene_name, ensembl_id in GENES_OF_INTEREST.items():
        if ensembl_id in fpkm.index:
            expr = fpkm.loc[ensembl_id]
            gene_expr[gene_name] = expr
            print(f"\n{gene_name} ({ensembl_id}):")
            print(f"  Overall mean: {expr.mean():.2f} ± {expr.std():.2f}")
            print(f"  Range: [{expr.min():.2f}, {expr.max():.2f}]")

    expr_df = pd.DataFrame(gene_expr)
    expr_df = expr_df.merge(metadata, left_index=True, right_index=True, how='left')

    return expr_df

def analyze_by_age_and_genotype(expr_df):
    """Detailed analysis by age group and genotype"""
    print("\n" + "="*80)
    print("EXPRESSION BY AGE AND GENOTYPE")
    print("="*80)

    timepoints = ['2M', '6M', '10M']
    genotypes = ['WT', 'Q111']
    genes = list(GENES_OF_INTEREST.keys())

    results = []

    for tp in timepoints:
        print(f"\n{'-'*80}")
        print(f"TIMEPOINT: {tp}")
        print(f"{'-'*80}")

        for gene in genes:
            wt_data = expr_df[(expr_df['timepoint'] == tp) &
                             (expr_df['genotype'] == 'WT')][gene]
            q111_data = expr_df[(expr_df['timepoint'] == tp) &
                               (expr_df['genotype'] == 'Q111')][gene]

            if len(wt_data) > 0 and len(q111_data) > 0:
                # T-test
                t_stat, p_val = stats.ttest_ind(q111_data, wt_data)

                # Effect size
                pooled_std = np.sqrt(((len(q111_data)-1)*q111_data.std()**2 +
                                     (len(wt_data)-1)*wt_data.std()**2) /
                                    (len(q111_data) + len(wt_data) - 2))
                cohens_d = (q111_data.mean() - wt_data.mean()) / pooled_std if pooled_std > 0 else 0

                fold_change = q111_data.mean() / wt_data.mean() if wt_data.mean() > 0 else 0

                print(f"\n  {gene}:")
                print(f"    WT (n={len(wt_data)}):   {wt_data.mean():.2f} ± {wt_data.std():.2f}")
                print(f"    Q111 (n={len(q111_data)}): {q111_data.mean():.2f} ± {q111_data.std():.2f}")
                print(f"    Fold change: {fold_change:.3f}x")
                print(f"    t-test: t={t_stat:.3f}, p={p_val:.4f}")
                print(f"    Cohen's d: {cohens_d:.3f}")

                if p_val < 0.05:
                    direction = "↑ higher" if q111_data.mean() > wt_data.mean() else "↓ lower"
                    print(f"    ✓ Q111 is {direction} (p < 0.05)")
                else:
                    print(f"    ✗ No significant difference")

                results.append({
                    'timepoint': tp,
                    'gene': gene,
                    'wt_mean': wt_data.mean(),
                    'wt_std': wt_data.std(),
                    'wt_n': len(wt_data),
                    'q111_mean': q111_data.mean(),
                    'q111_std': q111_data.std(),
                    'q111_n': len(q111_data),
                    'fold_change': fold_change,
                    't_statistic': t_stat,
                    'p_value': p_val,
                    'cohens_d': cohens_d
                })

    results_df = pd.DataFrame(results)
    return results_df

def correlations_by_age(expr_df):
    """Test gene-gene correlations within each age group"""
    print("\n" + "="*80)
    print("GENE-GENE CORRELATIONS BY AGE")
    print("="*80)

    timepoints = ['2M', '6M', '10M']
    genotypes = ['WT', 'Q111']
    gene_pairs = [('mHTT', 'UBC'), ('mHTT', 'POLR2A'), ('UBC', 'POLR2A')]

    results = []

    for tp in timepoints:
        print(f"\n{'-'*80}")
        print(f"TIMEPOINT: {tp}")
        print(f"{'-'*80}")

        for geno in genotypes:
            subset = expr_df[(expr_df['timepoint'] == tp) &
                           (expr_df['genotype'] == geno)]

            if len(subset) < 4:
                print(f"\n  {geno}: n={len(subset)} (too few samples)")
                continue

            print(f"\n  {geno} (n={len(subset)}):")

            for gene1, gene2 in gene_pairs:
                if gene1 in subset.columns and gene2 in subset.columns:
                    data = subset[[gene1, gene2]].dropna()

                    if len(data) >= 4:
                        pearson_r, pearson_p = stats.pearsonr(data[gene1], data[gene2])
                        spearman_r, spearman_p = stats.spearmanr(data[gene1], data[gene2])

                        print(f"    {gene1} vs {gene2}:")
                        print(f"      Pearson r={pearson_r:.3f}, p={pearson_p:.4f}")

                        if pearson_p < 0.05:
                            print(f"      ✓ Significant")
                        else:
                            print(f"      ✗ Not significant")

                        results.append({
                            'timepoint': tp,
                            'genotype': geno,
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

def temporal_changes_within_genotype(expr_df):
    """Test if gene expression changes over time within each genotype"""
    print("\n" + "="*80)
    print("TEMPORAL CHANGES WITHIN GENOTYPES")
    print("="*80)

    genotypes = ['WT', 'Q111']
    genes = list(GENES_OF_INTEREST.keys())
    timepoints = ['2M', '6M', '10M']

    results = []

    for geno in genotypes:
        print(f"\n{'-'*80}")
        print(f"GENOTYPE: {geno}")
        print(f"{'-'*80}")

        geno_data = expr_df[expr_df['genotype'] == geno]

        for gene in genes:
            print(f"\n  {gene}:")

            # Get data for each timepoint
            data_2m = geno_data[geno_data['timepoint'] == '2M'][gene].dropna()
            data_6m = geno_data[geno_data['timepoint'] == '6M'][gene].dropna()
            data_10m = geno_data[geno_data['timepoint'] == '10M'][gene].dropna()

            print(f"    2M (n={len(data_2m)}): {data_2m.mean():.2f} ± {data_2m.std():.2f}")
            print(f"    6M (n={len(data_6m)}): {data_6m.mean():.2f} ± {data_6m.std():.2f}")
            print(f"    10M (n={len(data_10m)}): {data_10m.mean():.2f} ± {data_10m.std():.2f}")

            # ANOVA to test if there's any difference across timepoints
            if len(data_2m) > 0 and len(data_6m) > 0 and len(data_10m) > 0:
                f_stat, p_anova = stats.f_oneway(data_2m, data_6m, data_10m)
                print(f"    ANOVA: F={f_stat:.3f}, p={p_anova:.4f}")

                if p_anova < 0.05:
                    print(f"    ✓ Significant change over time")
                else:
                    print(f"    ✗ No significant temporal change")

                # Correlation with time (using numeric months)
                time_data = []
                expr_data = []
                for tp, data in [('2M', data_2m), ('6M', data_6m), ('10M', data_10m)]:
                    months = int(tp[:-1])
                    time_data.extend([months] * len(data))
                    expr_data.extend(data.values)

                if len(time_data) > 2:
                    corr_r, corr_p = stats.pearsonr(time_data, expr_data)
                    print(f"    Correlation with age: r={corr_r:.3f}, p={corr_p:.4f}")

                    results.append({
                        'genotype': geno,
                        'gene': gene,
                        'mean_2m': data_2m.mean(),
                        'mean_6m': data_6m.mean(),
                        'mean_10m': data_10m.mean(),
                        'anova_f': f_stat,
                        'anova_p': p_anova,
                        'age_corr_r': corr_r,
                        'age_corr_p': corr_p
                    })

    results_df = pd.DataFrame(results)
    return results_df

def plot_temporal_trajectories(expr_df):
    """Plot gene expression trajectories over time"""
    genes = list(GENES_OF_INTEREST.keys())
    genotypes = ['WT', 'Q111']
    timepoints = ['2M', '6M', '10M']
    time_numeric = {'2M': 2, '6M': 6, '10M': 10}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    colors = {'WT': 'green', 'Q111': 'red'}

    for idx, gene in enumerate(genes):
        ax = axes[idx]

        for geno in genotypes:
            means = []
            sems = []

            for tp in timepoints:
                data = expr_df[(expr_df['genotype'] == geno) &
                              (expr_df['timepoint'] == tp)][gene]
                means.append(data.mean())
                sems.append(data.sem())

            x = [time_numeric[tp] for tp in timepoints]

            # Plot line with error bars
            ax.errorbar(x, means, yerr=sems,
                       label=geno, color=colors[geno],
                       marker='o', markersize=10, linewidth=2.5,
                       capsize=5, capthick=2, alpha=0.8)

            # Plot individual points
            for tp in timepoints:
                data = expr_df[(expr_df['genotype'] == geno) &
                              (expr_df['timepoint'] == tp)][gene]
                x_jitter = time_numeric[tp] + np.random.normal(0, 0.1, len(data))
                ax.scatter(x_jitter, data, color=colors[geno],
                          alpha=0.3, s=40, zorder=1)

        ax.set_xlabel('Age (months)', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'{gene} Expression (FPKM)', fontsize=12, fontweight='bold')
        ax.set_title(f'{gene} Over Time', fontsize=13, fontweight='bold')
        ax.set_xticks([2, 6, 10])
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('figures/q111_temporal_trajectories.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Temporal trajectories saved to figures/q111_temporal_trajectories.png")
    plt.close()

def plot_correlations_by_age(expr_df):
    """Plot gene-gene correlations separately for each age"""
    timepoints = ['2M', '6M', '10M']
    gene_pairs = [('mHTT', 'UBC'), ('mHTT', 'POLR2A'), ('UBC', 'POLR2A')]

    fig, axes = plt.subplots(3, 3, figsize=(18, 16))

    for row_idx, (gene1, gene2) in enumerate(gene_pairs):
        for col_idx, tp in enumerate(timepoints):
            ax = axes[row_idx, col_idx]

            # Plot WT
            wt_data = expr_df[(expr_df['genotype'] == 'WT') &
                             (expr_df['timepoint'] == tp)]
            if len(wt_data) > 0:
                ax.scatter(wt_data[gene1], wt_data[gene2],
                          c='green', label='WT', s=80, alpha=0.6,
                          edgecolors='black', linewidth=0.5)

            # Plot Q111
            q111_data = expr_df[(expr_df['genotype'] == 'Q111') &
                               (expr_df['timepoint'] == tp)]
            if len(q111_data) > 0:
                ax.scatter(q111_data[gene1], q111_data[gene2],
                          c='red', label='Q111', s=80, alpha=0.6,
                          edgecolors='black', linewidth=0.5)

            # Combined regression line
            combined = expr_df[expr_df['timepoint'] == tp][[gene1, gene2]].dropna()
            if len(combined) > 2:
                z = np.polyfit(combined[gene1], combined[gene2], 1)
                p = np.poly1d(z)
                x_line = np.linspace(combined[gene1].min(), combined[gene1].max(), 100)
                ax.plot(x_line, p(x_line), 'k--', linewidth=2, alpha=0.5)

                # Calculate correlation for combined
                r, p_val = stats.pearsonr(combined[gene1], combined[gene2])

                ax.text(0.05, 0.95, f'r={r:.2f}\np={p_val:.3f}',
                       transform=ax.transAxes, va='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            ax.set_xlabel(f'{gene1} (FPKM)', fontsize=10, fontweight='bold')
            ax.set_ylabel(f'{gene2} (FPKM)', fontsize=10, fontweight='bold')
            ax.set_title(f'{tp}', fontsize=11, fontweight='bold')
            ax.legend(loc='lower right', fontsize=9)
            ax.grid(True, alpha=0.3)

    # Overall title
    fig.suptitle('Gene Correlations by Age (WT and Q111)',
                fontsize=16, fontweight='bold', y=1.00)

    plt.tight_layout()
    plt.savefig('figures/q111_correlations_by_age.png', dpi=300, bbox_inches='tight')
    print(f"✓ Age-specific correlations saved to figures/q111_correlations_by_age.png")
    plt.close()

def plot_heatmap_summary(temporal_df, comparison_df):
    """Create heatmap summaries of temporal changes and comparisons"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Heatmap 1: Q111 vs WT fold changes over time
    genes = ['mHTT', 'UBC', 'POLR2A']
    timepoints = ['2M', '6M', '10M']

    fc_matrix = np.zeros((len(genes), len(timepoints)))
    p_matrix = np.zeros((len(genes), len(timepoints)))

    for i, gene in enumerate(genes):
        for j, tp in enumerate(timepoints):
            row = comparison_df[(comparison_df['gene'] == gene) &
                               (comparison_df['timepoint'] == tp)]
            if len(row) > 0:
                fc_matrix[i, j] = row['fold_change'].values[0]
                p_matrix[i, j] = row['p_value'].values[0]

    sns.heatmap(fc_matrix, annot=True, fmt='.3f',
               xticklabels=timepoints, yticklabels=genes,
               cmap='RdBu_r', center=1.0, vmin=0.9, vmax=1.1,
               cbar_kws={'label': 'Q111/WT Fold Change'},
               ax=axes[0])

    # Add significance markers
    for i in range(len(genes)):
        for j in range(len(timepoints)):
            if p_matrix[i, j] < 0.05:
                axes[0].text(j + 0.5, i + 0.2, '*',
                           ha='center', va='center',
                           fontsize=20, fontweight='bold', color='black')

    axes[0].set_title('Q111/WT Fold Change by Age\n(* = p < 0.05)',
                     fontsize=13, fontweight='bold')

    # Heatmap 2: Temporal correlation with age (within genotype)
    age_corr_matrix = np.zeros((len(genes), 2))

    for i, gene in enumerate(genes):
        for j, geno in enumerate(['WT', 'Q111']):
            row = temporal_df[(temporal_df['gene'] == gene) &
                             (temporal_df['genotype'] == geno)]
            if len(row) > 0:
                age_corr_matrix[i, j] = row['age_corr_r'].values[0]

    sns.heatmap(age_corr_matrix, annot=True, fmt='.3f',
               xticklabels=['WT', 'Q111'], yticklabels=genes,
               cmap='RdBu_r', center=0.0, vmin=-0.5, vmax=0.5,
               cbar_kws={'label': 'Correlation with Age'},
               ax=axes[1])

    axes[1].set_title('Correlation with Age\n(within genotype)',
                     fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.savefig('figures/q111_temporal_heatmaps.png', dpi=300, bbox_inches='tight')
    print(f"✓ Summary heatmaps saved to figures/q111_temporal_heatmaps.png")
    plt.close()

def main():
    """Main analysis workflow"""

    # Load and filter data
    fpkm, metadata = load_and_filter_data()

    # Extract gene expression
    expr_df = extract_gene_expression(fpkm, metadata)

    # Analysis by age and genotype
    comparison_df = analyze_by_age_and_genotype(expr_df)
    comparison_df.to_csv('results/q111_wt_by_age_comparison.csv', index=False)
    print(f"\n✓ Age-specific comparisons saved to results/q111_wt_by_age_comparison.csv")

    # Correlations by age
    corr_by_age_df = correlations_by_age(expr_df)
    corr_by_age_df.to_csv('results/q111_correlations_by_age.csv', index=False)
    print(f"✓ Age-specific correlations saved to results/q111_correlations_by_age.csv")

    # Temporal changes within genotype
    temporal_df = temporal_changes_within_genotype(expr_df)
    temporal_df.to_csv('results/q111_temporal_changes.csv', index=False)
    print(f"✓ Temporal changes saved to results/q111_temporal_changes.csv")

    # Create visualizations
    plot_temporal_trajectories(expr_df)
    plot_correlations_by_age(expr_df)
    plot_heatmap_summary(temporal_df, comparison_df)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nOutput files:")
    print("  - results/q111_wt_by_age_comparison.csv")
    print("  - results/q111_correlations_by_age.csv")
    print("  - results/q111_temporal_changes.csv")
    print("  - figures/q111_temporal_trajectories.png")
    print("  - figures/q111_correlations_by_age.png")
    print("  - figures/q111_temporal_heatmaps.png")

if __name__ == '__main__':
    main()
