#!/usr/bin/env python3
"""
RNA-seq analysis of GSE65774: Huntington's Disease mouse model
Transcriptome profiling across different CAG repeat lengths and timepoints
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 100

class GSE65774Analyzer:
    """Analyzer for GSE65774 Huntington's Disease RNA-seq data"""

    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.counts = None
        self.fpkm = None
        self.metadata = None
        self.log_fpkm = None

    def load_data(self):
        """Load count and FPKM data"""
        print("Loading data files...")
        self.counts = pd.read_csv(
            f'{self.data_dir}/GSE65774_Striatum_mRNA_counts_processedData.txt',
            sep='\t', index_col=0
        )
        self.fpkm = pd.read_csv(
            f'{self.data_dir}/GSE65774_Striatum_mRNA_FPKM_processedData.txt',
            sep='\t', index_col=0
        )
        print(f"✓ Loaded {self.counts.shape[0]} genes across {self.counts.shape[1]} samples")

    def parse_metadata(self):
        """Parse sample metadata from series matrix file"""
        print("\nParsing sample metadata...")

        # Read the series matrix file
        with open('GSE65774_series_matrix.txt', 'r') as f:
            lines = f.readlines()

        # Extract sample titles
        title_line = [l for l in lines if l.startswith('!Sample_title')][0]
        titles = title_line.split('\t')[1:]
        titles = [t.strip().strip('"') for t in titles]

        # Extract GEO accessions
        acc_line = [l for l in lines if l.startswith('!Sample_geo_accession')][0]
        accessions = acc_line.split('\t')[1:]
        accessions = [a.strip().strip('"') for a in accessions]

        # Parse metadata from titles
        metadata_list = []
        for title, acc in zip(titles, accessions):
            # Parse genotype (Q number or WT)
            if 'WT' in title or 'Wild Type' in title:
                genotype = 'WT'
                cag_repeats = 0
            else:
                # Extract Q number
                import re
                q_match = re.search(r'Q(\d+)', title)
                if q_match:
                    cag_repeats = int(q_match.group(1))
                    genotype = f'Q{cag_repeats}'
                else:
                    genotype = 'Unknown'
                    cag_repeats = -1

            # Extract timepoint
            if '2 month' in title:
                timepoint = '2M'
                timepoint_months = 2
            elif '6 month' in title:
                timepoint = '6M'
                timepoint_months = 6
            elif '10 month' in title:
                timepoint = '10M'
                timepoint_months = 10
            else:
                timepoint = 'Unknown'
                timepoint_months = -1

            # Extract sex
            if 'Female' in title:
                sex = 'F'
            elif 'Male' in title:
                sex = 'M'
            else:
                sex = 'Unknown'

            metadata_list.append({
                'geo_accession': acc,
                'title': title,
                'genotype': genotype,
                'cag_repeats': cag_repeats,
                'timepoint': timepoint,
                'timepoint_months': timepoint_months,
                'sex': sex
            })

        self.metadata = pd.DataFrame(metadata_list)

        # Match metadata to count matrix columns (data columns use different IDs)
        # For now, assume ordering matches
        if len(self.metadata) == len(self.counts.columns):
            self.metadata['sample_id'] = self.counts.columns

        # Save metadata
        self.metadata.to_csv(f'{self.data_dir}/metadata_complete.csv', index=False)
        print(f"✓ Parsed metadata for {len(self.metadata)} samples")
        print(f"  CAG repeat lengths: {sorted(self.metadata['cag_repeats'].unique())}")
        print(f"  Timepoints: {sorted(self.metadata['timepoint'].unique())}")

        return self.metadata

    def quality_control(self):
        """Perform QC checks on the data"""
        print("\n" + "="*80)
        print("QUALITY CONTROL ANALYSIS")
        print("="*80)

        # Library sizes
        lib_sizes = self.counts.sum(axis=0)
        print(f"\nLibrary sizes (total counts per sample):")
        print(f"  Mean: {lib_sizes.mean():,.0f}")
        print(f"  Median: {lib_sizes.median():,.0f}")
        print(f"  Range: [{lib_sizes.min():,.0f}, {lib_sizes.max():,.0f}]")

        # Gene detection
        genes_per_sample = (self.counts > 0).sum(axis=0)
        print(f"\nGenes detected per sample (count > 0):")
        print(f"  Mean: {genes_per_sample.mean():,.0f}")
        print(f"  Range: [{genes_per_sample.min():,.0f}, {genes_per_sample.max():,.0f}]")

        # Zero count genes
        zero_genes = (self.counts.sum(axis=1) == 0).sum()
        print(f"\nGenes with zero counts across all samples: {zero_genes}")
        print(f"Genes detected in at least one sample: {(self.counts.sum(axis=1) > 0).sum()}")

        # Create QC plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Library size distribution
        axes[0, 0].hist(lib_sizes / 1e6, bins=30, edgecolor='black')
        axes[0, 0].set_xlabel('Library Size (millions)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distribution of Library Sizes')
        axes[0, 0].axvline(lib_sizes.median() / 1e6, color='red', linestyle='--',
                          label=f'Median: {lib_sizes.median()/1e6:.1f}M')
        axes[0, 0].legend()

        # Genes detected per sample
        axes[0, 1].hist(genes_per_sample, bins=30, edgecolor='black')
        axes[0, 1].set_xlabel('Number of Genes Detected')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Genes Detected per Sample')

        # Mean expression distribution
        mean_expr = np.log10(self.fpkm.mean(axis=1) + 1)
        axes[1, 0].hist(mean_expr, bins=50, edgecolor='black')
        axes[1, 0].set_xlabel('log10(Mean FPKM + 1)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Distribution of Mean Gene Expression')

        # Sample correlation heatmap (random subset)
        np.random.seed(42)
        sample_subset = np.random.choice(self.counts.columns, min(20, len(self.counts.columns)), replace=False)
        corr = np.log2(self.counts[sample_subset] + 1).corr()
        sns.heatmap(corr, ax=axes[1, 1], cmap='coolwarm', center=0.95,
                    vmin=0.9, vmax=1.0, square=True, cbar_kws={'label': 'Pearson r'})
        axes[1, 1].set_title('Sample Correlation (random 20 samples)')

        plt.tight_layout()
        plt.savefig('figures/qc_overview.png', dpi=300, bbox_inches='tight')
        print("\n✓ QC plots saved to figures/qc_overview.png")
        plt.close()

    def pca_analysis(self):
        """Perform PCA on log-transformed FPKM data"""
        print("\n" + "="*80)
        print("PRINCIPAL COMPONENT ANALYSIS")
        print("="*80)

        # Filter low-expressed genes (mean FPKM > 1)
        expressed_genes = self.fpkm.mean(axis=1) > 1
        fpkm_filtered = self.fpkm[expressed_genes]
        print(f"\nFiltered to {expressed_genes.sum()} genes with mean FPKM > 1")

        # Log transform
        self.log_fpkm = np.log2(fpkm_filtered + 1)

        # Standardize
        scaler = StandardScaler()
        fpkm_scaled = scaler.fit_transform(self.log_fpkm.T)

        # PCA
        pca = PCA()
        pca_coords = pca.fit_transform(fpkm_scaled)

        # Create PCA dataframe
        pca_df = pd.DataFrame(
            pca_coords[:, :10],
            columns=[f'PC{i+1}' for i in range(10)],
            index=self.log_fpkm.columns
        )

        # Merge with metadata
        pca_df = pca_df.merge(self.metadata.set_index('sample_id'),
                             left_index=True, right_index=True, how='left')

        # Explained variance
        var_explained = pca.explained_variance_ratio_ * 100
        print(f"\nVariance explained by first 5 PCs:")
        for i in range(5):
            print(f"  PC{i+1}: {var_explained[i]:.2f}%")
        print(f"  Cumulative (PC1-5): {var_explained[:5].sum():.2f}%")

        # Create PCA plots
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))

        # PC1 vs PC2 colored by CAG repeats
        scatter1 = axes[0, 0].scatter(pca_df['PC1'], pca_df['PC2'],
                                      c=pca_df['cag_repeats'], cmap='viridis',
                                      s=80, alpha=0.7, edgecolors='black', linewidth=0.5)
        axes[0, 0].set_xlabel(f'PC1 ({var_explained[0]:.2f}% variance)')
        axes[0, 0].set_ylabel(f'PC2 ({var_explained[1]:.2f}% variance)')
        axes[0, 0].set_title('PCA colored by CAG Repeat Length')
        plt.colorbar(scatter1, ax=axes[0, 0], label='CAG Repeats')

        # PC1 vs PC2 colored by timepoint
        timepoint_colors = {'2M': 'lightblue', '6M': 'orange', '10M': 'darkred'}
        for tp in ['2M', '6M', '10M']:
            mask = pca_df['timepoint'] == tp
            axes[0, 1].scatter(pca_df.loc[mask, 'PC1'], pca_df.loc[mask, 'PC2'],
                              c=timepoint_colors[tp], label=tp, s=80, alpha=0.7,
                              edgecolors='black', linewidth=0.5)
        axes[0, 1].set_xlabel(f'PC1 ({var_explained[0]:.2f}% variance)')
        axes[0, 1].set_ylabel(f'PC2 ({var_explained[1]:.2f}% variance)')
        axes[0, 1].set_title('PCA colored by Timepoint')
        axes[0, 1].legend(title='Age')

        # PC1 vs PC2 colored by sex
        sex_colors = {'M': 'blue', 'F': 'red'}
        for s in ['M', 'F']:
            mask = pca_df['sex'] == s
            axes[1, 0].scatter(pca_df.loc[mask, 'PC1'], pca_df.loc[mask, 'PC2'],
                              c=sex_colors[s], label=s, s=80, alpha=0.7,
                              edgecolors='black', linewidth=0.5)
        axes[1, 0].set_xlabel(f'PC1 ({var_explained[0]:.2f}% variance)')
        axes[1, 0].set_ylabel(f'PC2 ({var_explained[1]:.2f}% variance)')
        axes[1, 0].set_title('PCA colored by Sex')
        axes[1, 0].legend(title='Sex')

        # Scree plot
        axes[1, 1].bar(range(1, 21), var_explained[:20], edgecolor='black')
        axes[1, 1].set_xlabel('Principal Component')
        axes[1, 1].set_ylabel('Variance Explained (%)')
        axes[1, 1].set_title('Scree Plot (first 20 PCs)')
        axes[1, 1].set_xticks(range(1, 21, 2))

        plt.tight_layout()
        plt.savefig('figures/pca_analysis.png', dpi=300, bbox_inches='tight')
        print("\n✓ PCA plots saved to figures/pca_analysis.png")
        plt.close()

        return pca_df

    def differential_expression(self, group1_mask, group2_mask, group1_name, group2_name):
        """
        Perform differential expression analysis between two groups
        using log2 fold change and t-test
        """
        print(f"\n{'='*80}")
        print(f"DIFFERENTIAL EXPRESSION: {group1_name} vs {group2_name}")
        print(f"{'='*80}")

        # Get sample IDs for each group
        group1_samples = self.metadata[group1_mask]['sample_id'].values
        group2_samples = self.metadata[group2_mask]['sample_id'].values

        print(f"\n{group1_name}: {len(group1_samples)} samples")
        print(f"{group2_name}: {len(group2_samples)} samples")

        # Filter to expressed genes
        expressed = (self.fpkm > 1).sum(axis=1) >= 3
        fpkm_filtered = self.fpkm[expressed]
        print(f"\nAnalyzing {expressed.sum()} genes (FPKM > 1 in >= 3 samples)")

        # Calculate mean expression
        group1_mean = fpkm_filtered[group1_samples].mean(axis=1)
        group2_mean = fpkm_filtered[group2_samples].mean(axis=1)

        # Log2 fold change
        log2fc = np.log2((group1_mean + 1) / (group2_mean + 1))

        # T-test
        pvalues = []
        for gene in fpkm_filtered.index:
            g1_vals = fpkm_filtered.loc[gene, group1_samples].values
            g2_vals = fpkm_filtered.loc[gene, group2_samples].values
            try:
                _, pval = stats.ttest_ind(g1_vals, g2_vals)
                pvalues.append(pval)
            except:
                pvalues.append(1.0)

        # Create results dataframe
        de_results = pd.DataFrame({
            'gene_id': fpkm_filtered.index,
            f'{group1_name}_mean': group1_mean.values,
            f'{group2_name}_mean': group2_mean.values,
            'log2_fold_change': log2fc.values,
            'pvalue': pvalues
        })

        # Adjusted p-values (Benjamini-Hochberg)
        from statsmodels.stats.multitest import multipletests
        _, padj, _, _ = multipletests(de_results['pvalue'], method='fdr_bh')
        de_results['padj'] = padj

        # Significance categories
        de_results['significant'] = (de_results['padj'] < 0.05) & (np.abs(de_results['log2_fold_change']) > 1)
        de_results['regulation'] = 'NS'
        de_results.loc[(de_results['log2_fold_change'] > 1) & (de_results['padj'] < 0.05), 'regulation'] = 'UP'
        de_results.loc[(de_results['log2_fold_change'] < -1) & (de_results['padj'] < 0.05), 'regulation'] = 'DOWN'

        # Sort by significance
        de_results = de_results.sort_values('padj')

        # Summary
        n_up = (de_results['regulation'] == 'UP').sum()
        n_down = (de_results['regulation'] == 'DOWN').sum()
        print(f"\nDifferentially expressed genes (|log2FC| > 1, padj < 0.05):")
        print(f"  Upregulated: {n_up}")
        print(f"  Downregulated: {n_down}")
        print(f"  Total: {n_up + n_down}")

        # Save results
        comparison_name = f"{group1_name}_vs_{group2_name}".replace(' ', '_')
        de_results.to_csv(f'results/de_{comparison_name}.csv', index=False)
        print(f"\n✓ DE results saved to results/de_{comparison_name}.csv")

        # Volcano plot
        self._plot_volcano(de_results, group1_name, group2_name)

        # Top genes table
        print(f"\nTop 10 upregulated genes:")
        print(de_results[de_results['regulation'] == 'UP'][['gene_id', 'log2_fold_change', 'padj']].head(10).to_string(index=False))

        print(f"\nTop 10 downregulated genes:")
        print(de_results[de_results['regulation'] == 'DOWN'][['gene_id', 'log2_fold_change', 'padj']].head(10).to_string(index=False))

        return de_results

    def _plot_volcano(self, de_results, group1_name, group2_name):
        """Create volcano plot"""
        plt.figure(figsize=(10, 8))

        # Plot non-significant genes
        ns = de_results[de_results['regulation'] == 'NS']
        plt.scatter(ns['log2_fold_change'], -np.log10(ns['padj']),
                   c='gray', s=10, alpha=0.3, label='NS')

        # Plot significant genes
        up = de_results[de_results['regulation'] == 'UP']
        down = de_results[de_results['regulation'] == 'DOWN']

        plt.scatter(up['log2_fold_change'], -np.log10(up['padj']),
                   c='red', s=20, alpha=0.6, label=f'UP ({len(up)})')
        plt.scatter(down['log2_fold_change'], -np.log10(down['padj']),
                   c='blue', s=20, alpha=0.6, label=f'DOWN ({len(down)})')

        # Threshold lines
        plt.axhline(-np.log10(0.05), color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        plt.axvline(1, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        plt.axvline(-1, color='black', linestyle='--', linewidth=0.8, alpha=0.5)

        plt.xlabel('log2 Fold Change', fontsize=12)
        plt.ylabel('-log10(adjusted p-value)', fontsize=12)
        plt.title(f'Volcano Plot: {group1_name} vs {group2_name}', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)

        comparison_name = f"{group1_name}_vs_{group2_name}".replace(' ', '_')
        plt.savefig(f'figures/volcano_{comparison_name}.png', dpi=300, bbox_inches='tight')
        print(f"✓ Volcano plot saved to figures/volcano_{comparison_name}.png")
        plt.close()


def main():
    """Main analysis workflow"""

    # Create output directories
    import os
    os.makedirs('figures', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    print("="*80)
    print("GSE65774 - Huntington's Disease RNA-seq Analysis")
    print("="*80)

    # Initialize analyzer
    analyzer = GSE65774Analyzer()

    # Load data
    analyzer.load_data()

    # Parse metadata
    metadata = analyzer.parse_metadata()

    # Quality control
    analyzer.quality_control()

    # PCA
    pca_df = analyzer.pca_analysis()

    # Differential expression analyses
    print("\n" + "="*80)
    print("DIFFERENTIAL EXPRESSION ANALYSES")
    print("="*80)

    # Example 1: Q175 vs WT at 6 months
    q175_mask = (metadata['genotype'] == 'Q175') & (metadata['timepoint'] == '6M')
    wt_mask = (metadata['genotype'] == 'WT') & (metadata['timepoint'] == '6M')
    de_q175_vs_wt_6m = analyzer.differential_expression(
        q175_mask, wt_mask, 'Q175_6M', 'WT_6M'
    )

    # Example 2: Q140 vs WT at 10 months
    q140_mask = (metadata['genotype'] == 'Q140') & (metadata['timepoint'] == '10M')
    wt_mask = (metadata['genotype'] == 'WT') & (metadata['timepoint'] == '10M')
    de_q140_vs_wt_10m = analyzer.differential_expression(
        q140_mask, wt_mask, 'Q140_10M', 'WT_10M'
    )

    # Example 3: Q111 vs Q20 at 6 months (comparing CAG repeat lengths)
    q111_mask = (metadata['genotype'] == 'Q111') & (metadata['timepoint'] == '6M')
    q20_mask = (metadata['genotype'] == 'Q20') & (metadata['timepoint'] == '6M')
    de_q111_vs_q20_6m = analyzer.differential_expression(
        q111_mask, q20_mask, 'Q111_6M', 'Q20_6M'
    )

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nOutput files:")
    print("  - figures/qc_overview.png")
    print("  - figures/pca_analysis.png")
    print("  - figures/volcano_*.png")
    print("  - results/de_*.csv")
    print("  - data/metadata_complete.csv")


if __name__ == '__main__':
    main()
