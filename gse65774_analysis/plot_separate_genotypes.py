#!/usr/bin/env python3
"""
Create separate visualizations for WT and Q111 mice
- Separate correlation plots for each genotype
- Side-by-side comparisons
- Temporal trajectories split by genotype
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
    """Load data filtered to Q111 and WT"""
    print("Loading Q111 and WT data...")

    fpkm = pd.read_csv('data/GSE65774_Striatum_mRNA_FPKM_processedData.txt',
                       sep='\t', index_col=0)
    metadata = pd.read_csv('data/metadata_complete.csv')
    metadata = metadata.set_index('sample_id')
    metadata = metadata.loc[fpkm.columns]

    # Filter to Q111 and WT only
    q111_wt_mask = metadata['genotype'].isin(['Q111', 'WT'])
    fpkm_filtered = fpkm.loc[:, q111_wt_mask]
    metadata_filtered = metadata[q111_wt_mask]

    # Extract gene expression
    gene_expr = {}
    for gene_name, ensembl_id in GENES_OF_INTEREST.items():
        if ensembl_id in fpkm_filtered.index:
            gene_expr[gene_name] = fpkm_filtered.loc[ensembl_id]

    expr_df = pd.DataFrame(gene_expr)
    expr_df = expr_df.merge(metadata_filtered, left_index=True, right_index=True, how='left')

    print(f"✓ Loaded {len(expr_df)} samples (WT: {(expr_df['genotype']=='WT').sum()}, Q111: {(expr_df['genotype']=='Q111').sum()})")

    return expr_df

def plot_wt_only_correlations(expr_df):
    """Create correlation plots for WT mice only"""
    print("\nCreating WT-only correlation plots...")

    wt_data = expr_df[expr_df['genotype'] == 'WT']
    timepoints = ['2M', '6M', '10M']
    gene_pairs = [('mHTT', 'UBC'), ('mHTT', 'POLR2A'), ('UBC', 'POLR2A')]

    # Create figure
    fig, axes = plt.subplots(3, 3, figsize=(18, 16))

    timepoint_colors = {'2M': '#a8dadc', '6M': '#457b9d', '10M': '#1d3557'}

    for row_idx, (gene1, gene2) in enumerate(gene_pairs):
        for col_idx, tp in enumerate(timepoints):
            ax = axes[row_idx, col_idx]

            tp_data = wt_data[wt_data['timepoint'] == tp]

            if len(tp_data) > 0:
                # Scatter plot
                ax.scatter(tp_data[gene1], tp_data[gene2],
                          c=timepoint_colors[tp], s=100, alpha=0.7,
                          edgecolors='black', linewidth=1)

                # Regression line
                if len(tp_data) > 2:
                    z = np.polyfit(tp_data[gene1], tp_data[gene2], 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(tp_data[gene1].min(), tp_data[gene1].max(), 100)
                    ax.plot(x_line, p(x_line), 'r--', linewidth=2.5, alpha=0.8)

                # Statistics
                if len(tp_data) >= 3:
                    r, p_val = stats.pearsonr(tp_data[gene1], tp_data[gene2])

                    # Significance stars
                    if p_val < 0.001:
                        sig = '***'
                    elif p_val < 0.01:
                        sig = '**'
                    elif p_val < 0.05:
                        sig = '*'
                    else:
                        sig = 'ns'

                    # Stats box
                    stats_text = f'r = {r:.3f}\np = {p_val:.4f}\n{sig}'
                    ax.text(0.05, 0.95, stats_text,
                           transform=ax.transAxes, va='top', ha='left',
                           bbox=dict(boxstyle='round', facecolor='white',
                                   edgecolor='black', alpha=0.8),
                           fontsize=11, fontweight='bold')

                    # Sample size
                    ax.text(0.95, 0.05, f'n = {len(tp_data)}',
                           transform=ax.transAxes, va='bottom', ha='right',
                           fontsize=10, style='italic')

            ax.set_xlabel(f'{gene1} (FPKM)', fontsize=12, fontweight='bold')
            ax.set_ylabel(f'{gene2} (FPKM)', fontsize=12, fontweight='bold')
            ax.set_title(f'{tp}', fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3)

    # Overall title
    fig.suptitle('Wild Type (WT) Gene Correlations by Age',
                fontsize=18, fontweight='bold', y=0.995)

    plt.tight_layout()
    plt.savefig('figures/WT_only_correlations.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: figures/WT_only_correlations.png")
    plt.close()

def plot_q111_only_correlations(expr_df):
    """Create correlation plots for Q111 mice only"""
    print("\nCreating Q111-only correlation plots...")

    q111_data = expr_df[expr_df['genotype'] == 'Q111']
    timepoints = ['2M', '6M', '10M']
    gene_pairs = [('mHTT', 'UBC'), ('mHTT', 'POLR2A'), ('UBC', 'POLR2A')]

    # Create figure
    fig, axes = plt.subplots(3, 3, figsize=(18, 16))

    timepoint_colors = {'2M': '#ffc8dd', '6M': '#e85d75', '10M': '#9d0208'}

    for row_idx, (gene1, gene2) in enumerate(gene_pairs):
        for col_idx, tp in enumerate(timepoints):
            ax = axes[row_idx, col_idx]

            tp_data = q111_data[q111_data['timepoint'] == tp]

            if len(tp_data) > 0:
                # Scatter plot
                ax.scatter(tp_data[gene1], tp_data[gene2],
                          c=timepoint_colors[tp], s=100, alpha=0.7,
                          edgecolors='black', linewidth=1)

                # Regression line
                if len(tp_data) > 2:
                    z = np.polyfit(tp_data[gene1], tp_data[gene2], 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(tp_data[gene1].min(), tp_data[gene1].max(), 100)
                    ax.plot(x_line, p(x_line), 'r--', linewidth=2.5, alpha=0.8)

                # Statistics
                if len(tp_data) >= 3:
                    r, p_val = stats.pearsonr(tp_data[gene1], tp_data[gene2])

                    # Significance stars
                    if p_val < 0.001:
                        sig = '***'
                    elif p_val < 0.01:
                        sig = '**'
                    elif p_val < 0.05:
                        sig = '*'
                    else:
                        sig = 'ns'

                    # Stats box
                    stats_text = f'r = {r:.3f}\np = {p_val:.4f}\n{sig}'
                    ax.text(0.05, 0.95, stats_text,
                           transform=ax.transAxes, va='top', ha='left',
                           bbox=dict(boxstyle='round', facecolor='white',
                                   edgecolor='black', alpha=0.8),
                           fontsize=11, fontweight='bold')

                    # Sample size
                    ax.text(0.95, 0.05, f'n = {len(tp_data)}',
                           transform=ax.transAxes, va='bottom', ha='right',
                           fontsize=10, style='italic')

            ax.set_xlabel(f'{gene1} (FPKM)', fontsize=12, fontweight='bold')
            ax.set_ylabel(f'{gene2} (FPKM)', fontsize=12, fontweight='bold')
            ax.set_title(f'{tp}', fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3)

    # Overall title
    fig.suptitle('Q111 Mutant Gene Correlations by Age',
                fontsize=18, fontweight='bold', y=0.995)

    plt.tight_layout()
    plt.savefig('figures/Q111_only_correlations.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: figures/Q111_only_correlations.png")
    plt.close()

def plot_separate_temporal_trajectories(expr_df):
    """Create separate trajectory plots for WT and Q111"""
    print("\nCreating separate temporal trajectory plots...")

    genes = list(GENES_OF_INTEREST.keys())
    timepoints = ['2M', '6M', '10M']
    time_numeric = {'2M': 2, '6M': 6, '10M': 10}

    # Create two separate figures - one for WT, one for Q111
    for genotype in ['WT', 'Q111']:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        geno_data = expr_df[expr_df['genotype'] == genotype]
        color = 'green' if genotype == 'WT' else 'red'

        for idx, gene in enumerate(genes):
            ax = axes[idx]

            means = []
            sems = []

            for tp in timepoints:
                data = geno_data[geno_data['timepoint'] == tp][gene]
                means.append(data.mean())
                sems.append(data.sem())

            x = [time_numeric[tp] for tp in timepoints]

            # Plot line with error bars
            ax.errorbar(x, means, yerr=sems,
                       color=color, marker='o', markersize=12,
                       linewidth=3, capsize=8, capthick=3, alpha=0.8,
                       label=f'{genotype} mean ± SEM')

            # Plot individual points
            for tp in timepoints:
                data = geno_data[geno_data['timepoint'] == tp][gene]
                x_jitter = time_numeric[tp] + np.random.normal(0, 0.15, len(data))
                ax.scatter(x_jitter, data, color=color,
                          alpha=0.4, s=60, zorder=1, edgecolors='black', linewidth=0.5)

            # Add statistics for temporal change
            time_data = []
            expr_data = []
            for tp in timepoints:
                tp_data = geno_data[geno_data['timepoint'] == tp][gene]
                months = int(tp[:-1])
                time_data.extend([months] * len(tp_data))
                expr_data.extend(tp_data.values)

            if len(time_data) > 2:
                corr_r, corr_p = stats.pearsonr(time_data, expr_data)

                # Add stats text
                stats_text = f'Age correlation:\nr = {corr_r:.3f}\np = {corr_p:.4f}'
                ax.text(0.05, 0.95, stats_text,
                       transform=ax.transAxes, va='top', ha='left',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
                       fontsize=10)

            ax.set_xlabel('Age (months)', fontsize=13, fontweight='bold')
            ax.set_ylabel(f'{gene} Expression (FPKM)', fontsize=13, fontweight='bold')
            ax.set_title(f'{gene}', fontsize=14, fontweight='bold')
            ax.set_xticks([2, 6, 10])
            ax.set_xlim([1, 11])
            ax.legend(loc='best', fontsize=11)
            ax.grid(True, alpha=0.3)

        fig.suptitle(f'{genotype} Expression Over Time',
                    fontsize=18, fontweight='bold')

        plt.tight_layout()
        plt.savefig(f'figures/{genotype}_temporal_trajectory.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: figures/{genotype}_temporal_trajectory.png")
        plt.close()

def plot_side_by_side_comparison(expr_df):
    """Create side-by-side WT vs Q111 for each gene pair at each age"""
    print("\nCreating side-by-side WT vs Q111 comparison plots...")

    gene_pairs = [('mHTT', 'UBC'), ('mHTT', 'POLR2A'), ('UBC', 'POLR2A')]
    timepoints = ['2M', '6M', '10M']

    for gene1, gene2 in gene_pairs:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        for col_idx, tp in enumerate(timepoints):
            # WT plot (top row)
            ax_wt = axes[0, col_idx]
            wt_data = expr_df[(expr_df['genotype'] == 'WT') &
                             (expr_df['timepoint'] == tp)]

            if len(wt_data) > 0:
                ax_wt.scatter(wt_data[gene1], wt_data[gene2],
                            c='green', s=100, alpha=0.6,
                            edgecolors='black', linewidth=1)

                if len(wt_data) > 2:
                    z = np.polyfit(wt_data[gene1], wt_data[gene2], 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(wt_data[gene1].min(), wt_data[gene1].max(), 100)
                    ax_wt.plot(x_line, p(x_line), 'k--', linewidth=2.5, alpha=0.8)

                    r, p_val = stats.pearsonr(wt_data[gene1], wt_data[gene2])
                    sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'

                    ax_wt.text(0.05, 0.95, f'WT - {tp}\nr = {r:.3f}\np = {p_val:.4f}\n{sig}',
                             transform=ax_wt.transAxes, va='top',
                             bbox=dict(boxstyle='round', facecolor='lightgreen',
                                     edgecolor='black', alpha=0.8),
                             fontsize=11, fontweight='bold')

                    ax_wt.text(0.95, 0.05, f'n = {len(wt_data)}',
                             transform=ax_wt.transAxes, va='bottom', ha='right',
                             fontsize=10, style='italic')

            ax_wt.set_ylabel(f'{gene2} (FPKM)', fontsize=12, fontweight='bold')
            ax_wt.grid(True, alpha=0.3)
            if col_idx == 1:
                ax_wt.set_title('Wild Type (WT)', fontsize=14, fontweight='bold')

            # Q111 plot (bottom row)
            ax_q111 = axes[1, col_idx]
            q111_data = expr_df[(expr_df['genotype'] == 'Q111') &
                               (expr_df['timepoint'] == tp)]

            if len(q111_data) > 0:
                ax_q111.scatter(q111_data[gene1], q111_data[gene2],
                              c='red', s=100, alpha=0.6,
                              edgecolors='black', linewidth=1)

                if len(q111_data) > 2:
                    z = np.polyfit(q111_data[gene1], q111_data[gene2], 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(q111_data[gene1].min(), q111_data[gene1].max(), 100)
                    ax_q111.plot(x_line, p(x_line), 'k--', linewidth=2.5, alpha=0.8)

                    r, p_val = stats.pearsonr(q111_data[gene1], q111_data[gene2])
                    sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'

                    ax_q111.text(0.05, 0.95, f'Q111 - {tp}\nr = {r:.3f}\np = {p_val:.4f}\n{sig}',
                               transform=ax_q111.transAxes, va='top',
                               bbox=dict(boxstyle='round', facecolor='lightcoral',
                                       edgecolor='black', alpha=0.8),
                               fontsize=11, fontweight='bold')

                    ax_q111.text(0.95, 0.05, f'n = {len(q111_data)}',
                               transform=ax_q111.transAxes, va='bottom', ha='right',
                               fontsize=10, style='italic')

            ax_q111.set_xlabel(f'{gene1} (FPKM)', fontsize=12, fontweight='bold')
            ax_q111.set_ylabel(f'{gene2} (FPKM)', fontsize=12, fontweight='bold')
            ax_q111.grid(True, alpha=0.3)
            if col_idx == 1:
                ax_q111.set_title('Q111 Mutant', fontsize=14, fontweight='bold')

        fig.suptitle(f'{gene1} vs {gene2} Correlations: WT vs Q111 by Age',
                    fontsize=18, fontweight='bold', y=0.995)

        plt.tight_layout()

        filename = f"{gene1}_vs_{gene2}_WT_vs_Q111.png".replace(' ', '_')
        plt.savefig(f'figures/{filename}', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: figures/{filename}")
        plt.close()

def main():
    """Main workflow"""

    print("="*80)
    print("CREATING SEPARATE WT AND Q111 VISUALIZATIONS")
    print("="*80)

    # Load data
    expr_df = load_data()

    # Create separate plots
    plot_wt_only_correlations(expr_df)
    plot_q111_only_correlations(expr_df)
    plot_separate_temporal_trajectories(expr_df)
    plot_side_by_side_comparison(expr_df)

    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  - figures/WT_only_correlations.png")
    print("  - figures/Q111_only_correlations.png")
    print("  - figures/WT_temporal_trajectory.png")
    print("  - figures/Q111_temporal_trajectory.png")
    print("  - figures/mHTT_vs_UBC_WT_vs_Q111.png")
    print("  - figures/mHTT_vs_POLR2A_WT_vs_Q111.png")
    print("  - figures/UBC_vs_POLR2A_WT_vs_Q111.png")

if __name__ == '__main__':
    main()
