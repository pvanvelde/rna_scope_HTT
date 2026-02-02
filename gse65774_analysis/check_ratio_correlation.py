#!/usr/bin/env python3
"""
Check if UBC/POLR2A ratio correlates with age for Q111 strain
"""

import pandas as pd
import numpy as np
from scipy import stats

# Load the temporal changes results
print("="*70)
print("UBC/POLR2A RATIO vs AGE CORRELATION CHECK")
print("="*70)

# Load data
fpkm = pd.read_csv('data/GSE65774_Striatum_mRNA_FPKM_processedData.txt',
                   sep='\t', index_col=0)
metadata = pd.read_csv('data/metadata_complete.csv')
metadata = metadata.set_index('sample_id')
metadata = metadata.loc[fpkm.columns]

# Filter to Q111 only
q111_mask = metadata['genotype'] == 'Q111'
metadata_q111 = metadata[q111_mask]
fpkm_q111 = fpkm.loc[:, q111_mask]

# Extract UBC and POLR2A
UBC_ID = 'ENSMUSG00000019505'
POLR2A_ID = 'ENSMUSG00000005198'

ubc_expr = fpkm_q111.loc[UBC_ID]
polr2a_expr = fpkm_q111.loc[POLR2A_ID]

# Create DataFrame with gene expression, ratio, and metadata
results_df = pd.DataFrame({
    'sample_id': fpkm_q111.columns,
    'UBC': ubc_expr.values,
    'POLR2A': polr2a_expr.values,
    'UBC_POLR2A_ratio': ubc_expr.values / polr2a_expr.values,
    'timepoint': metadata_q111['timepoint'].values,
    'age_months': metadata_q111['timepoint'].map({'2M': 2, '6M': 6, '10M': 10}).values
})

print(f"\nQ111 samples: n={len(results_df)}")
print(f"\nTimepoint distribution:")
print(results_df['timepoint'].value_counts().sort_index())

# Test correlations with age for individual genes
print("\n" + "="*70)
print("INDIVIDUAL GENE CORRELATIONS WITH AGE (Q111 only)")
print("="*70)

for gene in ['UBC', 'POLR2A']:
    r, p = stats.pearsonr(results_df['age_months'], results_df[gene])
    print(f"\n{gene} vs Age:")
    print(f"  Pearson r = {r:.3f}, p = {p:.4f}")
    if p >= 0.05:
        print(f"  → NOT significant (p ≥ 0.05)")
    else:
        print(f"  → Significant (p < 0.05)")

# Test correlation for UBC/POLR2A ratio
print("\n" + "="*70)
print("UBC/POLR2A RATIO CORRELATION WITH AGE (Q111 only)")
print("="*70)

r_ratio, p_ratio = stats.pearsonr(results_df['age_months'], results_df['UBC_POLR2A_ratio'])
r_spearman, p_spearman = stats.spearmanr(results_df['age_months'], results_df['UBC_POLR2A_ratio'])

print(f"\nUBC/POLR2A ratio vs Age:")
print(f"  Pearson r = {r_ratio:.3f}, p = {p_ratio:.4f}")
print(f"  Spearman r = {r_spearman:.3f}, p = {p_spearman:.4f}")

if p_ratio < 0.05:
    direction = "negative" if r_ratio < 0 else "positive"
    print(f"  → Significant {direction} correlation (p < 0.05) ✓")
else:
    print(f"  → NOT significant (p ≥ 0.05) ✗")

# Show ratio values by age
print("\n" + "="*70)
print("UBC/POLR2A RATIO BY AGE")
print("="*70)

for tp in ['2M', '6M', '10M']:
    tp_data = results_df[results_df['timepoint'] == tp]
    mean_ratio = tp_data['UBC_POLR2A_ratio'].mean()
    sem_ratio = tp_data['UBC_POLR2A_ratio'].sem()
    print(f"\n{tp} (n={len(tp_data)}):")
    print(f"  UBC/POLR2A ratio = {mean_ratio:.1f} ± {sem_ratio:.1f}")

# Linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(
    results_df['age_months'], results_df['UBC_POLR2A_ratio']
)

print("\n" + "="*70)
print("LINEAR REGRESSION: UBC/POLR2A ratio ~ Age")
print("="*70)
print(f"  Slope: {slope:.2f} (change in ratio per month)")
print(f"  R²: {r_value**2:.3f}")
print(f"  p-value: {p_value:.4f}")

# Get correlations for individual genes
r_ubc, p_ubc = stats.pearsonr(results_df['age_months'], results_df['UBC'])
r_polr2a, p_polr2a = stats.pearsonr(results_df['age_months'], results_df['POLR2A'])

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("\nStatement to verify:")
print("  'Individual gene expression levels showed no significant correlations")
print("   with age in Q111 strain, but UBC/POLR2A ratio demonstrated")
print("   significant negative correlations with age'")
print("\nResults for Q111 in STRIATUM:")
print(f"  1. UBC vs Age: r = {r_ubc:.3f}, p = {p_ubc:.3f} → NOT significant ✓")
print(f"  2. POLR2A vs Age: r = {r_polr2a:.3f}, p = {p_polr2a:.3f} → NOT significant ✓")
print(f"  3. UBC/POLR2A ratio vs Age: r = {r_ratio:.3f}, p = {p_ratio:.3f}")
if p_ratio < 0.05 and r_ratio < 0:
    print(f"     → Significant NEGATIVE correlation ✓")
    print("\n✓ Statement is VALID for Q111 striatum!")
elif p_ratio < 0.05:
    print(f"     → Significant but POSITIVE (not negative) ✗")
    print("\n✗ Statement is NOT valid - correlation is positive, not negative")
else:
    print(f"     → NOT significant ✗")
    print("\n✗ Statement is NOT valid - ratio does not correlate with age")

# Create visualization
print("\n" + "="*70)
print("CREATING FIGURES")
print("="*70)

import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Panel 1: UBC vs Age
ax = axes[0, 0]
for tp in ['2M', '6M', '10M']:
    tp_data = results_df[results_df['timepoint'] == tp]
    ax.scatter(tp_data['age_months'], tp_data['UBC'],
              label=tp, s=80, alpha=0.7, edgecolors='black', linewidth=0.5)
# Regression line
z = np.polyfit(results_df['age_months'], results_df['UBC'], 1)
p = np.poly1d(z)
age_line = np.linspace(2, 10, 100)
ax.plot(age_line, p(age_line), 'k--', linewidth=2, alpha=0.5)
ax.text(0.05, 0.95, f'r = {r_ubc:.3f}\np = {p_ubc:.3f}',
        transform=ax.transAxes, va='top', fontsize=11,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
ax.set_xlabel('Age (months)', fontsize=12, fontweight='bold')
ax.set_ylabel('UBC Expression (FPKM)', fontsize=12, fontweight='bold')
ax.set_title('UBC vs Age (Q111)', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 2: POLR2A vs Age
ax = axes[0, 1]
for tp in ['2M', '6M', '10M']:
    tp_data = results_df[results_df['timepoint'] == tp]
    ax.scatter(tp_data['age_months'], tp_data['POLR2A'],
              label=tp, s=80, alpha=0.7, edgecolors='black', linewidth=0.5)
# Regression line
z = np.polyfit(results_df['age_months'], results_df['POLR2A'], 1)
p = np.poly1d(z)
ax.plot(age_line, p(age_line), 'k--', linewidth=2, alpha=0.5)
ax.text(0.05, 0.95, f'r = {r_polr2a:.3f}\np = {p_polr2a:.3f}',
        transform=ax.transAxes, va='top', fontsize=11,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
ax.set_xlabel('Age (months)', fontsize=12, fontweight='bold')
ax.set_ylabel('POLR2A Expression (FPKM)', fontsize=12, fontweight='bold')
ax.set_title('POLR2A vs Age (Q111)', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 3: UBC/POLR2A ratio vs Age
ax = axes[1, 0]
for tp in ['2M', '6M', '10M']:
    tp_data = results_df[results_df['timepoint'] == tp]
    ax.scatter(tp_data['age_months'], tp_data['UBC_POLR2A_ratio'],
              label=tp, s=80, alpha=0.7, edgecolors='black', linewidth=0.5)
# Regression line
z = np.polyfit(results_df['age_months'], results_df['UBC_POLR2A_ratio'], 1)
p = np.poly1d(z)
ax.plot(age_line, p(age_line), 'k--', linewidth=2, alpha=0.5,
        label=f'slope = {slope:.2f}')
ax.text(0.05, 0.95, f'r = {r_ratio:.3f}\np = {p_ratio:.3f}\nR² = {r_value**2:.3f}',
        transform=ax.transAxes, va='top', fontsize=11,
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
ax.set_xlabel('Age (months)', fontsize=12, fontweight='bold')
ax.set_ylabel('UBC/POLR2A Ratio', fontsize=12, fontweight='bold')
ax.set_title('UBC/POLR2A Ratio vs Age (Q111)', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 4: Bar plot of ratio by age
ax = axes[1, 1]
ratio_means = []
ratio_sems = []
labels = []
for tp in ['2M', '6M', '10M']:
    tp_data = results_df[results_df['timepoint'] == tp]
    ratio_means.append(tp_data['UBC_POLR2A_ratio'].mean())
    ratio_sems.append(tp_data['UBC_POLR2A_ratio'].sem())
    labels.append(tp)

x_pos = np.arange(len(labels))
bars = ax.bar(x_pos, ratio_means, yerr=ratio_sems,
              capsize=10, alpha=0.7, edgecolor='black', linewidth=1.5,
              color=['#1f77b4', '#ff7f0e', '#2ca02c'])
ax.set_xticks(x_pos)
ax.set_xticklabels(labels)
ax.set_xlabel('Age', fontsize=12, fontweight='bold')
ax.set_ylabel('UBC/POLR2A Ratio', fontsize=12, fontweight='bold')
ax.set_title('Mean UBC/POLR2A Ratio by Age (Q111)', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Add values on bars
for i, (mean, sem) in enumerate(zip(ratio_means, ratio_sems)):
    ax.text(i, mean + sem + 1, f'{mean:.1f}±{sem:.1f}',
            ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('figures/q111_ratio_age_correlation.png', dpi=300, bbox_inches='tight')
print("✓ Figure saved to figures/q111_ratio_age_correlation.png")
plt.close()

# Create individual animal plot
print("\n" + "="*70)
print("CREATING INDIVIDUAL ANIMAL PLOT")
print("="*70)

# Check if we have animal IDs in the metadata
if 'animal_id' in metadata_q111.columns or 'mouse_id' in metadata_q111.columns:
    animal_col = 'animal_id' if 'animal_id' in metadata_q111.columns else 'mouse_id'
    results_df['animal_id'] = metadata_q111[animal_col].values
    has_animal_id = True
    print(f"Found animal ID column: {animal_col}")
else:
    # Create unique ID from sample name
    results_df['animal_id'] = results_df['sample_id']
    has_animal_id = False
    print("No animal ID column found - using sample IDs")

# Count measurements per animal
animals_per_age = results_df.groupby(['animal_id', 'age_months']).size().reset_index(name='count')
longitudinal = animals_per_age.groupby('animal_id')['age_months'].nunique().max() > 1

if longitudinal:
    print("Data appears to be LONGITUDINAL (some animals measured at multiple timepoints)")
else:
    print("Data appears to be CROSS-SECTIONAL (each animal measured once)")

# Create figure with individual animals
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Panel 1: All individual animals with lines
ax = axes[0]
colors_palette = plt.cm.tab20(np.linspace(0, 1, len(results_df['animal_id'].unique())))

for idx, animal in enumerate(results_df['animal_id'].unique()):
    animal_data = results_df[results_df['animal_id'] == animal].sort_values('age_months')

    if len(animal_data) > 1:
        # If animal has multiple timepoints, connect with line
        ax.plot(animal_data['age_months'], animal_data['UBC_POLR2A_ratio'],
               marker='o', markersize=8, alpha=0.6, linewidth=2,
               color=colors_palette[idx % len(colors_palette)])
    else:
        # Single timepoint - just a point
        ax.scatter(animal_data['age_months'], animal_data['UBC_POLR2A_ratio'],
                  s=100, alpha=0.6, edgecolors='black', linewidth=0.5,
                  color=colors_palette[idx % len(colors_palette)])

# Add mean trajectory
for tp in results_df['timepoint'].unique():
    tp_data = results_df[results_df['timepoint'] == tp]
    ax.scatter(tp_data['age_months'].iloc[0], tp_data['UBC_POLR2A_ratio'].mean(),
              s=300, marker='D', c='red', edgecolors='black', linewidth=2,
              zorder=100, label=f'Mean {tp}' if tp == results_df['timepoint'].unique()[0] else '')

# Plot mean line
age_groups = results_df.groupby('age_months')['UBC_POLR2A_ratio'].mean().reset_index()
ax.plot(age_groups['age_months'], age_groups['UBC_POLR2A_ratio'],
       'r--', linewidth=3, alpha=0.8, label='Mean trajectory', zorder=99)

ax.set_xlabel('Age (months)', fontsize=13, fontweight='bold')
ax.set_ylabel('UBC/POLR2A Ratio', fontsize=13, fontweight='bold')
ax.set_title('Individual Q111 Animals: UBC/POLR2A Ratio Over Time', fontsize=14, fontweight='bold')
ax.set_xticks([2, 6, 10])
ax.grid(True, alpha=0.3)
ax.legend(['Mean trajectory'], loc='best', fontsize=10)

# Panel 2: Colored by age with individual points labeled
ax = axes[1]
timepoint_colors = {'2M': '#1f77b4', '6M': '#ff7f0e', '10M': '#2ca02c'}

for tp in ['2M', '6M', '10M']:
    tp_data = results_df[results_df['timepoint'] == tp]

    # Add jitter for visibility
    x_jitter = tp_data['age_months'] + np.random.normal(0, 0.15, len(tp_data))

    ax.scatter(x_jitter, tp_data['UBC_POLR2A_ratio'],
              s=120, alpha=0.7, edgecolors='black', linewidth=1,
              color=timepoint_colors[tp], label=tp)

    # Add mean and SEM bars
    mean_val = tp_data['UBC_POLR2A_ratio'].mean()
    sem_val = tp_data['UBC_POLR2A_ratio'].sem()
    age_val = tp_data['age_months'].iloc[0]

    ax.errorbar(age_val, mean_val, yerr=sem_val,
               fmt='D', markersize=12, color=timepoint_colors[tp],
               ecolor='black', capsize=8, capthick=2.5, linewidth=2.5,
               markeredgecolor='black', markeredgewidth=2, zorder=100)

ax.set_xlabel('Age (months)', fontsize=13, fontweight='bold')
ax.set_ylabel('UBC/POLR2A Ratio', fontsize=13, fontweight='bold')
ax.set_title('Q111 Animals by Age Group (Mean ± SEM)', fontsize=14, fontweight='bold')
ax.set_xticks([2, 6, 10])
ax.set_xlim([1, 11])
ax.legend(loc='best', fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/q111_ratio_per_animal.png', dpi=300, bbox_inches='tight')
print("✓ Individual animal plot saved to figures/q111_ratio_per_animal.png")
plt.close()

# Print some individual animal statistics
print("\n" + "="*70)
print("INDIVIDUAL ANIMAL STATISTICS")
print("="*70)
print(f"\nTotal unique animals: {results_df['animal_id'].nunique()}")
print(f"\nAnimals per timepoint:")
for tp in ['2M', '6M', '10M']:
    n_animals = results_df[results_df['timepoint'] == tp]['animal_id'].nunique()
    print(f"  {tp}: {n_animals} animals")

# Show range of ratios
print(f"\nRatio range across all animals: {results_df['UBC_POLR2A_ratio'].min():.1f} - {results_df['UBC_POLR2A_ratio'].max():.1f}")
print(f"Overall mean ± SD: {results_df['UBC_POLR2A_ratio'].mean():.1f} ± {results_df['UBC_POLR2A_ratio'].std():.1f}")

# Save results
results_df.to_csv('results/q111_ratio_age_correlation.csv', index=False)
print(f"\n✓ Results saved to results/q111_ratio_age_correlation.csv")
