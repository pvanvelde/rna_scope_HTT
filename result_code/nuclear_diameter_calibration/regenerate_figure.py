#!/usr/bin/env python3
"""
Regenerate the nuclear diameter calibration figure without the ugly stats box.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from pathlib import Path

# Set up paths
SCRIPT_DIR = Path(__file__).parent
DATA_FILE = SCRIPT_DIR / 'nuclear_diameter_annotations_filtered.csv'
STATS_FILE = SCRIPT_DIR / 'nuclear_diameter_statistics.csv'
IMG_FILE = SCRIPT_DIR / 'img.png'
OUTPUT_PREFIX = SCRIPT_DIR / 'nuclear_diameter_figure'

# Load data
df = pd.read_csv(DATA_FILE)
stats = pd.read_csv(STATS_FILE)

# Extract statistics
n = int(stats['n'].values[0])
diameter_mean = stats['diameter_mean'].values[0]
diameter_std = stats['diameter_std'].values[0]
diameter_median = stats['diameter_median'].values[0]
volume_mean = stats['volume_mean'].values[0]
volume_std = stats['volume_std'].values[0]
volume_median = stats['volume_median'].values[0]

print(f"Loaded {n} measurements")
print(f"Diameter: {diameter_mean:.2f} ± {diameter_std:.2f} µm (median: {diameter_median:.2f})")
print(f"Volume: {volume_mean:.1f} ± {volume_std:.1f} µm³ (median: {volume_median:.1f})")

# Create figure - 3 panels (removed panel D)
fig = plt.figure(figsize=(10, 6))

# Define grid: 2 rows - top row for image, bottom row for histograms
gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1.2, 1],
                      hspace=0.35, wspace=0.3)

# Panel A: Example image with annotations (spans top row)
ax_a = fig.add_subplot(gs[0, :])
if IMG_FILE.exists():
    img = imread(IMG_FILE)
    ax_a.imshow(img)
    ax_a.set_title('A) Example: DAPI MIP with diameter annotations', fontsize=10, loc='left')
else:
    ax_a.text(0.5, 0.5, 'Image not available', ha='center', va='center', transform=ax_a.transAxes)
ax_a.axis('off')

# Panel B: Diameter histogram
ax_b = fig.add_subplot(gs[1, 0])
diameters = df['diameter_um'].values
ax_b.hist(diameters, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
ax_b.axvline(diameter_mean, color='red', linestyle='-', linewidth=2, label=f'Mean: {diameter_mean:.1f} µm')
ax_b.axvline(diameter_median, color='orange', linestyle='--', linewidth=2, label=f'Median: {diameter_median:.1f} µm')
ax_b.set_xlabel('Nuclear Diameter (µm)', fontsize=10)
ax_b.set_ylabel('Count', fontsize=10)
ax_b.set_title('B) Diameter Distribution', fontsize=10, loc='left')
ax_b.legend(loc='upper right', fontsize=8, framealpha=0.9)
ax_b.text(0.02, 0.98, f'n = {n}', transform=ax_b.transAxes, fontsize=9,
          verticalalignment='top', fontweight='bold')

# Panel C: Volume histogram
ax_c = fig.add_subplot(gs[1, 1])
volumes = df['volume_um3'].values
ax_c.hist(volumes, bins=20, color='darkgreen', edgecolor='black', alpha=0.7)
ax_c.axvline(volume_mean, color='red', linestyle='-', linewidth=2, label=f'Mean: {volume_mean:.0f} µm³')
ax_c.axvline(volume_median, color='orange', linestyle='--', linewidth=2, label=f'Median: {volume_median:.0f} µm³')
ax_c.set_xlabel('Nuclear Volume (µm³)', fontsize=10)
ax_c.set_ylabel('Count', fontsize=10)
ax_c.set_title('C) Volume Distribution', fontsize=10, loc='left')
ax_c.legend(loc='upper right', fontsize=8, framealpha=0.9)

# Adjust layout
plt.tight_layout()

# Save figure
for ext in ['svg', 'pdf', 'png']:
    output_file = f"{OUTPUT_PREFIX}.{ext}"
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")

plt.close()
print("Done!")
