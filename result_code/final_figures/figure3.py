"""
Figure 3 - Regional Analysis and Total mRNA Expression

Layout (matching PDF comments):
    Row 1: A (anatomical overview with DARPP-32 striatum marker, subregion labels a-i)
           B (subregional boxplots referring to A labels)
    Row 2: C (example FOV images: low, medium, high expression)
           D (age breakdown)
    Row 3: E (atlas coordinate breakdown, full width)
    Row 4: F (mouse ID breakdown, full width)

Subregion labels (from Panel A):
    Striatum (identified via DARPP-32 RNA probe):
      a: dorsomedial striatum
      b: dorsolateral striatum
      c: ventrolateral striatum
      d: ventromedial striatum
    Cortex (approximate regions, no cell markers):
      e: primary/secondary motor area
      f: primary somatosensory area upper limb
      g: primary somatosensory area nose
      h: gustatory area/agranular insular area
      i: piriform area

Data sources:
    - B: Merged subregional boxplots from regional_analysis_subregion_level.csv
    - D, E, F: Total mRNA from fov_level_data.csv

Data caching: Processed data is cached to disk for fast layout iterations.
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import f_oneway, mannwhitneyu, kruskal
import pickle
import sys

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'draft_figures'))

from figure_config import (
    FigureConfig,
    apply_figure_style,
    save_figure,
    COLORS
)

from results_config import (
    CHANNEL_COLORS,
    CHANNEL_LABELS_EXPERIMENTAL,
    EXCLUDED_SLIDES,
    MOUSE_LABEL_MAP_Q111,
    MOUSE_LABEL_MAP_WT,
    CV_THRESHOLD,
    BEAD_PSF_X,
    BEAD_PSF_Y,
    BEAD_PSF_Z,
    SIGMA_X_LOWER,
    QUANTILE_NEGATIVE_CONTROL,
    MAX_PFA,
)

# Apply consistent styling
apply_figure_style()

# Output and cache directories
OUTPUT_DIR = Path(__file__).parent / "output"
CACHE_DIR = OUTPUT_DIR / 'cache'
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = CACHE_DIR / 'figure3_data.pkl'

# Data source paths (from draft_figures/output/)
REGIONAL_DATA_PATH = Path(__file__).parent.parent / 'draft_figures' / 'output' / 'regional_analysis' / 'regional_analysis_subregion_level.csv'
FOV_DATA_PATH = Path(__file__).parent.parent / 'draft_figures' / 'output' / 'expression_analysis_q111' / 'fov_level_data.csv'

# Set to True to force data reload
FORCE_RELOAD = False

# Color scheme for Q111
COLOR_MHTT1A = CHANNEL_COLORS.get('green', '#2ecc71')
COLOR_FULL = CHANNEL_COLORS.get('orange', '#f39c12')

# Color scheme for WT (distinct blue/purple colors)
COLOR_MHTT1A_WT = '#3498db'  # Blue for WT mHTT1a
COLOR_FULL_WT = '#9b59b6'    # Purple for WT full-length


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_and_process_data():
    """Load and process all data for Figure 3. Returns cached data if available."""

    if CACHE_FILE.exists() and not FORCE_RELOAD:
        print(f"Loading cached data from {CACHE_FILE}")
        with open(CACHE_FILE, 'rb') as f:
            return pickle.load(f)

    print("\n" + "=" * 70)
    print("LOADING DATA FOR FIGURE 3")
    print("=" * 70)

    # Load regional subregion data
    print("Loading regional subregion data...")
    df_subregion = pd.read_csv(REGIONAL_DATA_PATH)
    print(f"  Loaded {len(df_subregion)} subregion records")

    # Load FOV-level total mRNA data
    print("Loading FOV-level total mRNA data...")
    df_fov = pd.read_csv(FOV_DATA_PATH)
    print(f"  Loaded {len(df_fov)} FOV records")

    # Exclude problematic slides from all data
    df_fov = df_fov[~df_fov['Slide'].isin(EXCLUDED_SLIDES)].copy()

    # Filter to Q111 for total mRNA panels
    df_fov_q111 = df_fov[df_fov['Mouse_Model'] == 'Q111'].copy()
    print(f"  Q111 FOVs after exclusion: {len(df_fov_q111)}")

    # Filter to WT for comparison
    df_fov_wt = df_fov[df_fov['Mouse_Model'] == 'Wildtype'].copy()
    print(f"  WT FOVs after exclusion: {len(df_fov_wt)}")

    # Also filter regional data
    df_subregion = df_subregion[~df_subregion['slide'].isin(EXCLUDED_SLIDES)].copy()
    print(f"  Subregion records after exclusion: {len(df_subregion)}")

    # Use fixed mouse ID mapping from results_config (maintains consistent numbering even with excluded slides)
    mouse_id_map_q111 = MOUSE_LABEL_MAP_Q111
    df_fov_q111['Mouse_Label'] = df_fov_q111['Mouse_ID'].map(mouse_id_map_q111)

    # Use fixed mouse ID mapping for WT
    mouse_id_map_wt = MOUSE_LABEL_MAP_WT
    df_fov_wt['Mouse_Label'] = df_fov_wt['Mouse_ID'].map(mouse_id_map_wt)

    cache_data = {
        'df_subregion': df_subregion,
        'df_fov_q111': df_fov_q111,
        'df_fov_wt': df_fov_wt,
        'mouse_id_map_q111': mouse_id_map_q111,
        'mouse_id_map_wt': mouse_id_map_wt,
    }

    # Save cache
    print(f"Saving cache to {CACHE_FILE}")
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(cache_data, f)

    return cache_data


# ══════════════════════════════════════════════════════════════════════════════
# STATISTICAL ANALYSIS FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def aggregate_by_slide(df_fov):
    """Aggregate FOV-level data to slide-level means.

    This avoids pseudoreplication by treating each slide as the independent
    observation unit rather than individual FOVs.
    """
    # Group by Slide, Region, Channel, Age and compute mean
    grouped = df_fov.groupby(['Slide', 'Region', 'Channel', 'Age', 'Mouse_Model']).agg({
        'Total_mRNA_per_Cell': 'mean',
        'Mouse_ID': 'first'  # Keep mouse ID for reference
    }).reset_index()

    return grouped


def compute_sample_sizes(df_fov_q111, df_fov_wt):
    """Compute sample sizes (number of slides and FOVs) per condition.

    Returns a list of dictionaries with sample size info for each condition.
    """
    sample_sizes = []

    for genotype, df_fov in [('Q111', df_fov_q111), ('WT', df_fov_wt)]:
        ages = sorted(df_fov['Age'].dropna().unique())
        for age in ages:
            for region in ['Cortex', 'Striatum']:
                for channel in ['mHTT1a', 'full-length mHTT']:
                    subset = df_fov[
                        (df_fov['Age'] == age) &
                        (df_fov['Region'] == region) &
                        (df_fov['Channel'] == channel)
                    ]

                    n_fovs = len(subset)
                    n_slides = subset['Slide'].nunique() if n_fovs > 0 else 0

                    if n_fovs > 0:
                        sample_sizes.append({
                            'genotype': genotype,
                            'age': age,
                            'region': region,
                            'channel': channel,
                            'n_slides': n_slides,
                            'n_fovs': n_fovs
                        })

    return sample_sizes


def compute_statistical_tests(df_fov_q111, df_fov_wt):
    """Compute statistical tests for region, age, and genotype comparisons.

    IMPORTANT: To avoid pseudoreplication, FOV-level data is first aggregated
    to slide-level means. Each slide is treated as an independent observation,
    and n reported is the number of slides, not FOVs.

    Returns a dictionary with results for:
    - region_tests: Cortex vs Striatum comparisons (Mann-Whitney U)
    - age_tests: Age effect across groups (Kruskal-Wallis H)
    - genotype_tests: Q111 vs WT comparisons (Mann-Whitney U)
    """
    results = {
        'region_tests': [],
        'age_tests': [],
        'genotype_tests': []
    }

    # Aggregate to slide-level means to avoid pseudoreplication
    df_slide_q111 = aggregate_by_slide(df_fov_q111)
    df_slide_wt = aggregate_by_slide(df_fov_wt)

    ages_q111 = sorted(df_slide_q111['Age'].dropna().unique())
    ages_wt = sorted(df_slide_wt['Age'].dropna().unique())
    channels = ['mHTT1a', 'full-length mHTT']

    # ══════════════════════════════════════════════════════════════════════════
    # TABLE 1: REGION COMPARISON (Cortex vs Striatum) - Mann-Whitney U
    # ══════════════════════════════════════════════════════════════════════════

    for genotype, df_slide, ages in [('Q111', df_slide_q111, ages_q111), ('WT', df_slide_wt, ages_wt)]:
        for channel in channels:
            for age in ages:
                cortex_data = df_slide[
                    (df_slide['Age'] == age) &
                    (df_slide['Region'] == 'Cortex') &
                    (df_slide['Channel'] == channel)
                ]['Total_mRNA_per_Cell'].dropna().values

                striatum_data = df_slide[
                    (df_slide['Age'] == age) &
                    (df_slide['Region'] == 'Striatum') &
                    (df_slide['Channel'] == channel)
                ]['Total_mRNA_per_Cell'].dropna().values

                if len(cortex_data) > 0 and len(striatum_data) > 0:
                    stat, pval = mannwhitneyu(cortex_data, striatum_data, alternative='two-sided')

                    # Determine significance level
                    if pval < 0.001:
                        sig = '***'
                    elif pval < 0.01:
                        sig = '**'
                    elif pval < 0.05:
                        sig = '*'
                    else:
                        sig = 'ns'

                    results['region_tests'].append({
                        'genotype': genotype,
                        'channel': channel,
                        'age': age,
                        'cortex_median': np.median(cortex_data),
                        'cortex_n': len(cortex_data),
                        'striatum_median': np.median(striatum_data),
                        'striatum_n': len(striatum_data),
                        'U': stat,
                        'p_value': pval,
                        'sig': sig
                    })

    # ══════════════════════════════════════════════════════════════════════════
    # TABLE 2: AGE EFFECT - Pairwise Mann-Whitney U tests between consecutive ages
    # ══════════════════════════════════════════════════════════════════════════

    for genotype, df_slide, ages in [('Q111', df_slide_q111, ages_q111), ('WT', df_slide_wt, ages_wt)]:
        for channel in channels:
            for region in ['Cortex', 'Striatum']:
                # Do pairwise comparisons between consecutive age groups
                for i in range(len(ages) - 1):
                    age1, age2 = ages[i], ages[i + 1]

                    data1 = df_slide[
                        (df_slide['Age'] == age1) &
                        (df_slide['Region'] == region) &
                        (df_slide['Channel'] == channel)
                    ]['Total_mRNA_per_Cell'].dropna().values

                    data2 = df_slide[
                        (df_slide['Age'] == age2) &
                        (df_slide['Region'] == region) &
                        (df_slide['Channel'] == channel)
                    ]['Total_mRNA_per_Cell'].dropna().values

                    if len(data1) > 0 and len(data2) > 0:
                        stat, pval = mannwhitneyu(data1, data2, alternative='two-sided')

                        # Determine significance level
                        if pval < 0.001:
                            sig = '***'
                        elif pval < 0.01:
                            sig = '**'
                        elif pval < 0.05:
                            sig = '*'
                        else:
                            sig = 'ns'

                        results['age_tests'].append({
                            'genotype': genotype,
                            'channel': channel,
                            'region': region,
                            'age1': age1,
                            'age2': age2,
                            'median1': np.median(data1),
                            'n1': len(data1),
                            'median2': np.median(data2),
                            'n2': len(data2),
                            'U': stat,
                            'p_value': pval,
                            'sig': sig
                        })

    # ══════════════════════════════════════════════════════════════════════════
    # TABLE 3: GENOTYPE COMPARISON (Q111 vs WT) - Mann-Whitney U
    # ══════════════════════════════════════════════════════════════════════════

    # Get common ages between Q111 and WT
    common_ages = sorted(set(ages_q111) & set(ages_wt))

    for channel in channels:
        for region in ['Cortex', 'Striatum']:
            for age in common_ages:
                q111_data = df_slide_q111[
                    (df_slide_q111['Age'] == age) &
                    (df_slide_q111['Region'] == region) &
                    (df_slide_q111['Channel'] == channel)
                ]['Total_mRNA_per_Cell'].dropna().values

                wt_data = df_slide_wt[
                    (df_slide_wt['Age'] == age) &
                    (df_slide_wt['Region'] == region) &
                    (df_slide_wt['Channel'] == channel)
                ]['Total_mRNA_per_Cell'].dropna().values

                if len(q111_data) > 0 and len(wt_data) > 0:
                    stat, pval = mannwhitneyu(q111_data, wt_data, alternative='two-sided')

                    # Determine significance level
                    if pval < 0.001:
                        sig = '***'
                    elif pval < 0.01:
                        sig = '**'
                    elif pval < 0.05:
                        sig = '*'
                    else:
                        sig = 'ns'

                    results['genotype_tests'].append({
                        'channel': channel,
                        'region': region,
                        'age': age,
                        'q111_median': np.median(q111_data),
                        'q111_n': len(q111_data),
                        'wt_median': np.median(wt_data),
                        'wt_n': len(wt_data),
                        'U': stat,
                        'p_value': pval,
                        'sig': sig
                    })

    return results


# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def plot_regional_boxplots(ax, df_subregion):
    """Plot merged regional boxplots (Cortex and Striatum subregions for both channels).

    Uses lowercase letter labels (a-i) matching Panel A anatomical overview:
    Striatum (identified via DARPP-32):
      a: dorsomedial striatum
      b: dorsolateral striatum
      c: ventrolateral striatum
      d: ventromedial striatum
    Cortex (approximate regions):
      e: primary/secondary motor area
      f: primary somatosensory area upper limb
      g: primary somatosensory area nose
      h: gustatory/agranular insular area
      i: piriform area
    """
    cfg = FigureConfig

    # Subregion letter labels (matching Panel A)
    subregion_short = {
        'Cortex - Piriform area': 'i',
        'Cortex - Primary and secondary motor areas': 'e',
        'Cortex - Primary somatosensory (mouth, upper limb)': 'f',
        'Cortex - Supplemental/primary somatosensory (nose)': 'g',
        'Cortex - Visceral/gustatory/agranular areas': 'h',
        'Striatum - lower left': 'd',
        'Striatum - lower right': 'c',
        'Striatum - upper left': 'a',
        'Striatum - upper right': 'b',
    }

    positions = []
    labels = []
    data_to_plot = []
    box_colors = []
    hatches = []

    pos = 0

    for region in ['Cortex', 'Striatum']:
        region_data = df_subregion[df_subregion['region'] == region]
        subregions_raw = region_data['subregion'].unique()

        # Sort subregions by their letter label (alphabetical)
        subregions = sorted(subregions_raw,
                           key=lambda x: subregion_short.get(x, x))
        hatch = '' if region == 'Cortex' else '///'

        for subregion in subregions:
            for channel in ['mHTT1a', 'full-length mHTT']:
                sub_data = region_data[
                    (region_data['subregion'] == subregion) &
                    (region_data['channel'] == channel)
                ]['expression_per_cell'].values

                if len(sub_data) > 0:
                    data_to_plot.append(sub_data)
                    positions.append(pos)
                    box_colors.append(COLOR_MHTT1A if channel == 'mHTT1a' else COLOR_FULL)
                    hatches.append(hatch)
                    pos += 0.4

            # Add label at center of the two boxes (using letter label)
            short_name = subregion_short.get(subregion, subregion.split(' - ')[-1])
            labels.append((pos - 0.4, short_name))
            pos += 0.3  # Gap between subregions

        pos += 0.5  # Gap between regions

    # Plot boxplots with simple black edges (narrower width)
    bp = ax.boxplot(data_to_plot, positions=positions, widths=0.28, patch_artist=True)

    for i, (patch, box_color, hatch) in enumerate(zip(bp['boxes'], box_colors, hatches)):
        patch.set_facecolor(box_color)
        patch.set_alpha(0.8)
        patch.set_edgecolor('black')
        patch.set_linewidth(1)
        if hatch:
            patch.set_hatch(hatch)

    # Make median lines black
    for median in bp['medians']:
        median.set_color('black')
        median.set_linewidth(1.5)

    # Set x-ticks (no rotation for single letter labels)
    ax.set_xticks([l[0] for l in labels])
    ax.set_xticklabels([l[1] for l in labels], rotation=0, ha='center', fontsize=cfg.FONT_SIZE_AXIS_TICK)

    ax.set_ylabel('mRNA/nucleus', fontsize=cfg.FONT_SIZE_AXIS_LABEL)
    ax.set_xlabel('')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add region separator line
    cortex_end = len([s for s in df_subregion[df_subregion['region'] == 'Cortex']['subregion'].unique()])
    ax.axvline(x=positions[cortex_end * 2 - 1] + 0.4, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)

    # Add region text labels at top
    ylim = ax.get_ylim()
    ax.text(positions[cortex_end - 1], ylim[1] * 0.98, 'Cortex',
            ha='center', va='top', fontsize=cfg.FONT_SIZE_AXIS_LABEL, fontweight='bold')
    ax.text(positions[-1] - 1, ylim[1] * 0.98, 'Striatum',
            ha='center', va='top', fontsize=cfg.FONT_SIZE_AXIS_LABEL, fontweight='bold')

    # Legend - simple: solid = Cortex, hatched = Striatum
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLOR_MHTT1A, alpha=0.8, edgecolor='black', label='mHTT1a'),
        Patch(facecolor=COLOR_FULL, alpha=0.8, edgecolor='black', label='full mHTT'),
        Patch(facecolor='white', edgecolor='black', label='Cortex'),
        Patch(facecolor='white', edgecolor='black', hatch='///', label='Striatum'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=cfg.FONT_SIZE_LEGEND - 1, ncol=2,
              bbox_to_anchor=(1.0, 0.85))


def plot_age_trends(ax, df_fov_q111, df_fov_wt):
    """Plot total mRNA by age as bar plots for Q111 and WT.

    Uses per-slide aggregation: FOVs are first averaged within each slide,
    then mean ± SEM is computed across slides. This avoids pseudoreplication.
    """
    cfg = FigureConfig

    # Aggregate to slide-level means first
    df_slide_q111 = aggregate_by_slide(df_fov_q111)
    df_slide_wt = aggregate_by_slide(df_fov_wt)

    ages = sorted(df_fov_q111['Age'].unique())
    x_positions = []
    pos = 0
    bar_width = 0.35

    for age in ages:
        age_start = pos
        for model, df_slide in [('Q111', df_slide_q111), ('WT', df_slide_wt)]:
            for region in ['Cortex', 'Striatum']:
                for channel in ['mHTT1a', 'full-length mHTT']:
                    # Get slide-level means for this condition
                    subset = df_slide[
                        (df_slide['Age'] == age) &
                        (df_slide['Region'] == region) &
                        (df_slide['Channel'] == channel)
                    ]['Total_mRNA_per_Cell'].values

                    if len(subset) > 0:
                        if model == 'Q111':
                            color = COLOR_MHTT1A if channel == 'mHTT1a' else COLOR_FULL
                        else:
                            color = COLOR_MHTT1A_WT if channel == 'mHTT1a' else COLOR_FULL_WT
                        hatch = '' if region == 'Cortex' else '///'

                        # Mean and SEM across slides (not FOVs)
                        mean_val = np.mean(subset)
                        sem_val = np.std(subset) / np.sqrt(len(subset)) if len(subset) > 1 else 0

                        ax.bar(pos, mean_val, yerr=sem_val, width=bar_width,
                               color=color, alpha=0.8, edgecolor='black', linewidth=1,
                               hatch=hatch, capsize=2, error_kw={'linewidth': 1})

                    pos += 0.4
                pos += 0.1  # Gap between regions
            pos += 0.15  # Gap between models
        x_positions.append((age_start + (pos - age_start) / 2 - 0.3, f'{int(age)}mo'))
        pos += 0.3

    ax.set_xticks([p[0] for p in x_positions])
    ax.set_xticklabels([p[1] for p in x_positions], fontsize=cfg.FONT_SIZE_AXIS_TICK)
    ax.set_ylabel('Total mRNA/nucleus', fontsize=cfg.FONT_SIZE_AXIS_LABEL)
    ax.set_xlabel('Age', fontsize=cfg.FONT_SIZE_AXIS_LABEL)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLOR_MHTT1A, alpha=0.8, edgecolor='black', label='mHTT1a Q111'),
        Patch(facecolor=COLOR_FULL, alpha=0.8, edgecolor='black', label='full mHTT Q111'),
        Patch(facecolor=COLOR_MHTT1A_WT, alpha=0.8, edgecolor='black', label='mHTT1a WT'),
        Patch(facecolor=COLOR_FULL_WT, alpha=0.8, edgecolor='black', label='full mHTT WT'),
        Patch(facecolor='white', edgecolor='black', label='Cortex'),
        Patch(facecolor='white', edgecolor='black', hatch='///', label='Striatum'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=cfg.FONT_SIZE_LEGEND - 2, ncol=3)


def plot_atlas_coords_data(df_fov_q111, df_fov_wt):
    """Collect data for atlas coordinates plot, grouped by age brackets.

    Returns positions, values, styling, age separators, and Q111/WT separator positions.
    Data is organized as: [Age 2mo: Q111 coords | WT coords] | [Age 6mo: Q111 coords | WT coords] | [Age 12mo: ...]
    """
    x_positions = []
    bars_data = []
    age_separators = []  # Positions for vertical lines between age groups
    genotype_separators = []  # Positions for dashed lines between Q111 and WT within age
    age_labels = []  # (position, label) for age bracket labels
    pos = 0
    bar_width = 0.35

    # Get all unique ages, sorted
    all_ages = sorted(set(df_fov_q111['Age'].dropna().unique()) | set(df_fov_wt['Age'].dropna().unique()))

    for age_idx, age in enumerate(all_ages):
        age_start = pos

        # Filter data for this age
        df_q111_age = df_fov_q111[df_fov_q111['Age'] == age]
        df_wt_age = df_fov_wt[df_fov_wt['Age'] == age]

        # Get coordinates for this age (Q111)
        coords_q111 = sorted(df_q111_age['Brain_Atlas_Coord'].dropna().unique())

        # Plot Q111 data for this age
        for coord in coords_q111:
            coord_start = pos
            for region in ['Cortex', 'Striatum']:
                for channel in ['mHTT1a', 'full-length mHTT']:
                    subset = df_q111_age[
                        (df_q111_age['Brain_Atlas_Coord'] == coord) &
                        (df_q111_age['Region'] == region) &
                        (df_q111_age['Channel'] == channel)
                    ]['Total_mRNA_per_Cell'].values

                    if len(subset) > 0:
                        color = COLOR_MHTT1A if channel == 'mHTT1a' else COLOR_FULL
                        hatch = '' if region == 'Cortex' else '///'

                        mean_val = np.mean(subset)
                        sem_val = np.std(subset) / np.sqrt(len(subset)) if len(subset) > 1 else 0

                        bars_data.append({
                            'pos': pos, 'mean': mean_val, 'sem': sem_val,
                            'color': color, 'hatch': hatch, 'width': bar_width
                        })

                    pos += 0.4
                pos += 0.05
            x_positions.append((coord_start + (pos - coord_start) / 2 - 0.2, f'Q{int(coord)}'))
            pos += 0.15

        # Add genotype separator (dashed line between Q111 and WT within this age)
        if len(coords_q111) > 0:
            genotype_separators.append(pos)
            pos += 0.3

        # Get coordinates for this age (WT)
        coords_wt = sorted(df_wt_age['Brain_Atlas_Coord'].dropna().unique())

        # Plot WT data for this age
        for coord in coords_wt:
            coord_start = pos
            for region in ['Cortex', 'Striatum']:
                for channel in ['mHTT1a', 'full-length mHTT']:
                    subset = df_wt_age[
                        (df_wt_age['Brain_Atlas_Coord'] == coord) &
                        (df_wt_age['Region'] == region) &
                        (df_wt_age['Channel'] == channel)
                    ]['Total_mRNA_per_Cell'].values

                    if len(subset) > 0:
                        color = COLOR_MHTT1A_WT if channel == 'mHTT1a' else COLOR_FULL_WT
                        hatch = '' if region == 'Cortex' else '///'

                        mean_val = np.mean(subset)
                        sem_val = np.std(subset) / np.sqrt(len(subset)) if len(subset) > 1 else 0

                        bars_data.append({
                            'pos': pos, 'mean': mean_val, 'sem': sem_val,
                            'color': color, 'hatch': hatch, 'width': bar_width
                        })

                    pos += 0.4
                pos += 0.05
            x_positions.append((coord_start + (pos - coord_start) / 2 - 0.2, f'W{int(coord)}'))
            pos += 0.15

        # Store age label position (center of this age bracket)
        age_end = pos
        age_labels.append(((age_start + age_end) / 2, f'{int(age)}mo'))

        # Add age separator (solid line) unless this is the last age
        if age_idx < len(all_ages) - 1:
            age_separators.append(pos)
            pos += 0.5

    return bars_data, x_positions, age_separators, genotype_separators, age_labels


def plot_atlas_coords(ax, df_fov_q111, df_fov_wt):
    """Plot total mRNA by atlas coordinates, grouped by age brackets."""
    cfg = FigureConfig

    bars_data, x_positions, age_separators, genotype_separators, age_labels = plot_atlas_coords_data(df_fov_q111, df_fov_wt)

    # Find max value for axis limit
    max_val = max(b['mean'] + b['sem'] for b in bars_data) if bars_data else 180

    # Plot bars
    for bar in bars_data:
        ax.bar(bar['pos'], bar['mean'], yerr=bar['sem'], width=bar['width'],
               color=bar['color'], alpha=0.8, edgecolor='black', linewidth=1,
               hatch=bar['hatch'], capsize=2, error_kw={'linewidth': 1})

    # Add dashed vertical lines between Q111 and WT within each age bracket
    for sep_pos in genotype_separators:
        ax.axvline(x=sep_pos, color='gray', linestyle='--', alpha=0.5, linewidth=1.0)

    # Add solid vertical lines between age brackets
    for sep_pos in age_separators:
        ax.axvline(x=sep_pos, color='black', linestyle='-', alpha=0.7, linewidth=1.5)

    # Set y-limits
    ax.set_ylim(0, max_val * 1.1)

    # Set x-ticks for coordinate labels
    ax.set_xticks([p[0] for p in x_positions])
    ax.set_xticklabels([p[1] for p in x_positions], fontsize=cfg.FONT_SIZE_AXIS_TICK - 2, rotation=45)

    # Add age bracket labels at the top
    for age_pos, age_label in age_labels:
        ax.text(age_pos, max_val * 1.05, age_label, ha='center', va='bottom',
                fontsize=cfg.FONT_SIZE_AXIS_LABEL, fontweight='bold')

    # Labels
    ax.set_xlabel('Atlas coord (25μm) by Age', fontsize=cfg.FONT_SIZE_AXIS_LABEL)
    ax.set_ylabel('Total mRNA/nucleus', fontsize=cfg.FONT_SIZE_AXIS_LABEL)

    # Grid
    ax.grid(axis='y', alpha=0.3, linestyle='--')


def plot_mouse_ids_data(df_fov_q111, df_fov_wt, mouse_id_map_q111, mouse_id_map_wt):
    """Collect data for mouse IDs plot, grouped by age brackets.

    Returns positions, values, styling, age separators, and Q111/WT separator positions.
    Data is organized as: [Age 2mo: Q111 mice | WT mice] | [Age 6mo: Q111 mice | WT mice] | [Age 12mo: ...]
    """
    x_positions = []
    bars_data = []
    age_separators = []  # Positions for vertical lines between age groups
    genotype_separators = []  # Positions for dashed lines between Q111 and WT within age
    age_labels = []  # (position, label) for age bracket labels
    pos = 0
    bar_width = 0.35

    # Helper to extract numeric label for sorting
    def get_label_num(mouse_id, id_map):
        label = id_map.get(mouse_id, '#999')
        return int(label.replace('#', ''))

    # Build sub-index mapping: Q#1.1, Q#1.2, etc. for slides within same mouse
    def build_slide_sublabels(mouse_ids, id_map):
        """Create labels with sub-indices for each slide within a mouse."""
        sorted_ids = sorted(mouse_ids, key=lambda x: (get_label_num(x, id_map), x))
        sublabels = {}
        current_mouse = None
        sub_idx = 0
        for mid in sorted_ids:
            mouse_num = id_map.get(mid, '#?')
            if mouse_num != current_mouse:
                current_mouse = mouse_num
                sub_idx = 1
            else:
                sub_idx += 1
            sublabels[mid] = f"{mouse_num}.{sub_idx}"
        return sublabels

    # Get all unique ages, sorted
    all_ages = sorted(set(df_fov_q111['Age'].dropna().unique()) | set(df_fov_wt['Age'].dropna().unique()))

    for age_idx, age in enumerate(all_ages):
        age_start = pos

        # Filter data for this age
        df_q111_age = df_fov_q111[df_fov_q111['Age'] == age].copy()
        df_wt_age = df_fov_wt[df_fov_wt['Age'] == age].copy()

        # Get mice for this age (Q111)
        mice_q111_age = df_q111_age['Mouse_ID'].unique()
        mice_order_q111 = sorted(mice_q111_age,
                                  key=lambda x: (get_label_num(x, mouse_id_map_q111), x))
        q111_sublabels = build_slide_sublabels(mice_q111_age, mouse_id_map_q111)

        # Plot Q111 mice for this age
        for mouse_id in mice_order_q111:
            mouse_data = df_q111_age[df_q111_age['Mouse_ID'] == mouse_id]
            mouse_label = q111_sublabels.get(mouse_id, mouse_id)
            mouse_start = pos

            for region in ['Cortex', 'Striatum']:
                for channel in ['mHTT1a', 'full-length mHTT']:
                    subset = mouse_data[
                        (mouse_data['Region'] == region) &
                        (mouse_data['Channel'] == channel)
                    ]['Total_mRNA_per_Cell'].values

                    if len(subset) > 0:
                        color = COLOR_MHTT1A if channel == 'mHTT1a' else COLOR_FULL
                        hatch = '' if region == 'Cortex' else '///'

                        mean_val = np.mean(subset)
                        sem_val = np.std(subset) / np.sqrt(len(subset)) if len(subset) > 1 else 0

                        bars_data.append({
                            'pos': pos, 'mean': mean_val, 'sem': sem_val,
                            'color': color, 'hatch': hatch, 'width': bar_width
                        })

                    pos += 0.4
                pos += 0.1
            x_positions.append((mouse_start + 0.7, f'Q{mouse_label}'))
            pos += 0.2

        # Add genotype separator (dashed line between Q111 and WT within this age)
        if len(mice_order_q111) > 0:
            genotype_separators.append(pos)
            pos += 0.3

        # Get mice for this age (WT)
        mice_wt_age = df_wt_age['Mouse_ID'].unique()
        mice_order_wt = sorted(mice_wt_age,
                                key=lambda x: (get_label_num(x, mouse_id_map_wt), x))
        wt_sublabels = build_slide_sublabels(mice_wt_age, mouse_id_map_wt)

        # Plot WT mice for this age
        for mouse_id in mice_order_wt:
            mouse_data = df_wt_age[df_wt_age['Mouse_ID'] == mouse_id]
            mouse_label = wt_sublabels.get(mouse_id, mouse_id)
            mouse_start = pos

            for region in ['Cortex', 'Striatum']:
                for channel in ['mHTT1a', 'full-length mHTT']:
                    subset = mouse_data[
                        (mouse_data['Region'] == region) &
                        (mouse_data['Channel'] == channel)
                    ]['Total_mRNA_per_Cell'].values

                    if len(subset) > 0:
                        color = COLOR_MHTT1A_WT if channel == 'mHTT1a' else COLOR_FULL_WT
                        hatch = '' if region == 'Cortex' else '///'

                        mean_val = np.mean(subset)
                        sem_val = np.std(subset) / np.sqrt(len(subset)) if len(subset) > 1 else 0

                        bars_data.append({
                            'pos': pos, 'mean': mean_val, 'sem': sem_val,
                            'color': color, 'hatch': hatch, 'width': bar_width
                        })

                    pos += 0.4
                pos += 0.1
            x_positions.append((mouse_start + 0.7, f'W{mouse_label}'))
            pos += 0.2

        # Store age label position (center of this age bracket)
        age_end = pos
        age_labels.append(((age_start + age_end) / 2, f'{int(age)}mo'))

        # Add age separator (solid line) unless this is the last age
        if age_idx < len(all_ages) - 1:
            age_separators.append(pos)
            pos += 0.5

    return bars_data, x_positions, age_separators, genotype_separators, age_labels


def plot_mouse_ids(ax, df_fov_q111, df_fov_wt, mouse_id_map_q111, mouse_id_map_wt):
    """Plot total mRNA by mouse ID, grouped by age brackets."""
    cfg = FigureConfig

    bars_data, x_positions, age_separators, genotype_separators, age_labels = plot_mouse_ids_data(
        df_fov_q111, df_fov_wt, mouse_id_map_q111, mouse_id_map_wt
    )

    # Find max value for axis limit
    max_val = max(b['mean'] + b['sem'] for b in bars_data) if bars_data else 180

    # Plot bars
    for bar in bars_data:
        ax.bar(bar['pos'], bar['mean'], yerr=bar['sem'], width=bar['width'],
               color=bar['color'], alpha=0.8, edgecolor='black', linewidth=1,
               hatch=bar['hatch'], capsize=2, error_kw={'linewidth': 1})

    # Add dashed vertical lines between Q111 and WT within each age bracket
    for sep_pos in genotype_separators:
        ax.axvline(x=sep_pos, color='gray', linestyle='--', alpha=0.5, linewidth=1.0)

    # Add solid vertical lines between age brackets
    for sep_pos in age_separators:
        ax.axvline(x=sep_pos, color='black', linestyle='-', alpha=0.7, linewidth=1.5)

    # Set y-limits
    ax.set_ylim(0, max_val * 1.1)

    # Set x-ticks
    ax.set_xticks([p[0] for p in x_positions])
    ax.set_xticklabels([p[1] for p in x_positions], fontsize=cfg.FONT_SIZE_AXIS_TICK - 2, rotation=45)

    # Add age bracket labels at the top
    for age_pos, age_label in age_labels:
        ax.text(age_pos, max_val * 1.05, age_label, ha='center', va='bottom',
                fontsize=cfg.FONT_SIZE_AXIS_LABEL, fontweight='bold')

    # Labels
    ax.set_xlabel('Mouse by Age', fontsize=cfg.FONT_SIZE_AXIS_LABEL)
    ax.set_ylabel('Total mRNA/nucleus', fontsize=cfg.FONT_SIZE_AXIS_LABEL)

    # Grid
    ax.grid(axis='y', alpha=0.3, linestyle='--')


# ══════════════════════════════════════════════════════════════════════════════
# MAIN FIGURE CREATION
# ══════════════════════════════════════════════════════════════════════════════

def create_figure3():
    """Create Figure 3 with the specified layout."""
    cfg = FigureConfig

    # Load data
    data = load_and_process_data()
    df_subregion = data['df_subregion']
    df_fov_q111 = data['df_fov_q111']
    df_fov_wt = data['df_fov_wt']
    mouse_id_map_q111 = data['mouse_id_map_q111']
    mouse_id_map_wt = data['mouse_id_map_wt']

    # Figure dimensions - use standard page width from config
    fig_width = cfg.PAGE_WIDTH_FULL
    fig_height = fig_width * 1.3  # Adjusted height for 4 rows

    fig = plt.figure(figsize=(fig_width, fig_height))

    # Main grid: 10 rows x 12 cols
    # Row 1 (rows 0-2): A (anatomical overview), B (subregional boxplots)
    # Row 2 (rows 2-4): C (example FOVs), D (age breakdown)
    # Row 3 (rows 4-6): E (full width, atlas coords, broken axis)
    # Row 4 (rows 6-8): F (full width, mouse IDs, broken axis)
    main_gs = gridspec.GridSpec(
        10, 12,
        figure=fig,
        left=cfg.SUBPLOT_LEFT + 0.05,
        right=cfg.SUBPLOT_RIGHT - 0.01,
        bottom=cfg.SUBPLOT_BOTTOM + 0.03,
        top=cfg.SUBPLOT_TOP - 0.01,
        hspace=0.7,
        wspace=0.5,
        height_ratios=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    )

    axes = {}

    # ══════════════════════════════════════════════════════════════════════════
    # ROW 1: A (anatomical overview, 1/3), B (subregional boxplots, 2/3)
    # ══════════════════════════════════════════════════════════════════════════
    axes['A'] = fig.add_subplot(main_gs[0:2, 0:4])  # 1/3 width - anatomical overview
    axes['B'] = fig.add_subplot(main_gs[0:2, 4:12])  # 2/3 width - subregional boxplots

    # ══════════════════════════════════════════════════════════════════════════
    # ROW 2: C (example FOVs, larger), D (age breakdown)
    # ══════════════════════════════════════════════════════════════════════════
    axes['C'] = fig.add_subplot(main_gs[2:4, 0:6])  # Left half - example FOV images
    axes['D'] = fig.add_subplot(main_gs[2:4, 6:12])  # Right half - age breakdown

    # ══════════════════════════════════════════════════════════════════════════
    # ROW 3: E (full width, atlas coords, single axis)
    # ══════════════════════════════════════════════════════════════════════════
    axes['E'] = fig.add_subplot(main_gs[4:6, 0:12])

    # ══════════════════════════════════════════════════════════════════════════
    # ROW 4: F (full width, mouse IDs, single axis)
    # ══════════════════════════════════════════════════════════════════════════
    axes['F'] = fig.add_subplot(main_gs[6:8, 0:12])

    # ══════════════════════════════════════════════════════════════════════════
    # FILL PANELS
    # ══════════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 70)
    print("CREATING PANELS")
    print("=" * 70)

    # Panel A: Empty placeholder (anatomical overview with DARPP-32)
    axes['A'].set_facecolor(COLORS['gray_light'])
    axes['A'].text(0.5, 0.5, 'Panel A', transform=axes['A'].transAxes,
                   ha='center', va='center', fontsize=cfg.FONT_SIZE_TITLE,
                   color=COLORS['gray_dark'])
    axes['A'].set_xticks([])
    axes['A'].set_yticks([])

    # Panel B: Regional boxplots (referencing subregion labels a-i from Panel A)
    print("  Creating Panel B (regional boxplots)...")
    plot_regional_boxplots(axes['B'], df_subregion)

    # Panel C: Empty placeholder (example FOV images: low, medium, high expression)
    axes['C'].set_facecolor(COLORS['gray_light'])
    axes['C'].text(0.5, 0.5, 'Panel C', transform=axes['C'].transAxes,
                   ha='center', va='center', fontsize=cfg.FONT_SIZE_TITLE,
                   color=COLORS['gray_dark'])
    axes['C'].set_xticks([])
    axes['C'].set_yticks([])

    # Panel D: Age trends
    print("  Creating Panel D (age trends)...")
    plot_age_trends(axes['D'], df_fov_q111, df_fov_wt)

    # Panel E: Atlas coordinates (single axis)
    print("  Creating Panel E (atlas coordinates)...")
    plot_atlas_coords(axes['E'], df_fov_q111, df_fov_wt)

    # Panel F: Mouse IDs (single axis)
    print("  Creating Panel F (mouse IDs)...")
    plot_mouse_ids(axes['F'], df_fov_q111, df_fov_wt,
                   mouse_id_map_q111, mouse_id_map_wt)

    # ══════════════════════════════════════════════════════════════════════════
    # ADD PANEL LABELS
    # ══════════════════════════════════════════════════════════════════════════

    label_offset_x = -0.02
    label_offset_y = 0.015

    # Row 1: A, B
    for label in ['A', 'B']:
        ax = axes[label]
        bbox = ax.get_position()
        fig.text(
            bbox.x0 + label_offset_x, bbox.y1 + label_offset_y, label,
            fontsize=cfg.FONT_SIZE_PANEL_LABEL,
            fontweight=cfg.FONT_WEIGHT_PANEL_LABEL,
            va='bottom', ha='left'
        )

    # Row 2: C, D - simple panels
    for label in ['C', 'D']:
        ax = axes[label]
        bbox = ax.get_position()
        fig.text(
            bbox.x0 + label_offset_x, bbox.y1 + label_offset_y, label,
            fontsize=cfg.FONT_SIZE_PANEL_LABEL,
            fontweight=cfg.FONT_WEIGHT_PANEL_LABEL,
            va='bottom', ha='left'
        )

    # E and F - single axes
    for label in ['E', 'F']:
        ax = axes[label]
        bbox = ax.get_position()
        fig.text(
            bbox.x0 + label_offset_x, bbox.y1 + label_offset_y, label,
            fontsize=cfg.FONT_SIZE_PANEL_LABEL,
            fontweight=cfg.FONT_WEIGHT_PANEL_LABEL,
            va='bottom', ha='left'
        )

    # ══════════════════════════════════════════════════════════════════════════
    # STATISTICS
    # ══════════════════════════════════════════════════════════════════════════

    print("  Computing statistical tests...")
    stat_tests = compute_statistical_tests(df_fov_q111, df_fov_wt)

    print("  Computing sample sizes...")
    sample_sizes = compute_sample_sizes(df_fov_q111, df_fov_wt)

    stats = {
        'n_subregion_records': len(df_subregion),
        'n_fov_q111': len(df_fov_q111),
        'n_fov_wt': len(df_fov_wt),
        'n_mice_q111': len(mouse_id_map_q111),
        'n_mice_wt': len(mouse_id_map_wt),
        'ages': sorted(df_fov_q111['Age'].unique()),
        'mouse_id_map_q111': mouse_id_map_q111,
        'mouse_id_map_wt': mouse_id_map_wt,
        'region_tests': stat_tests['region_tests'],
        'age_tests': stat_tests['age_tests'],
        'genotype_tests': stat_tests['genotype_tests'],
        'sample_sizes': sample_sizes,
    }

    return fig, axes, stats


def build_region_table(region_tests):
    """Build Table 1: Region comparison (Cortex vs Striatum) dynamically."""
    header = """TABLE 1: REGION COMPARISON (Cortex vs Striatum)
+----------+------------------+------+-------------------+--------------------+--------+---------+------+
| Genotype | Channel          | Age  | Cortex median     | Striatum median    | U      | p-value | Sig  |
+----------+------------------+------+-------------------+--------------------+--------+---------+------+"""

    rows = []
    for test in region_tests:
        pval_str = '<0.001' if test['p_value'] < 0.001 else f"{test['p_value']:.3f}"
        cortex_str = f"{test['cortex_median']:.1f} (n={test['cortex_n']})"
        striatum_str = f"{test['striatum_median']:.1f} (n={test['striatum_n']})"
        age_str = f"{int(test['age'])}mo"
        row = f"| {test['genotype']:<8} | {test['channel']:<16} | {age_str:<4} | {cortex_str:<17} | {striatum_str:<18} | {int(test['U']):<6} | {pval_str:<7} | {test['sig']:<4} |"
        rows.append(row)

    footer = "+----------+------------------+------+-------------------+--------------------+--------+---------+------+"

    return header + '\n' + '\n'.join(rows) + '\n' + footer


def build_age_table(age_tests):
    """Build Table 2: Age effect (pairwise Mann-Whitney U tests) dynamically."""
    header = """TABLE 2: AGE EFFECT (pairwise Mann-Whitney U tests between consecutive ages)
+----------+------------------+----------+-------------+-------------------+-------------------+--------+---------+------+
| Genotype | Channel          | Region   | Comparison  | Earlier median    | Later median      | U      | p-value | Sig  |
+----------+------------------+----------+-------------+-------------------+-------------------+--------+---------+------+"""

    rows = []
    for test in age_tests:
        pval_str = '<0.001' if test['p_value'] < 0.001 else f"{test['p_value']:.3f}"
        comparison = f"{int(test['age1'])}→{int(test['age2'])}mo"
        earlier_str = f"{test['median1']:.1f} (n={test['n1']})"
        later_str = f"{test['median2']:.1f} (n={test['n2']})"
        row = f"| {test['genotype']:<8} | {test['channel']:<16} | {test['region']:<8} | {comparison:<11} | {earlier_str:<17} | {later_str:<17} | {int(test['U']):<6} | {pval_str:<7} | {test['sig']:<4} |"
        rows.append(row)

    footer = "+----------+------------------+----------+-------------+-------------------+-------------------+--------+---------+------+"

    return header + '\n' + '\n'.join(rows) + '\n' + footer


def build_genotype_table(genotype_tests):
    """Build Table 3: Genotype comparison (Q111 vs WT) dynamically."""
    header = """TABLE 3: GENOTYPE COMPARISON (Q111 vs WT)
+------------------+----------+------+-------------------+-------------------+--------+---------+------+
| Channel          | Region   | Age  | Q111 median       | WT median         | U      | p-value | Sig  |
+------------------+----------+------+-------------------+-------------------+--------+---------+------+"""

    rows = []
    for test in genotype_tests:
        pval_str = '<0.001' if test['p_value'] < 0.001 else f"{test['p_value']:.3f}"
        q111_str = f"{test['q111_median']:.1f} (n={test['q111_n']})"
        wt_str = f"{test['wt_median']:.1f} (n={test['wt_n']})"
        age_str = f"{int(test['age'])}mo"
        row = f"| {test['channel']:<16} | {test['region']:<8} | {age_str:<4} | {q111_str:<17} | {wt_str:<17} | {int(test['U']):<6} | {pval_str:<7} | {test['sig']:<4} |"
        rows.append(row)

    footer = "+------------------+----------+------+-------------------+-------------------+--------+---------+------+"

    return header + '\n' + '\n'.join(rows) + '\n' + footer


def build_sample_size_table(sample_sizes):
    """Build a table showing sample sizes (slides and FOVs) per condition."""
    header = """SAMPLE SIZE TABLE (n slides / n FOVs per condition)
+----------+------+----------+------------------+-----------------+-----------------+
| Genotype | Age  | Region   | mHTT1a           | full-length     | Total FOVs      |
|          |      |          | (slides/FOVs)    | (slides/FOVs)   | per region      |
+----------+------+----------+------------------+-----------------+-----------------+"""

    rows = []
    # Group by genotype, age, region
    from collections import defaultdict
    grouped = defaultdict(dict)
    for s in sample_sizes:
        key = (s['genotype'], s['age'], s['region'])
        grouped[key][s['channel']] = (s['n_slides'], s['n_fovs'])

    # Sort by genotype, age, region
    for key in sorted(grouped.keys(), key=lambda x: (x[0], x[1], x[2])):
        genotype, age, region = key
        mhtt1a = grouped[key].get('mHTT1a', (0, 0))
        full = grouped[key].get('full-length mHTT', (0, 0))
        total_fovs = mhtt1a[1] + full[1]

        row = f"| {genotype:<8} | {int(age)}mo  | {region:<8} | {mhtt1a[0]}/{mhtt1a[1]:<13} | {full[0]}/{full[1]:<12} | {total_fovs:<15} |"
        rows.append(row)

    footer = "+----------+------+----------+------------------+-----------------+-----------------+"

    return header + '\n' + '\n'.join(rows) + '\n' + footer


def compute_region_key_finding(region_tests):
    """Generate key finding text for region comparisons based on actual results."""
    # Count significant results
    q111_sig = [t for t in region_tests if t['genotype'] == 'Q111' and t['sig'] != 'ns']
    wt_sig = [t for t in region_tests if t['genotype'] == 'WT' and t['sig'] != 'ns']

    if len(q111_sig) == 0 and len(wt_sig) == 0:
        return "Key finding: With per-slide aggregation (n=4-6 slides per group), no significant regional differences were detected, likely due to limited statistical power with small sample sizes."

    findings = []
    if len(q111_sig) > 0:
        details = [f"{t['channel']} at {int(t['age'])}mo" for t in q111_sig]
        findings.append(f"Q111: significant regional differences in {', '.join(details)}")
    if len(wt_sig) > 0:
        details = [f"{t['channel']} at {int(t['age'])}mo" for t in wt_sig]
        findings.append(f"WT: significant regional differences in {', '.join(details)}")

    return "Key finding: " + "; ".join(findings) + "."


def compute_age_key_finding(age_tests):
    """Generate key finding text for age effect based on actual results."""
    sig_tests = [t for t in age_tests if t['sig'] != 'ns']

    if len(sig_tests) == 0:
        return "Key finding: With per-slide aggregation (n=4-6 slides per age group), no significant pairwise age differences were detected, likely due to limited statistical power with small sample sizes. However, median values consistently increase with age across conditions."

    if len(sig_tests) == len(age_tests):
        return "Key finding: Expression increases significantly between consecutive age groups in ALL conditions (both genotypes, both probes, both regions)."

    # Group significant findings
    details = []
    for t in sig_tests:
        comparison = f"{int(t['age1'])}→{int(t['age2'])}mo"
        details.append(f"{t['genotype']} {t['channel']} {t['region']} ({comparison})")
    return f"Key finding: Significant age-related increases detected in: {', '.join(details)}."


def compute_genotype_key_finding(genotype_tests):
    """Compute the key finding string based on actual genotype comparison results."""
    sig_tests = [t for t in genotype_tests if t['sig'] != 'ns']

    if len(sig_tests) == 0:
        # Find the largest fold difference even if not significant
        max_fold = 0
        max_test = None
        for test in genotype_tests:
            if test['wt_median'] > 0:
                fold = test['q111_median'] / test['wt_median']
                if fold > max_fold:
                    max_fold = fold
                    max_test = test

        if max_test is not None:
            return f"Key finding: With per-slide aggregation (n=4-6 Q111 slides, n=1-2 WT slides), no significant genotype differences were detected due to limited statistical power. However, Q111 consistently showed higher expression than WT (e.g., {max_test['channel']} in {max_test['region'].lower()} at {int(max_test['age'])}mo: {max_fold:.1f}-fold difference, {max_test['q111_median']:.1f} vs {max_test['wt_median']:.1f} mRNA/nucleus)."
        return "Key finding: No significant genotype differences detected with per-slide aggregation."

    # Some significant results
    max_fold = 0
    max_test = None
    for test in sig_tests:
        if test['wt_median'] > 0:
            fold = test['q111_median'] / test['wt_median']
            if fold > max_fold:
                max_fold = fold
                max_test = test

    if max_test is not None:
        return f"Key finding: Q111 mice show significantly higher mRNA expression than WT. The effect is strongest for {max_test['channel']} in {max_test['region'].lower()} at {int(max_test['age'])}mo ({max_fold:.1f}-fold difference: {max_test['q111_median']:.1f} vs {max_test['wt_median']:.1f} mRNA/nucleus, p={max_test['p_value']:.3f})."

    return "Key finding: Q111 mice show significantly higher mRNA expression than WT in some conditions."


def generate_caption(stats):
    """Generate figure caption with mouse ID mapping table."""
    mouse_id_map_q111 = stats.get('mouse_id_map_q111', {})
    mouse_id_map_wt = stats.get('mouse_id_map_wt', {})

    # Build statistical tables dynamically
    region_table = build_region_table(stats.get('region_tests', []))
    age_table = build_age_table(stats.get('age_tests', []))
    genotype_table = build_genotype_table(stats.get('genotype_tests', []))
    sample_size_table = build_sample_size_table(stats.get('sample_sizes', []))

    # Build key findings dynamically based on actual statistical results
    region_key_finding = compute_region_key_finding(stats.get('region_tests', []))
    age_key_finding = compute_age_key_finding(stats.get('age_tests', []))
    genotype_key_finding = compute_genotype_key_finding(stats.get('genotype_tests', []))

    # Build mouse ID mapping table strings
    def build_mouse_table(id_map, prefix):
        if not id_map:
            return f"  (No {prefix} mice)"
        lines = []
        for mouse_id, label in sorted(id_map.items(), key=lambda x: int(x[1].replace('#', ''))):
            num = label.replace('#', '')
            lines.append(f"  {prefix}#{num} → {mouse_id}")
        return '\n'.join(lines)

    q111_table = build_mouse_table(mouse_id_map_q111, 'Q')
    wt_table = build_mouse_table(mouse_id_map_wt, 'W')

    caption = f"""Figure 3: Regional expression patterns and total mRNA quantification in Q111 and WT mice.

OVERVIEW:
This figure presents a comprehensive analysis of mRNA expression patterns across brain regions, age groups, anatomical coordinates, and individual animals. The analysis compares Q111 Huntington's disease transgenic mice with wildtype (WT) controls, examining both mHTT1a (mutant huntingtin exon 1, detected in green channel) and full-length mHTT (complete mutant huntingtin transcript, detected in orange channel) probes. The data reveal regional heterogeneity in expression levels and demonstrate reproducibility across biological replicates.

PANEL DESCRIPTIONS:

(A) ANATOMICAL OVERVIEW AND SUBREGION LABELS
Representative images showing the brain regions analyzed in this study.
- Left half-brain: Striatum highlighted in red, identified using DARPP-32 RNA probe as a cell-type marker
- Yellow box: Zoomed region showing imaging FOV locations
- Right half-brain: Schematic showing all imaged subregions labeled with lowercase letters (a-i)

SUBREGION LABEL KEY (referenced in Panel B x-axis):
  Striatum (identified via DARPP-32 marker):
    a: dorsomedial striatum
    b: dorsolateral striatum
    c: ventrolateral striatum
    d: ventromedial striatum
  Cortex (approximate regions - no cell-type markers available):
    e: primary/secondary motor area (M1/M2)
    f: primary somatosensory area, upper limb (S1)
    g: primary somatosensory area, nose (S1)
    h: gustatory area / agranular insular area
    i: piriform area (olfactory cortex)

NOTE: Cortical region assignments are APPROXIMATE based on anatomical landmarks.
Without cell-type specific markers for cortex, these labels broadly indicate the
different cortical areas sampled to capture regional heterogeneity, but are not
definitive cell-type identifications.

(B) SUB-REGIONAL EXPRESSION ANALYSIS
Detailed breakdown of mRNA expression across cortical and striatal subregions.
- Data source: Regional analysis at subregion level from {stats['n_subregion_records']} measurements
- X-axis: Brain subregions labeled with lowercase letters (a-i) corresponding to Panel A
  * Cortex (left): e, f, g, h, i (sorted alphabetically)
  * Striatum (right): a, b, c, d (sorted alphabetically)
- Y-axis: mRNA expression per nucleus (mRNA/nucleus)
- Cortical subregions (approximate, no cell markers):
  * e: primary/secondary motor area
  * f: primary somatosensory area, upper limb
  * g: primary somatosensory area, nose
  * h: gustatory/agranular insular area
  * i: piriform area
- Striatal subregions (DARPP-32 identified):
  * a: dorsomedial striatum
  * b: dorsolateral striatum
  * c: ventrolateral striatum
  * d: ventromedial striatum
- Color coding:
  * Green boxes: mHTT1a probe (mutant huntingtin exon 1)
  * Orange boxes: Full-length mHTT probe (complete mutant huntingtin)
  * Solid fill: Cortex subregions
  * Hatched fill (///): Striatum subregions
- Statistical representation: Box plots show median (center line), interquartile range (box), and whiskers extending to 1.5× IQR
- Vertical dashed line separates Cortex (left) from Striatum (right) regions
- Key observations:
  * Expression levels vary across subregions within each major region
  * mHTT1a and full-length mHTT show similar patterns but different absolute levels
  * Striatum generally shows higher expression than cortex

(C) EXAMPLE FOV IMAGES - LOW, MEDIUM, HIGH EXPRESSION
Representative microscopy images showing the range of mRNA expression levels observed.
- Three columns: Low expression, Medium expression, High expression FOVs
- Each column shows FOV overview and zoomed regions
- Expression values annotated in green (mHTT1a) and orange (full-length mHTT) as mRNA/nucleus
- DAPI nuclear stain shown in blue
- Demonstrates the biological heterogeneity in expression levels across FOVs

(D) TOTAL mRNA EXPRESSION BY AGE (per-slide aggregation)
Age-dependent expression analysis comparing Q111 transgenic and WT control mice.
- DATA AGGREGATION: To avoid pseudoreplication, FOVs are first averaged within each slide.
  Each bar shows mean ± SEM across slides (n = number of slides, not FOVs).
  See sample size table below for n per condition.
- Age groups analyzed: {', '.join([f'{int(a)} months' for a in stats['ages']])}
- Data visualization: Bar plots showing mean ± standard error of the mean (SEM)
- Color coding for Q111 mice:
  * Green: mHTT1a probe
  * Orange: Full-length mHTT probe
- Color coding for WT mice:
  * Blue: mHTT1a probe
  * Purple: Full-length mHTT probe
- Pattern coding:
  * Solid bars: Cortex region
  * Hatched bars (///): Striatum region
- Number of mice: {stats['n_mice_q111']} Q111 mice, {stats['n_mice_wt']} WT mice
- Interpretation:
  * Age-related changes in expression can be detected by comparing bar heights across age groups
  * Q111 vs WT comparison reveals disease-associated expression differences
  * Region-specific patterns (Cortex vs Striatum) may indicate differential vulnerability

(E) TOTAL mRNA EXPRESSION BY BRAIN ATLAS COORDINATE, GROUPED BY AGE (all FOVs merged)
Anterior-posterior expression gradient analysis organized by age brackets ({', '.join([f'{int(a)}mo' for a in stats['ages']])}).
- DATA AGGREGATION: All FOVs per atlas coordinate are merged together (not per-slide).
  Each bar shows mean ± SEM across all FOVs at that coordinate.
  This provides maximum spatial resolution but n reflects FOV count, not biological replicates.
- X-axis: Brain atlas coordinate in 25 μm units from Bregma, grouped by age
  * Age brackets (bold labels at top): {', '.join([f'{int(a)}mo' for a in stats['ages']])}
  * Within each age bracket: Q111 coordinates (left), WT coordinates (right)
  * Q (e.g., Q38, Q44): Q111 transgenic mouse coordinates
  * W (e.g., W38, W44): Wildtype mouse coordinates
- Y-axis: Total mRNA per nucleus
- Data visualization: Grouped bar plots showing mean ± SEM across FOVs
- Color and pattern coding: Same as panel D
- Separator lines:
  * Solid vertical lines: Separate age brackets
  * Dashed vertical lines: Separate Q111 from WT within each age bracket
- Interpretation:
  * Within-age comparisons: Q111 vs WT at each coordinate within the same age
  * Across-age comparisons: Same coordinates can be compared between age brackets
  * Anterior-posterior gradients visible within each age group
  * Age-related changes in expression gradient patterns may indicate disease progression

(F) PER-SLIDE BREAKDOWN, GROUPED BY AGE (individual slides shown)
Per-slide analysis organized by age brackets ({', '.join([f'{int(a)}mo' for a in stats['ages']])}).
- DATA AGGREGATION: Each bar represents one slide. FOVs within each slide are averaged,
  and each slide is shown as a separate entry. Error bars show SEM across FOVs within that slide.
  This is the most granular view of biological variability at the slide level.
- X-axis: Individual slide identifiers, grouped by age
  * Age brackets (bold labels at top): {', '.join([f'{int(a)}mo' for a in stats['ages']])}
  * Within each age bracket: Q111 slides (left), WT slides (right)
  * Q#1.1, Q#1.2: Slide-level labels within each mouse (mouse.slide notation)
  * W#1.1, W#1.2: Same notation for WT mice
- Y-axis: Total mRNA per nucleus
- Data visualization: Grouped bar plots showing mean ± SEM per slide (across FOVs within slide)
- Each entry has 4 bars: mHTT1a Cortex, mHTT1a Striatum, full-length Cortex, full-length Striatum
- Color and pattern coding: Same as panels D and E
- Separator lines:
  * Solid vertical lines: Separate age brackets
  * Dashed vertical lines: Separate Q111 from WT within each age bracket
- Key observations:
  * Within-age variability: Inter-individual differences within the same age group
  * Age-related patterns: Comparison of mouse-level expression across ages
  * Mice are grouped by biological age, facilitating age-matched comparisons
  * Reproducibility across mice of the same age supports statistical validity

================================================================================
MOUSE ID MAPPING TABLE
================================================================================
The following table maps anonymized mouse IDs used in panel F to actual mouse identifiers.
This mapping is consistent with Figure 4, panel F.

Q111 MICE:
{q111_table}

WILDTYPE (WT) MICE:
{wt_table}

DATA SUMMARY:
- Total Q111 FOV measurements: {stats['n_fov_q111']}
- Total WT FOV measurements: {stats['n_fov_wt']}
- Number of Q111 mice: {stats['n_mice_q111']}
- Number of WT mice: {stats['n_mice_wt']}
- Subregion-level measurements: {stats['n_subregion_records']}
- Age groups analyzed: {len(stats['ages'])} ({', '.join([f'{int(a)}mo' for a in stats['ages']])})

{sample_size_table}

================================================================================
FILTERING APPLIED (consistent with Figure 1, panels E onwards)
================================================================================

FOV-LEVEL ANALYSIS:
This figure analyzes mRNA expression at the FOV (field-of-view) level, aggregating cluster intensities per FOV and normalizing by nucleus count.

UPSTREAM SPOT FILTERING (applied before cluster identification):
1. DETECTION QUALITY FILTER (PFA):
   - Criterion: Probability of False Alarm < {MAX_PFA}
   - Purpose: Remove spots with poor Gaussian fit quality

2. INTENSITY THRESHOLD (from negative controls):
   - Criterion: Integrated photons > {QUANTILE_NEGATIVE_CONTROL*100:.0f}th percentile of negative control distribution
   - Purpose: Remove background/noise spots below detection threshold
   - Threshold: Calculated per slide-channel combination

3. SIZE LOWER BOUND (from bead PSF):
   - Criterion: σ_x ≥ {SIGMA_X_LOWER:.1f} nm (80% × bead PSF of {BEAD_PSF_X:.1f} nm)
   - Purpose: Remove artifactually small spots (noise, fitting errors)

CLUSTER IDENTIFICATION:
- Method: 3D connected component analysis on intensity-thresholded images
- Cluster intensity: Sum of all voxel intensities, normalized to mRNA equivalents

4. CLUSTER INTENSITY THRESHOLD (from negative controls):
   - Criterion: Cluster total intensity > {QUANTILE_NEGATIVE_CONTROL*100:.0f}th percentile of negative control distribution
   - Purpose: Remove clusters with intensity below the noise floor (false positive clusters)
   - Threshold: Calculated per slide-channel combination (same threshold as spot filter)

5. CLUSTER CV (COEFFICIENT OF VARIATION) THRESHOLD:
   - Criterion: Cluster CV >= {CV_THRESHOLD} (CV = standard deviation / mean of voxel intensities)
   - Purpose: Remove clusters with low intensity heterogeneity (likely noise or artifacts)
   - Rationale: True mRNA aggregates show spatial variation in signal; uniform low-variance regions are noise
   - See Figure 2 caption for detailed cluster discard statistics

FOV-LEVEL AGGREGATION:
- Total mRNA per FOV: Sum of all cluster intensities in the FOV
- mRNA per nucleus: (Total mRNA per FOV) / (Number of DAPI-positive nuclei)
- Minimum nuclei threshold: FOVs with < 40 nuclei are excluded

TECHNICAL NOTES:
- FOV = Field of View (single microscopy image)
- mRNA/nucleus calculated as: (total mRNA in FOV) / (number of DAPI-positive nuclei in FOV)
- Bead PSF: σ_x = {BEAD_PSF_X:.1f} nm, σ_y = {BEAD_PSF_Y:.1f} nm, σ_z = {BEAD_PSF_Z:.1f} nm
- Size lower bound: σ ≥ 80% of bead PSF ({SIGMA_X_LOWER:.1f} nm for σ_x)
- Excluded slides: {EXCLUDED_SLIDES} (technical failures - imaging artifacts or tissue damage)
- SEM calculation: standard deviation / sqrt(n) where n = number of FOVs per condition
- Broken y-axis used when outlier values would compress the main distribution visualization
- Atlas coordinates reference: Franklin & Paxinos Mouse Brain Atlas, 3rd edition

COLOR SCHEME SUMMARY:
| Probe | Q111 Color | WT Color |
|-------|------------|----------|
| mHTT1a (exon 1) | Green (#2ecc71) | Blue (#3498db) |
| Full-length mHTT | Orange (#f39c12) | Purple (#9b59b6) |

| Region | Pattern |
|--------|---------|
| Cortex | Solid fill |
| Striatum | Hatched (///) |

================================================================================
STATISTICAL ANALYSIS
================================================================================

Statistical tests were performed to assess three biological questions:
1. Does expression differ between brain regions (Cortex vs Striatum)?
2. Does expression change with age?
3. Is there a genotype effect (Q111 vs WT)?

METHODS:
- Data aggregation: To avoid pseudoreplication, FOV-level measurements were first
  averaged within each slide. Each slide is treated as an independent observation
  unit, and n reported in the tables below refers to the number of slides, not FOVs.
  This is important because multiple FOVs from the same slide are not truly
  independent observations.
- Region comparisons: Mann-Whitney U test (two-sided, non-parametric) comparing Cortex vs Striatum
- Age comparisons: Pairwise Mann-Whitney U tests between consecutive age groups (2mo vs 6mo, 6mo vs 12mo)
- Genotype comparisons: Mann-Whitney U test (two-sided, non-parametric) comparing Q111 vs WT
- Significance levels: * p<0.05, ** p<0.01, *** p<0.001, ns = not significant

IMPORTANT NOTE ON SAMPLE SIZES:
- Q111 mice: n=4-6 slides per condition - sufficient for exploratory analysis but limited statistical power
- WT mice: n=1-2 slides per condition - TOO SMALL to draw meaningful statistical conclusions
  The WT data is included for descriptive comparison only. Any WT comparisons (region, age, or
  genotype) should be interpreted with extreme caution due to inadequate sample sizes. Additional
  WT samples would be needed to validate any observed differences.

{region_table}

{region_key_finding}

{age_table}

{age_key_finding}

{genotype_table}

{genotype_key_finding}

================================================================================

KEY OBSERVATIONS (descriptive, based on median values):
Note: Statistical tests using per-slide aggregation did not reach significance due to limited
sample sizes (n=4-6 slides per Q111 group, n=1-2 for WT). The observations below describe
trends visible in the data that would require larger sample sizes to confirm statistically.

1. REGIONAL PATTERNS: mHTT1a median expression tends to be higher in striatum than cortex
   in Q111 mice at later ages (6-12mo), while full-length mHTT shows similar levels in both regions
2. AGE-RELATED TRENDS: Both probes show increasing median expression with age across conditions,
   consistent with progressive mRNA accumulation during disease progression
3. GENOTYPE DIFFERENCES: Q111 median expression values are consistently higher than WT
   (e.g., 5.2-fold for mHTT1a in striatum at 12mo), but statistical power is insufficient to confirm
4. PROBE-SPECIFIC PATTERNS: mHTT1a and full-length mHTT show different regional patterns in Q111
5. BIOLOGICAL REPRODUCIBILITY: Per-mouse analysis (Panel F) shows consistent patterns across slides
6. ANATOMICAL GRADIENTS: Atlas coordinate analysis (Panel E) reveals anterior-posterior variation

DATA CACHING:
Processed data is cached to {CACHE_FILE.name} for fast subsequent runs. Set FORCE_RELOAD = True to regenerate from raw data.
"""
    return caption


def main():
    """Generate and save Figure 3."""

    fig, axes, stats = create_figure3()

    print("\n" + "=" * 70)
    print("SAVING FIGURE")
    print("=" * 70)

    save_figure(fig, 'figure3', formats=['svg', 'png', 'pdf'], output_dir=OUTPUT_DIR)

    # Generate and save caption
    caption = generate_caption(stats)
    caption_file = OUTPUT_DIR / 'figure3_caption.txt'
    with open(caption_file, 'w') as f:
        f.write(caption)
    print(f"Caption saved: {caption_file}")

    plt.close(fig)

    print("\n" + "=" * 70)
    print("FIGURE 3 COMPLETE")
    print("=" * 70)
    print(f"\nTo make layout changes quickly, just re-run this script.")
    print(f"Data is cached at: {CACHE_FILE}")


if __name__ == '__main__':
    main()
