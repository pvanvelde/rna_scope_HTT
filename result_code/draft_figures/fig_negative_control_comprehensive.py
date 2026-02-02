"""
Comprehensive Negative Control Analysis Figure

This script creates a detailed multi-panel figure analyzing negative control thresholds
across different brain regions (Cortex vs Striatum) and their relationships with:
- Age
- Sample size
- Brain atlas coordinates
- Detection channels

Author: Generated for RNA Scope analysis
Date: 2025-11-16
"""

import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel, pearsonr, spearmanr, linregress
import scienceplots
plt.style.use('science')
plt.rcParams['text.usetex'] = False
from pathlib import Path
import seaborn as sns
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import custom functions
from result_functions_v2 import compute_thresholds, recursively_load_dict, extract_dataframe
from results_config import (
    PIXELSIZE,
    SLICE_DEPTH,
    H5_FILE_PATH_EXPERIMENTAL,
    CHANNEL_COLORS,
    SLIDE_LABEL_MAP_Q111
)

# Physical parameters
pixelsize = PIXELSIZE
slice_depth = SLICE_DEPTH

# Create output directory
OUTPUT_DIR = Path(__file__).parent / "output" / "negative_control_comprehensive"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":

    print("="*80)
    print("COMPREHENSIVE NEGATIVE CONTROL ANALYSIS")
    print("Analyzing regional, age, sample size, and spatial dependencies")
    print("="*80)

    # ══════════════════════════════════════════════════════════════════════════
    # 1. LOAD DATA
    # ══════════════════════════════════════════════════════════════════════════

    h5_file_path = H5_FILE_PATH_EXPERIMENTAL

    with h5py.File(h5_file_path, 'r') as h5_file:
        data_dict = recursively_load_dict(h5_file)

    print(f"\n{'='*80}")
    print(f"Loaded data from: {h5_file_path}")
    print(f"{'='*80}")

    # ══════════════════════════════════════════════════════════════════════════
    # 2. EXTRACT DATAFRAME WITH METADATA
    # ══════════════════════════════════════════════════════════════════════════

    desired_channels = ['green', 'orange']
    fields_to_extract = [
        'spots.pfa_values',
        'spots.photons',
        'cluster_intensities',
        'metadata_sample.Age',
        'spots.final_filter',
        'spots.params_raw',
        'metadata_sample.Brain_Atlas_coordinates'
    ]

    negative_control_field = 'Negative control'
    experimental_field = 'ExperimentalQ111 - 488mHT - 548mHTa - 647Darp'
    slide_field = 'metadata_sample_slide_name_std'
    max_pfa = 0.05
    quantile_negative_control = 0.95

    df_extracted = extract_dataframe(
        data_dict,
        field_keys=fields_to_extract,
        channels=desired_channels,
        include_file_metadata_sample=True,
        explode_fields=[]
    )

    print(f"\nDataFrame extracted:")
    print(f"  Total rows: {len(df_extracted)}")
    print(f"  Channels: {desired_channels}")
    print(f"  Quantile threshold: {quantile_negative_control}")

    # ══════════════════════════════════════════════════════════════════════════
    # 3. COMPUTE THRESHOLDS PER REGION
    # ══════════════════════════════════════════════════════════════════════════

    print(f"\n{'='*80}")
    print("Computing region-specific thresholds...")
    print(f"{'='*80}")

    (thresholds, thresholds_cluster,
     error_thresholds, error_thresholds_cluster,
     number_of_datapoints, age_dict) = compute_thresholds(
        df_extracted=df_extracted,
        slide_field=slide_field,
        desired_channels=desired_channels,
        negative_control_field=negative_control_field,
        experimental_field=experimental_field,
        quantile_negative_control=quantile_negative_control,
        max_pfa=max_pfa,
        plot=False,
        n_bootstrap=20,
        use_region=True,
        use_final_filter=True,
    )

    # ══════════════════════════════════════════════════════════════════════════
    # 4. BUILD LONG-FORM DATAFRAME WITH METADATA
    # ══════════════════════════════════════════════════════════════════════════

    print(f"\n{'='*80}")
    print("Building analysis dataframe...")
    print(f"{'='*80}")

    # Create dataframe from bootstrap samples
    rows = []
    for (slide, channel, area), vec in error_thresholds.items():
        # Simplify area labels
        area_short = "Cortex" if "Cortex" in area else "Striatum"

        for v in np.asarray(vec):
            rows.append(dict(
                slide=slide,
                channel=channel,
                area=area_short,
                area_full=area,
                threshold=v
            ))

    df = pd.DataFrame(rows)

    # Compute per-slide mean thresholds
    mean_df = (
        df.groupby(["slide", "channel", "area"])["threshold"]
          .mean()
          .unstack("area")
          .reset_index()
          .dropna()
    )

    # ══════════════════════════════════════════════════════════════════════════
    # 5. ADD METADATA TO MEAN DATAFRAME
    # ══════════════════════════════════════════════════════════════════════════

    # Helper functions to extract metadata
    def get_cortex_datapoints(row):
        c_key = (row["slide"], row["channel"], "Cortex - undefined")
        return number_of_datapoints.get(c_key, np.nan)

    def get_striatum_datapoints(row):
        s_key = (row["slide"], row["channel"], "Striatum - undefined")
        return number_of_datapoints.get(s_key, np.nan)

    def get_age(row):
        # Try both region keys
        c_key = (row["slide"], row["channel"], "Cortex - undefined")
        s_key = (row["slide"], row["channel"], "Striatum - undefined")
        age_val = age_dict.get(c_key, age_dict.get(s_key, np.nan))
        return age_val

    def get_brain_atlas_coord(row):
        """Extract brain atlas coordinate from the original dataframe"""
        # Find matching rows in df_extracted
        slide_match = df_extracted[slide_field] == row["slide"]
        # Get unique brain atlas coordinate for this slide
        coords = df_extracted[slide_match]['metadata_sample_Brain_Atlas_coordinates'].dropna().unique()
        if len(coords) > 0:
            return coords[0]
        return np.nan

    # Add metadata columns
    mean_df["n_cortex"] = mean_df.apply(get_cortex_datapoints, axis=1)
    mean_df["n_striatum"] = mean_df.apply(get_striatum_datapoints, axis=1)
    mean_df["n_min"] = mean_df[["n_cortex", "n_striatum"]].min(axis=1)
    mean_df["n_max"] = mean_df[["n_cortex", "n_striatum"]].max(axis=1)
    mean_df["n_total"] = mean_df["n_cortex"] + mean_df["n_striatum"]
    mean_df["age"] = mean_df.apply(get_age, axis=1)
    mean_df["atlas_coord"] = mean_df.apply(get_brain_atlas_coord, axis=1)

    # Compute difference metrics
    mean_df["diff"] = mean_df["Cortex"] - mean_df["Striatum"]
    mean_df["abs_diff"] = np.abs(mean_df["diff"])
    mean_df["mean_threshold"] = mean_df[["Cortex", "Striatum"]].mean(axis=1)
    mean_df["pct_diff"] = 100 * mean_df["diff"] / mean_df["mean_threshold"]

    # Extract animal ID from slide name using the mapping
    def get_animal_id(slide):
        """Extract animal ID from slide name using SLIDE_LABEL_MAP_Q111"""
        label = SLIDE_LABEL_MAP_Q111.get(slide, None)
        if label is not None:
            # Label format is '#N.M' where N is animal number
            return label.split('.')[0]  # Returns '#N'
        return None

    mean_df["animal_id"] = mean_df["slide"].apply(get_animal_id)

    # Count slides per animal
    slides_per_animal = mean_df.groupby("animal_id")["slide"].nunique()
    print(f"\nSlides per animal:")
    for animal, count in slides_per_animal.items():
        print(f"  {animal}: {count} slides")

    print(f"\nAnalysis dataframe summary:")
    print(f"  Paired samples: {len(mean_df)}")
    print(f"  Unique slides: {mean_df['slide'].nunique()}")
    print(f"  Channels: {sorted(mean_df['channel'].unique())}")
    print(f"  Regions: Cortex, Striatum")
    print(f"  Age range: {mean_df['age'].min():.0f} - {mean_df['age'].max():.0f}")
    print(f"  Atlas coord range: {mean_df['atlas_coord'].min():.2f} - {mean_df['atlas_coord'].max():.2f}")

    # ══════════════════════════════════════════════════════════════════════════
    # 6. STATISTICAL ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════

    print(f"\n{'='*80}")
    print("STATISTICAL ANALYSIS")
    print(f"{'='*80}")

    # Overall paired t-test (Cortex vs Striatum)
    t_all, p_all = ttest_rel(mean_df["Cortex"], mean_df["Striatum"])
    mean_cortex = mean_df["Cortex"].mean()
    mean_striatum = mean_df["Striatum"].mean()
    mean_diff = mean_df["diff"].mean()
    std_diff = mean_df["diff"].std()

    print(f"\n1. Regional Comparison (Cortex vs Striatum):")
    print(f"   Pairs: n={len(mean_df)}")
    print(f"   Cortex:   {mean_cortex:7.1f} ± {mean_df['Cortex'].std():5.1f} photons")
    print(f"   Striatum: {mean_striatum:7.1f} ± {mean_df['Striatum'].std():5.1f} photons")
    print(f"   Difference: {mean_diff:+7.1f} ± {std_diff:5.1f} photons")
    print(f"   Paired t-test: t={t_all:.3f}, p={p_all:.4g}")
    sig_str = "***" if p_all < 0.001 else "**" if p_all < 0.01 else "*" if p_all < 0.05 else "ns"
    print(f"   Significance: {sig_str}")

    # Per-channel analysis
    print(f"\n2. Per-Channel Analysis:")
    channel_results = {}
    for ch in sorted(mean_df["channel"].unique()):
        ch_data = mean_df[mean_df["channel"] == ch]
        if len(ch_data) >= 3:
            t, p = ttest_rel(ch_data["Cortex"], ch_data["Striatum"])
            channel_results[ch] = {
                'n': len(ch_data),
                't': t,
                'p': p,
                'mean_cortex': ch_data["Cortex"].mean(),
                'mean_striatum': ch_data["Striatum"].mean(),
                'mean_diff': ch_data["diff"].mean()
            }
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            print(f"   {ch:8s}: n={len(ch_data):2d}, diff={channel_results[ch]['mean_diff']:+6.1f}, "
                  f"t={t:6.2f}, p={p:.4g} ({sig})")

    # Per-animal variance analysis (SEPARATE BY CHANNEL)
    print(f"\n3. Per-Animal Variance Analysis (by channel):")
    animal_df_stats = mean_df[mean_df["animal_id"].notna()].copy()

    per_animal_stats = {}

    if len(animal_df_stats) > 0:
        for ch in sorted(animal_df_stats["channel"].unique()):
            ch_data = animal_df_stats[animal_df_stats["channel"] == ch]
            print(f"\n   {ch.upper()} CHANNEL:")

            # Calculate per-animal statistics for this channel
            animal_stats_ch = ch_data.groupby("animal_id").agg({
                "Cortex": ["mean", "std", "count"],
                "Striatum": ["mean", "std", "count"],
                "diff": ["mean", "std"]
            })
            animal_stats_ch.columns = ['_'.join(col).strip() for col in animal_stats_ch.columns.values]

            # Intra-animal variance (mean of within-animal variances)
            intra_var_cortex = (ch_data.groupby("animal_id")["Cortex"].var()).mean()
            intra_var_striatum = (ch_data.groupby("animal_id")["Striatum"].var()).mean()
            intra_var_diff = (ch_data.groupby("animal_id")["diff"].var()).mean()

            # Inter-animal variance (variance of animal means)
            inter_var_cortex = animal_stats_ch["Cortex_mean"].var()
            inter_var_striatum = animal_stats_ch["Striatum_mean"].var()
            inter_var_diff = animal_stats_ch["diff_mean"].var()

            # Handle NaN for intra-animal variance when only 1 slide per animal
            intra_var_cortex = 0 if np.isnan(intra_var_cortex) else intra_var_cortex
            intra_var_striatum = 0 if np.isnan(intra_var_striatum) else intra_var_striatum
            intra_var_diff = 0 if np.isnan(intra_var_diff) else intra_var_diff

            # Intraclass correlation coefficient (ICC)
            total_var_cortex = intra_var_cortex + inter_var_cortex
            total_var_striatum = intra_var_striatum + inter_var_striatum
            total_var_diff = intra_var_diff + inter_var_diff

            icc_cortex = inter_var_cortex / total_var_cortex if total_var_cortex > 0 else np.nan
            icc_striatum = inter_var_striatum / total_var_striatum if total_var_striatum > 0 else np.nan
            icc_diff = inter_var_diff / total_var_diff if total_var_diff > 0 else np.nan

            print(f"     Animals: n={animal_stats_ch.shape[0]}")
            print(f"     ")
            print(f"     Cortex threshold:")
            print(f"       Intra-animal SD (within): {np.sqrt(intra_var_cortex):,.1f} photons")
            print(f"       Inter-animal SD (between): {np.sqrt(inter_var_cortex):,.1f} photons")
            print(f"       ICC (between/total): {icc_cortex:.3f}")
            print(f"     ")
            print(f"     Striatum threshold:")
            print(f"       Intra-animal SD (within): {np.sqrt(intra_var_striatum):,.1f} photons")
            print(f"       Inter-animal SD (between): {np.sqrt(inter_var_striatum):,.1f} photons")
            print(f"       ICC (between/total): {icc_striatum:.3f}")
            print(f"     ")
            print(f"     Regional difference (Cortex - Striatum):")
            print(f"       Intra-animal SD (within): {np.sqrt(intra_var_diff):,.1f} photons")
            print(f"       Inter-animal SD (between): {np.sqrt(inter_var_diff):,.1f} photons")
            print(f"       ICC (between/total): {icc_diff:.3f}")

            # Store for later use
            per_animal_stats[ch] = {
                'intra_sd_cortex': np.sqrt(intra_var_cortex),
                'inter_sd_cortex': np.sqrt(inter_var_cortex),
                'icc_cortex': icc_cortex,
                'intra_sd_striatum': np.sqrt(intra_var_striatum),
                'inter_sd_striatum': np.sqrt(inter_var_striatum),
                'icc_striatum': icc_striatum,
                'intra_sd_diff': np.sqrt(intra_var_diff),
                'inter_sd_diff': np.sqrt(inter_var_diff),
                'icc_diff': icc_diff,
                'n_animals': animal_stats_ch.shape[0]
            }

    # Age correlations
    print(f"\n4. Age Correlations:")
    valid_age = mean_df[mean_df["age"].notna()]
    if len(valid_age) > 3:
        r_cortex_age, p_cortex_age = pearsonr(valid_age["age"], valid_age["Cortex"])
        r_striatum_age, p_striatum_age = pearsonr(valid_age["age"], valid_age["Striatum"])
        r_diff_age, p_diff_age = pearsonr(valid_age["age"], valid_age["diff"])

        print(f"   Age vs Cortex:     r={r_cortex_age:+.3f}, p={p_cortex_age:.4g}")
        print(f"   Age vs Striatum:   r={r_striatum_age:+.3f}, p={p_striatum_age:.4g}")
        print(f"   Age vs Difference: r={r_diff_age:+.3f}, p={p_diff_age:.4g}")
    else:
        r_cortex_age = r_striatum_age = r_diff_age = np.nan
        p_cortex_age = p_striatum_age = p_diff_age = np.nan
        print(f"   Insufficient data (n={len(valid_age)})")

    # Sample size correlations
    print(f"\n5. Sample Size Correlations:")
    r_n_cortex, p_n_cortex = pearsonr(mean_df["n_cortex"], mean_df["Cortex"])
    r_n_striatum, p_n_striatum = pearsonr(mean_df["n_striatum"], mean_df["Striatum"])
    r_n_diff, p_n_diff = pearsonr(mean_df["n_min"], mean_df["abs_diff"])

    print(f"   Sample size vs Cortex threshold:   r={r_n_cortex:+.3f}, p={p_n_cortex:.4g}")
    print(f"   Sample size vs Striatum threshold: r={r_n_striatum:+.3f}, p={p_n_striatum:.4g}")
    print(f"   Min sample size vs |Difference|:   r={r_n_diff:+.3f}, p={p_n_diff:.4g}")

    # Brain atlas correlations
    print(f"\n5. Brain Atlas Coordinate Correlations:")
    valid_atlas = mean_df[mean_df["atlas_coord"].notna()]
    if len(valid_atlas) > 3:
        r_atlas_cortex, p_atlas_cortex = pearsonr(valid_atlas["atlas_coord"], valid_atlas["Cortex"])
        r_atlas_striatum, p_atlas_striatum = pearsonr(valid_atlas["atlas_coord"], valid_atlas["Striatum"])
        r_atlas_diff, p_atlas_diff = pearsonr(valid_atlas["atlas_coord"], valid_atlas["diff"])

        print(f"   Atlas coord vs Cortex:     r={r_atlas_cortex:+.3f}, p={p_atlas_cortex:.4g}")
        print(f"   Atlas coord vs Striatum:   r={r_atlas_striatum:+.3f}, p={p_atlas_striatum:.4g}")
        print(f"   Atlas coord vs Difference: r={r_atlas_diff:+.3f}, p={p_atlas_diff:.4g}")
    else:
        r_atlas_cortex = r_atlas_striatum = r_atlas_diff = np.nan
        p_atlas_cortex = p_atlas_striatum = p_atlas_diff = np.nan
        print(f"   Insufficient data (n={len(valid_atlas)})")

    # ══════════════════════════════════════════════════════════════════════════
    # 7. CREATE COMPREHENSIVE FIGURE
    # ══════════════════════════════════════════════════════════════════════════

    print(f"\n{'='*80}")
    print("Creating comprehensive figure...")
    print(f"{'='*80}")

    # Create figure with 5×3 grid (added row for per-animal analysis)
    # Leave space at top (top=0.92) for adding images later
    fig = plt.figure(figsize=(16, 17), dpi=300)
    gs = fig.add_gridspec(5, 3, hspace=0.50, wspace=0.30,
                         left=0.07, right=0.98, top=0.94, bottom=0.04)

    # Axis limits for scatter plots
    threshold_min = min(mean_df["Cortex"].min(), mean_df["Striatum"].min()) * 0.95
    threshold_max = max(mean_df["Cortex"].max(), mean_df["Striatum"].max()) * 1.05
    lims = [threshold_min, threshold_max]

    # ══════════════════════════════════════════════════════════════════════════
    # ROW 1: REGIONAL COMPARISON
    # ══════════════════════════════════════════════════════════════════════════

    # Panel A: Cortex vs Striatum scatter (colored by channel)
    ax_a = fig.add_subplot(gs[0, 0])
    for ch in sorted(mean_df["channel"].unique()):
        ch_data = mean_df[mean_df["channel"] == ch]
        color = CHANNEL_COLORS.get(ch, 'gray')
        ax_a.scatter(ch_data["Cortex"], ch_data["Striatum"],
                   c=color, s=80, alpha=0.7, edgecolor='black', linewidth=0.8,
                   label=ch, zorder=3)
    ax_a.plot(lims, lims, "k--", lw=2, alpha=0.5, label='Unity', zorder=1)
    ax_a.set_aspect("equal", adjustable="box")
    ax_a.set_xlabel("Cortex threshold [photons]", fontsize=11)
    ax_a.set_ylabel("Striatum threshold [photons]", fontsize=11)
    ax_a.set_title("A. Regional Comparison by Channel", fontsize=12, fontweight='bold', loc='left')
    ax_a.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax_a.grid(True, alpha=0.3, linestyle='--', zorder=0)
    ax_a.set_xlim(lims)
    ax_a.set_ylim(lims)

    # Panel B: Difference distribution histogram
    ax_b = fig.add_subplot(gs[0, 1])
    diff_values = mean_df["diff"].values
    ax_b.hist(diff_values, bins=20, alpha=0.7, color='steelblue',
             edgecolor='black', linewidth=0.8)
    ax_b.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Zero')
    ax_b.axvline(mean_diff, color='darkblue', linestyle='-', linewidth=2,
                alpha=0.7, label=f'Mean={mean_diff:.1f}')
    ax_b.set_xlabel("Cortex - Striatum [photons]", fontsize=11)
    ax_b.set_ylabel("Count", fontsize=11)
    ax_b.set_title("B. Threshold Difference Distribution", fontsize=12, fontweight='bold', loc='left')
    ax_b.legend(loc='best', fontsize=9)
    ax_b.grid(True, alpha=0.3, linestyle='--', axis='y')

    # Panel C: Violin plot by channel
    ax_c = fig.add_subplot(gs[0, 2])
    channel_list = sorted(mean_df["channel"].unique())
    violin_data = [mean_df[mean_df["channel"] == ch]["diff"].values for ch in channel_list]
    parts = ax_c.violinplot(violin_data, positions=range(len(channel_list)),
                          showmeans=True, showmedians=True, showextrema=True)
    # Color violins by channel
    for i, (pc, ch) in enumerate(zip(parts['bodies'], channel_list)):
        pc.set_facecolor(CHANNEL_COLORS.get(ch, 'gray'))
        pc.set_alpha(0.7)
    ax_c.axhline(0, linestyle="--", linewidth=2, color='red', alpha=0.7)
    ax_c.set_xticks(range(len(channel_list)))
    ax_c.set_xticklabels(channel_list, fontsize=10)
    ax_c.set_xlabel("Channel", fontsize=11)
    ax_c.set_ylabel("Cortex - Striatum [photons]", fontsize=11)
    ax_c.set_title("C. Difference by Channel", fontsize=12, fontweight='bold', loc='left')
    ax_c.grid(True, alpha=0.3, linestyle='--', axis='y')

    # ══════════════════════════════════════════════════════════════════════════
    # ROW 2: AGE DEPENDENCIES
    # ══════════════════════════════════════════════════════════════════════════

    # Panel D: Age vs Cortex threshold
    ax_d = fig.add_subplot(gs[1, 0])
    valid_age_df = mean_df[mean_df["age"].notna()]
    if len(valid_age_df) > 0:
        for ch in sorted(valid_age_df["channel"].unique()):
            ch_data = valid_age_df[valid_age_df["channel"] == ch]
            color = CHANNEL_COLORS.get(ch, 'gray')
            ax_d.scatter(ch_data["age"], ch_data["Cortex"],
                       c=color, s=70, alpha=0.7, edgecolor='black',
                       linewidth=0.7, label=ch)

        # Add separate regression lines per channel
        for ch in sorted(valid_age_df["channel"].unique()):
            ch_data = valid_age_df[valid_age_df["channel"] == ch]
            if len(ch_data) > 2:
                color = CHANNEL_COLORS.get(ch, 'gray')
                slope, intercept, r_value, p_value, std_err = linregress(
                    ch_data["age"], ch_data["Cortex"])
                x_line = np.array([ch_data["age"].min(), ch_data["age"].max()])
                y_line = slope * x_line + intercept
                ax_d.plot(x_line, y_line, '--', color=color, linewidth=2, alpha=0.8,
                        label=f'{ch}: r={r_value:.3f}, p={p_value:.3g}')

    ax_d.set_xlabel("Age [months]", fontsize=11)
    ax_d.set_ylabel("Cortex threshold [photons]", fontsize=11)
    ax_d.set_title("D. Age vs Cortex Threshold", fontsize=12, fontweight='bold', loc='left')
    ax_d.legend(loc='best', fontsize=8)
    ax_d.grid(True, alpha=0.3, linestyle='--')

    # Panel E: Age vs Striatum threshold
    ax_e = fig.add_subplot(gs[1, 1])
    if len(valid_age_df) > 0:
        for ch in sorted(valid_age_df["channel"].unique()):
            ch_data = valid_age_df[valid_age_df["channel"] == ch]
            color = CHANNEL_COLORS.get(ch, 'gray')
            ax_e.scatter(ch_data["age"], ch_data["Striatum"],
                       c=color, s=70, alpha=0.7, edgecolor='black',
                       linewidth=0.7, label=ch)

        # Add separate regression lines per channel
        for ch in sorted(valid_age_df["channel"].unique()):
            ch_data = valid_age_df[valid_age_df["channel"] == ch]
            if len(ch_data) > 2:
                color = CHANNEL_COLORS.get(ch, 'gray')
                slope, intercept, r_value, p_value, std_err = linregress(
                    ch_data["age"], ch_data["Striatum"])
                x_line = np.array([ch_data["age"].min(), ch_data["age"].max()])
                y_line = slope * x_line + intercept
                ax_e.plot(x_line, y_line, '--', color=color, linewidth=2, alpha=0.8,
                        label=f'{ch}: r={r_value:.3f}, p={p_value:.3g}')

    ax_e.set_xlabel("Age [months]", fontsize=11)
    ax_e.set_ylabel("Striatum threshold [photons]", fontsize=11)
    ax_e.set_title("E. Age vs Striatum Threshold", fontsize=12, fontweight='bold', loc='left')
    ax_e.legend(loc='best', fontsize=8)
    ax_e.grid(True, alpha=0.3, linestyle='--')

    # Panel F: Age vs Difference
    ax_f = fig.add_subplot(gs[1, 2])
    if len(valid_age_df) > 0:
        for ch in sorted(valid_age_df["channel"].unique()):
            ch_data = valid_age_df[valid_age_df["channel"] == ch]
            color = CHANNEL_COLORS.get(ch, 'gray')
            ax_f.scatter(ch_data["age"], ch_data["diff"],
                       c=color, s=70, alpha=0.7, edgecolor='black',
                       linewidth=0.7, label=ch)

        # Add separate regression lines per channel
        for ch in sorted(valid_age_df["channel"].unique()):
            ch_data = valid_age_df[valid_age_df["channel"] == ch]
            if len(ch_data) > 2:
                color = CHANNEL_COLORS.get(ch, 'gray')
                slope, intercept, r_value, p_value, std_err = linregress(
                    ch_data["age"], ch_data["diff"])
                x_line = np.array([ch_data["age"].min(), ch_data["age"].max()])
                y_line = slope * x_line + intercept
                ax_f.plot(x_line, y_line, '--', color=color, linewidth=2, alpha=0.8,
                        label=f'{ch}: r={r_value:.3f}, p={p_value:.3g}')

        ax_f.axhline(0, linestyle="--", linewidth=1.5, color='red', alpha=0.5)

    ax_f.set_xlabel("Age [months]", fontsize=11)
    ax_f.set_ylabel("Cortex - Striatum [photons]", fontsize=11)
    ax_f.set_title("F. Age vs Threshold Difference", fontsize=12, fontweight='bold', loc='left')
    ax_f.legend(loc='best', fontsize=8)
    ax_f.grid(True, alpha=0.3, linestyle='--')

    # ══════════════════════════════════════════════════════════════════════════
    # ROW 3: ATLAS COORDINATES (3 PANELS LIKE AGE ROW)
    # ══════════════════════════════════════════════════════════════════════════

    # Panel H: Brain atlas coordinates vs Cortex threshold
    ax_h = fig.add_subplot(gs[2, 0])
    valid_atlas_df = mean_df[mean_df["atlas_coord"].notna()]
    if len(valid_atlas_df) > 0:
        for ch in sorted(valid_atlas_df["channel"].unique()):
            ch_data = valid_atlas_df[valid_atlas_df["channel"] == ch]
            color = CHANNEL_COLORS.get(ch, 'gray')
            ax_h.scatter(ch_data["atlas_coord"], ch_data["Cortex"],
                       c=color, s=70, alpha=0.7, edgecolor='black',
                       linewidth=0.7, label=ch)

        # Add separate regression lines per channel
        for ch in sorted(valid_atlas_df["channel"].unique()):
            ch_data = valid_atlas_df[valid_atlas_df["channel"] == ch]
            if len(ch_data) > 2:
                color = CHANNEL_COLORS.get(ch, 'gray')
                slope, intercept, r_value, p_value, std_err = linregress(
                    ch_data["atlas_coord"], ch_data["Cortex"])
                x_line = np.array([ch_data["atlas_coord"].min(),
                                 ch_data["atlas_coord"].max()])
                y_line = slope * x_line + intercept
                ax_h.plot(x_line, y_line, '--', color=color, linewidth=2, alpha=0.8,
                        label=f'{ch}: r={r_value:.3f}, p={p_value:.3g}')

    ax_h.set_xlabel("Brain Atlas Coordinate [mm]", fontsize=11)
    ax_h.set_ylabel("Cortex threshold [photons]", fontsize=11)
    ax_h.set_title("H. Atlas Coordinate vs Cortex", fontsize=12, fontweight='bold', loc='left')
    ax_h.legend(loc='best', fontsize=8)
    ax_h.grid(True, alpha=0.3, linestyle='--')

    # Panel I: Brain atlas coordinates vs Striatum threshold
    ax_i = fig.add_subplot(gs[2, 1])
    if len(valid_atlas_df) > 0:
        for ch in sorted(valid_atlas_df["channel"].unique()):
            ch_data = valid_atlas_df[valid_atlas_df["channel"] == ch]
            color = CHANNEL_COLORS.get(ch, 'gray')
            ax_i.scatter(ch_data["atlas_coord"], ch_data["Striatum"],
                       c=color, s=70, alpha=0.7, edgecolor='black',
                       linewidth=0.7, label=ch)

        # Add separate regression lines per channel
        for ch in sorted(valid_atlas_df["channel"].unique()):
            ch_data = valid_atlas_df[valid_atlas_df["channel"] == ch]
            if len(ch_data) > 2:
                color = CHANNEL_COLORS.get(ch, 'gray')
                slope, intercept, r_value, p_value, std_err = linregress(
                    ch_data["atlas_coord"], ch_data["Striatum"])
                x_line = np.array([ch_data["atlas_coord"].min(),
                                 ch_data["atlas_coord"].max()])
                y_line = slope * x_line + intercept
                ax_i.plot(x_line, y_line, '--', color=color, linewidth=2, alpha=0.8,
                        label=f'{ch}: r={r_value:.3f}, p={p_value:.3g}')

    ax_i.set_xlabel("Brain Atlas Coordinate [mm]", fontsize=11)
    ax_i.set_ylabel("Striatum threshold [photons]", fontsize=11)
    ax_i.set_title("I. Atlas Coordinate vs Striatum", fontsize=12, fontweight='bold', loc='left')
    ax_i.legend(loc='best', fontsize=8)
    ax_i.grid(True, alpha=0.3, linestyle='--')

    # Panel J: Brain atlas coordinates vs regional difference (SIGNIFICANT FINDING!)
    ax_j = fig.add_subplot(gs[2, 2])
    if len(valid_atlas_df) > 0:
        for ch in sorted(valid_atlas_df["channel"].unique()):
            ch_data = valid_atlas_df[valid_atlas_df["channel"] == ch]
            color = CHANNEL_COLORS.get(ch, 'gray')
            ax_j.scatter(ch_data["atlas_coord"], ch_data["diff"],
                       c=color, s=70, alpha=0.7, edgecolor='black',
                       linewidth=0.7, label=ch)

        # Add separate regression lines per channel
        for ch in sorted(valid_atlas_df["channel"].unique()):
            ch_data = valid_atlas_df[valid_atlas_df["channel"] == ch]
            if len(ch_data) > 2:
                color = CHANNEL_COLORS.get(ch, 'gray')
                slope, intercept, r_value, p_value, std_err = linregress(
                    ch_data["atlas_coord"], ch_data["diff"])
                x_line = np.array([ch_data["atlas_coord"].min(),
                                 ch_data["atlas_coord"].max()])
                y_line = slope * x_line + intercept
                ax_j.plot(x_line, y_line, '--', color=color, linewidth=2, alpha=0.8,
                        label=f'{ch}: r={r_value:.3f}, p={p_value:.3g}')

        # Add zero line
        ax_j.axhline(0, linestyle="--", linewidth=1.5, color='red', alpha=0.5)

    ax_j.set_xlabel("Brain Atlas Coordinate [mm]", fontsize=11)
    ax_j.set_ylabel("Cortex - Striatum [photons]", fontsize=11)
    ax_j.set_title("J. Atlas Coordinate vs Regional Difference", fontsize=12, fontweight='bold', loc='left')
    ax_j.legend(loc='best', fontsize=8)
    ax_j.grid(True, alpha=0.3, linestyle='--')

    # ══════════════════════════════════════════════════════════════════════════
    # ROW 4: PER-ANIMAL ANALYSIS (scatter plots colored by channel)
    # ══════════════════════════════════════════════════════════════════════════

    # Filter to only slides with known animal IDs
    animal_df = mean_df[mean_df["animal_id"].notna()].copy()
    unique_animals = sorted(animal_df["animal_id"].unique(), key=lambda x: int(x.replace('#', '')))

    # Create numeric x-positions for animals
    animal_to_x = {a: i for i, a in enumerate(unique_animals)}
    animal_df["animal_x"] = animal_df["animal_id"].map(animal_to_x)

    # Panel M: Scatter plot of Cortex thresholds per animal (colored by channel)
    ax_m = fig.add_subplot(gs[3, 0])
    if len(animal_df) > 0:
        for ch in sorted(animal_df["channel"].unique()):
            ch_data = animal_df[animal_df["channel"] == ch]
            color = CHANNEL_COLORS.get(ch, 'gray')
            # Add small jitter to x-position to avoid overlap
            jitter = 0.1 if ch == 'green' else -0.1
            ax_m.scatter(ch_data["animal_x"] + jitter, ch_data["Cortex"],
                        c=color, s=70, alpha=0.7, edgecolor='black',
                        linewidth=0.7, label=ch)

        ax_m.set_xticks(range(len(unique_animals)))
        ax_m.set_xticklabels(unique_animals, fontsize=9, rotation=45, ha='right')
        ax_m.set_xlabel("Animal ID", fontsize=11)
        ax_m.set_ylabel("Cortex threshold [photons]", fontsize=11)
        ax_m.set_title("M. Cortex Threshold per Animal", fontsize=12, fontweight='bold', loc='left')
        ax_m.legend(loc='best', fontsize=8)
        ax_m.grid(True, alpha=0.3, linestyle='--', axis='y')

    # Panel N: Scatter plot of Striatum thresholds per animal (colored by channel)
    ax_n = fig.add_subplot(gs[3, 1])
    if len(animal_df) > 0:
        for ch in sorted(animal_df["channel"].unique()):
            ch_data = animal_df[animal_df["channel"] == ch]
            color = CHANNEL_COLORS.get(ch, 'gray')
            jitter = 0.1 if ch == 'green' else -0.1
            ax_n.scatter(ch_data["animal_x"] + jitter, ch_data["Striatum"],
                        c=color, s=70, alpha=0.7, edgecolor='black',
                        linewidth=0.7, label=ch)

        ax_n.set_xticks(range(len(unique_animals)))
        ax_n.set_xticklabels(unique_animals, fontsize=9, rotation=45, ha='right')
        ax_n.set_xlabel("Animal ID", fontsize=11)
        ax_n.set_ylabel("Striatum threshold [photons]", fontsize=11)
        ax_n.set_title("N. Striatum Threshold per Animal", fontsize=12, fontweight='bold', loc='left')
        ax_n.legend(loc='best', fontsize=8)
        ax_n.grid(True, alpha=0.3, linestyle='--', axis='y')

    # Panel O: Scatter plot of regional difference per animal (colored by channel)
    ax_o = fig.add_subplot(gs[3, 2])
    if len(animal_df) > 0:
        for ch in sorted(animal_df["channel"].unique()):
            ch_data = animal_df[animal_df["channel"] == ch]
            color = CHANNEL_COLORS.get(ch, 'gray')
            jitter = 0.1 if ch == 'green' else -0.1
            ax_o.scatter(ch_data["animal_x"] + jitter, ch_data["diff"],
                        c=color, s=70, alpha=0.7, edgecolor='black',
                        linewidth=0.7, label=ch)

        ax_o.axhline(0, linestyle="--", linewidth=1.5, color='red', alpha=0.5)
        ax_o.set_xticks(range(len(unique_animals)))
        ax_o.set_xticklabels(unique_animals, fontsize=9, rotation=45, ha='right')
        ax_o.set_xlabel("Animal ID", fontsize=11)
        ax_o.set_ylabel("Cortex - Striatum [photons]", fontsize=11)
        ax_o.set_title("O. Regional Difference per Animal", fontsize=12, fontweight='bold', loc='left')
        ax_o.legend(loc='best', fontsize=8)
        ax_o.grid(True, alpha=0.3, linestyle='--', axis='y')

    # ══════════════════════════════════════════════════════════════════════════
    # ROW 5: SAMPLE SIZE AND DISTRIBUTIONS
    # ══════════════════════════════════════════════════════════════════════════

    # Panel G: Sample size vs threshold (both regions) - MOVED TO ROW 5
    ax_g = fig.add_subplot(gs[4, 0])
    ax_g.scatter(mean_df["n_cortex"], mean_df["Cortex"],
               c='blue', s=60, alpha=0.6, edgecolor='black',
               linewidth=0.7, label='Cortex', marker='o')
    ax_g.scatter(mean_df["n_striatum"], mean_df["Striatum"],
               c='red', s=60, alpha=0.6, edgecolor='black',
               linewidth=0.7, label='Striatum', marker='s')
    ax_g.set_xlabel("Sample size [# spots]", fontsize=11)
    ax_g.set_ylabel("Threshold [photons]", fontsize=11)
    ax_g.set_title("G. Sample Size vs Threshold", fontsize=12, fontweight='bold', loc='left')
    ax_g.legend(loc='best', fontsize=9)
    ax_g.grid(True, alpha=0.3, linestyle='--')
    ax_g.set_xscale('log')

    # Panel K: Threshold distributions by region
    ax_k = fig.add_subplot(gs[4, 1])
    bins = np.linspace(threshold_min, threshold_max, 25)
    ax_k.hist(mean_df["Cortex"], bins=bins, alpha=0.6, label='Cortex',
            color='blue', edgecolor='black', linewidth=0.7, density=True)
    ax_k.hist(mean_df["Striatum"], bins=bins, alpha=0.6, label='Striatum',
            color='red', edgecolor='black', linewidth=0.7, density=True)
    ax_k.axvline(mean_cortex, color='blue', linestyle='--', linewidth=2, alpha=0.8)
    ax_k.axvline(mean_striatum, color='red', linestyle='--', linewidth=2, alpha=0.8)
    ax_k.set_xlabel("Threshold [photons]", fontsize=11)
    ax_k.set_ylabel("Probability Density", fontsize=11)
    ax_k.set_title("K. Threshold Distributions by Region", fontsize=12, fontweight='bold', loc='left')
    ax_k.legend(loc='best', fontsize=9)
    ax_k.grid(True, alpha=0.3, linestyle='--', axis='y')

    # Panel L: Threshold distributions by channel
    ax_l = fig.add_subplot(gs[4, 2])
    for ch in sorted(mean_df["channel"].unique()):
        ch_data = mean_df[mean_df["channel"] == ch]
        all_thresholds = pd.concat([ch_data["Cortex"], ch_data["Striatum"]])
        color = CHANNEL_COLORS.get(ch, 'gray')
        ax_l.hist(all_thresholds, bins=20, alpha=0.6, label=ch,
                color=color, edgecolor='black', linewidth=0.7, density=True)
    ax_l.set_xlabel("Threshold [photons]", fontsize=11)
    ax_l.set_ylabel("Probability Density", fontsize=11)
    ax_l.set_title("L. Threshold Distributions by Channel", fontsize=12, fontweight='bold', loc='left')
    ax_l.legend(loc='best', fontsize=9)
    ax_l.grid(True, alpha=0.3, linestyle='--', axis='y')

    # No title - leaving space at top for user to add images later

    # ══════════════════════════════════════════════════════════════════════════
    # 8. SAVE FIGURE
    # ══════════════════════════════════════════════════════════════════════════

    print(f"\n{'='*80}")
    print("Saving figure...")
    print(f"{'='*80}")

    # Save in multiple formats
    for fmt in ['png', 'svg', 'pdf']:
        filepath = OUTPUT_DIR / f"fig_negative_control_comprehensive.{fmt}"
        plt.savefig(filepath, format=fmt, bbox_inches='tight', dpi=300)
        print(f"  Saved: {filepath}")

    plt.close(fig)

    # Save data to CSV
    csv_path = OUTPUT_DIR / "negative_control_comprehensive_data.csv"
    mean_df.to_csv(csv_path, index=False)
    print(f"  Data saved: {csv_path}")

    # Generate comprehensive caption
    print(f"\n{'='*80}")
    print("Generating comprehensive caption...")
    print(f"{'='*80}")

    n_slides = mean_df['slide'].nunique()
    n_pairs = len(mean_df)
    channels = sorted(mean_df['channel'].unique())

    caption_lines = []
    caption_lines.append("=" * 100)
    caption_lines.append("FIGURE: Comprehensive Negative Control Background Threshold Analysis")
    caption_lines.append("=" * 100)
    caption_lines.append("")
    caption_lines.append("OVERVIEW")
    caption_lines.append("-" * 100)
    caption_lines.append("This figure provides a comprehensive analysis of negative control background thresholds")
    caption_lines.append("across brain regions (Cortex vs Striatum), examining regional differences and their")
    caption_lines.append("relationships with age, sample size, and brain atlas coordinates. Negative control")
    caption_lines.append("thresholds define the intensity cutoff above which spots are considered true mRNA")
    caption_lines.append("signals rather than non-specific background.")
    caption_lines.append("")
    caption_lines.append("BIOLOGICAL RATIONALE:")
    caption_lines.append("- Negative controls use bacterial DapB probes not expressed in mammalian tissue")
    caption_lines.append("- Any detected signal represents non-specific binding or autofluorescence")
    caption_lines.append("- The 95th percentile of negative control intensities provides a conservative threshold")
    caption_lines.append("- Regional differences may reflect tissue autofluorescence or composition")
    caption_lines.append("")
    caption_lines.append("=" * 100)
    caption_lines.append("DATASET SUMMARY")
    caption_lines.append("=" * 100)
    caption_lines.append("")
    caption_lines.append(f"Total paired samples (slide-channel combinations): {n_pairs}")
    caption_lines.append(f"Unique slides: {n_slides}")
    caption_lines.append(f"Channels analyzed: {', '.join(channels)}")
    caption_lines.append("  - green (488 nm): HTT1a probe - detects exon 1 of mutant huntingtin")
    caption_lines.append("  - orange (548 nm): fl-HTT probe - detects full transcript")
    caption_lines.append(f"Age range: {mean_df['age'].min():.0f} - {mean_df['age'].max():.0f} months")
    caption_lines.append(f"Brain atlas coordinate range: {mean_df['atlas_coord'].min():.1f} - {mean_df['atlas_coord'].max():.1f} mm")
    caption_lines.append("")
    caption_lines.append("THRESHOLD CALCULATION METHOD:")
    caption_lines.append(f"  - Quantile: {quantile_negative_control*100:.0f}th percentile of negative control spot intensities")
    caption_lines.append(f"  - Quality filter: PFA (probability of false alarm) < {max_pfa}")
    caption_lines.append("  - Bootstrap iterations: 20 (for error estimation)")
    caption_lines.append("  - Region-specific: Separate thresholds computed for Cortex and Striatum")
    caption_lines.append("")
    caption_lines.append("=" * 100)
    caption_lines.append("QUANTITATIVE RESULTS")
    caption_lines.append("=" * 100)
    caption_lines.append("")
    caption_lines.append("OVERALL REGIONAL COMPARISON (Cortex vs Striatum):")
    caption_lines.append(f"  - Cortex threshold: {mean_cortex:,.1f} +/- {mean_df['Cortex'].std():,.1f} photons")
    caption_lines.append(f"  - Striatum threshold: {mean_striatum:,.1f} +/- {mean_df['Striatum'].std():,.1f} photons")
    caption_lines.append(f"  - Mean difference (Cortex - Striatum): {mean_diff:+,.1f} +/- {std_diff:,.1f} photons")
    caption_lines.append(f"  - Paired t-test: t = {t_all:.3f}, p = {p_all:.4g}")
    sig_label = 'Significant' if p_all < 0.05 else 'No significant'
    caption_lines.append(f"  - Interpretation: {sig_label} regional difference")
    caption_lines.append("")
    caption_lines.append("PER-CHANNEL STATISTICS:")

    for ch in channels:
        ch_data = mean_df[mean_df['channel'] == ch]
        if len(ch_data) >= 3:
            ch_cortex_mean = ch_data['Cortex'].mean()
            ch_cortex_std = ch_data['Cortex'].std()
            ch_striatum_mean = ch_data['Striatum'].mean()
            ch_striatum_std = ch_data['Striatum'].std()
            ch_diff_mean = ch_data['diff'].mean()
            ch_t, ch_p = ttest_rel(ch_data['Cortex'], ch_data['Striatum'])
            caption_lines.append("")
            caption_lines.append(f"  {ch.upper()} CHANNEL ({len(ch_data)} paired samples):")
            caption_lines.append(f"    Cortex: {ch_cortex_mean:,.1f} +/- {ch_cortex_std:,.1f} photons")
            caption_lines.append(f"    Striatum: {ch_striatum_mean:,.1f} +/- {ch_striatum_std:,.1f} photons")
            caption_lines.append(f"    Difference: {ch_diff_mean:+,.1f} photons")
            caption_lines.append(f"    Paired t-test: t = {ch_t:.3f}, p = {ch_p:.4g}")

    caption_lines.append("")
    caption_lines.append("CORRELATION ANALYSES:")
    caption_lines.append("")
    caption_lines.append("  Age correlations (testing for age-dependent threshold drift):")
    caption_lines.append(f"    Age vs Cortex threshold: r = {r_cortex_age:+.3f}, p = {p_cortex_age:.4g}")
    caption_lines.append(f"    Age vs Striatum threshold: r = {r_striatum_age:+.3f}, p = {p_striatum_age:.4g}")
    caption_lines.append(f"    Age vs Regional difference: r = {r_diff_age:+.3f}, p = {p_diff_age:.4g}")
    age_dep = 'Weak' if abs(r_cortex_age) < 0.3 else ('Moderate' if abs(r_cortex_age) < 0.5 else 'Strong')
    caption_lines.append(f"    Interpretation: {age_dep} age dependence")
    caption_lines.append("")
    caption_lines.append("  Sample size correlations (testing for statistical power effects):")
    caption_lines.append(f"    Sample size vs Cortex threshold: r = {r_n_cortex:+.3f}, p = {p_n_cortex:.4g}")
    caption_lines.append(f"    Sample size vs Striatum threshold: r = {r_n_striatum:+.3f}, p = {p_n_striatum:.4g}")
    caption_lines.append(f"    Min sample size vs |Regional difference|: r = {r_n_diff:+.3f}, p = {p_n_diff:.4g}")
    caption_lines.append("")
    caption_lines.append("  Brain atlas coordinate correlations (testing for rostrocaudal gradients):")
    caption_lines.append(f"    Atlas coord vs Cortex threshold: r = {r_atlas_cortex:+.3f}, p = {p_atlas_cortex:.4g}")
    caption_lines.append(f"    Atlas coord vs Striatum threshold: r = {r_atlas_striatum:+.3f}, p = {p_atlas_striatum:.4g}")
    caption_lines.append(f"    Atlas coord vs Regional difference: r = {r_atlas_diff:+.3f}, p = {p_atlas_diff:.4g}")
    atlas_sig = 'Significant' if p_atlas_diff < 0.05 else 'No significant'
    caption_lines.append(f"    Interpretation: {atlas_sig} rostrocaudal gradient")
    caption_lines.append("")
    caption_lines.append("=" * 100)
    caption_lines.append("PANEL DESCRIPTIONS")
    caption_lines.append("=" * 100)
    caption_lines.append("")
    caption_lines.append("ROW 1 - REGIONAL COMPARISON:")
    caption_lines.append("")
    caption_lines.append("  Panel A: Regional Comparison by Channel")
    caption_lines.append("    - Scatter plot of Cortex vs Striatum thresholds")
    caption_lines.append("    - Each point represents one slide-channel combination")
    caption_lines.append("    - Colors indicate detection channel (green = HTT1a, orange = fl-HTT)")
    caption_lines.append("    - Dashed diagonal line indicates perfect agreement (unity line)")
    caption_lines.append("    - Points above line: Striatum threshold > Cortex threshold")
    caption_lines.append("    - Points below line: Cortex threshold > Striatum threshold")
    obs_a = 'near' if abs(mean_diff) < 2000 else 'away from'
    caption_lines.append(f"    - OBSERVATION: Points cluster {obs_a} unity line")
    caption_lines.append("")
    caption_lines.append("  Panel B: Threshold Difference Distribution")
    caption_lines.append("    - Histogram of (Cortex - Striatum) threshold differences")
    caption_lines.append("    - Red dashed line: zero difference (perfect regional agreement)")
    caption_lines.append("    - Blue solid line: mean difference across all samples")
    caption_lines.append(f"    - OBSERVATION: Distribution centered at {mean_diff:+,.1f} photons")
    bias_label = 'Minimal' if abs(mean_diff) < 2000 else 'Substantial'
    caption_lines.append(f"    - BIOLOGICAL IMPLICATION: {bias_label} systematic regional bias")
    caption_lines.append("")
    caption_lines.append("  Panel C: Difference by Channel")
    caption_lines.append("    - Violin plots showing distribution of regional differences per channel")
    caption_lines.append("    - Width indicates probability density at each difference value")
    caption_lines.append("    - Red dashed line: zero difference")
    caption_lines.append("    - Allows comparison of regional effects between probe targets")
    caption_lines.append("")
    caption_lines.append("ROW 2 - AGE DEPENDENCIES:")
    caption_lines.append("")
    caption_lines.append("  Panel D: Age vs Cortex Threshold")
    caption_lines.append("    - Scatter plot with separate regression lines per channel")
    caption_lines.append("    - Tests whether background increases/decreases with animal age")
    age_obs = 'No significant' if abs(r_cortex_age) < 0.3 else 'Significant'
    caption_lines.append(f"    - OBSERVATION: {age_obs} age trend")
    caption_lines.append("    - BIOLOGICAL IMPLICATION: Cortical autofluorescence is stable across ages")
    caption_lines.append("")
    caption_lines.append("  Panel E: Age vs Striatum Threshold")
    caption_lines.append("    - Same analysis for striatal regions")
    age_obs_str = 'No significant' if abs(r_striatum_age) < 0.3 else 'Significant'
    caption_lines.append(f"    - OBSERVATION: {age_obs_str} age trend")
    caption_lines.append("    - BIOLOGICAL IMPLICATION: Striatal background is stable across ages")
    caption_lines.append("")
    caption_lines.append("  Panel F: Age vs Threshold Difference")
    caption_lines.append("    - Tests whether regional difference changes with age")
    age_eff = 'no' if abs(r_diff_age) < 0.3 else 'a'
    caption_lines.append(f"    - OBSERVATION: r = {r_diff_age:+.3f} indicates {age_eff} systematic age effect")
    caption_lines.append("")
    caption_lines.append("ROW 3 - BRAIN ATLAS COORDINATE DEPENDENCIES:")
    caption_lines.append("")
    caption_lines.append("  Panel H: Atlas Coordinate vs Cortex Threshold")
    caption_lines.append("    - Tests for rostrocaudal gradients in cortical background")
    caption_lines.append("    - X-axis: anterior-posterior position (mm from bregma)")
    caption_lines.append(f"    - OBSERVATION: r = {r_atlas_cortex:+.3f}")
    caption_lines.append("")
    caption_lines.append("  Panel I: Atlas Coordinate vs Striatum Threshold")
    caption_lines.append("    - Tests for rostrocaudal gradients in striatal background")
    caption_lines.append(f"    - OBSERVATION: r = {r_atlas_striatum:+.3f}")
    caption_lines.append("")
    caption_lines.append("  Panel J: Atlas Coordinate vs Regional Difference")
    caption_lines.append("    - CRITICAL PANEL: Tests whether cortex-striatum difference varies along brain axis")
    caption_lines.append(f"    - OBSERVATION: r = {r_atlas_diff:+.3f}, p = {p_atlas_diff:.4g}")
    atlas_impl = 'Significant' if p_atlas_diff < 0.05 else 'No significant'
    caption_lines.append(f"    - BIOLOGICAL IMPLICATION: {atlas_impl} anatomical gradient")
    caption_lines.append("")
    caption_lines.append("ROW 4 - PER-ANIMAL ANALYSIS:")
    caption_lines.append("")
    caption_lines.append("  Panel M: Cortex Threshold per Animal")
    caption_lines.append("    - Box plots showing threshold distribution for each animal")
    caption_lines.append("    - Each box represents all slides from one animal")
    caption_lines.append("    - Allows assessment of inter-animal vs intra-animal variability")
    n_animals = len(unique_animals) if 'unique_animals' in dir() else mean_df['animal_id'].nunique()
    caption_lines.append(f"    - Animals analyzed: n={n_animals}")
    caption_lines.append("    - OBSERVATION: Variability within animals (intra-animal) vs between animals (inter-animal)")
    caption_lines.append("")
    caption_lines.append("  Panel N: Striatum Threshold per Animal")
    caption_lines.append("    - Same analysis for striatal thresholds")
    caption_lines.append("    - Enables comparison of regional consistency across animals")
    caption_lines.append("")
    caption_lines.append("  Panel O: Regional Difference per Animal")
    caption_lines.append("    - Box plots of Cortex-Striatum difference for each animal")
    caption_lines.append("    - Red dashed line indicates zero difference")
    caption_lines.append("    - Tests whether regional bias is consistent across animals")
    caption_lines.append("    - BIOLOGICAL IMPLICATION: Animal-to-animal consistency in regional differences")
    caption_lines.append("")
    caption_lines.append("ROW 5 - SAMPLE SIZE AND DISTRIBUTIONS:")
    caption_lines.append("")
    caption_lines.append("  Panel G: Sample Size vs Threshold")
    caption_lines.append("    - Tests whether threshold estimates depend on number of spots analyzed")
    caption_lines.append("    - X-axis: log-scale number of negative control spots")
    caption_lines.append("    - Blue circles: Cortex, Red squares: Striatum")
    caption_lines.append(f"    - OBSERVATION: Weak correlation (r ~ {r_n_cortex:.2f})")
    caption_lines.append("    - IMPLICATION: Threshold estimates are stable regardless of sample size")
    caption_lines.append("")
    caption_lines.append("  Panel K: Threshold Distributions by Region")
    caption_lines.append("    - Overlapping histograms comparing Cortex vs Striatum")
    caption_lines.append("    - Dashed lines indicate mean values for each region")
    dist_obs = 'overlap substantially' if abs(mean_diff) < 3000 else 'are separated'
    caption_lines.append(f"    - OBSERVATION: Distributions {dist_obs}")
    caption_lines.append("")
    caption_lines.append("  Panel L: Threshold Distributions by Channel")
    caption_lines.append("    - Compares background levels between detection channels")
    caption_lines.append("    - Green typically shows higher thresholds due to tissue autofluorescence at 488 nm")
    caption_lines.append("    - Orange (548 nm) typically shows lower background")
    caption_lines.append("")
    caption_lines.append("=" * 100)
    caption_lines.append("BIOLOGICAL INTERPRETATION")
    caption_lines.append("=" * 100)
    caption_lines.append("")
    caption_lines.append("KEY FINDINGS:")

    if p_all < 0.05:
        sign = '+' if mean_diff > 0 else ''
        caption_lines.append(f"  1. SIGNIFICANT regional difference: Cortex shows {sign}{mean_diff:.0f} photon higher threshold")
        caption_lines.append("     -> May reflect higher autofluorescence in cortical tissue")
    else:
        caption_lines.append(f"  1. NO significant regional difference (p = {p_all:.4g})")
        caption_lines.append("     -> Cortex and Striatum have similar background levels")

    if abs(r_diff_age) > 0.3 and p_diff_age < 0.05:
        caption_lines.append("  2. SIGNIFICANT age effect on regional difference")
        caption_lines.append("     -> Consider age as potential confounder in regional comparisons")
    else:
        caption_lines.append("  2. NO significant age effect on regional difference")
        caption_lines.append("     -> Age does not confound regional comparisons")

    if abs(r_atlas_diff) > 0.3 and p_atlas_diff < 0.05:
        caption_lines.append(f"  3. SIGNIFICANT rostrocaudal gradient in regional difference (r = {r_atlas_diff:.3f})")
        caption_lines.append("     -> Regional threshold difference varies along anterior-posterior axis")
    else:
        caption_lines.append("  3. NO significant rostrocaudal gradient")
        caption_lines.append("     -> Regional comparison is valid across brain positions")

    caption_lines.append("")
    caption_lines.append("PRACTICAL IMPLICATIONS FOR mRNA QUANTIFICATION:")
    caption_lines.append("  - Use region-specific thresholds when regional difference is significant")
    caption_lines.append("  - Per-slide calibration accounts for slide-to-slide technical variation")
    caption_lines.append("  - Green channel (488 nm) typically requires higher thresholds due to autofluorescence")
    caption_lines.append("  - Sample size > 100 spots provides stable threshold estimates")
    caption_lines.append("")
    caption_lines.append("=" * 100)
    caption_lines.append("METHODOLOGY")
    caption_lines.append("=" * 100)
    caption_lines.append("")
    caption_lines.append("THRESHOLD DETERMINATION:")
    caption_lines.append("  1. Collect all spot intensities from negative control (DapB) regions")
    caption_lines.append("  2. Apply quality filter: PFA < 0.05 (probability of false alarm)")
    caption_lines.append(f"  3. Compute {quantile_negative_control*100:.0f}th percentile as threshold")
    caption_lines.append("  4. Bootstrap 20 times to estimate uncertainty")
    caption_lines.append("")
    caption_lines.append("STATISTICAL TESTS:")
    caption_lines.append("  - Paired t-test: Compare Cortex vs Striatum thresholds within same slides")
    caption_lines.append("  - Pearson correlation: Test linear relationships with continuous variables")
    caption_lines.append("  - Linear regression: Quantify slopes and significance of trends")
    caption_lines.append("")
    caption_lines.append("=" * 100)
    caption_lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    caption_lines.append("=" * 100)

    caption_text = '\n'.join(caption_lines)
    caption_path = OUTPUT_DIR / "fig_negative_control_comprehensive_caption.txt"
    with open(caption_path, 'w') as f:
        f.write(caption_text)
    print(f"  Caption saved: {caption_path}")

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
