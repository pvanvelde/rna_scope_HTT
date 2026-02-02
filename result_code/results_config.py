"""
Global configuration parameters for RNA Scope analysis.

This file contains all shared parameters used across analysis scripts.
Modify these values here to change them across all scripts that import this config.
"""

# ══════════════════════════════════════════════════════════════════════════
# IMAGING PARAMETERS
# ══════════════════════════════════════════════════════════════════════════

# Pixel size in nm
PIXELSIZE = 162.5  # nm

# Z-slice depth in nm
SLICE_DEPTH = 500  # nm

# Voxel size in μm³
VOXEL_SIZE = 0.1625 * 0.1625 * 0.5  # μm³ per voxel

# Mean nuclear volume for cell count estimation
MEAN_NUCLEAR_VOLUME = 716  # μm³




# ══════════════════════════════════════════════════════════════════════════
# BEAD PSF CALIBRATION VALUES
# ══════════════════════════════════════════════════════════════════════════
# These are from fluorescent microsphere calibration - DO NOT CHANGE
# Green channel bead PSF: 185±12, 187±13, 573±45 nm

BEAD_PSF_X = 185.0  # nm
BEAD_PSF_Y = 187.0  # nm
BEAD_PSF_Z = 573.0  # nm

# Lower bounds for sigma filtering (80% of bead PSF)
# Spots smaller than this are likely noise/artifacts
SIGMA_LOWER_FRACTION = 0.8
SIGMA_X_LOWER = BEAD_PSF_X * SIGMA_LOWER_FRACTION  # 148.0 nm
SIGMA_Y_LOWER = BEAD_PSF_Y * SIGMA_LOWER_FRACTION  # 149.6 nm
SIGMA_Z_LOWER = BEAD_PSF_Z * SIGMA_LOWER_FRACTION  # 458.4 nm

# Upper bounds for sigma filtering (from break_sigma analysis)
# Derived from biphasic breakpoints in CHANNEL_PARAMS (converted to nm)
# Green: [1.7897, 1.7914, 1.3973] pixels -> 290.8, 291.1, 698.7 nm
# Orange: [1.7542, 1.7556, 1.4014] pixels -> 285.1, 285.3, 700.7 nm
# Using the larger of the two channels as upper bound
SIGMA_X_UPPER = 291.0  # nm (from green break_sigma[0] * 162.5)
SIGMA_Y_UPPER = 291.0  # nm (from green break_sigma[1] * 162.5)
SIGMA_Z_UPPER = 701.0  # nm (from orange break_sigma[2] * 500)

# Convenience tuples for xlim in plots
SIGMA_X_XLIM = (SIGMA_X_LOWER, SIGMA_X_UPPER)
SIGMA_Y_XLIM = (SIGMA_Y_LOWER, SIGMA_Y_UPPER)
SIGMA_Z_XLIM = (SIGMA_Z_LOWER, SIGMA_Z_UPPER)


# ══════════════════════════════════════════════════════════════════════════
# THRESHOLD COMPUTATION PARAMETERS
# ══════════════════════════════════════════════════════════════════════════

# Maximum p-value for spot detection (PFA filter)
MAX_PFA = 0.05

# Quantile of negative control for threshold determination
QUANTILE_NEGATIVE_CONTROL = 0.95

# Coefficient of Variance (CV) threshold for cluster quality filtering
# Clusters with CV < CV_THRESHOLD are excluded (low CV = poor quality)
# CV >= CV_THRESHOLD means good quality cluster
CV_THRESHOLD = 0.5

# Number of bootstrap samples for threshold estimation
N_BOOTSTRAP = 1

# Use region-specific thresholds (False = slide-wide thresholds)
USE_REGION_THRESHOLDS = False

# Use final filter for spot selection
USE_FINAL_FILTER = True


# ══════════════════════════════════════════════════════════════════════════
# QUALITY CONTROL PARAMETERS
# ══════════════════════════════════════════════════════════════════════════

# Minimum nuclei threshold for FOV filtering
# FOVs with fewer cells are excluded to avoid artifacts from poor DAPI segmentation
MIN_NUCLEI_THRESHOLD = 40.0

# Minimum number of spots for peak intensity calculation
MIN_SPOTS_FOR_PEAK = 50

# Slides to exclude from analysis (technical failures identified in QC)
# These slides show abnormally low UBC expression (100-1000x below normal) in BOTH
# cortex and striatum, indicating technical failures (poor hybridization, tissue damage, etc.)
# See: result_code/positive_control_analysis/outlier_analysis/ubc_outlier_detection.png
# and: result_code/positive_control_analysis/outlier_analysis/recommended_exclusion_list.txt
EXCLUDED_SLIDES = [
    'm1a2',  # UBC: 0.012 (Cortex) / 0.022 (Striatum) mRNA/cell - extreme technical failure
    'm1b5',  # UBC: 0.203 (Cortex) / 0.790 (Striatum) mRNA/cell - technical failure
    'm2b5',
    'm2b2',
    'm2a7',
    'm2a1',
    'm2b4',
    'm2b5',
    'm3b4',
    'm1a1',
    'm1b2',
    'm1b3',
    'm1b4',
    'm3b5'

]

# This maps individual slide names to consistent mouse labels
# Sorted by age (numeric), then by mouse number within each age group
SLIDE_LABEL_MAP_Q111 = {
    # 2 month old mice (#1-#3)
    'm2a2': '#1.1',   # Q111 2mo #1 UNT Late STR
    'm2a7': '#1.2',   # Q111 2mo #1 UNT Mid STR
    'm2a4': '#2.1',   # Q111 2mo #2 UNT Mid STR
    'm2b2': '#2.2',   # Q111 2mo #2 UNT Early STR
    'm3b2': '#2.3',   # Q111 2mo #2 UNT early STR
    'm1b1': '#3.1',   # Q111 2mo #3 UNT Mid STR
    'm3a1': '#3.2',   # Q111 2mo #3 UNT Early STR
    'm3a3': '#3.3',   # Q111 2mo #3 UNT Late STR
    # 6 month old mice (#4-#6)
    'm1a1': '#4.1',   # Q111 6mo #1 NTC Early STR
    'm3a2': '#4.2',   # Q111 6mo #1 NTC Late STR
    'm1a4': '#5.1',   # Q111 6mo #2 NTC Mid STR
    'm1a5': '#5.2',   # Q111 6mo #2 NTC Late STR
    'm2b4': '#5.3',   # Q111 6mo #2 NTC Mid STR
    'm2b5': '#5.4',   # Q111 6mo #2 NTC Late STR (excluded but kept for consistent numbering)
    'm3a5': '#6.1',   # Q111 6mo #3 NTC Mid STR
    'm3b3': '#6.2',   # Q111 6mo #3 NTC Late STR
    'm3b5': '#6.3',   # Q111 6mo #3 NTC Early STR
    # 12 month old mice (#7-#9)
    'm2a3': '#7.1',   # Q111 12mo #1 aCSF Early STR
    'm2a8': '#7.2',   # Q111 12mo #1 aCSF Late STR
    'm2b7': '#7.3',   # Q111 12mo #1 aCSF Mid STR
    'm2b1': '#8.1',   # Q111 12mo #2 aCSF Late STR
    'm3b4': '#8.2',   # Q111 12mo #2 aCSF Early STR
    'm1b3': '#9.1',   # Q111 12mo #3 aCSF Mid STR
    'm1b4': '#9.2',   # Q111 12mo #3 aCSF Late STR
}

SLIDE_LABEL_MAP_WT = {
    # 2 month old mice (#1)
    'm1b2': '#1',   # WT 2mo #2 UNT Late STR
    'm3a4': '#1',   # WT 2mo #2 UNT Early STR
    # 12 month old mice (#2)
    'm2a6': '#2',   # WT 12mo #3 aCSF Mid STR
    'm2b6': '#2',   # WT 12mo #3 aCSF Late STR
}


# ══════════════════════════════════════════════════════════════════════════
# DATA FILE PATHS
# ══════════════════════════════════════════════════════════════════════════

from pathlib import Path

# Base directory for results (relative to this config file)
RESULTS_BASE_DIR = Path(__file__).parent

# HDF5 data files
# H5_FILE_PATH_EXPERIMENTAL = '/home/pieter/development/rna_scope_data/global_merged_20251109_133004.h5'
H5_FILE_PATH_EXPERIMENTAL = '/home/grunwaldlab/UMass Medical School Dropbox/Pieter Fop Van Velde/rna_scope_data_v3_including_cv/global_merged_20251216_060124.h5'
# H5_FILE_PATH_EXPERIMENTAL = '/home/pieter/Documents/Slide M2 - A3_results.h5'

H5_FILE_PATH_BEAD = '/home/grunwaldlab/UMass Medical School Dropbox/Pieter Fop Van Velde/rna_scope_data_v2/rna_scope_data_beads/global_merged_20251114_213336.h5'

# Legacy: single file path (points to experimental by default)
H5_FILE_PATH = H5_FILE_PATH_EXPERIMENTAL

# File paths as list (for scripts that iterate over multiple files)
H5_FILE_PATHS_EXPERIMENTAL = [H5_FILE_PATH_EXPERIMENTAL]
H5_FILE_PATHS_BEAD = [H5_FILE_PATH_BEAD]

# CSV summary files
# SUMMARY_CSV_PATH_EXPERIMENTAL = '/home/pieter/development/rna_scope_data/global_merged_20251109_133004.csv'
SUMMARY_CSV_PATH_EXPERIMENTAL = '/home/grunwaldlab/UMass Medical School Dropbox/Pieter Fop Van Velde/rna_scope_data_v3_including_cv/global_merged_20251216_060124.csv'
# SUMMARY_CSV_PATH_EXPERIMENTAL = '/home/pieter/Documents/Slide M2 - A3_summary.csv'

SUMMARY_CSV_PATH_BEAD = '/home/grunwaldlab/UMass Medical School Dropbox/Pieter Fop Van Velde/rna_scope_data_v2/rna_scope_data_beads/global_merged_20251114_213336.csv'

# CSV summary file paths as list (for scripts that iterate over multiple files)
SUMMARY_CSV_PATHS_EXPERIMENTAL = [SUMMARY_CSV_PATH_EXPERIMENTAL]
SUMMARY_CSV_PATHS_BEAD = [SUMMARY_CSV_PATH_BEAD]


# ══════════════════════════════════════════════════════════════════════════
# OUTPUT DIRECTORIES
# All generated outputs go to draft_figures/output/ for easy cleanup
# ══════════════════════════════════════════════════════════════════════════

# Base output directory for all draft figures
DRAFT_FIGURES_OUTPUT = RESULTS_BASE_DIR / 'draft_figures' / 'output'

# Specific output directories (all under draft_figures/output/)
OUTPUT_DIR_NEGATIVE_CONTROL = DRAFT_FIGURES_OUTPUT / 'negative_control_comprehensive'
OUTPUT_DIR_POSITIVE_CONTROL_CORRELATION = DRAFT_FIGURES_OUTPUT / 'positive_control_vs_experimental'
OUTPUT_DIR_NORMALIZED_PLOTS = DRAFT_FIGURES_OUTPUT / 'filtering_figures'
OUTPUT_DIR_CLUSTER_COMPOSITION = DRAFT_FIGURES_OUTPUT / 'cluster_composition_analysis'
OUTPUT_DIR_CROSS_CHANNEL = DRAFT_FIGURES_OUTPUT / 'cross_channel_correlation'
OUTPUT_DIR_OUTLIER_TRACING = DRAFT_FIGURES_OUTPUT / 'outlier_tracing_analysis'
OUTPUT_DIR_INTENSITY_NORMALIZATION = DRAFT_FIGURES_OUTPUT / 'intensity_normalization_analysis'
OUTPUT_DIR_CLUSTER_EXTREME = DRAFT_FIGURES_OUTPUT / 'cluster_properties_extreme_vs_normal'
OUTPUT_DIR_POSITIVE_CONTROL = DRAFT_FIGURES_OUTPUT / 'positive_control_comprehensive'
OUTPUT_DIR_COMPREHENSIVE = DRAFT_FIGURES_OUTPUT / 'expression_analysis_q111'
OUTPUT_DIR_BEAD_PSF = RESULTS_BASE_DIR / 'bead_psf_plots'  # Keep bead PSF separate (calibration data)


# ══════════════════════════════════════════════════════════════════════════
# CHANNEL DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════

# Channels to extract from data
CHANNELS = ['blue', 'green', 'orange']

# Channel-specific parameters for spot detection and filtering
CHANNEL_PARAMS = {
    "green": {
        "channel_index": 1,
        "mode_name": "mHTT1a",
        "min_size": 100,
        "max_size": 100000,
        "detect_and_fit": True,
        "detect_labels": True,
        "intensity_threshold": 5,
        "sigma": [1.6161, 1.6156, 1.3325],
        "break_sigma": [1.7897, 1.7914, 1.3973],
    },
    "orange": {
        "channel_index": 2,
        "mode_name": "full length mHTT",
        "min_size": 100,
        "max_size": 100000,
        "detect_and_fit": True,
        "detect_labels": True,
        "intensity_threshold": 5,
        "sigma": [1.5307, 1.5258, 1.3292],
        "break_sigma": [1.7542, 1.7556, 1.4014],
    },
}

# Channels to analyze (excluding DAPI/blue)
CHANNELS_TO_ANALYZE = ['green', 'orange']

# Channel labels for experimental data
CHANNEL_LABELS_EXPERIMENTAL = {
    'green': 'mHTT1a',
    'orange': 'full length mHTT'
}

# Channel labels for positive controls
CHANNEL_LABELS_POSITIVE_CONTROL = {
    'green': 'POLR2A (low)',
    'orange': 'UBC (high)'
}

# Channel colors for plotting
CHANNEL_COLORS = {
    'green': 'green',
    'orange': 'orange',
    'blue': 'blue'
}


# ══════════════════════════════════════════════════════════════════════════
# REGION DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════

# Striatum subregions
STRIATUM_SUBREGIONS = [
    "Striatum - lower left",
    "Striatum - lower right",
    "Striatum - upper left",
    "Striatum - upper right",
]

# Cortex subregions
CORTEX_SUBREGIONS = [
    "Cortex - Piriform area",
    "Cortex - Primary and secondary motor areas",
    "Cortex - Primary somatosensory (mouth, upper limb)",
    "Cortex - Supplemental/primary somatosensory (nose)",
    "Cortex - Visceral/gustatory/agranular areas",
]


# ══════════════════════════════════════════════════════════════════════════
# SAMPLE/METADATA FIELD NAMES
# ══════════════════════════════════════════════════════════════════════════

# Slide name field
SLIDE_FIELD = 'metadata_sample_slide_name_std'

# Field identifying negative controls
NEGATIVE_CONTROL_FIELD = 'Negative control'

# Field identifying experimental samples
EXPERIMENTAL_FIELD = 'ExperimentalQ111 - 488mHT - 548mHTa - 647Darp'

# Field identifying positive controls
POSITIVE_CONTROL_FIELD = 'Positive control'


# ══════════════════════════════════════════════════════════════════════════
# FIELDS TO EXTRACT FROM HDF5
# ══════════════════════════════════════════════════════════════════════════

# Standard fields to extract for most analyses
STANDARD_FIELDS_TO_EXTRACT = [
    'spots_sigma_var.params_raw',
    'spots.params_raw',
    'cluster_intensities',
    'num_cells',
    'label_sizes',
    'metadata_sample.Age',
    'spots.final_filter',
    'metadata_sample.Brain_Atlas_coordinates'
]


# ══════════════════════════════════════════════════════════════════════════
# PLOTTING PARAMETERS
# ══════════════════════════════════════════════════════════════════════════

# Default figure DPI
FIGURE_DPI = 300

# Default figure format
FIGURE_FORMAT = 'svg'

# Plot style
PLOT_STYLE = 'science'

# Use LaTeX for text rendering
USE_LATEX = False


# ══════════════════════════════════════════════════════════════════════════
# STATISTICAL PARAMETERS
# ══════════════════════════════════════════════════════════════════════════

# Significance levels
ALPHA_LEVEL = 0.05
ALPHA_BONFERRONI = 0.05  # Will be adjusted based on number of comparisons

# Minimum sample size for statistical tests
MIN_SAMPLE_SIZE_FOR_TEST = 3


# ══════════════════════════════════════════════════════════════════════════
# KDE PARAMETERS
# ══════════════════════════════════════════════════════════════════════════

# Bandwidth method for KDE (kernel density estimation)
KDE_BANDWIDTH_METHOD = 'scott'  # Options: 'scott', 'silverman', or numeric value

# Number of points to sample KDE for peak detection
KDE_N_POINTS = 1000

# Percentiles for KDE range
KDE_PERCENTILE_LOW = 1
KDE_PERCENTILE_HIGH = 99


# ══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════

def get_channel_label(channel, probe_set='experimental'):
    """
    Get the appropriate channel label based on probe set.

    Parameters
    ----------
    channel : str
        Channel name ('green' or 'orange')
    probe_set : str
        Type of probe set ('experimental' or 'positive_control')

    Returns
    -------
    str
        Channel label
    """
    if probe_set.lower() in ['positive_control', 'positive control']:
        return CHANNEL_LABELS_POSITIVE_CONTROL.get(channel, channel)
    else:
        return CHANNEL_LABELS_EXPERIMENTAL.get(channel, channel)


def get_voxels_per_cell():
    """
    Calculate voxels per cell based on mean nuclear volume.

    Returns
    -------
    float
        Number of voxels per cell
    """
    return MEAN_NUCLEAR_VOLUME / VOXEL_SIZE


def print_config_summary():
    """Print a summary of the current configuration."""
    print("="*70)
    print("RNA SCOPE ANALYSIS CONFIGURATION")
    print("="*70)
    print("\nImaging Parameters:")
    print(f"  Pixel size: {PIXELSIZE} nm")
    print(f"  Slice depth: {SLICE_DEPTH} nm")
    print(f"  Voxel size: {VOXEL_SIZE:.6f} μm³")
    print(f"  Mean nuclear volume: {MEAN_NUCLEAR_VOLUME} μm³")

    print("\nThreshold Parameters:")
    print(f"  Max PFA: {MAX_PFA}")
    print(f"  Negative control quantile: {QUANTILE_NEGATIVE_CONTROL}")
    print(f"  Bootstrap samples: {N_BOOTSTRAP}")

    print("\nQuality Control:")
    print(f"  Min nuclei threshold: {MIN_NUCLEI_THRESHOLD}")
    print(f"  Min spots for peak: {MIN_SPOTS_FOR_PEAK}")
    print(f"  Excluded slides: {EXCLUDED_SLIDES} (n={len(EXCLUDED_SLIDES)})")

    print("\nChannels:")
    print(f"  All channels: {CHANNELS}")
    print(f"  Analysis channels: {CHANNELS_TO_ANALYZE}")

    print("="*70)


if __name__ == "__main__":
    # Print configuration when run directly
    print_config_summary()
