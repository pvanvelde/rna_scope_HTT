"""
Configuration for Final Figures

This module defines all styling parameters for publication-quality figures.
Import this config in all figure scripts to ensure consistent styling.

Usage:
    from figure_config import FigureConfig, apply_figure_style
    apply_figure_style()  # Apply rcParams globally
"""

import matplotlib.pyplot as plt
import matplotlib as mpl


class FigureConfig:
    """Central configuration for all figure parameters."""

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE AND PANEL DIMENSIONS
    # ══════════════════════════════════════════════════════════════════════════

    # Standard journal widths (in mm, converted to inches for matplotlib)
    PAGE_WIDTH_FULL_MM = 180       # Full page width (typical for most journals)
    PAGE_WIDTH_SINGLE_MM = 85      # Single column width
    PAGE_WIDTH_1_5_COL_MM = 114    # 1.5 column width

    # Convert to inches (matplotlib default unit)
    PAGE_WIDTH_FULL = PAGE_WIDTH_FULL_MM / 25.4      # ~7.09 inches
    PAGE_WIDTH_SINGLE = PAGE_WIDTH_SINGLE_MM / 25.4  # ~3.35 inches
    PAGE_WIDTH_1_5_COL = PAGE_WIDTH_1_5_COL_MM / 25.4  # ~4.49 inches

    # Default figure width (full page)
    FIGURE_WIDTH = PAGE_WIDTH_FULL

    # ══════════════════════════════════════════════════════════════════════════
    # FONTS
    # ══════════════════════════════════════════════════════════════════════════

    # Font family - Arial/Helvetica is standard for most journals
    FONT_FAMILY = 'sans-serif'
    FONT_SANS_SERIF = ['Arial', 'Helvetica', 'DejaVu Sans']

    # Font sizes (in points)
    FONT_SIZE_BASE = 7           # Base font size
    FONT_SIZE_AXIS_LABEL = 8     # Axis labels (x, y labels)
    FONT_SIZE_AXIS_TICK = 7      # Tick labels
    FONT_SIZE_TITLE = 9          # Subplot titles
    FONT_SIZE_LEGEND = 7         # Legend text
    FONT_SIZE_ANNOTATION = 6     # Small annotations

    # Panel labels (A, B, C, etc.)
    FONT_SIZE_PANEL_LABEL = 12   # Panel labels
    FONT_WEIGHT_PANEL_LABEL = 'bold'
    PANEL_LABEL_OFFSET_X = -0.12  # Offset from axes (in axes fraction)
    PANEL_LABEL_OFFSET_Y = 1.05   # Offset from axes (in axes fraction)

    # ══════════════════════════════════════════════════════════════════════════
    # LINE AND MARKER PROPERTIES
    # ══════════════════════════════════════════════════════════════════════════

    LINE_WIDTH = 1.0             # Default line width
    LINE_WIDTH_THIN = 0.5        # Thin lines (e.g., grid)
    LINE_WIDTH_THICK = 1.5       # Thick lines (emphasis)
    LINE_WIDTH_AXES = 0.8        # Axes spines

    MARKER_SIZE = 4              # Default marker size
    MARKER_SIZE_SMALL = 2        # Small markers
    MARKER_SIZE_LARGE = 6        # Large markers

    # ══════════════════════════════════════════════════════════════════════════
    # COLORS
    # ══════════════════════════════════════════════════════════════════════════

    # Primary color scheme (colorblind-friendly)
    COLOR_Q111_MHTT1A = '#2ecc71'    # Green - Q111 mHTT1a
    COLOR_Q111_FULL = '#f39c12'      # Orange - Q111 full-length
    COLOR_WT_MHTT1A = '#3498db'      # Blue - WT mHTT1a
    COLOR_WT_FULL = '#9b59b6'        # Purple - WT full-length

    # Neutral colors
    COLOR_BLACK = '#2c3e50'
    COLOR_GRAY_DARK = '#7f8c8d'
    COLOR_GRAY_LIGHT = '#bdc3c7'
    COLOR_WHITE = '#ffffff'

    # Statistical significance
    COLOR_SIGNIFICANT = '#e74c3c'    # Red for significant
    COLOR_NOT_SIGNIFICANT = '#95a5a6'  # Gray for not significant

    # ══════════════════════════════════════════════════════════════════════════
    # EXPORT SETTINGS
    # ══════════════════════════════════════════════════════════════════════════

    DPI = 300                    # DPI for raster formats
    DPI_DISPLAY = 150            # DPI for screen display

    # Output formats
    FORMAT_VECTOR = 'svg'        # Primary vector format
    FORMAT_RASTER = 'png'        # Primary raster format
    FORMAT_PRINT = 'pdf'         # For printing/submission

    # SVG settings
    SVG_FONTTYPE = 'none'        # 'none' = text as text, 'path' = text as paths

    # ══════════════════════════════════════════════════════════════════════════
    # SPACING AND LAYOUT
    # ══════════════════════════════════════════════════════════════════════════

    # Subplot spacing (as fraction of figure)
    SUBPLOT_LEFT = 0.08
    SUBPLOT_RIGHT = 0.97
    SUBPLOT_BOTTOM = 0.08
    SUBPLOT_TOP = 0.95
    SUBPLOT_WSPACE = 0.30        # Horizontal space between subplots
    SUBPLOT_HSPACE = 0.35        # Vertical space between subplots

    # ══════════════════════════════════════════════════════════════════════════
    # STATISTICAL ANNOTATIONS
    # ══════════════════════════════════════════════════════════════════════════

    SIGNIFICANCE_LEVELS = {
        0.001: '***',
        0.01: '**',
        0.05: '*',
        1.0: 'ns'
    }


def apply_figure_style():
    """Apply the figure style configuration to matplotlib rcParams."""

    cfg = FigureConfig

    # Font settings
    plt.rcParams['font.family'] = cfg.FONT_FAMILY
    plt.rcParams['font.sans-serif'] = cfg.FONT_SANS_SERIF
    plt.rcParams['font.size'] = cfg.FONT_SIZE_BASE

    # Axis settings
    plt.rcParams['axes.labelsize'] = cfg.FONT_SIZE_AXIS_LABEL
    plt.rcParams['axes.titlesize'] = cfg.FONT_SIZE_TITLE
    plt.rcParams['axes.linewidth'] = cfg.LINE_WIDTH_AXES
    plt.rcParams['axes.labelweight'] = 'normal'

    # Tick settings
    plt.rcParams['xtick.labelsize'] = cfg.FONT_SIZE_AXIS_TICK
    plt.rcParams['ytick.labelsize'] = cfg.FONT_SIZE_AXIS_TICK
    plt.rcParams['xtick.major.width'] = cfg.LINE_WIDTH_THIN
    plt.rcParams['ytick.major.width'] = cfg.LINE_WIDTH_THIN
    plt.rcParams['xtick.major.size'] = 3
    plt.rcParams['ytick.major.size'] = 3

    # Legend
    plt.rcParams['legend.fontsize'] = cfg.FONT_SIZE_LEGEND
    plt.rcParams['legend.frameon'] = False

    # Lines
    plt.rcParams['lines.linewidth'] = cfg.LINE_WIDTH
    plt.rcParams['lines.markersize'] = cfg.MARKER_SIZE

    # Figure settings
    plt.rcParams['figure.dpi'] = cfg.DPI_DISPLAY
    plt.rcParams['savefig.dpi'] = cfg.DPI
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['savefig.pad_inches'] = 0.02

    # SVG settings (text as text, not paths)
    plt.rcParams['svg.fonttype'] = cfg.SVG_FONTTYPE

    # PDF settings
    plt.rcParams['pdf.fonttype'] = 42  # TrueType fonts

    # Remove top and right spines by default
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False


def add_panel_label(ax, label, x_offset=None, y_offset=None):
    """
    Add a panel label (A, B, C, etc.) to an axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to add the label to
    label : str
        The label text (e.g., 'A', 'B', 'C')
    x_offset : float, optional
        X offset in axes fraction (default from config)
    y_offset : float, optional
        Y offset in axes fraction (default from config)
    """
    cfg = FigureConfig

    if x_offset is None:
        x_offset = cfg.PANEL_LABEL_OFFSET_X
    if y_offset is None:
        y_offset = cfg.PANEL_LABEL_OFFSET_Y

    ax.text(
        x_offset, y_offset, label,
        transform=ax.transAxes,
        fontsize=cfg.FONT_SIZE_PANEL_LABEL,
        fontweight=cfg.FONT_WEIGHT_PANEL_LABEL,
        va='top',
        ha='left'
    )


def get_significance_symbol(p_value):
    """
    Get significance symbol for a p-value.

    Parameters
    ----------
    p_value : float
        The p-value to convert

    Returns
    -------
    str
        Significance symbol ('***', '**', '*', or 'ns')
    """
    for threshold, symbol in sorted(FigureConfig.SIGNIFICANCE_LEVELS.items()):
        if p_value < threshold:
            return symbol
    return 'ns'


def save_figure(fig, filename, formats=None, output_dir=None):
    """
    Save figure in multiple formats.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to save
    filename : str
        Base filename (without extension)
    formats : list, optional
        List of formats to save (default: ['svg', 'png', 'pdf'])
    output_dir : Path, optional
        Output directory (default: current directory)
    """
    from pathlib import Path

    if formats is None:
        formats = [FigureConfig.FORMAT_VECTOR]

    if output_dir is None:
        output_dir = Path('.')
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    for fmt in formats:
        filepath = output_dir / f"{filename}.{fmt}"
        fig.savefig(filepath, format=fmt, dpi=FigureConfig.DPI)
        print(f"Saved: {filepath}")


# Color palette as a simple dictionary for easy access
COLORS = {
    'q111_mhtt1a': FigureConfig.COLOR_Q111_MHTT1A,
    'q111_full': FigureConfig.COLOR_Q111_FULL,
    'wt_mhtt1a': FigureConfig.COLOR_WT_MHTT1A,
    'wt_full': FigureConfig.COLOR_WT_FULL,
    'black': FigureConfig.COLOR_BLACK,
    'gray_dark': FigureConfig.COLOR_GRAY_DARK,
    'gray_light': FigureConfig.COLOR_GRAY_LIGHT,
    'significant': FigureConfig.COLOR_SIGNIFICANT,
    'not_significant': FigureConfig.COLOR_NOT_SIGNIFICANT,
}
