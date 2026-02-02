"""
Supplementary Figure Registry

This file defines all supplementary figures, their source scripts, output paths,
and dependencies. Used by the runner script to generate figures.

Based on v5_draft.tex supplementary figures section.
"""

from pathlib import Path

# Base directories
RESULT_CODE_DIR = Path(__file__).parent.parent
DRAFT_FIGURES_DIR = RESULT_CODE_DIR / "draft_figures"
OUTPUT_DIR = DRAFT_FIGURES_DIR / "output"

# ══════════════════════════════════════════════════════════════════════════════
# SUPPLEMENTARY FIGURE REGISTRY
# ══════════════════════════════════════════════════════════════════════════════
#
# Each entry contains:
#   - name: Short identifier (e.g., "S1")
#   - title: Descriptive title
#   - script: Path to the generating script (relative to result_code/)
#   - output: Expected output path (relative to result_code/)
#   - tex_path: Path used in the LaTeX document
#   - description: Brief description of the figure content
#   - dependencies: List of other figures or data this depends on
#
# ══════════════════════════════════════════════════════════════════════════════

SUPPLEMENTARY_FIGURES = {
    # Prerequisite data generation (not published figures, but needed by other figures)
    "DATA_EXPR": {
        "name": "DATA_EXPR",
        "title": "FOV-level expression data generation",
        "script": "draft_figures/fig_expression_analysis_q111.py",
        "output": "draft_figures/output/expression_analysis_q111/fov_level_data.csv",
        "tex_path": None,  # Not a published figure
        "description": "Generates FOV-level mHTT expression data used by S5, S7, S10, S11, and final figures",
        "dependencies": [],
    },
    "DATA_REGIONAL": {
        "name": "DATA_REGIONAL",
        "title": "Regional analysis data generation",
        "script": "draft_figures/fig_regional_analysis.py",
        "output": "draft_figures/output/regional_analysis/regional_analysis_subregion_level.csv",
        "tex_path": None,  # Not a published figure
        "description": "Generates regional analysis data used by figure3",
        "dependencies": ["DATA_EXPR"],
    },
    "DATA_CLUSTER": {
        "name": "DATA_CLUSTER",
        "title": "Cluster properties data generation",
        "script": "draft_figures/fig_cluster_properties_extreme_vs_normal.py",
        "output": "draft_figures/output/cluster_properties_extreme_vs_normal/cluster_level_data.csv",
        "tex_path": None,  # Not a published figure
        "description": "Generates cluster property data used by figure5",
        "dependencies": ["DATA_EXPR"],
    },
    "S1": {
        "name": "S1",
        "title": "Bead-based PSF calibration",
        "script": None,  # Pre-generated from bead analysis
        "output": "bead_psf_plots/green_['Beads_0.2um']_gaussian_fit.pdf",
        "tex_path": "figures/methods/green_['Beads_0.2um']_gaussian_fit.pdf",
        "description": "Distributions of calibrated PSF widths from 0.2um TetraSpeck beads",
        "dependencies": [],
    },
    "S2": {
        "name": "S2",
        "title": "Negative controls establish per-slide detection thresholds",
        "script": "draft_figures/fig_negative_threshold.py",
        "output": "draft_figures/output/fig_negative_threshold.pdf",
        "tex_path": "figures/results/fig_negative_threshold.pdf",
        "description": "Analysis of negative control spots for threshold determination",
        "dependencies": [],
    },
    "S3": {
        "name": "S3",
        "title": "Empirical definition of the single-molecule regime via breakpoint analysis",
        "script": "draft_figures/fig_single_breakpoint_v3.py",
        "output": "draft_figures/output/single_breakpoint/fig_single_breakpoint_v3.pdf",
        "tex_path": "figures/results/fig_single_breakpoint_v3.pdf",
        "description": "Biphasic intensity-size relationships and breakpoint detection",
        "dependencies": ["S1"],
    },
    "S4": {
        "name": "S4",
        "title": "Validation of single-molecule filtering",
        "script": "draft_figures/fig_filtering_v2.py",
        "output": "draft_figures/output/filtering_figures/fig_filtering_v2.pdf",
        "tex_path": "figures/results/fig_filtering_v2.pdf",
        "description": "Filtering workflow and single-molecule identification validation",
        "dependencies": ["S3"],
    },
    "S5": {
        "name": "S5",
        "title": "Negative control thresholds show no significant correlations",
        "script": "draft_figures/fig_negative_control_threshold_correlation.py",
        "output": "draft_figures/output/negative_control_threshold_correlation/fig_negative_control_threshold_correlation.pdf",
        "tex_path": "figures/results3/fig_negative_control_threshold_correlation.pdf",
        "description": "Correlation analysis between thresholds and experimental expression",
        "dependencies": ["S12", "DATA_EXPR"],  # Needs S12 for thresholds, DATA_EXPR for FOV data
    },
    "S6": {
        "name": "S6",
        "title": "Positive control analysis validates assay dynamic range",
        "script": "draft_figures/fig_positive_control_comprehensive.py",
        "output": "draft_figures/output/positive_control_comprehensive/fig_positive_control_comprehensive.pdf",
        "tex_path": "figures/results2/fig_positive_control_comprehensive.pdf",
        "description": "POLR2A and UBC housekeeping gene expression analysis",
        "dependencies": [],
    },
    "S7": {
        "name": "S7",
        "title": "Positive control housekeeping genes correlate with experimental probes",
        "script": "draft_figures/fig_positive_control_vs_experimental_correlation.py",
        "output": "draft_figures/output/positive_control_vs_experimental/fig_positive_control_vs_experimental_correlation.pdf",
        "tex_path": "figures/results3/fig_positive_control_vs_experimental_correlation.pdf",
        "description": "Cross-channel correlation between housekeeping and experimental probes",
        "dependencies": ["S6", "DATA_EXPR"],  # Needs S6 for positive control data, DATA_EXPR for FOV data
    },
    "S8": {
        "name": "S8",
        "title": "Partial correlation analysis rules out RNA quality as confounder",
        "script": "draft_figures/fig_positive_control_vs_experimental_correlation.py",
        "output": "draft_figures/output/positive_control_vs_experimental/fig_rna_quality_confounder_analysis.pdf",
        "tex_path": "figures/results3/fig_rna_quality_confounder_analysis.pdf",
        "description": "RNA quality confounder analysis using partial correlations",
        "dependencies": ["S7"],
    },
    "S9": {
        "name": "S9",
        "title": "Method comparison between RNAscope and RNA-seq",
        "script": "draft_figures/fig_method_comparison_rnascope_vs_rnaseq.py",
        "output": "draft_figures/output/method_comparison/fig_method_comparison_rnascope_vs_rnaseq.pdf",
        "tex_path": "figures/results2/fig_method_comparison_rnascope_vs_rnaseq.pdf",
        "description": "Comparison of expression measurements between methods",
        "dependencies": [],
    },
    "S10": {
        "name": "S10",
        "title": "Clustered mRNA expression dominates total signal",
        "script": "draft_figures/fig_clustered_mrna_comprehensive.py",
        "output": "draft_figures/output/clustered_mrna_comprehensive/fig_clustered_mrna_comprehensive.pdf",
        "tex_path": "figures/results5/fig_clustered_mrna_comprehensive.pdf",
        "description": "Clustered mRNA quantification across ages and regions",
        "dependencies": ["S4", "DATA_EXPR"],  # Needs S4 for filtering, DATA_EXPR for FOV data
    },
    "S11": {
        "name": "S11",
        "title": "Single mRNA spots contribute minimally to total expression",
        "script": "draft_figures/fig_single_mrna_comprehensive.py",
        "output": "draft_figures/output/single_mrna_comprehensive/fig_single_mrna_comprehensive.pdf",
        "tex_path": "figures/results5/fig_single_mrna_comprehensive.pdf",
        "description": "Single spot density analysis",
        "dependencies": ["S4", "DATA_EXPR"],  # Needs S4 for filtering, DATA_EXPR for FOV data
    },
    "S12": {
        "name": "S12",
        "title": "Comprehensive negative control analysis",
        "script": "draft_figures/fig_negative_control_comprehensive.py",
        "output": "draft_figures/output/negative_control_comprehensive/fig_negative_control_comprehensive.pdf",
        "tex_path": "figures/results2/fig_negative_control_comprehensive.pdf",
        "description": "Region-independent detection threshold analysis",
        "dependencies": ["S2"],
    },
    "S13": {
        "name": "S13",
        "title": "Cluster size distributions for mHTT1a transcripts",
        "script": "draft_figures/fig_cluster_size_distributions.py",
        "output": "draft_figures/output/cluster_size_distributions/cluster_size_distributions_mHTT1a.pdf",
        "tex_path": "figures/results6/cluster_size_distributions_mHTT1a.pdf",
        "description": "mHTT1a cluster size probability distributions",
        "dependencies": ["S4"],
    },
    "S14": {
        "name": "S14",
        "title": "Cluster size distributions for full-length mHTT transcripts",
        "script": "draft_figures/fig_cluster_size_distributions.py",
        "output": "draft_figures/output/cluster_size_distributions/cluster_size_distributions_full_length_mHTT.pdf",
        "tex_path": "figures/results6/cluster_size_distributions_full_length_mHTT.pdf",
        "description": "Full-length mHTT cluster size probability distributions",
        "dependencies": ["S4"],
    },
}


def get_figure_info(fig_id: str) -> dict:
    """Get information about a specific figure."""
    return SUPPLEMENTARY_FIGURES.get(fig_id.upper())


def get_all_figures() -> dict:
    """Get all supplementary figures."""
    return SUPPLEMENTARY_FIGURES


def get_figures_by_script(script_name: str) -> list:
    """Get all figures generated by a specific script."""
    return [
        fig for fig in SUPPLEMENTARY_FIGURES.values()
        if fig["script"] and script_name in fig["script"]
    ]


def get_generation_order() -> list:
    """
    Return figure IDs in order that respects dependencies.
    Uses topological sort.
    """
    # Build dependency graph
    order = []
    visited = set()

    def visit(fig_id):
        if fig_id in visited:
            return
        visited.add(fig_id)
        fig = SUPPLEMENTARY_FIGURES.get(fig_id)
        if fig:
            for dep in fig.get("dependencies", []):
                visit(dep)
            order.append(fig_id)

    for fig_id in SUPPLEMENTARY_FIGURES:
        visit(fig_id)

    return order


if __name__ == "__main__":
    print("=" * 70)
    print("SUPPLEMENTARY FIGURE REGISTRY")
    print("=" * 70)

    print(f"\nTotal figures: {len(SUPPLEMENTARY_FIGURES)}")
    print("\nGeneration order (respecting dependencies):")
    for i, fig_id in enumerate(get_generation_order(), 1):
        fig = SUPPLEMENTARY_FIGURES[fig_id]
        deps = ", ".join(fig["dependencies"]) if fig["dependencies"] else "none"
        print(f"  {i:2d}. {fig_id}: {fig['title'][:50]}... (deps: {deps})")

    print("\n" + "=" * 70)
