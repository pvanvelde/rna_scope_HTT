# GSE65774 RNA-seq Analysis: Huntington's Disease Mouse Model

## Overview

This analysis examines transcriptome profiling data from GSE65774, a study of Huntington's disease knock-in mouse models. The dataset contains RNA-seq data from striatum tissue across multiple CAG repeat lengths, timepoints, and biological conditions.

**Citation:** Langfelder P, Cantle JP, Chatzopoulou D, Wang N et al. Integrated genomics and proteomics define huntingtin CAG length-dependent networks in mice. Nat Neurosci 2016 Apr;19(4):623-33. PMID: 26900923

## Dataset Information

- **Organism:** *Mus musculus* (C57BL/6)
- **Tissue:** Striatum
- **Platform:** Illumina HiSeq 2000 (GPL13112)
- **Total Samples:** 208
- **Total Genes:** 39,179 (Ensembl IDs)
- **Genes Detected:** 30,513 genes with expression in at least one sample

### Experimental Design

**CAG Repeat Lengths:**
- Wild Type (Q0)
- Q20 (control)
- Q80
- Q92
- Q111
- Q140
- Q175

**Timepoints:**
- 2 months
- 6 months
- 10 months

**Sample Groups:**
- Heterozygous knock-in mice (HET): 160 samples
- Wild Type littermates (WT): 48 samples
- Both male and female mice included

## Files and Directory Structure

```
gse65774_analysis/
├── data/
│   ├── GSE65774_Striatum_mRNA_counts_processedData.txt  # Raw count matrix
│   ├── GSE65774_Striatum_mRNA_FPKM_processedData.txt    # FPKM normalized values
│   ├── sample_metadata.csv                               # Initial metadata
│   └── metadata_complete.csv                             # Complete parsed metadata
├── figures/
│   ├── qc_overview.png                                   # QC metrics and plots
│   ├── pca_analysis.png                                  # PCA visualizations
│   └── volcano_*.png                                     # Volcano plots for DE
├── results/
│   └── de_*.csv                                          # Differential expression results
├── analyze_gse65774.py                                   # Main analysis script
└── README.md                                             # This file
```

## Analysis Pipeline

### 1. Data Download and Preprocessing

- Downloaded processed count and FPKM data from GEO
- Extracted and organized 208 samples
- Parsed sample metadata from GEO series matrix

### 2. Quality Control

**Library Size Statistics:**
- Mean: 36,799,689 reads
- Median: 35,669,276 reads
- Range: [29,120,490 - 49,550,126]

**Gene Detection:**
- Mean genes per sample: 21,216
- Genes with zero counts: 8,666
- Genes detected (count > 0): 30,513

### 3. Principal Component Analysis

Performed PCA on 13,358 genes with mean FPKM > 1.

**Variance Explained:**
- PC1: 29.61%
- PC2: 19.75%
- PC3: 9.85%
- PC4: 4.99%
- PC5: 4.20%
- **Cumulative (PC1-5): 68.41%**

**Key Observations:**
- Clear separation by experimental factors visible in PCA
- Sex, timepoint, and CAG repeat length contribute to sample variation
- Good sample clustering indicates biological signal

### 4. Differential Expression Analysis

Analyzed three representative comparisons:

#### Comparison 1: Q175 vs WT at 6 months
- Q175: 8 samples
- WT: 48 samples
- Genes analyzed: 14,727
- Significant DE genes (|log2FC| > 1, padj < 0.05): 0

#### Comparison 2: Q140 vs WT at 10 months
- Q140: 8 samples
- WT: 8 samples
- Genes analyzed: 14,727
- Significant DE genes (|log2FC| > 1, padj < 0.05): 0

#### Comparison 3: Q111 vs Q20 at 6 months
- Q111: 8 samples
- Q20: 8 samples
- Genes analyzed: 14,727
- Significant DE genes (|log2FC| > 1, padj < 0.05): 0

**Note:** No genes met the stringent significance criteria (|log2FC| > 1, padj < 0.05) in these comparisons. This suggests:
1. Subtle transcriptional changes requiring more sensitive methods
2. Need for less stringent thresholds or different normalization
3. Biological heterogeneity in disease progression
4. Consider using specialized RNA-seq DE tools (DESeq2, edgeR, limma-voom)

## Usage

### Running the Analysis

```bash
# Ensure you're in the analysis directory
cd gse65774_analysis

# Install required packages
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels

# Run the complete analysis
python3 analyze_gse65774.py
```

### Customizing Analyses

The `GSE65774Analyzer` class provides methods for:

- `load_data()` - Load count and FPKM matrices
- `parse_metadata()` - Extract sample information
- `quality_control()` - Generate QC metrics and plots
- `pca_analysis()` - Perform PCA and visualization
- `differential_expression()` - Compare gene expression between groups

Example custom analysis:

```python
from analyze_gse65774 import GSE65774Analyzer

# Initialize
analyzer = GSE65774Analyzer()
analyzer.load_data()
analyzer.parse_metadata()

# Custom comparison: Q175 vs WT at 10 months
q175_mask = (analyzer.metadata['genotype'] == 'Q175') & \
            (analyzer.metadata['timepoint'] == '10M')
wt_mask = (analyzer.metadata['genotype'] == 'WT') & \
          (analyzer.metadata['timepoint'] == '10M')

de_results = analyzer.differential_expression(
    q175_mask, wt_mask, 'Q175_10M', 'WT_10M'
)
```

## Output Files

### Visualizations

1. **qc_overview.png** - Four-panel QC summary:
   - Library size distribution
   - Genes detected per sample
   - Mean gene expression distribution
   - Sample correlation heatmap

2. **pca_analysis.png** - Four-panel PCA visualization:
   - PC1 vs PC2 colored by CAG repeat length
   - PC1 vs PC2 colored by timepoint
   - PC1 vs PC2 colored by sex
   - Scree plot (variance explained)

3. **volcano_*.png** - Volcano plots for each comparison showing:
   - log2 fold change vs -log10(adjusted p-value)
   - Upregulated genes (red)
   - Downregulated genes (blue)
   - Non-significant genes (gray)

### Data Files

1. **de_*.csv** - Differential expression results with columns:
   - `gene_id` - Ensembl gene ID
   - `[group]_mean` - Mean FPKM for each group
   - `log2_fold_change` - Log2 fold change
   - `pvalue` - Raw p-value from t-test
   - `padj` - Adjusted p-value (Benjamini-Hochberg)
   - `significant` - Boolean significance flag
   - `regulation` - UP/DOWN/NS classification

2. **metadata_complete.csv** - Complete sample metadata:
   - `geo_accession` - GEO sample accession
   - `title` - Full sample description
   - `genotype` - CAG repeat genotype
   - `cag_repeats` - Number of CAG repeats
   - `timepoint` - Age category (2M/6M/10M)
   - `timepoint_months` - Numeric months
   - `sex` - M/F
   - `sample_id` - Sample identifier in data matrices

## Next Steps and Recommendations

1. **Advanced Differential Expression:**
   - Use DESeq2, edgeR, or limma-voom for more sensitive detection
   - Apply batch effect correction if needed
   - Consider time-series analysis across all three timepoints

2. **Functional Analysis:**
   - Gene ontology (GO) enrichment analysis
   - KEGG pathway analysis
   - Gene set enrichment analysis (GSEA)

3. **Gene Co-expression Networks:**
   - WGCNA to identify co-regulated gene modules
   - Correlation with CAG repeat length
   - Network visualization

4. **Integration with External Data:**
   - Compare with human Huntington's disease datasets
   - Integrate with proteomics data from same study
   - Cross-reference with known HD pathways

5. **Additional Comparisons:**
   - Progressive changes: 2M vs 6M vs 10M within genotypes
   - CAG length correlation analysis
   - Sex-specific effects

## Dependencies

- Python 3.7+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- statsmodels
- scipy

## Data Source

Original data available at:
- **GEO:** https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE65774
- **SRA:** SRP053398

## Analysis Date

Generated: 2025-11-02
