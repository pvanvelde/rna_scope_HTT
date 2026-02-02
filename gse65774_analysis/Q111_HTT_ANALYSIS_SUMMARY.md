# Q111 Mice: HTT, UBC, and POLR2 Expression Analysis

## Summary of Findings

This analysis focused exclusively on Q111 Huntington's disease knock-in mice (n=24) to test the hypothesis that **higher mHTT expression correlates with higher UBC and POLR2 expression**.

---

## Key Results

### 1. Gene Expression Levels in Q111 Mice

| Gene | Mean FPKM | Median FPKM | Range | Description |
|------|-----------|-------------|-------|-------------|
| **mHTT** | 4.16 | 3.92 | [3.42, 6.26] | Mutant Huntingtin |
| **UBC** | 920.09 | 907.52 | [845.11, 1030.48] | Ubiquitin C (very highly expressed) |
| **POLR2A** | 17.68 | 15.90 | [13.32, 23.96] | RNA Polymerase II subunit A |

---

### 2. Correlation Analysis Results

#### **mHTT vs UBC**
- **Pearson r = 0.250** (p = 0.238) ❌ **NOT SIGNIFICANT**
- **Spearman ρ = 0.355** (p = 0.089) ❌ **NOT SIGNIFICANT**
- **Conclusion:** No significant correlation between mHTT and UBC expression in Q111 mice

#### **mHTT vs POLR2A**
- **Pearson r = 0.151** (p = 0.483) ❌ **NOT SIGNIFICANT**
- **Spearman ρ = 0.276** (p = 0.192) ❌ **NOT SIGNIFICANT**
- **Conclusion:** No significant correlation between mHTT and POLR2A expression in Q111 mice

#### **UBC vs POLR2A**
- **Pearson r = 0.746** (p = 2.86×10⁻⁵) ✅ **HIGHLY SIGNIFICANT**
- **Spearman ρ = 0.759** (p = 1.70×10⁻⁵) ✅ **HIGHLY SIGNIFICANT**
- **Conclusion:** Strong positive correlation between UBC and POLR2A expression

---

### 3. Q111 vs Wild Type Comparison

| Gene | Q111 Mean | WT Mean | Fold Change | p-value | Significant? |
|------|-----------|---------|-------------|---------|--------------|
| **mHTT** | 4.16 | 4.38 | 0.95× | 0.354 | ❌ No |
| **UBC** | 920.09 | 939.17 | 0.98× | 0.181 | ❌ No |
| **POLR2A** | 17.68 | 15.27 | **1.16×** | **0.003** | ✅ **Yes** |

**Key Finding:** POLR2A is significantly elevated in Q111 mice compared to WT (Cohen's d = 0.73, indicating a medium-to-large effect size).

---

## Interpretation

### Hypothesis Testing

**Original Hypothesis:** Higher mHTT expression goes with higher UBC and higher POLR2 expression.

**Result:** ❌ **NOT SUPPORTED**

### What We Found Instead:

1. **mHTT expression does NOT significantly correlate with UBC or POLR2A** in Q111 mice
   - The correlations are weak (r < 0.3) and not statistically significant
   - mHTT expression is relatively stable across Q111 samples (SD = 0.71)

2. **UBC and POLR2A are strongly correlated with each other** (r = 0.75, p < 0.001)
   - This likely reflects general transcriptional activity or cell state
   - Both are housekeeping-related genes

3. **POLR2A is elevated in Q111 vs WT** (p = 0.003)
   - This suggests altered transcriptional machinery in HD mice
   - Effect is independent of mHTT expression levels within Q111 group

4. **mHTT and UBC expression are similar between Q111 and WT**
   - mHTT: 4.16 vs 4.38 (not significantly different)
   - UBC: 920 vs 939 (not significantly different)

---

## Biological Interpretation

### Why might the hypothesis not be supported?

1. **Limited dynamic range of mHTT expression:**
   - Within Q111 mice, mHTT expression varies only modestly (3.42-6.26 FPKM)
   - All Q111 mice have the same CAG repeat length (111 repeats)
   - Greater variation might be seen when comparing across multiple CAG lengths

2. **Post-transcriptional regulation:**
   - The toxic effects of mutant huntingtin may occur at the protein level
   - Protein aggregation, not mRNA levels, may drive cellular stress responses
   - UBC upregulation might be triggered by protein aggregates, not mRNA

3. **Cellular compensation mechanisms:**
   - Cells may maintain homeostatic UBC levels despite HD pathology
   - POLR2A elevation might reflect a compensatory transcriptional response

4. **Timepoint considerations:**
   - Effects may be more pronounced at later stages (10 months)
   - Early timepoints (2 months) may not show strong correlations yet

### Why is POLR2A elevated in Q111?

The significant elevation of POLR2A in Q111 mice (16% increase, p = 0.003) suggests:
- **Altered transcriptional machinery** in response to HD pathology
- Possible compensatory increase in RNA Pol II to maintain transcription
- May reflect cellular stress response independent of mHTT mRNA levels

### The UBC-POLR2A correlation

The strong correlation (r = 0.75) between UBC and POLR2A likely reflects:
- Both are abundant, essential housekeeping genes
- Coordinated regulation of protein homeostasis and transcription
- General indicator of cellular metabolic state

---

## Recommendations for Further Analysis

### 1. Test correlation across CAG repeat lengths
Instead of looking within Q111 only, compare:
- Q20 vs Q80 vs Q92 vs Q111 vs Q140 vs Q175
- Use CAG repeat length as a continuous variable
- This provides greater dynamic range for correlation analysis

### 2. Examine HTT protein levels
- RNA levels may not reflect protein aggregation
- Consider proteomics data if available
- Look at HTT aggregate staining data

### 3. Analyze by timepoint
- Separate analyses at 2M, 6M, and 10M
- Progressive disease may show emerging correlations
- Time-series analysis could reveal temporal dynamics

### 4. Include other UPS (Ubiquitin-Proteasome System) genes
Look at additional UPS components:
- UBB, UBA52 (other ubiquitin genes)
- PSMC1, PSMD4 (proteasome subunits)
- UCHL1, USP14 (deubiquitinases)

### 5. Examine stress response genes
- Heat shock proteins (HSPs)
- Chaperones (HSPA, DNAJ family)
- Autophagy markers (SQSTM1, MAP1LC3B)

---

## Output Files

All analysis outputs are located in the `gse65774_analysis` directory:

### Visualizations
- `figures/q111_correlations.png` - Scatter plots showing gene-gene correlations
- `figures/q111_timepoint_expression.png` - Expression across 2M, 6M, 10M timepoints
- `figures/q111_vs_wt_comparison.png` - Comparison of expression levels Q111 vs WT

### Data Tables
- `results/q111_gene_correlations.csv` - Detailed correlation statistics
- `results/q111_vs_wt_comparison.csv` - Statistical comparison results

### Scripts
- `analyze_q111_htt_correlation.py` - Reusable analysis script

---

## Statistical Methods

- **Correlation:** Pearson (parametric) and Spearman (non-parametric)
- **Group comparison:** Independent t-test
- **Effect size:** Cohen's d
- **Significance threshold:** p < 0.05

**Sample sizes:**
- Q111 mice: n = 24 (8 per timepoint)
- WT mice: n = 64 (comparison group)

---

## Conclusion

The hypothesis that **higher mHTT expression correlates with higher UBC and POLR2 expression** is **not supported** within Q111 mice. However, the analysis revealed that:

1. ✅ **POLR2A is significantly elevated in Q111 vs WT**, suggesting altered transcriptional machinery
2. ✅ **UBC and POLR2A are strongly correlated**, reflecting coordinated cellular processes
3. ❌ **mHTT expression does not predict UBC or POLR2A levels** within the Q111 group

**Next steps:** Analyze correlations across multiple CAG repeat lengths (Q20-Q175) to test if broader variation in HD severity reveals relationships between these genes.

---

**Analysis Date:** 2025-11-02
**Dataset:** GSE65774 (Langfelder et al., 2016, Nature Neuroscience)
**Genes Analyzed:**
- mHTT: ENSMUSG00000029189
- UBC: ENSMUSG00000019505
- POLR2A: ENSMUSG00000005198
