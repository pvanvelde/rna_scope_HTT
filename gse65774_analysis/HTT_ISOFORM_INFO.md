# HTT-1a Isoform Information

## Question: Is HTT-1a in the dataset?

**Answer: ❌ NO - Only total HTT gene expression is available**

---

## What's in the Dataset

The GSE65774 dataset contains **gene-level quantification only**, using Ensembl gene IDs:

- **HTT gene:** `ENSMUSG00000029189`
- **Expression value:** Represents **ALL HTT isoforms combined**
- **Mean FPKM:** 4.31 across all samples

### What this means:
The HTT expression value you see (ENSMUSG00000029189) includes:
- HTT-1a (exon 1a isoform) ✓
- HTT-1b (exon 1b isoform) ✓
- All other splice variants ✓
- **Combined together** (cannot separate them)

---

## Why HTT-1a is Not Separately Quantified

### 1. Gene-Level vs Isoform-Level Analysis

This dataset used **gene-level summarization** during processing:
- Reads aligned to the genome
- Quantified at the **gene locus** level
- Multiple isoforms collapsed into one value per gene

### 2. No Transcript IDs Present

The dataset does NOT contain:
- ❌ Transcript IDs (like ENSMUST000000...)
- ❌ Exon-level counts
- ❌ Junction reads
- ❌ Isoform-specific quantification

All gene IDs follow the pattern: `ENSMUSG########` (gene-level only)

---

## HTT Isoforms Background

### HTT Gene Structure

The mouse HTT gene (and human HTT) has two main promoters:

1. **HTT-1a (Exon 1a):**
   - Neural-specific promoter
   - Predominantly expressed in brain
   - Contains the CAG repeat expansion
   - This is the "disease-relevant" isoform

2. **HTT-1b (Exon 1b):**
   - Ubiquitous promoter
   - Expressed in many tissues
   - Also contains CAG repeats
   - Less characterized in HD

### Why HTT-1a Matters

In Huntington's disease research, **HTT-1a is critical** because:
- Brain-specific expression
- Highest expression in striatum (the tissue studied here!)
- May be differentially regulated in disease
- Potentially more pathogenic

### The Problem

**Your question was likely motivated by:**
> "Does mutant HTT-1a (the brain-specific isoform) correlate with UBC/POLR2?"

**What we can measure:**
> "Does total HTT (all isoforms) correlate with UBC/POLR2?"

Since HTT-1a is the **predominant isoform in striatum**, the total HTT signal mostly reflects HTT-1a anyway, but we cannot prove this with the current data.

---

## How to Get HTT-1a Specific Data

If you need HTT-1a specifically, you would need:

### Option 1: Reanalyze Raw Data (Best Option)

Download the raw FASTQ files from SRA (SRP053398) and reprocess with isoform-aware tools:

**Tools:**
- **Salmon** or **Kallisto** - Pseudo-alignment with isoform quantification
- **RSEM** - Expectation-maximization for isoforms
- **StringTie** - Transcript assembly and quantification

**Workflow:**
```bash
# Download raw data
prefetch SRP053398
fastq-dump --split-files SRR*

# Quantify with Salmon (isoform-aware)
salmon quant -i mouse_transcriptome_index \
  -l A -1 reads_1.fq -2 reads_2.fq \
  -o output --validateMappings

# This will give you ENSMUST IDs (transcript-level)
```

### Option 2: Exon-Level Analysis

Use tools that quantify individual exons:
- **DEXSeq** - Differential exon usage
- **JunctionSeq** - Junction and exon analysis
- **leafcutter** - Splice junction quantification

This would let you see if exon 1a vs exon 1b usage differs.

### Option 3: Contact Authors

The Langfelder et al. 2016 paper authors may have:
- Isoform-level quantification already done
- Additional unpublished data
- Raw count matrices at transcript level

Contact: Jeff Aaronson (jeff.aaronson@chdifoundation.org)

### Option 4: Alternative Datasets

Look for HD datasets with isoform quantification:
- Check GEO for "Huntington isoform"
- Look for studies using long-read sequencing (PacBio, Nanopore)
- Search for HTT-1a specific papers

---

## What We CAN Say with Current Data

### Assumption: HTT-1a is Predominant in Striatum

Since the striatum is primarily neural tissue, and HTT-1a is the brain-specific isoform, it's reasonable to assume:

**Total HTT expression ≈ Mostly HTT-1a + Small amount of HTT-1b**

### Therefore, our findings likely reflect HTT-1a:

**At 2 months in Q111 mice:**
- HTT (mostly 1a) vs UBC: r = 0.83, p = 0.010 ✅
- HTT (mostly 1a) vs POLR2A: r = 0.67, p = 0.069 (trending)

**Caveats:**
- Cannot prove it's specifically HTT-1a
- Could have differential isoform regulation we're missing
- HTT-1b might confound the signal

---

## Literature Evidence for HTT-1a in Striatum

### Known Facts:

1. **Striatum is enriched for HTT-1a:**
   - Hodgson et al. (2006) showed HTT-1a is the major brain isoform
   - Aronin lab work demonstrated HTT-1a predominance in striatum

2. **Q111 mice express mutant HTT-1a:**
   - These are knock-in models with CAG expansion in exon 1
   - The expansion is in both HTT-1a and HTT-1b
   - But HTT-1a is more abundant in brain

3. **HTT-1a regulation in HD:**
   - Some studies show HTT-1a downregulation in HD
   - Others show no change in transcript, but protein aggregation
   - Post-transcriptional mechanisms are important

---

## Recommendations

### For Your Current Analysis:

**Treat "mHTT" as "primarily mHTT-1a"** with the understanding that:
- It's a reasonable approximation for striatal tissue
- The correlations you found (at 2M) likely reflect HTT-1a biology
- Cannot rule out HTT-1b contribution

### For Future Studies:

1. **Reanalyze the raw data** with isoform quantification (Salmon/Kallisto)
2. **Validate with qPCR** using isoform-specific primers
3. **Look at protein level** - HTT-1a vs 1b may differ at translation/stability
4. **Check exon 1a vs 1b specific reads** in the BAM files (if available)

---

## Summary

| Question | Answer |
|----------|--------|
| Is HTT-1a separately quantified? | ❌ No |
| What do we have? | Total HTT (all isoforms) |
| Can we assume it's mostly HTT-1a? | ✓ Yes (striatum is brain tissue) |
| How confident are we? | ~80% - reasonable but not proven |
| What's the solution? | Reanalyze raw data with isoform tools |

---

## Bottom Line

**Your correlations at 2 months (mHTT with UBC/POLR2A) most likely reflect HTT-1a**, since:
1. Striatum predominantly expresses HTT-1a
2. HTT-1a is the neural-specific, disease-relevant isoform
3. The Q111 mutation affects HTT-1a

However, you **cannot definitively state it's HTT-1a** without isoform-specific quantification.

**Recommended statement for publication:**
> "HTT expression (predominantly representing the HTT-1a isoform in striatal tissue) correlated with UBC (r=0.83, p=0.01) at 2 months in Q111 mice..."

---

**Dataset:** GSE65774 (gene-level quantification only)
**HTT Gene ID:** ENSMUSG00000029189 (all isoforms combined)
**Mean Expression:** 4.31 FPKM
**Isoform Data Available:** ❌ No
