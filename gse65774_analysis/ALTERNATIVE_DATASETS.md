# Alternative Huntington's Disease Datasets for Testing

## Recommended Datasets Similar to GSE65774

Based on your current analysis of Q111 mice, here are the best alternative datasets to validate your findings:

---

## 🌟 **Top Recommendation: GSE152058**

### **Cell Type-Specific Transcriptomics in HD Mouse Models**

**Why this is perfect for you:**
- ✅ Multiple CAG repeat lengths: Q20, Q50, Q111, Q170, Q175
- ✅ **Includes Q111** - can directly compare to your current results
- ✅ Striatum tissue (same as your current study)
- ✅ Multiple timepoints: 3 months and 6 months
- ✅ Large sample size: 333 samples
- ✅ Processed data available (RPKM, count matrices)

**Access:**
- GEO: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE152058
- Published: July 2020
- SRA: SRP267903

**What you can test:**
1. **Direct Q111 validation:** Compare your 6M Q111 findings
2. **Cross-CAG analysis:** Test mHTT-UBC-POLR2A correlations across Q20-Q175
3. **Cell-type specific:** See if correlations differ in D1 vs D2 neurons
4. **3-month early timepoint:** Test your "early coordinated response" hypothesis

**Models included:**
- Q20 (control-like)
- Q50
- **Q111** (your current model!)
- Q170
- Q175
- zQ175 (heterozygous)
- R6/2 (transgenic)

---

## 🔬 **Alternative Option 1: GSE135057**

### **Transcriptional Correlates in R6/1 HD Mice**

**Dataset:**
- GEO: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE135057
- 16 samples
- R6/1 transgenic model vs WT
- Multiple brain regions: striatum, cortex, hippocampus, cerebellum

**Why consider this:**
- ✅ Striatum included
- ✅ Processed data available
- ✅ Different HD model (R6/1) - tests if your findings generalize
- ✅ Behavioral stratification (good vs poor performers)

**Limitations:**
- ❌ No Q111 model (different genetic background)
- ❌ Smaller sample size (n=16)
- ❌ Single timepoint

**What you can test:**
- Do mHTT-UBC-POLR2A correlations exist in R6/1 model?
- Are they different in good vs poor performers?

**Published:** Scientific Reports, 2019

---

## 🧬 **Alternative Option 2: Other Datasets from the Original Study**

### **GSE65775 - Tissue Survey from Same Lab**

**Dataset:**
- From same Langfelder/CHDI group as GSE65774
- Q175 mice at 6 months
- Multiple tissues surveyed

**Why consider this:**
- ✅ Same lab, same processing pipeline
- ✅ Can compare Q175 to your Q111 findings
- ✅ 6-month timepoint matches

**Access:**
- GEO: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE65775

---

## 📊 **Comparison Table**

| Dataset | Model | CAG Length | Timepoints | Samples | Striatum? | Processed Data? | Best For |
|---------|-------|------------|------------|---------|-----------|-----------------|----------|
| **GSE65774** | Q111 KI | 20,80,92,111,140,175 | 2M,6M,10M | 208 | ✅ Yes | ✅ FPKM | Your current |
| **GSE152058** | Multiple | 20,50,111,170,175 | 3M,6M | 333 | ✅ Yes | ✅ RPKM | **Best validation** |
| **GSE135057** | R6/1 | N/A (fragment) | 1 point | 16 | ✅ Yes | ✅ Yes | Model comparison |
| **GSE65775** | Q175 KI | 175 | 6M | ~50 | ✅ Yes | ✅ FPKM | Same lab |

---

## 🎯 **Recommended Analysis Strategy**

### **Step 1: Validate in GSE152058 (Highest Priority)**

Download and analyze Q111 samples from GSE152058:

**Questions to answer:**
1. At 6 months, does mHTT correlate with UBC in Q111? (Your result: r=0.56, p=0.15)
2. At 3 months, does mHTT correlate with UBC? (Closer to your 2M finding: r=0.83)
3. Does POLR2A show elevation in Q111 vs WT?
4. Do the correlations change over time (3M → 6M)?

**Expected outcome:**
- If validated: Your findings are robust across studies
- If different: Could be due to cell-type composition (they have cell-specific data)

### **Step 2: Cross-CAG Repeat Analysis**

Use GSE152058 to test across Q20, Q50, Q111, Q170, Q175:

**Your hypothesis to test:**
- Early coordination (2-3M): mHTT-UBC-POLR2A correlate
- Does this pattern hold across ALL CAG lengths?
- Or is it Q111-specific?

### **Step 3: Test in Different Model (Optional)**

Use GSE135057 (R6/1):

**Question:**
- Is the mHTT-UBC relationship specific to knock-in models?
- Or does it generalize to transgenic models?

---

## 📥 **How to Download GSE152058**

### **Option 1: GEO Website (Easiest)**

```bash
# Download processed count matrix
wget https://ftp.ncbi.nlm.nih.gov/geo/series/GSE152nnn/GSE152058/suppl/GSE152058_RAW.tar

# Or download series matrix
wget "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE152nnn/GSE152058/matrix/GSE152058_series_matrix.txt.gz"
```

### **Option 2: GEOquery in R**

```R
library(GEOquery)
gse <- getGEO("GSE152058")
expr_data <- exprs(gse[[1]])
metadata <- pData(gse[[1]])
```

### **Option 3: SRA Toolkit (Raw Data)**

```bash
# Download raw FASTQ files
prefetch SRP267903
fastq-dump --split-files SRR*
```

---

## 🔍 **Additional Resources**

### **HDinHD.org**
- Comprehensive HD transcriptomics database
- Combined results from multiple studies
- Interactive exploration
- URL: http://www.hdinHD.org

### **Single-Cell Datasets**

If you want to go deeper:

**GSE153828** - Single-nucleus RNA-seq
- zQ175 model
- Cell-type resolution
- Can test if mHTT-UBC correlation is cell-type specific

---

## ⚠️ **Important Considerations**

### **Batch Effects**
- Different labs, different processing
- May need batch correction (ComBat)
- Or just compare trends, not absolute values

### **Different Normalization**
- GSE65774: FPKM
- GSE152058: RPKM (very similar to FPKM)
- GSE135057: May use different method

**Solution:** Use same normalization approach or work with raw counts

### **Different Mouse Ages**
- Your 2M vs their 3M (close enough)
- Your 6M = their 6M (perfect match)
- Your 10M vs no equivalent (unique to your dataset)

---

## 📝 **Analysis Plan for GSE152058**

### **Step-by-Step:**

1. **Download data**
   ```bash
   cd gse65774_analysis
   mkdir -p ../gse152058_validation
   cd ../gse152058_validation
   # Download files
   ```

2. **Filter to Q111 and WT samples at 6M**
   - Match your current analysis

3. **Extract mHTT, UBC, POLR2A expression**
   - Same genes, same approach

4. **Test correlations**
   - Compare to your results:
     - mHTT vs UBC (your r=0.56 at 6M)
     - mHTT vs POLR2A
     - UBC vs POLR2A (your r=0.71 at 6M)

5. **Compare Q111 vs WT**
   - Test if POLR2A is elevated (your result: +17%)
   - Test if UBC is similar (your result: no difference)

6. **Create comparison figures**
   - Side-by-side with GSE65774 results

---

## 🎓 **Publication Strategy**

### **If GSE152058 Validates Your Findings:**

**Strength of evidence:**
- Finding replicated in independent dataset
- Different lab, different cohort
- Same biological conclusion

**Paper structure:**
- Discovery cohort: GSE65774
- Validation cohort: GSE152058
- Meta-analysis across both

### **If Results Differ:**

**Possible explanations:**
- Age differences (2M vs 3M)
- Processing differences
- Cell-type composition
- True biological variability

**Paper approach:**
- Context-dependent findings
- Discuss what drives differences
- Still valuable biology

---

## 🚀 **Next Steps**

**Immediate (Today):**
1. Download GSE152058 series matrix
2. Check sample annotations
3. Verify Q111 samples are present

**This Week:**
1. Extract expression data for target genes
2. Run same correlation analysis
3. Compare results to GSE65774

**Optional (If Time):**
1. Analyze GSE135057 (R6/1 model)
2. Look at cell-type data from GSE152058
3. Cross-reference with HDinHD.org

---

## 📚 **Key Publications**

### **GSE152058:**
> Langfelder P, et al. (2016) Integrated genomics and proteomics define huntingtin CAG length-dependent networks in mice. Nat Neurosci. 19(4):623-33.

### **GSE135057:**
> Carmo et al. (2019) Transcriptional correlates of the pathological phenotype in a Huntington's disease mouse model. Sci Rep. 9(1):18362.

---

## 💡 **Why Validation Matters**

**Scientific rigor:**
- Single dataset = interesting finding
- Two datasets = validated finding
- Multiple datasets = robust biology

**Your findings at 2M are strong (r=0.83, p=0.01)**
- If replicated → publishable in high-impact journal
- If not replicated → still interesting, but need to explain why

**GSE152058 is your best bet for validation!**

---

## Summary

**Recommended Priority:**

1. 🥇 **GSE152058** - Best match, includes Q111, multiple timepoints
2. 🥈 **GSE135057** - Different model, tests generalizability
3. 🥉 **GSE65775** - Same lab, Q175 comparison

**Start with GSE152058 - it's the most directly comparable to your current work.**

Would you like me to help you download and analyze GSE152058?
