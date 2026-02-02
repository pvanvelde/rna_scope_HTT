# FPKM Explanation: What It Is and Whether to Normalize

## What is FPKM?

**FPKM** = **Fragments Per Kilobase of transcript per Million mapped reads**

### The Formula:
```
FPKM = (# of fragments mapping to gene) × 10⁹
       ─────────────────────────────────────────
       (total # of fragments) × (gene length in bp)
```

### What Each Part Corrects For:

1. **Per Million mapped reads:** Corrects for sequencing depth differences between samples
2. **Per Kilobase:** Corrects for gene length (longer genes get more reads)

### Example:
- Gene length: 2,000 bp (2 kb)
- Fragments aligned to gene: 800
- Total fragments in library: 40,000,000 (40M)

```
FPKM = 800 × 10⁹ / (40,000,000 × 2,000)
     = 800,000,000,000 / 80,000,000,000
     = 10 FPKM
```

---

## Is FPKM Normalized?

### ✅ YES - Already Normalized for:

1. **Library size (sequencing depth)**
   - Different samples have different total read counts
   - FPKM normalizes to "per million reads"
   - Accounts for technical variation in sequencing

2. **Gene length**
   - Longer genes naturally get more reads
   - FPKM normalizes to "per kilobase"
   - Allows comparison between genes of different lengths

### ❌ NO - Does NOT Normalize for:

1. **RNA composition effects**
   - If one gene is massively upregulated, it "steals" reads from others
   - FPKM doesn't account for this compositional bias

2. **GC content bias**
   - Some sequences are harder to sequence than others
   - FPKM doesn't correct for sequence-specific biases

3. **Biological replicates**
   - FPKM is calculated per sample
   - Doesn't account for variation between replicates

---

## Should You Normalize FPKM Further?

### For Your Analysis: **NO, FPKM is Appropriate**

You're using FPKM for:
1. **Correlation analysis** (within samples)
2. **Comparing the same genes** across conditions
3. **Fold change calculations** (Q111 vs WT)

FPKM is **already normalized** for these purposes.

### When You DON'T Need Additional Normalization:

✅ **Comparing the same gene across samples** (what you're doing)
- Example: mHTT in Q111 vs mHTT in WT
- FPKM is fine

✅ **Correlations within samples**
- Example: Does mHTT correlate with UBC?
- FPKM is fine

✅ **Fold changes and t-tests**
- Example: Is POLR2A higher in Q111 vs WT?
- FPKM is fine

### When You WOULD Need Different Normalization:

❌ **Comparing different genes to each other**
- Example: "Is mHTT expressed more than UBC?"
- FPKM is NOT ideal (use TPM instead)
- But you're not doing this

❌ **Differential expression with DESeq2/edgeR**
- These tools require **raw counts**, not FPKM
- They apply their own sophisticated normalization

❌ **RNA composition is very different**
- Example: One sample has massive rRNA contamination
- Would need more robust methods (TMM, RLE)

---

## FPKM vs Other Units

| Method | When to Use | Notes |
|--------|-------------|-------|
| **FPKM** | Same gene across samples | What you have |
| **TPM** | Comparing genes within sample | Sum to 1M per sample |
| **Raw counts** | DESeq2, edgeR | For advanced DE analysis |
| **log2(FPKM+1)** | Visualization, clustering | Reduces skewness |
| **Z-scores** | Heatmaps | Standardizes to mean=0, SD=1 |

---

## For Your Specific Analyses

### 1. **Correlation Analysis** (mHTT vs UBC)
- **Use:** Raw FPKM values ✅
- **Why:** Pearson correlation works on continuous data
- **Alternative:** Could use log2(FPKM+1) to reduce outliers, but not necessary

### 2. **Q111 vs WT Comparisons** (t-tests, fold change)
- **Use:** Raw FPKM values ✅
- **Why:** Already normalized for library size and gene length
- **Current approach:** Correct

### 3. **Temporal Changes** (2M → 6M → 10M)
- **Use:** Raw FPKM values ✅
- **Why:** Comparing same genes over time
- **Current approach:** Correct

### 4. **Visualizations** (scatter plots, box plots)
- **Use:** Raw FPKM values ✅
- **Why:** Absolute values are meaningful and interpretable
- **Current approach:** Correct

---

## What Your FPKM Values Mean

### mHTT: ~4 FPKM
- **Low-moderate expression**
- About 4 fragments per kilobase per million reads
- This is reasonable for huntingtin (not a super abundant gene)

### UBC: ~900-1000 FPKM
- **Very high expression**
- Ubiquitin C is a housekeeping gene
- Expected to be highly expressed

### POLR2A: ~15 FPKM
- **Moderate expression**
- RNA Polymerase II subunit
- Expected to be moderately expressed

### Typical FPKM Ranges:
- **< 1:** Very low / not detected
- **1-10:** Low to moderate
- **10-100:** Moderate to high
- **100-1000:** High
- **> 1000:** Very high (housekeeping genes)

---

## Should You Log-Transform?

### When to Use log2(FPKM + 1):

**Advantages:**
- Reduces influence of extreme outliers
- Makes data more normally distributed
- Better for some statistical tests

**Use for:**
- PCA analysis
- Hierarchical clustering
- Heatmaps

**You could use it for:**
- Correlation analysis (won't change rank correlation much)
- But raw FPKM is fine

### Your Current Approach is Fine

You're using raw FPKM, which is appropriate because:
1. Your genes span reasonable ranges
2. No extreme outliers visible
3. Sample sizes are small (n=8) - log won't help much
4. Your correlations are robust

---

## Common Misconceptions

### Myth 1: "FPKM is not normalized"
❌ **FALSE** - FPKM IS normalized for library size and gene length

### Myth 2: "Never use FPKM for anything"
❌ **FALSE** - FPKM is fine for many purposes (like yours)

### Myth 3: "TPM is always better than FPKM"
❌ **FALSE** - For comparing same gene across samples, FPKM = TPM essentially

### Myth 4: "Must log-transform FPKM"
❌ **FALSE** - Only if you have extreme outliers or need normal distribution

---

## Bottom Line for Your Analysis

### ✅ Your Current Use of FPKM is **CORRECT**

**You are:**
- Comparing the same genes across conditions ✓
- Using correlation analysis ✓
- Calculating fold changes ✓
- Performing t-tests ✓

**FPKM is already normalized for your purposes.**

### No Additional Normalization Needed

Your analysis is statistically sound. The FPKM values provided by GEO are:
- Quality controlled
- Normalized for sequencing depth
- Normalized for gene length
- Ready for the analyses you're performing

---

## If You Wanted to Be Extra Rigorous

### Optional Additional Steps (Not Required):

1. **Check for outliers:**
   - Plot FPKM distributions
   - Remove samples with extreme values
   - (You don't have this problem)

2. **Log-transform for visualization:**
   - Use log2(FPKM+1) for PCA plots
   - Makes differences more visible
   - Doesn't change your conclusions

3. **Batch effect correction:**
   - If samples were sequenced in different batches
   - Use ComBat or similar
   - (Probably not an issue here)

4. **Use TPM instead:**
   - Very similar to FPKM for your purposes
   - Won't change results

---

## Conclusion

**Question:** "What unit is FPKM, should this be normalized?"

**Answer:**
- **FPKM** = Fragments Per Kilobase per Million reads
- **Already normalized** for library size and gene length
- **No additional normalization needed** for your analyses
- Your current approach is statistically appropriate

**Your correlations, t-tests, and comparisons using FPKM are valid and publishable.**

---

## Reference

If you want to cite the FPKM method:
> Trapnell C, Williams BA, Pertea G, et al. Transcript assembly and quantification by RNA-Seq reveals unannotated transcripts and isoform switching during cell differentiation. Nat Biotechnol. 2010;28(5):511-515.

---

**Dataset:** GSE65774 (processed FPKM values from GEO)
**Your Analysis:** Appropriate use of FPKM normalization ✅
