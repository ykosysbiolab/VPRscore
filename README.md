# VPRscore
A lightweight framework for estimating sample-level risk scores leveraging LLM-driven variant effects.

---

## Overview
VPRscore estimates the functional impact of genetic variants by comparing the **sequence-context probability** of reference vs. alternate alleles using a pretrained DNA language model (Nucleotide Transformer).

Variant-level signals are combined with CADD annotations and aggregated to produce a **sample-level VPRscore**.

<img src="./main.png" title="VPRscore_workflow"/>

---

## Installation

```bash
git clone https://github.com/ykosysbiolab/VPRscore.git
cd VPRscore/env/

#Using conda
conda env create -f environment.yml       # replace with your YAML file name if different
conda activate vprscore                   # use the name defined in the YAML

#Using pip
pip install -r requirements.txt
```

## Input Requirements

### Required

- VCF file with variants
  - single-sample or multi-sample VCF depending on the workflow:  
    - Two-step (multi-sample) mode: a merged multi-sample VCF containing all variants across samples
    - One-step (single-sample) mode: a per-sample VCF.
  - VCFs should be normalized to biallelic sites
    ```
    bcftools norm -m -both input.vcf.gz -Oz -o input.biallelic.vcf.gz
    bcftools index input.biallelic.vcf.gz
    ```
  - Must be on the same reference build as the FASTA and CADD
    
- Reference genome FASTA 
  - e.g., GRCh38.fa
  - Indexed with samtools faidx
 
- CADD tsv.gz (for annotation integration)
  - Download whole_genome_SNVs.tsv.gz and whole_genome_SNVs.tsv.gz.tbi from the official CADD website: https://cadd.gs.washington.edu/download
  - Use the file matching your genome build (e.g. "All possible SNVs of GRCh38/hg38" for hg38) 

- Target region BED file
  - Genomic intervals where VPRscore should be computed (e.g. candidate diseases genes, exome panel, or GWAS loci).
  - Using a region file is strongly recommended: running over the entire genome is computationally expensive for large VCFs.

## Usage

### 1. Input preparation

### 2.1 Multi-sample mode (recommended)
Compute variant-level VPR and sample-level VPRscores from a merged multi-sample VCF in a single command:
  
   ```bash
    python src/run_multisample_vprscore.py \
    --vcf merged.biallelic.vcf.gz \
    --fasta GRCh38.fa \
    --regions regions.bed \
    --cadd cadd_preprocessed.tsv.gz \
    --alpha 0.5 \
    --beta 0.2 \
    --out ./multisample_vprscore.tsv
  ```

### 2.2 Alternative: one-step workflow (single-sample mode)
For small datasets or quick testing, VPRscore can also be computed directly from a single-sample VCF in one step.

  ```bash  
  python3 run_singlesample_vprscore.py \
    --vcf example/tmp_interval.filtered.vcf.gz \
    --cadd example/tmp_interval.cadd.tsv.gz \
    --fasta example/chr19.fa
    --out out.txt
    &> logs
  ```

## Inputs / Output

### Inputs
- `--vcf` : Biallelic VCF for a single sample.
- `--fasta` : Reference genome FASTA
- `--regions` : BED file with target regions
- `--cadd` : Preprocessed CADD table.
- `--alpha` : Variant-level weight for combining sequence-based risk and CADD. (alpha weight for vpr, default=0.5)
- `--beta` : Scaling parameter controlling variant-count weighting. (larger beta increases variant-count weighting, default=0.2)

### Output
- `multisample_vprscore.tsv` : per-sample VPRscore results
- `singlesample_vprscore.txt` : single VPRscore value result
---

#### Contact
For questions or issues:
- p3159@konkuk.ac.kr
