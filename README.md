# VPRscore
A lightweight framework for estimating sample-level risk scores leveraging LLM-driven variant effects.

---

## Overview
VPRscore estimates the functional impact of genetic variants by comparing the **sequence-context probability** of reference vs. alternate alleles using a pretrained DNA language model (Nucleotide Transformer).

Variant-level signals are combined with (optional) CADD annotations and aggregated to produce a **sample-level VPRscore**.

<img src="./img/main.png" title="VPRscore_workflow"/>

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

- **VCF file** with variants
- **Reference genome FASTA** (e.g., GRCh38.fa)

### Optional

- **CADD tsv.gz** (for annotation integration)


---

## 📮 Contact

For questions or issues:

**nayoungpark@konkuk.ac.kr**
