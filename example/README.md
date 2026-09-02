# VPRscore (refactored)

A pipeline that combines a Nucleotide-Transformer-based variant risk score
(VPR) with CADD to compute variant-level and sample-level risk scores.

## Structure

```
src/
  vpr_engine.py             Shared logic (sequence extraction, NT model
                             inference, normalization). jax/haiku/
                             nucleotide_transformer are only imported when a
                             VPREngine() is actually created (lazy) -> the
                             --cadd-only mode works without these heavy
                             packages installed.
  prep_vprscore_input.py    Step 1: filter a VCF to target regions (BED) and
                             subset the CADD file to those variants.
  calc_vprPrep.py           Step 2: compute variant-level scores (no
                             genotypes needed). Supports --cadd-only and
                             --chrom.
  merge_vprPrep.py          Step 2b: merge results run in parallel with
                             --chrom splits into one table (duplicate
                             variants are logged and the first value is
                             kept).
  aggregate_vprscore.py     Step 3: variant-level table + VCF genotypes ->
                             sample-level VPRscore. Works with single- or
                             multi-sample VCFs. --vcf accepts multiple files
                             (e.g. per-chromosome VCFs); samples are matched
                             by ID, giving results identical to running on
                             one genome-wide VCF.
  run_singlesample_vprscore.py
                             (kept for backward compatibility) one-shot
                             wrapper that runs calc_vprPrep.py +
                             aggregate_vprscore.py in sequence.
env/
  requirements-cadd-only.txt  Minimal dependencies for --cadd-only (standard
                               library only).
  requirements-full.txt       Full dependencies needed to also use the NT
                               model (jax/haiku, etc.).
  environment.yml             Conda environment (kept as-is).
```

## Changes from the previous version (including bug fixes)

- Consolidated model/sequence logic that was duplicated between
  `calc_multisample_vprPrep.py` / `run_singlesample_vprscore.py` into a
  single `vpr_engine.py`.
- **[Bug fix]** Fixed an asymmetric truncation bug where the REF sequence
  wasn't truncated to `max_tokens` the same way the ALT sequence was
  (`_run_forward` now applies truncation identically to both).
- **[Bug fix]** `prep_vprscore_input.py` used to unconditionally strip the
  `chr` prefix in the CADD subset output -> it now keeps the VCF's original
  chrom notation. This means `samtools faidx` lookups no longer break when
  the reference FASTA uses prefixed naming like `chr19`.
- **[Bug fix]** Fixed a `ZeroDivisionError` in what was
  `run_single_sample_vprscore()` when a sample had zero variants.
- **[Bug fix]** Fixed `find_variant_token()` indexing directly into its
  result without checking for `None` when the variant position couldn't be
  found -> that variant is now logged as a warning and recorded as NA
  instead of crashing the pipeline.
- **[Bug fix]** Fixed the SNV check only validating ALT and not REF (added
  `len(ref)==1`).
- **[New]** `aggregate_vprscore.py --vcf` now accepts multiple files (e.g.
  per-chromosome VCFs). Sample columns are matched by ID rather than
  position and accumulated, so there's no need to concat into a
  genome-wide VCF first (verified to produce numerically identical results
  to a genome-wide concat, even when sample order differs across files).
- Removed dead arguments that weren't actually used (e.g. some instances of
  `--vcf`, `--regions`).
- Replaced the local absolute-path dependency
  (`file:///mss_dc/...`) in `env/requirements.txt` with a git-installable
  one (`requirements-full.txt`).
- **[Normalization fix]** Changed `normalize_cadd()`'s `max_cadd` default
  from `30` to `5.656496`. Column 5 (`cols[4]`) of the `--cadd` input is
  CADD's **RawScore**, not PHRED (the official CADD tsv column order is
  `Chrom Pos Ref Alt RawScore PHRED`). The old `max_cadd=30` came from
  PHRED-scale convention (PHRED>=30 = top 0.1%), which doesn't fit RawScore
  (mostly in the -6..+6 range) -- applying it crushed nearly every variant
  toward 0, and negative RawScores passed straight through as a negative
  `n_cadd`. The new default is the **99.9th percentile** of RawScore,
  measured by streaming the entire `whole_genome_SNVs.tsv.gz` with
  systematic sampling (NR%1000; reproduced across two independent samples
  as 5.781617 / 5.656496) -- i.e. the PHRED>=30 convention carried over to
  the RawScore scale. Negative RawScore is now also clipped to risk 0 (the
  same design principle `normalize_or` uses for OR>=1). Override with
  `calc_vprPrep.py --max-cadd` (recompute this if you're on a different
  CADD version/build).
  **Note**: this changes numeric results from before -- pass `--max-cadd 30`
  to reproduce the old behavior for comparison.

## New features

### CADD-only mode
```bash
python3 src/calc_vprPrep.py \
  --cadd example/tmp_interval.cadd.tsv.gz \
  --out ./prep.cadd_only.txt \
  --cadd-only
```
This works without `jax`/`haiku`/`nucleotide_transformer` installed
(`env/requirements-cadd-only.txt` is sufficient). The `n_vpr` column in the
output is filled with `NA`, and the next step (`aggregate_vprscore.py`)
automatically recognizes this and scores using CADD alone (ignoring
`alpha`).

### Per-chromosome parallelization + merging

Variant-level computation (`calc_vprPrep.py`) doesn't need genotypes, so it
can be run fully independently per chromosome in parallel. Sample-level
summation in `aggregate_vprscore.py` is also associative, so there's no
need to pre-merge into a genome-wide VCF.

```bash
# 1) Compute variant-level scores per chromosome (no genotypes needed, independent)
python3 src/calc_vprPrep.py --cadd cadd.19.tsv.gz --fasta ref.fa --chrom 19 --out prep.19.txt
python3 src/calc_vprPrep.py --cadd cadd.20.tsv.gz --fasta ref.fa --chrom 20 --out prep.20.txt

# 2) Merge the results (duplicate variants are logged and the first value kept)
python3 src/merge_vprPrep.py --inputs prep.19.txt prep.20.txt --out prep.merged.txt

# 3) If the VCF is also split by chromosome, pass the files directly --
#    no need to concat into a genome-wide VCF; samples are matched by ID
#    and accumulated
python3 src/aggregate_vprscore.py \
  --vcf sample.chr19.vcf.gz sample.chr20.vcf.gz \
  --vprPrep prep.merged.txt --alpha 0.5 --beta 0.2 \
  --out score.tsv
```

When passing multiple files to `--vcf`, all files must share the same
sample set (subsets are fine); column order can differ across files since
matching is done by sample ID. An error is raised if a later file contains
a sample ID not present in the first VCF.

## Full pipeline example (multi-sample, runnable as-is with the example data)

```bash
python3 src/prep_vprscore_input.py \
  --vcf example/example_multi.vcf.gz \
  --cadd example/tmp_interval_multi.cadd.tsv.gz \
  --regions example/tmp_interval.bed \
  --out-prefix ./prep/multi

python3 src/calc_vprPrep.py \
  --cadd ./prep/multi.cadd.tsv.gz \
  --cadd-only \
  --out ./multi_prep.txt
# To also use the actual NT model, use --fasta example/chr19.fa instead of
# --cadd-only (chr19.fa must be downloaded separately via the link below --
# only its .fai is included in this repo).

python3 src/aggregate_vprscore.py \
  --vcf ./prep/multi.filtered.vcf.gz \
  --vprPrep ./multi_prep.txt \
  --alpha 0.5 --beta 0.2 \
  --out ./multisample_vprscore.tsv
```

The file passed to `--cadd` in the `prep_vprscore_input.py` step must be the
genome-wide CADD original, and must be **bgzip-compressed with a tabix
index** (`bgzip file.tsv && tabix -s1 -b2 -e2 file.tsv.gz`). A
`whole_genome_SNVs.tsv.gz` downloaded from sources like UK Biobank is
already in this format. `example/tmp_interval*.cadd.tsv.gz` have also been
re-indexed with bgzip/tabix for this to run end to end.

Single-sample runs use the same scripts as-is (just use a VCF with one
sample). For a one-shot run:

```bash
python3 src/run_singlesample_vprscore.py \
  --vcf ./prep/sample1.filtered.vcf.gz \
  --fasta example/chr19.fa \
  --cadd ./prep/sample1.cadd.tsv.gz \
  --alpha 0.5 --beta 0.2 \
  --out ./singlesample_vprscore.tsv
```

## Note: CADD values in example/ data

The 5th-column values (roughly 3.8-6.9) in `example/tmp_interval*.cadd.tsv.gz`
are closer to the PHRED range than the actual RawScore distribution
(roughly -6..+6, see above). These look like demo values -- with the new
RawScore-based `max_cadd` default, most of them will clip to 1.0. This
doesn't affect pipeline behavior, but reproducing this exact value with a
real CADD RawScore subset may not give the same numbers.

## Verification status

- All scripts pass `py_compile` / `ast.parse`.
- The `--cadd-only` path was run **end to end through the full 3-step
  pipeline** (`prep_vprscore_input.py` -> `calc_vprPrep.py` ->
  `aggregate_vprscore.py`) on the example data, confirming a clean exit and
  correct per-sample VPRscore output (works without jax/haiku installed).
  The `--chrom` filter + `merge_vprPrep.py` merge path was also verified
  separately.
- `merge_vprPrep.py`'s duplicate-variant detection/warning logic was
  exercised and verified.
- **The NT-model path (without `--cadd-only`) could not be run in this
  environment** since jax/haiku/nucleotide_transformer aren't installed
  here. The `_run_forward`/`_compute_odds_ratio` logic preserves the
  original code's flow (only the truncation symmetry was fixed), so no
  behavioral difference is expected, but running it once against real
  model weights in a GPU environment and comparing against prior results is
  recommended.
