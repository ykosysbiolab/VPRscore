#!/usr/bin/env python3
"""
aggregate_vprscore.py (formerly run_multisample_vprscore.py)

Combines the variant-level table produced by calc_vprPrep.py with a VCF's
genotypes to compute sample-level VPRscores.

Works with either a single-sample or multi-sample VCF (handles however many
sample columns are on the #CHROM line) -- so no separate "single-sample
script" is needed.

--vcf can also take multiple files (e.g. per-chromosome VCF splits).
Sample-level summation (sample_scores[j] += s_i) is associative regardless
of order, so there's no need to concat into a genome-wide VCF first --
each file can just be scanned and accumulated in turn. Sample columns
across multiple VCFs are matched by ID, not position (safe even if sample
order differs, or only partially overlaps, across files).
"""

import argparse
import math
import re
import sys

from vpr_engine import open_maybe_gzip, norm_chrom


def extract_sample_ids(vcf_path):
    with open_maybe_gzip(vcf_path) as f:
        for line in f:
            if line.startswith("#CHROM"):
                return line.strip().split("\t")[9:]
    raise ValueError(f"Could not find a #CHROM line in {vcf_path}.")


def load_variant_table(variant_table_path):
    """
    Rows where n_vpr is "NA" (i.e. produced by calc_vprPrep.py --cadd-only)
    are stored with n_vpr=None, and only the CADD score is used for them
    when summing.
    """
    variant_dict = {}
    with open_maybe_gzip(variant_table_path) as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 6:
                continue
            chrom, pos, ref, alt, n_vpr, n_cadd = cols[:6]
            key = (norm_chrom(chrom), int(pos), ref, alt)
            n_vpr_val = None if n_vpr.strip().upper() == "NA" else float(n_vpr)
            variant_dict[key] = (n_vpr_val, float(n_cadd))
    return variant_dict


def compute_single_variant_score(n_vpr, n_cadd, alpha):
    """s_i = alpha * n_vpr + (1-alpha) * n_cadd.
    If n_vpr is None (a CADD-only-scored variant), only CADD is used."""
    if n_vpr is None:
        return n_cadd
    return alpha * n_vpr + (1.0 - alpha) * n_cadd


def process_vcf(vcf_path, variant_table, alpha, sample_index, sample_scores, sample_counts):
    """Scan a single VCF and accumulate into sample_scores/sample_counts.
    sample_index: {sample_id: global_index} -- maps a sample_id seen in any
    VCF to the same position in the global arrays. Safe even if sample
    order differs across VCFs, or sample sets only partially overlap.
    Returns: how many observations in this VCF used a CADD-only
    (n_vpr=NA) score."""
    n_cadd_only_obs = 0
    with open_maybe_gzip(vcf_path) as f:
        col_to_global = None
        for line in f:
            if not line.strip():
                continue
            if line.startswith("#CHROM"):
                local_samples = line.strip().split("\t")[9:]
                unknown = [s for s in local_samples if s not in sample_index]
                if unknown:
                    raise ValueError(
                        f"{vcf_path}: found sample ID(s) not present in the first VCF: "
                        f"{unknown[:5]}{' ...' if len(unknown) > 5 else ''} -- "
                        "all VCFs must share the same sample set (subsets are OK)."
                    )
                col_to_global = [sample_index[s] for s in local_samples]
                continue
            if line.startswith("#"):
                continue
            if col_to_global is None:
                raise ValueError(f"{vcf_path}: found a data line before the #CHROM header.")
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 10:
                continue
            chrom_raw, pos, ref, alt = cols[0], int(cols[1]), cols[3], cols[4]
            if "," in alt:
                continue
            key = (norm_chrom(chrom_raw), pos, ref, alt)
            if key not in variant_table:
                continue
            n_vpr, n_cadd = variant_table[key]
            if n_vpr is None:
                n_cadd_only_obs += 1
            s_i = compute_single_variant_score(n_vpr, n_cadd, alpha)

            fmt = cols[8].split(":")
            gt_idx = fmt.index("GT") if "GT" in fmt else 0

            for local_j, global_j in enumerate(col_to_global):
                fields = cols[9 + local_j].split(":")
                if gt_idx >= len(fields):
                    continue
                gt = fields[gt_idx]
                if gt in {".", "./.", ".|."}:
                    continue
                alleles = re.split(r"[/|]", gt)
                if "1" in alleles:
                    sample_scores[global_j] += s_i
                    sample_counts[global_j] += 1
    return n_cadd_only_obs


def main():
    parser = argparse.ArgumentParser(
        description="Compute sample-level VPRscores from one or more VCFs (single- or "
                    "multi-sample, e.g. per-chromosome splits) and a precomputed variant-level "
                    "table. Multiple --vcf files are matched by sample ID and accumulated "
                    "together, so a genome-wide concat is not required."
    )
    parser.add_argument("--vcf", required=True, nargs="+",
                         help="One or more biallelic VCFs, e.g. per-chromosome splits sharing "
                              "the same samples. Sample columns are matched by ID across files, "
                              "not by column position.")
    parser.add_argument("--vprPrep", required=True,
                         help="calc_vprPrep.py (or merge_vprPrep.py) output: #chr pos ref alt n_vpr n_cadd")
    parser.add_argument("--alpha", type=float, default=0.5,
                         help="Weight for combining n_vpr and n_cadd (default: 0.5)")
    parser.add_argument("--beta", type=float, default=0.2,
                         help="Scaling parameter for variant-count weighting (default: 0.2)")
    parser.add_argument("--out", "--output_file", dest="output_file",
                         default="sample_vprscore.tsv",
                         help="Output file name (default: sample_vprscore.tsv). "
                              "--output_file kept as alias for backward compatibility.")
    args = parser.parse_args()

    variant_table = load_variant_table(args.vprPrep)

    # Use the first VCF's sample order as the global reference order.
    sample_ids = extract_sample_ids(args.vcf[0])
    sample_index = {s: i for i, s in enumerate(sample_ids)}
    num_samples = len(sample_ids)
    sample_scores = [0.0] * num_samples
    sample_counts = [0] * num_samples
    n_cadd_only_obs = 0

    for vcf_path in args.vcf:
        sys.stderr.write(f"[aggregate_vprscore] Processing {vcf_path} ...\n")
        n_cadd_only_obs += process_vcf(
            vcf_path, variant_table, args.alpha, sample_index, sample_scores, sample_counts,
        )

    if n_cadd_only_obs > 0:
        sys.stderr.write(
            f"[aggregate_vprscore] {n_cadd_only_obs} variant-sample observations "
            f"used CADD-only score (n_vpr=NA), alpha ignored for those.\n"
        )

    with open(args.output_file, "w") as out:
        print("Sample_ID\tScore_Sum\tCount\tAverage\tVPRscore", file=out)
        for s, total, n in zip(sample_ids, sample_scores, sample_counts):
            if n == 0:
                mean, vprs = 0.0, 0.0
            else:
                mean = total / n
                vprs = mean * (1 + args.beta * math.log(n))
            out.write(f"{s}\t{total:.6f}\t{n}\t{mean:.6f}\t{vprs:.6f}\n")

    sys.stderr.write(f"[aggregate_vprscore] Done -> {args.output_file}\n")


if __name__ == "__main__":
    main()
