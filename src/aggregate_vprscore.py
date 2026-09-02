#!/usr/bin/env python3
"""
aggregate_vprscore.py (구 run_multisample_vprscore.py)

calc_vprPrep.py가 만든 variant-level 테이블 + VCF의 genotype을 합쳐서
sample-level VPRscore를 계산한다.

VCF는 single-sample이든 multi-sample이든 상관없다 (#CHROM 라인의 샘플
컬럼 수만큼 알아서 처리) -- 그래서 별도의 "single-sample 스크립트"가
필요 없다.
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
    raise ValueError(f"{vcf_path}에서 #CHROM 라인을 찾을 수 없습니다.")


def load_variant_table(variant_table_path):
    """
    n_vpr이 "NA"인 행(=calc_vprPrep.py --cadd-only로 만든 결과)은
    n_vpr=None으로 저장해두고, 합산 시 CADD 점수만 사용한다.
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
    n_vpr이 None(CADD-only로 계산된 variant)이면 CADD만 사용한다."""
    if n_vpr is None:
        return n_cadd
    return alpha * n_vpr + (1.0 - alpha) * n_cadd


def main():
    parser = argparse.ArgumentParser(
        description="Compute sample-level VPRscores from a VCF (single- or multi-sample) "
                    "and a precomputed variant-level table."
    )
    parser.add_argument("--vcf", required=True, help="Biallelic VCF (single- or multi-sample).")
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
    sample_ids = extract_sample_ids(args.vcf)
    num_samples = len(sample_ids)
    sample_scores = [0.0] * num_samples
    sample_counts = [0] * num_samples
    n_cadd_only_obs = 0

    with open_maybe_gzip(args.vcf) as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
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
            s_i = compute_single_variant_score(n_vpr, n_cadd, args.alpha)

            fmt = cols[8].split(":")
            gt_idx = fmt.index("GT") if "GT" in fmt else 0

            for j in range(num_samples):
                fields = cols[9 + j].split(":")
                if gt_idx >= len(fields):
                    continue
                gt = fields[gt_idx]
                if gt in {".", "./.", ".|."}:
                    continue
                alleles = re.split(r"[/|]", gt)
                if "1" in alleles:
                    sample_scores[j] += s_i
                    sample_counts[j] += 1

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
