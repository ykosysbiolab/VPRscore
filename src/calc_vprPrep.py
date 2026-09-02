#!/usr/bin/env python3
"""
calc_vprPrep.py

variant-level VPR(+CADD) 점수 테이블을 만든다.
Sample-level aggregation과 완전히 분리된 레이어: genotype 필요 없이
(chrom,pos,ref,alt) 목록만 있으면 된다. 그래서 single-sample과
multi-sample 워크플로우가 이 스크립트를 그대로 공유한다
(sample-level 합산은 aggregate_vprscore.py에서).

--cadd-only 를 쓰면 NT 모델(jax/haiku/nucleotide_transformer)이
전혀 필요 없다 -- 그 패키지들이 설치되어 있지 않아도 동작한다.

--chrom 으로 특정 염색체만 처리해서 병렬로 돌린 뒤,
merge_vprPrep.py로 합칠 수 있다.
"""

import argparse
import sys

from vpr_engine import (
    open_maybe_gzip,
    normalize_or,
    normalize_cadd,
    VPREngine,
    DEFAULT_MAX_CADD_RAW,
)


def run_vpr_prep(cadd, fasta, out_path, cadd_only=False, chrom_filter=None,
                  max_cadd=DEFAULT_MAX_CADD_RAW):
    engine = None
    if cadd_only:
        sys.stderr.write(
            "[calc_vprPrep] --cadd-only enabled: NT model will NOT be loaded, "
            "n_vpr will be written as NA.\n"
        )
    else:
        engine = VPREngine()  # 여기서 jax/haiku/nucleotide_transformer가 import & 모델 로딩됨

    n_rows = 0
    n_na = 0
    with open_maybe_gzip(cadd) as fin, open(out_path, "w") as fout:
        fout.write("#chr\tpos\tref\talt\tn_vpr\tn_cadd\n")

        for line in fin:
            if line.startswith("#"):
                continue
            cols = line.strip().split("\t")
            if len(cols) < 5:
                continue
            chrom, pos, ref_allele, alt_allele, cadd_raw = (
                cols[0], int(cols[1]), cols[2], cols[3], float(cols[4])
            )
            if chrom_filter is not None and chrom != chrom_filter:
                continue

            cadd_score = normalize_cadd(cadd_raw, max_cadd=max_cadd)

            if cadd_only:
                fout.write(f"{chrom}\t{pos}\t{ref_allele}\t{alt_allele}\tNA\t{cadd_score}\n")
                n_rows += 1
                n_na += 1
                continue

            or_score = engine.score_variant(chrom, pos, ref_allele, alt_allele, fasta)
            if or_score is None:
                fout.write(f"{chrom}\t{pos}\t{ref_allele}\t{alt_allele}\tNA\t{cadd_score}\n")
                n_rows += 1
                n_na += 1
                continue

            risk_score = normalize_or(or_score)
            print(f"{chrom}\t{pos}\t{or_score}", file=sys.stderr)
            fout.write(f"{chrom}\t{pos}\t{ref_allele}\t{alt_allele}\t{risk_score}\t{cadd_score}\n")
            n_rows += 1

    sys.stderr.write(
        f"[calc_vprPrep] Done. {n_rows} rows written to {out_path} "
        f"({n_na} as NA).\n"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute variant-level VPR(+CADD) scores from a prepped CADD table."
    )
    parser.add_argument(
        "--cadd", required=True,
        help="Preprocessed CADD table (tsv[.gz]), e.g. output of prep_vprscore_input.py.",
    )
    parser.add_argument(
        "--fasta", required=False, default=None,
        help="Reference genome FASTA (indexed with samtools faidx). Not needed with --cadd-only.",
    )
    parser.add_argument(
        "--out", required=True,
        help="Output variant-level table (tsv): #chr pos ref alt n_vpr n_cadd",
    )
    parser.add_argument(
        "--cadd-only", action="store_true",
        help="Skip the NT model entirely; only compute CADD-based scores (n_vpr=NA). "
             "No jax/haiku/nucleotide_transformer required.",
    )
    parser.add_argument(
        "--chrom", required=False, default=None,
        help="Only process this chromosome (as written in the CADD file). "
             "Useful to parallelize by chromosome; merge results with merge_vprPrep.py.",
    )
    parser.add_argument(
        "--max-cadd", type=float, required=False, default=DEFAULT_MAX_CADD_RAW,
        help="Upper clip bound for CADD normalization: n_cadd = clip(raw,0,max_cadd)/max_cadd. "
             "This assumes --cadd column 5 is CADD RawScore (not PHRED). Default "
             f"({DEFAULT_MAX_CADD_RAW}) is the ~99.9th percentile RawScore measured "
             "genome-wide on whole_genome_SNVs.tsv.gz; recompute for a different CADD "
             "version/build and pass it explicitly if it differs meaningfully.",
    )
    args = parser.parse_args()
    if not args.cadd_only and not args.fasta:
        parser.error("--fasta is required unless --cadd-only is set.")
    return args


def main():
    args = parse_args()
    run_vpr_prep(
        cadd=args.cadd,
        fasta=args.fasta,
        out_path=args.out,
        cadd_only=args.cadd_only,
        chrom_filter=args.chrom,
        max_cadd=args.max_cadd,
    )


if __name__ == "__main__":
    main()
