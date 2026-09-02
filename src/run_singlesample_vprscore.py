#!/usr/bin/env python3
"""
run_singlesample_vprscore.py

기존 REAME의 "one-step" 사용법과의 호환을 위해 남겨둔 wrapper 스크립트.
내부적으로는 calc_vprPrep.py -> aggregate_vprscore.py 를 순서대로 실행한다
(둘 다 이제 vpr_engine.py를 공유하는 동일 로직).

single-sample VCF를 가정하지만, sample이 여러 개인 VCF를 넣어도
aggregate_vprscore.py가 알아서 각 샘플별 결과를 낸다.
"""

import argparse
import os
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(
        description="One-step VPRscore for a single-sample VCF "
                    "(wrapper around calc_vprPrep.py + aggregate_vprscore.py)."
    )
    parser.add_argument("--vcf", required=True,
                         help="Single-sample biallelic VCF (used to read genotypes for aggregation).")
    parser.add_argument("--fasta", required=False, default=None,
                         help="Reference genome FASTA. Not needed with --cadd-only.")
    parser.add_argument("--cadd", required=True, help="Preprocessed CADD table (tsv[.gz]).")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--out", required=True, help="Final sample-level VPRscore output file.")
    parser.add_argument("--cadd-only", action="store_true",
                         help="Skip the NT model; score using CADD only.")
    parser.add_argument("--max-cadd", type=float, required=False, default=None,
                         help="Forwarded to calc_vprPrep.py --max-cadd (CADD RawScore clip bound). "
                              "Omit to use calc_vprPrep.py's default.")
    args = parser.parse_args()

    if not args.cadd_only and not args.fasta:
        parser.error("--fasta is required unless --cadd-only is set.")

    here = os.path.dirname(os.path.abspath(__file__))
    fd, prep_path = tempfile.mkstemp(suffix=".vprPrep.txt")
    os.close(fd)

    try:
        prep_cmd = [
            sys.executable, os.path.join(here, "calc_vprPrep.py"),
            "--cadd", args.cadd,
            "--out", prep_path,
        ]
        if args.cadd_only:
            prep_cmd.append("--cadd-only")
        else:
            prep_cmd += ["--fasta", args.fasta]
        if args.max_cadd is not None:
            prep_cmd += ["--max-cadd", str(args.max_cadd)]
        sys.stderr.write("[run_singlesample_vprscore] " + " ".join(prep_cmd) + "\n")
        subprocess.run(prep_cmd, check=True)

        agg_cmd = [
            sys.executable, os.path.join(here, "aggregate_vprscore.py"),
            "--vcf", args.vcf,
            "--vprPrep", prep_path,
            "--alpha", str(args.alpha),
            "--beta", str(args.beta),
            "--out", args.out,
        ]
        sys.stderr.write("[run_singlesample_vprscore] " + " ".join(agg_cmd) + "\n")
        subprocess.run(agg_cmd, check=True)
    finally:
        if os.path.exists(prep_path):
            os.remove(prep_path)


if __name__ == "__main__":
    main()
