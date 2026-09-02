#!/usr/bin/env python3
"""
merge_vprPrep.py

calc_vprPrep.py를 --chrom으로 나눠서 병렬로 돌린 결과 파일들
(혹은 --cadd-only로 만든 결과와 모델로 만든 결과가 섞인 경우도)
하나의 vprPrep 테이블로 합친다.

- 헤더는 한 번만 씀
- (chrom,pos,ref,alt) key가 여러 파일에 중복되면 경고를 남기고 처음 값만 유지
- 입력 파일들은 순서 보장 없이 그대로 이어붙임 (정렬이 필요하면 별도로)
"""

import argparse
import sys

from vpr_engine import open_maybe_gzip

HEADER = "#chr\tpos\tref\talt\tn_vpr\tn_cadd\n"


def merge(inputs, out_path):
    seen = {}
    n_rows = 0
    n_dupe = 0
    n_malformed = 0

    with open(out_path, "w") as fout:
        fout.write(HEADER)
        for path in inputs:
            with open_maybe_gzip(path) as f:
                for line in f:
                    if not line.strip() or line.startswith("#"):
                        continue
                    cols = line.rstrip("\n").split("\t")
                    if len(cols) < 6:
                        n_malformed += 1
                        sys.stderr.write(
                            f"[merge_vprPrep] Warning: malformed line in {path}, "
                            f"skipped: {line.strip()}\n"
                        )
                        continue
                    key = tuple(cols[:4])
                    if key in seen:
                        n_dupe += 1
                        sys.stderr.write(
                            f"[merge_vprPrep] Warning: duplicate variant {key} "
                            f"in {path} (first seen in {seen[key]}), keeping first.\n"
                        )
                        continue
                    seen[key] = path
                    fout.write(line if line.endswith("\n") else line + "\n")
                    n_rows += 1

    sys.stderr.write(
        f"[merge_vprPrep] Merged {len(inputs)} file(s) -> {out_path}: "
        f"{n_rows} rows, {n_dupe} duplicates skipped, {n_malformed} malformed lines skipped.\n"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge chr-split (or otherwise chunked) vprPrep output files into one table."
    )
    parser.add_argument(
        "--inputs", required=True, nargs="+",
        help="vprPrep output files to merge, space-separated (supports .gz).",
    )
    parser.add_argument("--out", required=True, help="Merged output table (tsv).")
    return parser.parse_args()


def main():
    args = parse_args()
    merge(args.inputs, args.out)


if __name__ == "__main__":
    main()
