#!/usr/bin/env python3
"""
merge_vprPrep.py

Merges the output files from running calc_vprPrep.py in parallel with
--chrom splits (or a mix of --cadd-only and NT-model outputs) into a single
vprPrep table.

- The header is written only once.
- If a (chrom,pos,ref,alt) key is duplicated across files, a warning is
  logged and only the first value is kept.
- Input files are concatenated in the given order with no re-sorting
  (sort separately if needed).
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
