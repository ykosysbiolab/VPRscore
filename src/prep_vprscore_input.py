#!/usr/bin/env python3

import argparse
import gzip
import os
import subprocess
import sys
import tempfile
import shutil


def ensure_bgzipped_vcf(vcf_path):
    """
    """
    if vcf_path.endswith(".vcf.gz"):
        return vcf_path, False  # (path, is_temp)

    if vcf_path.endswith(".vcf"):
        gz_path = vcf_path + ".gz"
        sys.stderr.write(f"[prep_vpr_inputs] bgzip compressing {vcf_path} -> {gz_path}\n")
        with open(gz_path, "wb") as out:
            subprocess.run(["bgzip", "-c", vcf_path], stdout=out, check=True)
            subprocess.run(["tabix", gz_path], stdout=out, check=True)
        return gz_path, True

    sys.stderr.write(
        f"[prep_vpr_inputs] Warning: {vcf_path} extension not .vcf/.vcf.gz, "
        "bcftools may fail.\n"
    )
    return vcf_path, False


def open_maybe_gzip(path):
    """Open a file that may be gzipped."""
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "r")


def filter_vcf_by_regions(vcf_path, regions_path, out_vcf_path):
    """
    Use bcftools to filter VCF by BED regions and produce a bgzipped VCF + index.
    """
    vcf_bgz, is_temp = ensure_bgzipped_vcf(vcf_path)

    cmd_view = [
        "bcftools", "view",
        "-R", regions_path,
        "-Oz",
        "-o", out_vcf_path,
        vcf_bgz,
    ]
    sys.stderr.write("[prep_vpr_inputs] Running: " + " ".join(cmd_view) + "\n")
    subprocess.run(cmd_view, check=True)

    cmd_index = ["bcftools", "index", out_vcf_path]
    sys.stderr.write("[prep_vpr_inputs] Running: " + " ".join(cmd_index) + "\n")
    subprocess.run(cmd_index, check=True)
    
    # if is_temp:
    #     try:
    #         os.remove(vcf_bgz)
    #         os.remove(vcf_bgz + ".csi")
    #     except FileNotFoundError:
    #         pass



def collect_variant_keys_from_vcf(vcf_path):
    """
    Read a (possibly gzipped) VCF and collect (chrom, pos, ref, alt) keys
    for all variant lines.

    Assumes biallelic VCF (ALT에 콤마 없는 상태).
    """
    variant_keys = set()
    n_lines = 0

    with open_maybe_gzip(vcf_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 5:
                continue
            chrom = cols[0]
            pos = int(cols[1])
            ref = cols[3]
            alt = cols[4]
            if "," in alt:
                sys.stderr.write(
                    f"[prep_vpr_inputs] Warning: multi-allelic site found and skipped: "
                    f"{chrom}:{pos} {ref}>{alt}\n"
                )
                continue
            variant_keys.add((chrom, pos, ref, alt))
            n_lines += 1

    sys.stderr.write(
        f"[prep_vpr_inputs] Collected {len(variant_keys)} unique variants "
        f"from filtered VCF ({n_lines} lines).\n"
    )
    return variant_keys


def strip_chr(chrom: str) -> str:
    c = chrom.strip()
    if c.lower().startswith("chr"):
        return c[3:]
    return c


def subset_cadd(cadd_path, regions_path, variant_keys, out_cadd_path):
    """
    From a (possibly gzipped) CADD file, extract only rows matching
    (chrom, pos, ref, alt) in variant_keys.

    Assumes CADD header contains columns like: Chrom, Pos, Ref, Alt, RawScore ...
    """
    matched = 0
    total = 0

    variant_keys_nochr = set(
        (strip_chr(chrom), pos, ref, alt)
        for chrom, pos, ref, alt in variant_keys
    )
    cmd = ["tabix", cadd_path, "-R", regions_path]
    sys.stderr.write("[prep_vpr_inputs] Running: " + " ".join(cmd) + "\n")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        text=True,
    )

    with gzip.open(out_cadd_path, "wt") as fout:
        for line in proc.stdout:
            if not line.strip():
                continue
            cols = line.split("\t")
            
            try:
                chrom = strip_chr(cols[0])  # 혹시라도 chr 있으면 한 번 더 정리
                pos = int(cols[1])
                ref = cols[2]
                alt = cols[3]
                raw = cols[4]
            except (IndexError, ValueError):
                continue
            
            total += 1
            key = (chrom, pos, ref, alt)
            if key in variant_keys_nochr:
                matched += 1
                fout.write(f"{chrom}\t{pos}\t{ref}\t{alt}\t{raw}\n")

    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"tabix exited with code {ret}")

    sys.stderr.write(
        f"[prep_vpr_inputs] Finished CADD subset: matched {matched} variants "
        f"out of ~{total} scanned lines.\n"
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Prepare VPRscore inputs by filtering VCF to target regions and "
            "subsetting CADD to the resulting variants."
        )
    )
    parser.add_argument(
        "--vcf",
        required=True,
        help="Input biallelic VCF (.vcf or .vcf.gz).",
    )
    parser.add_argument(
        "--cadd",
        required=True,
        help="CADD whole-genome file (.tsv or .tsv.gz).",
    )
    parser.add_argument(
        "--regions",
        required=True,
        help="BED file with target regions.",
    )
    parser.add_argument(
        "--out-prefix",
        required=True,
        help="Prefix for output files (e.g. prep/sample1).",
    )
    args = parser.parse_args()

    out_vcf = args.out_prefix + ".filtered.vcf.gz"
    out_cadd = args.out_prefix + ".cadd.tsv.gz"

    # 1) VCF를 BED로 필터링
    filter_vcf_by_regions(args.vcf, args.regions, out_vcf)

    # 2) 필터링된 VCF에서 variant key 수집
    variant_keys = collect_variant_keys_from_vcf(out_vcf)

    # 3) CADD에서 해당 variant만 subset
    subset_cadd(args.cadd, args.regions ,variant_keys, out_cadd)

    sys.stderr.write(
        f"[prep_vpr_inputs] Done.\n"
        f"  Filtered VCF : {out_vcf}\n"
        f"  Subset CADD  : {out_cadd}\n"
    )


if __name__ == "__main__":
    main()
