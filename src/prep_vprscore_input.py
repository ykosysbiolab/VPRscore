#!/usr/bin/env python3

import argparse
import gzip
import subprocess
import sys

from vpr_engine import open_maybe_gzip, norm_chrom


def ensure_bgzipped_vcf(vcf_path):
    """Return as-is if it's already .vcf.gz; if it's .vcf, bgzip + tabix-index it first."""
    if vcf_path.endswith(".vcf.gz"):
        return vcf_path, False  # (path, is_temp)

    if vcf_path.endswith(".vcf"):
        gz_path = vcf_path + ".gz"
        sys.stderr.write(f"[prep_vpr_inputs] bgzip compressing {vcf_path} -> {gz_path}\n")
        with open(gz_path, "wb") as out:
            subprocess.run(["bgzip", "-c", vcf_path], stdout=out, check=True)
        subprocess.run(["tabix", "-p", "vcf", gz_path], check=True)
        return gz_path, True

    sys.stderr.write(
        f"[prep_vpr_inputs] Warning: {vcf_path} extension not .vcf/.vcf.gz, "
        "bcftools may fail.\n"
    )
    return vcf_path, False


def filter_vcf_by_regions(vcf_path, regions_path, out_vcf_path):
    """Filter a VCF to BED regions with bcftools, producing a bgzipped, indexed output."""
    vcf_bgz, is_temp = ensure_bgzipped_vcf(vcf_path)

    cmd_view = ["bcftools", "view", "-R", regions_path, "-Oz", "-o", out_vcf_path, vcf_bgz]
    sys.stderr.write("[prep_vpr_inputs] Running: " + " ".join(cmd_view) + "\n")
    subprocess.run(cmd_view, check=True)

    cmd_index = ["bcftools", "index", out_vcf_path]
    sys.stderr.write("[prep_vpr_inputs] Running: " + " ".join(cmd_index) + "\n")
    subprocess.run(cmd_index, check=True)

    if is_temp:
        import os
        for p in (vcf_bgz, vcf_bgz + ".tbi"):
            try:
                os.remove(p)
            except FileNotFoundError:
                pass


def collect_variant_keys_from_vcf(vcf_path):
    """
    Collect (chrom, pos, ref, alt) keys from the filtered VCF.
    chrom is kept exactly as written in the VCF (subset_cadd below rewrites
    the CADD output using this original notation, so it always matches the
    FASTA/samtools faidx naming convention).
    Assumes a biallelic VCF (no comma in ALT).
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


def subset_cadd(cadd_path, regions_path, variant_keys, out_cadd_path):
    """
    Extract only the (chrom, pos, ref, alt) rows present in variant_keys from
    the CADD file and write them to out_cadd_path.

    Matching is done regardless of a 'chr' prefix (via norm_chrom), but the
    **output always uses the VCF's original chrom notation.** The CADD
    file's chrom notation (usually no prefix) can differ from the VCF/FASTA's
    (e.g. 'chr19'); the next step (calc_vprPrep.py) passes this output file's
    chrom column straight to samtools faidx, so failing to unify the notation
    here causes a silent sequence-fetch failure (eventually crashing with an
    IndexError).
    """
    matched = 0
    total = 0

    # norm_chrom(key) -> original VCF chrom notation
    key_to_orig_chrom = {
        (norm_chrom(chrom), pos, ref, alt): chrom
        for chrom, pos, ref, alt in variant_keys
    }

    cmd = ["tabix", cadd_path, "-R", regions_path]
    sys.stderr.write("[prep_vpr_inputs] Running: " + " ".join(cmd) + "\n")

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)

    with gzip.open(out_cadd_path, "wt") as fout:
        for line in proc.stdout:
            if not line.strip():
                continue
            cols = line.split("\t")
            try:
                chrom = norm_chrom(cols[0])
                pos = int(cols[1])
                ref = cols[2]
                alt = cols[3]
                raw = cols[4].strip()
            except (IndexError, ValueError):
                continue

            total += 1
            key = (chrom, pos, ref, alt)
            if key in key_to_orig_chrom:
                matched += 1
                orig_chrom = key_to_orig_chrom[key]
                fout.write(f"{orig_chrom}\t{pos}\t{ref}\t{alt}\t{raw}\n")

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
    parser.add_argument("--vcf", required=True, help="Input biallelic VCF (.vcf or .vcf.gz).")
    parser.add_argument("--cadd", required=True, help="CADD whole-genome file (.tsv or .tsv.gz).")
    parser.add_argument("--regions", required=True, help="BED file with target regions.")
    parser.add_argument("--out-prefix", required=True, help="Prefix for output files (e.g. prep/sample1).")
    args = parser.parse_args()

    out_vcf = args.out_prefix + ".filtered.vcf.gz"
    out_cadd = args.out_prefix + ".cadd.tsv.gz"

    filter_vcf_by_regions(args.vcf, args.regions, out_vcf)
    variant_keys = collect_variant_keys_from_vcf(out_vcf)
    subset_cadd(args.cadd, args.regions, variant_keys, out_cadd)

    sys.stderr.write(
        f"[prep_vpr_inputs] Done.\n"
        f"  Filtered VCF : {out_vcf}\n"
        f"  Subset CADD  : {out_cadd}\n"
    )


if __name__ == "__main__":
    main()
