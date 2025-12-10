import argparse
import os
import math
import re
import gzip


def open_maybe_gzip(path):
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "r")

def extract_sample_ids(vcf_path):
    """
    """
    with open_maybe_gzip(vcf_path) as f:
        for line in f:
            if line.startswith("#CHROM"):
                return line.strip().split("\t")[9:]
    raise ValueError(f"{vcf_path}에서 #CHROM 라인을 찾을 수 없습니다.")


def norm_chrom(chrom):
    c = chrom.strip()
    if c.lower().startswith("chr"):
        return c[3:]
    return c

def load_variant_table(variant_table_path):
    """
    """
    variant_dict = {}
    with open_maybe_gzip(variant_table_path) as f:
        for line in f:
            if not line.strip():
                continue
            if line.startswith("#"):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 6:
                continue
            chrom, pos, ref, alt, n_vpr, n_cadd = cols[:6]
            key = (norm_chrom(chrom), int(pos), ref, alt)
            variant_dict[key] = (float(n_vpr), float(n_cadd))
    return variant_dict

def compute_single_variant_score(n_vpr, n_cadd, alpha):
    """
    s_i = α * n_vpr + (1 - α) * n_cadd
    """
    return alpha * n_vpr + (1.0 - alpha) * n_cadd


def main():
    parser = argparse.ArgumentParser(description="Compute sample-level VPRscores from a merged VCF and precomputed per-variant scores.")
    parser.add_argument(
        '--vcf',
        required=True,
        help='Merged multi-sample biallelic VCF (e.g. merged.biallelic.vcf.gz)'
    )
    parser.add_argument(
        '--vprPrep',
        required=True,
        help='Step1 output file with columns: #chr pos ref alt n_vpr n_cadd'
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.5,
        help='Variant-level weight for combining n_vpr and n_cadd (default: 0.5)'
    )
    parser.add_argument(
        '--beta',
        type=float,
        default=0.2,
        help='Scaling parameter for variant-count weighting in VPRscore (default: 0.2)'
    )
    parser.add_argument(
        '--output_file',
        default="sample_vprscore.tsv",
        help='Output file name (default: sample_vprscore.tsv)'
    )
    args = parser.parse_args()

    vcf_path = args.vcf
    variant_table_path = args.vprPrep
    alpha = args.alpha
    beta = args.beta
    output_file = args.output_file
    variant_table = load_variant_table(variant_table_path)

    sample_ids = extract_sample_ids(vcf_path)
    num_samples = len(sample_ids)
    sample_scores = [0.0] * num_samples
    sample_counts = [0] * num_samples

    with open_maybe_gzip(vcf_path) as f:
        for line in f:
            if not line.strip():
                continue
            if line.startswith("#"):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 10:
                continue
            chrom_raw = cols[0]
            pos = int(cols[1])
            ref = cols[3]
            alt = cols[4]
            if "," in alt:
                continue
            key = (norm_chrom(chrom_raw), pos, ref, alt)
            if key not in variant_table:
                continue
            n_vpr, n_cadd = variant_table[key]
            s_i = compute_single_variant_score(n_vpr, n_cadd, alpha)
            fmt = cols[8].split(":")
            gt_idx = fmt.index("GT") if "GT" in fmt else 0
            sample_fields = cols[9:]

            for j in range(num_samples):
                sample_field = cols[9 + j]
                fields = sample_field.split(":")
                if gt_idx >= len(fields):
                    continue
                gt = fields[gt_idx]

                if gt in {".", "./.", ".|."}:
                    continue
                alleles = re.split(r"[\/|]", gt)
                if "1" in alleles:
                    sample_scores[j] += s_i
                    sample_counts[j] += 1
    
    with open(output_file, "w") as out:
        print("Sample_ID\tScore_Sum\tCount\tAverage\ttVPRscore", file=out)
        for s, total, n in zip(sample_ids, sample_scores, sample_counts):
            if n == 0:
                mean = 0.0
                vprs = 0.0
            else:
                mean = total / n
                vprs = mean * (1 + beta * math.log(n))
            out.write(f"{s}\t{total:.6f}\t{n}\t{mean:.6f}\t{vprs:.6f}\n")





if __name__ == "__main__":
    main()
