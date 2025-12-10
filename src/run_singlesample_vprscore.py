import argparse
import os
import subprocess
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import sys
import re
import math
import gzip
from nucleotide_transformer.pretrained import get_pretrained_model


def get_sequence_from_fasta(chrom, start, end, fasta_file):
    cmd = f"samtools faidx {fasta_file} {chrom}:{start}-{end}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    lines = result.stdout.strip().split("\n")
    if len(lines) <= 1:
        return ""
    return "".join(lines[1:]).replace("\n", "")

def get_variant_sequence(chrom, start, end, fasta_file, vcf_file):
    cmd = f"samtools faidx {fasta_file} {chrom}:{start}-{end} | bcftools consensus {vcf_file}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    lines = result.stdout.strip().split("\n")
    if len(lines) <= 1:
        return "" 
    return "".join(lines[1:]).replace("\n", "")

def compute_odds_ratio_v2(ref_seq, var_seq, tokenizer, parameters, forward_fn, max_tokens, target_offset):
    ref_logits, var_logits,reftargetTokenIdx,targetTokenIdx,target_token_id, ref_target_token_id, seq_length, ref_seq_length  = run_nt_model_v2(ref_seq, var_seq, tokenizer, parameters, forward_fn, max_tokens, target_offset)
    ref_logits = jnp.squeeze(ref_logits, axis=0)
    var_logits = jnp.squeeze(var_logits, axis=0)
    ref_logits = ref_logits[1 : (ref_seq_length)]
    var_logits = var_logits[1 : (seq_length)]
    ref_prob = jax.nn.softmax(ref_logits, axis=-1)
    var_prob = jax.nn.softmax(var_logits, axis=-1)
    return_prob = var_prob[targetTokenIdx][target_token_id]
    ref_return_prob = ref_prob[reftargetTokenIdx][ref_target_token_id]
    # Log likelihood
    log_likelihood_ref = jnp.log(ref_return_prob)
    log_likelihood_var = jnp.log(return_prob)
    # Log odds ratio
    log_odds_ratio = log_likelihood_var - log_likelihood_ref
    odds_ratio = jnp.exp(log_odds_ratio)
    return odds_ratio

def run_nt_model_v2(ref_sequence,sequence, tokenizer, parameters, forward_fn, max_tokens, target_offset):
    sequences = [sequence.upper()]
    tokens_ids = [b[1] for b in tokenizer.batch_tokenize(sequences)]
    tokens_str = [b[0] for b in tokenizer.batch_tokenize(sequences)]
    ref_sequences = [ref_sequence.upper()]
    ref_tokens_ids = [b[1] for b in tokenizer.batch_tokenize(ref_sequences)]
    ref_tokens_str = [b[0] for b in tokenizer.batch_tokenize(ref_sequences)]
    if len(tokens_ids[0]) > max_tokens:
        tokens_ids[0] = tokens_ids[0][:max_tokens]
    token_offset, target_token_str, seq_length  = find_variant_token(tokens_str,target_offset)
    ref_token_offset, ref_target_token_str, ref_seq_length  = find_variant_token(ref_tokens_str,target_offset)
    # print("ref: ", ref_target_token_str, "alt: ", target_token_str)
    ref_tokens = jnp.asarray(ref_tokens_ids, dtype=jnp.int32)
    tokens = jnp.asarray(tokens_ids, dtype=jnp.int32)
    random_key = jax.random.PRNGKey(0)
    ref_output = forward_fn.apply(parameters, random_key, ref_tokens)
    output = forward_fn.apply(parameters, random_key, tokens)
    target_token_id = tokenizer.token_to_id(target_token_str)
    ref_target_token_id = tokenizer.token_to_id(ref_target_token_str)
    ref_logits = ref_output["logits"]
    var_logits = output["logits"]
    logits = output["logits"]
    return ref_logits,logits, ref_token_offset,token_offset, target_token_id, ref_target_token_id, seq_length, ref_seq_length

def find_variant_token(tokens, variant_offset):
    sequence_tokens = tokens[0][1:]  
    full_sequence = ""
    token_start_positions = []  
    valid_token_count = 0
    for token in sequence_tokens:
        token_start_positions.append(len(full_sequence))  
        full_sequence += token
        if token != "<pad>": 
            valid_token_count += 1
    for i, start_pos in enumerate(token_start_positions):
        end_pos = start_pos + len(sequence_tokens[i]) 
        if start_pos <= variant_offset < end_pos:
            return i, sequence_tokens[i], valid_token_count  
    return None, None, valid_token_count  


def normalize_or(or_val, epsilon=1e-4):
    or_clipped = min(max(or_val, epsilon), 1)
    return -math.log10(or_clipped) / -math.log10(epsilon)

def normalize_cadd(cadd_score, max_cadd=30):
    return min(cadd_score, max_cadd) / max_cadd

def compute_single_variant_score(risk, cadd=None, alpha=0.5):
    if cadd is None:
        return risk
    return alpha * risk + (1 - alpha) * cadd

def compute_prs(scores, beta=0.2):
    n = len(scores)
    if n == 0:
        return 0
    
    return (sum(scores) / n) * (1 + beta * math.log(n))
    
def open_maybe_gzip(path):
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "r")


def run_single_sample_vprscore(vcf, fasta, cadd, alpha, beta, out_path):
    """
    Single-sample VPRscore pipeline (one-step mode).
    """
    input_length = 10000
    max_tokens = 2000
    model_name = "500M_multi_species_v2"

    parameters, forward_fn, tokenizer, config = get_pretrained_model(
        model_name=model_name, max_positions=max_tokens
    )
    forward_fn = hk.transform(forward_fn)

    scores = []
    scores_with_cadd = []
    with open_maybe_gzip(cadd) as file:
        for line in file:
            if line.startswith("#"):
                continue
            cols = line.strip().split("\t")
            chrom, target_pos, ref_allele, alt_allele, cadd_raw = cols[0], int(cols[1]), cols[2], cols[3], float(cols[4])
            if alt_allele not in ["A", "C", "G", "T"]:
                continue
            results = chrom+"\t"+str(target_pos)
            start = target_pos - (input_length // 2)
            end = target_pos + (input_length // 2)
            target_offset = target_pos - (start) 
            ref_seq = get_sequence_from_fasta(chrom, start, end, fasta)
            ref_seq = re.sub(r"[^ACGTacgt]", "N", ref_seq)
            
            seq_list = list(ref_seq)
            center_idx = input_length // 2   
            seq_list[center_idx] = alt_allele
            var_seq = "".join(seq_list)
            
            or_score = compute_odds_ratio_v2(ref_seq, var_seq, tokenizer, parameters, forward_fn, max_tokens, target_offset)
            results = results+"\t"+str(or_score)
            print(results, file=sys.stderr)
            
            risk_score = normalize_or(or_score)
            cadd_score = normalize_cadd(cadd_raw)
            single_score_cadd = compute_single_variant_score(risk_score, cadd_score, alpha=alpha)
            scores_with_cadd.append(single_score_cadd)
    
    prs = compute_prs(scores_with_cadd, beta=beta)
    with open(out_path, "w") as fout:
        fout.write("# VPRscore (single-sample mode)\n")
        fout.write(f"VPRscore\t{prs:.6f}\n")

    print(prs)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run single-sample VPRscore pipeline from a single-sample VCF."
    )
    parser.add_argument(
        "--vcf",
        required=True,
        help="Single-sample biallelic VCF (e.g. sample1.biallelic.vcf.gz)",
    )
    parser.add_argument(
        "--fasta",
        required=True,
        help="Reference genome FASTA (indexed with samtools faidx)",
    )
    parser.add_argument(
        "--regions",
        required=False,
        help="Target region BED file where VPRscore should be computed.",
    )
    parser.add_argument(
        "--cadd",
        required=True,
        help="Preprocessed CADD table (tsv[.gz]) matching the reference build.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Weight for combining sequence-based risk and CADD (default: 0.5).",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.2,
        help="Scaling parameter for variant-count weighting (default: 0.2).",
    )
    parser.add_argument(
        "--out",
        required=True,
        default="out.txt",
        help="Output file for this sample's VPRscore "
             "(e.g. ./singlesample_vprscore.txt)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_single_sample_vprscore(
        vcf=args.vcf,
        fasta=args.fasta,
        cadd=args.cadd,
        alpha=args.alpha,
        beta=args.beta,
        out_path = args.out
    )

if __name__ == "__main__":
    main()
