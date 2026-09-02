"""
vpr_engine.py

Shared module for VPRscore's variant-level scoring logic.
- Imported by calc_vprPrep.py.
- jax / haiku / nucleotide_transformer are heavy GPU dependencies, so they are
  only imported when a VPREngine instance is actually created (i.e. when the
  NT model is used). This means --cadd-only mode can import this module
  without those packages installed.
"""

import gzip
import math
import re
import subprocess
import sys

INPUT_LENGTH = 10000
MAX_TOKENS = 2000
MODEL_NAME = "500M_multi_species_v2"


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def open_maybe_gzip(path):
    """Open a file transparently, whether or not it's gzip-compressed."""
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "r")


def norm_chrom(chrom):
    """Strip a 'chr' prefix (e.g. 'chr19' -> '19') to build a matching key.
    The original chrom notation must still be used for output/lookup
    (see prep_vprscore_input.py)."""
    c = chrom.strip()
    if c.lower().startswith("chr"):
        return c[3:]
    return c


def normalize_or(or_val, epsilon=1e-4):
    """Odds ratio -> [0,1] risk score. OR>=1 (probability actually increases)
    is clipped to risk 0 (only the decreasing direction is treated as 'risk'
    by design)."""
    or_clipped = min(max(or_val, epsilon), 1)
    return -math.log10(or_clipped) / -math.log10(epsilon)


# 99.9th percentile of CADD RawScore (the value pulled from cols[4], NOT
# PHRED). Measured by streaming the entire UKB whole_genome_SNVs.tsv.gz
# (900M+ lines, NR%1000 systematic sampling, reproduced twice: 5.781617 /
# 5.656496). The old hardcoded max_cadd=30 came from PHRED-scale convention
# (PHRED>=30 = top 0.1%) and didn't match the RawScore scale (mostly -6..+6)
# -- replaced with this value.
DEFAULT_MAX_CADD_RAW = 5.656496


def normalize_cadd(cadd_score, max_cadd=DEFAULT_MAX_CADD_RAW):
    """CADD RawScore -> [0,1] risk score.
    Negative RawScore (= looks more like an observed/benign-pattern variant)
    is clipped to risk 0 (same design principle as normalize_or clipping
    OR>=1 to risk 0). max_cadd is the top 0.1% (99.9th percentile) point,
    i.e. the PHRED>=30 convention carried over to the RawScore scale."""
    clipped = max(cadd_score, 0.0)
    return min(clipped, max_cadd) / max_cadd


def find_variant_token(tokens, variant_offset):
    """Find the index of the token that contains variant_offset (a
    character-level position) in a tokenized sequence.
    Returns (None, None, valid_token_count) if not found."""
    sequence_tokens = tokens[0][1:]  # drop the CLS token
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


def get_sequence_from_fasta(chrom, start, end, fasta_file):
    """Fetch the chrom:start-end sequence via samtools faidx.
    Uses a list of args instead of shell=True (avoids injection/quoting
    issues); on failure, logs the reason to stderr and returns an empty
    string."""
    cmd = ["samtools", "faidx", fasta_file, f"{chrom}:{start}-{end}"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(
            f"[vpr_engine] samtools faidx failed for {chrom}:{start}-{end} "
            f"(fasta={fasta_file}): {result.stderr.strip()}\n"
        )
        return ""
    lines = result.stdout.strip().split("\n")
    if len(lines) <= 1:
        return ""
    return "".join(lines[1:]).replace("\n", "")


# ---------------------------------------------------------------------------
# NT model usage (only instantiated when NOT running --cadd-only)
# ---------------------------------------------------------------------------

class VPREngine:
    """Nucleotide Transformer-based odds-ratio calculator.
    The model is loaded as soon as an instance is created (heavyweight)."""

    def __init__(self, model_name=MODEL_NAME, max_tokens=MAX_TOKENS,
                 input_length=INPUT_LENGTH):
        # Heavy dependencies are only imported here (lazy import)
        import haiku as hk
        import jax
        import jax.numpy as jnp
        from nucleotide_transformer.pretrained import get_pretrained_model

        self._jax = jax
        self._jnp = jnp
        self.max_tokens = max_tokens
        self.input_length = input_length

        parameters, forward_fn, tokenizer, config = get_pretrained_model(
            model_name=model_name, max_positions=max_tokens
        )
        self.parameters = parameters
        self.forward_fn = hk.transform(forward_fn)
        self.tokenizer = tokenizer
        self.config = config

    def _run_forward(self, sequence):
        """Tokenize a sequence -> truncate (if needed) -> model forward pass.
        Both ref and var sequences go through this same function, so the
        asymmetric truncation bug that existed before is now structurally
        impossible."""
        jnp = self._jnp
        tok = self.tokenizer.batch_tokenize([sequence.upper()])
        tokens_str = [b[0] for b in tok]
        tokens_ids = [b[1] for b in tok]
        if len(tokens_ids[0]) > self.max_tokens:
            tokens_ids[0] = tokens_ids[0][: self.max_tokens]
            tokens_str[0] = tokens_str[0][: self.max_tokens]
        tokens = jnp.asarray(tokens_ids, dtype=jnp.int32)
        random_key = self._jax.random.PRNGKey(0)
        output = self.forward_fn.apply(self.parameters, random_key, tokens)
        return output["logits"], tokens_str

    def _compute_odds_ratio(self, ref_seq, var_seq, target_offset):
        jnp = self._jnp
        jax = self._jax
        ref_logits, ref_tokens_str = self._run_forward(ref_seq)
        var_logits, var_tokens_str = self._run_forward(var_seq)

        ref_idx, ref_token, ref_len = find_variant_token(ref_tokens_str, target_offset)
        var_idx, var_token, var_len = find_variant_token(var_tokens_str, target_offset)
        if ref_idx is None or var_idx is None:
            sys.stderr.write(
                "[vpr_engine] Warning: variant offset outside tokenized "
                "sequence (possibly truncated), skipping variant.\n"
            )
            return None

        ref_token_id = self.tokenizer.token_to_id(ref_token)
        var_token_id = self.tokenizer.token_to_id(var_token)

        ref_logits = jnp.squeeze(ref_logits, axis=0)[1:ref_len]
        var_logits = jnp.squeeze(var_logits, axis=0)[1:var_len]
        ref_prob = jax.nn.softmax(ref_logits, axis=-1)[ref_idx][ref_token_id]
        var_prob = jax.nn.softmax(var_logits, axis=-1)[var_idx][var_token_id]

        log_odds_ratio = jnp.log(var_prob) - jnp.log(ref_prob)
        return float(jnp.exp(log_odds_ratio))

    def score_variant(self, chrom, pos, ref_allele, alt_allele, fasta_file):
        """Compute the odds ratio for a single (chrom, pos, ref, alt).
        Returns None if it's not a SNV (len(ref) != 1 or len(alt) != 1),
        the sequence can't be fetched, or the token position can't be found
        (the caller records this as NA)."""
        if len(ref_allele) != 1 or len(alt_allele) != 1 or alt_allele.upper() not in "ACGT":
            return None

        half = self.input_length // 2
        start = pos - half
        end = pos + half
        target_offset = pos - start

        ref_seq = get_sequence_from_fasta(chrom, start, end, fasta_file)
        if not ref_seq or len(ref_seq) <= target_offset:
            sys.stderr.write(
                f"[vpr_engine] Warning: could not fetch sequence for "
                f"{chrom}:{pos}, skipping.\n"
            )
            return None
        ref_seq = re.sub(r"[^ACGTacgt]", "N", ref_seq)

        seq_list = list(ref_seq)
        seq_list[target_offset] = alt_allele.upper()
        var_seq = "".join(seq_list)

        return self._compute_odds_ratio(ref_seq, var_seq, target_offset)
