"""
vpr_engine.py

VPRscore의 variant-level 계산 로직을 한 곳에 모아둔 공용 모듈.
- calc_vprPrep.py 에서 import 해서 씀
- jax / haiku / nucleotide_transformer 는 무거운 GPU 의존성이라, 실제로
  VPREngine 인스턴스를 만들 때(=NT 모델을 쓸 때)만 import 한다.
  즉 --cadd-only 로 돌리는 경우에는 이 패키지들이 설치되어 있지 않아도
  이 모듈을 문제없이 import 할 수 있다.
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
# 공용 유틸
# ---------------------------------------------------------------------------

def open_maybe_gzip(path):
    """gzip이든 아니든 알아서 열어주는 헬퍼."""
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "r")


def norm_chrom(chrom):
    """'chr19' -> '19' 처럼 chr 접두어만 벗겨서 '매칭용 키'를 만든다.
    실제 출력/조회에는 원본 chrom 표기를 그대로 써야 한다 (아래 prep_vprscore_input.py 참고).
    """
    c = chrom.strip()
    if c.lower().startswith("chr"):
        return c[3:]
    return c


def normalize_or(or_val, epsilon=1e-4):
    """Odds ratio -> [0,1] 위험도 점수. OR>=1(발현 확률이 오히려 증가)인
    경우는 위험도 0으로 clip 한다 (감소 방향만 '위험'으로 취급하는 설계)."""
    or_clipped = min(max(or_val, epsilon), 1)
    return -math.log10(or_clipped) / -math.log10(epsilon)


# CADD RawScore(cols[4]에서 뽑은 값, PHRED 아님) 기준 99.9th percentile.
# UKB whole_genome_SNVs.tsv.gz 전체(9억+ 줄, NR%1000 systematic sampling,
# 2회 재현: 5.781617 / 5.656496)를 스트리밍으로 훑어서 얻은 값.
# 원래 하드코딩돼 있던 max_cadd=30은 PHRED 스케일(PHRED>=30 = 상위 0.1%)
# 관습에서 온 숫자라 RawScore(대부분 -6~+6대)에는 안 맞았음 -- 이 값으로 교체.
DEFAULT_MAX_CADD_RAW = 5.656496


def normalize_cadd(cadd_score, max_cadd=DEFAULT_MAX_CADD_RAW):
    """CADD RawScore -> [0,1] 위험도 점수.
    음수 RawScore(=관찰된/정상 변이 패턴에 더 가까움)는 위험도 0으로 clip
    (normalize_or가 OR>=1을 위험도 0으로 clip하는 것과 동일한 설계 원칙).
    max_cadd는 상위 0.1%(99.9th percentile) 지점으로, PHRED>=30 관습을
    RawScore 스케일로 옮겨온 값."""
    clipped = max(cadd_score, 0.0)
    return min(clipped, max_cadd) / max_cadd


def find_variant_token(tokens, variant_offset):
    """토큰화된 시퀀스에서 variant_offset(문자 단위 위치)이 속한 토큰의
    인덱스를 찾는다. 못 찾으면 (None, None, valid_token_count) 반환."""
    sequence_tokens = tokens[0][1:]  # CLS 토큰 제외
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
    """samtools faidx로 chrom:start-end 서열을 가져온다.
    shell=True 대신 리스트 인자로 호출 (인젝션/따옴표 문제 방지),
    실패하면 stderr에 이유를 남기고 빈 문자열을 돌려준다."""
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
# NT 모델을 실제로 쓰는 부분 (--cadd-only 가 아닐 때만 인스턴스화됨)
# ---------------------------------------------------------------------------

class VPREngine:
    """Nucleotide Transformer 기반 odds-ratio 계산기.
    인스턴스를 만드는 순간 모델이 로딩된다 (무거운 작업)."""

    def __init__(self, model_name=MODEL_NAME, max_tokens=MAX_TOKENS,
                 input_length=INPUT_LENGTH):
        # 무거운 의존성은 여기서만 import (지연 import)
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
        """시퀀스 하나를 토큰화 -> (필요시) truncate -> 모델 forward.
        ref/var 양쪽 모두 동일하게 이 함수를 통해서 truncate 되므로
        (예전 버그였던) 비대칭 truncation이 구조적으로 불가능하다."""
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
        """(chrom,pos,ref,alt) 하나에 대한 odds ratio를 계산한다.
        SNV가 아니거나(ref/alt 길이 != 1), 서열을 못 가져오거나,
        토큰 위치를 못 찾으면 None을 반환한다 (호출부에서 NA로 기록)."""
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
