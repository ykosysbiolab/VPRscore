# VPRscore (refactored)

Nucleotide-Transformer 기반 variant risk score(VPR) + CADD를 결합해
variant-level, sample-level 점수를 계산하는 파이프라인.

## 구조

```
src/
  vpr_engine.py             공용 로직 (서열 추출, NT 모델 추론, 정규화).
                             jax/haiku/nucleotide_transformer는 VPREngine()을
                             실제로 만들 때만 import됨 (lazy) -> --cadd-only
                             모드는 이 무거운 패키지들 없이도 동작함.
  prep_vprscore_input.py    1단계: VCF를 target region(BED)으로 필터링하고,
                             CADD 파일을 그 variant들로 subset.
  calc_vprPrep.py           2단계: variant-level 점수 계산 (genotype 불필요).
                             --cadd-only, --chrom 지원.
  merge_vprPrep.py          2-b단계: --chrom으로 나눠 돌린 결과들을 하나로 병합
                             (중복 variant는 경고 후 첫 값 유지).
  aggregate_vprscore.py     3단계: variant-level 테이블 + VCF genotype ->
                             sample-level VPRscore. single/multi-sample VCF 겸용.
  run_singlesample_vprscore.py
                             (하위호환용) calc_vprPrep.py + aggregate_vprscore.py를
                             순서대로 실행하는 원샷 wrapper.
env/
  requirements-cadd-only.txt  --cadd-only 전용 최소 의존성 (표준 라이브러리만 사용).
  requirements-full.txt       NT 모델까지 쓸 때 필요한 전체 의존성 (jax/haiku 등).
  environment.yml             conda 환경 (기존 유지).
```

## 이전 버전과의 차이 (버그 수정 포함)

- `calc_multisample_vprPrep.py` / `run_singlesample_vprscore.py`에 중복돼 있던
  모델·서열 로직을 `vpr_engine.py` 하나로 통합.
- **[버그 수정]** REF 시퀀스가 ALT 시퀀스와 다르게 `max_tokens`로 잘리지 않던
  비대칭 truncation 버그 수정 (`_run_forward`가 양쪽에 동일하게 적용됨).
- **[버그 수정]** `prep_vprscore_input.py`가 CADD subset 출력에 chr 접두어를
  무조건 제거해서 쓰던 것을 수정 -> 이제 VCF의 원본 chrom 표기를 그대로
  유지해서 씀. 참조 FASTA가 `chr19`처럼 접두어 있는 명명을 쓰는 경우에도
  `samtools faidx` 조회가 깨지지 않음.
- **[버그 수정]** `run_single_sample_vprscore()`에서 variant가 0개일 때
  `ZeroDivisionError` 나던 것 수정.
- **[버그 수정]** `find_variant_token()`이 variant 위치를 못 찾을 때
  (`None` 반환) 방어 코드 없이 바로 인덱싱하던 부분 수정 -> 이제 해당
  variant는 경고 로그를 남기고 NA로 처리, 파이프라인이 죽지 않음.
- **[버그 수정]** SNV 여부 체크가 ALT만 하고 REF는 안 하던 것 수정 (`len(ref)==1` 추가).
- `--vcf`, `--regions` 등 실제로 안 쓰이던 죽은 인자 정리.
- `env/requirements.txt`의 로컬 절대경로(`file:///mss_dc/...`) 의존성을
  git 설치 방식으로 교체 (`requirements-full.txt`).
- **[정규화 수정]** `normalize_cadd()`의 `max_cadd` 기본값을 `30`에서
  `5.656496`으로 교체. `--cadd` 입력의 5번째 컬럼(`cols[4]`)은 CADD의
  **RawScore**이지 PHRED가 아닌데(공식 CADD tsv 컬럼 순서:
  `Chrom Pos Ref Alt RawScore PHRED`), 기존 `max_cadd=30`은 PHRED 관습
  (PHRED>=30 = 상위 0.1%)에서 온 값이라 RawScore(대부분 -6~+6대)에
  적용하면 거의 모든 variant가 0 근처로 눌리고, 음수 RawScore는 그대로
  음수 `n_cadd`가 되는 문제가 있었음. 새 기본값은
  `whole_genome_SNVs.tsv.gz` 전체를 systematic sampling(NR%1000)으로
  스트리밍해서 구한 RawScore의 **99.9th percentile**(두 차례 독립
  샘플링에서 5.781617 / 5.656496으로 재현) -- PHRED>=30 관습을
  RawScore 스케일로 옮겨온 값. 동시에 음수 RawScore는 위험도 0으로
  clip하도록 함 (`normalize_or`가 OR>=1을 0으로 clip하는 것과 동일한
  설계 원칙). `calc_vprPrep.py --max-cadd`로 오버라이드 가능
  (CADD 버전/빌드가 다르면 재계산해서 넘길 것).
  **주의**: 이 변경은 기존 결과와 수치가 달라짐 -- 이전 결과와 비교하려면
  `--max-cadd 30`으로 예전 동작을 재현할 수 있음.

## 신규 기능

### CADD-only 모드
```bash
python3 src/calc_vprPrep.py \
  --cadd example/tmp_interval.cadd.tsv.gz \
  --out ./prep.cadd_only.txt \
  --cadd-only
```
`jax`/`haiku`/`nucleotide_transformer`가 설치되어 있지 않아도 동작합니다
(`env/requirements-cadd-only.txt`만으로 충분). 출력의 `n_vpr` 컬럼은 `NA`로
채워지고, 다음 단계(`aggregate_vprscore.py`)가 이를 자동으로 인식해
CADD 점수만으로 계산합니다 (`alpha` 무시).

### chr 단위 병렬 처리 + 병합
```bash
# chr별로 병렬 실행
python3 src/calc_vprPrep.py --cadd cadd.19.tsv.gz --fasta ref.fa --chrom 19 --out prep.19.txt
python3 src/calc_vprPrep.py --cadd cadd.20.tsv.gz --fasta ref.fa --chrom 20 --out prep.20.txt

# 결과 병합 (중복 variant는 경고 후 첫 값 유지)
python3 src/merge_vprPrep.py --inputs prep.19.txt prep.20.txt --out prep.merged.txt
```

## 전체 파이프라인 예시 (multi-sample)

```bash
python3 src/prep_vprscore_input.py \
  --vcf example/example_multi.vcf.gz \
  --cadd example/full_cadd.tsv.gz \
  --regions example/tmp_interval.bed \
  --out-prefix ./prep/multi

python3 src/calc_vprPrep.py \
  --cadd ./prep/multi.cadd.tsv.gz \
  --fasta example/chr19.fa \
  --out ./multi_prep.txt

python3 src/aggregate_vprscore.py \
  --vcf ./prep/multi.filtered.vcf.gz \
  --vprPrep ./multi_prep.txt \
  --alpha 0.5 --beta 0.2 \
  --out ./multisample_vprscore.tsv
```

Single-sample도 동일한 스크립트를 그대로 씁니다 (VCF에 샘플이 1개뿐이면 됨).
원샷으로 하고 싶으면:

```bash
python3 src/run_singlesample_vprscore.py \
  --vcf ./prep/sample1.filtered.vcf.gz \
  --fasta example/chr19.fa \
  --cadd ./prep/sample1.cadd.tsv.gz \
  --alpha 0.5 --beta 0.2 \
  --out ./singlesample_vprscore.tsv
```

## 참고: example/ 데이터의 CADD 값

`example/tmp_interval*.cadd.tsv.gz`의 5번째 컬럼 값(3.8~6.9대)은 실제
RawScore 분포(-6~+6대, 아래 참고)보다는 PHRED에 가까운 범위입니다.
데모용으로 만들어진 값으로 보이며, 새 기본 `max_cadd`(RawScore 기준)를
적용하면 대부분 1.0으로 clip됩니다 -- 파이프라인 동작 자체엔 문제
없지만, 실제 CADD RawScore subset으로 재현 테스트할 때는 이 값이
그대로 재현되지 않을 수 있습니다.

## 검증 상태

- 모든 스크립트 `py_compile` / `ast.parse` 통과.
- `--cadd-only` 경로는 example 데이터(single/multi 둘 다)로 end-to-end 실행 확인
  (jax/haiku 미설치 상태에서도 정상 동작 확인).
- `merge_vprPrep.py`의 중복 variant 감지/경고 로직 실행 확인.
- **NT 모델 경로(`--cadd-only` 없이)는 이 환경에 jax/haiku/nucleotide_transformer가
  설치되어 있지 않아 실행 검증을 못 했습니다.** `_run_forward`/`_compute_odds_ratio`
  로직은 원본 코드 흐름을 그대로 유지(양쪽 truncation만 수정)했으므로 동작상
  차이는 없을 것으로 예상되지만, GPU 환경에서 실제 모델로 한 번 돌려서
  기존 결과와 대조해보는 걸 권장합니다.
