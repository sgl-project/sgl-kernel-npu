#!/usr/bin/env bash
# Sweep unidex_copy vs torch_index_copy benchmark across hit rates and block dims.
#
# Usage:
#   bash scripts/enumerate_bench_unidex_copy.sh
#
# Override params via environment:
#   HIT_RATES="0.5 1.0" BLOCK_DIMS="8 24" bash scripts/enumerate_bench_unidex_copy.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_SCRIPT="${SCRIPT_DIR}/../benchmark/bench_unidex_copy.py"

# ---------- tunable parameter lists ----------
HIT_RATES="${HIT_RATES:-0.5}"
BLOCK_DIMS="${BLOCK_DIMS:-8 16 24 32 48}"
TOPK_LIST="${TOPK_LIST:-2048}"

# fixed arguments
BATCH_SIZE="${BATCH_SIZE:-8}"
SRC_ROWS="${SRC_ROWS:-131072}"
DST_ROWS="${DST_ROWS:-0}"
SRC_IDX_MODE="${SRC_IDX_MODE:-random}"
DST_IDX_MODE="${DST_IDX_MODE:-arange}"
DTYPE="${DTYPE:-float16}"
HEAD_NUM="${HEAD_NUM:-1}"
HEAD_DIM="${HEAD_DIM:-576}"
TOKEN_BYTES="${TOKEN_BYTES:-1152}"
WARMUP="${WARMUP:-5}"
PERF_ITERS="${PERF_ITERS:-50}"
ACC_ITERS="${ACC_ITERS:-1}"
SEED="${SEED:-20260609}"

cd "${SCRIPT_DIR}/.."

for topk in ${TOPK_LIST}; do
  for block_dim in ${BLOCK_DIMS}; do
    for hit_rate in ${HIT_RATES}; do
      echo "============================================================"
      echo "unidex_copy bench: topk=${topk} block_dim=${block_dim} hit_rate=${hit_rate}"
      echo "============================================================"
      python3 "${BENCH_SCRIPT}" \
        --directions d2d \
        --baselines unidex torch_index_copy \
        --batch-size "${BATCH_SIZE}" \
        --topk "${topk}" \
        --src-rows "${SRC_ROWS}" \
        --dst-rows "${DST_ROWS}" \
        --src-index-mode "${SRC_IDX_MODE}" \
        --dst-index-mode "${DST_IDX_MODE}" \
        --hit-rate "${hit_rate}" \
        --dtype "${DTYPE}" \
        --head-num "${HEAD_NUM}" \
        --head-dim "${HEAD_DIM}" \
        --token-bytes "${TOKEN_BYTES}" \
        --block-dim "${block_dim}" \
        --warmup "${WARMUP}" \
        --perf-iters "${PERF_ITERS}" \
        --accuracy-iters "${ACC_ITERS}" \
        --seed "${SEED}"
      echo ""
    done
  done
done
