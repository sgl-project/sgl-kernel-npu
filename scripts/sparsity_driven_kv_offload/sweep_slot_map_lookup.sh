#!/usr/bin/env bash
# Sweep slot_map_lookup vs any+argmax benchmark across hit ratios and block dims.
#
# Usage:
#   bash scripts/sparsity_driven_kv_offload/sweep_slot_map_lookup.sh
#
# Override params via environment:
#   HIT_RATIOS="0.5 1.0" BLOCK_DIMS="8 24" \
#     bash scripts/sparsity_driven_kv_offload/sweep_slot_map_lookup.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BENCH_SCRIPT="${REPO_ROOT}/benchmark/sparsity_driven_kv_offload/bench_slot_map_lookup.py"

# ---------- tunable parameter lists ----------
HIT_RATIOS="${HIT_RATIOS:-0.5}"
BLOCK_DIMS="${BLOCK_DIMS:-8 16 24 32 48}"
TOPK_LENS="${TOPK_LENS:-2048}"

# fixed arguments
SIZE="${SIZE:-16}"
BS="${BS:-4}"
MAX_CTX="${MAX_CTX:-131072}"
DEVICE_LEN="${DEVICE_LEN:-12800}"
WARMUP="${WARMUP:-3}"
ITERS="${ITERS:-10}"
ACC_REPEAT="${ACC_REPEAT:-1}"
PERF_REPEAT="${PERF_REPEAT:-5}"
SEED="${SEED:-20260514}"

cd "${REPO_ROOT}"

for topk_len in ${TOPK_LENS}; do
  for block_dim in ${BLOCK_DIMS}; do
    for hit_ratio in ${HIT_RATIOS}; do
      echo "============================================================"
      echo "slot_map bench: topk_len=${topk_len} block_dim=${block_dim} hit_ratio=${hit_ratio}"
      echo "============================================================"
      python3 "${BENCH_SCRIPT}" \
        --size "${SIZE}" \
        --bs "${BS}" \
        --topk-len "${topk_len}" \
        --max-context-len "${MAX_CTX}" \
        --device-len "${DEVICE_LEN}" \
        --hit-ratio "${hit_ratio}" \
        --block-dim "${block_dim}" \
        --warmup "${WARMUP}" \
        --iters "${ITERS}" \
        --accuracy-repeat "${ACC_REPEAT}" \
        --perf-repeat "${PERF_REPEAT}" \
        --seed "${SEED}"
      echo ""
    done
  done
done
