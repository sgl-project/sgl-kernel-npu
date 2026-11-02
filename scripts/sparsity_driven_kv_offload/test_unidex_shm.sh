#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
NPU_DEVICE_ID="${1:-${NPU_DEVICE_ID:-0}}"

export PYTHONPATH="${REPO_ROOT}/python/sgl_kernel_npu${PYTHONPATH:+:${PYTHONPATH}}"

cd "${REPO_ROOT}"

echo "[1/2] Running UniDex Copy unit tests on NPU ${NPU_DEVICE_ID}"

python3 - "${NPU_DEVICE_ID}" \
  "${REPO_ROOT}/tests/python/sgl_kernel_npu/test_unidex_copy.py" <<'PY'
import runpy
import sys

import torch
import torch_npu  # noqa: F401

device_id = int(sys.argv[1])
test_file = sys.argv[2]

torch.npu.set_device(device_id)
sys.argv = [test_file, "-v"]
runpy.run_path(test_file, run_name="__main__")
PY

echo "[2/2] Running H2D/D2H benchmark smoke test"

python3 benchmark/sparsity_driven_kv_offload/bench_unidex_copy.py \
  --directions h2d d2h \
  --baselines unidex torch_index_copy \
  --device-id "${NPU_DEVICE_ID}" \
  --batch-size 1 \
  --topk 8 \
  --src-rows 32 \
  --dst-rows 8 \
  --dtype float16 \
  --head-num 1 \
  --head-dim 16 \
  --token-bytes 32 \
  --block-dim 8 \
  --hit-rates 0.0 0.5 1.0 \
  --warmup 1 \
  --perf-iters 5 \
  --accuracy-iters 2

echo "UniDex SHM tests passed."
