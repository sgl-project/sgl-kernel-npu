#!/usr/bin/env bash
# Run from the PR #939 sgl-kernel-npu checkout, or pass its path as argument 1.
# Usage: bash test_pr939_a5_cann91.sh /path/to/sgl-kernel-npu
set -euo pipefail

repo_dir="${1:-$PWD}"
cd "$repo_dir"
python_bin="${PYTHON_BIN:-python3}"
npu_device="${NPU_DEVICE:-0}"
result_dir="$PWD/test-results/a5-cann91-$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$result_dir"

files=(
  tests/python/operators/attention/test_sparse_flash_mla_metadata.py
  tests/python/operators/attention/test_sparse_flash_mla.py
  tests/python/operators/attention/test_mixed_quant_sparse_flash_mla_metadata.py
  tests/python/operators/attention/test_mixed_quant_sparse_flash_mla.py
  tests/python/operators/attention/test_quant_lightning_indexer.py
  tests/python/operators/attention/test_quant_lightning_indexer_metadata.py
  tests/python/operators/attention/test_compressor.py
  tests/python/operators/kvcacheio/test_kv_compress_epilog.py
  tests/python/operators/kvcacheio/test_indexer_quant_cache.py
  tests/python/operators/norm/test_mhc_pre.py
  tests/python/operators/norm/test_mhc_post.py
  tests/python/operators/rope/test_inplace_partial_rotary_mul.py
  tests/python/operators/moe/test_moe_gating_top_k.py
)
for test_file in "${files[@]}"; do
  if [[ ! -f "$test_file" ]]; then
    echo "Missing test file: $test_file. Use the PR #939 checkout." >&2
    exit 2
  fi
done

"$python_bin" - "$npu_device" <<'PY' 2>&1 | tee "$result_dir/environment.log"
import sys
import torch
import torch_npu
import sgl_kernel_npu
from sgl_kernel_npu.operators.utils._transformer_compat import (
    cann_backend,
    detect_cann_version,
)
from sgl_kernel_npu.operators.utils.generation import Generation, get_hardware

version = detect_cann_version()
backend = cann_backend()
if tuple(map(int, version.split('.')))[:2] != (9, 1):
    raise SystemExit(f'Expected CANN 9.1.x, got {version}')
if backend != 'legacy':
    raise SystemExit(f'Unexpected backend: {backend}')
if not torch.npu.is_available():
    raise SystemExit('NPU is unavailable')
device = int(sys.argv[1])
torch.npu.set_device(device)
name = torch.npu.get_device_name(device)
if get_hardware().generation != Generation.ARCH35:
    raise SystemExit(f'This test suite requires ARCH35 (A5 target), got {name}')
# Print the selected A5 chip.
print('Python:', sys.executable)
print('Torch:', torch.__version__)
print('CANN:', version)
print('Backend:', backend)
print('Device:', device, name, '(A5)')
print('Kernel package:', sgl_kernel_npu.__file__)
print('Legacy custom/native operators must be installed; the kernel loader loads them on demand.')
PY

printf '\nRunning PR #939 A5-supported accuracy tests (performance disabled).\n'
printf 'Compressor BSH runs only if enabled in your test file; its C++ fix requires rebuilding.\n'
printf 'Results: %s\n\n' "$result_dir"
set +e
"$python_bin" -m pytest "${files[@]}" \
  --impl auto --impl-ref torch_native \
  --npu-device "$npu_device" \
  --skip-acc-check=false --enable-perf=false \
  -v -s -ra \
  --junitxml="$result_dir/results.xml" \
  2>&1 | tee "$result_dir/pytest.log"
pipeline_status=("${PIPESTATUS[@]}")
set -e
pytest_status="${pipeline_status[0]}"
log_status="${pipeline_status[1]}"
printf '\npytest exit code: %s\nResults: %s\n' "$pytest_status" "$result_dir"
if [[ "$pytest_status" -ne 0 ]]; then
  exit "$pytest_status"
fi
exit "$log_status"
