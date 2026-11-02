#!/usr/bin/env bash
# Run from a Linux environment with CANN, torch_npu and NPU Triton installed.
set -euo pipefail
arch=${1:-}
case "$arch" in
    2201) ops=alloc_extend,assign_cache_op ;;
    3510) ops=alloc_extend,assign_cache_op,kv_compress_epilog ;;
    *) echo "Usage: bash $0 {2201|3510} [--tp2]" >&2; exit 2 ;;
esac
if (( $# > 2 )) || [[ -n ${2:-} && $2 != --tp2 ]]; then
    echo "Optional second argument: --tp2" >&2; exit 2
fi
: "${ASCEND_HOME_PATH:?Initialize your CANN environment first}"
python=${PYTHON:-python3}
cd "$(dirname "$0")/.."
git rev-parse HEAD
"$python" setup.py --make_clean --arch-version "$arch" --ops "$ops" --artifact wheel
shopt -s nullglob
wheels=(output/sgl_kernel_npu-*.whl)
if (( ${#wheels[@]} != 1 )); then
    echo "Expected one wheel in output/, found ${#wheels[@]}" >&2; exit 1
fi
"$python" -m pip install --force-reinstall --no-deps "${wheels[0]}"

# Enable this wheel's vendor before pytest or torchrun initializes torch_npu.
vendor=$("$python" -c 'from importlib.metadata import distribution; print(distribution("sgl-kernel-npu").locate_file("sgl_kernel_npu/_cann/vendors/sgl_kernel_npu").resolve())')
export ASCEND_CUSTOM_OPP_PATH="$vendor${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
"$python" - "$arch" "$ops" "$vendor" "${2:-}" <<'PY'
import json
import sys
from pathlib import Path

import sgl_kernel_npu
import torch
from sgl_kernel_npu.operators.utils.generation import get_hardware

arch, ops, vendor, tp = sys.argv[1:]
root = Path(sgl_kernel_npu.__file__).resolve().parent
manifest = json.loads((root / "_build_manifest.json").read_text())
assert (root / "_cann/vendors/sgl_kernel_npu").resolve() == Path(vendor), root
assert manifest["build_mode"] == "REFACTOR" and manifest["arch_version"] == arch, manifest
assert set(manifest["operators"]) == set(ops.split(",")), manifest
assert str(get_hardware().arch_id) == arch, get_hardware()
assert str(root / "lib/libsgl_kernel_npu.so") in torch.ops.loaded_libraries, root
if tp and torch.npu.device_count() < 2:
    raise SystemExit("--tp2 requires two visible NPUs")
print(f"Testing installed package {root}, CANN operators {ops}")
PY

base=tests/python/operators
common=(--skip-acc-check=false --enable-perf=false -v -s)
run_test() { "$python" -m pytest "$@" "${common[@]}"; }
for impl in ascendc torch_native; do
    run_test "$base/mem_cache/test_alloc_extend.py" --impl="$impl" --impl-ref=cpu
done
run_test "$base/mem_cache/test_assign_cache_op.py" --impl=ascendc --impl-ref=torch_native
if [[ $arch == 3510 ]]; then
    run_test "$base/mem_cache/test_kv_compress_epilog.py" --impl=ascendc --impl-ref=cpu
    run_test "$base/mem_cache/test_kv_compress_epilog.py" -k 'not graph_replay' --impl=torch_native --impl-ref=cpu
fi
tp_test="$base/norm/test_split_qkv_tp_rmsnorm_rope.py"
for impl in triton torch_native; do
    run_test "$tp_test" -k single_rank --impl="$impl" --impl-ref=cpu
    if [[ ${2:-} == --tp2 ]]; then
        "$python" -m torch.distributed.run --standalone --nproc-per-node=2 -m pytest \
            "$tp_test" -k two_rank --impl="$impl" --impl-ref=cpu "${common[@]}"
    fi
done
