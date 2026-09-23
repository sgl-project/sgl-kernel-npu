# A5 native INT8 FuseEP validation

This port enables `FuseMode.DISPATCH_FFN_COMBINE` (mode 2) on Ascend950. It is
the kernel dependency for the INT8 A5 regression reported while testing
[SGLang #40516](https://github.com/sgl-project/sglang/pull/40516).

Both grouped matrix multiplies retain INT8 inputs and weights with INT32
accumulation. Per-channel weight scales are passed to the A5 Fixpipe; the
vector epilogues apply per-token scales, SwiGLU, INT8 requantization, and
BF16 output conversion. Checkpoints are not converted to FP8 or BF16.

The A5 build includes the operator definition, tiling and ACLNN wrappers.
The shared pipeline selects CATLASS v1.6.1's `Ascend950` copies and Fixpipe,
uses CANN's A5 HCCL context accessors and communication engine, and obtains
the routing core count and UB size from the platform. A3 keeps its existing
CATLASS dependency, HCCL ABI and routing configuration.

## Validation status

This port is not ready for use until A5 numerical validation passes. The
initial port and loader follow-up compiled in A5/A3 CI, but manual A5 testing
at 17 tokens, hidden 2048, gate/up width 1536, 128 experts and top-k 8 produced
finite unfused outputs and NaN fused outputs on all four ranks.

The follow-up corrects GEMM2's per-token dequantization: its `Cast` and `Muls`
calls used `isSetMask=false` without explicitly setting the vector mask.
Ascend950's count-based APIs do not reset the SPR mask used by those calls.
The epilogue now passes explicit element counts for the tile and each row,
avoiding dependence on a previous operation's mask. Both matrix multiplies
still use INT8 inputs and weights with INT32 accumulation.

This fixes an identified mask-state dependency; it is not yet proof that the
reported A5 NaNs are fully resolved. Rerun the distributed kernel and SGLang
regressions below with the rebuilt wheel. Static checks and compilation do
not establish NPU numerical correctness.

The [epilogue mask regression](../tests/ascendc/fuseep_epilogue/README.md)
isolates the production dequantization code on one NPU with NaN-filled UB
and explicit full/restricted masks. On A3, the old implementation failed
all nine restricted-mask cases; the corrected implementation passed all
18 cases exactly. The test also compiles for Ascend950 with CANN 9.1.

## Build on A5

Use the A5 CANN and torch-npu environment supported by this repository. The
build fetches CATLASS v1.6.1 and requires the system `moe_distribute_base.h`
already used by A5 `FusedDeepMoe`.

```bash
bash build.sh -a deepep Ascend950
python -m pip install --force-reinstall --no-deps output/deep_ep*.whl
```

Verify the installed custom operator exports both symbols:

```bash
find output python/deep_ep -name 'libcust_opapi.so' -exec \
  nm -D --defined-only {} \; | grep -E 'aclnnDispatchFFNCombine(GetWorkspaceSize)?$'
```

Use a fresh process after installing the wheel. A stale DeepEP/custom OPP
installation still produces the missing-`aclnnDispatchFFNCombine` exception.

The runtime loads `libcust_opapi.so` relative to the actual `deep_ep_cpp`
extension before consulting the process search path. Changing
`LD_LIBRARY_PATH` inside Python does not update glibc's startup search path;
an older external custom library must not override the library in the wheel.
Missing-symbol errors report the selected library path and any load failure.
The A5 build checks both exports after installing the custom OPP and stops
before creating the wheel if either symbol is absent. Kernel build failures
also stop the build script.

If the same error persists, run this read-only check with the Python
interpreter and environment used to launch SGLang, and save the output:

```bash
python - <<'PY'
import ctypes
import importlib.metadata
import importlib.util
import os
import sys
from pathlib import Path

import torch
import torch_npu

print("Python:", sys.executable)
print("DeepEP version:", importlib.metadata.version("deep_ep"))
spec = importlib.util.find_spec("deep_ep")
print("DeepEP package:", spec.origin)
package = Path(spec.origin).resolve().parent
library = package / "vendors/hwcomputing/op_api/lib/libcust_opapi.so"
print("Bundled library:", library, "exists:", library.is_file())
if library.is_file():
    try:
        handle = ctypes.CDLL(str(library), mode=os.RTLD_NOW | os.RTLD_LOCAL)
        for name in ("aclnnDispatchFFNCombine", "aclnnDispatchFFNCombineGetWorkspaceSize"):
            print(name, "exported:", hasattr(handle, name))
    except OSError as error:
        print("Load failed:", error)
PY
```

An absent export means the installed custom OPP does not contain this port;
a load error identifies a missing dependency or incompatible library.
If both exports are present, compare the package path with the traceback
and the library path in the updated runtime error, and restart all workers
after installing the matching DeepEP wheel. The host-only loader regression
can be run on Linux with `python3 tests/python/deepep/test_op_api_loader.py -v`;
it does not compile or validate the NPU kernel.

## Distributed INT8 kernel regression

The existing test compares against unfused INT8 dispatch, grouped matmul,
SwiGLU quantization and combine. Its reference explicitly selects INT8 on
A5 (where `use_fp8=True` selects FP8), asserts integer tensor dtypes, checks
output error and exact expert token counts, and can repeat on the same
buffers to exercise communication state reuse. The original mean absolute
error threshold of `1e-2` is unchanged.

The test reports whether expert receive counts match before checking output
values. Non-finite outputs fail explicitly with the rank, NaN/Inf counts and
sample coordinates, before calculating an error metric. Save these lines
and use `--debug` to include the weight formats if a failure persists.

From the repository root, run the Qwen3-30B-A3B expert shapes (hidden 2048,
gate/up width 1536, 128 experts, top-k 8) at decode, tail and prefill sizes:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export HCCL_BUFFSIZE=200
export DEEPEP_HCCL_BUFFSIZE=200
for tokens in 1 17 128; do
  timeout 600s python tests/python/deepep/test_dispatch_ffn_combine.py \
    --num-processes 4 --num-tokens "$tokens" --hidden 2048 \
    --moe-intermediate-size 1536 --num-experts 128 --num-topk 8 \
    --repeat 3 --skip-benchmark || exit 1
done

# Concentrate traffic on one rank; other ranks receive no expert tokens.
timeout 600s python tests/python/deepep/test_dispatch_ffn_combine.py \
  --num-processes 4 --num-tokens 17 --hidden 2048 \
  --moe-intermediate-size 1536 --num-experts 128 --num-topk 8 \
  --active-ranks 0 --repeat 3 --skip-benchmark
```

Also rebuild for A3 (`bash build.sh -a deepep Ascend910_9382`), install that
wheel on A3 and run the same regression. Do not reuse an A5 wheel on A3.

## Isolate a remaining numerical failure

The A5 run after the GEMM2 mask fix, before the routing synchronization fix,
produced finite outputs and matching expert counts but mean errors of
0.206055-0.261719 (limit 0.01). After installing `bc12a46`, the reported
17-token reproduction passed all three repetitions on four ranks with mean
errors of 0.001182556-0.001762390. All three single-device diagnostics below
also passed on A5. The full-model logprob regression still fails; these
standalone successes do not establish full-model correctness.

Input dynamic quantization also lacked a V-to-S event between `ReduceMax` and
the scalar `GetValue` that reads its result. `PIPE_V` only orders vector work;
the kernel build disables automatic synchronization. Both full-load and
gather paths now synchronize this dependency and wait for the scalar read
before reusing the scale buffer on Vector. Before/after full-load tests both
passed on A3; the A5 distributed improvement above followed this correction.

Run the following single-device diagnostics on A5 from this repository root.
They compile the production headers directly, without installing a DeepEP
wheel. The epilogue tests use the CATLASS headers fetched by the DeepEP build.
Use a free device and pass its visible device index as the last argument.

```bash
cmake -S tests/ascendc/fuseep_routing -B build/fuseep-routing \
  -DASC_DIR="$ASCEND_HOME_PATH/compiler/tikcpp/ascendc_kernel_cmake" \
  -DCATLASS_ARCH=3510
cmake --build build/fuseep-routing -j2
./build/fuseep-routing/test_routing_quant 0

cmake -S tests/ascendc/fuseep_epilogue -B build/fuseep-epilogue \
  -DASC_DIR="$ASCEND_HOME_PATH/compiler/tikcpp/ascendc_kernel_cmake" \
  -DCATLASS_ARCH=3510
cmake --build build/fuseep-epilogue -j2
./build/fuseep-epilogue/test_swiglu 0
./build/fuseep-epilogue/test_epilogue 0
```

Save the complete outputs and exit codes. `test_routing_quant` checks the
input INT8 values, scales, routing indices and expert counts exactly.
`test_swiglu` isolates per-token dequantization, activation and requantization
between the two GEMMs. `test_epilogue` checks GEMM2's per-token dequantization
with exact reference values. These tests do not cover the INT8 matrix kernels,
HCCL transport or final unpermute/combine; a pass narrows the investigation
but does not replace the unchanged distributed regression above.

## SGLang regression

With the corresponding SGLang #40516 changes and the rebuilt A5 wheel,
run the existing INT8 checkpoint without skipping A5:

```bash
PYTHONPATH="$PWD/python:${PYTHONPATH:-}" \
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
SGLANG_TEST_MODEL_PATH=/path/to/Qwen-MoE-W8A8 \
SGLANG_TEST_TP_SIZE=4 \
SGLANG_TEST_FUSEEP_MODES=2 \
SGLANG_TEST_LOG_DIR=/tmp/fuseep-a5-int8 \
python -u test/registered/npu/basic_function/parameter/test_npu_fuseep_mode.py -v
```

Run this command from the SGLang repository. It compares the same INT8
checkpoint with the unfused backend, including batched prefill/decode,
concurrent arithmetic requests and teacher-forced logprobs (mean absolute
difference `< 0.1`, maximum `< 0.6`). Use the Qwen3.5 model override for the
shared-expert regression. Performance and graph execution need separate
validation; these commands make no performance claim.

### Compare real MoE inputs when the standalone test passes

The standalone reference feeds GEMM1's INT32 output into fused dequantization
and SwiGLU. SGLang's unfused path instead rounds GEMM1's dequantized output to
BF16 before activation. It also uses expert TP for the `none` backend and EP
for FuseEP. A full-model comparison alone cannot isolate these differences
from a kernel error.

Run the diagnostic below in the same environment/checkpoint as the failing
test. No wheel rebuild is required. It copies SGLang's Python sources to the
new output directory, instruments mode 2 and runs the original regression.
The installed sources, checkpoint, fused outputs, assertions and exit status
are preserved. Allow approximately 70 MiB for the private source copy.
The test and both servers run with the same Python interpreter and explicitly
select that source copy after Python startup. Their logs print
`[FUSEEP_DIAGNOSTIC_IMPORT]` with the selected source path. The launcher also
bypasses environment proxies for localhost and streams the test console.

```bash
python tests/python/deepep/diagnose_sglang_fuseep.py \
  --sglang-root /home/wzy/sgl-sglang \
  --devices 0,1,2,3 \
  --output-dir /tmp/fuseep-real-input-a5
```

The model defaults to the original test's configuration, including
`SGLANG_TEST_MODEL_PATH`; use `--model /path/to/Qwen-MoE-W8A8` to override it.
Choose free devices and a new output directory for each run. An unchanged
full-model assertion failure is expected if the precision issue persists.

For each layer and each first-seen prefill size (at least four local tokens
by default), the diagnostic replays the same hidden states, expert IDs,
router weights and loaded expert weights through two unfused INT8 references:

- INT32 GEMM1, dequantization/SwiGLU/requantization, BF16 GEMM2 and combine.
- BF16 GEMM1, SwiGLU/requantization, BF16 GEMM2 and combine.

Both references explicitly request INT8, including on A5. They use a separate
HCCL group and communication window and never replace the model's fused
output. The report includes exact receive-count agreement, finite checks,
absolute error, relative squared error and output magnitude. The reference
comparison shares loaded weights/routing with the fused path; it does not
independently validate checkpoint loading or the router.

Collect `layer-summary.json`, `layers/rank*.jsonl`, `layers/progress-rank*.json`,
`baseline.json`, `mode2.json`, `baseline.log`, `mode2.log` and `test-console.log`
from the output directory. If no comparisons were recorded, numerical checks
are `null` (unknown), not `false`. The summary lists available artifacts and
each rank's last entered stage, including skipped-input reasons. A completed
original test with zero comparisons does not establish a numerical failure
in either reference: check the import markers and progress records first.
The per-layer data distinguishes a same-input operator discrepancy from accumulated
full-model changes; it does not relax or replace the original precision test.
