# Tree indexer TopK (AscendC, A2/A3)

Replaces only the final score-to-index selection in the DSpark tree indexer.
Gather and score computation remain in Triton. This operator has no vLLM
dependency and uses the existing `EXEC_KERNEL_CMD` / generated kernel launcher
build path. It is included by `build.sh -a kernels` for the A2/A3 family, not A5.

## Interface and selection

```python
import torch
import sgl_kernel_npu  # registers the compiled operator

indices = torch.ops.npu.tree_indexer_topk(scores, k)
```

- Input: contiguous NPU FP32 `[rows, slots]`, `slots > 0`.
- `1 <= k <= 1024`; the serving case uses `k=512` (model `index_topk`, not
  speculative `topk=2`). `k` may exceed the number of slots.
- Output: INT32 `[rows, 1, k]`. Select by descending score, break ties by lower
  logical index, emit selected indices in ascending order, pad with `-1`.
- NaN and negative infinity are invalid; positive infinity is valid. Signed
  zeros compare equal. Empty batches are allowed.

The kernel assigns rows to vector cores. Each core sorts 1024 scores at a time
with the repository's `LIServiceVec::SortAll` helper, retains a running top-K
list with `MrgSort`, and reads its K-th score. Two subsequent scans count strict
winners and compact selected indices, taking threshold ties in index order.
Native sort tie ordering does not affect the selected result. UB buffers total
44 KiB + 256 bytes, independent of history length; no device workspace is used.
This replaces the Triton bisection's repeated scans and avoids the int64
`tl.topk`/`tl.sort` compilation path that failed in memory planning on A3.

This first implementation uses one core per row (up to the available vector
cores). It does not distribute a single long row across multiple cores.

## Build and validate

Run in the NPU environment with CANN and torch_npu already configured:

```bash
git switch dspark-topk
git pull --ff-only
bash build.sh -a kernels
python3 -m pip install --force-reinstall --no-deps output/sgl_kernel_npu-*.whl
python3 benchmark/spec_decode/bench_tree_indexer_topk.py --compare-triton
```

Use the current build's wheel if `output/` contains multiple matching versions.
The build script auto-detects the SoC; an explicit target, if needed, must match
the machine (for example `bash build.sh -a kernels Ascend910_9362`).

The standalone script first checks a CPU model of the algorithm, then checks
the actual NPU output against an independent stable-sort reference, including
partial tiles, same-score boundaries, fewer than K valid slots, and core reuse.
It then times `[13, slots]` inputs for slots 1024, 2048, 4096 and 8192, with
K=512. `--compare-triton` also runs the working integer-bisection fallback,
**not** the failing `tl.topk` implementation. First-call wall time and warmed
stream/wall intervals are reported separately; the stream interval can include
host launch gaps, so use profiling for pure device kernel time. Correctness
checks run before timing, so `first_call_ms` is not process-cold startup time.

For the algorithm-only check on a machine without an NPU:

```bash
python3 benchmark/spec_decode/bench_tree_indexer_topk.py --cpu-only
```

CPU success does not establish that the AscendC kernel compiles or runs correctly.
Native compilation, NPU correctness and throughput must be verified on the target
CANN/device combination before using the new path for benchmark conclusions.

## Enable in SGLang

Add this variable **before starting a new server process**:

```bash
export SGL_TREE_INDEXER_TOPK_IMPL=ascendc
```

The default is `triton`, preserving existing behavior. With `ascendc`,
`SGL_TREE_INDEXER_TL_TOPK` does not control selection. A missing native operator
raises a rebuild/install error rather than silently falling back. Short contexts
that bypass indexer scoring entirely still bypass this TopK operator.

Keep the existing tree backend and native tree attention settings:

```bash
export SGLANG_DSPARK_TREE_BACKEND=sgl_kernel_npu
export SGLANG_DSPARK_TREE_ATTN_IMPL=ascendc
```

To return to the working bisection path, restart with:

```bash
export SGL_TREE_INDEXER_TOPK_IMPL=triton
export SGL_TREE_INDEXER_TL_TOPK=0
```

After standalone correctness passes, reproduce the original long-output request
past the first real C4 TopK selection, then repeat the same serving benchmark.
This change makes no claim about end-to-end throughput until those runs finish.

## Per-verify metadata reuse in SGLang

The matching SGLang `dspark-topk` adapter can cache the exact maximum compressed
length, validate `query_cu` once, and reuse the 1024-element placeholder metadata
tensor across indexer layers in the same immutable `NodePlan`. A new verify step
gets a new plan; explicit caller-owned lengths retain normal validation. No
candidate bounds, score computation or TopK implementation are changed.

Update **both** repositories and reinstall the kernel Python package for this
adapter change. Its new keyword-only `max_length` / `validate_values` arguments
are Python API additions; the AscendC binary interface has not changed.
Direct kernel callers still validate device values by default, including checking
a supplied maximum against the actual maximum.

The adapter defaults to reuse. To compare the old per-layer preparation with
the new per-step preparation, keep `SGL_TREE_INDEXER_TOPK_IMPL=ascendc` and the
same workload, restarting the server with each setting:

```bash
export SGLANG_DSPARK_INDEXER_REUSE_METADATA=1  # reuse within a verify step
# export SGLANG_DSPARK_INDEXER_REUSE_METADATA=0  # original preparation
```

For a trace with 21 real indexer calls per step, indexer `cu` equality checks and
maximum-length readbacks should drop from 21 to 1 per step, as should placeholder
metadata creation. Total `item`/`max` counts also include other model paths and
will not drop to one. Measure serving throughput separately from profiling.
