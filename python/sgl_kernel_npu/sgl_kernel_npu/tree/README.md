# `sgl_kernel_npu.tree` — DSV4 DSpark tree operators

Python-only Triton port of the four DSV4 DSpark tree operators from
**vllm-ascend `8947f1f9f`**. The authoritative semantics (signatures, dtypes,
strides, ancestor/mask rules, metadata layout) live in
[`docs/tree_ops_semantics.md`](../../../../docs/tree_ops_semantics.md). This file
is the package-level contract and the operator quick reference.

## Wave 1 scope

Wave 1 ships:

- `_common.py` — Triton-free validation and tree/page geometry helpers.
- `__init__.py` — a lazy public API. Importing `sgl_kernel_npu.tree` does **not**
  import Triton, `torch_npu`, or a kernel module.
- `README.md` — this document.
- `docs/tree_ops_semantics.md` — the implementation brief (D1–D8 + per-op math).
- `benchmark/spec_decode/bench_tree_ops.py` — CLI harness. `--dry-run` runs the
  independent CPU reference; non-dry-run measures on NPU.

Wave 1 does **not** ship the four Triton kernels, C++/CMake changes, or
`scripts/run_kernel_tests.sh` edits. Requesting a kernel before it exists raises
an `ImportError` with the planned module name.

## Public API

| Canonical name | Short alias | AscendC source | Planned module |
|---|---|---|---|
| `compressor_tree_project` | `tree_project` | `compressor_tree_project` | `tree/compressor_tree_project.py` |
| `compressor_tree` | `tree_compressor` | `compressor_tree` | `tree/compressor_tree.py` |
| `quant_lightning_indexer_tree` | `tree_indexer` | `quant_lightning_indexer_tree` | `tree/quant_lightning_indexer_tree.py` |
| `sparse_attn_sharedkv_tree` | `tree_attention` | `sparse_attn_sharedkv_tree` | `tree/sparse_attn_sharedkv_tree.py` |

Resolution is lazy:

```python
import sgl_kernel_npu.tree as tree

tree.common.pick_specialization(4, 512, 64)   # (4, 2, 512, 64)
tree.compressor_tree_project                 # ImportError until Wave 2 lands
```

The shared helpers are importable directly:

```python
from sgl_kernel_npu.tree._common import (
    ape_row, tree_step, tree_node, paged_slot,
    align_up, ceil_div, pick_specialization, get_num_vectorcores,
)
```

`_common.py` imports only `torch` and the standard library. `torch_npu` and
Triton are imported lazily inside `get_num_vectorcores()` (which reuses
`indexer/flash_block_score_decode.py::_get_vectorcore_num_safe`). This keeps
CPU-only reference runs possible.

## Locked decisions (D1–D8)

| # | Decision |
|---|---|
| **D1** | Semantic contract is the AscendC math at `8947f1f9f`, re-expressed as exact PyTorch. AscendC-specific rounding (per-512 bf16 probability blocks, fp16 head-reduction order) is documented but not bit-reproduced. Acceptance is tolerance-based. |
| **D2** | `metadata` (`int32[1024]`) is accepted, validated for dtype/shape/contiguity, and **otherwise ignored**. A zeroed metadata and a realistic metadata must produce identical output. |
| **D3** | Top-k tie order is not contractual. Kernels use `(score desc, index asc)`; tests compare multisets and score sets. |
| **D4** | `layout_q` defaults to `"TND"`; `"BSND"` raises `NotImplementedError`. |
| **D5** | The reference lives in `tests/python/sgl_kernel_npu/tree_reference.py`, imports only `torch`, and is never imported by production code. |
| **D6** | `ratio ∈ {4,128}` and `head_dim ∈ {128,512}` are Triton `constexpr` specializations (four per op). |
| **D7** | `compressor_tree` / `sparse_attn_sharedkv_tree` accept `rope_dim` / `cmp_ratio` at runtime, assert supported values, then dispatch to constexpr kernels. |
| **D8** | No NPU is required to merge Wave 1/2 except for `npu`-marked tests. The numerical GREEN gate is the A3 command block below. |

## Operator quick contracts

Full details and source citations are in `docs/tree_ops_semantics.md`.

### `compressor_tree_project(x, wkv, wgate, nodes, ape, positions, cu) -> None`

- `x` BF16 `[T,H]`; `wkv`/`wgate` BF16 `[c*d,H]`; `c = 2` for ratio 4, `c = 1`
  for ratio 128; `d ∈ {128,512}`.
- `nodes` FP32 `[T+1,1,2*c*d]` is **in-place output**; row 0 is reserved and must
  not be written.
- `ape` FP32 `[ratio,c*d]`; `positions` int32 `[T]` = `(prefix+depth) % ratio`;
  `cu` int32 `[T+1]` **must equal `arange(T+1)`** (one query per node).
- Computes `kv = x @ wkv.T`, `score = x @ wgate.T + ape[r]`, and writes
  `[kv(c*d) | score(c*d)]`. No aggregation, RMSNorm, or RoPE.
- Binding-level dtype is BF16 only (`8947f1f9f:…/torch_binding.h:10`).

### `compressor_tree(state, nodes, norm, cos, sin, table, prefix, ancestors, depths, requests, ratio, rope_dim, epsilon) -> Tensor[n,d] BF16`

- `state` FP32 paged `[pages,rows,2*c*d]` with `stride(1) >= 2*c*d`;
  `nodes` FP32 `[(n+1)*2*c*d]`; `norm` FP32 `[d]`; `cos/sin` FP32 `[...,>=64]`
  with `stride(1) == 1`.
- `table` int32; **`prefix` is int32**; `ancestors` / `depths` / `requests` are
  int64. (Source correction: the earlier plan said prefix int64; see
  `8947f1f9f:…/torch_binding.h:27-30` and `…/compressor_tree.cpp:131-132`.)
- Emits only when `(pos+1) % ratio == 0`; otherwise the output row is exactly
  zero. C4 interleaves previous/current halves; C128 uses one window.
- Algorithm: window max → `exp/sum/weighted sum` over valid ancestors or paged
  slots → RMSNorm with `eps` **inside** `sqrt` → interleaved RoPE
  (`i^1`, sign `-1/+1`, `ropePos = pos+1-ratio`) → BF16.
- State inputs are read-only; non-emitting nodes produce zeros.

### `quant_lightning_indexer_tree(query, history, weights, query_scale, history_scale, nodes, node_scale, ancestors, prefixes, requests, table, cu, lengths, residual, metadata, topk) -> int32[n,1,topk]`

- `query` int8 `[n,64,128]`; `history` int8 `[pages,block,1,128]` with
  `stride(1)==128`, `stride(3)==1`; `history_scale` FP16 `[pages,block,1,1]`;
  `weights`/`query_scale` FP16 `[n,64]`; `nodes` int8 `[n,128]`;
  `node_scale` FP16 `[n]`.
- `ancestors` int64; `prefixes`/`table`/`cu`/`lengths`/`residual`/`metadata`
  int32; `requests` int64; `topk ∈ [1,2048]`.
- Score: int8 QK → int32 → `relu` → `*2^-10` → FP16 rounding → `w*qs` FP16 →
  FP32 head sum → `*k_scale`.
- Mapping: `k < prefix/4` uses paged history; otherwise
  `step = (k+1)*4-1-prefix`, `node = ancestors[row,step]`. Invalid → score
  `-inf`, index `-1`.
- Top-k: 2048-wide accumulator, `(score desc, index asc)`, `-1` padding.
  Select-all is a Python-side shortcut when `max(1, max_kv//4) <= topk`.

### `sparse_attn_sharedkv_tree(q, *, ...) -> (out, softmax_lse)`

- `q` FP16/BF16 `[T,N1,512]`; `ori_kv`/`cmp_kv` paged with `N2 == 1`.
- `ori_sparse_indices` int32 `[T,1,W]`, `W` multiple of 128, physical slots and
  `-1` padding. C4 adds logical `cmp_sparse_indices` `[T,1,K]`; C128 omits it.
- Tree pool inputs (`cmp_tree_kv` `[nodes,512]`, `tree_ancestors` int64
  `[T,max_depth]`, `tree_prefix` int32 `[T]`) are **all-or-none**.
- C4 filters logical compressed positions; C128 traverses full CFA history.
  Logical positions never become physical cache indices. QK and PV use the
  same original-KV slots.
- Softmax includes an optional sink as a zero-value key with initial weight 1;
  `probabilities` are cast to the KV dtype before PV; `LSE = ln(S)+M`.
  An empty row returns `out=0, lse=0`.
- `layout_q="TND"`, `layout_kv="PA_ND"`, `ori_mask_mode=4`, `cmp_mask_mode=3`.

## “Metadata is validated but ignored”

Both `quant_lightning_indexer_tree` and `sparse_attn_sharedkv_tree` take
`metadata: int32[1024]` for ABI compatibility with the AscendC operators.
**D2 applies: validate `dtype == int32`, `numel == 1024`, and contiguity, then
ignore the contents.** The 36-AIC × 8 + 72-AIV × 8 partition is an AscendC
scheduling detail; the Triton kernels compute their own grid. Do not read this
tensor in a kernel. The layout is documented in
`docs/tree_ops_semantics.md` §1.5 and in the two `*_metadata.h` headers at
`8947f1f9f:csrc/attention/{quant_lightning_indexer_tree,sparse_attn_sharedkv_tree}/op_kernel/*_metadata.h:24-64`.

## Tests and commands

### CPU-only (no NPU, no Triton, no `.so`)

Reference self-tests (owned by the reference agent):

```bash
python -m pytest tests/python/sgl_kernel_npu/test_tree_reference.py -q
```

Import/contract checks for this package:

```bash
python -m py_compile python/sgl_kernel_npu/sgl_kernel_npu/tree/_common.py \
  python/sgl_kernel_npu/sgl_kernel_npu/tree/__init__.py
```

Note: importing `sgl_kernel_npu.tree` still executes the repository root
`sgl_kernel_npu/__init__.py`, which loads `libsgl_kernel_npu.so` and imports
`torch_npu`. On a CPU-only box, use `tree_reference.py` directly (it imports
only `torch`). The `tree` package itself does not add a Triton/NPU import.

### NPU / A3 (numerical GREEN gate)

```bash
export PYTHONPATH=$PWD/python/sgl_kernel_npu:$PYTHONPATH
ls python/sgl_kernel_npu/sgl_kernel_npu/lib/libsgl_kernel_npu.so
python -m pytest tests/python/sgl_kernel_npu/test_tree_*.py -q
bash scripts/run_kernel_tests.sh speculative
```

The tree tests are not yet registered in `SPECULATIVE_TESTS`
(`scripts/run_kernel_tests.sh:70-75`); registration is a later wave task.

### Benchmark

```bash
# CPU dry-run: runs the independent reference, emits JSON/CSV, no NPU timing.
python benchmark/spec_decode/bench_tree_ops.py --op all --dry-run \
  --json /tmp/tree_dry.json --csv /tmp/tree_dry.csv

# A3 measurement: production kernels vs torch-reference baseline.
python benchmark/spec_decode/bench_tree_ops.py --op all --baseline torch \
  --warmup 10 --reps 50 --csv perf/tree_perf.csv --json perf/tree_perf.json
```

`--dry-run` never imports production Triton modules. Non-dry-run lazily imports
`torch_npu` and the tree kernels, uses `torch.npu.synchronize()` and
`torch.npu.Event`, and raises a clear error when no NPU is available.

## File map

```
python/sgl_kernel_npu/sgl_kernel_npu/tree/
├── __init__.py                       # lazy public API (this package)
├── _common.py                        # validation + geometry (Triton-free)
├── README.md                         # this file
├── compressor_tree_project.py         # Wave 2 (Triton)
├── compressor_tree.py                 # Wave 2 (Triton)
├── quant_lightning_indexer_tree.py    # Wave 3 (Triton)
└── sparse_attn_sharedkv_tree.py       # Wave 3 (Triton)
```

Tests (not created by this package):

```
tests/python/sgl_kernel_npu/tree_reference.py        # independent CPU oracle (D5)
tests/python/sgl_kernel_npu/tree_fixtures.py         # deterministic fixtures
tests/python/sgl_kernel_npu/test_tree_reference.py   # hand-computed goldens
tests/python/sgl_kernel_npu/test_tree_<op>.py        # per-op contract + NPU parity
tests/python/sgl_kernel_npu/test_tree_integration.py # full chain
```

## Source pin and corrections

- Source: `vllm-ascend` commit `8947f1f9f`.
- Paths: `csrc/attention/{compressor_tree_project,compressor_tree,
  quant_lightning_indexer_tree,sparse_attn_sharedkv_tree}/`.
- Corrections recorded in `docs/tree_ops_semantics.md` §0:
  - `compressor_tree.prefix` is int32, not int64.
  - `compressor_tree_project` is BF16-only at the Python binding even though
    the op definition advertises FP16 as well.
