# sparse_attention_score

Native AscendC block-sparse attention kernel for MiniMax-M3, statically linked
into `libsgl_kernel_npu.so` and registered as `torch.ops.npu.npu_sparse_attention_score`.

This is the **main attention** step (post-indexer): given a set of pre-selected
top-k KV blocks per query, it computes sparse attention only over those blocks
instead of full attention. It replaces the Triton split-K fallback on Ascend NPU
when available — self-contained, with no PTA plugin and no `ASCEND_CUSTOM_OPP_PATH`.

## Overview

MiniMax-M3 sparse attention runs in two steps:
1. **Indexer** — scores KV blocks and selects top-k (Triton kernel in sglang).
2. **Main attention** — attention over the selected blocks (**this op**).

This kernel implements step 2 as a native direct-launch cube kernel (CUTLASS-style
parameterized GEMM + online softmax), migrated from the PTA (CANN aclnn L0)
3-layer architecture to a 2-layer native architecture (no L0 Op API layer).

## Interface

### C++ HOST_API

```cpp
namespace sglang::npu_kernel {

HOST_API at::Tensor sparse_attention_score(
    const at::Tensor &query,            // [T, N, D] TND layout
    const at::Tensor &key,              // [blockNum, blockSize, kvHead, D]
    const at::Tensor &value,            // [blockNum, blockSize, kvHead, D]
    const at::Tensor &select_idx,       // [kvHead, maxQSeqlen, topK] (0-based, -1 pad)
    const at::Tensor &block_table,      // [batch, maxBlocksPerBatch]
    const c10::optional<at::Tensor> &select_num_idx,        // [kvHead, maxQSeqlen]
    const c10::optional<at::Tensor> &q_dequant_scale,       // fp8 only
    const c10::optional<at::Tensor> &k_dequant_scale,       // fp8 only
    const c10::optional<at::Tensor> &v_dequant_scale,       // fp8 only
    const c10::optional<at::Tensor> &actual_seq_lengths,    // [batch]
    const c10::optional<at::Tensor> &actual_seq_lengths_kv, // [batch]
    int64_t num_key_value_heads,
    double scale_value,
    int64_t block_size,
    int64_t top_k,
    int64_t inner_precise);

}  // namespace sglang::npu_kernel
```

### Python

```python
out = torch.ops.npu.npu_sparse_attention_score(
    query, key, value, select_idx, block_table,
    select_num_idx=...,                # optional
    q_dequant_scale=None,              # fp8 only
    k_dequant_scale=None,
    v_dequant_scale=None,
    actual_seq_lengths=...,            # [batch], decode: ones(batch)
    actual_seq_lengths_kv=...,         # [batch], KV length per request
    num_key_value_heads=...,
    scale_value=...,                   # default: 1/sqrt(D)
    block_size=128,
    top_k=17,                          # M3: 16 scored + 1 local
    inner_precise=0,
)
```

## Inputs / Outputs

| Tensor | Shape | Dtype | Notes |
|---|---|---|---|
| `query` | `[T, N, D]` | bf16/fp16/fp8 | TND layout; T=total Q tokens |
| `key` / `value` | `[blockNum, blockSize, kvHead, D]` | same as query | blocked paged KV |
| `select_idx` | `[kvHead, maxQSeqlen, topK]` | int32 | 0-based block ids; `-1` = skip (OOB/unused) |
| `block_table` | `[batch, maxBlocksPerBatch]` | int32 | logical→physical page map |
| `select_num_idx` | `[kvHead, maxQSeqlen]` | int32 | valid count per (kvHead, query) |
| `actual_seq_lengths` | `[batch]` | int32 | Q length per request (decode: 1) |
| `actual_seq_lengths_kv` | `[batch]` | int32 | KV length per request |
| **output** `attentionOut` | `[T, N, D]` | same as query (fp8→fp16) | attention output |

`softmaxLse [T, N, 1]` fp32 is allocated internally for the kernel to write but
**not returned** (sglang consumes only `attentionOut`).

## Design

### 2-Layer Native Architecture

```
Python: torch.ops.npu.npu_sparse_attention_score(q, k, v, sel, bt, ...)
  ↓
Host API (libsgl_kernel_npu.so):
  op_host/sparse_attention_score.cpp  -> HOST_API + EXEC_KERNEL_CMD
  ↓
Kernel (same .so, ascendc_library static link):
  op_kernel/sparse_attention_score.cpp -> tilingKey dispatch
  op_kernel/sparse_attention_score_kernel_interface.cpp -> arch22/arch35
  op_kernel/attn_infra/ + tla/ + arch22/ + arch35/  (CUTLASS-style infra)
```

No CANN aclnn L0 Op API layer (the PTA `op_api/` directory was removed). Tiling
and infershape are done directly in the host API, not via the CANN graph engine.

### Host API Flow (`op_host/sparse_attention_score.cpp`)

1. Allocate `attentionOut` (query shape; fp8 input → fp16 output) + `softmaxLse` `[T,N,1]` fp32.
2. Fill `SAInfo` from at::Tensor shapes + function attrs + `PlatformAscendCManager`.
3. `SATiling::DoTiling(info)` → `SATilingData` + `workspaceSize` + `blockDim`.
4. `captureMap`: cache the tiling blob by hash of all tiling-relevant dims
   (cuda-graph replay: first call H2D-copies into a global buffer; replays reuse
   the cached device pointer, no per-forward `aclrtMemcpy`).
5. Allocate `workspace = at::empty({workspaceSize})`.
6. `EXEC_KERNEL_CMD(sparse_attention_score, blockDim, <15 GM args>)` → launch.

### Tiling (`op_host/tiling/`)

- `SATilingData` (`sparse_attention_score_tiling_data.h`): POD struct, byte-identical
  mirror of the kernel-side `SparseAttn::SparseAttentionScoreTilingData`
  (`op_kernel/kernel_common.hpp`). Host and kernel share this binary contract.
- `SAInfo`: all tiling inputs (shapes + attrs + platform), filled by the host API.
  No `gert::TilingContext` dependency (the native `ge_helper::TilingContext` deletes
  `SetBlockDim`/`SetTilingKey`/`GetRawTilingData`/`GetPlatformInfo`).
- `SATiling::DoTiling`: computes `totalTaskNum = totalQTokens * kvHeads`,
  `blockDim = min(totalTaskNum, aicNum)`, workspace size, tilingKey, and L1 tile
  params. Mirrors the PTA `SASATiling` logic. `ParseSeqlens` is dropped (dead code
  — tiling does not read `actualSeqLengths` host data, so it is cuda-graph safe).

### Kernel (`op_kernel/`)

- Entry: `extern "C" __global__ __aicore__ void sparse_attention_score(...)` (15 GM args).
- Reads `tilingKey` from the tiling blob (`reinterpret_cast<SparseAttentionScoreTilingData*>`)
  and dispatches by `__CCE_AICORE__`:
  - `220` (arch22 / Ascend910B): `SasaInferIntfRegularArch22<half|bfloat16_t, float>`
  - `310` (arch35 / Ascend910_93): `SasaInferIntfRegular` / `SasaInferInterfaceFullQuant`
- The 3 Sasa interface functions are `__aicore__ inline` (device functions, not
  `__global__`) so they can be called at runtime from the kernel entry (the PTA
  version used compile-time `#if TILING_KEY_VAR` which is unavailable in native mode).
- `attn_infra/` (~15.5K lines): CUTLASS-style parameterized GEMM/epilogue/layout
  infrastructure (arch abstraction, block GEMM, online softmax, rescale-O, mask2idx).
- `tla/` (~1.5K lines): compile-time Tensor Layout DSL.
- `arch22/` / `arch35/`: per-arch kernel implementations.

### tilingKey

| Key | Arch | Dtype | D | blockSize |
|---|---|---|---|---|
| 20001 | arch22 | fp16 | 128 | 128 |
| 20002 | arch22 | bf16 | 128 | 128 |
| 10001 | arch35 | fp16 | 128 | 128 |
| 10002 | arch35 | bf16 | 128 | 128 |
| 10003 | arch35 | fp8 | 128 | 128 |

## `inner_precise` and precision

`inner_precise` selects the inner accumulation path and is the single biggest
knob for EAGLE3 accept rate:

| value | meaning | use |
|---|---|---|
| `0` | low-precision mixed | **do not use** — drops EAGLE3 accept rate (draft-target divergence) |
| `4` | LOW_HIGH_MIXED (high-precision) | **required** — validated bf16-correct; mandatory for fp8 |

The sglang wrapper passes `inner_precise=4`. Verified against a CPU online-softmax
golden (decode + multi-batch verify shapes, bf16): `max_diff ≈ 4.9e-4` (bf16 floor),
no NaN/Inf. The historical "precision problem" that got this op deleted was the
wrapper passing `inner_precise=0`, not the kernel — keep `4`.

## Build

Registered in `csrc/CMakeLists.txt`:
- `OP_SRCS`: `op_host/sparse_attention_score.cpp` + `op_host/tiling/sparse_attention_score_tiling.cpp`
- `WORKSPACE_KERNEL_SRCS`: `op_kernel/sparse_attention_score.cpp`
- `ascendc_include_directories(workspace_kernel ...)`: `op_kernel/` (for attn_infra/tla/arch headers)
- `target_include_directories(...)`: `op_host/` + `op_host/tiling/`

torch registration: `include/sgl_kenel_npu_ops.h` (declaration) +
`csrc/pytorch_extensions.cpp` (`m.def` + `m.impl`).

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
bash build.sh -a kernels Ascend910_9382
```

## Usage

### Auto-detect (sglang integration)

The op is auto-detected by `_get_native_sparse_op()` in
`sgl_kernel_npu/attention/gqa_share_sparse_attention.py`:

```python
@lru_cache(maxsize=1)
def _get_native_sparse_op():
    try:
        import sgl_kernel_npu  # registers torch.ops.npu.*
        return getattr(torch.ops.npu, "npu_sparse_attention_score", None)
    except (ImportError, RuntimeError, AttributeError):
        return None
```

- **Available** → routes decode/verify main attention through this ascendc op.
- **Unavailable** → falls back to the Triton split-K path.
- Gate: `SGLANG_MINIMAX_NPU_NATIVE_ATTN=1` enables it (default off → Triton). The
  legacy `NATIVE_DECODE`/`NATIVE_VERIFY`/`NATIVE_SPARSE_LIB` names are dead. The
  backend primes the probe at `__init__` (pre cuda-graph capture).

### Direct call

```python
import torch, torch_npu, sgl_kernel_npu
out = torch.ops.npu.npu_sparse_attention_score(q, k, v, sel, bt, ...)
```

## Files

```
csrc/sparse_attention_score/
├── op_host/
│   ├── sparse_attention_score.h          # HOST_API declaration
│   ├── sparse_attention_score.cpp        # HOST_API implementation (7-step flow)
│   └── tiling/
│       ├── sparse_attention_score_tiling_data.h  # SATilingData POD (host/kernel contract)
│       ├── sparse_attention_score_tiling.h       # SAInfo + SATiling class
│       └── sparse_attention_score_tiling.cpp     # DoTiling implementation
├── op_kernel/
│   ├── sparse_attention_score.cpp                # kernel entry (15 GM args, tilingKey dispatch)
│   ├── sparse_attention_score_tilingkey.h        # tilingKey constants
│   ├── sparse_attention_score_kernel_interface.cpp  # SasaInfer* device functions
│   ├── kernel_common.hpp                         # kernel-side tiling data struct (contract)
│   ├── arch22/                                   # Ascend910B kernel
│   ├── arch35/                                   # Ascend910_93 kernel
│   ├── attn_infra/                               # CUTLASS-style GEMM/epilogue/layout infra
│   └── tla/                                      # Tensor Layout DSL
```

## Notes

- **cuda-graph safe**: workspace via the NPU caching allocator (graph memory pool);
  `actual_seq_lengths_kv` is sglang's static int32 buffer refreshed out-of-graph
  each forward; tiling blob cached in `captureMap` (hash of tiling-relevant dims).
- **M3 config**: `num_kv_heads=1` (TP=16 per-rank), `num_q_heads=8` (group=8),
  `head_dim=128`, `block_size=128`, `top_k=17` (16 scored + 1 local), bf16.
- **e2e (benchmark_serving_auto, 16K/48, cuda-graph)**: accept length 3.88 vs
  Triton 3.56 (≥ baseline, no drop); faster decode TPOT. Self-contained build —
  no `ASCEND_CUSTOM_OPP_PATH` / external vendor package.
- **Precision harness**: `native_attn_ref/test_precision_kernel.py` (this op,
  `inner_precise=4`, vs CPU online-softmax golden) and `test_precision.py`
  (aclnn reference variant).
