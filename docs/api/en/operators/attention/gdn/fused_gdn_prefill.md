# fused_gdn_prefill

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Unsupported | Supported |
| `torch_native` | Unverified | Supported |
| `tilelang` | Not implemented | Not implemented |
| `ascendc` | Not implemented | Not implemented |

## Signature

The default option is `auto`, registered to the same entry point as `triton` on `Generation.ARCH35`.

```python
from sgl_kernel_npu.operators.attention.gdn import (
    build_gdn_prefill_chunk_meta,
    fused_gdn_prefill,
)

def fused_gdn_prefill(
    mixed_qkv,
    a,
    b,
    A_log,
    dt_bias,
    ssm_states,
    cache_indices,
    cu_seqlens,
    chunk_indices,
    chunk_offsets,
    num_k_heads,
    num_v_heads,
    total_valid_tokens=None,
    ckpt_chunk_idx=None,
    ckpt_dst_slots=None,
    *,
    impl="auto",
):
    ...
```

## Description

Executes the full Gated Delta Net (GDN) chunked prefill pipeline on Ascend 950 (`arch35`) in 3 fused Triton stages without copying `V` out of `mixed_qkv`:

1. **Stage 0 (L2Norm + Gating)**: Computes row-wise L2 normalization on `Q` and `K` (`eps = 1e-6`) and GDN gating in float32:
   $$g_t = -\exp(A_{\log}) \cdot \operatorname{softplus}(a_t + \text{dt\_bias}), \quad \beta_t = \sigma(b_t)$$
   writing `g_T [NV, T]` and `beta_T [NV, T]` via 2D UB block-transpose for contiguous bursts.
2. **Stage 1 (GQA-Interleaved Prepare)**: Deduplicates $K K^\top$ and $Q K^\top$ across GQA sibling V-heads (`GQA=3`, `2×GQA=2`, or generic), computes local `cumsum(g)` over `BT=64`, unscaled intra-chunk attention `ATTN [NT, NV, 64, 64]`, exact 2-level 8×8 block Euler triangular inversion $(I + L)^{-1}$ in FP16, and WY factors $(W, U)$ in BF16 reading `V` directly from `mixed_qkv`.
3. **Stage 2 (Fused `ChunkFwdH + ChunkFwdO`)**: Keeps recurrent state $H \in \mathbb{R}^{D_v \times D_k}$ on-chip, updates `ssm_states[cache_indices]` in place, optionally writes single-point Radix Cache checkpoints at `ckpt_chunk_idx * 64` into `ssm_states[ckpt_dst_slots]`, and scales the FP32 output once by $D^{-1/2}$.

## Parameters and return value

| Parameter | Type | Shape | Dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- | --- |
| `mixed_qkv` | `torch.Tensor` | `[T, (2*NK + NV)*128]` | `torch.bfloat16` | Required | Contiguous packed QKV tensor on NPU. |
| `a` | `torch.Tensor` | `[T, NV]` | `torch.bfloat16`, `torch.float32` | Required | Contiguous pre-softplus decay projection. |
| `b` | `torch.Tensor` | `[T, NV]` | `torch.bfloat16`, `torch.float32` | Required | Contiguous pre-sigmoid beta projection. |
| `A_log` | `torch.Tensor` | `[NV]` | `torch.float32` | Required | Per-head log-decay scale parameter. |
| `dt_bias` | `torch.Tensor` | `[NV]` | `torch.bfloat16`, `torch.float32` | Required | Per-head decay bias parameter. |
| `ssm_states` | `torch.Tensor` | `[num_slots, NV, 128, 128]` | `torch.bfloat16`, `torch.float32` | Required | Contiguous writable SSM state pool updated in place. |
| `cache_indices` | `torch.Tensor` | `[B]` | `torch.int32` | Required | Active state slot index per sequence. |
| `cu_seqlens` | `torch.Tensor` | `[B + 1]` | `torch.int32` | Required | Cumulative sequence lengths (`cu_seqlens[0] == 0`). |
| `chunk_indices` | `torch.Tensor` | `[NT, 2]` | `torch.int32` | Required | `(seq_id, chunk_id)` mapping from `build_gdn_prefill_chunk_meta`. |
| `chunk_offsets` | `torch.Tensor` | `[B + 1]` | `torch.int32` | Required | Cumulative chunk offsets from `build_gdn_prefill_chunk_meta`. |
| `num_k_heads` | `int` | — | — | Required | Number of Q/K heads (power of 2, divides `num_v_heads`). |
| `num_v_heads` | `int` | — | — | Required | Number of V heads. |
| `total_valid_tokens` | `int` or `None` | — | — | `None` | Valid token count; trailing padded rows `[total_valid_tokens:]` are zeroed. |
| `ckpt_chunk_idx` | `torch.Tensor` or `None` | `[B]` | `torch.int32` | `None` | Optional chunk index at which to snapshot entering state `H`. |
| `ckpt_dst_slots` | `torch.Tensor` or `None` | `[B]` | `torch.int32` | `None` | Destination slots in `ssm_states` for checkpoints (`< 0` skips). |
| `impl` | `str` | — | — | `"auto"` | Keyword-only; `"auto"`, `"triton"` (`ARCH35` only), or `"torch_native"`. |

| Return value | Type | Shape | Dtype | Description and constraints |
| --- | --- | --- | --- | --- |
| `output` | `torch.Tensor` | `[T, NV, 128]` | `torch.bfloat16` | Newly allocated prefill attention output on the same NPU device. |

## triton backend

- Registered for `Generation.ARCH35` (Ascend 950). Supports `head_dim = 128`, `chunk_size = 64`, power-of-two `num_k_heads`, and arbitrary integer `GQA = num_v_heads // num_k_heads` (specialized 3-way and 4-way interleaved kernels for `GQA=3` and `GQA=2` with even `NK`, plus a generic fallback).
- Uses 64-bit token-stride addressing (`tl.int64`) to support long-context batches without `int32` wraparound.

## torch_native backend

- Independent FP32 PyTorch step-by-step recurrent implementation updating `ssm_states` and optional checkpoints in place.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.attention.gdn import (
    build_gdn_prefill_chunk_meta,
    fused_gdn_prefill,
)

nk, nv, d = 16, 48, 128
seq_lens = [129, 64]
T = sum(seq_lens)
mixed_qkv = torch.randn(T, (2 * nk + nv) * d, device="npu", dtype=torch.bfloat16)
a = torch.randn(T, nv, device="npu", dtype=torch.bfloat16)
b = torch.randn(T, nv, device="npu", dtype=torch.bfloat16)
A_log = torch.randn(nv, device="npu", dtype=torch.float32) * 0.2 - 1.0
dt_bias = torch.randn(nv, device="npu", dtype=torch.bfloat16) * 0.1
ssm_states = torch.zeros(4, nv, d, d, device="npu", dtype=torch.bfloat16)
cache_indices = torch.tensor([0, 1], device="npu", dtype=torch.int32)
cu_seqlens = torch.tensor([0, 129, 193], device="npu", dtype=torch.int32)
cidx, coff, _, total_valid = build_gdn_prefill_chunk_meta(seq_lens, mixed_qkv.device)

out = fused_gdn_prefill(
    mixed_qkv, a, b, A_log, dt_bias, ssm_states, cache_indices, cu_seqlens,
    cidx, coff, nk, nv, total_valid_tokens=total_valid, impl="auto",
)
```
