# chunk_gated_delta_rule_npu

## Hardware support

| Implementation | ARCH22 | ARCH35 |
| --- | --- | --- |
| `triton` (also `auto`) | Unverified for this composition | Unverified for this composition |

## Signature

```python
from sgl_kernel_npu.operators.attention.gdn import chunk_gated_delta_rule_npu

def chunk_gated_delta_rule_npu(
    q, k, v, g, beta, scale=None, initial_state=None,
    output_final_state=True, cu_seqlens=None, head_first=False,
    use_qk_l2norm_in_kernel=False, *, impl="auto",
):
    ...
```

`auto` selects `triton`. This is the legacy `fla.chunk` decomposed wrapper,
separate from the fused CANN `chunk_gated_delta_rule` API.

## Description

Copy the legacy forward orchestration without changing its function bodies.
Only implementation imports are redirected to the migrated implementations:
optional Q/K L2 normalization, chunk-local cumulative sum, scaled KKT,
triangular inverse, WY recomputation, chunk state update and output reconstruction.
The default chunk width is 64. Intermediate dtypes, casts and launch choices
remain those of the existing implementations.

The mathematical token recurrence is
`S_t = exp(g_t) * S_(t-1) + beta_t * k_t ⊗ (v_t - k_t^T exp(g_t) S_(t-1))`,
`o_t = scale * q_t^T S_t`. Heads share Q/K according to `H/Hg`.
Optional L2 normalization uses `x / sqrt(sum(x*x) + 1e-6)` before the recurrence.
The chunk implementation retains its existing low-precision rounding points.

## Parameters and return value

| Parameter | Type / shape | Default | Meaning |
| --- | --- | --- | --- |
| `q`, `k` | NPU Tensor `[B,T,Hg,K]` | Required | Matching FP16/BF16 dtype and device. |
| `v` | NPU Tensor `[B,T,H,V]` | Required | Same dtype as q/k; H is a multiple of Hg. |
| `g` | NPU Tensor `[B,T,H]` | Required | Per-token log decay, usually FP32; not cumulative. |
| `beta` | NPU Tensor `[B,T,H]` | Required | Update weights. |
| `scale` | float or None | None | None means `K**-0.5`; explicit zero is retained. |
| `initial_state` | NPU Tensor `[N,H,K,V]` or None | None | None means zero state. |
| `output_final_state` | bool | True | Return the final state, or None when false. |
| `cu_seqlens` | NPU int32/int64 Tensor `[N+1]` or None | None | Packed B=1 boundaries; first=0, last=T, positive lengths. |
| `head_first` | bool | False | True raises the legacy DeprecationWarning exception. |
| `use_qk_l2norm_in_kernel` | bool | False | Normalize Q/K using the existing L2 implementation. |
| `impl` | str | `"auto"` | `auto` or `triton`; no backend fallback. |

| Return | Shape / dtype | Meaning |
| --- | --- | --- |
| `o` | `[B,T,H,V]`, q.dtype | Attention output. |
| `final_state` | `[N,H,K,V]`, FP32 or None | Final recurrent states. |
| `h` | `[1,total_chunks,H,K,V]` for packed input, q.dtype | States entering each chunk, in sequence/chunk order. |

## Triton implementation

Inputs are made contiguous and execution enters the input tensor's NPU device
context, preserving the legacy input guard. The copied wrapper retains
`torch.compiler.disable`; this change does not add full-graph support.
The ordinary path uses K,V <= 128 as supported by the migrated state kernel.
Use fixed B=1 or packed B=1 multi-sequence inputs: fixed B>1 inherits the
existing state/output kernels' gate-offset limitation. No new validation or
kernel correction is introduced. Do not mutate cached sequence boundaries in place.

`GDN_USE_MEGA_GDN=1` selects the byte-identical legacy mega helper (128-token
chunks, head dimension 128, B=1 assumptions); it requires the migrated native
`mega_chunk_gdn` operator selected in the standalone ARCH22 build. Its unchanged
helper lives under `implementation/ascendc/attention/gdn`; see [mega_chunk_gdn](mega_chunk_gdn.md). `GDN_RECOMPUTE_SUPPRESS_LEVEL` retains
its import-time behavior for intermediate retention. Both switches preserve
legacy behavior; the default path uses migrated Triton implementations.

No new `torch_native` production backend is added in this import-only migration.
Tests use an independent tokenwise recurrence for all three returns on both
decomposed and mega paths, without importing the old `fla` implementation.
Device accuracy and performance require A3/A5 execution.

## Example

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.attention.gdn import chunk_gated_delta_rule_npu

q = torch.randn(1, 65, 4, 128, device="npu", dtype=torch.bfloat16)
k = torch.randn_like(q)
v = torch.randn(1, 65, 8, 128, device="npu", dtype=q.dtype)
g = -torch.rand(1, 65, 8, device="npu", dtype=torch.float32)
beta = torch.rand(1, 65, 8, device="npu", dtype=q.dtype)
cu_seqlens = torch.tensor([0, 31, 65], device="npu", dtype=torch.int32)
o, final_state, h = chunk_gated_delta_rule_npu(
    q, k, v, g, beta, cu_seqlens=cu_seqlens,
    use_qk_l2norm_in_kernel=True, impl="triton",
)
```
