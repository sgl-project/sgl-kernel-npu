# chunk_gated_delta_rule_npu

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Supported | Supported |
| `torch_native` | Supported | Supported |

## Signature

Omitting `impl` or passing `"auto"` selects `"triton"` without automatic fallback.
The forward pass uses decomposed Triton kernels.

```python
from sgl_kernel_npu.operators.attention.gdn import chunk_gated_delta_rule_npu

def chunk_gated_delta_rule_npu(
    q, k, v, g, beta, scale=None, initial_state=None,
    output_final_state=True, cu_seqlens=None, head_first=False,
    use_qk_l2norm_in_kernel=False, *, impl="auto",
):
    ...
```

## Description

Computes the gated delta rule forward pass, returning attention output, optional
final recurrent states, and the states entering each chunk. Each group of `H/Hg`
value heads shares one query/key head. `g` contains per-token log decay;
`beta` controls the state update.

The forward pass processes chunks of 64 tokens. Optional Q/K L2 normalization
computes `x / sqrt(sum(x*x) + 1e-6)` before attention. `scale=None` uses `K**-0.5`;
an explicit zero scale is preserved. Packed sequences have independent states.

This is the decomposed GDN interface used by SGLang's linear-attention path.
It is registered separately from the fused CANN `chunk_gated_delta_rule` API.

## Parameters and return value

| Parameter | Type | Shape | Dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- | --- |
| `q`, `k` | `torch.Tensor` | `[B,T,Hg,K]` | `torch.float16`, `torch.bfloat16` | Required | Query and key tensors on the same NPU; matching dtype. |
| `v` | `torch.Tensor` | `[B,T,H,V]` | Same as `q` | Required | Value tensor on the same NPU; `H` must be a multiple of `Hg`. |
| `g` | `torch.Tensor` | `[B,T,H]` | `torch.float32`, `torch.float16`, `torch.bfloat16` | Required | Per-token log decay, before cumulative summation; on the input NPU. |
| `beta` | `torch.Tensor` | `[B,T,H]` | `torch.float32`, `torch.float16`, `torch.bfloat16` | Required | State-update weights on the input NPU. |
| `scale` | `float` or `None` | — | — | `None` | Output scale; `None` uses `K**-0.5`. |
| `initial_state` | `torch.Tensor` or `None` | `[N,H,K,V]` | `torch.float32`, `torch.float16`, `torch.bfloat16` | `None` | Initial states on the input NPU; `None` starts from zero. |
| `output_final_state` | `bool` | — | — | `True` | Whether to return final states. |
| `cu_seqlens` | `torch.Tensor` or `None` | `[N+1]` | `torch.int32`, `torch.int64` | `None` | Packed sequence boundaries on the input NPU: first value 0, last value `T`, positive sequence lengths. `None` means fixed-length input. |
| `head_first` | `bool` | — | — | `False` | Only `False` is supported; `True` raises `DeprecationWarning`. |
| `use_qk_l2norm_in_kernel` | `bool` | — | — | `False` | Normalize Q/K along the last dimension before attention. |
| `impl` | `str` | — | — | `"auto"` | Keyword-only; accepts `"auto"`, `"triton"`, or `"torch_native"`. |

| Return value | Type | Shape | Dtype | Description and constraints |
| --- | --- | --- | --- | --- |
| `o` | `torch.Tensor` | `[B,T,H,V]` | `q.dtype` | Attention output on `q.device`; inputs are not modified. |
| `final_state` | `torch.Tensor` or `None` | `[N,H,K,V]` | `torch.float32` | Final states on `q.device`; `None` when `output_final_state=False`. |
| `h` | `torch.Tensor` | `[1,NT,H,K,V]` | `q.dtype` | States entering each chunk on `q.device`, ordered by sequence and chunk. |

Use fixed `B=1` or packed `B=1` inputs with `K,V <= 128`. `N` is the number
of sequences; without `cu_seqlens`, `N=1`. `NT` is the total number of chunks,
summing `ceil(sequence_length / chunk_size)` over sequences. Packed initial
states must have `N=len(cu_seqlens)-1`. Do not modify cached sequence boundaries
in place. Shape, dtype and device requirements are caller preconditions; only
some are explicitly validated.

## triton backend

- Makes tensor inputs contiguous and selects their NPU device context.
- Composes chunk-local cumulative sum, scaled KKT, triangular inverse, WY
  recomputation, state update and output reconstruction with 64-token chunks.
- Provides forward execution for inference, preserving the existing intermediate
  dtypes and casts.

## torch_native backend

- Computes an independent tokenwise recurrence on CPU in FP64 and restores the
  output dtypes and device. Supports the same normalization, scale and state options.
- Provides the reference implementation for accuracy checks.

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
    use_qk_l2norm_in_kernel=True,
)
```
