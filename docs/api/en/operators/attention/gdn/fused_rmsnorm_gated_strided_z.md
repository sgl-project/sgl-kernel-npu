# fused_rmsnorm_gated_strided_z

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Unverified | Supported |
| `torch_native` | Unverified | Supported |
| `tilelang` | Not implemented | Not implemented |
| `ascendc` | Not implemented | Not implemented |

## Signature

The default option is `auto`, currently registered to the same entry point as `triton`.

```python
from sgl_kernel_npu.operators.attention.gdn import fused_rmsnorm_gated_strided_z

def fused_rmsnorm_gated_strided_z(
    core_attn_out,
    projected_states_qkvz,
    norm_weight,
    z_offset,
    eps=1e-6,
    *,
    impl="auto",
):
    ...
```

## Description

Computes per-head gated RMSNorm while reading `z` directly from a strided slice of packed `projected_states_qkvz` (`[:, z_offset : z_offset + NV * D]`), avoiding a full-tensor copy of `z`:

$$Y_{t,h,j} = \frac{X_{t,h,j}}{\sqrt{\frac{1}{D}\sum_{k=0}^{D-1} X_{t,h,k}^2 + \varepsilon}} \cdot w_j \cdot \operatorname{SiLU}(Z_{t,h,j})$$

All reductions, weight scaling, and SiLU gating are computed in float32 and cast back to `core_attn_out.dtype`.

## Parameters and return value

| Parameter | Type | Shape | Dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- | --- |
| `core_attn_out` | `torch.Tensor` | `[T, NV, D]` | `torch.bfloat16`, `torch.float16` | Required | Contiguous attention output tensor on NPU. |
| `projected_states_qkvz` | `torch.Tensor` | `[T, QKVZ_DIM]` | `torch.bfloat16`, `torch.float16` | Required | Contiguous packed QKVZ projection tensor on NPU. |
| `norm_weight` | `torch.Tensor` | `[D]` | `torch.bfloat16`, `torch.float16`, `torch.float32` | Required | Contiguous per-head RMSNorm weight vector. |
| `z_offset` | `int` | — | — | Required | Starting column index of `z` in `projected_states_qkvz` (`z_offset + NV * D <= QKVZ_DIM`). |
| `eps` | `float` | — | — | `1e-6` | Stability epsilon added to mean square. |
| `impl` | `str` | — | — | `"auto"` | Keyword-only; `"auto"`, `"triton"`, or `"torch_native"`. |

| Return value | Type | Shape | Dtype | Description and constraints |
| --- | --- | --- | --- | --- |
| `output` | `torch.Tensor` | `[T, NV * D]` | Same as `core_attn_out` | Newly allocated contiguous gated RMSNorm output. |

## triton backend

- Schedules up to 56 persistent AIV programs with full-row `HEADS_PER_PROG = NV` when `T >= 56` and `NV <= 64`, and adaptive head-group tiling for small decode batches (`T <= 16`).
- Uses `tl.int64` token-stride indexing so long-context prefills (`T > 131072`) do not overflow `int32`.

## torch_native backend

- Independent PyTorch implementation computing float32 RMSNorm and `F.silu` on the sliced `z` view.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.attention.gdn import fused_rmsnorm_gated_strided_z

T, nk, nv, d = 32, 16, 48, 128
qkv_dim = (2 * nk + nv) * d
core = torch.randn(T, nv, d, device="npu", dtype=torch.bfloat16)
qkvz = torch.randn(T, qkv_dim + nv * d, device="npu", dtype=torch.bfloat16)
w = torch.randn(d, device="npu", dtype=torch.bfloat16)

y = fused_rmsnorm_gated_strided_z(core, qkvz, w, qkv_dim, eps=1e-6, impl="auto")
```
