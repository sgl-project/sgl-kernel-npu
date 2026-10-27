# fused_rope_qk_mqa

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Unverified | Unverified |
| `torch_native` | Unverified | Unverified |
| `tilelang` | Not implemented | Not implemented |
| `ascendc` | Not implemented | Not implemented |

## Signature

```python
from sgl_kernel_npu.operators.rope import fused_rope_qk_mqa

def fused_rope_qk_mqa(
    query, key, cos_sin, rotary_dim, is_neox_style, *, impl="auto"
):
    ...
```

`auto` and `triton` are registered to the same entry point. The default uses Triton, with backend modules imported lazily. Unsupported inputs or unavailable implementations raise an error without automatic fallback.

## Description

Applies rotary position embedding (RoPE) to the first `rotary_dim` channels of Query and Key. Their head counts may differ; Key is not expanded to match the Query head count.

Let the rotary dimension be $R$ and $m=R/2$. Each cache row stores `[cos | sin]`, with $c_{t,j}=\mathrm{cos\_sin}_{t,j}$ and $s_{t,j}=\mathrm{cos\_sin}_{t,j+m}$. For any Query or Key head, the channel pairs are:

$$
(a_j,b_j)=\begin{cases}
(j,j+m), & \text{is\_neox\_style=True},\\
(2j,2j+1), & \text{is\_neox\_style=False}.
\end{cases}
$$

$$
Y_{t,h,a_j}=X_{t,h,a_j}c_{t,j}-X_{t,h,b_j}s_{t,j},\qquad
Y_{t,h,b_j}=X_{t,h,a_j}s_{t,j}+X_{t,h,b_j}c_{t,j}.
$$

The unrotated tail $Y_{t,h,R:D}$ is copied from the input. Cache row `t` is used directly for token `t`; cache rows must already have been selected by position and aligned with the tokens. This interface takes no position indices and performs no gather. Additional rows beyond `T` are ignored.

## Parameters and return values

| Parameter | Type | Shape | Dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- | --- |
| `query` | `torch.Tensor` | `[T, Hq, D]` | `torch.float32`, `torch.float16`, `torch.bfloat16` | Required | Query input; every dimension must be positive. |
| `key` | `torch.Tensor` | `[T, Hk, D]` | Same as query | Required | `Hk > 0`; shares query's token count, head dimension, device, and dtype. No divisibility relationship between head counts is required. |
| `cos_sin` | `torch.Tensor` | `[N, rotary_dim]`, `N >= T` | `torch.float32`, `torch.float16`, `torch.bfloat16` | Required | First half of each row stores cos; second half stores sin. Must share query's device, but may have a different dtype. |
| `rotary_dim` | `int` | — | — | Required | Python integer excluding bool; must be positive, even, and no greater than `D`. |
| `is_neox_style` | `bool` | — | — | Required | Python bool; True pairs the two halves, False pairs adjacent even and odd channels. |
| `impl` | `str` | — | — | `"auto"` | Keyword-only; accepts `"auto"`, `"triton"`, or `"torch_native"`. |

All three inputs require `torch.strided` layout. Non-contiguous inputs and read-only views with zero strides are supported. Invalid metadata is rejected with assertions.

| Return value | Type | Shape | Dtype | Description and constraints |
| --- | --- | --- | --- | --- |
| `out_query` | `torch.Tensor` | `[T, Hq, D]` | query.dtype | Newly allocated Query output on query's device. |
| `out_key` | `torch.Tensor` | `[T, Hk, D]` | key.dtype | Newly allocated Key output on key's device. |

Returns `(out_query, out_key)`. Inputs are not modified, and unrotated tail values are preserved.

## triton backend

- Preserves the computation of the kernel in `norm/fused_rope_qk_mqa.py`. The original file and import path remain unchanged.
- All three tensors must be on an NPU. `Hq`, `Hk`, and `rotary_dim // 2` must be powers of two. `D - rotary_dim` may be zero; otherwise it must also be a power of two. These requirements come from unmasked `tl.arange` ranges.
- Each program processes one token. `T` does not need to be a power of two.

## torch_native backend

- Uses independent PyTorch mathematical operations without calling Triton. Supports general valid dimensions without power-of-two restrictions on head counts, half rotary dimensions, or tail lengths.
- Matches [Triton's type promotion rules](https://triton-lang.org/main/python-api/triton-semantics.html): mixing float16 and bfloat16 uses float16; float32 is used if either operand is float32. Results are cast back to the query/key dtype. Low-precision intermediate rounding and fused multiply-add operations can cause numerical differences. Bitwise equality is not guaranteed; use dtype-appropriate tolerances.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.rope import fused_rope_qk_mqa

query = torch.randn(17, 8, 128, device="npu", dtype=torch.float32)
key = torch.randn(17, 1, 128, device="npu", dtype=torch.float32)
# Each row already corresponds to a token; construct or select angles by position.
angles = torch.randn(17, 32, device="npu", dtype=torch.float32)
cos_sin = torch.cat((angles.cos(), angles.sin()), dim=-1)

q_out, k_out = fused_rope_qk_mqa(query, key, cos_sin, 64, True)
q_ref, k_ref = fused_rope_qk_mqa(
    query, key, cos_sin, 64, True, impl="torch_native"
)
torch.testing.assert_close(q_out, q_ref, rtol=1e-4, atol=1e-4)
torch.testing.assert_close(k_out, k_ref, rtol=1e-4, atol=1e-4)
torch.testing.assert_close(q_out[..., 64:], query[..., 64:], rtol=0, atol=0)
torch.testing.assert_close(k_out[..., 64:], key[..., 64:], rtol=0, atol=0)
```
