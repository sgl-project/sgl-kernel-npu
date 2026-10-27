# fused_variance

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Supported | Supported |
| `torch_native` | Supported | Supported |

## Signature

The default option is `auto`, currently registered to the same entry point as `triton`. Omitting `impl` or passing `"auto"` selects Triton.

```python
from sgl_kernel_npu.operators.norm import fused_variance

def fused_variance(x, *, impl="auto"):
    ...
```

## Description

Computes the per-token mean square over the last dimension. It is the first phase of a tensor-parallel RMSNorm. For input $X \in \mathbb{R}^{B\times L\times C}$:

$$
v_{bl} = \frac{1}{C}\sum_{j=0}^{C-1}X_{blj}^{2}
$$

The result has shape `[B, L, 1]` and keeps the input dtype. It is a **local** variance: under tensor parallelism each rank holds only `C` channels, so the caller must `all_reduce(AVG)` the result before passing it to `fused_rsqrt_mul`. Calling this operator alone does not complete an RMSNorm.

## Parameters and return value

The tables describe the recommended calling convention and current implementation requirements. The public operator forwards arguments without input validation.

| Parameter | Type | Shape | Dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- | --- |
| `x` | `torch.Tensor` | `[B, L, C]` | `torch.float32` (this path uses fp32 only) | Required | Input to reduce. Call with `B > 0, L > 0, C > 0`. Triton requires a contiguous three-dimensional NPU tensor whose `C` is a power of two. |
| `impl` | `str` | — | — | `"auto"` | Keyword-only; accepts `"auto"`, `"triton"`, or `"torch_native"`. `auto` currently selects the Triton implementation. |

| Return value | Type | Shape | Dtype | Description and constraints |
| --- | --- | --- | --- | --- |
| `variance` | `torch.Tensor` | `[B, L, 1]` | Same as `x` | Newly allocated local mean square on the same device as `x`. Inputs are not modified. |

Use finite inputs; cross-backend NaN behavior is unspecified. The public interface neither validates inputs nor switches implementations automatically. Package initialization still requires `torch_npu` and the native shared library.

## triton backend

- Uses a CANN Triton kernel that flattens `B*L` tokens and reduces per row; the grid is the vector core count.
- Autotunes `block_l ∈ {96, 64, 32, 16, 8, 4, 2, 1}` keyed by `num_tokens`.
- Uses `tl.arange(0, C)` for the reduction dimension, so `C` must be a power of two; requires contiguous three-dimensional input.
- Accumulation follows the input dtype. The production path passes float32, matching the reference implementation.

## torch_native backend

- Uses `x.square().sum(dim=-1, keepdim=True) / C`, keeps the input dtype, and supports arbitrary leading dimensions and non-contiguous inputs.
- Reduction order may differ from Triton; float32 differences stay within the default tolerances.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators import list_impls
from sgl_kernel_npu.operators.norm import fused_rsqrt_mul, fused_variance

x = torch.randn(1, 4096, 512, device="npu", dtype=torch.float32)
weight = torch.randn(512, device="npu", dtype=torch.float32)

print(list_impls(fused_variance))  # ['auto', 'torch_native', 'triton']
v_default = fused_variance(x)  # Default auto
v_triton = fused_variance(x, impl="triton")
v_native = fused_variance(x, impl="torch_native")
torch.testing.assert_close(v_default, v_triton, rtol=0, atol=0)
torch.testing.assert_close(v_triton, v_native, rtol=1e-5, atol=1e-5)

# A real tensor-parallel call all-reduces variance (AVG) at this point.
y = fused_rsqrt_mul(x, v_triton, weight, 1e-6)
```
