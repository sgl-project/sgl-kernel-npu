# fused_rsqrt_mul

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Supported | Supported |
| `torch_native` | Supported | Supported |

## Signature

The default option is `auto`, currently registered to the same entry point as `triton`. Omitting `impl` or passing `"auto"` selects Triton.

```python
from sgl_kernel_npu.operators.norm import fused_rsqrt_mul

def fused_rsqrt_mul(x, variance, weight, eps=1e-6, *, impl="auto"):
    ...
```

## Description

Applies the scale and weight of a tensor-parallel RMSNorm using an already synchronized variance. It is the second phase of `fused_variance`. For input $X \in \mathbb{R}^{B\times L\times C}$, variance $V \in \mathbb{R}^{B\times L\times 1}$, and weight $w \in \mathbb{R}^{C}$:

$$
Y_{blj} = X_{blj}\cdot\frac{1}{\sqrt{V_{bl}+\varepsilon}}\cdot w_{j}
$$

`variance` must already contain the tensor-parallel average (`all_reduce(AVG)`); it is usually the output of `fused_variance`. The output keeps the input dtype and inputs are not modified.

## Parameters and return value

The tables describe the recommended calling convention and current implementation requirements. The public operator forwards arguments without input validation.

| Parameter | Type | Shape | Dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- | --- |
| `x` | `torch.Tensor` | `[B, L, C]` | `torch.float32` (this path uses fp32 only) | Required | Input to normalize. Triton requires a contiguous three-dimensional NPU tensor; `C` need not be a power of two. |
| `variance` | `torch.Tensor` | `[B, L, 1]` or `[B*L]` | Same as `x` | Required | Rank-averaged variance on the same device as `x`. Triton reads it row-contiguously. |
| `weight` | `torch.Tensor` | `[C]` | Same as `x` | Required | Multiplicative weight per channel on the same device as `x`. Triton requires it to be contiguous. |
| `eps` | `float` | — | — | `1e-6` | Stability term added to the variance; a finite positive value is recommended. |
| `impl` | `str` | — | — | `"auto"` | Keyword-only; accepts `"auto"`, `"triton"`, or `"torch_native"`. `auto` currently selects the Triton implementation. |

| Return value | Type | Shape | Dtype | Description and constraints |
| --- | --- | --- | --- | --- |
| `output` | `torch.Tensor` | `[B, L, C]` | Same as `x` | Newly allocated output on the same device as `x`. Inputs are not modified. |

Use finite inputs; cross-backend NaN behavior is unspecified. The public interface neither validates inputs nor switches implementations automatically. Package initialization still requires `torch_npu` and the native shared library.

## triton backend

- Uses a CANN Triton kernel tiled over `(num_tokens, C)`; the grid is the vector core count.
- Autotunes `(block_l, block_c)` keyed by `num_tokens` and `hidden_size`. Tail rows and columns are masked, so `C` need not be a power of two.
- Each program walks tasks with a `kernel_num` stride; the output is allocated with `torch.empty_like(x)`.
- Arithmetic follows the input dtype. The production path passes float32, matching the reference implementation.

## torch_native backend

- Uses the broadcast expression `x * torch.rsqrt(variance + eps) * weight`, keeps the input dtype, and supports non-contiguous inputs.
- A flat `[B*L]` variance is interpreted as `[B, L, 1]`, matching the row-wise layout the Triton kernel reads.
- `rsqrt` and reduction order may differ from Triton; float32 differences stay within the default tolerances.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators import list_impls
from sgl_kernel_npu.operators.norm import fused_rsqrt_mul, fused_variance

x = torch.randn(1, 4096, 512, device="npu", dtype=torch.float32)
weight = torch.randn(512, device="npu", dtype=torch.float32)

variance = fused_variance(x)
# A real tensor-parallel call all-reduces variance (AVG) at this point.
print(list_impls(fused_rsqrt_mul))  # ['auto', 'torch_native', 'triton']
y_default = fused_rsqrt_mul(x, variance, weight, 1e-6)  # Default auto
y_triton = fused_rsqrt_mul(x, variance, weight, 1e-6, impl="triton")
y_native = fused_rsqrt_mul(x, variance, weight, 1e-6, impl="torch_native")
torch.testing.assert_close(y_default, y_triton, rtol=0, atol=0)
torch.testing.assert_close(y_triton, y_native, rtol=1e-5, atol=1e-5)
```
