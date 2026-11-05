# fused_rmsnorm_without_weight

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Supported | Supported |
| `torch_native` | Supported | Supported |

## Signature

The default option is `auto`, currently registered to the same entry point as `triton`. Omitting `impl` or passing `"auto"` selects Triton.

```python
from sgl_kernel_npu.operators.norm import fused_rmsnorm_without_weight

def fused_rmsnorm_without_weight(x, eps, *, impl="auto"):
    ...
```

## Description

Applies weightless, bias-free RMS normalization over the last dimension. For input $X \in \mathbb{R}^{B\times L\times C}$, for every $(b,l)$ row:

$$
v = \frac{1}{C}\sum_{j=0}^{C-1}X_{j}^{2}
$$

$$
Y_{j} = \frac{X_{j}}{\sqrt{v+\varepsilon}}
$$

The result is cast back to the input dtype; inputs are not modified.

## Parameters and return value

The tables describe the recommended calling convention and current implementation requirements. The public operator forwards arguments without input validation.

| Parameter | Type | Shape | Dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- | --- |
| `x` | `torch.Tensor` | `[B, L, C]` | `torch.float32`, `torch.float16`, `torch.bfloat16` | Required | Input to normalize. Triton requires a contiguous three-dimensional NPU tensor whose `C` is a power of two. |
| `eps` | `float` | — | — | Required | Stability term added to the mean square; a finite positive value such as `1e-6` is recommended. |
| `impl` | `str` | — | — | `"auto"` | Keyword-only; accepts `"auto"`, `"triton"`, or `"torch_native"`. `auto` currently selects the Triton implementation. |

| Return value | Type | Shape | Dtype | Description and constraints |
| --- | --- | --- | --- | --- |
| `output` | `torch.Tensor` | `[B, L, C]` | Same as `x` | Newly allocated output on the same device as `x`. Inputs are not modified. |

Use finite inputs; cross-backend NaN behavior is unspecified. The public interface neither validates inputs nor switches implementations automatically. Package initialization still requires `torch_npu` and the native shared library.

## triton backend

- Uses a CANN Triton kernel that flattens `B*L` tokens and reduces per row; non-quantized output only.
- Autotunes `block_l ∈ {64, 32, 16, 8, 4, 2, 1}` keyed by `num_tokens` and `C`; the grid is the vector core count. Candidate tiles are limited to `max(8192, C)` elements to bound FP32 intermediate storage.
- Uses `tl.arange(0, C)` for the reduction dimension, so `C` must be a power of two; requires contiguous three-dimensional input.
- Input values are promoted to float32 before squaring. Mean-square reduction and normalization use float32; the result is cast back to the input dtype.

## torch_native backend

- Uses PyTorch operations: cast to float32, compute the mean square and `rsqrt`, then cast back to the input dtype; supports arbitrary leading dimensions and non-contiguous inputs.
- Reduction order and the `rsqrt` implementation may differ slightly from Triton; results agree within usual floating-point tolerances.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators import list_impls
from sgl_kernel_npu.operators.norm import fused_rmsnorm_without_weight

x = torch.randn(2, 16, 2048, device="npu", dtype=torch.bfloat16)

print(list_impls(fused_rmsnorm_without_weight))  # ['auto', 'torch_native', 'triton']
y_default = fused_rmsnorm_without_weight(x, 1e-6)  # Default auto selects Triton
y_auto = fused_rmsnorm_without_weight(x, 1e-6, impl="auto")
y_triton = fused_rmsnorm_without_weight(x, 1e-6, impl="triton")
y_native = fused_rmsnorm_without_weight(x, 1e-6, impl="torch_native")
torch.testing.assert_close(y_default, y_triton, rtol=0, atol=0)
torch.testing.assert_close(y_auto, y_triton, rtol=0, atol=0)
torch.testing.assert_close(y_triton, y_native, rtol=1e-2, atol=1e-2)
```
