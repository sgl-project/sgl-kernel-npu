# l2norm_fwd

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Supported | Supported |
| `torch_native` | Supported | Supported |

## Signature

```python
from sgl_kernel_npu.operators.norm import l2norm_fwd

def l2norm_fwd(x, eps=1e-6, output_dtype=None, *, impl="auto"):
    ...
```

`auto` and `triton` currently refer to the same entry point. Execution failures do not trigger an automatic fallback.

## Description

Applies L2 normalization along the last dimension, preserving the input shape:

$$
Y_{\ldots,j}=\frac{X_{\ldots,j}}{\sqrt{\sum_{k=0}^{D-1}X_{\ldots,k}^{2}+\varepsilon}}
$$

`eps` is added to the sum of squares inside the square root. The sum is not divided by the dimension length. Computation uses FP32 intermediates and casts the result to the requested output dtype, defaulting to the input dtype.

## Parameters and return value

| Parameter | Type | Shape | Dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- | --- |
| `x` | `torch.Tensor` | `[..., D]` | `torch.float32`, `torch.float16`, `torch.bfloat16` | Required | At least one dimension with `D > 0`. Triton requires a contiguous NPU tensor. Leading dimensions are merged into rows using `view`. |
| `eps` | `float` | — | — | `1e-6` | Stability term added to the sum of squares. Use a finite, nonnegative value; a positive value avoids zero denominators for zero input. |
| `output_dtype` | `torch.dtype` or `None` | — | `torch.float32`, `torch.float16`, `torch.bfloat16` or — | `None` | None selects x.dtype; otherwise the output is allocated with this dtype. |
| `impl` | `str` | — | — | `"auto"` | Keyword-only; accepts `"auto"`, `"triton"` or `"torch_native"`. |

| Return value | Type | Shape | Dtype | Description |
| --- | --- | --- | --- | --- |
| `output` | `torch.Tensor` | `x.shape` | `output_dtype`, defaulting to `x.dtype` | Newly allocated output on the input device. The input is not modified. |

The public entry point does not add contiguous copies. Both implementations require an input compatible with `view(-1, D)`. Triton has a feature-width limit: `D` must not exceed `65536 // x.element_size()` or it raises `RuntimeError`. The limit is based on the input dtype.

## triton backend

- Uses `l2norm_fwd_kernel_opt` for `D <= 512` and `l2norm_fwd_kernel` for `D > 512`.
- Uses FP32 intermediates with width-dependent tiling and launch configuration. Both internal kernels share one public entry point.

## torch_native backend

- Converts input to FP32 and computes `values * torch.rsqrt(values.square().sum(-1, keepdim=True) + eps)`.
- Writes to an output with the requested dtype and restores the original shape.

## Example

Install a wheel containing this API and the NPU runtime dependencies before running the example.

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.norm import l2norm_fwd

torch.npu.set_device(0)
x = torch.randn(2, 3, 128, device="npu", dtype=torch.bfloat16)

y_default = l2norm_fwd(x)  # Keeps BF16 output
y_triton = l2norm_fwd(x, output_dtype=torch.float32, impl="triton")
y_native = l2norm_fwd(x, output_dtype=torch.float32, impl="torch_native")

print(y_triton.shape, y_triton.dtype)  # torch.Size([2, 3, 128]), torch.float32
```
