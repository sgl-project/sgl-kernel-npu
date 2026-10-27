# l1_norm

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Registered; unverified | Registered; unverified |
| `torch_native` | Registered; unverified | Registered; unverified |

Both generations are registered. Compilation, accuracy and performance validation on real NPUs is pending.

## Signature

```python
from sgl_kernel_npu.operators.norm import l1_norm

def l1_norm(input, *, impl="auto"):
    ...
```

`auto` and `triton` currently refer to the same entry point. Execution failures do not trigger an automatic fallback.

## Description

Divides each row of a two-dimensional input by its signed sum:

$$
Y_{i,j}=\frac{X_{i,j}}{\sum_{k=0}^{H-1}X_{i,k}}
$$

The denominator does not take absolute values or add epsilon. This operation can renormalize selected expert weights. Zero row sums retain the original floating-point division behavior and can produce NaN or Inf.

## Parameters and return value

| Parameter | Type | Shape | Dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- | --- |
| `input` | `torch.Tensor` | `[B, H]` | `torch.float32`, `torch.float16`, `torch.bfloat16` | Required | `H > 0`. Triton requires a contiguous NPU tensor and a power-of-two `H`. |
| `impl` | `str` | — | — | `"auto"` | Keyword-only; accepts `"auto"`, `"triton"` or `"torch_native"`. |

| Return value | Type | Shape | Dtype | Description |
| --- | --- | --- | --- | --- |
| `output` | `torch.Tensor` | `[B, H]` | `torch.float32` | Newly allocated output on the input device. The input is not modified. |

The public entry point adds neither input validation nor contiguous copies. Triton addresses elements as `row * H + column` without reading the tensor's row stride. Callers must ensure that input slices satisfy the contiguous layout requirement.

## triton backend

- Uses the original `l1_norm_kernel` to sum and normalize each row, writing FP32 output.
- Uses `tl.arange(0, H)`, requiring a power-of-two width.

## torch_native backend

- Converts the input to FP32, then computes `values / values.sum(dim=-1, keepdim=True)`.
- Preserves signed row sums and zero-denominator behavior for accuracy comparisons.

## Example

Install a wheel containing this API and the NPU runtime dependencies before running the example.

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.norm import l1_norm

torch.npu.set_device(0)
weights = 0.5 + torch.rand(16, 8, device="npu", dtype=torch.bfloat16)

y_default = l1_norm(weights)
y_triton = l1_norm(weights, impl="triton")
y_native = l1_norm(weights, impl="torch_native")

print(y_triton.shape, y_triton.dtype)  # torch.Size([16, 8]), torch.float32
```
