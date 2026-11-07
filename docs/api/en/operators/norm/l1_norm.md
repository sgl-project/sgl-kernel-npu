# l1_norm

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Supported | Supported |
| `torch_native` | Supported | Supported |

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

The denominator does not take absolute values or add epsilon. This operation can renormalize selected expert weights. Zero row sums can produce NaN or Inf.

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

- Uses `l1_norm_kernel` to sum and normalize each row, writing FP32 output.
- Uses `tl.arange(0, H)`, requiring a power-of-two width.

## torch_native backend

- Computes with FP32 NumPy values on CPU: `values = input.to(torch.float32).cpu().numpy()`, then `values / values.sum(axis=-1, keepdims=True)`.
- Converts the NumPy result to an FP32 Tensor and returns it to the input device, preserving the public return contract.
- Requires NumPy. NPU input is copied to the CPU and results are copied back. This accuracy reference is unsuitable for execution inside NPU graph capture.

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
