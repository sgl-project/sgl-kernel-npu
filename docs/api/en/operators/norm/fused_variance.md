# fused_variance

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Registered; unverified | Registered; unverified |
| `torch_native` | Registered; unverified | Registered; unverified |

Both generations are registered. Compilation, accuracy and performance validation on real NPUs is pending.

## Signature

```python
from sgl_kernel_npu.operators.norm import fused_variance

def fused_variance(x, *, impl="auto"):
    ...
```

`auto` and `triton` currently refer to the same entry point. Execution failures do not trigger an automatic fallback.

## Description

Computes the mean square over the last dimension of a three-dimensional input, retaining that dimension:

$$
Y_{b,l,0}=\frac{1}{C}\sum_{j=0}^{C-1}X_{b,l,j}^{2}
$$

Despite the variance name, the calculation does not subtract the input mean. This API only computes a local mean square; it does not apply RMSNorm scaling, epsilon or cross-rank communication.

## Parameters and return value

| Parameter | Type | Shape | Dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- | --- |
| `x` | `torch.Tensor` | `[B, L, C]` | `torch.float32`, `torch.float16`, `torch.bfloat16` | Required | `C > 0`. Triton requires a contiguous NPU tensor and a power-of-two `C`. |
| `impl` | `str` | — | — | `"auto"` | Keyword-only; accepts `"auto"`, `"triton"` or `"torch_native"`. |

| Return value | Type | Shape | Dtype | Description |
| --- | --- | --- | --- | --- |
| `output` | `torch.Tensor` | `[B, L, 1]` | `x.dtype` | Newly allocated output on the input device. The input is not modified. |

The public entry point preserves the original calling logic without adding validation or contiguous copies. Callers must satisfy the selected backend's input requirements.

## triton backend

- Uses the original `fused_variance_kernel` and autotune configuration, reading input in contiguous layout.
- The hidden dimension is expanded with `tl.arange(0, C)`, requiring a power-of-two width.

## torch_native backend

- Currently computes `x.square().float().mean(dim=-1, keepdim=True).to(x.dtype)`.
- Squares in the input dtype, converts to FP32 for the mean, then casts back to the input dtype.
- Preserves the mean-square shape and dtype contract. Compare backend results using accuracy tolerances.

## Example

Install a wheel containing this API and the NPU runtime dependencies before running the example.

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.norm import fused_variance

torch.npu.set_device(0)
x = torch.randn(2, 17, 128, device="npu", dtype=torch.float32)

y_default = fused_variance(x)
y_triton = fused_variance(x, impl="triton")
y_native = fused_variance(x, impl="torch_native")

print(y_triton.shape, y_triton.dtype)  # torch.Size([2, 17, 1]), torch.float32
```
