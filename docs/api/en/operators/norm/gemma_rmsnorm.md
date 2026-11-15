# gemma_rmsnorm

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `torch_npu` | Supported | Supported |
| `torch_native` | Supported | Supported |

## Signature

`auto` selects `torch_npu`.

```python
from sgl_kernel_npu.operators.norm import gemma_rmsnorm


def gemma_rmsnorm(input, weight, eps=1e-06, *, impl="auto"): ...
```

## Description

Computes RMS normalization followed by the Gemma scale `1 + weight`, returning the normalized output and reciprocal RMS.

## Parameters and return value

All tensors must be on the same NPU device.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `input` | Tensor | `Required` | Floating NPU [...,H] input. |
| `weight` | Tensor | `Required` | RMSNorm weight of shape [H], where H is the input's last dimension. Same dtype as input. |
| `eps` | float | `1e-06` | RMSNorm stabilization constant. |
| `impl` | str | `'auto'` | auto selects torch_npu; explicit backend selection. |

| Return | Type | Description |
| --- | --- | --- |
| result | Tuple[Tensor,Tensor] | Output has input shape/dtype; FP32 reciprocal RMS has shape [...,1]. |

## torch_npu backend

- ARCH22 calls torch_npu.npu_gemma_rms_norm directly. ARCH35 computes 1+weight in the weight dtype and calls torch_npu.npu_rms_norm.

## torch_native backend

- Computes reciprocal RMS from the FP32 mean square and rsqrt, multiplies by the input and Gemma scale, and converts output to the input dtype.
- Computes `1 + weight` in FP32 on ARCH22 and in the weight dtype on ARCH35 before converting it to FP32.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.norm import gemma_rmsnorm

x = torch.randn(4, 2304, device="npu", dtype=torch.bfloat16)
w = torch.zeros(2304, device="npu", dtype=x.dtype)
y, rstd = gemma_rmsnorm(x, w, impl="torch_npu")
```
