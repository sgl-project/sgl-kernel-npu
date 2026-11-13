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

Computes RMS normalization followed by the Gemma scale (1+weight), returning both normalized output and reciprocal RMS. ARCH22 uses the native Gemma operation; ARCH35 forms 1+weight in the stored weight dtype before RMSNorm, retaining that rounding boundary.

## Parameters and return value

All tensors must be on the same NPU device.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `input` | Tensor | `Required` | Floating NPU [...,H] input. |
| `weight` | Tensor | `Required` | Convolution: [channels,width]. Gemma: [H], same dtype as input. |
| `eps` | float | `1e-06` | RMSNorm stabilization constant. |
| `impl` | str | `'auto'` | auto selects torch_npu; explicit backend selection. |

| Return | Type | Description |
| --- | --- | --- |
| result | Tuple[Tensor,Tensor] | Output has input shape/dtype; FP32 reciprocal RMS has shape [...,1]. |

## torch_npu

Uses the architecture-appropriate native NPU RMSNorm binding.

## torch_native

Uses PyTorch tensor operations on the input device.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.norm import gemma_rmsnorm

x = torch.randn(4, 2304, device="npu", dtype=torch.bfloat16)
w = torch.zeros(2304, device="npu", dtype=x.dtype)
y, rstd = gemma_rmsnorm(x, w, impl="torch_npu")
```
