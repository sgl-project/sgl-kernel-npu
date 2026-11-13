# swiglu_oai

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Supported | Supported |
| `torch_native` | Supported | Supported |

## Signature

`auto` selects `triton`.

```python
from sgl_kernel_npu.operators.activation import swiglu_oai


def swiglu_oai(hidden_states, dim, gemm1_alpha, gemm1_clamp_limit, *, impl="auto"): ...
```

## Description

Applies GPT-OSS-style SwiGLU to interleaved gate/up channels. Even channels are gate and odd channels are up. Gate is capped at limit; up is clamped to [-limit,limit]. The output is gate*sigmoid(alpha*gate)*(up+1), flattened to [rows,dim/2].

## Parameters and return value

All tensors must be on the same NPU device.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `hidden_states` | Tensor | `Required` | Contiguous floating [rows,2*D] input. |
| `dim` | int | `Required` | Packed gate/up width, equal to the last input dimension. |
| `gemm1_alpha` | float | `Required` | Sigmoid input multiplier. |
| `gemm1_clamp_limit` | float | `Required` | Gate upper bound and symmetric up bound. |
| `impl` | str | `'auto'` | auto selects triton; explicit backend selection. |

| Return | Type | Description |
| --- | --- | --- |
| result | Tensor | [rows,dim/2] output, input dtype/device. |

## triton

Uses Triton-Ascend kernels.

## torch_native

Uses PyTorch tensor operations on the input device.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.activation import swiglu_oai

x = torch.randn(4, 5760, device="npu", dtype=torch.bfloat16)
y = swiglu_oai(x, 5760, 1.702, 7.0, impl="triton")
```
