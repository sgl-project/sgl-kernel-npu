# fused_kda_gate

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Supported | Supported |
| `torch_native` | Supported | Supported |

## Signature

`auto` selects `triton`.

```python
from sgl_kernel_npu.operators.attention.kda import fused_kda_gate


def fused_kda_gate(
    gate,
    A_log,
    head_dim,
    *,
    gate_bias=None,
    beta=1.0,
    threshold=20.0,
    lower_bound=None,
    impl="auto"
): ...
```

## Description

Activates KDA channel gates in FP32. The default computes -exp(A_log)*softplus(gate+bias). A lower_bound switches to lower_bound*sigmoid(exp(A_log)*(gate+bias)). The last packed dimension is expanded to [heads,head_dim].

## Parameters and return value

All tensors must be on the same NPU device.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `gate` | Tensor | `Required` | Floating [...,heads*head_dim] packed gates. |
| `A_log` | Tensor | `Required` | Floating [heads] (or equivalent singleton-expanded shape) decay weights. |
| `head_dim` | int | `Required` | Channels per gate head. |
| `gate_bias` | Tensor or None | `None` | Optional floating [heads*head_dim] additive gate bias. |
| `beta` | float | `1.0` | Softplus multiplier or SiTU saturation parameter, according to the operation. |
| `threshold` | float | `20.0` | Softplus linear-branch threshold. |
| `lower_bound` | float or None | `None` | Select bounded sigmoid gates when supplied; None selects softplus. |
| `impl` | str | `'auto'` | auto selects triton; explicit backend selection. |

| Return | Type | Description |
| --- | --- | --- |
| result | Tensor | FP32 [...,heads,head_dim] gate activation. |

## triton

Uses Triton-Ascend kernels.

## torch_native

Uses PyTorch tensor operations on the input device.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.attention.kda import fused_kda_gate

gate = torch.randn(4, 256, device="npu", dtype=torch.bfloat16)
a = torch.zeros(2, device="npu")
y = fused_kda_gate(gate, a, 128, impl="triton")
```
