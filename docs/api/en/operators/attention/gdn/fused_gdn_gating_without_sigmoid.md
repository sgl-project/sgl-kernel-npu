# fused_gdn_gating_without_sigmoid

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Supported | Supported |
| `torch_native` | Supported | Supported |

## Signature

`auto` selects `triton`.

```python
from sgl_kernel_npu.operators.attention.gdn import fused_gdn_gating_without_sigmoid


def fused_gdn_gating_without_sigmoid(
    A_log, a, b, dt_bias, beta=1.0, threshold=20.0, *, impl="auto"
): ...
```

## Description

Computes `g = -exp(A_log) * softplus(a + dt_bias, beta, threshold)` in FP32.

## Parameters and return value

All tensors must be on the same NPU device.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `A_log` | Tensor | `Required` | Floating [heads] (or equivalent singleton-expanded shape) decay weights. |
| `a` | Tensor | `Required` | Floating-point gate input of shape [tokens,heads]. |
| `b` | Tensor | `Required` | Gate tensor paired with `a`. |
| `dt_bias` | Tensor | `Required` | Floating-point gate bias of shape [heads]. |
| `beta` | float | `1.0` | Nonzero softplus coefficient. |
| `threshold` | float | `20.0` | Softplus linear-branch threshold. |
| `impl` | str | `'auto'` | auto selects triton; explicit backend selection. |

| Return | Type | Description |
| --- | --- | --- |
| result | Tuple[Tensor,Tensor] | FP32 `g` of shape [1,tokens,heads] and the original `b` tensor. |

## triton backend

- Fuses bias addition, thresholded softplus, and per-head decay scaling for contiguous gate input `a`.

## torch_native backend

- Uses `exp`, `log1p`, and `where` for thresholded softplus, then applies the decay scale.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.attention.gdn import fused_gdn_gating_without_sigmoid

a = torch.randn(16, 8, device="npu", dtype=torch.bfloat16)
b = torch.randn_like(a)
w = torch.zeros(8, device="npu")
g, raw_beta = fused_gdn_gating_without_sigmoid(w, a, b, w, impl="triton")
```
