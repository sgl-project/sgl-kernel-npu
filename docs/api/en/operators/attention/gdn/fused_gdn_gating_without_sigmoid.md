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

Computes FP32 g=-exp(A_log)*softplus(a+dt_bias,beta,threshold), returning g with a leading singleton and forwarding b unchanged. The second result is the original b tensor, with no sigmoid applied.

## Parameters and return value

All tensors must be on the same NPU device.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `A_log` | Tensor | `Required` | Floating [heads] (or equivalent singleton-expanded shape) decay weights. |
| `a` | Tensor | `Required` | GDN: [tokens,heads]. KDA: [tokens,Hk,K] with optional leading singleton. |
| `b` | Tensor | `Required` | Floating [tokens,Hv], optionally [1,tokens,Hv] for KDA. |
| `dt_bias` | Tensor | `Required` | GDN: [heads]; KDA: [Hk,K] gate bias. |
| `beta` | float | `1.0` | Softplus multiplier or SiTU saturation parameter, according to the operation. |
| `threshold` | float | `20.0` | Softplus linear-branch threshold. |
| `impl` | str | `'auto'` | auto selects triton; explicit backend selection. |

| Return | Type | Description |
| --- | --- | --- |
| result | Tuple[Tensor,Tensor] | FP32 [1,tokens,heads] g and the unmodified b tensor. |

## triton

Uses Triton-Ascend kernels.

## torch_native

Uses PyTorch tensor operations on the input device.

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
