# grouped_gemma_rmsnorm

## Hardware support

| impl | arch22 | arch35 |
| --- | --- | --- |
| `triton` / `auto` | Yes | No |
| `torch_native` | Yes | No |

## Signature

`auto` → `triton`.

```python
from sgl_kernel_npu.operators.norm import grouped_gemma_rmsnorm

def grouped_gemma_rmsnorm(x, weight, group_size, eps, *, impl="auto"):
    ...
```

## Description

Grouped Gemma RMSNorm for contiguous BF16 x[R,10240], weight[10240], group_size=2560 and finite positive eps. Each group has independent statistics and weights. Returns BF16 shaped as x. auto composes public grouped RMSNorm with FP32 (1+weight).

All tensors are on the same NPU. Inputs are not modified; zero rows return empty output. `auto` aliases `triton`, including its native PyTorch substeps. No error-driven fallback is installed. `torch_native` is an independent reference: short convolution uses native NPU convolution; the other references use CPU arithmetic (FP64 for floating-point oracles) and return to the input device. These references synchronize and are intended for accuracy checks outside graph capture.

Each group computes `x * rsqrt(mean(x*x) + eps) * (1 + weight)` with FP32 statistics and scaling, then rounds to BF16.

## Parameters and return value

| Parameter | Type | Shape / dtype | Default | Constraints |
| --- | --- | --- | --- | --- |
| `x` | Tensor | [R,10240], BF16 | required | Contiguous NPU input. |
| `weight` | Tensor | [10240], BF16 | required | Contiguous on x.device; add 1 in FP32. |
| `group_size` | int | — | required | 2560; four independent groups. |
| `eps` | float | — | required | Finite and positive. |
| `impl` | str | — | "auto" | Keyword-only: auto, triton, torch_native. |

| Return | Type | Shape | Dtype | Device / layout |
| --- | --- | --- | --- | --- |
| `output` | Tensor | x.shape | BF16 | Input device; contiguous, newly allocated. |

## triton

Default inference implementation. Metadata checks reject unsupported inputs before launch; input contents must satisfy the caller constraints above. Supports graph capture and replay with in-place input and weight updates. Native substeps, where present, are part of this implementation.

## torch_native

Independent accuracy reference for the documented input contract. Synchronizing host transfers are permitted here; run this implementation outside graph capture. It is never selected automatically after a failure.

## Example

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.norm import grouped_gemma_rmsnorm

x = torch.randn(1, 10240, device="npu", dtype=torch.bfloat16)
weight = torch.zeros(10240, device="npu", dtype=x.dtype)
y = grouped_gemma_rmsnorm(x, weight, 2560, 1e-6, impl="auto")
```

The independent CPU FP64 regression uses atol=0.005 and rtol=0.01 for BF16 output. All shapes, including R=4096 and R=8193, use the same tolerance; graph/eager agreement alone is not an accuracy result.
