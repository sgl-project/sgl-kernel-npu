# grouped_gemma_rmsnorm

## Hardware support

| impl | arch22 | arch35 |
| --- | --- | --- |
| `triton` / `auto` | Supported | Supported |
| `torch_native` | Supported | Supported |

## Signature

`auto` → `triton`.

```python
from sgl_kernel_npu.operators.norm import grouped_gemma_rmsnorm

def grouped_gemma_rmsnorm(x, weight, group_size, eps, *, impl="auto"):
    ...
```

## Description

Grouped Gemma RMSNorm for contiguous BF16 x[R,10240], weight[10240], group_size=2560 and finite positive eps. Each group has independent statistics and weights. Returns BF16 shaped as x. The Triton implementation reuses the existing normalization kernel with FP32 (1+weight).

All tensors are on the same NPU. Inputs are not modified; zero rows return empty output.

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

Default implementation. Checks input metadata before launch. Supports graph capture and replay after in-place input updates.

## torch_native

Independent CPU FP64 reference; returns results to the input device. Use outside graph capture. Select explicitly with `impl="torch_native"`.

## Example

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.norm import grouped_gemma_rmsnorm

x = torch.randn(1, 10240, device="npu", dtype=torch.bfloat16)
weight = torch.zeros(10240, device="npu", dtype=x.dtype)
y = grouped_gemma_rmsnorm(x, weight, 2560, 1e-6, impl="auto")
```
