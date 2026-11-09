# hc_combine

## Hardware support

| impl | arch22 | arch35 |
| --- | --- | --- |
| `triton` / `auto` | Yes | No |
| `torch_native` | Yes | No |

## Signature

`auto` → `triton`.

```python
from sgl_kernel_npu.operators.hyper_conn import hc_combine

def hc_combine(block, residual, normed, weight, hc, hs, *, impl="auto"):
    ...
```

## Description

HC gated residual injection: contiguous BF16 block[R,2560], residual/normed[R,10240], weight[4,10240], hc=4, hs=2560. Returns BF16 residual + block*2*sigmoid(normed@weight.T/4), without modifying inputs. auto selects the Triton/native hybrid.

All tensors are on the same NPU. Inputs are not modified; zero rows return empty output. `auto` aliases `triton`, including its native PyTorch substeps. No error-driven fallback is installed. `torch_native` is an independent reference: short convolution uses native NPU convolution; the other references use CPU arithmetic (FP64 for floating-point oracles) and return to the input device. These references synchronize and are intended for accuracy checks outside graph capture.

## Parameters and return value

| Parameter | Type | Shape / dtype | Default | Constraints |
| --- | --- | --- | --- | --- |
| `block` | Tensor | [R,2560], BF16 | required | Contiguous NPU input. |
| `residual / normed` | Tensor | [R,10240], BF16 | required | Contiguous on block.device. |
| `weight` | Tensor | [4,10240], BF16 | required | Contiguous on block.device. |
| `hc / hs` | int | — | required | 4 / 2560. |
| `impl` | str | — | "auto" | Keyword-only: auto, triton, torch_native. |

| Return | Type | Shape | Dtype | Device / layout |
| --- | --- | --- | --- | --- |
| `output` | Tensor | [R,10240] | BF16 | Input device; contiguous, newly allocated. |

## triton

Default inference implementation. Metadata checks reject unsupported inputs before launch; input contents must satisfy the caller constraints above. Supports graph capture and replay with in-place input and weight updates. Native substeps, where present, are part of this implementation.

## torch_native

Independent accuracy reference for the documented input contract. Synchronizing host transfers are permitted here; run this implementation outside graph capture. It is never selected automatically after a failure.

## Example

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.hyper_conn import hc_combine

residual = torch.randn(1, 10240, device="npu", dtype=torch.bfloat16)
block = torch.randn(1, 2560, device="npu", dtype=residual.dtype)
weight = torch.zeros(4, 10240, device="npu", dtype=residual.dtype)
y = hc_combine(block, residual, residual, weight, 4, 2560, impl="auto")
```
