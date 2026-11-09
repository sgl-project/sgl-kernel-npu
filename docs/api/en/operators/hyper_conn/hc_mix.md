# hc_mix

## Hardware support

| impl | arch22 | arch35 |
| --- | --- | --- |
| `triton` / `auto` | Yes | No |
| `torch_native` | Yes | No |

## Signature

`auto` → `triton`.

```python
from sgl_kernel_npu.operators.hyper_conn import hc_mix

def hc_mix(x, down, up, hc, hs, *, impl="auto"):
    ...
```

## Description

HC sigmoid-gated mixing. Contiguous BF16 x[R,10240], down[320,10240], up[10240,320], hc=4, hs=2560. Returns BF16[R,2560], using FP32 projections and SiLU(x@down.T/4). auto selects the Triton/native hybrid.

All tensors are on the same NPU. Inputs are not modified; zero rows return empty output. `auto` aliases `triton`, including its native PyTorch substeps. No error-driven fallback is installed. `torch_native` is an independent reference: short convolution uses native NPU convolution; the other references use CPU arithmetic (FP64 for floating-point oracles) and return to the input device. These references synchronize and are intended for accuracy checks outside graph capture.

Let `h = SiLU(x @ down.T / 4)` and `g = sigmoid(h @ up.T)`. Reshape x and g to `[R,4,2560]`; output `sum(g*x, dim=1)/4`. Intermediates are FP32; only the output rounds to BF16.

## Parameters and return value

| Parameter | Type | Shape / dtype | Default | Constraints |
| --- | --- | --- | --- | --- |
| `x` | Tensor | [R,10240], BF16 | required | Contiguous NPU input. |
| `down` | Tensor | [320,10240], BF16 | required | Contiguous on x.device. |
| `up` | Tensor | [10240,320], BF16 | required | Contiguous on x.device. |
| `hc / hs` | int | — | required | 4 / 2560. |
| `impl` | str | — | "auto" | Keyword-only: auto, triton, torch_native. |

| Return | Type | Shape | Dtype | Device / layout |
| --- | --- | --- | --- | --- |
| `output` | Tensor | [R,2560] | BF16 | Input device; contiguous, newly allocated. |

## triton

Default inference implementation. Metadata checks reject unsupported inputs before launch; input contents must satisfy the caller constraints above. Supports graph capture and replay with in-place input and weight updates. Native substeps, where present, are part of this implementation.

## torch_native

Independent accuracy reference for the documented input contract. Synchronizing host transfers are permitted here; run this implementation outside graph capture. It is never selected automatically after a failure.

## Example

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.hyper_conn import hc_mix

x = torch.randn(1, 10240, device="npu", dtype=torch.bfloat16)
down = torch.zeros(320, 10240, device="npu", dtype=x.dtype)
up = torch.zeros(10240, 320, device="npu", dtype=x.dtype)
y = hc_mix(x, down, up, 4, 2560, impl="auto")
```
