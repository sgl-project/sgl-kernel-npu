# short_conv

## Hardware support

| impl | arch22 | arch35 |
| --- | --- | --- |
| `triton` / `auto` | Yes | No |
| `torch_native` | Yes | No |

## Signature

`auto` → `triton`.

```python
from sgl_kernel_npu.operators.mamba import short_conv

def short_conv(conv_input, weight, dilation, *, single_token=False, impl="auto"):
    ...
```

## Description

Depthwise dilation-3 PLE convolution. Input [B,10240,9+W], W=1..4; matching BF16/FP16 weight [10240,1,4]. Returns contiguous [B,W,10240], or [B,10240] for single_token=True. auto selects Triton.

All tensors are on the same NPU. Inputs are not modified; zero rows return empty output. `auto` aliases `triton`, including its native PyTorch substeps. No error-driven fallback is installed. `torch_native` is an independent reference: short convolution uses native NPU convolution; the other references use CPU arithmetic (FP64 for floating-point oracles) and return to the input device. These references synchronize and are intended for accuracy checks outside graph capture.

`out[b,t,c] = sum_j conv_input[b,c,t+3*j] * weight[c,0,j]`, j=0..3. Products accumulate in FP32 and round once to the input dtype.

## Parameters and return value

| Parameter | Type | Shape / dtype | Default | Constraints |
| --- | --- | --- | --- | --- |
| `conv_input` | Tensor | [B,10240,9+W], BF16/FP16 | required | Contiguous NPU tensor; W=1..4. |
| `weight` | Tensor | [10240,1,4], conv_input.dtype | required | Contiguous on the same NPU. |
| `dilation` | int | — | required | Exactly 3. |
| `single_token` | bool | — | False | Only W=1; remove the token dimension. |
| `impl` | str | — | "auto" | Keyword-only: auto, triton, torch_native. |

| Return | Type | Shape | Dtype | Device / layout |
| --- | --- | --- | --- | --- |
| `output` | Tensor | [B,W,10240] or [B,10240] | conv_input.dtype | Input device; contiguous, newly allocated. |

## triton

Default inference implementation. Metadata checks reject unsupported inputs before launch; input contents must satisfy the caller constraints above. Supports graph capture and replay with in-place input and weight updates. Native substeps, where present, are part of this implementation.

## torch_native

Independent accuracy reference for the documented input contract. Synchronizing host transfers are permitted here; run this implementation outside graph capture. It is never selected automatically after a failure.

## Example

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.mamba import short_conv

x = torch.randn(1, 10240, 13, device="npu", dtype=torch.bfloat16)
w = torch.randn(10240, 1, 4, device="npu", dtype=x.dtype)
y = short_conv(x, w, 3, impl="auto")
```
