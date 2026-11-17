# short_conv

## Hardware support

| impl | arch22 | arch35 |
| --- | --- | --- |
| `triton` / `auto` | Supported | Supported |
| `torch_native` | Supported | Supported |

## Signature

`auto` → `triton`.

```python
from sgl_kernel_npu.operators.mamba import short_conv

def short_conv(conv_input, weight, dilation, *, single_token=False, impl="auto"):
    ...
```

## Description

Depthwise dilation-3 PLE convolution. Input [B,10240,9+W], W=1..4; matching BF16/FP16 weight [10240,1,4]. Returns contiguous [B,W,10240], or [B,10240] for single_token=True. auto selects Triton.

All tensors are on the same NPU. Inputs are not modified; zero rows return empty output.

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

Default implementation. Checks input metadata before launch. Supports graph capture and replay after in-place input updates.

## torch_native

Independent reference using native NPU convolution. Use outside graph capture. Select explicitly with `impl="torch_native"`.

## Example

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.mamba import short_conv

x = torch.randn(1, 10240, 13, device="npu", dtype=torch.bfloat16)
w = torch.randn(10240, 1, 4, device="npu", dtype=x.dtype)
y = short_conv(x, w, 3, impl="auto")
```
