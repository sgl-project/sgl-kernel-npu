# swiglu_quant

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Supported | Supported |
| `torch_native` | Supported | Supported |

## Signature

`auto` selects `triton`.

```python
from sgl_kernel_npu.operators.activation import swiglu_quant


def swiglu_quant(
    x,
    group_list,
    group_list_type,
    need_quant=True,
    do_limit=False,
    limit=7.0,
    *,
    impl="auto"
): ...
```

## Description

Applies split-layout SwiGLU to routed MoE rows and optionally quantizes each row to INT8. do_limit clamps SiLU(gate) from above and clamps up on both sides before multiplication. Quantization uses max(abs(value))/127, floor(value/scale+0.5), and saturation to [-128,127]. Unrouted rows and scales without quantization are unspecified.

## Parameters and return value

All tensors must be on the same NPU device.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `x` | Tensor | `Required` | Contiguous floating [rows,2*D] gate/up halves. |
| `group_list` | Tensor | `Required` | Contiguous INT32/INT64 expert counts. Type 0 requires an extra terminal total at storage offset numel(), outside the logical view. |
| `group_list_type` | int | `Required` | 1 for counts; 0 for cumulative metadata with terminal storage. |
| `need_quant` | bool | `True` | Return INT8 values and FP32 row scales when true. |
| `do_limit` | bool | `False` | Clamp SiLU(gate) and up before multiplication. |
| `limit` | float | `7.0` | Clamp magnitude. |
| `impl` | str | `'auto'` | auto selects triton; explicit backend selection. |

| Return | Type | Description |
| --- | --- | --- |
| result | Tuple[Tensor,Tensor] | [rows,D] values and [rows] FP32 scales; scales are unspecified without quantization. |

## triton

Uses Triton-Ascend kernels.

## torch_native

Uses PyTorch tensor operations on the input device.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.activation import swiglu_quant

x = torch.randn(4, 3072, device="npu", dtype=torch.bfloat16)
groups = torch.tensor([4], device="npu", dtype=torch.int64)
y, scale = swiglu_quant(x, groups, 1, impl="triton")
```
