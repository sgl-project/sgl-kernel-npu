# situ

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Supported | Supported |
| `torch_native` | Supported | Supported |

## Signature

`auto` selects `triton`.

```python
from sgl_kernel_npu.operators.activation import situ


def situ(
    hidden_states,
    group_list,
    group_list_type,
    *,
    need_quant,
    beta=4.0,
    linear_beta=25.0,
    impl="auto"
): ...
```

## Description

Grouped SiTU applies beta*tanh(gate/beta)*sigmoid(gate) to the first half of each row and multiplies by the second half, optionally bounded by linear_beta*tanh(up/linear_beta). Only routed rows are written. Optional INT8 output uses per-row max-absolute scales, a 1e-30 scale floor, floor(x/scale+0.5), and saturation to [-128,127].

## Parameters and return value

All tensors must be on the same NPU device.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `hidden_states` | Tensor | `Required` | Contiguous floating [rows,2*D] input. |
| `group_list` | Tensor | `Required` | Contiguous INT32/INT64 expert counts. Type 0 requires an extra terminal total at storage offset numel(), outside the logical view. |
| `group_list_type` | int | `Required` | 1 for counts; 0 for cumulative metadata with terminal storage. |
| `need_quant` | bool | `Required` | Return INT8 values and FP32 row scales when true. |
| `beta` | float | `4.0` | Softplus multiplier or SiTU saturation parameter, according to the operation. |
| `linear_beta` | float or None | `25.0` | SiTU up-channel bound; None disables the tanh bound. |
| `impl` | str | `'auto'` | auto selects triton; explicit backend selection. |

| Return | Type | Description |
| --- | --- | --- |
| result | Tuple[Tensor, Tensor or None] | [rows,D] activation or INT8 output and [rows] FP32 scales (None without quantization). |

## triton

Uses Triton-Ascend kernels.

## torch_native

Uses PyTorch tensor operations on the input device.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.activation import situ

x = torch.randn(4, 3072, device="npu", dtype=torch.bfloat16)
groups = torch.tensor([4], device="npu", dtype=torch.int64)
y, scale = situ(x, groups, 1, need_quant=True, impl="triton")
```
