# flatten_patches

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `torch_native` | Unverified | Verified |

## Signature

```python
from sgl_kernel_npu.operators.multimodal import flatten_patches

def flatten_patches(patches, patch_size, merge_size, temporal_patch_size, *, impl="auto"):
    ...
```

`auto` selects `torch_native`.

## Description

Convert image or video pixels into flattened patches for model input. Repeat the final frame when needed to complete a temporal patch, then group spatial patches by the requested merge size.

Return the flattened features together with the temporal, height and width grid sizes. The layout conversion preserves the required patch and feature order while using tensors of at most eight dimensions. Inputs are not modified; identity layouts may share storage with the input.

## Parameters

| Parameter | Type | Description and constraints |
| --- | --- | --- |
| `patches` | Tensor `[B,T,C,H,W]` | NPU pixel tensor; FP32, FP16 or BF16 tested by the test cases. All dimensions must be positive. Noncontiguous inputs are supported. |
| `patch_size` | int | Positive spatial patch size `P`; required. |
| `merge_size` | int | Positive spatial merge factor `M`; `H` and `W` must be divisible by `P*M`; required. |
| `temporal_patch_size` | int | Positive number of frames per patch `TP`; required. |
| `impl` | str | `auto` (default) or `torch_native`. |

## Return value

| Value | Type | Description |
| --- | --- | --- |
| `flatten_patches` | Tensor `[B,grid_t*grid_h*grid_w,C*TP*P*P]` | Same dtype/device as input. |
| `grid_t` | int | Frame count after padding divided by `TP`. |
| `grid_h` | int | `H/P`. |
| `grid_w` | int | `W/P`. |

## torch_native

Uses frame expansion/concatenation and two layout transformations of at most eight dimensions.

## Example

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.multimodal import flatten_patches

pixels = torch.randn((1, 3, 3, 28, 28), device="npu", dtype=torch.bfloat16)
patches, grid_t, grid_h, grid_w = flatten_patches(
    pixels, patch_size=14, merge_size=2, temporal_patch_size=2, impl="torch_native"
)
assert patches.shape == (1, 8, 1176)
assert (grid_t, grid_h, grid_w) == (2, 2, 2)
```
