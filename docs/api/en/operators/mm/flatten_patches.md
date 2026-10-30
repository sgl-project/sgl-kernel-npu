# flatten_patches

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `torch_native` | Unverified | Unverified |

The implementation uses PyTorch tensor operations. NPU execution and performance
of this extracted API have not yet been validated on either architecture.

## Signature

```python
from sgl_kernel_npu.operators.mm import flatten_patches

def flatten_patches(patches, patch_size, merge_size, temporal_patch_size, *, impl="auto"):
    ...
```

`auto` selects `torch_native`.

## Description

Convert GLM image/video pixels into flattened patches. Repeat the final frame
until the frame count is divisible by `temporal_patch_size`. Spatial tokens are
ordered by temporal grid, merged block row, merged block column, row within the
merged block and column within the merged block. Features are ordered by channel,
frame within the temporal patch, pixel row and pixel column.

With `P=patch_size`, `M=merge_size`, `TP=temporal_patch_size`, the grid is
`(ceil(T/TP), H/P, W/P)`. The implementation uses at most eight dimensions for
each reshape/permute, avoiding the upstream 10-D permutation on NPU. Inputs are
read-only; the output may alias the input for identity layouts.

Extracted from `_npu_safe_flatten_patches` in SGLang
[`glm_vl_utils.py` at 932adcd205](https://github.com/IceShip423/sglang/blob/932adcd20532c57e59e89076e49bf71789650293/python/sglang/srt/hardware_backend/npu/modules/glm_vl_utils.py).

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

Uses frame expansion/concatenation and two low-rank layout transformations.
This layout-only operator has one implementation; `auto` is an alias, not an
independent reference. Tests compare bitwise with the upstream 10-D transformation
executed on CPU, then transferred to the input device. No 10-D reference runs on
NPU. The tests provide optional profiling for representative image/video shapes;
no performance results are claimed here.

## Example

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.mm import flatten_patches

pixels = torch.randn((1, 3, 3, 28, 28), device="npu", dtype=torch.bfloat16)
patches, grid_t, grid_h, grid_w = flatten_patches(
    pixels, patch_size=14, merge_size=2, temporal_patch_size=2, impl="torch_native"
)
assert patches.shape == (1, 8, 1176)
assert (grid_t, grid_h, grid_w) == (2, 2, 2)
```
