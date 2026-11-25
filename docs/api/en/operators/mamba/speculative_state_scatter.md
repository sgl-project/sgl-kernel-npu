# speculative_state_scatter

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Supported | Supported |
| `torch_native` | Supported | Supported |

## Signature

`auto` selects `triton`.

```python
from sgl_kernel_npu.operators.mamba import speculative_state_scatter


def speculative_state_scatter(
    dst, src, dst_indices, src_indices, step_indices, *, impl="auto"
): ...
```

## Description

Commits one speculative convolution state snapshot per request across all layers. For request `r`, it copies `src[:, src_indices[r], step_indices[r]]` to `dst[:, dst_indices[r]]` in place. The indices are converted to INT32. A negative value in any of the three indices skips that request and leaves its destination untouched.

## Parameters and return value

Place all tensors on the same NPU. The state tail can have one to three axes and must match between the two tensors. Non-contiguous state tensors are supported when their destination slices are writable and disjoint.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `dst` | Tensor | Required | Writable persistent state of shape `[layers, slots, *state]`. |
| `src` | Tensor | Required | Snapshots of shape `[layers, scratch, steps, *state]` with the same dtype as `dst`. |
| `dst_indices` | Tensor | Required | Integer `[requests]` destination slots. Active slots must be in range and distinct across requests. |
| `src_indices` | Tensor | Required | Integer `[requests]` source slots. Active slots must be in range. |
| `step_indices` | Tensor | Required | Integer `[requests]` snapshot steps. Active steps must be in range. |
| `impl` | str | `"auto"` | Selects the implementation. |

| Return | Type | Description |
| --- | --- | --- |
| result | Tensor | The original `dst` tensor, including when the request list is empty. |

## triton backend

- Copies each selected snapshot using the strides of the source and destination tensors.
- Masks negative indices on the device so a captured NPU graph can replay with different index contents.

## torch_native backend

- Filters inactive requests with PyTorch and uses indexed reads and in-place writes to copy the remaining snapshots.
- Use the Triton implementation for NPU graph capture because the PyTorch index selection has a data-dependent size.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.mamba import speculative_state_scatter

dst = torch.zeros(2, 4, 3, 128, device="npu", dtype=torch.bfloat16)
src = torch.randn(2, 3, 4, 3, 128, device="npu", dtype=dst.dtype)
dst_indices = torch.tensor([2], device="npu", dtype=torch.int32)
src_indices = torch.tensor([1], device="npu", dtype=torch.int32)
step_indices = torch.tensor([3], device="npu", dtype=torch.int32)
result = speculative_state_scatter(
    dst, src, dst_indices, src_indices, step_indices, impl="triton"
)
assert result is dst
```
