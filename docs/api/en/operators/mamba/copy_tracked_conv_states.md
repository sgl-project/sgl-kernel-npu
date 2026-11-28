# copy_tracked_conv_states

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Unverified | Supported |
| `torch_native` | Supported | Supported |

## Signature

`auto` selects `triton`.

```python
from sgl_kernel_npu.operators.mamba import copy_tracked_conv_states


def copy_tracked_conv_states(
    conv_states, src_slots, dst_slots, track_steps, *, impl="auto"
): ...
```

## Description

Performs a single-launch cross-layer masked snapshot copy during speculative Radix Cache state tracking: for each request `b` satisfying `track_steps[b] >= 0`, `src_slots[b] >= 0`, `dst_slots[b] >= 0`, and `src_slots[b] != dst_slots[b]`, copies `conv_states[:, src_slots[b]]` into `conv_states[:, dst_slots[b]]` in place. Untracked rows exit early inside the kernel without self-copy overhead.

## Parameters and return value

All tensors must reside on the same NPU device.

| Parameter | Type | Shape | Dtype | Default | Description |
| --- | --- | --- | --- | --- | --- |
| `conv_states` | `torch.Tensor` | `[layers, pool, ...]` (>=2D) | Any numeric dtype | Required | Cross-layer convolution state pool updated in place. |
| `src_slots` | `torch.Tensor` | `[B]` | Integer | Required | Source slot indices (`< 0` skips row). |
| `dst_slots` | `torch.Tensor` | `[B]` | Integer | Required | Destination slot indices (`< 0` or equal to source skips row). |
| `track_steps` | `torch.Tensor` | `[B]` | Integer | Required | Tracking step indices (`< 0` marks untracked rows to skip). |
| `impl` | `str` | — | — | `"auto"` | Keyword-only; `"auto"`, `"triton"`, or `"torch_native"`. |

| Return value | Type | Description |
| --- | --- | --- |
| `result` | `torch.Tensor` | Original `conv_states` alias updated in place. |

## triton backend

- Uses a persistent 56-AIV grid (`BLOCK_E=4096`, `tl.int64` offsets) for contiguous state pools so all layers and requests complete in one launch.
- Falls back cleanly to masked slice copy for non-contiguous views while preserving strides.

## torch_native backend

- Independent per-request PyTorch slice copy reference cloning source slices prior to write-back.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.mamba import copy_tracked_conv_states

conv_states = torch.randn(48, 72, 10, 10240, device="npu", dtype=torch.bfloat16)
src_slots = torch.tensor([0, 1, 2, 3], device="npu", dtype=torch.int64)
dst_slots = torch.tensor([10, 11, 12, 13], device="npu", dtype=torch.int64)
track_steps = torch.tensor([2, -1, 0, -1], device="npu", dtype=torch.int64)
copy_tracked_conv_states(conv_states, src_slots, dst_slots, track_steps, impl="auto")
```
