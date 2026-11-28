# conv_state_rollback

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Supported | Supported |
| `torch_native` | Supported | Supported |

## Signature

`auto` selects `triton`.

```python
from sgl_kernel_npu.operators.mamba import conv_state_rollback


def conv_state_rollback(
    conv_states,
    state_indices,
    step_indices,
    draft_token_num,
    *,
    active_tail=None,
    impl="auto",
): ...
```

## Description

Rolls convolution history to the right by draft_token_num-1-step for each selected request across all layers. Only the shifted suffix is overwritten; the leading part stays unchanged. When `active_tail` is provided (for example `active_tail=3` for `kernel_size=4` causal depthwise conv1d), only the trailing `active_tail` active window rows are restored to reduce HBM traffic. Non-contiguous state views retain their original storage and strides.

## Parameters and return value

All tensors must be on the same NPU device.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `conv_states` | Tensor | `Required` | Convolution states of shape [layers, pool, window, channels], updated in place. |
| `state_indices` | Tensor | `Required` | Integer [requests] rollback slots, converted to INT32. |
| `step_indices` | Tensor | `Required` | Integer [requests] accepted-step indices, converted to INT32. |
| `draft_token_num` | int | `Required` | Number of draft tokens. |
| `active_tail` | int or None | `None` | Optional number of trailing active window rows to restore; `None` rolls back the full window. |
| `impl` | str | `'auto'` | auto selects triton; explicit backend selection. |

| Return | Type | Description |
| --- | --- | --- |
| result | Tensor or None | Original conv_states alias; None for an empty selection. |

## triton backend

- Computes each request’s rollback shift as draft_token_num-1-step_indices and moves the history window in the original state view.
- Uses a persistent 56-AIV channel-tiled kernel (`BLOCK_D <= 8192`) for contiguous 4D states, supporting arbitrary non-power-of-two channel widths (such as `10240`), and preloads all 3 source rows into registers for `active_tail=3`.
- Uses a channel-tiled strided kernel for non-contiguous layouts, preserving original storage and strides.

## torch_native backend

- Computes the rollback shift per request and moves state history with slicing and copying.
- Clones the source slice before copying to avoid overwriting unread data in overlapping regions, preserving the destination object and layout.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.mamba import conv_state_rollback

state = torch.randn(2, 4, 6, 128, device="npu", dtype=torch.bfloat16)
slots = torch.tensor([1], device="npu", dtype=torch.int32)
steps = torch.tensor([1], device="npu", dtype=torch.int32)
conv_state_rollback(state, slots, steps, 4, impl="triton")
```
