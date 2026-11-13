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
    conv_states, state_indices, step_indices, draft_token_num, *, impl="auto"
): ...
```

## Description

Rolls convolution history to the right by draft_token_num-1-step for each selected request across all layers. Only the shifted suffix is overwritten; the leading part stays unchanged. Non-contiguous state views retain their original storage and strides.

## Parameters and return value

All tensors must be on the same NPU device.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `conv_states` | Tensor | `Required` | Prefill: [pool,channels,width-1]; rollback: [layers,pool,window,channels]. |
| `state_indices` | Tensor | `Required` | Integer [requests] rollback slots, converted to INT32. |
| `step_indices` | Tensor | `Required` | Integer [requests] accepted-step indices, converted to INT32. |
| `draft_token_num` | int | `Required` | Number of draft tokens. |
| `impl` | str | `'auto'` | auto selects triton; explicit backend selection. |

| Return | Type | Description |
| --- | --- | --- |
| result | Tensor or None | Original conv_states alias; None for an empty selection. |

## triton

Uses Triton-Ascend kernels.

## torch_native

Uses PyTorch tensor operations on the input device.

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
