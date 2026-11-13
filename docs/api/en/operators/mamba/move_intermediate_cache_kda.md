# move_intermediate_cache_kda

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Supported | Supported |
| `torch_native` | Supported | Supported |

## Signature

`auto` selects `triton`.

```python
from sgl_kernel_npu.operators.mamba import move_intermediate_cache_kda


def move_intermediate_cache_kda(
    ssm_states,
    intermediate_state_cache,
    dst_indices_tensor,
    src_indices_tensor,
    last_steps_tensor,
    h_block_size=1,
    *,
    impl="auto"
): ...
```

## Description

Commits accepted KDA snapshots to persistent state across all layers, honoring the destination head, V and K strides. Negative last-step entries are skipped; an empty selection returns the original destination unchanged.

## Parameters and return value

All tensors must be on the same NPU device.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `ssm_states` | Tensor | `Required` | Writable floating [layers,pool,H,V,K] persistent state. |
| `intermediate_state_cache` | Tensor | `Required` | Floating [layers,pool,steps,H,V,K], contiguous inside H/V/K. |
| `dst_indices_tensor` | Tensor | `Required` | Integer [requests] destination slots, distinct for concurrent requests. |
| `src_indices_tensor` | Tensor | `Required` | Integer [requests] source slots. |
| `last_steps_tensor` | Tensor | `Required` | Integer [requests] snapshot step; negative skips the request. |
| `h_block_size` | int | `1` | Head tiling size; does not change mathematical results. |
| `impl` | str | `'auto'` | auto selects triton; explicit backend selection. |

| Return | Type | Description |
| --- | --- | --- |
| result | Tensor | The original ssm_states tensor, updated in place. |

## triton

Uses Triton-Ascend kernels.

## torch_native

Uses PyTorch tensor operations on the input device.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.mamba import move_intermediate_cache_kda

state = torch.zeros(2, 4, 2, 128, 128, device="npu", dtype=torch.bfloat16)
snapshots = torch.randn(2, 4, 3, 2, 128, 128, device="npu", dtype=state.dtype)
slot = torch.tensor([1], device="npu", dtype=torch.int32)
step = torch.tensor([2], device="npu", dtype=torch.int32)
move_intermediate_cache_kda(state, snapshots, slot, slot, step, impl="triton")
```
