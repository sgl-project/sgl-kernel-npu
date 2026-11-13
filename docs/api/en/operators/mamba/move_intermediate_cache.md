# move_intermediate_cache

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Supported | Supported |
| `torch_native` | Supported | Supported |

## Signature

`auto` selects `triton`.

```python
from sgl_kernel_npu.operators.mamba import move_intermediate_cache


def move_intermediate_cache(
    ssm_states,
    intermediate_state_cache,
    dst_indices_tensor,
    src_indices_tensor,
    last_steps_tensor,
    h_block_size=2,
    *,
    impl="auto"
): ...
```

## Description

Copies each selected speculative snapshot into the persistent SSM cache across all layers. Requests with a negative last step leave their destination untouched. Within each head this interface copies physical contiguous V*K storage; use move_intermediate_cache_kda for a logically transposed destination.

## Parameters and return value

All tensors must be on the same NPU device.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `ssm_states` | Tensor | `Required` | Writable floating [layers,pool,H,V,K] persistent state. |
| `intermediate_state_cache` | Tensor | `Required` | Floating [layers,pool,steps,H,V,K], contiguous inside H/V/K. |
| `dst_indices_tensor` | Tensor | `Required` | Integer [requests] destination slots, distinct for concurrent requests. |
| `src_indices_tensor` | Tensor | `Required` | Integer [requests] source slots. |
| `last_steps_tensor` | Tensor | `Required` | Integer [requests] snapshot step; negative skips the request. |
| `h_block_size` | int | `2` | Head tiling size; does not change mathematical results. |
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

from sgl_kernel_npu.operators.mamba import move_intermediate_cache

state = torch.zeros(2, 4, 2, 128, 128, device="npu", dtype=torch.bfloat16)
snapshots = torch.randn(2, 4, 3, 2, 128, 128, device="npu", dtype=state.dtype)
slot = torch.tensor([1], device="npu", dtype=torch.int32)
step = torch.tensor([2], device="npu", dtype=torch.int32)
move_intermediate_cache(state, snapshots, slot, slot, step, impl="triton")
```
