# zero_experts_compute_identity

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Supported | Supported |
| `torch_native` | Supported | Supported |

## Signature

`auto` selects `triton`.

```python
from sgl_kernel_npu.operators.moe import zero_experts_compute_identity


def zero_experts_compute_identity(
    expert_indices,
    expert_scales,
    num_experts,
    zero_expert_type,
    hidden_states,
    identity_mask_value=0,
    *,
    impl="auto"
): ...
```

## Description

Computes the weighted identity-expert contribution for LongCat MoE. Routes with expert ID >= num_experts contribute their weights times hidden_states, then their scales are zeroed and IDs replaced. If every route is identity, the first replacement ID is zero. Both routing tensors are modified in place.

## Parameters and return value

All tensors must be on the same NPU device.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `expert_indices` | Tensor | `Required` | Writable contiguous integer [tokens,topk] routes. |
| `expert_scales` | Tensor | `Required` | Writable contiguous floating [tokens,topk] route weights. |
| `num_experts` | int | `Required` | IDs at or above this count are identity experts. |
| `zero_expert_type` | str | `Required` | Use identity; other strings do not select other computations. |
| `hidden_states` | Tensor | `Required` | Floating contiguous [tokens,hidden] states. |
| `identity_mask_value` | int | `0` | Replacement route ID for identity experts. |
| `impl` | str | `'auto'` | auto selects triton; explicit backend selection. |

| Return | Type | Description |
| --- | --- | --- |
| result | Tensor | [tokens,hidden] weighted identity output with hidden_states dtype; modifies routing IDs and scales. |

## triton

Uses Triton-Ascend kernels.

## torch_native

Uses PyTorch tensor operations on the input device.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.moe import zero_experts_compute_identity

indices = torch.tensor([[0, 16]], device="npu", dtype=torch.int32)
scales = torch.tensor([[0.75, 0.25]], device="npu")
x = torch.randn(1, 128, device="npu", dtype=torch.bfloat16)
y = zero_experts_compute_identity(indices, scales, 16, "identity", x, impl="triton")
```
