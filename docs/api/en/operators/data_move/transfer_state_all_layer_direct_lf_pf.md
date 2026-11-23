# transfer_state_all_layer_direct_lf_pf

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `ascendc_aiv` | Supported | Supported |
| `aclrt` | Supported | Supported |
| `torch_native` | Supported | Supported |

## Signature

```python
from sgl_kernel_npu.operators.data_move import TransferFlag, transfer_state_all_layer_direct_lf_pf

def transfer_state_all_layer_direct_lf_pf(device_states, host_states, device_indices, host_indices, flags=TransferFlag.FAST2D, *, impl='auto'): ...
```

`auto` selects `ascendc_aiv`; `aclrt` forwards to the pre-existing ACL runtime op `torch.ops.npu.transfer_state_all_layer_direct_lf_pf` with an identical signature.

## Description

Back up every layer of the selected Mamba/SSM state slots from the NPU to host memory (D2H only),
using paired device/host slot indices. Device components are layer-first `[layers, slots, *state_shape]`;
host components are page-first `[slots, layers, 1, *state_shape]`. Each index pair transfers every
layer of one slot. Up to 16 components may be submitted in one call with independent layer counts
(e.g. a main `[L, slots, ...]` component plus `[1, slots, ...]` draft components), and host
components may be layer-narrowed views of one registered pool. Unselected slots remain unchanged.

## Parameters and return value

| Parameter | Type | Shape | Dtype | Default | Description |
| --- | --- | --- | --- | --- | --- |
| `device_states` | Tensor list | [L,D,...] each | Matching payload dtype | Required | Layer-first device state components; layer counts may differ, slot payload trailing dims must be flat-packed (contiguous), permuted in-slot views are rejected. |
| `host_states` | Tensor list | [H,L,1,...] each | Matching payload dtype | Required | Host state components paired one-to-one with the device components; may be layer-narrowed views of one registered pool. |
| `device_indices` | Tensor | [N] | int64 | Required | Non-negative device slot indices valid for every component. |
| `host_indices` | Tensor | [N] | int64 | Required | Corresponding non-negative host slot indices; destinations must be unique. |
| `flags` | TransferFlag | — | — | TransferFlag.FAST2D | 2D copy flag FAST2D. |
| `impl` | str | — | — | 'auto' | auto, aclrt, ascendc_aiv, or torch_native. |

| Return value | Type | Description |
| --- | --- | --- |
| `None` | NoneType | Updates the destination in place. Synchronize the submitting stream before reading it or reusing the source. |

## aclrt backend

Forwards to the pre-existing ACL runtime op `torch.ops.npu.transfer_state_all_layer_direct_lf_pf` (original implementation, identical signature); the host buffers must be pinned but need no DVA registration.

## ascendc_aiv backend

Submits an asynchronous AIV copy to the current NPU stream. Host memory must be registered through
`register_acl_host_memory` (aclrt backend) or a MemFabric pool; the registration table is the only
DVA source and unregistered memory fails fast. Each (slot, layer) row moves through 88KB UB double
buffering, and wider rows are split automatically. Slot indices stay in device memory and the kernel
derives every row address from the layout metadata, so indices never round-trip through the CPU.
The component count is capped at 16 by the static tiling metadata.

## torch_native backend

PyTorch reference: `index_select` gathers the selected device slots, `movedim` moves the layer
dimension after the slot dimension, and each component writes back through its layer-narrowed host
views; the host buffers must be pinned. Does not call the `ascendc_aiv` implementation.

## Example

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.data_move import transfer_state_all_layer_direct_lf_pf
from sgl_kernel_npu.operators.mem_cache.memfabric.host_memory import register_acl_host_memory

dev = torch.randn(4, 8, 128, device="npu")
host = torch.zeros(8, 4, 1, 128)
register_acl_host_memory(host)
ids = torch.arange(8, dtype=torch.int64, device="npu")
transfer_state_all_layer_direct_lf_pf([dev], [host], ids, ids)
torch.npu.synchronize()
print(host[0, :, 0])
```
