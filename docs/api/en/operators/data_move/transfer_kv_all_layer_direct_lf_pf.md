# transfer_kv_all_layer_direct_lf_pf

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `aclrt` | Supported | Supported |
| `torch_native` | Supported | Supported |

## Signature

```python
from sgl_kernel_npu.operators.data_move import TransferFlag, transfer_kv_all_layer_direct_lf_pf

def transfer_kv_all_layer_direct_lf_pf(device_states, host_states, device_indices, host_indices, flags=TransferFlag.FAST2D, *, impl='auto'): ...
```

`auto` selects `aclrt`.

## Description

Back up every layer of selected NPU state slots to pinned host memory (D2H only). Each device
component uses `[layers, slots, *state_shape]` ; its host counterpart uses
`[slots, layers, 1,
*state_shape]` . Each paired device/host slot index transfers all layers of
every component. Components may have different shapes and dtypes, such as Mamba temporal and
convolution states. Unselected host slots remain unchanged.

## Parameters and return value

| Parameter | Type | Shape | Dtype | Default | Description |
| --- | --- | --- | --- | --- | --- |
| `device_states` | Sequence[Tensor] | [L,D,...] | Matching payload dtype | Required | Layer-first device components [layers, slots, ...]. |
| `host_states` | Sequence[Tensor] | [H,L,1,...] | Matching payload dtype | Required | Corresponding pinned host components [slots, layers, 1, ...]. |
| `device_indices` | Tensor | [N] | int64 | Required | Device slot indices (token indices for KV). |
| `host_indices` | Tensor | [N] | int64 | Required | Matching host slot or token indices. |
| `flags` | TransferFlag | — | — | TransferFlag.FAST2D | TransferFlag.FAST2D for 2D copies. |
| `impl` | str | — | — | 'auto' | auto (ACL Runtime), aclrt, or torch_native. |

| Return value | Type | Description |
| --- | --- | --- |
| `None` | NoneType | Updates the destination in place. Synchronize the submitting stream before reading it or reusing the source. |

## aclrt backend

Submits asynchronous 2D transfers to the current NPU stream.

## torch_native backend

An independent PyTorch reference iterates over components, slot pairs and layers, copying
`dev[layer, d] → host[h, layer, 0]` . Dense payloads are flattened in physical storage order
with `as_strided` and transferred with `copy_(..., non_blocking=True)` , without calling the
native operator.

## Example

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.data_move import transfer_kv_all_layer_direct_lf_pf

dev = torch.ones(2, 4, 8, device="npu")
host = torch.zeros(4, 2, 1, 8, pin_memory=True)
slots = torch.tensor([1, 3])
transfer_kv_all_layer_direct_lf_pf([dev], [host], slots, slots)
torch.npu.synchronize()
print(host[1, :, 0])
```
