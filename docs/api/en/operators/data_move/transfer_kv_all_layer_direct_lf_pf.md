# transfer_kv_all_layer_direct_lf_pf

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `aclrt` | Supported | Supported |
| `ascendc_aiv` | Supported | Supported |
| `torch_native` | Supported | Supported |

## Signature

```python
from sgl_kernel_npu.operators.data_move import TransferFlag, transfer_kv_all_layer_direct_lf_pf

def transfer_kv_all_layer_direct_lf_pf(device_states, host_states, device_indices, host_indices, flags=TransferFlag.FAST2D, *, impl='auto'): ...
```

`auto` selects `aclrt`.

## Description

Back up every layer of selected NPU state slots to pinned host memory (D2H only). Each device
component uses `[layers, slots, *state_shape]`; its host counterpart uses
`[slots, layers, 1,
*state_shape]`. Each paired device/host slot index transfers all layers of
every component. Components may have different shapes and dtypes, such as Mamba temporal and
convolution states. Unselected host slots remain unchanged.

## Parameters and return value

| Parameter | Type | Shape | Dtype | Default | Description |
| --- | --- | --- | --- | --- | --- |
| `device_states` | Sequence[Tensor] | [L,D,...] | Matching payload dtype | Required | Layer-first device components [layers, slots, ...]; each slot payload must be physically dense. |
| `host_states` | Sequence[Tensor] | [H,L,1,...] | Matching payload dtype | Required | Corresponding contiguous, pinned host components [slots, layers, 1, ...]. |
| `device_indices` | Tensor | [N] | int64 | Required | Non-negative device slot indices within every component's slot count; no page-size conversion. |
| `host_indices` | Tensor | [N] | int64 | Required | Corresponding non-negative host slot indices within every component's slot count. |
| `flags` | TransferFlag | — | — | TransferFlag.FAST2D | TransferFlag.FAST2D for 2D copies. |
| `impl` | str | — | — | 'auto' | auto (ACL Runtime), aclrt, ascendc_aiv, or torch_native. |

| Return value | Type | Description |
| --- | --- | --- |
| `None` | NoneType | Updates the destination in place. Synchronize the submitting stream before reading it or reusing the source. |

## aclrt backend

Submits asynchronous 2D transfers to the current NPU stream. Host components must be
contiguous and pinned. Device slot payloads must be physically dense, with the slot
pitch equal to the payload size. Device layers may have padding, provided the layer
pitch is at least the number of device slots times the slot pitch. Each device/host
component pair must have matching dtypes and payload shapes. All components and slot
indices are validated before any copy; ACL submission errors are propagated.

## torch_native backend

Adapts the all-layer Torch fallback from SGLang's `MambaPoolHost`: gather device slots with
`index_select`, move the layer dimension after the slot dimension, and write to the host.
Components are handled separately; physical storage views preserve dense transposed payloads.
Repeated destination indices are overwritten in input order.

## ascendc_aiv backend

Submits one fused `state_exchange` AIV kernel per call instead of one ACL copy per component. Host components must be registered in the DVA table (register_host_memory / register_acl_host_memory) before the call; destination slot indices must be unique across the submission, and NPU graph capture is not supported. Components may carry independent layer counts (main plus draft pools), and host components may be layer-narrowed views of one registered pool.

Build the CANN operator with `python3 setup.py --arch-version 2201 --ops state_exchange` for A3 or use `--arch-version 3510` for A5.

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
