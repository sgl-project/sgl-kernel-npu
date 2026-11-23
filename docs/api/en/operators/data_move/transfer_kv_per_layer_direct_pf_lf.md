# transfer_kv_per_layer_direct_pf_lf

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `aclrt` | Supported | Supported |
| `ascendc_aiv` | Supported | Supported |
| `torch_native` | Supported | Supported |

## Signature

```python
from sgl_kernel_npu.operators.data_move import TransferFlag, transfer_kv_per_layer_direct_pf_lf

def transfer_kv_per_layer_direct_pf_lf(src, dst, src_indices, dst_indices, layer_id, flags=TransferFlag.FAST2D, *, impl='auto'): ...
```

`auto` selects `aclrt`.

## Description

Restore one selected layer from pinned host memory to the NPU (H2D only). The source layout is
`[slots, layers, 1, *state_shape]`, and the destination is a single device-layer view
`[slots,
*state_shape]`. Each `src_indices[i] → dst_indices[i]` pair copies the source at
`layer_id`; indices may be nonconsecutive and host source slots may repeat. Only selected
destination slots are updated. This also applies to other state tensors with the same layout.

## Parameters and return value

| Parameter | Type | Shape | Dtype | Default | Description |
| --- | --- | --- | --- | --- | --- |
| `src` | Tensor | [H,L,1,...] | Matching payload dtype | Required | Contiguous, pinned host state [slots, layers, 1, ...]. |
| `dst` | Tensor | [D,...] | Matching payload dtype | Required | Destination NPU layer view [slots, ...]; each slot payload must be physically dense. |
| `src_indices` | Tensor | [N] | int64 | Required | Host slot indices in copy order. |
| `dst_indices` | Tensor | [N] | int64 | Required | Corresponding device slots. |
| `layer_id` | int | — | — | Required | Zero-based source layer. |
| `flags` | TransferFlag | — | — | TransferFlag.FAST2D | TransferFlag.FAST2D for 2D copies. |
| `impl` | str | — | — | 'auto' | auto (ACL Runtime), aclrt, ascendc_aiv, or torch_native. |

| Return value | Type | Description |
| --- | --- | --- |
| `None` | NoneType | Updates the destination in place. Synchronize the submitting stream before reading it or reusing the source. |

## aclrt backend

Submits asynchronous 2D transfers to the current NPU stream. The host source must be
contiguous and pinned. Destination slot payloads must be physically dense, with the
slot pitch equal to the payload size. Source and destination dtypes and payload
shapes must match. Slot indices must be non-negative and within their respective
buffer's slot count; `layer_id` must be within the host layer count. Inputs are
validated before any copy, and ACL submission errors are propagated.

## torch_native backend

Adapts the per-layer Torch fallback from SGLang's `MambaPoolHost`, mapping host slots to device
slots with `index_select/index_copy_`. Dense payloads are flattened in physical storage order
so transposed states match native byte copies. Repeated destination indices are overwritten in
input order.

## ascendc_aiv backend

Submits one fused `state_exchange` AIV kernel per call. The host source must be registered in the DVA table (register_host_memory / register_acl_host_memory); destination slot indices must be unique, and NPU graph capture is not supported. Layer-narrowed host views of one registered pool are accepted.

Build the CANN operator with `python3 setup.py --arch-version 2201 --ops state_exchange` for A3 or use `--arch-version 3510` for A5.

## Example

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.data_move import transfer_kv_per_layer_direct_pf_lf

host = torch.ones(4, 2, 1, 8, pin_memory=True)
layer = torch.zeros(4, 8, device="npu")
slots = torch.tensor([1, 3])
transfer_kv_per_layer_direct_pf_lf(host, layer, slots, slots, layer_id=1)
torch.npu.synchronize()
print(layer[1])
```
