# transfer_state_per_layer_direct_pf_lf

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `ascendc_aiv` | Supported | Supported |
| `torch_native` | Supported | Supported |

## Signature

```python
from sgl_kernel_npu.operators.data_move import TransferFlag, transfer_state_per_layer_direct_pf_lf

def transfer_state_per_layer_direct_pf_lf(src, dst, src_indices, dst_indices, layer_id, flags=TransferFlag.FAST2D, *, impl='auto'): ...
```

`auto` selects `ascendc_aiv`.

## Description

Restore one host layer of the selected Mamba/SSM state slots into a device layer view (H2D only),
using paired host/device slot indices. The host state is page-first `[slots, layers, 1, *state_shape]`
and the destination device layer view is `[slots, *state_shape]`; `layer_id` selects the host layer
to restore, and each index pair transfers exactly that layer. Unselected slots remain unchanged.

## Parameters and return value

| Parameter | Type | Shape | Dtype | Default | Description |
| --- | --- | --- | --- | --- | --- |
| `src` | Tensor | [H,L,1,...] | Matching payload dtype | Required | Host state [slots, layers, 1, *state_shape]. |
| `dst` | Tensor | [D,...] | Matching payload dtype | Required | Destination device layer view `[slots, *state_shape]`, typically one layer slice of the device pool; trailing dims must be flat-packed (contiguous). |
| `src_indices` | Tensor | [N] | int64 | Required | Host slot indices in copy order. |
| `dst_indices` | Tensor | [N] | int64 | Required | Corresponding device slot indices. |
| `layer_id` | int | — | — | Required | Zero-based source layer; must be smaller than the `src` layer count. |
| `flags` | TransferFlag | — | — | TransferFlag.FAST2D | 2D copy flag FAST2D. |
| `impl` | str | — | — | 'auto' | auto, ascendc_aiv, or torch_native. |

| Return value | Type | Description |
| --- | --- | --- |
| `None` | NoneType | Updates the destination in place. Synchronize the submitting stream before reading it or reusing the source. |

## ascendc_aiv backend

Submits an asynchronous AIV copy to the current NPU stream. Internally the transfer is normalized
into a single-component state_exchange submission: the host side takes the layer-narrowed view of
`src` at `layer_id`, the device side unsqueezes `dst` to `[1, slots, ...]`, and the offsets ride on
the base pointers and strides. Host memory must be registered through `register_acl_host_memory`
(aclrt backend) or a MemFabric pool; the registration table is the only DVA source and unregistered
memory fails fast.

## torch_native backend

PyTorch reference: reuses SGLang's `MambaPoolHost` per-layer restore path; the host buffer must be
pinned. Does not call the `ascendc_aiv` implementation.

## Example

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.data_move import transfer_state_per_layer_direct_pf_lf
from sgl_kernel_npu.operators.mem_cache.memfabric.host_memory import register_acl_host_memory

dev = torch.zeros(8, 128, device="npu")
host = torch.randn(8, 4, 1, 128)
register_acl_host_memory(host)
h, d = torch.arange(8, dtype=torch.int64, device="npu"), torch.arange(8, dtype=torch.int64, device="npu")
transfer_state_per_layer_direct_pf_lf(host, dev, h, d, layer_id=2)
torch.npu.synchronize()
print(dev)
```
