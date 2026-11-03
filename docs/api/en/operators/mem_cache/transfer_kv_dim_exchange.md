# transfer_kv_dim_exchange

HiCache uses this helper to copy selected KV pages between page-first host storage and layer-first NPU storage. The public import is `sgl_kernel_npu.kvcacheio.transfer_kv_dim_exchange`. The copy is enqueued on the current NPU stream. Tested on Ascend 910B4 (A2) with CANN 9.1.0.

## Signature

```python
transfer_kv_dim_exchange(
    device_indices, host_indices, device_k, host_k, device_v, host_v,
    device_index_k=None, host_index_k=None, device_index_k_scale=None,
    host_index_k_scale=None, page_size=128, direction=TransferDirection.H2D,
    flags=None,
)
```

## Parameters

| Parameter | Type and constraints |
| --- | --- |
| `device_indices`, `host_indices` | Flattened int64 token indices. Supply complete pages of the same size in transfer order on each side. |
| `device_k`, `host_k` | Matching K tensors with device layout `[layers, device_pages, page_size, heads, width]` and pinned host layout `[host_pages, layers, page_size, heads, width]`. Use the same payload dtype. |
| `device_v`, `host_v` | Corresponding V tensors in the same layouts, or two empty tensors for K-only data. |
| `device_index_k`, `host_index_k`, `device_index_k_scale`, `host_index_k_scale` | Optional index and scale tensor pairs. The wrapper pads a four-dimensional device scale with a final singleton dimension for the exchange. |
| `page_size`, `direction` | Positive tokens per page, default 128. Direction defaults to `TransferDirection.H2D`; `TransferDirection.D2H` copies to the host. |
| `flags` | An explicit `TransferFlag` takes precedence over the environment. |

## Return value

Returns `None` and writes the selected destination pages in place. Synchronize the stream before reading the result from the host.

## Ascend native backend

With `flags=None`, the default is `FAST2D`. Setting `SGL_KERNEL_NPU_COALESCE_KV_H2D=1` before import selects `FAST2D_COALESCED_H2D` for H2D. The native implementation can group dense K-only runs when both host and device page IDs advance together. Explicit `FAST2D`, K/V pairs and D2H keep page-wise copies.

## Example

```python
import torch
from sgl_kernel_npu.kvcacheio import TransferDirection, TransferFlag, transfer_kv_dim_exchange

page_size = 128
indices = torch.arange(2 * page_size, dtype=torch.int64)
host_k = torch.ones((4, 1, page_size, 1, 4), dtype=torch.int8).pin_memory()
device_k = torch.zeros((1, 4, page_size, 1, 4), dtype=torch.int8, device="npu")
no_v = torch.empty((0, 0, 0, 0, 0))
transfer_kv_dim_exchange(
    indices, indices, device_k, host_k, no_v, no_v,
    page_size=page_size, direction=TransferDirection.H2D,
    flags=TransferFlag.FAST2D_COALESCED_H2D,
)
torch.npu.synchronize()
assert torch.equal(device_k[:, :2].cpu(), host_k[:2].transpose(0, 1))
```
