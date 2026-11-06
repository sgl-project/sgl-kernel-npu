# transfer_kv_dim_exchange

HiCache uses this helper to copy selected KV pages between page-first host storage and layer-first NPU storage. The public import is `sgl_kernel_npu.kvcacheio.transfer_kv_dim_exchange`. The copy is enqueued on the current NPU stream. The original K-only PR and the queued-copy prototype were measured separately on Ascend 910B4 (A2) with CANN 9.1.0. The latest integration of the prototype has not been rerun on hardware.

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

With `flags=None`, the default is `FAST2D`. Setting `SGL_KERNEL_NPU_COALESCE_KV_H2D=1` before import selects `FAST2D_COALESCED_H2D` for H2D. For CPU int64 indices, the wrapper submits K, V, index and scale components in one queued native call. Each nonempty component is five-dimensional (four-dimensional components are padded with a final singleton dimension). The page payload must be contiguous; outer layer/page strides may contain gaps, including layer slices of packed host pools. Each host/device pair must have matching dtype and payload dimensions; different pairs can have different dtypes, widths and layer counts.

Both host and device page IDs must advance together before descriptors are combined. Repeated destinations retain input order. The planner uses actual byte strides and checks page alignment, complete token pages, bounds and the supported copy pitches before submitting. Unsupported payload strides are rejected by the queued path, not silently repacked.

The native call uses Torch-NPU `OpCommand::RunOpApi` and the current stream without draining the host task queue. It captures tensor owners until submission; the Python helper additionally holds the component views and indices until a completion event retires on a subsequent call from the same thread. Callers must keep the submitting thread alive until copies complete and must not modify source buffers before completion. An event protects lifetime, not concurrent writes.

Explicit `FAST2D` and D2H retain the existing implementation. Calls with non-CPU indices or unsupported tensor ranks retain the original #947 path. No private GLM environment variable, dynamic extension loader or SGLang monkey patch is needed.

## Migration and validation status

The queued implementation is ported from the previously measured GLM-5.2 extension (`glm52_cache_copy.cpp`, SHA256 `e19368569eb484d605b57920d57ce677c46814ceb3e82fcc2e2503d7d72a6960`). Its measured binary SHA256 was `c0bb68872171ccd5f41b4d9719f846feb6d002670138ff89d0a89c8c88664f71`.

Historical prototype tests passed 21 combinations in each of task-queue modes 0 and 2, five-component layer-slice byte checks, and six-generation lifetime/stream-order checks. Two diagnostic model replays completed 36/36 requests each. These are historical prototype results, not validation or a performance claim for the latest PR integration. Native registration, wrapper routing, device selection for completion events, and acceptance of repeated destinations are integration adaptations; no new NPU experiment was run for this migration.

The original #947 page-order/scale test remains applicable. The prototype's layer-slice, lifetime and invalid-index cases are carried into `tests/python/sgl_kernel_npu/test_queued_h2d.py` for subsequent hardware validation. Run that file separately with `TASK_QUEUE_ENABLE=0` and `TASK_QUEUE_ENABLE=2`.

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
