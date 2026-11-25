# transfer_kv_dim_exchange

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `aclrt` | Supported | Supported |
| `torch_native` | Supported | Supported |

## Signature

```python
from sgl_kernel_npu.operators.data_move import TransferDirection, TransferFlag, transfer_kv_dim_exchange

def transfer_kv_dim_exchange(device_indices, host_indices, device_k, host_k, device_v, host_v, device_index_k=None, host_index_k=None, device_index_k_scale=None, host_index_k_scale=None, page_size=128, direction=TransferDirection.H2D, flags=TransferFlag.FAST2D, *, impl='auto'): ...
```

`auto` selects `aclrt`.

## Description

Transfer selected KV-cache pages between the NPU and pinned host memory for H2D restore or D2H
backup. The device layout is `[layers, pages, tokens_per_page, heads, dim]`, while the host
layout is `[pages, layers, tokens_per_page, heads, dim]`. Paired token indices identify source
and destination pages; each transfer includes every layer of a page. K-only transfers and
optional V, indexer keys and scales are supported. Unselected pages remain unchanged.

## Parameters and return value

| Parameter | Type | Shape | Dtype | Default | Description |
| --- | --- | --- | --- | --- | --- |
| `device_indices` | Tensor | [N] | int64 | Required | Device slot indices (token indices for KV). |
| `host_indices` | Tensor | [N] | int64 | Required | Matching host slot or token indices. |
| `device_k` | Tensor | [L,D,P,A,C] | Matching payload dtype | Required | Device K [layers, pages, page_size, heads, dim]. |
| `host_k` | Tensor | [H,L,P,A,C] | Matching payload dtype | Required | Pinned host K [pages, layers, page_size, heads, dim]. |
| `device_v` | Tensor | [L,D,P,A,C] | Matching payload dtype | Required | Device V, or an empty tensor. When copied, layers, pages, page size, heads and dtype must match K; head dimension may differ. |
| `host_v` | Tensor | [H,L,P,A,C] | Matching payload dtype | Required | Host V, or an empty tensor. |
| `device_index_k` | Tensor | [L,D,P,A,C] | Matching payload dtype | None | Optional device indexer key cache. |
| `host_index_k` | Tensor | [H,L,P,A,C] | Matching payload dtype | None | Matching optional host indexer key cache. |
| `device_index_k_scale` | Tensor | [L,D,P,1] / [L,D,P,1,1] | FP32 | None | Optional per-token device indexer scale (4D or 5D). |
| `host_index_k_scale` | Tensor | [H,L,P,1,1] | FP32 | None | Matching optional 5D host scale. |
| `page_size` | int | — | — | 128 | Tokens per KV page; the index count must be divisible by this value. Only the first index of each group selects a page; other entries are ignored. |
| `direction` | TransferDirection | — | — | TransferDirection.H2D | TransferDirection.H2D or TransferDirection.D2H. |
| `flags` | TransferFlag | — | — | TransferFlag.FAST2D | TransferFlag.FAST2D for 2D copies. |
| `impl` | str | — | — | 'auto' | auto (ACL Runtime), aclrt, or torch_native. |

| Return value | Type | Description |
| --- | --- | --- |
| `None` | NoneType | Updates the destination in place. Synchronize the submitting stream before reading it or reusing the source. |

## aclrt backend

Submits asynchronous 2D transfers to the current NPU stream. Device/host pairs must have matching dtypes and page shapes, with pinned contiguous host storage. Device page payloads may be densely transposed, but pages and layers must retain packed pitches. All selected page indices are validated before any copy; ACL submission errors are propagated.

## torch_native backend

Adapts the PyTorch gather/scatter reference from SGLang's `MambaPoolHost`. The first token
index in each `page_size` group determines a page ID; a physical page is treated as one slot for
per-layer restore or all-layer backup. K, V and optional indexer/scale buffers are handled
separately without calling the native `aclrt` operator.

## Example

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.data_move import transfer_kv_dim_exchange

dev = torch.zeros(2, 4, 128, 1, 8, device="npu")
host = torch.ones(4, 2, 128, 1, 8, pin_memory=True)
tokens = torch.arange(128)
transfer_kv_dim_exchange(tokens, tokens, dev, host, torch.empty(0), torch.empty(0))
torch.npu.synchronize()
print(dev[:, 0, 0])
```
