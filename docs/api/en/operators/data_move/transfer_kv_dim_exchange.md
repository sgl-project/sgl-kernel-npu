# transfer_kv_dim_exchange

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `aclrt` | Supported | Supported |
| `triton` | Not implemented | Not implemented |
| `tilelang` | Not implemented | Not implemented |
| `torch_native` | Supported | Supported |

## Signature

```python
from sgl_kernel_npu.operators.data_move import TransferDirection, TransferFlag, transfer_kv_dim_exchange

def transfer_kv_dim_exchange(device_indices, host_indices, device_k, host_k, device_v, host_v, device_index_k=None, host_index_k=None, device_index_k_scale=None, host_index_k_scale=None, page_size=128, direction=TransferDirection.H2D, flags=TransferFlag.FAST2D, *, impl='auto'): ...
```

`auto` selects `aclrt`.

## Description

Copy selected KV pages across layer-first and page-first layouts.

## Parameters and return value

| Parameter | Type | Shape | Dtype | Default | Description |
| --- | --- | --- | --- | --- | --- |
| `device_indices` | Tensor | [N] | int64 | Required | Device slot indices (token indices for KV). |
| `host_indices` | Tensor | [N] | int64 | Required | Matching host slot or token indices. |
| `device_k` | Tensor | [L,D,P,A,C] | Matching payload dtype | Required | Device K [layers, pages, page_size, heads, dim]. |
| `host_k` | Tensor | [H,L,P,A,C] | Matching payload dtype | Required | Pinned host K [pages, layers, page_size, heads, dim]. |
| `device_v` | Tensor | [L,D,P,A,C] | Matching payload dtype | Required | Device V, or an empty tensor. |
| `host_v` | Tensor | [H,L,P,A,C] | Matching payload dtype | Required | Host V, or an empty tensor. |
| `device_index_k` | Tensor | [L,D,P,A,C] | Matching payload dtype | None | Optional device indexer key cache. |
| `host_index_k` | Tensor | [H,L,P,A,C] | Matching payload dtype | None | Matching optional host indexer key cache. |
| `device_index_k_scale` | Tensor | [L,D,P,1] / [L,D,P,1,1] | FP32 | None | Optional per-token device indexer scale (4D or 5D). |
| `host_index_k_scale` | Tensor | [H,L,P,1,1] | FP32 | None | Matching optional 5D host scale. |
| `page_size` | int | — | — | 128 | Tokens per KV page. |
| `direction` | TransferDirection | — | — | TransferDirection.H2D | TransferDirection.H2D or TransferDirection.D2H. |
| `flags` | TransferFlag | — | — | TransferFlag.FAST2D | TransferFlag.FAST2D for 2D copies. |
| `impl` | str | — | — | 'auto' | auto (ACL Runtime), aclrt, or torch_native. |

| Return value | Type | Description |
| --- | --- | --- |
| `None` | NoneType | Updates the destination in place. Synchronize the submitting stream before reading it or reusing the source. |

## aclrt backend

Submits asynchronous 2D transfers to the current NPU stream.

## torch_native backend

Uses PyTorch operations on the input devices.

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
