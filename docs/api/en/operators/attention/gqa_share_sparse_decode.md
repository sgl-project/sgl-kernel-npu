# gqa_share_sparse_decode

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Supported | Supported |
| `torch_native` | Supported | Supported |

## Signature

```python
import torch
from sgl_kernel_npu.operators.attention import gqa_share_sparse_decode

def gqa_share_sparse_decode(q, sink, k_cache_bnsd, v_cache_bnsd, block_table, seq_lens, block_size, topk_idx, sm_scale=None, num_topk_chunks=None, max_num_topk_chunks=8, req_to_token=None, req_pool_indices=None, max_num_blocks=None, num_pages=None, sanitize_page_ids=False, topk_blocks_per_step=1, prefetch_idx=False, num_warps=None, num_stages=None, *, impl='auto'):
    ...
```

`auto` aliases `triton`; it does not automatically fall back to the reference.

## Description

Gather selected KV pages and compute $softmax(QK^\mathsf{T}\,sm\_scale)V$, masking tokens beyond seq_lens. The optional sink adds a logit with zero value.

## Parameters and return values

Triton tensor inputs should share one NPU and use contiguous layouts; outputs reside on the input device.

| Parameter | Type | Shape / dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- |
| `q` | Tensor | `[B,QH,D]; float16/bfloat16` | Required | Decode queries. |
| `sink` | Tensor or None | `[QH,D]; q.dtype` | Required | Extra softmax key with zero value. |
| `k_cache_bnsd` | Tensor | `[pages,block_size,KVH,D]; q.dtype` | Required | Paged keys. |
| `v_cache_bnsd` | Tensor | `[pages,block_size,KVH,D]; q.dtype` | Required | Paged values with the same shape as keys. |
| `block_table` | Tensor or None | `[B,max_blocks]; int32/int64` | Required | Logical-to-physical pages; None in direct lookup mode. |
| `seq_lens` | Tensor | `[T] or [B]; int32` | Required | Causal KV length per query/request. |
| `block_size` | int | `—` | Required | Number of tokens per KV page; positive. |
| `topk_idx` | Tensor | `[KVH or QH,B,S]; int32` | Required | Selected logical blocks with -1 padding; per-QH input uses the first head of each group. |
| `sm_scale` | float or None | `—` | `None` | Attention scale; None selects D**-0.5. |
| `num_topk_chunks` | int or None | `—` | `None` | Power-of-two split count, no greater than max(1,S); None selects automatically. |
| `max_num_topk_chunks` | int | `—` | `8` | Upper bound for automatic split selection. |
| `req_to_token` | Tensor or None | `[requests,max_ctx]; int32/int64` | `None` | Token-slot mapping; direct lookup requires req_pool_indices. |
| `req_pool_indices` | Tensor or None | `[B]; int32/int64` | `None` | Request IDs for direct lookup. |
| `max_num_blocks` | int or None | `—` | `None` | Positive logical capacity, required for direct lookup. |
| `num_pages` | int or None | `—` | `None` | Positive physical page count, required for direct lookup. |
| `sanitize_page_ids` | bool | `—` | `False` | Clamp direct-lookup page IDs to the physical range. |
| `topk_blocks_per_step` | int | `—` | `1` | Power-of-two number of selected blocks per tile. |
| `prefetch_idx` | bool | `—` | `False` | Prefetch indices when processing one block per step. |
| `num_warps` | int or None | `—` | `None` | Triton launch setting; None selects the backend default. |
| `num_stages` | int or None | `—` | `None` | Triton pipeline setting; None selects the backend default. |
| `impl` | str | `—` | `'auto'` | Keyword-only: auto, triton or torch_native. |

| Return value | Type | Shape / dtype | Description |
| --- | --- | --- | --- |
| `output` | Tensor or None | `q.shape; q.dtype` | New sparse-attention output; inputs are not modified. |

## triton backend

Requires matching FP16/BF16 q/K/V on one NPU, QH divisible by KVH, and valid page mappings. The public Triton entry disables the optional native attention path.

## torch_native backend

Independent PyTorch operations provide the reference result. Host metadata transfers may synchronize the device; this backend is intended for accuracy comparison.

## Example

```python
import torch
from sgl_kernel_npu.operators.attention import gqa_share_sparse_decode

q = torch.randn(1, 4, 64, device="npu", dtype=torch.float16)
k = torch.randn(4, 16, 2, 64, device="npu", dtype=torch.float16)
v = torch.randn_like(k)
table = torch.tensor([[0, 1, 2, 3]], device="npu", dtype=torch.int32)
lengths = torch.tensor([32], device="npu", dtype=torch.int32)
topk = torch.tensor([[[0, 1]], [[0, 1]]], device="npu", dtype=torch.int32)
y = gqa_share_sparse_decode(q, None, k, v, table, lengths, 16, topk, impl="triton")
```
