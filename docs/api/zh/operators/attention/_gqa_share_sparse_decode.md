# gqa_share_sparse_decode

## 硬件支持

| Backend | arch22 | arch35 |
| --- | --- | --- |
| `triton` | 未验证 | 未验证 |
| `torch_native` | 未验证 | 未验证 |

## Signature

```python
from sgl_kernel_npu.operators.attention import gqa_share_sparse_decode
def gqa_share_sparse_decode(q, sink, k_cache_bnsd, v_cache_bnsd, block_table, seq_lens, block_size, topk_idx, sm_scale=None, num_topk_chunks=None, max_num_topk_chunks=8, req_to_token=None, req_pool_indices=None, max_num_blocks=None, num_pages=None, sanitize_page_ids=False, topk_blocks_per_step=1, prefetch_idx=False, num_warps=None, num_stages=None, *, impl="auto"): ...
```

`auto` selects `triton`.

## Description

在选中的分页 KV 块上计算 decode 注意力；可选 sink 提供一个值为零的 softmax 项。

## 参数与返回值

q: [B,QH,D]; K/V: [pages,block_size,KVH,D]; topk_idx: [KVH or QH,B,topk]; seq_lens: [B]; block_table: [B,max_blocks], or direct lookup via req_to_token and req_pool_indices. Returns [B,QH,D].

`impl` is `auto`, `triton`, or `torch_native`. The independent torch reference may synchronize or transfer sequence metadata to the host and is intended for correctness comparison, not performance.

## Triton

Migrated from the legacy implementation; supported shapes, dtypes, layouts and numerical behavior need hardware validation.

## torch_native

Independent PyTorch reference for numerical comparison.

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
