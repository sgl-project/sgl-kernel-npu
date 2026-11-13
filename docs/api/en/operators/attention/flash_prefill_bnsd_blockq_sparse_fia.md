# flash_prefill_bnsd_blockq_sparse_fia

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Supported | Supported |
| `torch_native` | Supported | Supported |

## Signature

`auto` selects `triton`.

```python
from sgl_kernel_npu.operators.attention import flash_prefill_bnsd_blockq_sparse_fia


def flash_prefill_bnsd_blockq_sparse_fia(
    q,
    k_cache_bnsd,
    v_cache_bnsd,
    topk_idx,
    seq_lens,
    per_query_req,
    req_to_token,
    block_size,
    sm_scale,
    num_pages,
    max_num_blocks,
    block_table_out=None,
    actual_kvlen_out=None,
    *,
    impl="auto"
): ...
```

## Description

Computes sparse causal prefill attention from selected logical KV blocks. It gathers physical pages, orders the current block after past blocks, limits that block to the query's causal length, and evaluates grouped-query attention. Only one KV head is supported; optional metadata buffers are updated in place.

## Parameters and return value

All tensors must be on the same NPU device.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `q` | Tensor | `Required` | KDA: [1,tokens,Hq,K]; FIA: FP16/BF16 [tokens,QH,D]. |
| `k_cache_bnsd` | Tensor | `Required` | FP16/BF16 [num_pages,block_size,1,D] cache. |
| `v_cache_bnsd` | Tensor | `Required` | Value cache, same shape/dtype as keys. |
| `topk_idx` | Tensor | `Required` | INT32 [1,tokens,topk+1] selected logical blocks, -1 padding, including the current block. |
| `seq_lens` | Tensor | `Required` | Integer per-request query lengths; FIA instead uses per-query causal KV lengths. |
| `per_query_req` | Tensor | `Required` | INT32 [tokens] request indices. |
| `req_to_token` | Tensor | `Required` | INT32 [requests,max_context] physical token slots. |
| `block_size` | int | `Required` | Tokens per physical KV page. |
| `sm_scale` | float or None | `Required` | Attention score scale; None uses head_dim**-0.5. |
| `num_pages` | int | `Required` | Physical cache page count, matching cache shape. |
| `max_num_blocks` | int | `Required` | Reserved maximum selected-block count. |
| `block_table_out` | Tensor or None | `None` | Optional writable INT32 [tokens,topk+1] page table. |
| `actual_kvlen_out` | Tensor or None | `None` | Optional writable INT32 [tokens] causal lengths. |
| `impl` | str | `'auto'` | auto selects triton; explicit backend selection. |

| Return | Type | Description |
| --- | --- | --- |
| result | Tensor | [tokens,QH,D] output with q dtype/device. |

## triton

Uses Triton page preparation and the native NPU fused attention API.

## torch_native

Uses PyTorch tensor operations on the input device.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.attention import flash_prefill_bnsd_blockq_sparse_fia

q = torch.randn(1, 16, 128, device="npu", dtype=torch.bfloat16)
k = torch.randn(2, 128, 1, 128, device="npu", dtype=q.dtype)
v = torch.randn_like(k)
blocks = torch.tensor([[[-1, 0]]], device="npu", dtype=torch.int32)
lengths = torch.tensor([4], device="npu", dtype=torch.int32)
request = torch.tensor([0], device="npu", dtype=torch.int32)
mapping = torch.arange(256, device="npu", dtype=torch.int32).view(1, -1)
y = flash_prefill_bnsd_blockq_sparse_fia(
    q, k, v, blocks, lengths, request, mapping, 128, None, 2, 2, impl="triton"
)
```
