# kpool_decode_update_and_maybe_write_cache_bf16

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Unverified | Unverified |

## Signature

```python
from sgl_kernel_npu.operators.mem_cache import (
    kpool_decode_update_and_maybe_write_cache_bf16,
)

def kpool_decode_update_and_maybe_write_cache_bf16(
    buf,
    tail_k,
    tail_score,
    key,
    slot_score,
    ape,
    block_tables,
    req_pool_indices,
    positions,
    seq_lens,
    out_cache_loc,
    *,
    pool_size,
    slots_per_page,
    head_dim=128,
    impl="auto",
):
    ...
```

## Description

Updates the request-local BF16 tail key and score rings for valid decode rows. A row is valid when its request index is in range, `out_cache_loc[row] != 0`, and `0 <= positions[row] < seq_lens[row]`.

When the current token completes a pool, the kernel computes a dimension-wise softmax over `score + ape` across that pool and writes the resulting BF16 weighted key to the flattened index cache. The cache page is obtained through `block_tables`. There is no Hadamard transform, FP8 quantization, or per-slot scale output.

## Parameters

| Parameter | Type | Shape | Dtype | Default | Description |
| --- | --- | --- | --- | --- | --- |
| `buf` | `torch.Tensor` | `[num_pages, slots_per_page * head_dim]` | `torch.bfloat16` | Required | Flattened index cache; updated in place when a pool completes. |
| `tail_k` | `torch.Tensor` | `[R, T, head_dim]` | `torch.bfloat16` | Required | Request-local key rings; updated in place for valid rows. |
| `tail_score` | `torch.Tensor` | Same as `tail_k` | `torch.float32` per the backend contract | Required | Scores paired with `tail_k`; updated in place for valid rows. |
| `key` | `torch.Tensor` | `[B, head_dim]` | `torch.bfloat16` | Required | Current decode keys. |
| `slot_score` | `torch.Tensor` | `[B, head_dim]` | `torch.float32` per the backend contract | Required | Scores paired with `key`. |
| `ape` | `torch.Tensor` | `[pool_size, head_dim]` | `torch.float32` | Required | Additive pooling-position scores. |
| `block_tables` | `torch.Tensor` | `[B, BLOCK_TABLE_COLS]` | `torch.int32` | Required | Token-page table used to locate a completed pool's cache page. |
| `req_pool_indices` | `torch.Tensor` | `[>= B]` | Integer; not explicitly checked by the wrapper | Required | Request/ring row for each batch item. |
| `positions` | `torch.Tensor` | `[>= B]` | Integer; not explicitly checked by the wrapper | Required | Current token position for each batch item. |
| `seq_lens` | `torch.Tensor` | `[>= B]` | Integer; not explicitly checked by the wrapper | Required | Sequence length for each batch item. |
| `out_cache_loc` | `torch.Tensor` | `[>= B]` | Integer; not explicitly checked by the wrapper | Required | Token cache locations; a zero entry suppresses that row's updates. Read only. |
| `pool_size` | `int` | — | — | Required | Keyword-only number of tokens in one pool. |
| `slots_per_page` | `int` | — | — | Required | Keyword-only number of pooled slots per flattened cache page. |
| `head_dim` | `int` | — | — | `128` | Keyword-only channel width; must match the tail's last dimension. |
| `impl` | `str` | — | — | `"auto"` | Keyword-only; accepts `"auto"` or `"triton"`. |

## Return value and in-place state

Returns `None`. `buf`, `tail_k`, and `tail_score` are the mutable state; the other tensors are read only. An empty batch returns without launching the kernel.

Pass contiguous tensors on the same NPU. The backend calls `.contiguous()` on its inputs, so a non-contiguous mutable input can be copied and would not receive caller-visible updates. Metadata values and page locations must be valid upstream; the wrapper checks only selected dtype and shape conditions.

## triton backend

- Requires the BF16 cache/key layout described above. Pool accumulation uses FP32 and the pooled key is converted to BF16.
- `auto` is registered to the same entry point as `triton`; it is not dynamic selection or fallback. Importing the registry still performs ACL architecture detection.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.mem_cache import (
    kpool_decode_update_and_maybe_write_cache_bf16,
)

b, r, t, h = 1, 1, 4, 128
buf = torch.zeros(1, 8 * h, device="npu", dtype=torch.bfloat16)
tail_k = torch.zeros(r, t, h, device="npu", dtype=torch.bfloat16)
tail_score = torch.zeros(r, t, h, device="npu", dtype=torch.float32)
key = torch.randn(b, h, device="npu", dtype=torch.bfloat16)
slot_score = torch.zeros(b, h, device="npu", dtype=torch.float32)
ape = torch.zeros(4, h, device="npu", dtype=torch.float32)
block_tables = torch.zeros(b, 1, device="npu", dtype=torch.int32)
metadata = dict(
    req_pool_indices=torch.zeros(b, device="npu", dtype=torch.int32),
    positions=torch.tensor([3], device="npu", dtype=torch.int32),
    seq_lens=torch.tensor([4], device="npu", dtype=torch.int32),
    out_cache_loc=torch.ones(b, device="npu", dtype=torch.int32),
)

kpool_decode_update_and_maybe_write_cache_bf16(
    buf, tail_k, tail_score, key, slot_score, ape, block_tables,
    **metadata, pool_size=4, slots_per_page=8,
)
```
