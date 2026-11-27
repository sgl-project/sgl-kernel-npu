# fused_lightning_indexer_manage

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `ascendc` | Unverified | Unsupported |

## Signature

```python
from sgl_kernel_npu.operators.indexer import fused_lightning_indexer_manage
```

```python
def fused_lightning_indexer_manage(
    index_weights, query, index_key_cache, index_block_table,
    actual_seq_lengths_query, actual_seq_lengths_key, offload_seq_lengths_key,
    num_cache_tokens, request_state, req_pool_entries, cache_slots_pool,
    topk_src_ids, topk_dst_slots, topk_miss_counts, miss_src_ids,
    miss_dst_slots, miss_counts, full_visible=False, *, impl="auto",
) -> None:
    ...
```

## Description

The operator computes index TopK and updates the resident slot map and the
miss plan in place. `impl="auto"` selects `"ascendc"` on ARCH22 with the native
operator included in the wheel.

## Parameters and return value

| Arguments | Contract |
| --- | --- |
| `index_weights`, `query`, `index_key_cache` | FP16 or BF16 tensors of shape `[T, H]`, `[T, H, 128]` and `[P, 128, 1, 128]`. |
| `index_block_table` | Paged index key mapping, int32 `[B, L]`. |
| `actual_seq_lengths_query`, `actual_seq_lengths_key`, `offload_seq_lengths_key` | Int32 `[B]` cumulative query ends, visible key lengths and previous offload lengths. |
| `num_cache_tokens`, `request_state`, `req_pool_entries` | Int32 `[B]` resident capacities, request states and rows in the persistent slot map. |
| `cache_slots_pool` | Mutable int32 `[R, S]` encoded source-to-slot state. |
| `topk_src_ids`, `topk_dst_slots`, `topk_miss_counts` | Mutable int32 `[T, 2048]`, `[T, 2048]` and `[T]` query plans. |
| `miss_src_ids`, `miss_dst_slots`, `miss_counts` | Mutable int32 `[B, 32768]`, `[B, 32768]` and `[B]` request miss plans. |
| `full_visible` | Include all causally visible positions in TopK selection. Defaults to `False`. |

`T` is the packed query length, `H` the index head count, `P` the number of key
pages and `B` the request count. Tensors must be contiguous on the same NPU.
The return value is `None`. See the [native contract and test command](../../../../../csrc/indexer/fused_lightning_indexer_manage/README.md).

## ascendc backend

Uses the wheel’s native LIM binding. The correctness suite compares the emitted slot plan with an independent CPU hotset and cache model.

## Runnable example

```python
import torch
import torch_npu
import sgl_kernel_npu
from sgl_kernel_npu.operators.indexer import fused_lightning_indexer_manage

def main():
    torch.npu.set_device(0)
    device, dtype = "npu:0", torch.bfloat16
    batch, heads, source_capacity = 1, 32, 8320
    weights = torch.full((batch, heads), 1 / heads, dtype=dtype, device=device)
    query = torch.zeros((batch, heads, 128), dtype=dtype, device=device)
    query[..., 0] = 1
    keys = torch.zeros((source_capacity // 128, 128, 1, 128), dtype=dtype, device=device)
    keys.view(-1, 128)[:, 0] = -8
    keys.view(-1, 128)[:2048, 0] = 8
    table = torch.arange(source_capacity // 128, dtype=torch.int32, device=device).view(1, -1)
    def vec(value):
        return torch.tensor([value], dtype=torch.int32, device=device)
    pool = torch.full((batch, source_capacity), -1, dtype=torch.int32, device=device)
    ids = torch.empty((batch, 2048), dtype=torch.int32, device=device)
    slots = torch.empty_like(ids)
    miss_src = torch.empty((batch, 32768), dtype=torch.int32, device=device)
    miss_dst = torch.empty_like(miss_src)
    miss_counts = vec(0)
    fused_lightning_indexer_manage(
        weights, query, keys, table, vec(1), vec(8193), vec(8192), vec(4096),
        vec(-2), vec(0), pool, ids, slots, vec(0), miss_src, miss_dst, miss_counts,
    )
    torch.npu.synchronize()
    assert torch.equal(ids.cpu().sort().values[0], torch.arange(2048, dtype=torch.int32))
    assert int(miss_counts.cpu()[0]) == 4096  # Cold fill populates the full resident capacity.

if __name__ == "__main__":
    main()
```
