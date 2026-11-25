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

```bash
python -m pytest tests/python/sgl_kernel_npu/test_fused_lightning_indexer_manage.py -q
```
