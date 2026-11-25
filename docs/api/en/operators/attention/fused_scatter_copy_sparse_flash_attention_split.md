# fused_scatter_copy_sparse_flash_attention_split

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `ascendc` | Unverified | Unsupported |

## Signature

```python
from sgl_kernel_npu.operators.attention import (
    fused_scatter_copy_sparse_flash_attention_split,
)
```

```python
def fused_scatter_copy_sparse_flash_attention_split(
    query, query_rope, hbm_ckv, hbm_rope, actual_seq_lengths_query,
    actual_seq_lengths_kv, num_cache_tokens, topk_src_ids, topk_dst_slots,
    topk_miss_counts, miss_src_ids, miss_dst_slots, miss_counts,
    hbm_block_table, dram_block_table, dram_kv_dev_ptr, attention_out,
    scale_value, *, full_visible=False, resident_refresh_tail=False,
    current=None, positions=None, valid=None, request_slots=None,
    req_to_token=None, resident_end=None, host_token_capacity=0,
    dram_rope_dev_ptr=0, impl="auto",
) -> None:
    ...
```

## Description

This operator consumes a LIM plan, copies missing KV to the split HBM cache and
computes sparse attention. `impl="auto"` selects `"ascendc"` on ARCH22 with the
native operator included in the wheel.

## Parameters and return value

| Arguments | Contract |
| --- | --- |
| `query`, `query_rope`, `attention_out` | FP16 or BF16 TND tensors, shapes `[T,H,512]`, `[T,H,64]`, `[T,H,512]`. |
| `hbm_ckv`, `hbm_rope` | Mutable paged caches, shapes `[P,128,1,512]` and `[P,128,1,64]`, with the query dtype. |
| `actual_seq_lengths_query`, `actual_seq_lengths_kv`, `num_cache_tokens` | Int32 `[B]` cumulative query ends, request-local KV lengths and cache capacities. |
| `topk_src_ids`, `topk_dst_slots`, `topk_miss_counts` | Int32 `[T,2048]`, `[T,2048]` and `[T]` query plans. |
| `miss_src_ids`, `miss_dst_slots`, `miss_counts` | Int32 `[B,M]`, `[B,M]` and `[B]` request miss plans. |
| `hbm_block_table`, `dram_block_table` | Int32 page tables with one row per request. |
| `dram_kv_dev_ptr`, `dram_rope_dev_ptr` | Registered NPU-accessible addresses for CKV and KPE in host memory. |
| `scale_value` | Attention scale. |
| `full_visible`, `resident_refresh_tail` | Full-visible mode and optional overlap for a LIM refresh suffix. Both default to `False`. |
| `current`, `positions`, `valid`, `request_slots`, `req_to_token`, `resident_end`, `host_token_capacity` | Optional full-visible resident staging group. See the native contract for shapes and bounds. |

The operator returns `None` and mutates the HBM caches, `attention_out` and the
optional `resident_end`. See the [native contract](../../../../../csrc/attention/fused_scatter_copy_sparse_flash_attention/README.md)
and [independent attention and replay tests](../../../../../tests/test_copy_sfa_split_hbm_resident.py).

## ascendc backend

Uses the wheel’s native split Copy-SFA binding. The correctness suite allocates registered host buffers and compares the output with independent PyTorch attention.

## Runnable example

```bash
python -m pytest tests/test_copy_sfa_split_hbm_resident.py -q
```
