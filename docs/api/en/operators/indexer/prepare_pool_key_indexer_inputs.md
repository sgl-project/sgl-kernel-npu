# prepare_pool_key_indexer_inputs

Prepare safe nonempty inputs for causal CANN `TND` / `PA_BBND` indexing
(`mask_mode=3`). `auto` and `triton` select the same Triton implementation;
`torch_native` selects the PyTorch reference.

```python
from sgl_kernel_npu.operators.indexer import prepare_pool_key_indexer_inputs

query_padded, weights_padded, ends_padded, lengths_padded, tails_padded, table_padded = (
    prepare_pool_key_indexer_inputs(query, weights, ends, lengths, tails, table)
)
```

| Input | Shape | Dtype | Output |
| --- | --- | --- | --- |
| `query` | `[T,H,D]` | BF16 | Append one zero row. |
| `weights` | `[T,H]` | BF16 | Append one zero row. |
| `ends` | `[R]` | int64 | Append `T+1`. |
| `lengths` | `[R]` | int64 | Clamp to at least one and append one. |
| `tails` | `[R]` | int64 | Append zero. |
| `table` | `[M,C]` | int32 | Append one zero row. |

All inputs share the query NPU; `M >= R`, `C >= 1`. Ends are nondecreasing,
within `[0,T]`, and exclude graph-padding tokens. Repeated ends represent
requests with no queries. With `M > R`, unused table rows including row `R`
must already contain valid block IDs. Physical cache block zero and every
empty real history's first block must be allocated and readable.

Strided inputs are normalized with `contiguous()`. The six outputs retain
the input dtypes, are contiguous, do not modify inputs, and use independent
allocations for every invocation. Capture records those allocations for reuse
on graph replay; no device-to-host synchronization is introduced.

Call CANN using the prepared inputs, crop its indices to `indices[:T]`, then
pack using the original positions and valid-query metadata. The synthetic
history and graph-padding candidates must not reach attention.

The kernel is copied unchanged from vllm-ascend
`vllm_ascend/ops/triton/prepare_nonempty_keypool_inputs.py` at
`e0e8dc91ce5a5a03f5cd18176bc6f4ec5fd0e4a8`; only imports and the public
function name are adapted. The upstream Apache-2.0 notices are retained.
The migration has not been validated on target NPU hardware.
