# pack_pool_key_indexer_indices

Pack causal CANN pool-key indices and restore each query's incomplete tail.
`auto` selects `triton`; `torch_native` is the independent PyTorch reference.

```python
from sgl_kernel_npu.operators.indexer import pack_pool_key_indexer_indices

output = pack_pool_key_indexer_indices(
    indices, positions, query_ends, topk, pool_size, output_buffer=None
)
```

Inputs are int32 `indices[T,C]`, int64 `positions[T]` and int64
`query_ends[B]`, all contiguous along their last dimension on the same NPU.
`C >= topk + pool_size - 1`. Query ends are the original nondecreasing
cumulative real-query boundaries, without the sentinel. Valid queries form a
prefix of length `query_ends[-1]`; remaining rows are graph padding. A nonempty
input requires at least one query-end entry. `topk` and `pool_size` follow the
causal CANN call that produced the indices.

For position `p`, the complete history ends at
`tail_start = (p+1) // pool_size * pool_size`. Keep the first
`min(tail_start, topk)` CANN indices, then append the raw positions in
`[tail_start,p+1)`. Unused columns and padding rows become `-1`. History
candidates are consumed without extra value filtering, matching the original
vllm-ascend kernel. Artificial history for a short request is discarded because
its complete-history count is zero.

The result is int32 `[T,topk+pool_size-1]`. By default the output aliases and
overwrites `indices`. An optional int32 output buffer may be wider and have
additional rows; it must have sufficient capacity and contiguous columns.
Its unused columns within the first `T` rows are also filled with `-1`.
Unlike vLLM's wrapper, this wrapper returns a two-dimensional tensor.

The Triton kernel is copied unchanged from
`vllm_ascend/ops/triton/glm5_next_cann_keypool.py::_pack_indices_kernel` at
vllm-ascend `e0e8dc91ce5a5a03f5cd18176bc6f4ec5fd0e4a8`. The public interface
uses `query_ends` in place of the former token-validity mask. Both repositories
must be updated together. Target NPU validation of this migration is pending.
