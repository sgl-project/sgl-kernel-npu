# indexer

Prepare causal metadata, pooled indices and decode indexer inputs.

Public import: `sgl_kernel_npu.operators.indexer`.

Select backends with `impl`; `auto` aliases the backend listed on each API page, with no runtime fallback. Registry import requires NPU architecture detection; implementations load lazily on first execution.

- [expand_causal_seqlens](_expand_causal_seqlens.md)
- [expand_pool_topk_and_append_tail_batched_npu](_expand_pool_topk_and_append_tail_batched_npu.md)
- [fused_decode_prepare_indexer](_fused_decode_prepare_indexer.md)

All three APIs map `auto` to `triton` and provide independent `torch_native` backends. `expand_causal_seqlens` can overwrite supplied output buffers; the other two allocate outputs.
