# indexer

Prepare causal metadata, pooled indices and decode indexer inputs.

Public import: `sgl_kernel_npu.operators.indexer`.

Select backends with `impl`; `auto` aliases the backend listed on each API page, with no runtime fallback. Registry import requires NPU architecture detection; implementations load lazily on first execution.

- [expand_causal_seqlens](_expand_causal_seqlens.md)
- [expand_kpool_group_and_tail_to_topk](_expand_kpool_group_and_tail_to_topk.md)
- [kpool_indexer_prepare_decode_input](_kpool_indexer_prepare_decode_input.md)

All three APIs map `auto` to `triton` and provide independent `torch_native` backends. `expand_causal_seqlens` can overwrite supplied output buffers; the other two allocate outputs.
