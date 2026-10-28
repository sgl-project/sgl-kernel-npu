# mem_cache

Construct pooled page tables and update KPool cache/tail state.

Public import: `sgl_kernel_npu.operators.mem_cache`.

Select backends with `impl`; `auto` aliases the backend listed on each API page, with no runtime fallback. Registry import requires NPU architecture detection; implementations load lazily on first execution.

- [build_pooled_page_table_64](_build_pooled_page_table_64.md)
- [kpool_assemble_softmax_write_cache_npu](_kpool_assemble_softmax_write_cache_npu.md)
- [kpool_decode_update_and_maybe_write_cache_bf16](_kpool_decode_update_and_maybe_write_cache_bf16.md)
- [kpool_write_tail_and_maybe_compress_npu](_kpool_write_tail_and_maybe_compress_npu.md)
- [scatter_kpool_tail_updates_npu](_scatter_kpool_tail_updates_npu.md)

All operations except page-table construction mutate state and return `None`; callers must prevent conflicting writes and initialize history. Page-table construction maps both `auto` and `torch_native` to the same implementation.
