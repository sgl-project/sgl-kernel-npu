# mem_cache

Update KPool cache and tail state.

Public import: `sgl_kernel_npu.operators.mem_cache`.

Select backends with `impl`; `auto` aliases the backend listed on each API page, with no runtime fallback. Registry import requires NPU architecture detection; implementations load lazily on first execution.

- [kpool_assemble_softmax_write_cache_npu](_kpool_assemble_softmax_write_cache_npu.md)
- [kpool_decode_update_and_maybe_write_cache_bf16](_kpool_decode_update_and_maybe_write_cache_bf16.md)
- [kpool_write_tail_and_maybe_compress_npu](_kpool_write_tail_and_maybe_compress_npu.md)
- [scatter_kpool_tail_updates_npu](_scatter_kpool_tail_updates_npu.md)

All four operations mutate state and return `None`; callers must prevent conflicting writes and initialize history. Pooled page-table construction belongs to [attention](../attention/README.md).
