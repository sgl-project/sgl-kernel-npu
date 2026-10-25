# API Index

| API | Supported backends |
| --- | --- |
| [rmsnorm_bias](operators/norm/rmsnorm_bias.md) | `triton`, `torch_native` |
| [build_pooled_page_table_64](operators/mem_cache/build_pooled_page_table_64.md) | `torch_native` |
| [kpool_decode_update_and_maybe_write_cache_bf16](operators/mem_cache/kpool_decode_update_and_maybe_write_cache_bf16.md) | `triton` |
| [kpool_assemble_softmax_write_cache_npu](operators/mem_cache/kpool_assemble_softmax_write_cache_npu.md) | `triton` |
| [kpool_write_tail_and_maybe_compress_npu](operators/mem_cache/kpool_write_tail_and_maybe_compress_npu.md) | `triton` |
| [scatter_kpool_tail_updates_npu](operators/mem_cache/scatter_kpool_tail_updates_npu.md) | `triton` |
| [fused_decode_prepare_indexer](operators/indexer/fused_decode_prepare_indexer.md) | `triton` |
| [expand_pool_topk_and_append_tail_batched_npu](operators/indexer/expand_pool_topk_and_append_tail_batched_npu.md) | `triton` |
| [causal_conv1d_target_verify_npu](operators/mamba/causal_conv1d_target_verify_npu.md) | `triton` |
