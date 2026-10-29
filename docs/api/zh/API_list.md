# API 索引

| 接口名称 | 支持的后端 |
| --- | --- |
| [kv_quant_sparse_attn_sharedkv](operators/attention/kv_quant_sparse_attn_sharedkv.md) | `ascendc`, `torch_native` |
| [kv_quant_sparse_attn_sharedkv_metadata_host](operators/attention/kv_quant_sparse_attn_sharedkv_metadata_host.md) | `host`, `torch_native` |
| [add_rmsnorm_bias](operators/norm/add_rmsnorm_bias.md) | `triton`、`torch_native` |
| [fused_rmsnorm_without_weight](operators/norm/fused_rmsnorm_without_weight.md) | `triton`、`torch_native` |
| [fused_rsqrt_mul](operators/norm/fused_rsqrt_mul.md) | `triton`、`torch_native` |
| [fused_variance](operators/norm/fused_variance.md) | `triton`、`torch_native` |
| [rmsnorm_bias](operators/norm/rmsnorm_bias.md) | `triton`、`torch_native` |
| [apply_token_bitmask](operators/sampler/apply_token_bitmask.md) | `ascendc`, `torch_native` |
| [l1_norm](operators/norm/l1_norm.md) | `triton`、`torch_native` |
| [l2norm_fwd](operators/norm/l2norm_fwd.md) | `triton`、`torch_native` |
| [layer_norm_fwd](operators/norm/layer_norm_fwd.md) | `triton`、`torch_native` |
| [fused_rope_qk_mqa](operators/rope/fused_rope_qk_mqa.md) | `triton`、`torch_native` |
| [row_argmax](operators/sample/row_argmax.md) | `triton`、`torch_native` |
| [markov_greedy_step_triton](operators/sample/markov_greedy_step_triton.md) | `triton`、`torch_native` |
| [build_out_tokens_triton](operators/sample/build_out_tokens_triton.md) | `triton`、`torch_native` |
| [markov_sample_step_triton](operators/sample/markov_sample_step_triton.md) | `triton`、`torch_native` |
| [pool_key_indexer](operators/attentions/_pool_key_indexer.md) | `aclnn` |
| [expand_causal_seqlens](operators/indexer/_expand_causal_seqlens.md) | `triton`, `torch_native` |
| [expand_pool_topk_and_append_tail_batched_npu](operators/indexer/_expand_pool_topk_and_append_tail_batched_npu.md) | `triton`, `torch_native` |
| [fused_decode_prepare_indexer](operators/indexer/_fused_decode_prepare_indexer.md) | `triton`, `torch_native` |
| [causal_conv1d_target_verify_npu](operators/mamba/_causal_conv1d_target_verify_npu.md) | `triton`, `torch_native` |
| [build_pooled_page_table_64](operators/attention/_build_pooled_page_table_64.md) | `torch_native` |
| [kpool_assemble_softmax_write_cache_npu](operators/mem_cache/_kpool_assemble_softmax_write_cache_npu.md) | `triton`, `torch_native` |
| [kpool_decode_update_and_maybe_write_cache_bf16](operators/mem_cache/_kpool_decode_update_and_maybe_write_cache_bf16.md) | `triton`, `torch_native` |
| [kpool_write_tail_and_maybe_compress_npu](operators/mem_cache/_kpool_write_tail_and_maybe_compress_npu.md) | `triton`, `torch_native` |
| [scatter_kpool_tail_updates_npu](operators/mem_cache/_scatter_kpool_tail_updates_npu.md) | `triton`, `torch_native` |
