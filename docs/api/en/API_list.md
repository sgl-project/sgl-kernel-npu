# API Index

| API | Supported backends |
| --- | --- |
| [kv_quant_sparse_attn_sharedkv](operators/attention/kv_quant_sparse_attn_sharedkv.md) | `ascendc`, `torch_native` |
| [kv_quant_sparse_attn_sharedkv_metadata_host](operators/attention/kv_quant_sparse_attn_sharedkv_metadata_host.md) | `host`, `torch_native` |
| [add_rmsnorm_bias](operators/norm/add_rmsnorm_bias.md) | `triton`, `torch_native` |
| [fused_rmsnorm_without_weight](operators/norm/fused_rmsnorm_without_weight.md) | `triton`, `torch_native` |
| [fused_rsqrt_mul](operators/norm/fused_rsqrt_mul.md) | `triton`, `torch_native` |
| [fused_variance](operators/norm/fused_variance.md) | `triton`, `torch_native` |
| [rmsnorm_bias](operators/norm/rmsnorm_bias.md) | `triton`, `torch_native` |
| [recompute_w_u_fwd](operators/attention/gdn/recompute_w_u_fwd.md) | `triton`, `torch_native` |
| [solve_tril](operators/attention/gdn/solve_tril.md) | `triton`, `torch_native` |
| [apply_token_bitmask](operators/sampler/apply_token_bitmask.md) | `ascendc`, `torch_native` |
| [l1_norm](operators/norm/l1_norm.md) | `triton`, `torch_native` |
| [l2norm_fwd](operators/norm/l2norm_fwd.md) | `triton`, `torch_native` |
| [layer_norm_fwd](operators/norm/layer_norm_fwd.md) | `triton`, `torch_native` |
| [fused_rope_qk_mqa](operators/rope/fused_rope_qk_mqa.md) | `triton`, `torch_native` |
| [gate_multiply](operators/elementwise/gate_multiply.md) | `triton`, `torch_native` |
| [row_argmax](operators/sample/row_argmax.md) | `triton`, `torch_native` |
| [markov_greedy_step_triton](operators/sample/markov_greedy_step_triton.md) | `triton`, `torch_native` |
| [build_out_tokens_triton](operators/sample/build_out_tokens_triton.md) | `triton`, `torch_native` |
| [markov_sample_step_triton](operators/sample/markov_sample_step_triton.md) | `triton`, `torch_native` |
| [pool_key_indexer](operators/indexer/pool_key_indexer.md) | `aclnn` |
| [expand_causal_seqlens](operators/indexer/expand_causal_seqlens.md) | `triton`, `torch_native` |
| [expand_kpool_group_and_tail_to_topk](operators/indexer/expand_kpool_group_and_tail_to_topk.md) | `triton`, `torch_native` |
| [kpool_indexer_prepare_decode_input](operators/indexer/kpool_indexer_prepare_decode_input.md) | `triton`, `torch_native` |
| [build_pooled_page_table](operators/indexer/build_pooled_page_table.md) | `torch_native` |
| [kpool_assemble_softmax_rotate_write_cache](operators/mem_cache/kpool_assemble_softmax_rotate_write_cache.md) | `triton`, `torch_native` |
| [kpool_decode_update_and_maybe_write_cache](operators/mem_cache/kpool_decode_update_and_maybe_write_cache.md) | `triton`, `torch_native` |
| [kpool_write_tail_and_maybe_compress](operators/mem_cache/kpool_write_tail_and_maybe_compress.md) | `triton`, `torch_native` |
| [scatter_kpool_tail_updates](operators/mem_cache/scatter_kpool_tail_updates.md) | `triton`, `torch_native` |
| [attn_res_mix_fused](operators/hyper_conn/attn_res_mix_fused.md) | `triton`, `torch_native` |
| [flatten_patches](operators/multimodal/flatten_patches.md) | `torch_native` |
| [flash_decode_bnsd_with_topk_idx](operators/indexer/flash_decode_bnsd_with_topk_idx.md) | `triton`, `torch_native` |

## Batch-invariant operators

These operators are wrappers over an external package and require additional
installation steps; see
[installation](operators/batch_invariant/installation.md).

| API | Supported backends |
| --- | --- |
| [mm](operators/batch_invariant/mm.md) | `batch_invariant_ops` |
| [matmul](operators/batch_invariant/matmul.md) | `batch_invariant_ops` |
| [reduce_mean](operators/batch_invariant/reduce_mean.md) | `batch_invariant_ops` |
| [reduce_sum](operators/batch_invariant/reduce_sum.md) | `batch_invariant_ops` |
| [softmax](operators/batch_invariant/softmax.md) | `batch_invariant_ops` |
| [log_softmax](operators/batch_invariant/log_softmax.md) | `batch_invariant_ops` |
| [add_rms_norm](operators/batch_invariant/add_rms_norm.md) | `batch_invariant_ops` |
| [fused_linear_online_max_sum](operators/batch_invariant/fused_linear_online_max_sum.md) | `batch_invariant_ops` |
| [fused_infer_attention_score](operators/batch_invariant/fused_infer_attention_score.md) | `batch_invariant_ops` |
| [fused_infer_attention_score_v2](operators/batch_invariant/fused_infer_attention_score_v2.md) | `batch_invariant_ops` |
