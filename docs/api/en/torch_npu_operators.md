# TorchNPU operator entry points

Import public functions from `sgl_kernel_npu.operators.<module>`. Both `impl="auto"` and `impl="torch_npu"` call the installed TorchNPU backend. The specific TorchNPU operator defines the accepted dtypes, layouts, shapes, and SDK versions. The table lists registered implementations. Hardware availability also depends on the argument combination.

| Module | Public entry points | Backends |
| --- | --- | --- |
| `activation` | [fast_gelu](operators/activation/fast_gelu.md), [geglu](operators/activation/geglu.md), [swiglu](operators/activation/swiglu.md) | `torch_npu`, `torch_native` |
| `attention` | [flash_attention_qlens](operators/attention/flash_attention_qlens.md), [paged_attention](operators/attention/paged_attention.md), [paged_attention_mla](operators/attention/paged_attention_mla.md), [attention_update](operators/attention/attention_update.md), [fused_infer_attention_score](operators/attention/fused_infer_attention_score.md), [fused_infer_attention_score_v2](operators/attention/fused_infer_attention_score_v2.md), [kv_quant_sparse_flash_attention_torch_npu](operators/attention/kv_quant_sparse_flash_attention_torch_npu.md), [mla_prolog_v3](operators/attention/mla_prolog_v3.md) | `torch_npu` |
| `data_move` | [dtype_cast](operators/data_move/dtype_cast.md), [format_cast](operators/data_move/format_cast.md), [prefetch](operators/data_move/prefetch.md) | `torch_npu` |
| `indexer` | [lightning_indexer](operators/indexer/lightning_indexer.md) | `torch_npu` |
| `matmul` | [dual_level_quant_matmul](operators/matmul/dual_level_quant_matmul.md), [quant_matmul](operators/matmul/quant_matmul.md), [transpose_quant_batchmatmul](operators/matmul/transpose_quant_batchmatmul.md), [weight_quant_batchmatmul](operators/matmul/weight_quant_batchmatmul.md) | `torch_npu` |
| `mem_cache` | [reshape_and_cache](operators/mem_cache/reshape_and_cache.md), [scatter_pa_kv_cache](operators/mem_cache/scatter_pa_kv_cache.md), [kv_rmsnorm_rope_cache](operators/mem_cache/kv_rmsnorm_rope_cache.md) | `torch_npu` |
| `norm` | [add_rms_norm](operators/norm/add_rms_norm.md), [rms_norm](operators/norm/rms_norm.md) | `torch_npu`, `torch_native` |
| `quant` | [dynamic_block_quant](operators/quant/dynamic_block_quant.md), [dynamic_dual_level_mx_quant](operators/quant/dynamic_dual_level_mx_quant.md), [dynamic_mx_quant](operators/quant/dynamic_mx_quant.md), [dynamic_quant](operators/quant/dynamic_quant.md) | `torch_npu` |
| `rope` | [interleave_rope](operators/rope/interleave_rope.md), [mrope](operators/rope/mrope.md), [rotary_mul](operators/rope/rotary_mul.md) | `torch_npu`, `torch_native` |
| `sampler` | [top_k_top_p](operators/sampler/top_k_top_p.md) | `torch_npu`, `torch_native` |
| `tensor_ops` | [scatter_nd_update_](operators/tensor_ops/scatter_nd_update_.md) | `torch_npu` |

## Registered generations

- The TorchNPU backends for `data_move.prefetch` and `rope.mrope` are registered on ARCH22. The `mrope` reference backend supports the default cache layout.
- The TorchNPU backends for `dynamic_dual_level_mx_quant`, `dynamic_mx_quant`, `dual_level_quant_matmul`, and `transpose_quant_batchmatmul` are registered on ARCH35.
- The DSA FP8 workload for `dynamic_block_quant` requires ARCH35. ARCH22 can use the INT8 mode supported by TorchNPU.

## Caller-provided buffers

- Use `fused_infer_attention_score_out` or `fused_infer_attention_score_v2_out` with `out=[attention_out, softmax_lse]` for preallocated FIA outputs. Each version also exposes a corresponding `*_get_max_workspace` function.
- `flash_attention_qlens`, `paged_attention`, and `paged_attention_mla` write to the caller's `out`. The KV-cache writers and `scatter_nd_update_` preserve TorchNPU's in-place updates.
- `kv_quant_sparse_flash_attention_torch_npu` accepts separate TorchNPU key/value and quantization parameters. The existing [packed FP8 KV API](operators/attention/kv_quant_sparse_flash_attention.md) accepts a packed cache.

```python
import torch
from sgl_kernel_npu.operators.quant import dynamic_quant

x = torch.randn(1, 128, device="npu", dtype=torch.float16)
values, scales = dynamic_quant(x, dst_type=torch.int8, impl="torch_npu")
```

[TorchNPU operator schemas](https://github.com/Ascend/op-plugin/blob/master/op_plugin/config/op_plugin_functions.yaml)
