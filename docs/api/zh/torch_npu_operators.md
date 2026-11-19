# TorchNPU 算子入口

从 `sgl_kernel_npu.operators.<模块>` 导入下表的公开函数。`impl="auto"` 和 `impl="torch_npu"` 都调用已安装的 TorchNPU。参数的 dtype、布局、shape 和 SDK 版本约束由具体 TorchNPU 算子决定。表中列出注册的实现，芯片可用性还取决于具体参数组合。

| 模块 | 公开入口 | 后端 |
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

## 代际范围

- `data_move.prefetch` 和 `rope.mrope` 的 TorchNPU 后端注册在 ARCH22。后者也可显式选择 `torch_native` 使用默认 cache 布局。
- `dynamic_dual_level_mx_quant`、`dynamic_mx_quant`、`dual_level_quant_matmul`、`transpose_quant_batchmatmul` 的 TorchNPU 后端注册在 ARCH35。
- `dynamic_block_quant` 的 DSA FP8 路径需要 ARCH35。ARCH22 可使用 TorchNPU 支持的 INT8 模式。

## 调用时需要保留的语义

- FIA 和 FIA V2 的预分配输出分别通过 `fused_infer_attention_score_out` 和 `fused_infer_attention_score_v2_out` 传入，使用 `out=[attention_out, softmax_lse]`。对应的 `*_get_max_workspace` 可以预先分配 workspace。
- `flash_attention_qlens`、`paged_attention` 和 `paged_attention_mla` 写入调用方传入的 `out`。KV cache 写入和 `scatter_nd_update_` 保留 TorchNPU 的原地更新语义。
- `kv_quant_sparse_flash_attention_torch_npu` 接收 TorchNPU 的独立 key/value 和量化参数。已有的 [packed FP8 KV 接口](operators/attention/kv_quant_sparse_flash_attention.md) 接收打包缓存。

```python
import torch
from sgl_kernel_npu.operators.quant import dynamic_quant

x = torch.randn(1, 128, device="npu", dtype=torch.float16)
values, scales = dynamic_quant(x, dst_type=torch.int8, impl="torch_npu")
```

[TorchNPU 算子定义](https://github.com/Ascend/op-plugin/blob/master/op_plugin/config/op_plugin_functions.yaml)
