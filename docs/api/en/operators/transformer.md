# DSV4 Transformer operator index

Each operator has an individual [English](../API_list.md) and [Chinese](../../zh/API_list.md) API page.

- [sparse_flash_mla](attention/sparse_flash_mla.md)
- [mixed_quant_sparse_flash_mla](attention/mixed_quant_sparse_flash_mla.md)
- [sparse_flash_mla_metadata](attention/sparse_flash_mla_metadata.md)
- [mixed_quant_sparse_flash_mla_metadata](attention/mixed_quant_sparse_flash_mla_metadata.md)
- [quant_lightning_indexer](attention/quant_lightning_indexer.md)
- [quant_lightning_indexer_metadata](attention/quant_lightning_indexer_metadata.md)
- [compressor](attention/compressor.md)
- [kv_compress_epilog](kvcacheio/kv_compress_epilog.md)
- [indexer_quant_cache](kvcacheio/indexer_quant_cache.md)
- [mhc_pre_dpskv4](hyper_conn/mhc_pre_dpskv4.md)
- [mhc_post_dpskv4](hyper_conn/mhc_post_dpskv4.md)
- [inplace_partial_rotary_mul](rope/inplace_partial_rotary_mul.md)
- [moe_gating_top_k](moe/moe_gating_top_k.md)

Shared implementation helpers use descriptive module names. Eager attention math is in `implementation/torch_native/attention/attention.py`; metadata descriptors are in `metadata.py`. Binding selection and schema guards are in `operators/utils/transformer_compat.py`, and SDK release parsing is in `operators/utils/cann.py`. Sparse-index conversion and original indexer argument mapping share `operators/attention/attention_utils.py`.

The initialized SDK selects the ABI. On CANN 9.1, each operator chooses a
source-specific implementation; on CANN 9.2 all 13 APIs register
`cann_ops_transformer_cann920`. A5 packed KV still requires a matching reader,
metadata and writer because the SDKs use different encodings. Kernel errors
never trigger a retry with another implementation.

## Verified CANN 9.1 implementation sources

These names identify the implementation source or execution backend. They do
not rename Python packages, Torch namespaces or ACLNN symbols. `cann_9_1`
remains an internal SDK/format label and is no longer a registered implementation.

| Public API | CANN 9.1 implementation | Verified source |
| --- | --- | --- |
| `attention.sparse_flash_mla` | `ops_transformer_cann910`<br>`ascendc` | [ops-transformer experimental](https://gitcode.com/cann/ops-transformer/tree/6f0d3ec620b338cab4f2d00322ff2e0d141419d2/experimental/attention/sparse_attn_sharedkv)<br>[sgl-kernel-npu DSpark device kernel](https://github.com/sgl-project/sgl-kernel-npu/blob/26ec66efe01f8acef43db55da679107702876b23/csrc/sparse_attn_sharedkv/op_kernel/sparse_attn_sharedkv.cpp) |
| `attention.sparse_flash_mla_metadata` | `ops_transformer_cann910`<br>`host` | [ops-transformer experimental AICPU metadata](https://gitcode.com/cann/ops-transformer/tree/6f0d3ec620b338cab4f2d00322ff2e0d141419d2/experimental/attention/sparse_attn_sharedkv_metadata)<br>[sgl-kernel-npu CPU scheduling](https://github.com/sgl-project/sgl-kernel-npu/blob/26ec66efe01f8acef43db55da679107702876b23/csrc/sparse_attn_sharedkv_metadata/op_host/sparse_attn_sharedkv_metadata.cpp) |
| `attention.mixed_quant_sparse_flash_mla` | `ops_transformer_cann910`<br>`ascendc` | [ops-transformer experimental](https://gitcode.com/cann/ops-transformer/tree/6f0d3ec620b338cab4f2d00322ff2e0d141419d2/experimental/attention/kv_quant_sparse_attn_sharedkv)<br>[sgl-kernel-npu DSpark device kernel](https://github.com/sgl-project/sgl-kernel-npu/blob/26ec66efe01f8acef43db55da679107702876b23/csrc/kv_quant_sparse_attn_sharedkv/op_kernel/kv_quant_sparse_attn_sharedkv.cpp) |
| `attention.mixed_quant_sparse_flash_mla_metadata` | `ops_transformer_cann910`<br>`host` | [ops-transformer experimental AICPU metadata](https://gitcode.com/cann/ops-transformer/tree/6f0d3ec620b338cab4f2d00322ff2e0d141419d2/experimental/attention/kv_quant_sparse_attn_sharedkv_metadata)<br>[sgl-kernel-npu CPU scheduling](https://github.com/sgl-project/sgl-kernel-npu/blob/26ec66efe01f8acef43db55da679107702876b23/csrc/kv_quant_sparse_attn_sharedkv_metadata/op_host/kv_quant_sparse_attn_sharedkv_metadata.cpp) |
| `attention.quant_lightning_indexer` | `ops_transformer_cann910` | [ops-transformer experimental](https://gitcode.com/cann/ops-transformer/tree/6f0d3ec620b338cab4f2d00322ff2e0d141419d2/experimental/attention/quant_lightning_indexer) |
| `attention.quant_lightning_indexer_metadata` | `ops_transformer_cann910` | [ops-transformer experimental AICPU metadata](https://gitcode.com/cann/ops-transformer/tree/6f0d3ec620b338cab4f2d00322ff2e0d141419d2/experimental/attention/quant_lightning_indexer_metadata) |
| `attention.compressor` | `ops_transformer_cann910`<br>`ascendc` | [A5 ops-transformer experimental](https://gitcode.com/cann/ops-transformer/tree/6f0d3ec620b338cab4f2d00322ff2e0d141419d2/experimental/attention/compressor)<br>[A3 sgl-kernel-npu device kernel](https://github.com/sgl-project/sgl-kernel-npu/blob/26ec66efe01f8acef43db55da679107702876b23/csrc/compressor/op_kernel/compressor.cpp) |
| `kvcacheio.kv_compress_epilog` | `ascendc` | [sgl-kernel-npu device kernel](https://github.com/sgl-project/sgl-kernel-npu/blob/26ec66efe01f8acef43db55da679107702876b23/csrc/kv_compress_epilog/op_kernel/kv_compress_epilog.cpp) |
| `kvcacheio.indexer_quant_cache` | `cann_recipes_infer_cann910` | [cann-recipes-infer; aclnnIndexerCompressEpilog](https://gitcode.com/cann/cann-recipes-infer/blob/35a476c2b1ea61990380a7115ea991023aedc7d0/ops/ascendc/torch_ops_extension/custom_ops/csrc/npu_indexer_compress_epilog.cpp) |
| `hyper_conn.mhc_pre_dpskv4` | `cann_recipes_infer_cann910` | [cann-recipes-infer; aclnnHcPre](https://gitcode.com/cann/cann-recipes-infer/blob/35a476c2b1ea61990380a7115ea991023aedc7d0/ops/ascendc/torch_ops_extension/custom_ops/csrc/npu_hc_pre.cpp) |
| `hyper_conn.mhc_post_dpskv4` | `cann_recipes_infer_cann910` | [cann-recipes-infer; aclnnHcPost](https://gitcode.com/cann/cann-recipes-infer/blob/35a476c2b1ea61990380a7115ea991023aedc7d0/ops/ascendc/torch_ops_extension/custom_ops/csrc/npu_hc_post.cpp) |
| `rope.inplace_partial_rotary_mul` | `cann_recipes_infer_cann910` | [cann-recipes-infer; aclnnInplacePartialRotaryMul](https://gitcode.com/cann/cann-recipes-infer/blob/35a476c2b1ea61990380a7115ea991023aedc7d0/ops/ascendc/torch_ops_extension/custom_ops/csrc/inplace_partial_rotary_mul.cpp) |
| `moe.moe_gating_top_k` | `cann_recipes_infer_cann910` | [cann-recipes-infer; aclnnMoeGatingTopKHash](https://gitcode.com/cann/cann-recipes-infer/blob/35a476c2b1ea61990380a7115ea991023aedc7d0/ops/ascendc/torch_ops_extension/custom_ops/csrc/npu_moe_gating_top_k.cpp) |

The experimental ops-transformer sources were inspected at CANN 9.1.0 commit
`6f0d3ec620b338cab4f2d00322ff2e0d141419d2`. Recipes bridge sources were checked
at `35a476c2b1ea61990380a7115ea991023aedc7d0`, also pinned by the main repository's
CANN 9.1 third-party profile. This verifies source ownership, not which binary
revision is installed on a remote NPU host.

The [original migration inventory](https://github.com/sgl-project/sglang/pull/47438)
describes kernel sources. Recipes' `custom_ops` extension also contains bridges
for the experimental ops-transformer kernels, so `import custom_ops` alone does
not identify a kernel's source. The recipes `npu_hc_pre` bridge, omitted from that
inventory, calls `aclnnHcPre`; its `npu_moe_gating_top_k` calls
`aclnnMoeGatingTopKHash`.

For CANN 9.1 attention, window-only inputs select the experimental implementation;
explicit sparse original-KV inputs select the checked-in `ascendc` compute and
`host` metadata pair. An explicitly selected source rejects the other input
contract. Compressor selects `ascendc` on A3 and `ops_transformer_cann910` on A5.
`host` metadata is CPU C++ scheduling code that copies the resulting table to NPU;
this is distinct from the experimental AICPU metadata kernels.

## Binding loading and CANN 9.2

The 9.2 name `cann_ops_transformer_cann920` is a registry key. The actual Python
package and Torch namespace remain `cann_ops_transformer`. MoE keeps the native
`torch.ops.npu.npu_moe_gating_top_k` binding, and compressor keeps the existing
`torch.ops.npu.compressor` fused RMSNorm/RoPE contract. The latter remains the
checked-in kernel, not the released core-only compressor.

For 9.1, `import custom_ops` registers custom bindings, while checked-in kernels
are registered by `sgl_kernel_npu`. An optional standalone registration library
can be supplied through `SGL_KERNEL_NPU_CANN_9_1_OPS_SO`. Runtime SDK detection,
operator package loading, and source provenance are separate concerns.
