# DSV4 Transformer operator index

Each operator has an individual [English](../API_list.md) and [Chinese](../../zh/API_list.md) API page.

- [sparse_flash_mla](attention/sparse_flash_mla.md)
- [mixed_quant_sparse_flash_mla](attention/mixed_quant_sparse_flash_mla.md)
- [sparse_flash_mla_metadata](attention/sparse_flash_mla_metadata.md)
- [mixed_quant_sparse_flash_mla_metadata](attention/mixed_quant_sparse_flash_mla_metadata.md)
- [quant_lightning_indexer](indexer/quant_lightning_indexer.md)
- [quant_lightning_indexer_metadata](indexer/quant_lightning_indexer_metadata.md)
- [compressor](attention/compressor.md)
- [kv_compress_epilog](mem_cache/kv_compress_epilog.md)
- [indexer_quant_cache](mem_cache/indexer_quant_cache.md)
- [mhc_pre_dpskv4](hyper_conn/mhc_pre_dpskv4.md)
- [mhc_post_dpskv4](hyper_conn/mhc_post_dpskv4.md)
- [inplace_partial_rotary_mul](rope/inplace_partial_rotary_mul.md)
- [moe_gating_top_k](moe/moe_gating_top_k.md)

Shared implementation helpers use descriptive module names. Eager attention math is in `implementation/torch_native/attention/attention.py`; metadata descriptors are in `metadata.py`. Each operator selects its SDK implementation directly using `detect_cann_version()` from `operators/utils/cann.py` and loads its bundled binding locally. MoE schema compatibility stays in `operators/moe/moe_gating_top_k.py`. Sparse-index conversion and original indexer argument mapping share `operators/attention/attention_utils.py`.

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
| `indexer.quant_lightning_indexer` | `ops_transformer_cann910` | [ops-transformer experimental](https://gitcode.com/cann/ops-transformer/tree/6f0d3ec620b338cab4f2d00322ff2e0d141419d2/experimental/attention/quant_lightning_indexer) |
| `indexer.quant_lightning_indexer_metadata` | `ops_transformer_cann910` | [ops-transformer experimental AICPU metadata](https://gitcode.com/cann/ops-transformer/tree/6f0d3ec620b338cab4f2d00322ff2e0d141419d2/experimental/attention/quant_lightning_indexer_metadata) |
| `attention.compressor` | `ops_transformer_cann910`<br>`ascendc` | [A5 ops-transformer experimental](https://gitcode.com/cann/ops-transformer/tree/6f0d3ec620b338cab4f2d00322ff2e0d141419d2/experimental/attention/compressor)<br>[A3 sgl-kernel-npu device kernel](https://github.com/sgl-project/sgl-kernel-npu/blob/26ec66efe01f8acef43db55da679107702876b23/csrc/compressor/op_kernel/compressor.cpp) |
| `mem_cache.kv_compress_epilog` | `ascendc` | [sgl-kernel-npu device kernel](https://github.com/sgl-project/sgl-kernel-npu/blob/26ec66efe01f8acef43db55da679107702876b23/csrc/kv_compress_epilog/op_kernel/kv_compress_epilog.cpp) |
| `mem_cache.indexer_quant_cache` | `cann_recipes_infer_cann910` | [cann-recipes-infer; aclnnIndexerCompressEpilog](https://gitcode.com/cann/cann-recipes-infer/blob/35a476c2b1ea61990380a7115ea991023aedc7d0/ops/ascendc/torch_ops_extension/custom_ops/csrc/npu_indexer_compress_epilog.cpp) |
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

The 9.2 name `cann_ops_transformer_cann920` is a registry key. The Torch
namespace remains `cann_ops_transformer`; its Python binding is loaded from the
bundled `sgl_kernel_npu._third_party.sgl_ops_transformer` package. MoE keeps the native
`torch.ops.npu.npu_moe_gating_top_k` binding, and compressor keeps the existing
`torch.ops.npu.compressor` fused RMSNorm/RoPE contract. The latter remains the
checked-in kernel, not the released core-only compressor.

For 9.1, the bundled `sgl_kernel_npu._third_party.sgl_cann_recipes_infer`
extension registers selected recipes bridges in the private
`torch.ops.sgl_cann_recipes_infer` namespace. The old experimental ACLNN kernels
are built from the pinned ops-transformer source and bundled as a separate vendor;
checked-in kernels require the existing native build's registrations. The existing public
mHC APIs also use a separately bundled released ops-transformer binding. Neither
SDK path requires an environment-installed `custom_ops` or
`cann_ops_transformer` Python package. The wheel still requires the matching CANN
SDK and torch_npu at runtime.

Wheel-private `libcust_opapi.so` libraries receive distinct SONAMEs during
staging. The filenames and absolute-path CANN lookups stay unchanged. This
prevents a later bare-name `dlopen("libcust_opapi.so")`, such as DeepEP's, from
reusing a previously loaded private vendor. `RTLD_LOCAL` alone does not isolate
SONAME lookup. Each third-party adapter isolates the staged vendors before
hashing its payload; `_build_manifest.json` records each dependency's
`opapi_soname` and hashes of the staged binaries. Rebuild and reinstall the
sgl-kernel-npu wheel to apply this packaging change; no DeepEP source change is
required. This isolates library selection and does not certify graph-capture
or communication accuracy on an NPU.


## Repository-owned DSV4 kernels

The root `setup.py` additionally builds the existing fused compressor into
`sgl_kernel_npu/lib/libsgl_compressor.so`. The build reuses this checkout's
compressor kernel, host implementation and architecture-specific tiling without
changing C++ sources. Its original Torch schema is extracted from
`csrc/pytorch_extensions.cpp`; package import loads the library before the first
NPU use and checks that `torch.ops.npu.compressor` is registered. The main library
continues to contain the selected migrated CANN adapters.

Compressor intermediates are kept under `build/compressor` for incremental
builds. `--make-clean` removes that directory along with the other local build
outputs, while preserving third-party dependency caches. The existing
`build.sh` native build remains unchanged. Legacy direct-launch DSpark bindings
still require their existing native build integration; this compressor fix does
not assert complete DSV4 service coverage.

CANN 9.1 A5 compressor uses the bundled recipes/experimental implementation;
CANN 9.1 A3 and CANN 9.2 A3/A5 use the repository's fused compressor.

Graph metadata allocation uses the size API beside each metadata operator:
`get_sparse_flash_mla_metadata_size()`,
`get_mixed_quant_sparse_flash_mla_metadata_size()`, and
`get_quant_lightning_indexer_metadata_size()`. These pure ABI queries do not
load a third-party bridge or execute an NPU kernel. The pinned CANN 9.1
metadata buffers and the pinned CANN 9.2 release-branch buffers have 1024 int32
entries.
Each metadata operator test checks its actual output against its size API.
