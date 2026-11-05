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
- [mhc_pre](norm/mhc_pre.md)
- [mhc_post](norm/mhc_post.md)
- [inplace_partial_rotary_mul](rope/inplace_partial_rotary_mul.md)
- [moe_gating_top_k](moe/moe_gating_top_k.md)

Shared implementation helpers use descriptive module names. Eager attention math is in `implementation/torch_native/attention/attention.py`; metadata descriptors are in `metadata.py`. Binding selection and schema guards are in `operators/utils/transformer_compat.py`, and SDK release parsing is in `operators/utils/cann.py`. Sparse-index conversion and original indexer argument mapping share `operators/attention/attention_utils.py`.

The indexer registers separate `cann_9_1` and `cann_ops_transformer_cann920` implementations; `auto` selects by initialized SDK. The 12 migrated Transformer implementation names are `cann_ops_transformer_cann920`; MoE retains `torch_npu`. The actual Python import and Torch namespace remain `cann_ops_transformer`. Packed KV still requires a paired reader, metadata and writer because CANN 9.1 and 9.2 use different storage encodings. No execution failure retries another backend.

The optional standalone CANN 9.1 registration library is configured with
`SGL_KERNEL_NPU_CANN_9_1_OPS_SO`. Set it only when `custom_ops` does not supply
the required bindings. Internal helpers, format constants, diagnostics and
reference parameters consistently use the `cann_9_1` name. The quantization
mode strings remain `fp8_bf16` (608 bytes) and `fp8_e8m0` (640 bytes).

## Implementation names and binding packages

`cann_ops_transformer_cann920` is an SGL operator implementation name, not a
Python package. It selects the existing CANN 9.2 path; Transformer bindings are
still loaded by `import cann_ops_transformer`. The fused compressor continues
to use `torch.ops.npu.compressor`, and MoE retains the `torch_npu` implementation.

The `cann_9_1` implementation is a compatibility path across original
`torch.ops.custom` and `torch.ops.npu` bindings. Importing `custom_ops` registers
the custom namespace. That package name does not establish a single source
repository or cover every migrated operator; deployments must supply their
matching original bindings. The SDK version selects the ABI, not a package name.
