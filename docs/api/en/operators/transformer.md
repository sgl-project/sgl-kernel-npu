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

Shared implementation helpers use descriptive module names. Eager attention math is in `implementation/torch_native/attention/attention.py`; metadata descriptors are in `metadata.py`. Binding selection and schema guards are in `operators/utils/transformer_compat.py`, and SDK release parsing is in `operators/utils/cann.py`. Attention pairing and sparse-index conversion share `operators/attention/attention_utils.py`.

The indexer registers separate `cann_9_1` and `cann_ops_transformer` implementations; `auto` selects by initialized SDK. The other migrated production names remain compatible. Packed KV still requires a paired reader, metadata and writer because CANN 9.1 and 9.2 use different storage encodings. No execution failure retries another backend.
