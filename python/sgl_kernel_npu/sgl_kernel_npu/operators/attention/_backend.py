"""Choose paired attention bindings before KV allocation or graph capture.

Metadata, readers and packed-KV writers must share one backend selection.
Operator-specific input conversions belong to their public operator modules.
"""

import functools

from ..utils._transformer_compat import (
    OperatorFamily,
    OperatorUnavailableError,
    _lookup,
    select_family,
)


@functools.lru_cache(maxsize=None)
def attention_family(quantized):
    prefix = "mixed_quant_sparse_flash_mla" if quantized else "sparse_flash_mla"
    old = (
        "npu_kv_quant_sparse_attn_sharedkv" if quantized else "npu_sparse_attn_sharedkv"
    )
    released = (
        ("cann_ops_transformer", prefix + "_metadata"),
        ("cann_ops_transformer", prefix),
    )
    legacy = (("custom", old + "_metadata"), ("custom", old))
    if quantized:
        # Reader, metadata and writer must agree on 608B/BF16-scale versus
        # 640B/E8M0-scale encoding before any cache is allocated.
        released += (("cann_ops_transformer", "kv_compress_epilog"),)
        legacy += (("npu", "kv_compress_epilog"),)
    return select_family(released, legacy)


@functools.lru_cache(maxsize=None)
def sparse_host_family(quantized):
    stem = "kv_quant_sparse_attn_sharedkv" if quantized else "sparse_attn_sharedkv"
    specs = (("npu", stem + "_metadata_host"), ("npu", stem))
    ops = _lookup(specs)
    if any(op is None for op in ops):
        raise OperatorUnavailableError(
            "Legacy explicit sparse attention requires the paired sgl-kernel-npu "
            f"host metadata and compute bindings: {specs}"
        )
    return OperatorFamily("legacy", ops, specs)


@functools.lru_cache(maxsize=1)
def indexer_family():
    return select_family(
        (
            ("cann_ops_transformer", "quant_lightning_indexer_metadata"),
            ("cann_ops_transformer", "quant_lightning_indexer"),
        ),
        (
            ("custom", "npu_quant_lightning_indexer_metadata"),
            ("custom", "npu_quant_lightning_indexer"),
        ),
    )
