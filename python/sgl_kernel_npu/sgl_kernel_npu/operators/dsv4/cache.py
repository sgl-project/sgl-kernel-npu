"""DSV4 cache operators with explicit implementation selection."""

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator


@register_impl(
    "dsv4.kv_compress_epilog", "auto", supported_generations={Generation.ARCH35}
)
@register_impl(
    "dsv4.kv_compress_epilog",
    "cann_ops_transformer",
    supported_generations={Generation.ARCH35},
)
def _cann_kv_compress_epilog(cache, x, slot_mapping, **kwargs):
    from ...implementation.cann_ops_transformer.dsv4 import kv_compress_epilog

    return kv_compress_epilog(cache, x, slot_mapping, **kwargs)


@register_impl(
    "dsv4.kv_compress_epilog", "torch_native", supported_generations={Generation.ARCH35}
)
def _torch_kv_compress_epilog(cache, x, slot_mapping, **kwargs):
    from ...implementation.torch_native.dsv4.cache import kv_compress_epilog

    return kv_compress_epilog(cache, x, slot_mapping, **kwargs)


@register_operator("dsv4.kv_compress_epilog")
def kv_compress_epilog(cache, x, slot_mapping, *, impl="auto", **kwargs):
    """Quantize and scatter A5 KV rows in place; -1 slots are skipped.

    impl: auto (released CANN), cann_ops_transformer, or torch_native.
    torch_native is an eager correctness reference, not a graph fallback.
    """
    return get_impl_by_name("dsv4.kv_compress_epilog", impl)(
        cache, x, slot_mapping, **kwargs
    )


@register_impl("dsv4.indexer_quant_cache", "auto")
@register_impl("dsv4.indexer_quant_cache", "cann_ops_transformer")
def _cann_indexer_quant_cache(cache, scale_cache, x, slot_mapping, **kwargs):
    from ...implementation.cann_ops_transformer.dsv4 import indexer_quant_cache

    return indexer_quant_cache(cache, scale_cache, x, slot_mapping, **kwargs)


@register_impl("dsv4.indexer_quant_cache", "torch_native")
def _torch_indexer_quant_cache(cache, scale_cache, x, slot_mapping, **kwargs):
    from ...implementation.torch_native.dsv4.cache import indexer_quant_cache

    return indexer_quant_cache(cache, scale_cache, x, slot_mapping, **kwargs)


@register_operator("dsv4.indexer_quant_cache")
def indexer_quant_cache(cache, scale_cache, x, slot_mapping, *, impl="auto", **kwargs):
    """Quantize and scatter indexer keys and fp32 scales in place.

    impl: auto (released CANN), cann_ops_transformer, or torch_native.
    torch_native is an eager correctness reference, not a graph fallback.
    """
    return get_impl_by_name("dsv4.indexer_quant_cache", impl)(
        cache, scale_cache, x, slot_mapping, **kwargs
    )
