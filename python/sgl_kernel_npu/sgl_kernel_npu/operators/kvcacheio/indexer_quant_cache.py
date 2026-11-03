"""Public indexer_quant_cache operator with registered implementations."""

from ..utils._cann_ops_transformer import get_op
from ..utils.registry import get_impl_by_name, register_impl, register_operator


@register_impl("kvcacheio.indexer_quant_cache", "auto")
@register_impl("kvcacheio.indexer_quant_cache", "cann_ops_transformer")
def _cann_indexer_quant_cache(cache, scale_cache, x, slot_mapping, **kwargs):
    get_op("indexer_quant_cache")(cache, scale_cache, x, slot_mapping, **kwargs)
    return cache, scale_cache


@register_impl("kvcacheio.indexer_quant_cache", "torch_native")
def _torch_indexer_quant_cache(cache, scale_cache, x, slot_mapping, **kwargs):
    from ...implementation.torch_native.kvcacheio.indexer_quant_cache import (
        indexer_quant_cache,
    )

    return indexer_quant_cache(cache, scale_cache, x, slot_mapping, **kwargs)


@register_operator("kvcacheio.indexer_quant_cache")
def indexer_quant_cache(cache, scale_cache, x, slot_mapping, *, impl="auto", **kwargs):
    """Quantize and scatter indexer keys and fp32 scales in place.

    impl: auto (released CANN), cann_ops_transformer, or torch_native.
    torch_native is an eager correctness reference, not a graph fallback.
    """
    return get_impl_by_name("kvcacheio.indexer_quant_cache", impl)(
        cache, scale_cache, x, slot_mapping, **kwargs
    )
