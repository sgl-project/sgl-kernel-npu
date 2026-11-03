"""Public indexer_quant_cache operator with registered implementations."""

from ..utils._transformer_compat import OperatorUnavailableError, select_family
from ..utils.registry import get_impl_by_name, register_impl, register_operator


@register_impl("kvcacheio.indexer_quant_cache", "auto")
@register_impl("kvcacheio.indexer_quant_cache", "cann_ops_transformer")
def _cann_indexer_quant_cache(cache, scale_cache, x, slot_mapping, **kwargs):
    family = select_family(
        (("cann_ops_transformer", "indexer_quant_cache"),),
        (("custom", "indexer_compress_epilog"),),
    )
    if family.legacy:
        if (
            kwargs.get("quant_mode", "fp8") != "fp8"
            or kwargs.get("x_scale", 1.0) != 1.0
            or not kwargs.get("round_scale", True)
        ):
            raise OperatorUnavailableError(
                "Legacy indexer epilog supports its original fp8 dynamic-quantization contract only"
            )
        family.ops[0](
            indexer_compress_cache=cache,
            indexer_compress_scale=scale_cache,
            x=x,
            slot_mapping=slot_mapping,
        )
    else:
        family.ops[0](cache, scale_cache, x, slot_mapping, **kwargs)
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

    impl: auto (released/legacy compatibility), cann_ops_transformer, or torch_native.
    torch_native is an eager correctness reference, not a graph fallback.
    """
    return get_impl_by_name("kvcacheio.indexer_quant_cache", impl)(
        cache, scale_cache, x, slot_mapping, **kwargs
    )
