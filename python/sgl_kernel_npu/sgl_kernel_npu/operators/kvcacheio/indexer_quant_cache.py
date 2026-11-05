"""Public indexer_quant_cache operator with registered implementations."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator
from ..utils.transformer_compat import (
    OperatorUnavailableError,
    cann_backend,
    select_family,
)


@register_impl("kvcacheio.indexer_quant_cache", "auto")
@register_impl("kvcacheio.indexer_quant_cache", "cann_ops_transformer")
def _cann_indexer_quant_cache(cache, scale_cache, x, slot_mapping, **kwargs):
    family = select_family(
        (("cann_ops_transformer", "indexer_quant_cache"),),
        (("custom", "indexer_compress_epilog"),),
    )
    if family.cann_9_1:
        if kwargs.get("quant_mode", "fp8") != "fp8":
            raise OperatorUnavailableError(
                "cann_9_1 indexer epilog supports its original fp8 dynamic-quantization contract only"
            )
        # Normal FP8 uses FP32 block scales. round_scale is MX-FP8-only,
        # and x_scale is HiFloat8-only; both are inactive in this mode.
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

    return indexer_quant_cache(
        cache,
        scale_cache,
        x,
        slot_mapping,
        _cann_9_1=cann_backend() == "cann_9_1",
        **kwargs,
    )


@register_operator("kvcacheio.indexer_quant_cache")
def indexer_quant_cache(cache, scale_cache, x, slot_mapping, *, impl="auto", **kwargs):
    """Quantize and scatter indexer keys and fp32 scales in place.

    For quant_mode="fp8", the original 9.1 binding uses unrounded FP32 scales;
    the released 9.2 binding also honors round_scale. x_scale is HiFloat8-only.
    impl: auto (released/cann_9_1 compatibility), cann_ops_transformer, or torch_native.
    torch_native is an eager correctness reference, not a graph fallback.
    """
    return get_impl_by_name("kvcacheio.indexer_quant_cache", impl)(
        cache, scale_cache, x, slot_mapping, **kwargs
    )
