"""Public indexer_quant_cache operator with registered implementations."""

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator
from ..utils.transformer_compat import (
    OperatorUnavailableError,
    cann_backend,
    require_op,
)


@register_impl(
    "mem_cache.indexer_quant_cache", "auto", supported_generations={Generation.ARCH35}
)
def _auto_indexer_quant_cache(cache, scale_cache, x, slot_mapping, **kwargs):
    backend = cann_backend()
    if backend == "cann_9_1":
        backend = "cann_recipes_infer_cann910"
    return get_impl_by_name("mem_cache.indexer_quant_cache", backend)(
        cache, scale_cache, x, slot_mapping, **kwargs
    )


@register_impl(
    "mem_cache.indexer_quant_cache",
    "cann_ops_transformer_cann920",
    supported_generations={Generation.ARCH35},
)
def _cann_indexer_quant_cache(cache, scale_cache, x, slot_mapping, **kwargs):
    op = require_op(
        "cann_ops_transformer",
        "indexer_quant_cache",
        backend="cann_ops_transformer_cann920",
    )
    op(cache, scale_cache, x, slot_mapping, **kwargs)
    return cache, scale_cache


@register_impl(
    "mem_cache.indexer_quant_cache",
    "cann_recipes_infer_cann910",
    supported_generations={Generation.ARCH35},
)
def _cann_9_1_indexer_quant_cache(cache, scale_cache, x, slot_mapping, **kwargs):
    op = require_op("custom", "indexer_compress_epilog", backend="cann_9_1")
    if kwargs.get("quant_mode", "fp8") != "fp8":
        raise OperatorUnavailableError(
            "cann_9_1 indexer epilog supports its original fp8 dynamic-quantization contract only"
        )
    # Normal FP8 uses FP32 block scales. round_scale is MX-FP8-only,
    # and x_scale is HiFloat8-only; both are inactive in this mode.
    op(
        indexer_compress_cache=cache,
        indexer_compress_scale=scale_cache,
        x=x,
        slot_mapping=slot_mapping,
    )
    return cache, scale_cache


@register_impl("mem_cache.indexer_quant_cache", "torch_native")
def _torch_indexer_quant_cache(cache, scale_cache, x, slot_mapping, **kwargs):
    from ...implementation.torch_native.mem_cache.indexer_quant_cache import (
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


@register_operator("mem_cache.indexer_quant_cache")
def indexer_quant_cache(cache, scale_cache, x, slot_mapping, *, impl="auto", **kwargs):
    """Quantize and scatter indexer keys and fp32 scales in place.

    Production implementations support A5/ARCH35; the eager reference is independent.

    For quant_mode="fp8", the original 9.1 binding uses unrounded FP32 scales;
    the released 9.2 binding also honors round_scale. x_scale is HiFloat8-only.
    impl: auto (SDK-selected), cann_recipes_infer_cann910, cann_ops_transformer_cann920, or torch_native.
    torch_native is an eager correctness reference, not a graph fallback.
    """
    return get_impl_by_name("mem_cache.indexer_quant_cache", impl)(
        cache, scale_cache, x, slot_mapping, **kwargs
    )
