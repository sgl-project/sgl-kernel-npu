"""Public indexer_quant_cache operator with registered implementations."""

import torch

from ..utils.cann import detect_cann_version
from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator

_cann_920_fn = None
_cann_910_fn = None
_torch_native_fn = None


@register_impl(
    "mem_cache.indexer_quant_cache", "auto", supported_generations={Generation.ARCH35}
)
def _auto_indexer_quant_cache(cache, scale_cache, x, slot_mapping, **kwargs):
    cann_version = detect_cann_version()
    if cann_version.startswith("9.1."):
        backend = "cann_recipes_infer_cann910"
    elif cann_version.startswith("9.2."):
        backend = "cann_ops_transformers_cann920"
    else:
        raise RuntimeError(
            f"Unsupported CANN release {cann_version}; expected CANN 9.1.x or 9.2.x"
        )
    return get_impl_by_name("mem_cache.indexer_quant_cache", backend)(
        cache, scale_cache, x, slot_mapping, **kwargs
    )


@register_impl(
    "mem_cache.indexer_quant_cache",
    "cann_ops_transformers_cann920",
    supported_generations={Generation.ARCH35},
)
def _cann_indexer_quant_cache(cache, scale_cache, x, slot_mapping, **kwargs):
    global _cann_920_fn
    cann_version = detect_cann_version()
    if not cann_version.startswith("9.2."):
        raise RuntimeError(
            f"This implementation requires CANN 9.2.x, got {cann_version}"
        )
    if _cann_920_fn is None:
        from ..._third_party.sgl_ops_transformer_dsv4 import ops

        _cann_920_fn = ops.indexer_quant_cache

    _cann_920_fn(cache, scale_cache, x, slot_mapping, **kwargs)
    return cache, scale_cache


@register_impl(
    "mem_cache.indexer_quant_cache",
    "cann_recipes_infer_cann910",
    supported_generations={Generation.ARCH35},
)
def _cann_9_1_indexer_quant_cache(cache, scale_cache, x, slot_mapping, **kwargs):
    global _cann_910_fn
    cann_version = detect_cann_version()
    if not cann_version.startswith("9.1."):
        raise RuntimeError(
            f"This implementation requires CANN 9.1.x, got {cann_version}"
        )
    if _cann_910_fn is None:
        from ..._third_party.sgl_cann_recipes_infer import indexer_compress_epilog

        _cann_910_fn = indexer_compress_epilog

    if kwargs.get("quant_mode", "fp8") != "fp8":
        raise RuntimeError(
            "cann_9_1 indexer epilog supports its original fp8 dynamic-quantization contract only"
        )
    # Normal FP8 uses FP32 block scales. round_scale is MX-FP8-only,
    # and x_scale is HiFloat8-only; both are inactive in this mode.
    _cann_910_fn(
        indexer_compress_cache=cache,
        indexer_compress_scale=scale_cache,
        x=x,
        slot_mapping=slot_mapping,
    )
    return cache, scale_cache


@register_impl("mem_cache.indexer_quant_cache", "torch_native")
def _torch_indexer_quant_cache(cache, scale_cache, x, slot_mapping, **kwargs):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.mem_cache.indexer_quant_cache import (
            indexer_quant_cache,
        )

        _torch_native_fn = indexer_quant_cache
    cann_version = detect_cann_version()
    if not cann_version.startswith(("9.1.", "9.2.")):
        raise RuntimeError(
            f"Unsupported CANN release {cann_version}; expected CANN 9.1.x or 9.2.x"
        )
    return _torch_native_fn(
        cache,
        scale_cache,
        x,
        slot_mapping,
        _cann_9_1=cann_version.startswith("9.1."),
        **kwargs,
    )


@register_operator("mem_cache.indexer_quant_cache")
def indexer_quant_cache(cache, scale_cache, x, slot_mapping, *, impl="auto", **kwargs):
    """Quantize and scatter indexer keys and fp32 scales in place.

    Production implementations support A5/ARCH35; the eager reference is independent.

    For quant_mode="fp8", the original 9.1 binding uses unrounded FP32 scales;
    the released 9.2 binding also honors round_scale. x_scale is HiFloat8-only.
    impl: auto (SDK-selected), cann_recipes_infer_cann910, cann_ops_transformers_cann920, or torch_native.
    torch_native is an eager correctness reference, not a graph fallback.
    """
    return get_impl_by_name("mem_cache.indexer_quant_cache", impl)(
        cache, scale_cache, x, slot_mapping, **kwargs
    )
