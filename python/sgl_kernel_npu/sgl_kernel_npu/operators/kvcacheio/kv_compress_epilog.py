"""Public kv_compress_epilog operator with registered implementations."""

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator


@register_impl(
    "kvcacheio.kv_compress_epilog", "auto", supported_generations={Generation.ARCH35}
)
@register_impl(
    "kvcacheio.kv_compress_epilog",
    "cann_ops_transformer",
    supported_generations={Generation.ARCH35},
)
def _cann_kv_compress_epilog(cache, x, slot_mapping, **kwargs):
    from ...implementation.cann_ops_transformer.kvcacheio.kv_compress_epilog import (
        kv_compress_epilog,
    )

    return kv_compress_epilog(cache, x, slot_mapping, **kwargs)


@register_impl(
    "kvcacheio.kv_compress_epilog",
    "torch_native",
    supported_generations={Generation.ARCH35},
)
def _torch_kv_compress_epilog(cache, x, slot_mapping, **kwargs):
    from ...implementation.torch_native.kvcacheio.kv_compress_epilog import (
        kv_compress_epilog,
    )

    return kv_compress_epilog(cache, x, slot_mapping, **kwargs)


@register_operator("kvcacheio.kv_compress_epilog")
def kv_compress_epilog(cache, x, slot_mapping, *, impl="auto", **kwargs):
    """Quantize and scatter A5 KV rows in place; -1 slots are skipped.

    impl: auto (released CANN), cann_ops_transformer, or torch_native.
    torch_native is an eager correctness reference, not a graph fallback.
    """
    return get_impl_by_name("kvcacheio.kv_compress_epilog", impl)(
        cache, x, slot_mapping, **kwargs
    )
