"""Public compressor operator with registered implementations."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator


@register_impl("attention.compressor", "auto")
@register_impl("attention.compressor", "cann_ops_transformer")
def _cann_compressor(x, wkv, wgate, state_cache, ape, norm_weight, **kwargs):
    from ...implementation.cann_ops_transformer.attention.compressor import compressor

    return compressor(x, wkv, wgate, state_cache, ape, norm_weight, **kwargs)


@register_impl("attention.compressor", "torch_native")
def _torch_compressor(x, wkv, wgate, state_cache, ape, norm_weight, **kwargs):
    from ...implementation.torch_native.attention.compressor import compressor

    return compressor(x, wkv, wgate, state_cache, ape, norm_weight, **kwargs)


@register_operator("attention.compressor")
def compressor(x, wkv, wgate, state_cache, ape, norm_weight, *, impl="auto", **kwargs):
    """Compress KV and update the compressor state cache in place.

    impl: auto (released CANN), cann_ops_transformer, or torch_native.
    torch_native is an eager correctness reference, not a graph fallback.
    """
    return get_impl_by_name("attention.compressor", impl)(
        x, wkv, wgate, state_cache, ape, norm_weight, **kwargs
    )
