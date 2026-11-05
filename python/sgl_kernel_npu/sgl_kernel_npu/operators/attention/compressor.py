"""Public compressor operator with registered implementations."""

from ..utils.generation import Generation, get_hardware
from ..utils.registry import get_impl_by_name, register_impl, register_operator
from ..utils.transformer_compat import select_family


@register_impl("attention.compressor", "auto")
@register_impl("attention.compressor", "cann_ops_transformer")
def _cann_compressor(x, wkv, wgate, state_cache, ape, norm_weight, **kwargs):
    # Keep the fused RMSNorm/RoPE contract: current native binding on A3/A5,
    # original recipe binding on legacy A5. The released core-only compressor
    # has a different computation boundary and is not an interchangeable call.
    legacy_namespace = (
        "custom" if get_hardware().generation == Generation.ARCH35 else "npu"
    )
    family = select_family(
        (("npu", "compressor"),),
        ((legacy_namespace, "compressor"),),
    )
    return family.ops[0](x, wkv, wgate, state_cache, ape, norm_weight, **kwargs)


@register_impl("attention.compressor", "torch_native")
def _torch_compressor(x, wkv, wgate, state_cache, ape, norm_weight, **kwargs):
    from ...implementation.torch_native.attention.compressor import compressor

    return compressor(x, wkv, wgate, state_cache, ape, norm_weight, **kwargs)


@register_operator("attention.compressor")
def compressor(x, wkv, wgate, state_cache, ape, norm_weight, *, impl="auto", **kwargs):
    """Compress KV and update the compressor state cache in place.

    impl: auto (released/legacy compatibility), cann_ops_transformer, or torch_native.
    torch_native is an eager correctness reference, not a graph fallback.
    """
    return get_impl_by_name("attention.compressor", impl)(
        x, wkv, wgate, state_cache, ape, norm_weight, **kwargs
    )
