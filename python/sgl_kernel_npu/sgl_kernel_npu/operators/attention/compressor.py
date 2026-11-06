"""Public compressor operator with registered implementations."""

from ..utils.generation import Generation, get_hardware
from ..utils.registry import get_impl_by_name, register_impl, register_operator
from ..utils.transformer_compat import cann_backend, require_op


@register_impl("attention.compressor", "auto")
def _auto_compressor(x, wkv, wgate, state_cache, ape, norm_weight, **kwargs):
    backend = cann_backend()
    if backend == "cann_9_1":
        backend = (
            "ops_transformer_cann910"
            if get_hardware().generation == Generation.ARCH35
            else "ascendc"
        )
    return get_impl_by_name("attention.compressor", backend)(
        x, wkv, wgate, state_cache, ape, norm_weight, **kwargs
    )


@register_impl("attention.compressor", "cann_ops_transformer_cann920")
def _cann_compressor(x, wkv, wgate, state_cache, ape, norm_weight, **kwargs):
    # Preserve the repository's fused RMSNorm/RoPE and state-cache contract.
    op = require_op("npu", "compressor", backend="cann_ops_transformer_cann920")
    return op(x, wkv, wgate, state_cache, ape, norm_weight, **kwargs)


@register_impl(
    "attention.compressor",
    "ops_transformer_cann910",
    supported_generations={Generation.ARCH35},
)
def _ops_transformer_cann910(x, wkv, wgate, state_cache, ape, norm_weight, **kwargs):
    op = require_op("custom", "compressor", backend="cann_9_1")
    return op(x, wkv, wgate, state_cache, ape, norm_weight, **kwargs)


@register_impl(
    "attention.compressor", "ascendc", supported_generations={Generation.ARCH22}
)
def _ascendc(x, wkv, wgate, state_cache, ape, norm_weight, **kwargs):
    op = require_op("npu", "compressor", backend="cann_9_1")
    return op(x, wkv, wgate, state_cache, ape, norm_weight, **kwargs)


@register_impl("attention.compressor", "torch_native")
def _torch_compressor(x, wkv, wgate, state_cache, ape, norm_weight, **kwargs):
    from ...implementation.torch_native.attention.compressor import compressor

    return compressor(x, wkv, wgate, state_cache, ape, norm_weight, **kwargs)


@register_operator("attention.compressor")
def compressor(x, wkv, wgate, state_cache, ape, norm_weight, *, impl="auto", **kwargs):
    """Compress KV and update the compressor state cache in place.

    impl: auto (SDK-selected), ops_transformer_cann910 (A5), ascendc (A3), cann_ops_transformer_cann920, or torch_native.
    torch_native is an eager correctness reference, not a graph fallback.
    """
    return get_impl_by_name("attention.compressor", impl)(
        x, wkv, wgate, state_cache, ape, norm_weight, **kwargs
    )
