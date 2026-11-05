"""Public mhc_post operator with registered implementations."""

from ..utils.generation import Generation, get_hardware
from ..utils.registry import get_impl_by_name, register_impl, register_operator
from ..utils.transformer_compat import cann_backend, require_op


@register_impl("norm.mhc_post", "auto")
def _auto_mhc_post(x, residual, post, comb):
    return get_impl_by_name("norm.mhc_post", cann_backend())(x, residual, post, comb)


@register_impl("norm.mhc_post", "cann_ops_transformer_cann920")
def _cann_mhc_post(x, residual, post, comb):
    op = require_op(
        "cann_ops_transformer", "mhc_post", backend="cann_ops_transformer_cann920"
    )
    return op(residual, comb, x, post)


@register_impl("norm.mhc_post", "cann_9_1")
def _cann_9_1_mhc_post(x, residual, post, comb):
    op = require_op("custom", "npu_hc_post", backend="cann_9_1")
    if get_hardware().generation == Generation.ARCH35:
        return op(
            x.unsqueeze(0), residual.unsqueeze(0), post.unsqueeze(0), comb.unsqueeze(0)
        ).squeeze(0)
    return op(x, residual, post, comb)


@register_impl("norm.mhc_post", "torch_native")
def _torch_mhc_post(x, residual, post, comb):
    from ...implementation.torch_native.norm.mhc_post import mhc_post

    return mhc_post(x, residual, post, comb)


@register_operator("norm.mhc_post")
def mhc_post(x, residual, post, comb, *, impl="auto"):
    """Mix the layer output and residual streams, preserving the output dtype.

    impl: auto (SDK-selected), cann_9_1, cann_ops_transformer_cann920, or torch_native.
    torch_native is an eager correctness reference, not a graph fallback.
    """
    return get_impl_by_name("norm.mhc_post", impl)(x, residual, post, comb)
