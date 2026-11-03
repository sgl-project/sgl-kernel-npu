"""Public mhc_post operator with registered implementations."""

from ..utils._cann_ops_transformer import get_op
from ..utils.registry import get_impl_by_name, register_impl, register_operator


@register_impl("norm.mhc_post", "auto")
@register_impl("norm.mhc_post", "cann_ops_transformer")
def _cann_mhc_post(x, residual, post, comb):
    return get_op("mhc_post")(residual, comb, x, post)


@register_impl("norm.mhc_post", "torch_native")
def _torch_mhc_post(x, residual, post, comb):
    from ...implementation.torch_native.norm.mhc_post import mhc_post

    return mhc_post(x, residual, post, comb)


@register_operator("norm.mhc_post")
def mhc_post(x, residual, post, comb, *, impl="auto"):
    """Mix the layer output and residual streams, preserving the output dtype.

    impl: auto (released CANN), cann_ops_transformer, or torch_native.
    torch_native is an eager correctness reference, not a graph fallback.
    """
    return get_impl_by_name("norm.mhc_post", impl)(x, residual, post, comb)
