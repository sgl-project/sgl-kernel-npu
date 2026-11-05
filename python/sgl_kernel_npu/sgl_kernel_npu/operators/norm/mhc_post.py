"""Public mhc_post operator with registered implementations."""

from ..utils.generation import Generation, get_hardware
from ..utils.registry import get_impl_by_name, register_impl, register_operator
from ..utils.transformer_compat import select_family


@register_impl("norm.mhc_post", "auto")
@register_impl("norm.mhc_post", "cann_ops_transformer")
def _cann_mhc_post(x, residual, post, comb):
    family = select_family(
        (("cann_ops_transformer", "mhc_post"),), (("custom", "npu_hc_post"),)
    )
    if not family.cann_9_1:
        return family.ops[0](residual, comb, x, post)
    if get_hardware().generation == Generation.ARCH35:
        return family.ops[0](
            x.unsqueeze(0), residual.unsqueeze(0), post.unsqueeze(0), comb.unsqueeze(0)
        ).squeeze(0)
    return family.ops[0](x, residual, post, comb)


@register_impl("norm.mhc_post", "torch_native")
def _torch_mhc_post(x, residual, post, comb):
    from ...implementation.torch_native.norm.mhc_post import mhc_post

    return mhc_post(x, residual, post, comb)


@register_operator("norm.mhc_post")
def mhc_post(x, residual, post, comb, *, impl="auto"):
    """Mix the layer output and residual streams, preserving the output dtype.

    impl: auto (released/cann_9_1 compatibility), cann_ops_transformer, or torch_native.
    torch_native is an eager correctness reference, not a graph fallback.
    """
    return get_impl_by_name("norm.mhc_post", impl)(x, residual, post, comb)
