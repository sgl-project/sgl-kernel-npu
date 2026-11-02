"""DSV4 mhc operators with explicit implementation selection."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator


@register_impl("dsv4.mhc_pre", "auto")
@register_impl("dsv4.mhc_pre", "cann_ops_transformer")
def _cann_mhc_pre(x, fn, scale, base, hc_mult, sinkhorn_iters, hc_eps, rms_norm_eps):
    from ...implementation.cann_ops_transformer.dsv4 import mhc_pre

    return mhc_pre(x, fn, scale, base, hc_mult, sinkhorn_iters, hc_eps, rms_norm_eps)


@register_impl("dsv4.mhc_pre", "torch_native")
def _torch_mhc_pre(x, fn, scale, base, hc_mult, sinkhorn_iters, hc_eps, rms_norm_eps):
    from ...implementation.torch_native.dsv4.mhc import mhc_pre

    return mhc_pre(x, fn, scale, base, hc_mult, sinkhorn_iters, hc_eps, rms_norm_eps)


@register_operator("dsv4.mhc_pre")
def mhc_pre(
    x, fn, scale, base, hc_mult, sinkhorn_iters, hc_eps, rms_norm_eps, *, impl="auto"
):
    """Return mixed activation, post weights and residual mixing matrix; input layernorm remains separate.

    impl: auto (released CANN), cann_ops_transformer, or torch_native.
    torch_native is an eager correctness reference, not a graph fallback.
    """
    return get_impl_by_name("dsv4.mhc_pre", impl)(
        x, fn, scale, base, hc_mult, sinkhorn_iters, hc_eps, rms_norm_eps
    )


@register_impl("dsv4.mhc_post", "auto")
@register_impl("dsv4.mhc_post", "cann_ops_transformer")
def _cann_mhc_post(x, residual, post, comb):
    from ...implementation.cann_ops_transformer.dsv4 import mhc_post

    return mhc_post(x, residual, post, comb)


@register_impl("dsv4.mhc_post", "torch_native")
def _torch_mhc_post(x, residual, post, comb):
    from ...implementation.torch_native.dsv4.mhc import mhc_post

    return mhc_post(x, residual, post, comb)


@register_operator("dsv4.mhc_post")
def mhc_post(x, residual, post, comb, *, impl="auto"):
    """Mix the layer output and residual streams, preserving the output dtype.

    impl: auto (released CANN), cann_ops_transformer, or torch_native.
    torch_native is an eager correctness reference, not a graph fallback.
    """
    return get_impl_by_name("dsv4.mhc_post", impl)(x, residual, post, comb)
