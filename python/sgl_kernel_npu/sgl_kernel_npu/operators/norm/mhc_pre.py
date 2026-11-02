"""Public mhc_pre operator with registered implementations."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator


@register_impl("norm.mhc_pre", "auto")
@register_impl("norm.mhc_pre", "cann_ops_transformer")
def _cann_mhc_pre(x, fn, scale, base, hc_mult, sinkhorn_iters, hc_eps, rms_norm_eps):
    from ...implementation.cann_ops_transformer.norm.mhc_pre import mhc_pre

    return mhc_pre(x, fn, scale, base, hc_mult, sinkhorn_iters, hc_eps, rms_norm_eps)


@register_impl("norm.mhc_pre", "torch_native")
def _torch_mhc_pre(x, fn, scale, base, hc_mult, sinkhorn_iters, hc_eps, rms_norm_eps):
    from ...implementation.torch_native.norm.mhc_pre import mhc_pre

    return mhc_pre(x, fn, scale, base, hc_mult, sinkhorn_iters, hc_eps, rms_norm_eps)


@register_operator("norm.mhc_pre")
def mhc_pre(
    x, fn, scale, base, hc_mult, sinkhorn_iters, hc_eps, rms_norm_eps, *, impl="auto"
):
    """Return mixed activation, post weights and residual mixing matrix; input layernorm remains separate.

    impl: auto (released CANN), cann_ops_transformer, or torch_native.
    torch_native is an eager correctness reference, not a graph fallback.
    """
    return get_impl_by_name("norm.mhc_pre", impl)(
        x, fn, scale, base, hc_mult, sinkhorn_iters, hc_eps, rms_norm_eps
    )
