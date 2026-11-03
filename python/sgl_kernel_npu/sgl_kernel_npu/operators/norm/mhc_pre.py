"""Public mhc_pre operator with registered implementations."""

from ..utils._transformer_compat import select_family
from ..utils.registry import get_impl_by_name, register_impl, register_operator


@register_impl("norm.mhc_pre", "auto")
@register_impl("norm.mhc_pre", "cann_ops_transformer")
def _cann_mhc_pre(x, fn, scale, base, hc_mult, sinkhorn_iters, hc_eps, rms_norm_eps):
    family = select_family(
        (("cann_ops_transformer", "mhc_pre_sinkhorn"),),
        (("custom", "npu_hc_pre"),),
    )
    shape = x.shape
    if family.legacy:
        y, post, comb = family.ops[0](
            x,
            fn,
            scale,
            base,
            hc_mult=hc_mult,
            hc_sinkhorn_iters=sinkhorn_iters,
            norm_eps=rms_norm_eps,
            hc_eps=hc_eps,
        )
        return y.to(x.dtype), post, comb
    h_in, h_post, h_res, *_ = family.ops[0](
        x.reshape(-1, 1, hc_mult, shape[-1]),
        fn,
        scale,
        base,
        hc_mult,
        sinkhorn_iters,
        hc_eps,
        rms_norm_eps,
        False,
    )
    return (
        h_in.reshape(-1, shape[-1]).to(x.dtype),
        h_post.reshape(-1, hc_mult),
        h_res.reshape(-1, hc_mult, hc_mult),
    )


@register_impl("norm.mhc_pre", "torch_native")
def _torch_mhc_pre(x, fn, scale, base, hc_mult, sinkhorn_iters, hc_eps, rms_norm_eps):
    from ...implementation.torch_native.norm.mhc_pre import mhc_pre

    return mhc_pre(x, fn, scale, base, hc_mult, sinkhorn_iters, hc_eps, rms_norm_eps)


@register_operator("norm.mhc_pre")
def mhc_pre(
    x, fn, scale, base, hc_mult, sinkhorn_iters, hc_eps, rms_norm_eps, *, impl="auto"
):
    """Return mixed activation, post weights and residual mixing matrix; input layernorm remains separate.

    impl: auto (released/legacy compatibility), cann_ops_transformer, or torch_native.
    torch_native is an eager correctness reference, not a graph fallback.
    """
    return get_impl_by_name("norm.mhc_pre", impl)(
        x, fn, scale, base, hc_mult, sinkhorn_iters, hc_eps, rms_norm_eps
    )
