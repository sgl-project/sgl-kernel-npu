"""Inference mHC projection and Sinkhorn normalization."""

import torch

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_cann_fn = None
_torch_native_fn = None


@register_impl("hyper_conn.mhc_pre_sinkhorn", "auto")
@register_impl("hyper_conn.mhc_pre_sinkhorn", "cann_ops_transformer")
def _cann_impl(
    x, phi, alpha, bias, hc_mult, num_iters, hc_eps, norm_eps, *, return_pre=False
):
    global _cann_fn
    if _cann_fn is None:
        from ..._third_party.sgl_ops_transformer.ops import mhc_pre_sinkhorn

        _cann_fn = mhc_pre_sinkhorn
    args = (x.unsqueeze(1), phi, alpha, bias, hc_mult, num_iters, hc_eps, norm_eps)
    if return_pre:
        # The upstream inference facade drops hPre. The raw binding exposes it
        # with the optional outputs enabled, without running a backward pass.
        y, post, comb, pre, *_ = torch.ops.cann_ops_transformer.mhc_pre_sinkhorn(
            *args, True
        )
        return y.squeeze(1), post.squeeze(1), comb.squeeze(1), pre.squeeze(1)
    y, post, comb = _cann_fn(*args)
    return y.squeeze(1), post.squeeze(1), comb.squeeze(1)


@register_impl("hyper_conn.mhc_pre_sinkhorn", "torch_native")
def _torch_native_impl(*args, return_pre=False):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.hyper_conn.mhc_pre_sinkhorn import (
            mhc_pre_sinkhorn,
        )

        _torch_native_fn = mhc_pre_sinkhorn
    return _torch_native_fn(*args, return_pre=return_pre)


@register_operator("hyper_conn.mhc_pre_sinkhorn")
def mhc_pre_sinkhorn(
    x,
    phi,
    alpha,
    bias,
    hc_mult=4,
    num_iters=20,
    hc_eps=1e-6,
    norm_eps=1e-6,
    *,
    return_pre=False,
    impl="auto",
):
    """Project [T,N,D] streams through the bundled CANN binding by default.

    phi has shape [(2+N)*N,N*D], alpha [3], and bias [(2+N)*N].
    Returns (hin [T,D], hpost [T,N], hres [T,N*N]), preserving vendor
    output dtypes. When return_pre is true, append hpre [T,N] in FP32.
    Input constraints are delegated to the selected backend.
    No output layer normalization is fused.
    """
    impl_fn = get_impl_by_name("hyper_conn.mhc_pre_sinkhorn", impl)
    if return_pre:
        return impl_fn(
            x, phi, alpha, bias, hc_mult, num_iters, hc_eps, norm_eps, return_pre=True
        )
    return impl_fn(x, phi, alpha, bias, hc_mult, num_iters, hc_eps, norm_eps)
