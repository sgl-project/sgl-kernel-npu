"""Inference mHC projection and Sinkhorn normalization."""

import torch

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_cann_fn = None
_torch_native_fn = None


@register_impl("hyper_conn.mhc_pre_sinkhorn", "auto")
@register_impl("hyper_conn.mhc_pre_sinkhorn", "cann")
def _cann_impl(x, phi, alpha, bias, hc_mult, num_iters, hc_eps, norm_eps):
    global _cann_fn
    if _cann_fn is None:
        from cann_ops_transformer.ops import mhc_pre_sinkhorn

        _cann_fn = mhc_pre_sinkhorn
    # CANN 9.1's wrapper requires BSND, even for packed tokens.
    y, post, comb = _cann_fn(
        x.unsqueeze(1), phi, alpha, bias, hc_mult, num_iters, hc_eps, norm_eps
    )
    return y.squeeze(1), post.squeeze(1), comb.squeeze(1)


@register_impl("hyper_conn.mhc_pre_sinkhorn", "torch_native")
def _torch_native_impl(*args):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.hyper_conn.mhc_pre_sinkhorn import (
            mhc_pre_sinkhorn,
        )

        _torch_native_fn = mhc_pre_sinkhorn
    return _torch_native_fn(*args)


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
    impl="auto",
):
    """Project [T,N,D] streams; auto selects CANN 9.1's public API.

    phi has shape [(2+N)*N,N*D], alpha [3], and bias [(2+N)*N].
    Returns (hin [T,D], hpost [T,N], hres [T,N,N]), preserving vendor
    output dtypes. Input constraints are delegated to the selected backend.
    No output layer normalization is fused.
    """
    impl_fn = get_impl_by_name("hyper_conn.mhc_pre_sinkhorn", impl)
    if x.shape[0] == 0:
        return (
            x.new_empty((0, x.shape[-1])),
            x.new_empty((0, hc_mult), dtype=torch.float32),
            x.new_empty((0, hc_mult, hc_mult), dtype=torch.float32),
        )
    return impl_fn(x, phi, alpha, bias, hc_mult, num_iters, hc_eps, norm_eps)
