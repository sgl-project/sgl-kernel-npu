"""Inference mHC post mapping, with the CANN public argument order."""

import torch

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_cann_fn = None
_torch_native_fn = None


@register_impl("hyper_conn.mhc_post", "auto")
@register_impl("hyper_conn.mhc_post", "cann")
def _cann_impl(x, h_res, h_out, h_post):
    global _cann_fn
    if _cann_fn is None:
        from cann_ops_transformer.ops import mhc_post

        _cann_fn = mhc_post
    return _cann_fn(x, h_res, h_out, h_post)


@register_impl("hyper_conn.mhc_post", "torch_native")
def _torch_native_impl(*args):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.hyper_conn.mhc_post import mhc_post

        _torch_native_fn = mhc_post
    return _torch_native_fn(*args)


@register_operator("hyper_conn.mhc_post")
def mhc_post(x, h_res, h_out, h_post, *, impl="auto"):
    """Return [T,N,D] post-mapped streams; auto selects CANN 9.1.

    x is [T,N,D], h_res [T,N,N], h_out [T,D], and h_post [T,N].
    Input constraints and output dtype are delegated to the selected backend.
    y[t,i,d] = h_post[t,i]*h_out[t,d] + sum_j h_res[t,i,j]*x[t,j,d].
    """
    impl_fn = get_impl_by_name("hyper_conn.mhc_post", impl)
    if x.shape[0] == 0:
        return torch.empty_like(x)
    return impl_fn(x, h_res, h_out, h_post)
