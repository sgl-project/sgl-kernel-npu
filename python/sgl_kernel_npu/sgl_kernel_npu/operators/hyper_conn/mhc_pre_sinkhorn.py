"""Inference mHC projection and Sinkhorn normalization."""

from functools import cache

import torch

from ... import _manifest
from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator

_cann_fn = None
_torch_native_fn = None


@cache
def _has_bundled_v2():
    return any(
        item["name"] == "sgl_ops_transformer_mhc"
        for item in _manifest.get("third_party", ())
    )


@register_impl("hyper_conn.mhc_pre_sinkhorn", "auto")
@register_impl("hyper_conn.mhc_pre_sinkhorn", "cann_ops_transformer")
def _cann_impl(
    x, phi, alpha, bias, hc_mult, num_iters, hc_eps, norm_eps, *, return_pre=False
):
    if _has_bundled_v2():
        return _cann_v2_impl(
            x,
            phi,
            alpha,
            bias,
            hc_mult,
            num_iters,
            hc_eps,
            norm_eps,
            return_pre=return_pre,
        )
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


@cache
def _v2_binding():
    from ..._third_party.sgl_ops_transformer_mhc.ops.mhc.mhc_pre_sinkhorn.mhc_pre_sinkhorn import (
        mhc_pre_sinkhorn_op_builder,
    )

    return mhc_pre_sinkhorn_op_builder.load().mhc_pre_sinkhorn


@register_impl(
    "hyper_conn.mhc_pre_sinkhorn",
    "cann_ops_transformer_v2",
    excluded_generations={Generation.ARCH35},
)
def _cann_v2_impl(
    x,
    phi,
    alpha,
    bias,
    hc_mult,
    num_iters,
    hc_eps,
    norm_eps,
    *,
    return_pre=False,
    pre_mix=None,
):
    if not _has_bundled_v2():
        raise RuntimeError(
            "mhc_pre_sinkhorn V2 requires a wheel that bundles the mHC V2 vendor"
        )
    if pre_mix is not None and (
        pre_mix.shape != x.shape[:-1] or pre_mix.dtype != torch.float32
    ):
        raise ValueError("pre_mix must be FP32 with shape x.shape[:-1]")
    # The upstream inference facade drops hPre. Call its native binding with
    # HF32 explicitly enabled and request backward buffers only for the legacy
    # return_pre call without pre_mix.
    need_backward = return_pre and pre_mix is None
    y, post, comb, pre, *_ = _v2_binding()(
        x,
        phi,
        alpha,
        bias,
        hc_mult,
        num_iters,
        hc_eps,
        norm_eps,
        need_backward,
        pre_mix,
        True,
    )
    return (y, post, comb, pre) if return_pre else (y, post, comb)


@register_impl("hyper_conn.mhc_pre_sinkhorn", "torch_native")
def _torch_native_impl(*args, return_pre=False, pre_mix=None):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.hyper_conn.mhc_pre_sinkhorn import (
            mhc_pre_sinkhorn,
        )

        _torch_native_fn = mhc_pre_sinkhorn
    return _torch_native_fn(*args, return_pre=return_pre, pre_mix=pre_mix)


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
    pre_mix=None,
    impl="auto",
):
    """Project [T,N,D] streams through the bundled CANN binding by default.

    phi has shape [(2+N)*N,N*D], alpha [3], and bias [(2+N)*N].
    Returns (hin [T,D], hpost [T,N], hres [T,N*N]), preserving vendor
    output dtypes. When return_pre is true, append hpre [T,N] in FP32.
    pre_mix optionally supplies the FP32 [T,N] weights used for hin and
    requires the cann_ops_transformer_v2 or torch_native implementation.
    Input constraints are delegated to the selected backend.
    No output layer normalization is fused.
    """
    impl_fn = get_impl_by_name("hyper_conn.mhc_pre_sinkhorn", impl)
    if pre_mix is not None:
        if impl not in ("cann_ops_transformer_v2", "torch_native"):
            raise ValueError("pre_mix requires cann_ops_transformer_v2 or torch_native")
        return impl_fn(
            x,
            phi,
            alpha,
            bias,
            hc_mult,
            num_iters,
            hc_eps,
            norm_eps,
            return_pre=return_pre,
            pre_mix=pre_mix,
        )
    if return_pre:
        return impl_fn(
            x, phi, alpha, bias, hc_mult, num_iters, hc_eps, norm_eps, return_pre=True
        )
    return impl_fn(x, phi, alpha, bias, hc_mult, num_iters, hc_eps, norm_eps)
