"""Fused inference mHC pre-mix with the previous sublayer's weights."""

from functools import cache

import torch

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator


@cache
def _binding():
    from ..._third_party.sgl_ops_transformer_mhc_v2.ops.mhc.sgl_mhc_pre_sinkhorn.sgl_mhc_pre_sinkhorn import (
        sgl_mhc_pre_sinkhorn_op_builder,
    )

    return sgl_mhc_pre_sinkhorn_op_builder.load().sgl_mhc_pre_sinkhorn


@register_impl(
    "hyper_conn.mhc_pre_sinkhorn_v2", "auto", excluded_generations={Generation.ARCH35}
)
@register_impl(
    "hyper_conn.mhc_pre_sinkhorn_v2",
    "cann_ops_transformer",
    excluded_generations={Generation.ARCH35},
)
def _cann_impl(x, phi, alpha, bias, hc_mult, num_iters, hc_eps, norm_eps, *, pre_mix):
    # Request hPre through the pre-mix path, with backward outputs disabled
    # and HF32 enabled.
    y, post, comb, pre, *_ = _binding()(
        x, phi, alpha, bias, hc_mult, num_iters, hc_eps, norm_eps, False, pre_mix, True
    )
    return y, post, comb, pre


@register_impl("hyper_conn.mhc_pre_sinkhorn_v2", "torch_native")
def _torch_native_impl(
    x, phi, alpha, bias, hc_mult, num_iters, hc_eps, norm_eps, *, pre_mix
):
    from ...implementation.torch_native.hyper_conn.mhc_pre_dpskv4 import mhc_pre_dpskv4

    y, post, comb, pre = mhc_pre_dpskv4(
        x, phi, alpha, bias, hc_mult, num_iters, hc_eps, norm_eps, pre_mix=pre_mix
    )
    return y, post, comb.flatten(1), pre


@register_operator("hyper_conn.mhc_pre_sinkhorn_v2")
def mhc_pre_sinkhorn_v2(
    x,
    phi,
    alpha,
    bias,
    hc_mult=4,
    num_iters=20,
    hc_eps=1e-6,
    norm_eps=1e-6,
    *,
    pre_mix,
    impl="auto",
):
    """Mix [T,N,D] residuals with FP32 [T,N] weights from the previous sublayer.

    Return (hin [T,D], hpost [T,N], hres [T,N*N], hpre [T,N]).
    The three coefficients are computed from the current residual.
    """
    if pre_mix.shape != x.shape[:-1] or pre_mix.dtype != torch.float32:
        raise ValueError("pre_mix must be FP32 with shape x.shape[:-1]")
    return get_impl_by_name("hyper_conn.mhc_pre_sinkhorn_v2", impl)(
        x, phi, alpha, bias, hc_mult, num_iters, hc_eps, norm_eps, pre_mix=pre_mix
    )
