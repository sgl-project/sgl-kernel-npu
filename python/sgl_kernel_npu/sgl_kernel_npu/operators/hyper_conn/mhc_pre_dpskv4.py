"""Public mhc_pre operator with registered implementations."""

import torch

from ..utils.cann import detect_cann_version
from ..utils.registry import get_impl_by_name, register_impl, register_operator


@register_impl("hyper_conn.mhc_pre_dpskv4", "auto")
def _auto_mhc_pre(x, fn, scale, base, hc_mult, sinkhorn_iters, hc_eps, rms_norm_eps):
    cann_version = detect_cann_version()
    if cann_version.startswith("9.1."):
        backend = "cann_recipes_infer_cann910"
    elif cann_version.startswith("9.2."):
        backend = "cann_ops_transformers_cann920"
    else:
        raise RuntimeError(
            f"Unsupported CANN release {cann_version}; expected CANN 9.1.x or 9.2.x"
        )
    return get_impl_by_name("hyper_conn.mhc_pre_dpskv4", backend)(
        x, fn, scale, base, hc_mult, sinkhorn_iters, hc_eps, rms_norm_eps
    )


@register_impl("hyper_conn.mhc_pre_dpskv4", "cann_ops_transformers_cann920")
def _cann_mhc_pre(x, fn, scale, base, hc_mult, sinkhorn_iters, hc_eps, rms_norm_eps):
    cann_version = detect_cann_version()
    if not cann_version.startswith("9.2."):
        raise RuntimeError(
            f"This implementation requires CANN 9.2.x, got {cann_version}"
        )
    from ..._third_party.sgl_ops_transformer_dsv4 import ops

    shape = x.shape
    h_in, h_post, h_res = ops.mhc_pre_sinkhorn(
        x.reshape(-1, 1, hc_mult, shape[-1]),
        fn,
        scale,
        base,
        hc_mult,
        sinkhorn_iters,
        hc_eps,
        rms_norm_eps,
    )
    return (
        h_in.reshape(-1, shape[-1]).to(x.dtype),
        h_post.reshape(-1, hc_mult),
        h_res.reshape(-1, hc_mult, hc_mult),
    )


@register_impl("hyper_conn.mhc_pre_dpskv4", "cann_recipes_infer_cann910")
def _cann_9_1_mhc_pre(
    x, fn, scale, base, hc_mult, sinkhorn_iters, hc_eps, rms_norm_eps
):
    cann_version = detect_cann_version()
    if not cann_version.startswith("9.1."):
        raise RuntimeError(
            f"This implementation requires CANN 9.1.x, got {cann_version}"
        )
    from ..._third_party import sgl_cann_recipes_infer  # noqa: F401

    op = getattr(torch.ops.sgl_cann_recipes_infer, "npu_hc_pre", None)
    if op is None:
        raise RuntimeError(
            f"Missing operator sgl_cann_recipes_infer.npu_hc_pre in the installed sgl-kernel-npu wheel for CANN {cann_version}"
        )
    y, post, comb = op(
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


@register_impl("hyper_conn.mhc_pre_dpskv4", "torch_native")
def _torch_mhc_pre(x, fn, scale, base, hc_mult, sinkhorn_iters, hc_eps, rms_norm_eps):
    from ...implementation.torch_native.hyper_conn.mhc_pre_dpskv4 import mhc_pre_dpskv4

    return mhc_pre_dpskv4(
        x, fn, scale, base, hc_mult, sinkhorn_iters, hc_eps, rms_norm_eps
    )


@register_operator("hyper_conn.mhc_pre_dpskv4")
def mhc_pre_dpskv4(
    x, fn, scale, base, hc_mult, sinkhorn_iters, hc_eps, rms_norm_eps, *, impl="auto"
):
    """Return mixed activation, post weights and residual mixing matrix; input layernorm remains separate.

    impl: auto (SDK-selected), cann_recipes_infer_cann910, cann_ops_transformers_cann920, or torch_native.
    torch_native is an eager correctness reference, not a graph fallback.
    """
    return get_impl_by_name("hyper_conn.mhc_pre_dpskv4", impl)(
        x, fn, scale, base, hc_mult, sinkhorn_iters, hc_eps, rms_norm_eps
    )
