"""Public mhc_pre operator with registered implementations."""

import torch

from ..utils.cann import detect_cann_version
from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator

_cann_920_fn = None
_cann_920_v2_fn = None
_cann_910_fn = None
_torch_native_fn = None


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
    global _cann_920_fn
    cann_version = detect_cann_version()
    if not cann_version.startswith("9.2."):
        raise RuntimeError(
            f"This implementation requires CANN 9.2.x, got {cann_version}"
        )
    if _cann_920_fn is None:
        from ..._third_party.sgl_ops_transformer_dsv4 import ops

        _cann_920_fn = ops.mhc_pre_sinkhorn

    shape = x.shape
    h_in, h_post, h_res = _cann_920_fn(
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
    global _cann_910_fn
    cann_version = detect_cann_version()
    if not cann_version.startswith("9.1."):
        raise RuntimeError(
            f"This implementation requires CANN 9.1.x, got {cann_version}"
        )
    if _cann_910_fn is None:
        from ..._third_party.sgl_cann_recipes_infer import npu_hc_pre

        _cann_910_fn = npu_hc_pre

    y, post, comb = _cann_910_fn(
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


@register_impl(
    "hyper_conn.mhc_pre_dpskv4",
    "cann_recipes_infer_cann920",
    supported_generations={Generation.ARCH35},
)
def _cann_9_2_mhc_pre_v2(
    x,
    fn,
    scale,
    base,
    hc_mult,
    sinkhorn_iters,
    hc_eps,
    rms_norm_eps,
    *,
    pre_mix=None,
    return_pre=False,
):
    global _cann_920_v2_fn
    cann_version = detect_cann_version()
    if not cann_version.startswith("9.2."):
        raise RuntimeError(
            f"This implementation requires CANN 9.2.x, got {cann_version}"
        )
    if _cann_920_v2_fn is None:
        from ..._third_party import sgl_cann_recipes_infer  # noqa: F401

        _cann_920_v2_fn = getattr(
            torch.ops.sgl_cann_recipes_infer, "npu_hc_pre_v2", None
        )
        if _cann_920_v2_fn is None:
            raise RuntimeError(
                "Missing operator sgl_cann_recipes_infer.npu_hc_pre_v2 in the installed "
                f"sgl-kernel-npu wheel for CANN {cann_version}"
            )
    y, post, comb, pre = _cann_920_v2_fn(
        x,
        fn,
        scale,
        base,
        pre_mix,
        hc_mult=hc_mult,
        hc_sinkhorn_iters=sinkhorn_iters,
        norm_eps=rms_norm_eps,
        hc_eps=hc_eps,
    )
    y = y.to(x.dtype)
    return (y, post, comb, pre) if return_pre else (y, post, comb)


@register_impl("hyper_conn.mhc_pre_dpskv4", "torch_native")
def _torch_mhc_pre(
    x,
    fn,
    scale,
    base,
    hc_mult,
    sinkhorn_iters,
    hc_eps,
    rms_norm_eps,
    *,
    pre_mix=None,
    return_pre=False,
):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.hyper_conn.mhc_pre_dpskv4 import (
            mhc_pre_dpskv4,
        )

        _torch_native_fn = mhc_pre_dpskv4

    outputs = _torch_native_fn(
        x,
        fn,
        scale,
        base,
        hc_mult,
        sinkhorn_iters,
        hc_eps,
        rms_norm_eps,
        pre_mix=pre_mix,
    )
    return outputs if return_pre else outputs[:3]


@register_operator("hyper_conn.mhc_pre_dpskv4")
def mhc_pre_dpskv4(
    x,
    fn,
    scale,
    base,
    hc_mult,
    sinkhorn_iters,
    hc_eps,
    rms_norm_eps,
    *,
    pre_mix=None,
    return_pre=False,
    impl="auto",
):
    """Return mixed activation, post weights and residual mixing matrix; input layernorm remains separate.

    pre_mix optionally supplies the [T,n] FP32 weights used to accumulate y, and
    return_pre appends the computed pre weights as a fourth output. Both need impl
    "cann_recipes_infer_cann920" (DSV4.1 fused kernel) or "torch_native"; auto stays
    on the DSV4 backend.
    torch_native is an eager correctness reference, not a graph fallback.

    impl: auto (SDK-selected), cann_ops_transformers_cann920,
    cann_recipes_infer_cann910, cann_recipes_infer_cann920, or torch_native.
    """
    if not isinstance(return_pre, bool):
        raise TypeError("return_pre must be bool")
    needs_pre = pre_mix is not None or return_pre
    if needs_pre and impl not in ("cann_recipes_infer_cann920", "torch_native"):
        raise ValueError(
            "pre_mix and return_pre require impl 'cann_recipes_infer_cann920' or "
            f"'torch_native', got {impl!r}"
        )
    impl_fn = get_impl_by_name("hyper_conn.mhc_pre_dpskv4", impl)
    if needs_pre:
        return impl_fn(
            x,
            fn,
            scale,
            base,
            hc_mult,
            sinkhorn_iters,
            hc_eps,
            rms_norm_eps,
            pre_mix=pre_mix,
            return_pre=return_pre,
        )
    return impl_fn(x, fn, scale, base, hc_mult, sinkhorn_iters, hc_eps, rms_norm_eps)
