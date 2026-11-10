"""Public mhc_post operator with registered implementations."""

import torch

from ..utils.cann import detect_cann_version
from ..utils.generation import Generation, get_hardware
from ..utils.registry import get_impl_by_name, register_impl, register_operator


@register_impl("hyper_conn.mhc_post_dpskv4", "auto")
def _auto_mhc_post(x, residual, post, comb):
    cann_version = detect_cann_version()
    if cann_version.startswith("9.1."):
        backend = "cann_recipes_infer_cann910"
    elif cann_version.startswith("9.2."):
        backend = "cann_ops_transformers_cann920"
    else:
        raise RuntimeError(
            f"Unsupported CANN release {cann_version}; expected CANN 9.1.x or 9.2.x"
        )
    return get_impl_by_name("hyper_conn.mhc_post_dpskv4", backend)(
        x, residual, post, comb
    )


@register_impl("hyper_conn.mhc_post_dpskv4", "cann_ops_transformers_cann920")
def _cann_mhc_post(x, residual, post, comb):
    cann_version = detect_cann_version()
    if not cann_version.startswith("9.2."):
        raise RuntimeError(
            f"This implementation requires CANN 9.2.x, got {cann_version}"
        )
    from ..._third_party.sgl_ops_transformer_dsv4 import ops

    return ops.mhc_post(residual, comb, x, post)


@register_impl("hyper_conn.mhc_post_dpskv4", "cann_recipes_infer_cann910")
def _cann_9_1_mhc_post(x, residual, post, comb):
    cann_version = detect_cann_version()
    if not cann_version.startswith("9.1."):
        raise RuntimeError(
            f"This implementation requires CANN 9.1.x, got {cann_version}"
        )
    from ..._third_party import sgl_cann_recipes_infer  # noqa: F401

    op = getattr(torch.ops.sgl_cann_recipes_infer, "npu_hc_post", None)
    if op is None:
        raise RuntimeError(
            f"Missing operator sgl_cann_recipes_infer.npu_hc_post in the installed sgl-kernel-npu wheel for CANN {cann_version}"
        )
    if get_hardware().generation == Generation.ARCH35:
        return op(
            x.unsqueeze(0), residual.unsqueeze(0), post.unsqueeze(0), comb.unsqueeze(0)
        ).squeeze(0)
    return op(x, residual, post, comb)


@register_impl("hyper_conn.mhc_post_dpskv4", "torch_native")
def _torch_mhc_post(x, residual, post, comb):
    from ...implementation.torch_native.hyper_conn.mhc_post_dpskv4 import (
        mhc_post_dpskv4,
    )

    return mhc_post_dpskv4(x, residual, post, comb)


@register_operator("hyper_conn.mhc_post_dpskv4")
def mhc_post_dpskv4(x, residual, post, comb, *, impl="auto"):
    """Mix the layer output and residual streams, preserving the output dtype.

    impl: auto (SDK-selected), cann_recipes_infer_cann910, cann_ops_transformers_cann920, or torch_native.
    torch_native is an eager correctness reference, not a graph fallback.
    """
    return get_impl_by_name("hyper_conn.mhc_post_dpskv4", impl)(x, residual, post, comb)
