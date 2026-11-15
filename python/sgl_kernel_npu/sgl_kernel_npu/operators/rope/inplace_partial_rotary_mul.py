"""Public inplace_partial_rotary_mul operator with registered implementations."""

import torch

from ..utils.cann import detect_cann_version
from ..utils.registry import get_impl_by_name, register_impl, register_operator


@register_impl("rope.inplace_partial_rotary_mul", "auto")
def _auto_inplace_partial_rotary_mul(x, cos, sin, **kwargs):
    cann_version = detect_cann_version()
    if cann_version.startswith("9.1."):
        backend = "cann_recipes_infer_cann910"
    elif cann_version.startswith("9.2."):
        backend = "cann_ops_transformers_cann920"
    else:
        raise RuntimeError(
            f"Unsupported CANN release {cann_version}; expected CANN 9.1.x or 9.2.x"
        )
    return get_impl_by_name("rope.inplace_partial_rotary_mul", backend)(
        x, cos, sin, **kwargs
    )


@register_impl("rope.inplace_partial_rotary_mul", "cann_ops_transformers_cann920")
def _cann_ops_transformer_inplace_partial_rotary_mul(x, cos, sin, **kwargs):
    cann_version = detect_cann_version()
    if not cann_version.startswith("9.2."):
        raise RuntimeError(
            f"This implementation requires CANN 9.2.x, got {cann_version}"
        )
    from ..._third_party.sgl_ops_transformer_dsv4 import ops

    ops.inplace_partial_rotary_mul(x, cos, sin, **kwargs)
    return x


@register_impl("rope.inplace_partial_rotary_mul", "cann_recipes_infer_cann910")
def _cann_9_1_inplace_partial_rotary_mul(x, cos, sin, **kwargs):
    cann_version = detect_cann_version()
    if not cann_version.startswith("9.1."):
        raise RuntimeError(
            f"This implementation requires CANN 9.1.x, got {cann_version}"
        )
    from ..._third_party import sgl_cann_recipes_infer  # noqa: F401

    op = getattr(torch.ops.sgl_cann_recipes_infer, "inplace_partial_rotary_mul", None)
    if op is None:
        raise RuntimeError(
            f"Missing operator sgl_cann_recipes_infer.inplace_partial_rotary_mul in the installed sgl-kernel-npu wheel for CANN {cann_version}"
        )
    op(x, cos, sin, **kwargs)
    return x


@register_impl("rope.inplace_partial_rotary_mul", "torch_native")
def _torch_inplace_partial_rotary_mul(x, cos, sin, **kwargs):
    from ...implementation.torch_native.rope.inplace_partial_rotary_mul import (
        inplace_partial_rotary_mul,
    )

    return inplace_partial_rotary_mul(x, cos, sin, **kwargs)


@register_operator("rope.inplace_partial_rotary_mul")
def inplace_partial_rotary_mul(
    x, cos, sin, *, rotary_mode="interleave", partial_slice=None, impl="auto"
):
    """Apply partial RoPE in place using fp32 intermediate arithmetic.

    impl: auto (SDK-selected), cann_recipes_infer_cann910, cann_ops_transformers_cann920, or torch_native.
    torch_native is an eager correctness reference, not a graph fallback.
    """
    return get_impl_by_name("rope.inplace_partial_rotary_mul", impl)(
        x,
        cos,
        sin,
        rotary_mode=rotary_mode,
        partial_slice=[0, x.shape[-1]] if partial_slice is None else partial_slice,
    )
