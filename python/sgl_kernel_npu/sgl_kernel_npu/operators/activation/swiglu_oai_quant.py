"""Public split-layout SwiGLU-OAI with optional per-row INT8 quantization."""

import torch

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_triton_fn = None
_torch_native_fn = None


@register_impl("activation.swiglu_oai_quant", "auto")
@register_impl("activation.swiglu_oai_quant", "triton")
def _triton_impl(x, alpha, limit, need_quant, group_list, group_list_type):
    global _triton_fn
    if _triton_fn is None:
        from ...implementation.triton.activation.swiglu_oai_quant import (
            swiglu_oai_quant,
        )

        _triton_fn = swiglu_oai_quant
    return _triton_fn(x, alpha, limit, need_quant, group_list, group_list_type)


@register_impl("activation.swiglu_oai_quant", "torch_native")
def _torch_native_impl(x, alpha, limit, need_quant, group_list, group_list_type):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.activation.swiglu_oai_quant import (
            swiglu_oai_quant,
        )

        _torch_native_fn = swiglu_oai_quant
    return _torch_native_fn(x, alpha, limit, need_quant, group_list, group_list_type)


def _check_inputs(group_list, group_list_type):
    """Retain only the legacy group encoding and index-dtype checks."""
    if group_list is not None:
        if group_list_type not in (0, 1):
            raise ValueError(f"group_list_type must be 0 or 1, got {group_list_type}")
        if group_list.dtype not in (torch.int32, torch.int64):
            raise ValueError(
                f"group_list dtype must be torch.int32 or torch.int64, "
                f"got {group_list.dtype}"
            )


@register_operator("activation.swiglu_oai_quant")
def swiglu_oai_quant(
    x,
    alpha,
    limit,
    need_quant=True,
    group_list=None,
    group_list_type=None,
    *,
    impl="auto",
):
    """Compute clamped ``gate * sigmoid(gate * alpha) * (up + 1)``.

    Args:
        x (torch.Tensor): FP16/BF16/FP32 input [..., 2D], split [gate | up].
        alpha (float): Finite sigmoid multiplier.
        limit (float): Finite positive clamp bound.
        need_quant (bool): Return per-row INT8 quantization when True.
            For auto/triton, output width D must be a multiple of 1536,
            as required by the legacy kernel.
        group_list (torch.Tensor): Optional nonempty 1-D INT32/INT64 tensor
            on x.device. Counts (type 1) or cumulative counts (type 0).
        group_list_type (int): Required with group_list; 0 or 1.
        impl (str): "auto" (Triton), "triton", or "torch_native".

    Returns:
        (out, scale): out has shape [..., D] and dtype INT8 if need_quant,
        otherwise x.dtype. FP32 scale has shape [prod(x.shape[:-1])].
        Grouped mode only defines the first R flattened rows, where R is
        group_list[-1] (type 0) or sum(group_list) (type 1); callers must
        guarantee 0 <= R <= the input row count. Unprocessed output rows
        and all scales when need_quant=False are unspecified. Zero rows
        quantize to zero with scale zero. Inputs are not modified.
    """
    _check_inputs(group_list, group_list_type)
    impl_fn = get_impl_by_name("activation.swiglu_oai_quant", impl)
    return impl_fn(x, alpha, limit, need_quant, group_list, group_list_type)
