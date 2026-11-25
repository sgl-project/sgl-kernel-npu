"""Public swiglu quant."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_triton_fn = None
_torch_native_fn = None


@register_impl("activation.swiglu_quant", "auto")
@register_impl("activation.swiglu_quant", "triton")
def _triton_impl(
    x, group_list, group_list_type, need_quant=True, do_limit=False, limit=7.0
):
    global _triton_fn
    if _triton_fn is None:
        from ...implementation.triton.activation.swiglu_quant import swiglu_quant

        _triton_fn = swiglu_quant
    return _triton_fn(x, group_list, group_list_type, need_quant, do_limit, limit)


@register_impl("activation.swiglu_quant", "torch_native")
def _torch_native_impl(
    x, group_list, group_list_type, need_quant=True, do_limit=False, limit=7.0
):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.activation.swiglu_quant import swiglu_quant

        _torch_native_fn = swiglu_quant
    return _torch_native_fn(x, group_list, group_list_type, need_quant, do_limit, limit)


@register_operator("activation.swiglu_quant")
def swiglu_quant(
    x,
    group_list,
    group_list_type,
    need_quant=True,
    do_limit=False,
    limit=7.0,
    *,
    impl="auto"
):
    """Applies split-layout SwiGLU to routed MoE rows and optionally quantizes each row to INT8. do_limit clamps SiLU(gate) from above and clamps up on both sides before multiplication. Quantization uses max(abs(value))/127, floor(value/scale+0.5), and saturation to [-128,127]. Unrouted rows and scales without quantization are unspecified.

    Args:
        x (Tensor): Contiguous floating [rows,2*D] gate/up halves. Default: Required.
        group_list (Tensor): Contiguous INT32/INT64 expert counts. Type 0 requires an extra terminal total at storage offset numel(), outside the logical view. Default: Required.
        group_list_type (int): 1 for counts; 0 for cumulative metadata with terminal storage. Default: Required.
        need_quant (bool): Return INT8 values and FP32 row scales when true. Default: True.
        do_limit (bool): Clamp SiLU(gate) and up before multiplication. Default: False.
        limit (float): Clamp magnitude. Default: 7.0.
        impl (str): auto selects triton; explicit backend selection. Default: 'auto'.

    Returns:
        [rows,D] values and [rows] FP32 scales; scales are unspecified without quantization.
    """
    return get_impl_by_name("activation.swiglu_quant", impl)(
        x, group_list, group_list_type, need_quant, do_limit, limit
    )
