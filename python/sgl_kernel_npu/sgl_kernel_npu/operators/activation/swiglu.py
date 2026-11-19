"""Public swiglu operator."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_OP_NAME = "activation.swiglu"
_torch_npu_fn = None
_torch_native_fn = None


@register_impl(_OP_NAME, "auto")
@register_impl(_OP_NAME, "torch_npu")
def _torch_npu_impl(input, dim=-1):
    global _torch_npu_fn
    if _torch_npu_fn is None:
        from torch_npu import npu_swiglu

        _torch_npu_fn = npu_swiglu
    return _torch_npu_fn(input, dim)


@register_impl(_OP_NAME, "torch_native")
def _torch_native_impl(input, dim=-1):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.activation.swiglu import swiglu

        _torch_native_fn = swiglu
    return _torch_native_fn(input, dim)


@register_operator(_OP_NAME)
def swiglu(input, dim=-1, *, impl="auto"):
    """Split a floating-point input in half along an even-sized dim.

    The output is SiLU(left) * right with half the size along dim.
    auto selects torch_npu.
    """
    return get_impl_by_name(_OP_NAME, impl)(input, dim)
