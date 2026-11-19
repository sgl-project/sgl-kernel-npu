"""Public geglu operator."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_OP_NAME = "activation.geglu"
_torch_npu_fn = None
_torch_native_fn = None


@register_impl(_OP_NAME, "auto")
@register_impl(_OP_NAME, "torch_npu")
def _torch_npu_impl(input, dim=-1, approximate=1, activate_left=False):
    global _torch_npu_fn
    if _torch_npu_fn is None:
        from torch_npu import npu_geglu

        _torch_npu_fn = npu_geglu
    return _torch_npu_fn(input, dim, approximate, activate_left)


@register_impl(_OP_NAME, "torch_native")
def _torch_native_impl(input, dim=-1, approximate=1, activate_left=False):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.activation.geglu import geglu

        _torch_native_fn = geglu
    return _torch_native_fn(input, dim, approximate, activate_left)


@register_operator(_OP_NAME)
def geglu(input, dim=-1, approximate=1, activate_left=False, *, impl="auto"):
    """Split input in half along dim and apply GELU to one half.

    The selected dimension of the floating-point input must be even.
    approximate=0 uses the erf formulation and 1 uses tanh. activate_left
    selects the first half when true and the second otherwise. Returns
    (gated_output, gelu_output), with half the size along dim.
    auto selects torch_npu.
    """
    return get_impl_by_name(_OP_NAME, impl)(input, dim, approximate, activate_left)
