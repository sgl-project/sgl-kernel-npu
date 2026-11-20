"""Public fast_gelu operator."""

import torch_npu

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_OP_NAME = "activation.fast_gelu"
_torch_native_fn = None


@register_impl(_OP_NAME, "auto")
@register_impl(_OP_NAME, "ops-nn")
def _torch_npu_impl(input):
    return torch_npu.npu_fast_gelu(input)


@register_impl(_OP_NAME, "torch_native")
def _torch_native_impl(input):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.activation.fast_gelu import fast_gelu

        _torch_native_fn = fast_gelu
    return _torch_native_fn(input)


@register_operator(_OP_NAME)
def fast_gelu(input, *, impl="auto"):
    """Apply x * sigmoid(1.702 * x) elementwise.

    input is a floating-point tensor of any supported shape. The output
    preserves its shape and dtype. auto selects ops-nn.
    """
    return get_impl_by_name(_OP_NAME, impl)(input)
