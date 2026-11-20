"""Public rms_norm operator."""

import torch_npu

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_OP_NAME = "norm.rms_norm"
_torch_native_fn = None


@register_impl(_OP_NAME, "auto")
@register_impl(_OP_NAME, "ops-nn")
def _torch_npu_impl(input, gamma, epsilon=1e-6):
    return torch_npu.npu_rms_norm(input, gamma, epsilon)


@register_impl(_OP_NAME, "torch_native")
def _torch_native_impl(input, gamma, epsilon=1e-6):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.norm.rms_norm import rms_norm

        _torch_native_fn = rms_norm
    return _torch_native_fn(input, gamma, epsilon)


@register_operator(_OP_NAME)
def rms_norm(input, gamma, epsilon=1e-6, *, impl="auto"):
    """Apply RMS normalization followed by multiplication by gamma.

    gamma matches the trailing dimensions of the floating-point input
    and defines the normalization axes. epsilon stabilizes the divisor.
    Returns (output, rstd), with float32 rstd and singleton dimensions
    along the reduction axes. auto selects ops-nn.
    """
    return get_impl_by_name(_OP_NAME, impl)(input, gamma, epsilon)
