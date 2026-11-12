"""ELU with explicit Ascend C or PyTorch implementation selection."""

import math

import torch
import torch_npu

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator

_IMPL_NAME = "examples.elu"
# The Ascend C Kernel is only declared and measured on arch22 (Ascend910).
_A2_A3_ONLY = {Generation.ARCH22}
_torch_native_fn = None


@register_impl(_IMPL_NAME, "auto", supported_generations=_A2_A3_ONLY)
@register_impl(_IMPL_NAME, "ascendc", supported_generations=_A2_A3_ONLY)
def _ascendc_impl(x, alpha):
    return torch.ops.npu.elu(x, float(alpha))


@register_impl(_IMPL_NAME, "torch_native")
def _torch_native_impl(x, alpha):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.examples.elu import elu as reference

        _torch_native_fn = reference

    return _torch_native_fn(x, alpha)


@register_operator(_IMPL_NAME)
def elu(x, alpha=1.0, *, impl="auto"):
    """Apply ELU (Exponential Linear Unit) element-wise.

    Returns a new tensor with the same shape and dtype as x; the input is not
    modified.

    Args:
        x: Contiguous float16 or float32 tensor on an Ascend NPU with at least
            one dimension and ND storage format. Any rank and element count is
            accepted; the operator treats the tensor as a flat element sequence.
        alpha: Coefficient of the negative branch. Must be finite. Default 1.0.
        impl: Keyword-only backend selector. auto (default) selects ascendc on
            arch22 without fallback; ascendc selects the Ascend C operator;
            torch_native selects the independent PyTorch reference. The Ascend C
            backend is not available on arch35.
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor, got " + type(x).__name__)
    if x.device.type != "npu":
        raise ValueError("x must be on an Ascend NPU")
    if x.dtype not in (torch.float16, torch.float32):
        raise ValueError("x must be float16 or float32, got " + str(x.dtype))
    if x.dim() < 1:
        raise ValueError("x must have at least one dimension")
    if x.numel() == 0:
        raise ValueError("x must contain at least one element")
    if not x.is_contiguous():
        raise ValueError("x must be contiguous")
    # ACL_FORMAT_ND is 2; do not silently reinterpret a blocked layout.
    if torch_npu.get_npu_format(x) != 2:
        raise ValueError("x must use ND storage format")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise TypeError("alpha must be a real number, got " + type(alpha).__name__)
    alpha = float(alpha)
    if not math.isfinite(alpha):
        raise ValueError("alpha must be finite, got " + str(alpha))
    return get_impl_by_name(_IMPL_NAME, impl)(x, alpha)
