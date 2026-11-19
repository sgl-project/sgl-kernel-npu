"""Public add_rms_norm operator."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_OP_NAME = "norm.add_rms_norm"
_torch_npu_fn = None
_torch_native_fn = None


@register_impl(_OP_NAME, "auto")
@register_impl(_OP_NAME, "torch_npu")
def _torch_npu_impl(x1, x2, gamma, epsilon=1e-6):
    global _torch_npu_fn
    if _torch_npu_fn is None:
        from torch_npu import npu_add_rms_norm

        _torch_npu_fn = npu_add_rms_norm
    return _torch_npu_fn(x1, x2, gamma, epsilon)


@register_impl(_OP_NAME, "torch_native")
def _torch_native_impl(x1, x2, gamma, epsilon=1e-6):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.norm.add_rms_norm import add_rms_norm

        _torch_native_fn = add_rms_norm
    return _torch_native_fn(x1, x2, gamma, epsilon)


@register_operator(_OP_NAME)
def add_rms_norm(x1, x2, gamma, epsilon=1e-6, *, impl="auto"):
    """Add x1 and x2, then apply RMS normalization and gamma.

    x1 and x2 have the same floating dtype and shape. gamma matches
    the trailing normalization dimensions, and epsilon stabilizes the
    divisor. Returns (output, float32 rstd, residual_sum) and preserves
    both inputs. auto selects torch_npu.
    """
    return get_impl_by_name(_OP_NAME, impl)(x1, x2, gamma, epsilon)
