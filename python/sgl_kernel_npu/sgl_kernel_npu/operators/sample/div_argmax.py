"""Public div_argmax operator and backend registration."""

from typing import Optional

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator

_triton_fn = None
_torch_native_fn = None


@register_impl("sample.div_argmax", "auto", supported_generations={Generation.ARCH35})
@register_impl("sample.div_argmax", "triton", supported_generations={Generation.ARCH35})
def _triton_impl(x, divisor, *, divisor_min=None, ignore_nan=False):
    global _triton_fn
    if _triton_fn is None:
        from ...implementation.triton.sample.div_argmax import div_argmax

        _triton_fn = div_argmax
    return _triton_fn(x, divisor, divisor_min=divisor_min, ignore_nan=ignore_nan)


@register_impl(
    "sample.div_argmax", "torch_native", supported_generations={Generation.ARCH35}
)
def _torch_native_impl(x, divisor, *, divisor_min=None, ignore_nan=False):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.sample.div_argmax import div_argmax

        _torch_native_fn = div_argmax
    return _torch_native_fn(x, divisor, divisor_min=divisor_min, ignore_nan=ignore_nan)


@register_operator("sample.div_argmax")
def div_argmax(x, divisor, *, divisor_min=None, ignore_nan=False, impl="auto"):
    """Row-wise ``(x / divisor).argmax(-1)``; auto selects triton.

    Args:
        x (Tensor): FP32 NPU [batch, vocab] with unit column stride and
            vocab >= 1. Default: Required.
        divisor (Tensor): FP32 [batch, vocab] on the same NPU with unit
            column stride. Default: Required.
        divisor_min (float): Clamps divisor from below before the division;
            NaN divisor entries stay NaN. Default: None.
        ignore_nan (bool): When true, a NaN quotient never wins and a row
            with nothing above -inf gives 0. Default: False.
        impl (str): auto selects triton; explicit backend selection.
            Default: 'auto'.

    Returns:
        int64 [batch] indices of the row-wise maximum quotient, lowest index
        on ties, newly allocated on the device of x.
    """
    return get_impl_by_name("sample.div_argmax", impl)(
        x, divisor, divisor_min=divisor_min, ignore_nan=ignore_nan
    )
