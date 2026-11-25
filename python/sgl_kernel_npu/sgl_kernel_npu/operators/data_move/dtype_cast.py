"""TorchNPU dtype conversion, including packed NPU dtypes."""

import torch_npu

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_OP_NAME = "data_move.dtype_cast"
_native_impl = None


@register_impl(_OP_NAME, "auto")
@register_impl(_OP_NAME, "ops-math")
def _torch_npu_impl(input, dtype, input_dtype):
    if input_dtype is None:
        return torch_npu.npu_dtype_cast(input, dtype)
    return torch_npu.npu_dtype_cast(input, dtype, input_dtype=input_dtype)


@register_impl(_OP_NAME, "torch_native")
def _torch_native_impl(*args, **kwargs):
    global _native_impl
    if _native_impl is None:
        from ...implementation.torch_native.data_move.cast import dtype_cast

        _native_impl = dtype_cast
    return _native_impl(*args, **kwargs)


@register_operator(_OP_NAME)
def dtype_cast(input, dtype, input_dtype=None, *, impl="auto"):
    """Cast an NPU tensor to dtype, optionally interpreting its input dtype.

    The installed TorchNPU defines packed dtype support and output shape.
    Auto selects ops-math.
    """
    return get_impl_by_name(_OP_NAME, impl)(input, dtype, input_dtype)
