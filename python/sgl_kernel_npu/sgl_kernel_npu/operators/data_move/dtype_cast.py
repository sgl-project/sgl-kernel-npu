"""TorchNPU dtype conversion, including packed NPU dtypes."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_OP_NAME = "data_move.dtype_cast"
_torch_npu_fn = None


@register_impl(_OP_NAME, "auto")
@register_impl(_OP_NAME, "torch_npu")
def _torch_npu_impl(input, dtype, input_dtype):
    global _torch_npu_fn
    if _torch_npu_fn is None:
        from torch_npu import npu_dtype_cast

        _torch_npu_fn = npu_dtype_cast
    if input_dtype is None:
        return _torch_npu_fn(input, dtype)
    return _torch_npu_fn(input, dtype, input_dtype=input_dtype)


@register_operator(_OP_NAME)
def dtype_cast(input, dtype, input_dtype=None, *, impl="auto"):
    """Cast an NPU tensor to dtype, optionally interpreting its input dtype.

    The installed TorchNPU defines packed dtype support and output shape.
    Auto selects torch_npu.
    """
    return get_impl_by_name(_OP_NAME, impl)(input, dtype, input_dtype)
