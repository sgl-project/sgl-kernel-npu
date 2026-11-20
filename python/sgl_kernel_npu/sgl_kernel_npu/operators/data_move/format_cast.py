"""TorchNPU physical storage-format conversion."""

import torch_npu

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_OP_NAME = "data_move.format_cast"
_native_impl = None


@register_impl(_OP_NAME, "auto")
@register_impl(_OP_NAME, "ops-math")
def _torch_npu_impl(input, acl_format, **kwargs):
    return torch_npu.npu_format_cast(input, acl_format, **kwargs)


@register_impl(_OP_NAME, "torch_native")
def _torch_native_impl(*args, **kwargs):
    global _native_impl
    if _native_impl is None:
        from ...implementation.torch_native.data_move.cast import format_cast

        _native_impl = format_cast
    return _native_impl(*args, **kwargs)


@register_operator(_OP_NAME)
def format_cast(
    input, acl_format, *, customize_dtype=None, input_dtype=None, impl="auto"
):
    """Convert an NPU tensor to an ACL format ID or a tensor's storage format.

    The optional dtype arguments describe packed input and output representations.
    Auto selects ops-math and returns the converted tensor.
    """
    dtypes = {}
    if customize_dtype is not None:
        dtypes["customize_dtype"] = customize_dtype
    if input_dtype is not None:
        dtypes["input_dtype"] = input_dtype
    return get_impl_by_name(_OP_NAME, impl)(input, acl_format, **dtypes)
