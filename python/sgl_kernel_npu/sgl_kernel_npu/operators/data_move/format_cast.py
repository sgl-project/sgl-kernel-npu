"""TorchNPU physical storage-format conversion."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_OP_NAME = "data_move.format_cast"
_torch_npu_fn = None


@register_impl(_OP_NAME, "auto")
@register_impl(_OP_NAME, "torch_npu")
def _torch_npu_impl(input, acl_format, **kwargs):
    global _torch_npu_fn
    if _torch_npu_fn is None:
        from torch_npu import npu_format_cast

        _torch_npu_fn = npu_format_cast
    return _torch_npu_fn(input, acl_format, **kwargs)


@register_operator(_OP_NAME)
def format_cast(
    input, acl_format, *, customize_dtype=None, input_dtype=None, impl="auto"
):
    """Convert an NPU tensor to an ACL format ID or a tensor's storage format.

    The optional dtype arguments describe packed input and output representations.
    Auto selects torch_npu and returns the converted tensor.
    """
    dtypes = {}
    if customize_dtype is not None:
        dtypes["customize_dtype"] = customize_dtype
    if input_dtype is not None:
        dtypes["input_dtype"] = input_dtype
    return get_impl_by_name(_OP_NAME, impl)(input, acl_format, **dtypes)
