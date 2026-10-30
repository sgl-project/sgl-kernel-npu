"""Public row_argmax operator and backend registration."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_triton_fn = None
_torch_native_fn = None


@register_impl("sample.row_argmax", "auto")
@register_impl("sample.row_argmax", "triton")
def _triton_impl(x):
    global _triton_fn
    if _triton_fn is None:
        from ...implementation.triton.sample.row_argmax import row_argmax

        _triton_fn = row_argmax
    return _triton_fn(x)


@register_impl("sample.row_argmax", "torch_native")
def _torch_native_impl(x):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.sample.row_argmax import row_argmax

        _torch_native_fn = row_argmax
    return _torch_native_fn(x)


@register_operator("sample.row_argmax")
def row_argmax(x, *, impl="auto"):
    """Dispatch to the selected row_argmax implementation."""
    return get_impl_by_name("sample.row_argmax", impl)(x)
