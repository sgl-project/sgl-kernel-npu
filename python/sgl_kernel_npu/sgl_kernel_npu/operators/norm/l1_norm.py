"""Public l1_norm with implementation selection."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_triton_fn = None
_torch_native_fn = None


@register_impl("norm.l1_norm", "auto")
@register_impl("norm.l1_norm", "triton")
def _triton_impl(input, valid_rows=None):
    global _triton_fn
    if _triton_fn is None:
        from ...implementation.triton.norm.l1_norm import l1_norm

        _triton_fn = l1_norm

    return _triton_fn(input, valid_rows)


@register_impl("norm.l1_norm", "torch_native")
def _torch_native_impl(input, valid_rows=None):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.norm.l1_norm import l1_norm

        _torch_native_fn = l1_norm

    return _torch_native_fn(input, valid_rows)


@register_operator("norm.l1_norm")
def l1_norm(input, *, valid_rows=None, impl="auto"):
    """Normalize each row and optionally zero rows at or beyond the device scalar valid_rows."""
    return get_impl_by_name("norm.l1_norm", impl)(input, valid_rows)
