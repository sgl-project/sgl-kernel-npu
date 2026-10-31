"""Public scale/shift operator with explicit, lazily loaded implementations."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_triton_fn = None
_torch_native_fn = None


@register_impl("norm.fused_scale_shift", "auto")
@register_impl("norm.fused_scale_shift", "triton")
def _triton_impl(x, scale, shift, scale_constant):
    global _triton_fn
    if _triton_fn is None:
        from ...implementation.triton.norm.scale_shift import fused_scale_shift

        _triton_fn = fused_scale_shift
    return _triton_fn(x, scale, shift, scale_constant)


@register_impl("norm.fused_scale_shift", "torch_native")
def _torch_native_impl(x, scale, shift, scale_constant):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.norm.scale_shift import fused_scale_shift

        _torch_native_fn = fused_scale_shift
    return _torch_native_fn(x, scale, shift, scale_constant)


@register_operator("norm.fused_scale_shift")
def fused_scale_shift(x, scale, shift, scale_constant=1.0, *, impl="auto"):
    """Apply scale and shift; auto selects Triton.

    Args:
        x (torch.Tensor): Input of shape [B, L, H].
        scale (torch.Tensor): One value or H flattened channel values.
            Triton requires H values when shift.numel() == x.numel().
        shift (torch.Tensor): One, H, or x.numel() flattened values.
        scale_constant (float): Added to scale for elementwise shift only;
            broadcast shift always uses 1.0, preserving the legacy behavior.
        impl (str): "auto", "triton", or "torch_native"; defaults to "auto".

    Returns:
        torch.Tensor: Newly allocated output with x's shape, dtype and device.
    """
    impl_fn = get_impl_by_name("norm.fused_scale_shift", impl)
    return impl_fn(x, scale, shift, scale_constant)
