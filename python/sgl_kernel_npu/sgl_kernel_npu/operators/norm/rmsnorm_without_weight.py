"""Public weightless RMSNorm operator."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_triton_fn = None
_torch_native_fn = None


# Importing this module registers these two functions, without importing Triton.
@register_impl("norm.rmsnorm_without_weight", "auto")
@register_impl("norm.rmsnorm_without_weight", "triton")
def _triton_impl(x, eps):
    global _triton_fn
    if _triton_fn is None:
        from ...implementation.triton.norm.rmsnorm_without_weight import (
            fused_rmsnorm_without_weight,
        )

        _triton_fn = fused_rmsnorm_without_weight

    return _triton_fn(x, eps)


@register_impl("norm.rmsnorm_without_weight", "torch_native")
def _torch_native_impl(x, eps):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.norm.rmsnorm_without_weight import (
            fused_rmsnorm_without_weight,
        )

        _torch_native_fn = fused_rmsnorm_without_weight

    return _torch_native_fn(x, eps)


@register_operator("norm.rmsnorm_without_weight")
def fused_rmsnorm_without_weight(x, eps, *, impl="auto"):
    """The default impl="auto" selects Triton.

    Apply RMS normalization over the last dimension without a weight or bias.

    Args:
        x (torch.Tensor): Input of shape [B, L, C]; normalization runs over C.
            Triton requires a contiguous three-dimensional NPU tensor whose C
            is a power of two.
        eps (float): Stability term added to the mean square.
        impl (str): Implementation name: "auto", "triton", or
            "torch_native". Defaults to "auto".

    Returns:
        torch.Tensor: Output of shape [B, L, C] with the same dtype and device
        as x.
    """
    impl_fn = get_impl_by_name("norm.rmsnorm_without_weight", impl)
    return impl_fn(x, eps)
