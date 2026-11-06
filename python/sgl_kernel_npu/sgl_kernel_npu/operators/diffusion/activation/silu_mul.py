"""Public bit-exact silu-multiply operator."""

from ...utils.registry import get_impl_by_name, register_impl, register_operator

_triton_fn = None
_torch_native_fn = None


@register_impl("diffusion.silu_mul", "auto")
@register_impl("diffusion.silu_mul", "triton")
def _triton_impl(a, b):
    global _triton_fn
    if _triton_fn is None:
        from ....implementation.triton.diffusion.activation.silu_mul import silu_mul

        _triton_fn = silu_mul
    return _triton_fn(a, b)


@register_impl("diffusion.silu_mul", "torch_native")
def _torch_native_impl(a, b):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ....implementation.torch_native.diffusion.activation.silu_mul import (
            silu_mul,
        )

        _torch_native_fn = silu_mul
    return _torch_native_fn(a, b)


@register_operator("diffusion.silu_mul")
def silu_mul(a, b, *, impl="auto"):
    """Compute ``silu(a) * b``; the default impl="auto" selects Triton.

    Reproduces the eager chain's BF16 rounding: the silu result is rounded to
    BF16 before the multiply, and the product is rounded once more on store.
    Unlike a single-kernel approximation, both boundaries are explicit, so the
    output matches ``torch.nn.functional.silu(a) * b`` bit for bit.

    Args:
        a (torch.Tensor): Gate input of shape [*S], BF16, contiguous, on an NPU.
        b (torch.Tensor): Value input of the same shape, dtype, device and
            layout as ``a``.
        impl (str): Implementation name: "auto", "triton" or "torch_native".
            Defaults to "auto". Keyword-only.

    Returns:
        torch.Tensor: Newly allocated BF16 result with ``a``'s shape and device.
        Inputs are not modified.
    """
    impl_fn = get_impl_by_name("diffusion.silu_mul", impl)
    return impl_fn(a, b)
