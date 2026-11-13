"""Public RMSNorm operator with the cast-before-weight rounding."""

from ...utils.registry import get_impl_by_name, register_impl, register_operator

_triton_fn = None
_torch_native_fn = None


@register_impl("diffusion.rmsnorm_cast_before_weight", "auto")
@register_impl("diffusion.rmsnorm_cast_before_weight", "triton")
def _triton_impl(x, weight, eps):
    global _triton_fn
    if _triton_fn is None:
        from ....implementation.triton.diffusion.norm.rmsnorm_cast_before_weight import (
            rmsnorm_cast_before_weight,
        )

        _triton_fn = rmsnorm_cast_before_weight
    return _triton_fn(x, weight, eps)


@register_impl("diffusion.rmsnorm_cast_before_weight", "torch_native")
def _torch_native_impl(x, weight, eps):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ....implementation.torch_native.diffusion.norm.rmsnorm_cast_before_weight import (
            rmsnorm_cast_before_weight,
        )

        _torch_native_fn = rmsnorm_cast_before_weight
    return _torch_native_fn(x, weight, eps)


@register_operator("diffusion.rmsnorm_cast_before_weight")
def rmsnorm_cast_before_weight(x, weight, eps, *, impl="auto"):
    """Compute RMSNorm with an explicit BF16 cast before the weight multiply.

    Reproduces ``RMSNorm(cast_x_before_out_mul=True)``: the normalized FP32
    value is rounded to the input dtype before multiplying the weight, and the
    product is rounded once more on store. The reduction is ``tl.sum``, which
    matches aten's row reduction on the validated NPU architecture.

    Args:
        x (torch.Tensor): Input of shape [..., D], BF16 or FP16, contiguous, on
            an NPU. ``D`` must be a power of two.
        weight (torch.Tensor): Multiplicative weight of shape [D], same dtype
            and device as ``x``, contiguous.
        eps (float): Stability term added to the mean square.
        impl (str): Implementation name: "auto", "triton" or "torch_native".
            Defaults to "auto". Keyword-only.

    Returns:
        torch.Tensor: Newly allocated result with ``x``'s shape and dtype.
        Inputs are not modified.

    The Triton backend requires at least 16 rows and a power-of-two last
    dimension; other shapes are rejected before launch.
    """
    impl_fn = get_impl_by_name("diffusion.rmsnorm_cast_before_weight", impl)
    return impl_fn(x, weight, eps)
