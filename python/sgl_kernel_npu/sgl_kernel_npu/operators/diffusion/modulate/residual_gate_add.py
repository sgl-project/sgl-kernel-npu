"""Public bit-exact residual gate add operator."""

from ...utils.registry import get_impl_by_name, register_impl, register_operator

_triton_fn = None
_torch_native_fn = None


@register_impl("diffusion.residual_gate_add", "auto")
@register_impl("diffusion.residual_gate_add", "triton")
def _triton_impl(residual, update, gate):
    global _triton_fn
    if _triton_fn is None:
        from ....implementation.triton.diffusion.modulate.residual_gate_add import (
            residual_gate_add,
        )

        _triton_fn = residual_gate_add
    return _triton_fn(residual, update, gate)


@register_impl("diffusion.residual_gate_add", "torch_native")
def _torch_native_impl(residual, update, gate):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ....implementation.torch_native.diffusion.modulate.residual_gate_add import (
            residual_gate_add,
        )

        _torch_native_fn = residual_gate_add
    return _torch_native_fn(residual, update, gate)


@register_operator("diffusion.residual_gate_add")
def residual_gate_add(residual, update, gate, *, impl="auto"):
    """Compute ``residual + update * gate``; impl="auto" selects Triton.

    Reproduces the eager chain's BF16 rounding: the product is rounded to BF16
    before the add, and the sum is rounded once more on store. The output
    matches separate eager ``mul`` and ``add`` calls bit for bit.

    Args:
        residual (torch.Tensor): Accumulator of shape [..., C], BF16,
            contiguous, on an NPU.
        update (torch.Tensor): Update of the same shape, dtype, device and
            layout as ``residual``.
        gate (torch.Tensor): Multiplier matching ``residual``'s shape, or one
            broadcast row of shape [1, C] / [C], BF16 on the same device.
        impl (str): Implementation name: "auto", "triton" or "torch_native".
            Defaults to "auto". Keyword-only.

    Returns:
        torch.Tensor: Newly allocated BF16 result with ``residual``'s shape and
        device. Inputs are not modified.

    The Triton backend requires a power-of-two last dimension and at least 16
    rows per tile; other shapes are rejected before launch.
    """
    impl_fn = get_impl_by_name("diffusion.residual_gate_add", impl)
    return impl_fn(residual, update, gate)
