"""Public same-shape sigmoid gating operator for attention outputs."""

import torch

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_triton_fn = None
_torch_native_fn = None


def _check_inputs(x: torch.Tensor, gate: torch.Tensor) -> None:
    """Retain only the legacy same-shape check before backend dispatch."""
    if x.shape != gate.shape:
        raise ValueError(
            f"x and gate must have the same shape, got {x.shape} vs {gate.shape}"
        )


@register_impl("activation.fused_sigmoid_mul", "auto")
@register_impl("activation.fused_sigmoid_mul", "triton")
def _triton_impl(x, gate):
    global _triton_fn
    if _triton_fn is None:
        from ...implementation.triton.activation.fused_sigmoid_mul import (
            fused_sigmoid_mul,
        )

        _triton_fn = fused_sigmoid_mul
    return _triton_fn(x, gate)


@register_impl("activation.fused_sigmoid_mul", "torch_native")
def _torch_native_impl(x, gate):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.activation.fused_sigmoid_mul import (
            fused_sigmoid_mul,
        )

        _torch_native_fn = fused_sigmoid_mul
    return _torch_native_fn(x, gate)


@register_operator("activation.fused_sigmoid_mul")
def fused_sigmoid_mul(
    x: torch.Tensor, gate: torch.Tensor, *, impl="auto"
) -> torch.Tensor:
    """Return ``(x.float() * sigmoid(gate.float())).to(x.dtype)``.

    ``auto`` selects the Triton implementation without automatic fallback.
    Both implementations compute in FP32 and return a new tensor, leaving
    the inputs unchanged. Non-contiguous inputs, scalars and empty tensors
    are supported. The Triton backend provides forward execution for inference.

    Args:
        x: FP32, FP16 or BF16 strided tensor on an NPU, any shape.
        gate: FP32, FP16 or BF16 tensor with the same shape and NPU as x.
            Its dtype may differ from x. Contains sigmoid inputs, not
            precomputed sigmoid values.
        impl: Keyword-only implementation name: auto, triton or torch_native.

    Returns:
        A tensor with x's shape, dtype and device.
    """
    _check_inputs(x, gate)
    impl_fn = get_impl_by_name("activation.fused_sigmoid_mul", impl)
    return impl_fn(x, gate)
