"""Public row-wise sigmoid gating operator."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_triton_fn = None
_torch_native_fn = None


@register_impl("activation.fused_sigmoid_mul_broadcast", "auto")
@register_impl("activation.fused_sigmoid_mul_broadcast", "triton")
def _triton_impl(x, gate):
    global _triton_fn
    if _triton_fn is None:
        from ...implementation.triton.activation.fused_sigmoid_mul_broadcast import (
            fused_sigmoid_mul_broadcast,
        )

        _triton_fn = fused_sigmoid_mul_broadcast
    return _triton_fn(x, gate)


@register_impl("activation.fused_sigmoid_mul_broadcast", "torch_native")
def _torch_native_impl(x, gate):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.activation.fused_sigmoid_mul_broadcast import (
            fused_sigmoid_mul_broadcast,
        )

        _torch_native_fn = fused_sigmoid_mul_broadcast
    return _torch_native_fn(x, gate)


def _check_inputs(x, gate):
    """Retain only the legacy input-rank and gate-shape checks."""
    if x.ndim != 2:
        raise ValueError(f"x must be 2-D, got ndim={x.ndim}, shape={x.shape}")

    bs = x.shape[0]

    if gate.ndim == 2:
        if gate.shape[0] != bs or gate.shape[1] != 1:
            raise ValueError(f"2-D gate must have shape ({bs}, 1), got {gate.shape}")
    elif gate.ndim != 1 or gate.shape[0] != bs:
        raise ValueError(f"gate must be ({bs},) or ({bs}, 1), got {gate.shape}")


@register_operator("activation.fused_sigmoid_mul_broadcast")
def fused_sigmoid_mul_broadcast(x, gate, *, impl="auto"):
    """Compute ``y[n, j] = x[n, j] * sigmoid(gate[n])`` with FP32 math.

    Args:
        x (torch.Tensor): NPU float32, float16 or bfloat16 input [N, D].
        gate (torch.Tensor): Gate logits [N] or [N, 1], with x's dtype and device.
            Each row uses one gate value across all D columns.
        impl (str): Keyword-only backend: auto (default), triton or torch_native.
            Auto selects Triton.

    Returns:
        torch.Tensor: Newly allocated [N, D] output with x's dtype and device.
        Triton returns contiguous output. The torch_native layout follows
        PyTorch operations. Non-contiguous inputs are accepted.
    """
    _check_inputs(x, gate)
    impl_fn = get_impl_by_name("activation.fused_sigmoid_mul_broadcast", impl)
    return impl_fn(x, gate)
