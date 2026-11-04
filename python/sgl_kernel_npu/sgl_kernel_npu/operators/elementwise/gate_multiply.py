"""Public BF16 gate multiply / output cast operator for GDN."""

import torch

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_triton_fn = None
_torch_native_fn = None


def _check_inputs(y, gate):
    """Reject inputs outside the operator contract before any kernel launch.

    Both backends share this contract, so it is checked once here instead of
    being duplicated per backend. Only metadata is inspected; no
    device-to-host data read happens.
    """
    if y.device.type != "npu" or y.device != gate.device:
        raise ValueError("y and gate must be on the same NPU")
    if (
        y.dtype != torch.bfloat16
        or gate.dtype != torch.float32
        or y.shape != gate.shape
    ):
        raise ValueError("expected matching BF16 y and FP32 gate shapes")
    if y.requires_grad or gate.requires_grad:
        raise ValueError("gate_multiply is inference only")


@register_impl("elementwise.gate_multiply", "auto")
@register_impl("elementwise.gate_multiply", "triton")
def _triton_impl(y, gate):
    global _triton_fn
    if _triton_fn is None:
        from ...implementation.triton.elementwise.gate_multiply import gate_multiply

        _triton_fn = gate_multiply
    return _triton_fn(y, gate)


@register_impl("elementwise.gate_multiply", "torch_native")
def _torch_native_impl(y, gate):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.elementwise.gate_multiply import (
            gate_multiply,
        )

        _torch_native_fn = gate_multiply
    return _torch_native_fn(y, gate)


@register_operator("elementwise.gate_multiply")
def gate_multiply(y, gate, *, impl="auto"):
    """Compute ``(y.float() * gate).to(torch.bfloat16)``; ``impl="auto"`` is Triton.

    GDN's sigmoid-after-normalization path promotes the BF16 normalized result
    to FP32, multiplies by the stored FP32 sigmoid gate, and casts the FP32
    product back to BF16. This inference-only operator performs that multiply
    and cast. The caller computes ``gate`` with its existing sigmoid before this
    call. The BF16 normalization rounding already applied to ``y`` is preserved;
    normalization is not fused and sigmoid is not approximated. This operator
    neither replaces the swish-gated layernorm kernel nor changes model
    dispatch.

    Args:
        y (torch.Tensor): BF16 normalized activation on an NPU, any shape.
        gate (torch.Tensor): FP32 sigmoid gate matching y's shape and device.
        impl (str): Implementation name: "auto", "triton" or "torch_native".
            Defaults to "auto". Keyword-only.

    Returns:
        torch.Tensor: Newly allocated BF16 result with y's shape and device.
        Inputs are not modified.

    The Triton backend uses a fused NPU kernel for contiguous inputs with at
    least 16,777,216 elements and otherwise evaluates the PyTorch expression;
    both paths are bitwise equal to the torch_native reference. The threshold
    was measured on 910C and is not a claim of benefit on every NPU.
    """
    _check_inputs(y, gate)
    impl_fn = get_impl_by_name("elementwise.gate_multiply", impl)
    return impl_fn(y, gate)
