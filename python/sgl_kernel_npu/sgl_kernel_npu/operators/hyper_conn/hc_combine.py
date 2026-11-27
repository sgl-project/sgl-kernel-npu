"""Public hyper-connection gated residual combine with Triton default and Ascend C backend."""

from typing import Optional

import torch
import torch.nn.functional as F

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator

_triton_fn = None
_torch_native_fn = None

# Unified Buffer sizing caps the per-branch hidden size and the branch count.
MAX_HIDDEN_SIZE = 4096
MAX_HC_COUNT = 8
WIDTH_ALIGNMENT = 16


def _tensors_overlap(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a is b:
        return True
    if a.device != b.device or a.numel() == 0 or b.numel() == 0:
        return False

    def _byte_bounds(t: torch.Tensor) -> tuple[int, int]:
        base = t.data_ptr()
        elem = t.element_size()
        if t.is_contiguous():
            return base, base + t.numel() * elem
        min_elem = 0
        max_elem = 0
        for size, stride in zip(t.shape, t.stride()):
            extent = (size - 1) * stride
            if extent >= 0:
                max_elem += extent
            else:
                min_elem += extent
        return base - (-min_elem) * elem, base + (max_elem + 1) * elem

    a_begin, a_end = _byte_bounds(a)
    b_begin, b_end = _byte_bounds(b)
    return a_begin < b_end and b_begin < a_end


def _check_inputs(block, residual, normed, weight, hc, hs, out=None):
    """Reject inputs outside the operator contract before any kernel launch."""
    tensors = (block, residual, normed, weight)
    if any(not isinstance(t, torch.Tensor) for t in tensors):
        raise ValueError("hc_combine requires BF16 tensor inputs")
    if any(t.dtype != torch.bfloat16 for t in tensors):
        raise ValueError("hc_combine requires BF16 inputs")
    if residual.device.type != "npu" or any(
        t.device != residual.device for t in tensors
    ):
        raise ValueError("all inputs must be on the same NPU")
    if any(t.ndim != 2 for t in tensors):
        raise ValueError("block, residual, normed and weight must be 2-D")
    if any(not t.is_contiguous() for t in tensors):
        raise ValueError("block, residual, normed and weight must be contiguous")
    if (
        isinstance(hc, bool)
        or not isinstance(hc, int)
        or hc <= 0
        or hc > MAX_HC_COUNT
        or hc & (hc - 1)
    ):
        raise ValueError("hc must be a power of two in [1, 8]")
    if (
        isinstance(hs, bool)
        or not isinstance(hs, int)
        or hs <= 0
        or hs > MAX_HIDDEN_SIZE
        or hs % WIDTH_ALIGNMENT
    ):
        raise ValueError("hs must be a multiple of 16 up to 4096")
    rows = residual.shape[0]
    total_width = hc * hs
    if residual.shape[1] != total_width:
        raise ValueError("residual must have shape [rows, hc * hs]")
    if normed.shape != (rows, total_width):
        raise ValueError("normed must have shape [rows, hc * hs]")
    if block.shape != (rows, hs):
        raise ValueError("block must have shape [rows, hs]")
    if weight.shape != (hc, total_width):
        raise ValueError("weight must have shape [hc, hc * hs]")
    if any(t.requires_grad for t in tensors):
        raise ValueError("hc_combine is inference only")
    if out is not None:
        if not isinstance(out, torch.Tensor):
            raise ValueError("out must be a torch.Tensor")
        if out.dtype != torch.bfloat16:
            raise ValueError("out must be BF16")
        if out.device != residual.device:
            raise ValueError("out must be on the same NPU as residual")
        if out.shape != residual.shape:
            raise ValueError("out must have the same shape as residual")
        if not out.is_contiguous():
            raise ValueError("out must be contiguous")
        if out.requires_grad:
            raise ValueError("hc_combine is inference only")
        if any(_tensors_overlap(out, t) for t in tensors):
            raise ValueError("out must not alias or overlap any input tensor")


@register_impl(
    "hyper_conn.hc_combine",
    "auto",
    supported_generations={Generation.ARCH22, Generation.ARCH35},
)
@register_impl(
    "hyper_conn.hc_combine",
    "triton",
    supported_generations={Generation.ARCH22, Generation.ARCH35},
)
def _triton_impl(block, residual, normed, weight, hc, hs, out=None):
    global _triton_fn
    if _triton_fn is None:
        from ...implementation.triton.hyper_conn.hc import combine

        _triton_fn = combine
    result = _triton_fn(block, residual, normed, weight, hc, hs)
    if out is not None:
        out.copy_(result)
        return out
    return result


# Native CANN/ACLNN AIV kernel declares SUPPORTED_ARCHS 2201, so impl="ascendc"
# is registered for arch22 only while auto remains on the Triton path.
@register_impl(
    "hyper_conn.hc_combine",
    "ascendc",
    supported_generations={Generation.ARCH22},
)
def _ascendc_impl(block, residual, normed, weight, hc, hs, out=None):
    if residual.shape[0] == 0:
        return out if out is not None else torch.empty_like(residual)
    gate_logits = F.linear(normed, weight)
    return torch.ops.npu.hyper_conn_branch_combine(
        block, residual, gate_logits, hc, out
    )


@register_impl(
    "hyper_conn.hc_combine",
    "torch_native",
    supported_generations={Generation.ARCH22, Generation.ARCH35},
)
def _torch_native_impl(block, residual, normed, weight, hc, hs, out=None):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.hyper_conn.hc import combine

        _torch_native_fn = combine
    if residual.shape[0] == 0 and out is not None:
        return out
    result = _torch_native_fn(block, residual, normed, weight, hc, hs)
    if out is not None:
        out.copy_(result)
        return out
    return result


@register_operator("hyper_conn.hc_combine")
def hc_combine(
    block: torch.Tensor,
    residual: torch.Tensor,
    normed: torch.Tensor,
    weight: torch.Tensor,
    hc: int,
    hs: int,
    out: Optional[torch.Tensor] = None,
    *,
    impl: str = "auto",
) -> torch.Tensor:
    """Project per-branch gates and inject ``block`` into ``residual``.

    The default ``impl="auto"`` selects the Triton/native FP32-intermediate
    path (``hc=4, hs=2560``) on both ``arch22`` and ``arch35``. Explicit
    ``impl="ascendc"`` selects the arch22 CANN/ACLNN path using BF16
    ``F.linear`` projection and the FP32-internal Ascend C AIV combine kernel
    with optional zero-copy pre-allocated ``out`` buffers. ``torch_native`` is
    an independent CPU FP64 reference.
    """
    _check_inputs(block, residual, normed, weight, hc, hs, out)
    impl_fn = get_impl_by_name("hyper_conn.hc_combine", impl)
    return impl_fn(block, residual, normed, weight, hc, hs, out)
