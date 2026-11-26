"""Public grouped Gemma RMSNorm with Triton default and explicit Ascend C backend."""

import math
from typing import Optional

import torch

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator

_triton_fn = None
_torch_native_fn = None

# Unified Buffer sizing caps the group width, which must also fill whole
# 16-byte BF16 copies.
MAX_GROUP_SIZE = 4096
GROUP_SIZE_ALIGNMENT = 8


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


def _check_inputs(x, weight, group_size, eps, out=None):
    """Reject inputs outside the operator contract before any kernel launch."""
    if not isinstance(x, torch.Tensor) or not isinstance(weight, torch.Tensor):
        raise ValueError("grouped_gemma_rmsnorm requires BF16 x and weight tensors")
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise ValueError("grouped_gemma_rmsnorm requires BF16 x and weight")
    if x.device.type != "npu" or weight.device != x.device:
        raise ValueError("x and weight must be on the same NPU")
    if x.ndim < 2 or weight.ndim != 1:
        raise ValueError("x must have at least 2 dimensions and weight must be 1-D")
    if not x.is_contiguous() or not weight.is_contiguous():
        raise ValueError("x and weight must be contiguous")
    if group_size is None:
        width = x.shape[-1]
    elif isinstance(group_size, bool) or not isinstance(group_size, int):
        raise ValueError("group_size must be an integer or None")
    else:
        width = group_size
    if (
        width <= 0
        or width > MAX_GROUP_SIZE
        or width % GROUP_SIZE_ALIGNMENT
        or x.shape[-1] == 0
        or x.shape[-1] % width
    ):
        raise ValueError(
            "group size must be a multiple of 8 up to 4096 and divide x's last dimension"
        )
    slices, remainder = divmod(weight.numel(), width)
    if remainder or slices == 0:
        raise ValueError("weight length must be a positive multiple of the group size")
    if (x.numel() // width) % slices:
        raise ValueError(
            "the group count must be a whole number of weight periods "
            "(weight length / group size)"
        )
    if isinstance(eps, bool) or not isinstance(eps, (int, float)):
        raise ValueError("eps must be a real number")
    if not math.isfinite(eps) or not (1.17549435e-38 <= eps <= 3.4028235e38):
        raise ValueError("eps must be positive and finite")
    if x.requires_grad or weight.requires_grad:
        raise ValueError("grouped_gemma_rmsnorm is inference only")
    if out is not None:
        if not isinstance(out, torch.Tensor):
            raise ValueError("out must be a torch.Tensor")
        if out.dtype != torch.bfloat16:
            raise ValueError("out must be BF16")
        if out.device != x.device:
            raise ValueError("out must be on the same NPU as x")
        if out.shape != x.shape:
            raise ValueError("out must have the same shape as x")
        if not out.is_contiguous():
            raise ValueError("out must be contiguous")
        if out.requires_grad:
            raise ValueError("grouped_gemma_rmsnorm is inference only")
        if _tensors_overlap(out, x) or _tensors_overlap(out, weight):
            raise ValueError("out must not alias or overlap x or weight")


@register_impl(
    "norm.grouped_gemma_rmsnorm",
    "auto",
    supported_generations={Generation.ARCH22, Generation.ARCH35},
)
@register_impl(
    "norm.grouped_gemma_rmsnorm",
    "triton",
    supported_generations={Generation.ARCH22, Generation.ARCH35},
)
def _triton_impl(x, weight, group_size, eps, out=None):
    global _triton_fn
    if _triton_fn is None:
        from ...implementation.triton.norm.grouped_gemma_rmsnorm import (
            grouped_gemma_rmsnorm,
        )

        _triton_fn = grouped_gemma_rmsnorm
    effective_group_size = x.shape[-1] if group_size is None else group_size
    result = _triton_fn(x, weight, effective_group_size, eps)
    if out is not None:
        out.copy_(result)
        return out
    return result


# Native CANN/ACLNN AIV kernel declares SUPPORTED_ARCHS 2201, so impl="ascendc"
# is registered for arch22 only while auto remains on the Triton path.
@register_impl(
    "norm.grouped_gemma_rmsnorm",
    "ascendc",
    supported_generations={Generation.ARCH22},
)
def _ascendc_impl(x, weight, group_size, eps, out=None):
    return torch.ops.npu.hyper_conn_grouped_norm(x, weight, group_size, eps, out)


@register_impl(
    "norm.grouped_gemma_rmsnorm",
    "torch_native",
    supported_generations={Generation.ARCH22, Generation.ARCH35},
)
def _torch_native_impl(x, weight, group_size, eps, out=None):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.norm.grouped_gemma_rmsnorm import (
            grouped_gemma_rmsnorm,
        )

        _torch_native_fn = grouped_gemma_rmsnorm
    if x.numel() == 0 and out is not None:
        return out
    result = _torch_native_fn(x, weight, group_size, eps)
    if out is not None:
        out.copy_(result)
        return out
    return result


@register_operator("norm.grouped_gemma_rmsnorm")
def grouped_gemma_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    group_size: Optional[int] = None,
    eps: float = 1e-6,
    out: Optional[torch.Tensor] = None,
    *,
    impl: str = "auto",
) -> torch.Tensor:
    """RMS-normalize fixed-width groups and scale by ``1 + weight``.

    The default ``impl="auto"`` selects Triton on both ``arch22`` and
    ``arch35`` (fixed ``x[R, 10240], weight[10240], group_size=2560``).
    Explicit ``impl="ascendc"`` selects the arch22 CANN/ACLNN AIV kernel, which
    supports general group widths (multiples of 8 up to 4096), shared or
    per-branch cycling weights, 2-D/3-D contiguous inputs, and zero-copy
    pre-allocated ``out`` buffers. ``torch_native`` is an independent CPU FP64
    reference.
    """
    _check_inputs(x, weight, group_size, eps, out)
    impl_fn = get_impl_by_name("norm.grouped_gemma_rmsnorm", impl)
    return impl_fn(x, weight, group_size, eps, out)
