"""Public hyper-connection gated residual combine operator."""

from typing import Optional

import torch
import torch.nn.functional as F

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator

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


def _check_inputs(
    block_output,
    residual,
    normed_residual,
    inject_weight,
    hc_count,
    hidden_size,
    out=None,
):
    """Reject inputs outside the operator contract before any kernel launch.

    Both backends share this contract, so it is checked once here instead of
    being duplicated per backend. Only metadata is inspected; no
    device-to-host data read happens.
    """
    tensors = (block_output, residual, normed_residual, inject_weight)
    if any(t.dtype != torch.bfloat16 for t in tensors):
        raise ValueError("hc_combine requires BF16 inputs")
    if residual.device.type != "npu" or any(
        t.device != residual.device for t in tensors
    ):
        raise ValueError("all inputs must be on the same NPU")
    if any(t.ndim != 2 for t in tensors):
        raise ValueError(
            "block_output, residual, normed_residual and inject_weight must be 2-D"
        )
    if any(not t.is_contiguous() for t in tensors):
        raise ValueError(
            "block_output, residual, normed_residual and inject_weight must be contiguous"
        )
    # The branch count divides the projected injection logit, and the kernel
    # multiplies by the reciprocal, so only a power of two stays exact.
    if (
        isinstance(hc_count, bool)
        or not isinstance(hc_count, int)
        or hc_count <= 0
        or hc_count > MAX_HC_COUNT
        or hc_count & (hc_count - 1)
    ):
        raise ValueError("hc_count must be a power of two in [1, 8]")
    if (
        isinstance(hidden_size, bool)
        or not isinstance(hidden_size, int)
        or hidden_size <= 0
        or hidden_size > MAX_HIDDEN_SIZE
        or hidden_size % WIDTH_ALIGNMENT
    ):
        raise ValueError("hidden_size must be a multiple of 16 up to 4096")
    rows = residual.shape[0]
    total_width = hc_count * hidden_size
    if residual.shape[1] != total_width:
        raise ValueError("residual must have shape [rows, hc_count * hidden_size]")
    if normed_residual.shape != (rows, total_width):
        raise ValueError(
            "normed_residual must have shape [rows, hc_count * hidden_size]"
        )
    if block_output.shape != (rows, hidden_size):
        raise ValueError("block_output must have shape [rows, hidden_size]")
    if inject_weight.shape != (hc_count, total_width):
        raise ValueError(
            "inject_weight must have shape [hc_count, hc_count * hidden_size]"
        )
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


# The native entry point is compiled by the A3-only operator set of an arch22
# build, so this implementation is not declared for arch35.
@register_impl(
    "hyper_conn.hc_combine", "auto", supported_generations={Generation.ARCH22}
)
@register_impl(
    "hyper_conn.hc_combine", "ascendc", supported_generations={Generation.ARCH22}
)
def _ascendc_impl(
    block_output,
    residual,
    normed_residual,
    inject_weight,
    hc_count,
    hidden_size,
    out=None,
):
    if residual.shape[0] == 0:
        return out if out is not None else torch.empty_like(residual)
    gate_logits = F.linear(normed_residual, inject_weight)
    return torch.ops.npu.hyper_conn_branch_combine(
        block_output, residual, gate_logits, hc_count, out
    )


@register_impl("hyper_conn.hc_combine", "torch_native")
def _torch_native_impl(
    block_output,
    residual,
    normed_residual,
    inject_weight,
    hc_count,
    hidden_size,
    out=None,
):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.hyper_conn.hc_combine import hc_combine

        _torch_native_fn = hc_combine
    if residual.shape[0] == 0 and out is not None:
        return out
    result = _torch_native_fn(
        block_output, residual, normed_residual, inject_weight, hc_count, hidden_size
    )
    if out is not None:
        out.copy_(result)
        return out
    return result


@register_operator("hyper_conn.hc_combine")
def hc_combine(
    block_output: torch.Tensor,
    residual: torch.Tensor,
    normed_residual: torch.Tensor,
    inject_weight: torch.Tensor,
    hc_count: int,
    hidden_size: int,
    out: Optional[torch.Tensor] = None,
    *,
    impl: str = "auto",
) -> torch.Tensor:
    """Project per-branch gates and inject ``block_output`` into ``residual``.

    The default ``impl="auto"`` selects Ascend C. One gate per ``(row, branch)``
    is projected from ``normed_residual`` and ``inject_weight``, scaled by
    ``2 * sigmoid(logit / hc_count)``, and used to scale ``block_output`` onto
    every branch of ``residual``:

    ``z = linear(normed_residual, inject_weight)``

    ``g[m, c] = 2 * sigmoid(z[m, c] / hc_count)``

    ``out[m, c, j] = bfloat16(residual[m, c, j] + block_output[m, j]
    * g[m, c])``

    Args:
        block_output (torch.Tensor): BF16 contiguous ``[rows, hidden_size]``
            sub-block output on an NPU.
        residual (torch.Tensor): BF16 contiguous
            ``[rows, hc_count * hidden_size]`` residual on block_output's
            device.
        normed_residual (torch.Tensor): BF16 contiguous
            ``[rows, hc_count * hidden_size]`` normalized residual on
            block_output's device.
        inject_weight (torch.Tensor): BF16 contiguous
            ``[hc_count, hc_count * hidden_size]`` injection projection weight
            on block_output's device.
        hc_count (int): Hyper-connection branch count; a power of two in [1, 8].
        hidden_size (int): Per-branch hidden size; a multiple of 16 up to 4096.
        out (torch.Tensor | None): Optional pre-allocated BF16 contiguous output
            tensor on residual's device with residual's shape, not aliasing or
            overlapping any input tensor. Defaults to None.
        impl (str): Implementation name: "auto", "ascendc" or "torch_native".
            Defaults to "auto". Keyword-only.

    Returns:
        torch.Tensor: Combined BF16 result with residual's shape and device.
        Returns ``out`` (written in place) when supplied, or a newly allocated
        tensor otherwise. Inputs are not modified; the residual is not updated
        in place.

    The gate projection runs through ``torch.nn.functional.linear``, while the
    combine stage evaluates the gate, branch scaling and residual addition in
    ``float32`` and rounds to ``bfloat16`` once when writing ``out``. Both
    implementations evaluate sigmoid as ``1 / (1 + exp(-x))`` with
    backend-specific transcendentals, so outputs may differ by up to a BF16 ulp
    rather than being bitwise equal.
    """
    _check_inputs(
        block_output,
        residual,
        normed_residual,
        inject_weight,
        hc_count,
        hidden_size,
        out,
    )
    impl_fn = get_impl_by_name("hyper_conn.hc_combine", impl)
    return impl_fn(
        block_output,
        residual,
        normed_residual,
        inject_weight,
        hc_count,
        hidden_size,
        out,
    )
