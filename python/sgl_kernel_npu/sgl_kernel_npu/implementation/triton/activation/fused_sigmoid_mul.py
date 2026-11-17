"""Fused same-shape sigmoid gating on Ascend vector cores."""

import torch
import triton
import triton.language as tl

from ..utils.triton_utils import get_device_properties

_ELEM_BLOCK_SIZE = 2048


@triton.jit
def _fused_sigmoid_mul_kernel(
    x_ptr,
    gate_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    block_start = pid * BLOCK_SIZE

    for offset in tl.range(block_start, n_elements, num_programs * BLOCK_SIZE):
        idx = offset + tl.arange(0, BLOCK_SIZE)
        mask = idx < n_elements
        x = tl.load(x_ptr + idx, mask=mask).to(tl.float32)
        g = tl.load(gate_ptr + idx, mask=mask).to(tl.float32)
        out = x * tl.sigmoid(g)
        tl.store(out_ptr + idx, out.to(x_ptr.dtype.element_ty), mask=mask)


def fused_sigmoid_mul(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Apply sigmoid gating after normalizing inputs to contiguous storage.

    The kernel and 2048-element grid-stride tiling match the legacy backend.
    """
    x = x.contiguous()
    gate = gate.contiguous()
    out = torch.empty_like(x)
    n = x.numel()
    if n == 0:
        return out

    _, num_vectorcore = get_device_properties()
    num_blocks = min(triton.cdiv(n, _ELEM_BLOCK_SIZE), num_vectorcore)
    num_blocks = max(num_blocks, 1)

    _fused_sigmoid_mul_kernel[(num_blocks,)](
        x,
        gate,
        out,
        n,
        BLOCK_SIZE=_ELEM_BLOCK_SIZE,
    )
    return out
