"""Row-wise sigmoid gating on Ascend vector cores."""

import torch
import triton
import triton.language as tl

from ..utils.triton_utils import get_device_properties


@triton.jit
def _fused_sigmoid_mul_broadcast_kernel(
    out_ptr,
    gate_ptr,
    x_ptr,
    num_rows,
    hidden_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    for row in tl.range(pid, num_rows, num_programs):
        g = tl.load(gate_ptr + row).to(tl.float32)
        g = tl.sigmoid(g)

        offs = tl.arange(0, BLOCK_SIZE)
        mask = offs < hidden_dim
        x = tl.load(x_ptr + row * hidden_dim + offs, mask=mask).to(tl.float32)
        out = x * g
        tl.store(
            out_ptr + row * hidden_dim + offs,
            out.to(x_ptr.dtype.element_ty),
            mask=mask,
        )


def fused_sigmoid_mul_broadcast(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Compute x * sigmoid(gate) where gate broadcasts over the last dim.

    Args:
        x: 2-D tensor of shape (N, D).
        gate: tensor of shape (N,) or (N, 1).

    Returns:
        A tensor with the same shape as x.
    """
    bs, hidden_dim = x.shape

    x = x.contiguous()
    gate = gate.contiguous()
    # Allocate after contiguous() so a transposed/strided x still yields a
    # contiguous output matching the kernel's row-major writes.
    out = torch.empty_like(x)

    BLOCK_SIZE = triton.next_power_of_2(hidden_dim)
    _, num_vectorcore = get_device_properties()
    n_rows = min(bs, num_vectorcore)

    _fused_sigmoid_mul_broadcast_kernel[(n_rows,)](
        out,
        gate,
        x,
        bs,
        hidden_dim=hidden_dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out
