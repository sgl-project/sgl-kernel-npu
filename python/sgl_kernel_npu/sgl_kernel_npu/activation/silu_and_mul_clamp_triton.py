from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _silu_and_mul_clamp_kernel(
    gate_up,
    out,
    weights,
    m,
    n,
    gate_up_stride,
    out_stride,
    limit,
    HAS_LIMIT: tl.constexpr,
    HAS_WEIGHTS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (rows[:, None] < m) & (cols[None, :] < n)
    base = gate_up + rows[:, None].to(tl.int64) * gate_up_stride
    gate = tl.load(base + cols[None, :], mask=mask, other=0.0).to(tl.float32)
    up = tl.load(base + n + cols[None, :], mask=mask, other=0.0).to(tl.float32)
    if HAS_LIMIT:
        gate = tl.minimum(gate, limit)
        up = tl.minimum(tl.maximum(up, -limit), limit)
    value = gate * tl.sigmoid(gate) * up
    if HAS_WEIGHTS:
        weight = tl.load(weights + rows, mask=rows < m, other=1.0).to(tl.float32)
        value *= weight[:, None]
    tl.store(
        out + rows[:, None].to(tl.int64) * out_stride + cols[None, :],
        value.to(out.dtype.element_ty),
        mask=mask,
    )


def silu_and_mul_clamp_triton(
    gate_up: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    swiglu_limit: Optional[float] = None,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute clamped SwiGLU on a ``[..., 2N]`` gate/up tensor."""
    if gate_up.stride(-1) != 1:
        gate_up = gate_up.contiguous()
    two_n = gate_up.shape[-1]
    assert two_n % 2 == 0, "gate_up last dim must be even"
    n = two_n // 2
    m = gate_up.numel() // two_n
    if out is None:
        out = gate_up.new_empty(*gate_up.shape[:-1], n)
    if m == 0:
        return out
    gate_up = gate_up.reshape(m, two_n)
    out_2d = out.view(m, n)
    if weights is not None:
        weights = weights.reshape(-1).contiguous()
        assert weights.numel() == m, "weights must have one entry per row"
    block_m, block_n, num_warps = (1, 256, 4) if m <= 16 else (8, 1024, 8)
    has_limit = swiglu_limit is not None and swiglu_limit > 0
    _silu_and_mul_clamp_kernel[(triton.cdiv(m, block_m), triton.cdiv(n, block_n))](
        gate_up,
        out_2d,
        weights,
        m,
        n,
        gate_up.stride(0),
        out_2d.stride(0),
        float(swiglu_limit) if has_limit else 0.0,
        HAS_LIMIT=has_limit,
        HAS_WEIGHTS=weights is not None,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=num_warps,
    )
    return out
