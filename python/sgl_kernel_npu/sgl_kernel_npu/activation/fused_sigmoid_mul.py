"""NPU-native fused sigmoid-gate-multiply kernels.

Two variants:
- ``fused_sigmoid_mul``: element-wise ``x * sigmoid(gate)`` when ``x`` and
  ``gate`` have identical shapes.
- ``fused_sigmoid_mul_broadcast``: broadcast ``x * sigmoid(gate)`` when ``gate``
  is ``(N,)`` or ``(N, 1)`` and ``x`` is ``(N, D)``.

Optimized for Ascend NPU: tile size and grid are chosen based on the runtime
AIV (vector core) count from
:func:`sgl_kernel_npu.utils.triton_utils.get_device_properties`. Kernels are
kept pure-vector (AIV only) and avoid AIC (matrix cores).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sgl_kernel_npu.utils.triton_utils import get_device_properties


# Tile size chosen by micro-benchmark on Ascend910 9362 for batch 16/32 and
# hidden dims 3584/4096/5120.  A 2048-element tile amortizes dispatch overhead
# while still generating one block per vector core.
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
    """Compute ``x * sigmoid(gate)`` in one fused AIV kernel (same-shape).

    Args:
        x: input tensor, any shape.
        gate: gating tensor, must have the same shape as ``x``.

    Returns:
        A tensor with the same shape as ``x``.
    """
    if x.shape != gate.shape:
        raise ValueError(
            f"x and gate must have the same shape, got {x.shape} vs {gate.shape}"
        )

    x = x.contiguous()
    gate = gate.contiguous()
    out = torch.empty_like(x)
    n = x.numel()
    if n == 0:
        return out

    _, num_vectorcore = get_device_properties()
    num_blocks = min(triton.cdiv(n, _ELEM_BLOCK_SIZE), num_vectorcore)
    num_blocks = max(num_blocks, 1)

    # Pass the contiguous tensors directly; Triton pointer offsets are flat, so
    # the original rank does not matter.  Avoiding view(-1) shaves Python wrapper
    # overhead observed on Ascend NPU.
    _fused_sigmoid_mul_kernel[(num_blocks,)](
        x,
        gate,
        out,
        n,
        BLOCK_SIZE=_ELEM_BLOCK_SIZE,
    )
    return out


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
    """Compute ``x * sigmoid(gate)`` where ``gate`` broadcasts over the last dim.

    Args:
        x: 2-D tensor of shape ``(N, D)``.
        gate: tensor of shape ``(N,)`` or ``(N, 1)``.

    Returns:
        A tensor with the same shape as ``x``.
    """
    if x.ndim != 2:
        raise ValueError(f"x must be 2-D, got ndim={x.ndim}, shape={x.shape}")

    bs, hidden_dim = x.shape
    out = torch.empty_like(x)

    # Accept gate as (N,) or (N, 1).  Passing it directly avoids the squeeze()
    # Python overhead that showed up in micro-benchmarks on Ascend NPU.
    if gate.ndim == 2:
        if gate.shape[1] != 1:
            raise ValueError(f"2-D gate must have shape ({bs}, 1), got {gate.shape}")
    elif gate.ndim != 1 or gate.shape[0] != bs:
        raise ValueError(f"gate must be ({bs},) or ({bs}, 1), got {gate.shape}")

    x = x.contiguous()
    gate = gate.contiguous()

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
