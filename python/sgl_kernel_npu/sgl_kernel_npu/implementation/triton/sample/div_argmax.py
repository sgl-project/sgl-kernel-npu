"""Row-wise ``(x / divisor).argmax(-1)`` for Triton-Ascend.

Used by sampling paths that divide before the argmax: Gumbel-max draft
sampling divides probabilities by exponential noise (``divisor_min`` clamps
the noise away from zero), and the Gemma4 renoise sampler divides by
per-token noise. Dividing outside the kernel would cost an extra full pass
over the logits, so the division is fused into the reduction.

A flat two-stage split keeps enough programs in flight for a handful of very
wide rows: each partial program covers one block, and the final kernel
reduces the per-block candidates.

The two-operand ``tl.reduce((v, i), combine)`` form lowers to a region
``tt.reduce`` that the Ascend triton-mlir-opt cannot parse, so the kernels
use one-operand ``tl.max``/``tl.min`` instead: ties resolve to the lowest
index, a NaN quotient never wins, and an all-NaN tile leaves its index at
the N sentinel. ``ignore_nan`` forces NaN quotients to -inf, so a row with
nothing above -inf gives 0.

Migrated from ``sglang.kernels.ops.speculative.row_argmax`` (Apache License 2.0).
"""

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _medium_argmax_partial_kernel(
    X,
    PV,
    PI,
    N: tl.constexpr,
    SX: tl.constexpr,
    SPLITS: tl.constexpr,
    BLOCK: tl.constexpr,
    DIVISOR=None,
    SD: tl.constexpr = 0,
    DIVISOR_MIN: tl.constexpr = None,
    IGNORE_NAN: tl.constexpr = False,
):
    row, part = tl.program_id(0), tl.program_id(1)
    ix = part * BLOCK + tl.arange(0, BLOCK)
    valid = ix < N
    v = tl.load(X + row.to(tl.int64) * SX + ix, valid, float("-inf"))
    if DIVISOR is not None:
        divisor = tl.load(DIVISOR + row.to(tl.int64) * SD + ix, valid, 1.0)
        if DIVISOR_MIN is not None:
            # clamp_min keeps a NaN divisor; tl.maximum would replace it.
            divisor = tl.where(
                divisor != divisor, divisor, tl.maximum(divisor, DIVISOR_MIN)
            )
        v = tl.div_rn(v, divisor)
    if IGNORE_NAN:
        v = tl.where(v == v, v, float("-inf"))
    i = tl.where(valid, ix, N)
    best_v = tl.max(v, 0)
    best_i = tl.min(tl.where(v == best_v, i, N), 0)
    tl.store(PV + row * SPLITS + part, best_v)
    tl.store(PI + row * SPLITS + part, best_i)


@triton.jit
def _medium_argmax_final_kernel(PV, PI, OUT, SPLITS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    part = tl.arange(0, BLOCK)
    v = tl.load(PV + row * SPLITS + part, part < SPLITS, float("-inf"))
    i = tl.load(PI + row * SPLITS + part, part < SPLITS, 0x7FFFFFFF)
    best_v = tl.max(v, 0)
    best_i = tl.min(tl.where(v == best_v, i, 0x7FFFFFFF), 0)
    tl.store(OUT + row, best_i.to(tl.int64))


def div_argmax(
    x: torch.Tensor,
    divisor: torch.Tensor,
    *,
    divisor_min: Optional[float] = None,
    ignore_nan: bool = False,
) -> torch.Tensor:
    """FP32 row-wise ``(x / divisor).argmax(-1)`` with unit column strides.

    ``divisor_min`` first applies ``divisor.clamp_min(divisor_min)``; a NaN
    divisor stays NaN. With ``ignore_nan`` a NaN quotient never wins and a row
    with nothing above -inf gives 0.
    """
    if x.ndim != 2:
        raise ValueError("x must have shape [batch, vocab]")
    if x.dtype != torch.float32:
        raise ValueError("x must use torch.float32")
    if x.device.type != "npu":
        raise ValueError("x must be on an NPU")
    if not 1 <= x.shape[1]:
        raise ValueError("x must have vocab_size >= 1")
    if x.stride(1) != 1:
        raise ValueError("x must have unit column stride")
    if divisor.shape != x.shape:
        raise ValueError("divisor must have the same shape as x")
    if divisor.dtype != x.dtype:
        raise ValueError("divisor must use torch.float32")
    if divisor.device != x.device:
        raise ValueError("divisor must be on the same NPU as x")
    if divisor.stride(1) != 1:
        raise ValueError("divisor must have unit column stride")

    rows, n = x.shape
    out = torch.empty(rows, dtype=torch.int64, device=x.device)
    if rows == 0:
        return out
    # Small blocks keep enough programs in flight when rows are few.
    block = 4096 if rows <= 256 else 8192
    splits = triton.cdiv(n, block)
    values = torch.empty((rows, splits), dtype=torch.float32, device=x.device)
    indices = torch.empty((rows, splits), dtype=torch.int32, device=x.device)
    _medium_argmax_partial_kernel[(rows, splits)](
        x,
        values,
        indices,
        n,
        x.stride(0),
        splits,
        block,
        DIVISOR=divisor,
        SD=divisor.stride(0),
        DIVISOR_MIN=divisor_min,
        IGNORE_NAN=ignore_nan,
        num_warps=4,
    )
    _medium_argmax_final_kernel[(rows,)](
        values, indices, out, splits, triton.next_power_of_2(splits), num_warps=1
    )
    return out
