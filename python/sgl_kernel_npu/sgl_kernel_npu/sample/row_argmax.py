"""Split-vocabulary FP32 row argmax, ported from SGLang for Triton-Ascend."""

import torch
import triton
import triton.language as tl
from sgl_kernel_npu.utils.triton_utils import get_device_properties


@triton.jit
def _argmax_partial_kernel(
    X, OUTV, OUTI, N, SX, ROWS: tl.constexpr, SPLITS: tl.constexpr, BLOCK: tl.constexpr
):
    for task in range(tl.program_id(0), ROWS * SPLITS, tl.num_programs(0)):
        row = (task // SPLITS).to(tl.int64)
        part = task % SPLITS
        per = tl.cdiv(N, SPLITS)
        start = part * per
        best_v = float("-inf")
        best_i = N
        for off in tl.range(start, tl.minimum(start + per, N), BLOCK):
            idx = off + tl.arange(0, BLOCK)
            m = idx < tl.minimum(start + per, N)
            v = tl.load(X + row * SX + idx, m, float("-inf"))
            cur_v = tl.max(v, 0)
            # lowest index among the maxima of this tile
            cur_i = tl.min(tl.where(v == cur_v, idx, N), 0)
            take = (cur_v > best_v) | ((cur_v == best_v) & (cur_i < best_i))
            best_i = tl.where(take, cur_i, best_i)
            best_v = tl.where(take, cur_v, best_v)
        tl.store(OUTV + (row * SPLITS + part) * 8, best_v)
        tl.store(OUTI + (row * SPLITS + part) * 8, best_i)


@triton.jit
def _argmax_final_kernel(
    PV,
    PI,
    OUT,
    ROWS: tl.constexpr,
    SPLITS: tl.constexpr,
    BLOCK: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    NAN_PRIORITY: tl.constexpr,
):
    for tile in range(tl.program_id(0), tl.cdiv(ROWS, BLOCK_ROWS), tl.num_programs(0)):
        row = tile * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        part = tl.arange(0, BLOCK)
        valid = (row[:, None] < ROWS) & (part[None, :] < SPLITS)
        offset = (row[:, None] * SPLITS + part[None, :]) * 8
        # Explicitly fill masked lanes after the strided 2D gathers: relying
        # on load(other=...) can leave DMA padding in the reduction on Ascend.
        v = tl.load(PV + offset, mask=valid, other=0.0)
        i = tl.load(PI + offset, mask=valid, other=0)
        v = tl.where(valid, v, float("-inf"))
        i = tl.where(valid, i, 0x7FFFFFFF)
        if NAN_PRIORITY:
            nan_i = tl.min(tl.where(v != v, i, 0x7FFFFFFF), 1)
            best_v = tl.max(tl.where(v != v, float("-inf"), v), 1)
        else:
            best_v = tl.max(v, 1)
        index = tl.min(tl.where(v == best_v[:, None], i, 0x7FFFFFFF), 1)
        if NAN_PRIORITY:
            index = tl.where(nan_i != 0x7FFFFFFF, nan_i, index)
        tl.store(OUT + row, index.to(tl.int64), mask=row < ROWS)


@triton.jit
def _medium_argmax_partial_kernel(
    X,
    PV,
    PI,
    N: tl.constexpr,
    SX: tl.constexpr,
    SPLITS: tl.constexpr,
    BLOCK: tl.constexpr,
    ROWS: tl.constexpr,
):
    for task in range(tl.program_id(0), ROWS * SPLITS, tl.num_programs(0)):
        row = (task // SPLITS).to(tl.int64)
        part = task % SPLITS
        ix = part * BLOCK + tl.arange(0, BLOCK)
        valid = ix < N
        v = tl.load(X + row * SX + ix, valid, float("-inf"))
        i = tl.where(valid, ix, N)
        # Separate reductions avoid a tuple-reduce lowering unsupported by
        # some Triton-Ascend toolchains, while retaining GPU NaN priority.
        nan_i = tl.min(tl.where(valid & (v != v), i, 0x7FFFFFFF), 0)
        best_v = tl.max(tl.where(v != v, float("-inf"), v), 0)
        best_i = tl.min(tl.where(valid & (v == best_v), i, N), 0)
        best_i = tl.where(nan_i != 0x7FFFFFFF, nan_i, best_i)
        best_v = tl.where(nan_i != 0x7FFFFFFF, float("nan"), best_v)
        tl.store(PV + (row * SPLITS + part) * 8, best_v)
        tl.store(PI + (row * SPLITS + part) * 8, best_i)


def row_argmax(x: torch.Tensor) -> torch.Tensor:
    """FP32 row argmax with unit column stride; ties use the first index.

    As on GPU, only the medium-row path gives NaNs priority. The small-row
    fast path is intended for finite logits and does not match torch NaN selection.
    """
    assert x.ndim == 2 and x.dtype == torch.float32 and x.stride(1) == 1
    rows, n = x.shape
    assert n > 0
    out = torch.empty(rows, dtype=torch.int64, device=x.device)
    if rows == 0:
        return out
    cores = get_device_properties()[1]
    block = 2048
    splits = 64 if rows <= 64 else triton.cdiv(n, block)
    # Scalar DMA stores on Ascend touch a 32-byte block. Padding each partial
    # prevents one core's reduction padding from clobbering another's result.
    values = torch.empty((rows, splits, 8), dtype=torch.float32, device=x.device)
    indices = torch.empty((rows, splits, 8), dtype=torch.int32, device=x.device)
    if rows <= 64:
        _argmax_partial_kernel[(min(cores, rows * splits),)](
            x, values, indices, n, x.stride(0), rows, SPLITS=splits, BLOCK=block
        )
    else:
        _medium_argmax_partial_kernel[(min(cores, rows * splits),)](
            x, values, indices, n, x.stride(0), splits, block, rows
        )
    block_rows = 8
    _argmax_final_kernel[(min(cores, triton.cdiv(rows, block_rows)),)](
        values,
        indices,
        out,
        rows,
        splits,
        max(8, triton.next_power_of_2(splits)),
        block_rows,
        NAN_PRIORITY=rows > 64,
    )
    return out
