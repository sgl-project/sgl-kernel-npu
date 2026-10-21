"""Tiled argmax and greedy prefix verification for DSPark on Ascend."""

import torch
import triton
import triton.language as tl
from sgl_kernel_npu.utils.triton_utils import get_device_properties


@triton.jit
def _argmax_kernel(x, out, rows, width, row_stride, col_stride, BLOCK: tl.constexpr):
    for row in range(tl.program_id(0), rows, tl.num_programs(0)):
        best = float("-inf")
        index = 2147483647
        nan_index = 2147483647
        for start in range(0, width, BLOCK):
            col = start + tl.arange(0, BLOCK)
            v = tl.load(
                x + row.to(tl.int64) * row_stride + col * col_stride,
                col < width,
                other=float("-inf"),
            ).to(tl.float32)
            nan_index = tl.minimum(
                nan_index,
                tl.min(tl.where((col < width) & (v != v), col, 2147483647), 0),
            )
            finite = tl.where(v == v, v, float("-inf"))
            tile_best = tl.max(finite, 0)
            tile_index = tl.min(
                tl.where((col < width) & (finite == tile_best), col, 2147483647), 0
            )
            index = tl.where(
                tile_best > best,
                tile_index,
                tl.where(tile_best == best, tl.minimum(index, tile_index), index),
            )
            best = tl.maximum(best, tile_best)
        tl.store(
            out + row, tl.where(nan_index < 2147483647, nan_index, index).to(tl.int64)
        )


def row_argmax(logits):
    rows, width = logits.shape
    assert width > 0
    out = torch.empty(rows, dtype=torch.int64, device=logits.device)
    _argmax_kernel[(max(1, min(rows, get_device_properties()[1])),)](
        logits,
        out,
        rows,
        width,
        logits.stride(0),
        logits.stride(1),
        BLOCK=1024,
        num_warps=1,
    )
    return out


@triton.jit
def _prefix_kernel(
    candidates,
    target,
    cutoff,
    lengths,
    bonus,
    trim,
    rows,
    width: tl.constexpr,
    s0,
    s1,
    HAS_CUTOFF: tl.constexpr,
    BLOCK: tl.constexpr,
):
    for row in range(tl.program_id(0), rows, tl.num_programs(0)):
        col = tl.arange(0, BLOCK)
        draft = tl.load(
            candidates + row.to(tl.int64) * s0 + (col + 1) * s1,
            col < width - 1,
            other=0,
        )
        predicted = tl.load(target + row * width + col, col < width - 1, other=0)
        length = tl.min(
            tl.where((col < width - 1) & (draft != predicted), col, width - 1), 0
        )
        capped = length
        if HAS_CUTOFF:
            capped = tl.minimum(length, tl.load(cutoff + row) - 1)
        tl.store(lengths + row, capped)
        tl.store(trim + row, length - capped)
        tl.store(bonus + row, tl.load(target + row * width + capped))


def accept_greedy(
    *,
    candidates,
    target_logits,
    verify_num_draft_tokens,
    cutoff_verify_lens=None,
    fused_argmax=False
):
    rows, width = candidates.shape
    assert rows > 0 and width == verify_num_draft_tokens
    assert target_logits.shape[0] == rows * width
    target = row_argmax(target_logits)
    lengths = torch.empty(rows, dtype=torch.int32, device=candidates.device)
    bonus = torch.empty(rows, dtype=torch.int64, device=candidates.device)
    trim = torch.empty_like(lengths)
    _prefix_kernel[(min(rows, get_device_properties()[1]),)](
        candidates,
        target,
        cutoff_verify_lens,
        lengths,
        bonus,
        trim,
        rows,
        width,
        candidates.stride(0),
        candidates.stride(1),
        HAS_CUTOFF=cutoff_verify_lens is not None,
        BLOCK=triton.next_power_of_2(max(width - 1, 1)),
        num_warps=1,
    )
    return lengths, bonus, trim


class AcceptGreedy:
    execute = staticmethod(accept_greedy)
    triton = staticmethod(accept_greedy)
