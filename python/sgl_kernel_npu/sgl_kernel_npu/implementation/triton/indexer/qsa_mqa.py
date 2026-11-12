"""QSA MQA scoring for packed keys and paged key caches.

Queries are contiguous BF16 Q[R,4,128]. Packed K[M,1,128] uses int32
starts/ends[R]; paged cache[P,16,1,128] uses int32 table[R,L] and
lengths[R], with output width 16*L. All inputs are contiguous on one NPU.

score[r,j] = sum_h relu(dot(Q[r,h], K_for_row_r[j])) / sqrt(128).
BF16 matrix products accumulate into FP32; invisible scores are -inf.
Packed bounds must satisfy 0<=start<=end<=M. Paged lengths must fit
16*L and live page IDs must be in [0,P). Live Q/K must be finite and
produce representable scores. These device contents are caller guarantees.
For example, paged length 17 writes 17 scores and pads the rest with -inf.

Packed tiles use 128 key columns below M=1024, otherwise 256, with at
most 16 or 8 query rows respectively. Paged tiles use one query row
and 256 keys; four heads are padded internally to a 16-row matrix tile.
These tile sizes balance parallelism and compiler resources. Programs
loop over tiles with the launch size capped at the compute-core count.

Inputs are unchanged. Warm up before capture and keep metadata fixed;
Q, keys, bounds and page tables may be updated in place before replay.
"""

import math
from functools import lru_cache

import torch
import triton
import triton.language as tl

__all__ = ["packed", "paged"]


@lru_cache(maxsize=None)
def _compute_cores(device_index):
    # Query a capability, not a device-name or generation string. Warmup caches
    # this host-side property before graph capture; tensor values stay on device.
    properties = triton.runtime.driver.active.utils.get_device_properties(device_index)
    cores = properties["num_aicore"]
    if cores <= 0:
        raise RuntimeError("Invalid NPU compute-core count")
    return cores


@triton.jit
def _score(
    Q,
    K,
    First,
    Last,
    Out,
    R: tl.constexpr,
    W: tl.constexpr,
    L: tl.constexpr,
    PAGED: tl.constexpr,
    BQ: tl.constexpr,
    BN: tl.constexpr,
    HQ: tl.constexpr,
    PROGRAMS: tl.constexpr,
):
    for tile in range(tl.program_id(0), tl.cdiv(R, BQ) * tl.cdiv(W, BN), PROGRAMS):
        row_base = (tile // tl.cdiv(W, BN)) * BQ
        col_tile = tile % tl.cdiv(W, BN)
        cols = col_tile * BN + tl.arange(0, BN)
        dims = tl.arange(0, 128)
        heads = tl.arange(0, HQ)
        rows = row_base + tl.arange(0, BQ)
        if PAGED:
            length = tl.load(Last + row_base)
            valid = (cols < W) & (cols < length)
            result = tl.full((BN,), -float("inf"), tl.float32)
            if col_tile * BN < length:
                pages = tl.load(First + row_base * L + cols // 16, valid, other=0).to(
                    tl.int64
                )
                k = tl.load(
                    K
                    + (pages[:, None] * 16 + cols[:, None] % 16) * 128
                    + dims[None, :],
                    valid[:, None],
                    other=0,
                )
                q = tl.load(
                    Q
                    + row_base.to(tl.int64) * 512
                    + heads[:, None] * 128
                    + dims[None, :],
                    heads[:, None] < 4,
                    other=0,
                )
                scores = tl.dot(q, tl.trans(k))
                result = tl.sum(tl.maximum(scores, 0), axis=0) / math.sqrt(128)
                result = tl.where(valid, result, -float("inf"))
            tl.store(Out + row_base.to(tl.int64) * W + cols, result, cols < W)
        else:
            starts = tl.load(First + rows, rows < R, other=W)
            ends = tl.load(Last + rows, rows < R, other=0)
            valid = (
                (rows[:, None] < R)
                & (cols[None, :] < W)
                & (cols[None, :] >= starts[:, None])
                & (cols[None, :] < ends[:, None])
            )
            # Compute packed tiles unconditionally, then apply row visibility.
            q = tl.load(
                Q + (row_base * 4 + heads[:, None]).to(tl.int64) * 128 + dims[None, :],
                row_base * 4 + heads[:, None] < R * 4,
                other=0,
            )
            k = tl.load(
                K + cols[:, None].to(tl.int64) * 128 + dims[None, :],
                cols[:, None] < W,
                other=0,
            )
            scores = tl.dot(q, tl.trans(k))
            scores = tl.reshape(tl.maximum(scores, 0), (BQ, 4, BN))
            result = tl.sum(scores, axis=1) / math.sqrt(128)
            result = tl.where(valid, result, -float("inf"))
            tl.store(
                Out + rows[:, None].to(tl.int64) * W + cols[None, :],
                result,
                (rows[:, None] < R) & (cols[None, :] < W),
            )


def _validate(q, k, first, last, paged, width=None):
    if q.ndim != 3 or q.shape[1:] != (4, 128):
        raise ValueError("Expected Q[R,4,128]")
    if q.device.type != "npu" or any(x.device != q.device for x in (k, first, last)):
        raise ValueError("All inputs must be on the same NPU")
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16:
        raise ValueError("Q and K must be BF16")
    if any(not x.is_contiguous() for x in (q, k, first, last)):
        raise ValueError("Inputs must be contiguous")
    if first.dtype != torch.int32 or last.dtype != torch.int32:
        raise ValueError("Metadata must be int32")
    if last.shape != (q.shape[0],):
        raise ValueError("Expected one length or end per query row")
    if paged:
        if k.ndim != 4 or k.shape[1:] != (16, 1, 128) or k.shape[0] == 0:
            raise ValueError("Expected cache[P,16,1,128] with P > 0")
        if first.ndim != 2 or first.shape[0] != q.shape[0]:
            raise ValueError("Expected table[R,L]")
        if not isinstance(width, int) or width != first.shape[1] * 16:
            raise ValueError("Output width must equal table.shape[1] * 16")
    elif k.ndim != 3 or k.shape[1:] != (1, 128) or first.shape != last.shape:
        raise ValueError("Expected packed K[M,1,128] and starts[R]")


def packed(q, k, starts, ends):
    """BF16 packed K[M,1,128] -> FP32[R,M]; bounds are device int32 arrays."""
    _validate(q, k, starts, ends, False)
    rows, width = q.shape[0], k.shape[0]
    out = torch.empty((rows, width), device=q.device, dtype=torch.float32)
    if not rows or not width:
        return out
    # For K widths of at least 1024, four or more 256-key tiles amortize
    # matrix/vector handoffs. Q[4096,4,128], K[1024,1,128] uses 8x256 tiles;
    # Q[1,4,128], K[65536,1,128] uses four padded rows with the same width.
    # Short K[48,1,128] or K[257,1,128] keeps 16x128 tiles to avoid wasting
    # a wide key tile and splitting an already small query workload.
    # This changes tile geometry only; packed ranges and output stay identical.
    bq_limit, bn = (8, 256) if width >= 1024 else (16, 128)
    bq = min(bq_limit, max(4, triton.next_power_of_2(rows)))
    programs = min(
        _compute_cores(q.device.index), triton.cdiv(rows, bq) * triton.cdiv(width, bn)
    )
    _score[(programs,)](
        q,
        k,
        starts,
        ends,
        out,
        rows,
        width,
        0,
        False,
        bq,
        bn,
        bq * 4,
        programs,
        enable_fp_fusion=False,
    )
    return out


def paged(q, cache, table, lengths, width):
    """BF16 cache[P,16,1,128] -> FP32[R,16*L]; each row has its own page map."""
    _validate(q, cache, table, lengths, True, width)
    rows = q.shape[0]
    out = torch.empty((rows, width), device=q.device, dtype=torch.float32)
    if not rows or not width:
        return out
    # Q[32,4,128] can represent B8 verify-W4, but each row has its own map.
    # Logical columns address cache[P,16,1,128] through table[R,L]. Never
    # reinterpret this cache as packed K, even when R=1. Write all 16*L
    # columns so short and empty graph rows retain exact -inf padding.
    # A 256-key tile joins 16 logical pages through masked cache loads. Pad
    # four heads to 16 only inside the matrix tile; the API still requires H4.
    # R=129 uses this same scoring strategy; the bounded grid loops over rows.
    bn = 256
    programs = min(_compute_cores(q.device.index), rows * triton.cdiv(width, bn))
    _score[(programs,)](
        q,
        cache,
        table,
        lengths,
        out,
        rows,
        width,
        table.shape[1],
        True,
        1,
        bn,
        16,
        programs,
        enable_fp_fusion=False,
    )
    return out
