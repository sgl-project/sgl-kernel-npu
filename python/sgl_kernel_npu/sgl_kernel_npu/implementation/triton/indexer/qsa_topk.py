"""Exact NPU QSA Top-K for finite valid scores and contiguous row bounds."""

from functools import lru_cache

import torch
import triton
import triton.language as tl
from triton.language.extra.cann import extension as ext

MAX_WIDTH = 2**24
# Bound each packed FP32 allocation; native workspace and outputs are additional.
WORKSPACE_BYTES = 128 * 1024 * 1024
# Use native selection for wide or large matrices; keep smaller cases in Triton.
TILED_MAX_WIDTH = 262144
HYBRID_MIN_ROWS = 256
HYBRID_MIN_COLUMNS = 2048
HYBRID_MIN_BYTES = 16 * 1024 * 1024


def select_implementation(rows, columns, topk):
    """Choose shortcut, tiled or native hybrid selection from host shapes.

    Empty inputs and widths no larger than K use the shortcut. Large
    matrices use the hybrid to bound tiled-sort work; smaller ones use
    Triton. The row and column thresholds are performance choices, not
    input limits. No scores or row bounds are read on the host.
    """
    if columns > MAX_WIDTH:
        raise ValueError(f"score width exceeds supported maximum {MAX_WIDTH}")
    if rows == 0 or columns <= topk:
        return "shortcut"
    score_bytes = rows * columns * 4
    if (
        columns > TILED_MAX_WIDTH
        or score_bytes >= WORKSPACE_BYTES
        or (
            rows >= HYBRID_MIN_ROWS
            and columns >= HYBRID_MIN_COLUMNS
            and score_bytes >= HYBRID_MIN_BYTES
        )
    ):
        return "hybrid"
    return "tiled"


@triton.jit
def _short(
    Lengths,
    Output,
    ROWS: tl.constexpr,
    K: tl.constexpr,
    BR: tl.constexpr,
    CORES: tl.constexpr,
):
    for block in range(tl.program_id(0), tl.cdiv(ROWS, BR), CORES):
        rows = block * BR + tl.arange(0, BR)
        length = tl.load(Lengths + rows.to(tl.int64), rows < ROWS, other=0)
        cols = tl.arange(0, K)
        result = tl.where(cols[None, :] < length[:, None], cols[None, :], -1)
        tl.store(
            Output + rows.to(tl.int64)[:, None] * K + cols[None, :],
            result,
            rows[:, None] < ROWS,
        )


@lru_cache(maxsize=None)
def _vector_cores(device):
    return triton.runtime.driver.active.utils.get_device_properties(device)[
        "num_vectorcore"
    ]


_SORT_BLOCK = 4096
_MAX_PROGRAMS = 65535


@triton.jit
def _reduce_candidates(
    Scores,
    Lengths,
    Starts,
    Values,
    Threshold,
    stride,
    WIDTH: tl.constexpr,
    OUT_WIDTH: tl.constexpr,
    K: tl.constexpr,
    HAS_STARTS: tl.constexpr,
    FIRST: tl.constexpr,
    FINAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1)
    length = tl.load(Lengths + row)
    if length > K:
        start = 0
        if FIRST and HAS_STARTS:
            start = tl.load(Starts + row)
        cols = tile * BLOCK + tl.arange(0, BLOCK)
        valid = cols < WIDTH
        if FIRST:
            valid = valid & (cols < length)
        x = tl.load(Scores + row * stride + start + cols, valid, other=-float("inf"))
        ordered = ext.sort(x, descending=True)
        if FINAL:
            cutoff = ext.get_element(ordered, (K - 1,))
            tl.store(Threshold + row, cutoff)
        else:
            best = ext.extract_slice(ordered, (0,), (K,), (1,))
            tl.store(Values + row * OUT_WIDTH + tile * K + tl.arange(0, K), best)


@triton.jit
def _counts(
    Scores,
    Lengths,
    Starts,
    Threshold,
    Counts,
    stride: tl.constexpr,
    K: tl.constexpr,
    HAS_STARTS: tl.constexpr,
    TILES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1)
    length = tl.load(Lengths + row)
    if length > K:
        start = 0
        if HAS_STARTS:
            start = tl.load(Starts + row)
        cols = tile * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(
            Scores + row * stride + start + cols, cols < length, other=-float("inf")
        )
        cutoff = tl.load(Threshold + row)
        greater = tl.sum(((cols < length) & (x > cutoff)).to(tl.int32), 0)
        equal = tl.sum(((cols < length) & (x == cutoff)).to(tl.int32), 0)
        tl.store(Counts + row * (2 * TILES) + tile, greater)
        tl.store(Counts + row * (2 * TILES) + TILES + tile, equal)


@triton.jit
def _emit_tiles(
    Scores,
    Lengths,
    Starts,
    Threshold,
    Counts,
    Output,
    stride: tl.constexpr,
    K: tl.constexpr,
    HAS_STARTS: tl.constexpr,
    TILES: tl.constexpr,
    BLOCK: tl.constexpr,
    CT: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1)
    length = tl.load(Lengths + row)
    if length <= K:
        if tile == 0:
            cols = tl.arange(0, K)
            tl.store(Output + row * K + cols, tl.where(cols < length, cols, -1))
    else:
        start = 0
        if HAS_STARTS:
            start = tl.load(Starts + row)
        t = tl.arange(0, CT)
        gs = tl.load(Counts + row * (2 * TILES) + t, t < TILES, other=0)
        es = tl.load(Counts + row * (2 * TILES) + TILES + t, t < TILES, other=0)
        total = tl.sum(gs, 0)
        gb = tl.sum(tl.where(t < tile, gs, 0), 0)
        eb = tl.sum(tl.where(t < tile, es, 0), 0)
        cols = tile * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(
            Scores + row * stride + start + cols, cols < length, other=-float("inf")
        )
        cutoff = tl.load(Threshold + row)
        # Exact index sorting produces contiguous writes for each category.
        greater = (cols < length) & (x > cutoff)
        equal = (cols < length) & (x == cutoff)
        local = tl.arange(0, BLOCK)
        gi = ext.sort(tl.where(greater, cols.to(tl.float32), float("inf")))
        ei = ext.sort(tl.where(equal, cols.to(tl.float32), float("inf")))
        gc = tl.sum(greater.to(tl.int32), 0)
        ec = tl.sum(equal.to(tl.int32), 0)
        tl.store(Output + row * K + gb + local, gi.to(tl.int32), local < gc)
        tl.store(
            Output + row * K + total + eb + local,
            ei.to(tl.int32),
            (local < ec) & (total + eb + local < K),
        )


@triton.jit
def _single_tile(
    Scores,
    Lengths,
    Starts,
    Output,
    stride,
    K: tl.constexpr,
    HAS_STARTS: tl.constexpr,
    BLOCK: tl.constexpr,
    ROWS: tl.constexpr,
    CORES: tl.constexpr,
):
    for row_id in range(tl.program_id(0), ROWS, CORES):
        row = row_id.to(tl.int64)
        length = tl.load(Lengths + row)
        out_cols = tl.arange(0, K)
        if length <= K:
            tl.store(
                Output + row * K + out_cols, tl.where(out_cols < length, out_cols, -1)
            )
        else:
            start = 0
            if HAS_STARTS:
                start = tl.load(Starts + row)
            cols = tl.arange(0, BLOCK)
            x = tl.load(
                Scores + row * stride + start + cols, cols < length, other=-float("inf")
            )
            ordered = ext.sort(x, descending=True)
            cutoff = ext.get_element(ordered, (K - 1,))
            # FP32 index keys are exact: 0 <= key <= 2*BLOCK <= 8192.
            keys = tl.where(
                (cols < length) & (x > cutoff),
                cols,
                tl.where((cols < length) & (x == cutoff), BLOCK + cols, 2 * BLOCK),
            )
            ordered_ids = ext.sort(keys.to(tl.float32))
            chosen = ext.extract_slice(ordered_ids, (0,), (K,), (1,)).to(tl.int32)
            chosen = tl.where(chosen >= BLOCK, chosen - BLOCK, chosen)
            tl.store(Output + row * K + out_cols, chosen)


def tiled_topk(score, lengths, topk, row_starts=None):
    rows, columns = score.shape
    output = torch.empty((rows, topk), dtype=torch.int32, device=score.device)
    starts = lengths if row_starts is None else row_starts
    if columns <= _SORT_BLOCK:
        block = max(topk, triton.next_power_of_2(columns))
        layout = (score.stride(0), topk, row_starts is not None, block)
        # One schedule for every R: one program per row up to the vector-core
        # count, then the same kernel loops over additional rows.
        cores = min(rows, _vector_cores(score.device.index))
        _single_tile[(cores,)](score, lengths, starts, output, *layout, rows, cores)
        return output
    # Each launch respects the product of grid axes, including all sort stages.
    tiles = triton.cdiv(columns, _SORT_BLOCK)
    chunk_rows = max(1, _MAX_PROGRAMS // tiles)
    for base in range(0, rows, chunk_rows):
        end = min(rows, base + chunk_rows)
        x, lens, out = score[base:end], lengths[base:end], output[base:end]
        starts = lens if row_starts is None else row_starts[base:end]
        threshold = torch.empty(end - base, dtype=torch.float32, device=score.device)
        source, width, first = x, columns, True
        while True:
            block = min(_SORT_BLOCK, max(topk, triton.next_power_of_2(width)))
            sort_tiles = triton.cdiv(width, block)
            final = sort_tiles <= 1
            # Keeping K from each tile is exact: every discarded value already
            # has at least K values no smaller than it in its own tile.
            values = (
                threshold
                if final
                else torch.empty(
                    (end - base, sort_tiles * topk),
                    dtype=torch.float32,
                    device=x.device,
                )
            )
            _reduce_candidates[(end - base, sort_tiles)](
                source,
                lens,
                starts,
                values,
                threshold,
                source.stride(0),
                width,
                sort_tiles * topk,
                topk,
                row_starts is not None,
                first,
                final,
                block,
            )
            if final:
                break
            source, width, first = values, sort_tiles * topk, False
        counts = torch.empty(
            (end - base, 2 * tiles), dtype=torch.int32, device=x.device
        )
        args = (x, lens, starts, threshold, counts)
        layout = (x.stride(0), topk, row_starts is not None, tiles, _SORT_BLOCK)
        _counts[(end - base, tiles)](*args, *layout)
        _emit_tiles[(end - base, tiles)](
            *args, out, *layout, triton.next_power_of_2(tiles)
        )
    return output


@triton.jit
def _pack(
    X,
    Lengths,
    Starts,
    Packed,
    stride,
    M: tl.constexpr,
    K: tl.constexpr,
    HAS_STARTS: tl.constexpr,
    TILES: tl.constexpr,
    TASKS: tl.constexpr,
    CORES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    for task in range(tl.program_id(0), TASKS, CORES):
        row = (task // TILES).to(tl.int64)
        cols = (task % TILES) * BLOCK + tl.arange(0, BLOCK)
        length = tl.load(Lengths + row)
        start = 0
        if HAS_STARTS:
            start = tl.load(Starts + row)
        # Only long-row valid scores are read. Invalid/short rows pack as -inf.
        values = tl.load(
            X + row * stride + start + cols,
            (cols < length) & (length > K),
            other=-float("inf"),
        )
        tl.store(Packed + row * M + cols, values, cols < M)


@triton.jit
def _finish(
    Lengths, Indices, Output, K: tl.constexpr, ROWS: tl.constexpr, CORES: tl.constexpr
):
    for rid in range(tl.program_id(0), ROWS, CORES):
        row = rid.to(tl.int64)
        length = tl.load(Lengths + row)
        ranks = tl.arange(0, K)
        if length <= K:
            ids = tl.where(ranks < length, ranks, -1)
        else:
            # Finite valid scores strictly outrank every packed -inf.
            # No value reads, infinity repair, prefix scan or score rescan.
            ids = tl.load(Indices + row * K + ranks).to(tl.int32)
        tl.store(Output + row * K + ranks, ids)


def hybrid_topk(score, lengths, topk, row_starts=None):
    """One native selection per row chunk; 32-column alignment is retained.

    At M<=2**24 a single aligned FP32 row fits in 64 MiB. Chunking rows bounds
    the pack allocation to 128 MiB; output/native/graph memory is additional.
    Each launch is capped at vector cores; task loops cover all rows and tails.
    There is no column subdivision or candidate merge in this implementation.
    """
    rows, columns = score.shape
    output = torch.empty((rows, topk), dtype=torch.int32, device=score.device)
    cores = _vector_cores(score.device.index)
    packed_width = triton.cdiv(columns, 32) * 32
    chunk = max(1, min(65535, WORKSPACE_BYTES // (packed_width * 4)))
    for begin in range(0, rows, chunk):
        end = min(rows, begin + chunk)
        x, lens, out = score[begin:end], lengths[begin:end], output[begin:end]
        starts = lens if row_starts is None else row_starts[begin:end]
        packed = torch.empty(
            (end - begin, packed_width), dtype=score.dtype, device=score.device
        )
        tiles = triton.cdiv(packed_width, 4096)
        tasks = (end - begin) * tiles
        grid = min(cores, tasks)
        _pack[(grid,)](
            x,
            lens,
            starts,
            packed,
            x.stride(0),
            packed_width,
            topk,
            row_starts is not None,
            tiles,
            tasks,
            grid,
            4096,
        )
        values, indices = torch.topk(packed, topk, dim=1, sorted=True)
        grid = min(cores, end - begin)
        _finish[(grid,)](lens, indices, out, topk, end - begin, grid)
    return output


def fast_topk(score, lengths, topk, row_starts=None):
    """Select sequence-local compressed block indices into int32 [R,K].

    score is FP32 [R,M] with contiguous columns; lengths and optional
    row_starts are contiguous int32 [R] on the same NPU. K is 512 or 2048,
    and M<=2**24. Each row selects score[r,start:start+length]. Callers
    must provide valid bounds and finite scores inside that interval.

    For length<=K, return [0,...,length-1] followed by -1 padding. Longer
    rows return K indices selecting the largest scores; ties may choose
    different indices and the output need not be score-sorted. For example,
    start=10 and returned index 2 refer to score column 12.

    Small matrices use tiled sorting to find the exact selection threshold.
    Larger ones pack valid intervals and call torch.topk per row chunk.
    Tiled launches stay within 65535 programs; each hybrid pack allocation
    is bounded by 128 MiB. Outputs and native workspace use additional memory.

    Inputs are unchanged. Warm up before graph capture; keep metadata and
    K fixed, and update scores and row bounds in place before replay.
    """
    if type(topk) is not int or topk not in (512, 2048):
        raise ValueError("QSA NPU top-k supports integer K=512 or 2048")
    if (
        not isinstance(score, torch.Tensor)
        or score.layout != torch.strided
        or score.device.type != "npu"
        or score.dtype != torch.float32
        or score.ndim != 2
        or score.stride(1) != 1
        or score.stride(0) < 0
    ):
        raise ValueError("score must be NPU FP32 [R,M], contiguous in columns")
    rows, columns = score.shape
    implementation = select_implementation(rows, columns, topk)
    bounds = (lengths,) if row_starts is None else (lengths, row_starts)
    for tensor in bounds:
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.layout != torch.strided
            or tensor.shape != (rows,)
            or tensor.dtype != torch.int32
            or tensor.device != score.device
            or not tensor.is_contiguous()
        ):
            raise ValueError(
                "row bounds must be contiguous int32 [R] on the score device"
            )
    limit = 2**63 - 1
    end = score.storage_offset() + max(rows - 1, 0) * score.stride(0) + columns
    if end * score.element_size() > limit or rows * topk * 4 > limit:
        raise ValueError("score/output byte offsets must fit signed int64")
    if implementation == "shortcut":
        output = torch.empty((rows, topk), dtype=torch.int32, device=score.device)
        if rows:
            cores = min(_vector_cores(score.device.index), triton.cdiv(rows, 4))
            _short[(cores,)](lengths, output, rows, topk, 4, cores)
        return output
    if implementation == "hybrid":
        return hybrid_topk(score, lengths, topk, row_starts)
    return tiled_topk(score, lengths, topk, row_starts)
