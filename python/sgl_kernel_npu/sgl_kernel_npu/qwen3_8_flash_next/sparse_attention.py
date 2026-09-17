"""Sparse attention over physical token slots, with FP32 split reductions."""

import torch
import triton
import triton.language as tl

# The NPU runtime limits the product of all grid dimensions per launch.
_MAX_GRID_PROGRAMS = 65535
_SPLITS = 8


@triton.jit
def _sparse_partials(
    Q,
    K,
    V,
    Slots,
    Partial,
    Maxima,
    Sums,
    Q_ROW: tl.constexpr,
    Q_HEAD: tl.constexpr,
    K_ROW: tl.constexpr,
    K_HEAD: tl.constexpr,
    V_ROW: tl.constexpr,
    V_HEAD: tl.constexpr,
    SLOT_ROW: tl.constexpr,
    SLOT_COL: tl.constexpr,
    HEADS: tl.constexpr,
    GROUP: tl.constexpr,
    DIM: tl.constexpr,
    WIDTH: tl.constexpr,
    SPLITS: tl.constexpr,
    TILES: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1)
    split = tl.program_id(2)
    dims = tl.arange(0, DIM)
    query = tl.load(Q + row * Q_ROW + head * Q_HEAD + dims).to(tl.float32)
    maximum = tl.full((), -float("inf"), tl.float32)
    total = tl.full((), 0, tl.float32)
    numerator = tl.full((DIM,), 0, tl.float32)
    for tile in range(TILES):
        cols = (split * TILES + tile) * BLOCK + tl.arange(0, BLOCK)
        slots = tl.load(
            Slots + row * SLOT_ROW + cols * SLOT_COL, cols < WIDTH, other=-1
        )
        valid = (cols < WIDTH) & (slots >= 0)
        # Mask before accessing cache: sentinel rows must never read dummy NaNs.
        safe_slots = tl.maximum(slots, 0).to(tl.int64)
        keys = tl.load(
            K + safe_slots[:, None] * K_ROW + (head // GROUP) * K_HEAD + dims[None, :],
            valid[:, None],
            other=0,
        ).to(tl.float32)
        values = tl.load(
            V + safe_slots[:, None] * V_ROW + (head // GROUP) * V_HEAD + dims[None, :],
            valid[:, None],
            other=0,
        ).to(tl.float32)
        scores = tl.sum(keys * query[None, :], axis=1) * SCALE
        scores = tl.where(valid, scores, -float("inf"))
        next_max = tl.maximum(maximum, tl.max(scores, axis=0))
        safe_max = tl.where(next_max == -float("inf"), 0, next_max)
        correction = tl.exp(maximum - safe_max)
        weights = tl.where(valid, tl.exp(scores - safe_max), 0)
        numerator = numerator * correction + tl.sum(weights[:, None] * values, axis=0)
        total = total * correction + tl.sum(weights, axis=0)
        maximum = next_max
    part = (row * HEADS + head) * SPLITS + split
    tl.store(Partial + part * DIM + dims, numerator)
    tl.store(Maxima + part, maximum)
    tl.store(Sums + part, total)


@triton.jit
def _merge_partials(
    Partial, Maxima, Sums, Output, DIM: tl.constexpr, SPLITS: tl.constexpr
):
    item = tl.program_id(0)
    splits = tl.arange(0, SPLITS)
    dims = tl.arange(0, DIM)
    part = item * SPLITS + splits
    maxima = tl.load(Maxima + part)
    sums = tl.load(Sums + part)
    maximum = tl.max(maxima, axis=0)
    maximum = tl.where(maximum == -float("inf"), 0, maximum)
    factors = tl.exp(maxima - maximum)
    partials = tl.load(Partial + part[:, None] * DIM + dims[None, :])
    denominator = tl.sum(sums * factors, axis=0)
    numerator = tl.sum(partials * factors[:, None], axis=0)
    result = numerator / tl.where(denominator > 0, denominator, 1)
    tl.store(Output + item * DIM + dims, result)


def can_run_sparse_attention(q, k, v, slots) -> bool:
    """Whether tensor metadata fits the supported layout and launch bounds."""
    return (
        q.device.type == "npu"
        and q.ndim == k.ndim == v.ndim == 3
        and slots.ndim == 2
        and q.dtype in (torch.float32, torch.float16, torch.bfloat16)
        and k.dtype == v.dtype == q.dtype
        and q.device == k.device == v.device == slots.device
        and k.shape == v.shape
        and q.shape[-1] == k.shape[-1]
        and q.shape[-1] in (64, 128, 256)
        and k.shape[1] > 0
        and 0 < q.shape[1] <= _MAX_GRID_PROGRAMS // _SPLITS
        and q.shape[1] % k.shape[1] == 0
        and q.shape[0] == slots.shape[0]
        and q.stride(-1) == k.stride(-1) == v.stride(-1) == 1
        and slots.dtype in (torch.int32, torch.int64)
    )


def sparse_attention(q, k, v, slots, softmax_scale=None):
    """Compute sparse GQA; nonnegative slots must index valid cache rows."""
    if not can_run_sparse_attention(q, k, v, slots):
        raise ValueError("Unsupported NPU sparse attention tensor configuration")
    rows, heads, dim = q.shape
    width = slots.shape[1]
    if rows == 0 or width == 0:
        return torch.zeros_like(q)
    splits = _SPLITS
    block = 32
    # Chunk query rows so both the split grid and the smaller merge grid fit.
    # Keep each row's split reductions unchanged, including in graph capture.
    chunk_rows = min(rows, _MAX_GRID_PROGRAMS // (heads * splits))
    partial = torch.empty(
        (chunk_rows, heads, splits, dim), device=q.device, dtype=torch.float32
    )
    maxima = torch.empty(
        (chunk_rows, heads, splits), device=q.device, dtype=torch.float32
    )
    sums = torch.empty_like(maxima)
    output = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    for start in range(0, rows, chunk_rows):
        count = min(chunk_rows, rows - start)
        query = q[start : start + count]
        indices = slots[start : start + count]
        _sparse_partials[(count, heads, splits)](
            query,
            k,
            v,
            indices,
            partial,
            maxima,
            sums,
            query.stride(0),
            query.stride(1),
            k.stride(0),
            k.stride(1),
            v.stride(0),
            v.stride(1),
            indices.stride(0),
            indices.stride(1),
            heads,
            heads // k.shape[1],
            dim,
            width,
            splits,
            triton.cdiv(width, splits * block),
            softmax_scale or dim**-0.5,
            block,
            enable_fp_fusion=False,
        )
        _merge_partials[(count * heads,)](
            partial,
            maxima,
            sums,
            output[start : start + count],
            dim,
            splits,
            enable_fp_fusion=False,
        )
    return output
