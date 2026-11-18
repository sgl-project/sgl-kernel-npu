"""Bounded BF16 hybrid path: masked gather and FP32 matrix softmax."""

import torch
import triton
import triton.language as tl


@triton.jit
def _gather(
    K,
    V,
    Slots,
    Keys,
    Values,
    K_ROW: tl.constexpr,
    K_HEAD: tl.constexpr,
    V_ROW: tl.constexpr,
    V_HEAD: tl.constexpr,
    SLOT_ROW: tl.constexpr,
    SLOT_COL: tl.constexpr,
    WIDTH: tl.constexpr,
    DIM: tl.constexpr,
    KV_HEAD,
    START_ROW,
    START_ITEM,
    ITEMS: tl.constexpr,
):
    item = START_ITEM + tl.program_id(0).to(tl.int64) * 32 + tl.arange(0, 32)
    row = item // WIDTH + START_ROW
    col = item % WIDTH
    slots = tl.load(Slots + row * SLOT_ROW + col * SLOT_COL, item < ITEMS, other=-1)
    valid = (item < ITEMS) & (slots >= 0)
    safe = tl.maximum(slots, 0).to(tl.int64)
    kv_head = tl.full((), 0, tl.int64) + KV_HEAD
    dims = tl.arange(0, DIM)
    keys = tl.load(
        K + safe[:, None] * K_ROW + kv_head * K_HEAD + dims[None, :],
        valid[:, None],
        other=0,
    ).to(tl.float32)
    values = tl.load(
        V + safe[:, None] * V_ROW + kv_head * V_HEAD + dims[None, :],
        valid[:, None],
        other=0,
    ).to(tl.float32)
    tl.store(Keys + item[:, None] * DIM + dims[None, :], keys, (item < ITEMS)[:, None])
    tl.store(
        Values + item[:, None] * DIM + dims[None, :], values, (item < ITEMS)[:, None]
    )


def torch_attention(q, k, v, slots, scale):
    """Compute selected attention with masked gathers and FP32 matrices.

    The caller validates BF16 Q[R,Hq,256], K/V[N,Hkv,256] and int32
    slots[R,S], handles empty inputs, and supplies the softmax scale.
    This path handles local head pairs 24:2, 12:1 and 6:1. KV heads are
    folded into the matrix batch; the full KV pool is never copied.

    Gather selected K/V, compute softmax(scale * QK), multiply by V, and
    cast the output to BF16. Negative slots are masked before gathering;
    all-padding rows return zero. Duplicate slots remain separate columns.

    Row chunks target 16 MiB for two FP32 gather buffers (8 bytes per
    entry per dimension), with at least one row. A single wide row may
    exceed that target. Gather tasks copy 32 entries and launches are
    capped at 65535 programs. Warm up before capturing fixed-shape replay.
    """
    if k.shape[1] > 1:
        return _batched_attention(q, k, v, slots, scale)
    rows, heads, dim = q.shape
    width = slots.shape[1]
    # Bound gathers independently of total row count; no per-row Torch loop.
    chunk_rows = max(1, min(32, (16 * 1024**2) // (width * dim * 8)))
    output = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    for start in range(0, rows, chunk_rows):
        count = min(chunk_rows, rows - start)
        keys = torch.empty((count, width, dim), device=q.device, dtype=torch.float32)
        values = torch.empty_like(keys)
        valid = slots[start : start + count] >= 0
        items = count * width
        for item_start in range(0, items, 65535 * 32):
            programs = min(65535, triton.cdiv(items - item_start, 32))
            _gather[(programs,)](
                k,
                v,
                slots,
                keys,
                values,
                k.stride(0),
                k.stride(1),
                v.stride(0),
                v.stride(1),
                *slots.stride(),
                width,
                dim,
                0,
                start,
                item_start,
                items,
                enable_fp_fusion=False,
            )
        query = q[start : start + count].float()
        scores = torch.bmm(query, keys.transpose(1, 2)) * scale
        scores = scores.masked_fill(scores == -float("inf"), float("nan"))
        scores = scores.masked_fill(~valid[:, None, :], -float("inf"))
        probabilities = torch.softmax(scores, -1)
        probabilities = torch.where(valid[:, None, :], probabilities, 0.0)
        result = torch.bmm(probabilities, values).to(q.dtype)
        output[start : start + count].copy_(result)
    return output


@triton.jit
def _gather_heads(
    K,
    V,
    Slots,
    Keys,
    Values,
    K_ROW: tl.constexpr,
    K_HEAD: tl.constexpr,
    V_ROW: tl.constexpr,
    V_HEAD: tl.constexpr,
    SLOT_ROW: tl.constexpr,
    SLOT_COL: tl.constexpr,
    WIDTH: tl.constexpr,
    DIM: tl.constexpr,
    KV_COUNT: tl.constexpr,
    HEAD_START,
    START_ROW,
    START_ITEM,
    ITEMS: tl.constexpr,
):
    item = START_ITEM + tl.program_id(0).to(tl.int64) * 32 + tl.arange(0, 32)
    row = item // (WIDTH * KV_COUNT) + START_ROW
    col = item % WIDTH
    head = item // WIDTH % KV_COUNT + HEAD_START
    slots = tl.load(Slots + row * SLOT_ROW + col * SLOT_COL, item < ITEMS, other=-1)
    valid = (item < ITEMS) & (slots >= 0)
    safe = tl.maximum(slots, 0).to(tl.int64)
    dims = tl.arange(0, DIM)
    keys = tl.load(
        K + safe[:, None] * K_ROW + head[:, None] * K_HEAD + dims[None, :],
        valid[:, None],
        other=0,
    ).to(tl.float32)
    values = tl.load(
        V + safe[:, None] * V_ROW + head[:, None] * V_HEAD + dims[None, :],
        valid[:, None],
        other=0,
    ).to(tl.float32)
    tl.store(Keys + item[:, None] * DIM + dims[None, :], keys, (item < ITEMS)[:, None])
    tl.store(
        Values + item[:, None] * DIM + dims[None, :], values, (item < ITEMS)[:, None]
    )


def _batched_attention(q, k, v, slots, scale):
    rows, heads, dim = q.shape
    kv_heads = k.shape[1]
    group = heads // kv_heads
    width = slots.shape[1]
    kv_chunk = kv_heads
    chunk_rows = max(1, min(32, (16 * 1024**2) // (width * dim * 8 * kv_chunk)))
    output = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    output_heads = output.view(rows, kv_heads, group, dim)
    for start in range(0, rows, chunk_rows):
        count = min(chunk_rows, rows - start)
        valid = slots[start : start + count] >= 0
        keys = torch.empty(
            (count * kv_heads, width, dim), device=q.device, dtype=torch.float32
        )
        values = torch.empty_like(keys)
        items = count * kv_heads * width
        for item_start in range(0, items, 65535 * 32):
            programs = min(65535, triton.cdiv(items - item_start, 32))
            _gather_heads[(programs,)](
                k,
                v,
                slots,
                keys,
                values,
                k.stride(0),
                k.stride(1),
                v.stride(0),
                v.stride(1),
                *slots.stride(),
                width,
                dim,
                kv_heads,
                0,
                start,
                item_start,
                items,
                enable_fp_fusion=False,
            )
        query = q[start : start + count].float().reshape(count * kv_heads, group, dim)
        scores = (torch.bmm(query, keys.transpose(1, 2)) * scale).reshape(
            count, kv_heads, group, width
        )
        scores = scores.masked_fill(scores == -float("inf"), float("nan"))
        scores = scores.masked_fill(~valid[:, None, None, :], -float("inf"))
        probabilities = torch.softmax(scores, -1)
        probabilities = torch.where(valid[:, None, None, :], probabilities, 0.0)
        result = torch.bmm(
            probabilities.reshape(count * kv_heads, group, width), values
        ).to(q.dtype)
        output_heads[start : start + count].copy_(
            result.reshape(count, kv_heads, group, dim)
        )
    return output
