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
    """Compute the production hybrid attention after wrapper metadata checks.

    Chunk shapes and computation
    ----------------------------
        Q       [R,Hq,256]       K/V pool [N,Hkv,256]
        slots   [R,S]           output   [R,Hq,256], contiguous BF16

    The wrapper owns the full input, numerical and graph contract, handles
    empty inputs and scale defaults, and dispatches TP1/2/4 here:
    (Hq,Hkv)=(24,2),(12,1),(6,1). This helper is not an independent reference
    fallback. K/V are contiguous pool views; the pool is never copied.

    Let C=Hkv (1 or 2), G=Hq/C (6 or 12), and b be rows in the current chunk.
    For C=1, Triton gathers selected keys and values as FP32 [b,S,256]:
        query [b,G,256] @ keys.transpose(1,2) [b,256,S]
            -> scores [b,G,S]
        probabilities = softmax(scale*scores, along S, padding masked)
        probabilities [b,G,S] @ values [b,S,256] -> [b,G,256]

    For C=2, fold row and KV-head into a batch of b*C:
        query [b*C,G,256], gathered keys/values [b*C,S,256]
        scores reshape to [b,C,G,S] to broadcast the [b,S] padding mask
        probabilities reshape back to [b*C,G,S] for the final bmm
        result [b*C,G,256] -> [b,C,G,256] -> output slice [b,Hq,256]
    Both matrix products and softmax use FP32; only the result is cast to BF16.
    G groups consecutive query heads sharing one KV head; duplicates in slots
    remain separate matrix columns and retain their softmax contributions.

    Row chunks and gather launches
    ------------------------------
        chunk_rows = max(1, min(32, (16*1024**2) // (S*256*8*C)))
    The 8 bytes per element are two FP32 gather buffers (K and V). Always
    process at least one row; the last chunk may be smaller. For S=2051:
    C=1 permits 3 rows per chunk, C=2 permits 1. Thus Q=[32,24,256] uses
    32 one-row chunks with two KV heads folded into each matrix batch.

    16 MiB is only a gather target, not total temporary or graph memory.
    A single large-S row can exceed it. Output, query conversion, scores,
    probabilities, masks and other intermediates are additional allocations.
    Chunking is resource management within the hybrid, not another algorithm.

    Each Triton gather task copies 32 selected entries of width 256. The
    flattened entry count is b*C*S; at most 65535 tasks are launched at once,
    with further launches advancing the entry offset. Pool and entry offsets
    use int64. Negative slots are masked before loads, and invalid entries
    are written as zero in the gathered buffers; slot 0 is never a dummy row.

    Padding, anomalies and replay
    -----------------------------
    Computed -Inf scores are changed to NaN before the padding mask supplies
    its own -Inf sentinel. After softmax, invalid probabilities are set to
    zero. This preserves exact all-padding zeros without clearing anomalies
    in valid data. Nonfinite values otherwise follow arithmetic; no pool scan,
    host synchronization, exception recovery, or service-stop guarantee is added.

    Fixed-shape gathers and matrices read updated Q/K/V/slots on graph replay.
    The public wrapper's warmup, fixed metadata and control requirements apply.
    Measurements justify retaining this path for other TP head pairs, with
    eager and memory costs; mathematics does not require a separate hybrid.
    Full-wrapper baselines, regressions and limits are in DELIVERY.md.
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
