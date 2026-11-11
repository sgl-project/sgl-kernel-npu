"""NPU sparse GQA over physical slots with masked loads and FP32 reductions."""

import torch
import triton
import triton.language as tl

from .qsa_torch_attention import torch_attention

_MAX_GRID_PROGRAMS = 65535


@triton.jit
def _sparse_partials(
    Q,
    K,
    V,
    Slots,
    Partial,
    Maxima,
    Sums,
    Output,
    Q_ROW: tl.constexpr,
    Q_HEAD: tl.constexpr,
    Q_DIM: tl.constexpr,
    K_ROW: tl.constexpr,
    K_HEAD: tl.constexpr,
    K_DIM: tl.constexpr,
    V_ROW: tl.constexpr,
    V_HEAD: tl.constexpr,
    V_DIM: tl.constexpr,
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
    START: tl.constexpr,
    WIDE_OFFSETS: tl.constexpr,
):
    local_item = tl.program_id(0).to(tl.int64)
    item = local_item + START
    row = item // HEADS
    head = item % HEADS
    split = tl.program_id(1)
    dims = tl.arange(0, DIM)
    if WIDE_OFFSETS:
        dims = dims.to(tl.int64)
    query = tl.load(Q + row * Q_ROW + head * Q_HEAD + dims * Q_DIM).to(tl.float32)
    maximum = tl.full((), -float("inf"), tl.float32)
    total = tl.full((), 0, tl.float32)
    numerator = tl.full((DIM,), 0, tl.float32)
    for tile in range(TILES):
        base = split * TILES + tile
        if WIDE_OFFSETS:
            base = base.to(tl.int64)
        cols = base * BLOCK + tl.arange(0, BLOCK)
        slots = tl.load(
            Slots + row * SLOT_ROW + cols * SLOT_COL, cols < WIDTH, other=-1
        )
        valid = (cols < WIDTH) & (slots >= 0)
        # Only negative padding is replaced; nonnegative slots are never clamped.
        if WIDE_OFFSETS:
            safe_slots = tl.maximum(slots, 0).to(tl.int64)
        else:
            safe_slots = tl.maximum(slots, 0).to(tl.int32)
        keys = tl.load(
            K
            + safe_slots[:, None] * K_ROW
            + (head // GROUP) * K_HEAD
            + dims[None, :] * K_DIM,
            valid[:, None],
            other=0,
        ).to(tl.float32)
        values = tl.load(
            V
            + safe_slots[:, None] * V_ROW
            + (head // GROUP) * V_HEAD
            + dims[None, :] * V_DIM,
            valid[:, None],
            other=0,
        ).to(tl.float32)
        scores = tl.sum(keys * query[None, :], axis=1) * SCALE
        # A selected -inf score is an anomaly, not a padding sentinel.
        scores = tl.where(scores == -float("inf"), float("nan"), scores)
        scores = tl.where(valid, scores, -float("inf"))
        next_max = tl.maximum(maximum, tl.max(scores, axis=0))
        safe_max = tl.where(next_max == -float("inf"), 0, next_max)
        correction = tl.exp(maximum - safe_max)
        weights = tl.where(valid, tl.exp(scores - safe_max), 0)
        numerator = numerator * correction + tl.sum(weights[:, None] * values, axis=0)
        total = total * correction + tl.sum(weights, axis=0)
        maximum = next_max
    if SPLITS == 1:
        result = numerator / tl.where(total > 0, total, 1)
        tl.store(Output + item * DIM + dims, result)
    else:
        part = local_item * SPLITS + split
        tl.store(Partial + part * DIM + dims, numerator)
        tl.store(Maxima + part, maximum)
        tl.store(Sums + part, total)


@triton.jit
def _update_head(
    keys,
    values,
    query,
    valid,
    maximum,
    total,
    numerator,
    SCALE: tl.constexpr,
    MASK_WEIGHTS: tl.constexpr = True,
):
    scores = tl.sum(keys * query[None, :], 1) * SCALE
    scores = tl.where(scores == -float("inf"), float("nan"), scores)
    scores = tl.where(valid, scores, -float("inf"))
    next_max = tl.maximum(maximum, tl.max(scores, 0))
    safe_max = tl.where(next_max == -float("inf"), 0, next_max)
    correction = tl.exp(maximum - safe_max)
    # Padding scores already have zero exponential weight for finite live tiles.
    # Valid anomalies still propagate through live weights and accumulators.
    weights = tl.exp(scores - safe_max)
    if MASK_WEIGHTS:
        weights = tl.where(valid, weights, 0)
    numerator = numerator * correction + tl.sum(weights[:, None] * values, 0)
    total = total * correction + tl.sum(weights, 0)
    return next_max, total, numerator


@triton.jit
def _store_head(
    Partial,
    Maxima,
    Sums,
    Output,
    local_item,
    item,
    split,
    maximum,
    total,
    numerator,
    DIM: tl.constexpr,
    SPLITS: tl.constexpr,
):
    dims = tl.arange(0, DIM)
    if SPLITS == 1:
        result = numerator / tl.where(total > 0, total, 1)
        tl.store(Output + item * DIM + dims, result)
    else:
        part = local_item * SPLITS + split
        tl.store(Partial + part * DIM + dims, numerator)
        tl.store(Maxima + part, maximum)
        tl.store(Sums + part, total)


@triton.jit
def _grouped_partials(
    Q,
    K,
    V,
    Slots,
    Partial,
    Maxima,
    Sums,
    Output,
    Q_ROW: tl.constexpr,
    Q_HEAD: tl.constexpr,
    Q_DIM: tl.constexpr,
    K_ROW: tl.constexpr,
    V_ROW: tl.constexpr,
    SLOT_ROW: tl.constexpr,
    SLOT_COL: tl.constexpr,
    DIM: tl.constexpr,
    WIDTH: tl.constexpr,
    SPLITS: tl.constexpr,
    TILES: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
    START_ROW: tl.constexpr,
):
    local_row = tl.program_id(0).to(tl.int64)
    row = local_row + START_ROW
    split = tl.program_id(1)
    dims = tl.arange(0, DIM)
    q0 = tl.load(Q + row * Q_ROW + dims * Q_DIM).to(tl.float32)
    q1 = tl.load(Q + row * Q_ROW + Q_HEAD + dims * Q_DIM).to(tl.float32)
    q2 = tl.load(Q + row * Q_ROW + 2 * Q_HEAD + dims * Q_DIM).to(tl.float32)
    m0 = tl.full((), -float("inf"), tl.float32)
    m1 = tl.full((), -float("inf"), tl.float32)
    m2 = tl.full((), -float("inf"), tl.float32)
    s0 = tl.full((), 0, tl.float32)
    s1 = tl.full((), 0, tl.float32)
    s2 = tl.full((), 0, tl.float32)
    a0 = tl.full((DIM,), 0, tl.float32)
    a1 = tl.full((DIM,), 0, tl.float32)
    a2 = tl.full((DIM,), 0, tl.float32)
    if SPLITS == 1:
        # Locate the last live slot without assuming a valid prefix.
        # Every replay reads current slots, including fully padded rows.
        last_valid = tl.full((), 0, tl.int32)
        scan = tl.full((), (TILES * BLOCK + 255) // 256 - 1, tl.int32)
        while (scan >= 0) & (last_valid == 0):
            offsets = scan * 256 + tl.arange(0, 256)
            scan_cols = split * TILES * BLOCK + offsets
            scan_slots = tl.load(
                Slots + row * SLOT_ROW + scan_cols * SLOT_COL,
                (scan_cols < WIDTH) & (offsets < TILES * BLOCK),
                other=-1,
            )
            last_valid = tl.maximum(
                last_valid, tl.max(tl.where(scan_slots >= 0, offsets + 1, 0), 0)
            )
            scan -= 1
        active_tiles = tl.cdiv(last_valid, BLOCK)
    else:
        active_tiles = TILES
    for tile in range(active_tiles):
        cols = (split * TILES + tile) * BLOCK + tl.arange(0, BLOCK)
        slots = tl.load(
            Slots + row * SLOT_ROW + cols * SLOT_COL, cols < WIDTH, other=-1
        )
        valid = (cols < WIDTH) & (slots >= 0)
        # An empty tile leaves every online-softmax state unchanged. Inspect
        # live slots on device: padding can occur before later valid tiles.
        # This also avoids gather copies introduced before mask selection.
        if tl.max(slots, 0) >= 0:
            safe_slots = tl.maximum(slots, 0).to(tl.int32)
            keys = tl.load(
                K + safe_slots[:, None] * K_ROW + dims[None, :], valid[:, None], other=0
            ).to(tl.float32)
            values = tl.load(
                V + safe_slots[:, None] * V_ROW + dims[None, :], valid[:, None], other=0
            ).to(tl.float32)
            m0, s0, a0 = _update_head(
                keys, values, q0, valid, m0, s0, a0, SCALE, SPLITS != 1
            )
            m1, s1, a1 = _update_head(
                keys, values, q1, valid, m1, s1, a1, SCALE, SPLITS != 1
            )
            m2, s2, a2 = _update_head(
                keys, values, q2, valid, m2, s2, a2, SCALE, SPLITS != 1
            )
    _store_head(
        Partial,
        Maxima,
        Sums,
        Output,
        local_row * 3,
        row * 3,
        split,
        m0,
        s0,
        a0,
        DIM,
        SPLITS,
    )
    _store_head(
        Partial,
        Maxima,
        Sums,
        Output,
        local_row * 3 + 1,
        row * 3 + 1,
        split,
        m1,
        s1,
        a1,
        DIM,
        SPLITS,
    )
    _store_head(
        Partial,
        Maxima,
        Sums,
        Output,
        local_row * 3 + 2,
        row * 3 + 2,
        split,
        m2,
        s2,
        a2,
        DIM,
        SPLITS,
    )


@triton.jit
def _merge_partials(
    Partial,
    Maxima,
    Sums,
    Output,
    DIM: tl.constexpr,
    SPLITS: tl.constexpr,
    START: tl.constexpr,
):
    item = tl.program_id(0).to(tl.int64)
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
    tl.store(Output + (item + START) * DIM + dims, result)


# Full tensor contract and dispatch rationale: sparse_attention below.
_SUPPORTED_HEADS = {(24, 2), (12, 1), (6, 1), (3, 1)}


def _supported_width(width):
    if not 1025 <= width <= 2**31 - 1:
        return False
    for selected_blocks in (512, 2048):
        unit = selected_blocks + 1
        if (width + 1) % unit == 0 and (width + 1) // unit >= 2:
            return True
        # Exists c>=3 with 3<=width+1-unit*c<=c (shared MTP tail).
        low = max(3, (width + 1 + unit) // (unit + 1))
        high = (width - 2) // unit
        if low <= high:
            return True
    return False


def can_run_sparse_attention(q, k, v, slots) -> bool:
    """Check production metadata only; no device-value reads or fallback."""
    return (
        q.device.type == "npu"
        and q.ndim == k.ndim == v.ndim == 3
        and slots.ndim == 2
        and q.dtype == k.dtype == v.dtype == torch.bfloat16
        and q.device == k.device == v.device == slots.device
        and k.shape == v.shape
        and q.shape[2] == k.shape[2] == 256
        and (q.shape[1], k.shape[1]) in _SUPPORTED_HEADS
        and q.shape[0] == slots.shape[0]
        and slots.dtype == torch.int32
        and k.shape[0] <= 2**31 - 1
        and _supported_width(slots.shape[1])
        and k.stride() == v.stride() == (k.shape[1] * 256, 256, 1)
        and q.stride(2) == 1
        and q.stride(1) >= 256
        and q.stride(0) >= (q.shape[1] - 1) * q.stride(1) + 256
        and slots.stride(1) == 1
        and slots.stride(0) >= slots.shape[1]
    )


def _configuration(q, k, v, slots):
    rows = q.shape[0]
    if rows not in (1, 2) and rows < 8:
        return 32, 8
    # Probe a conservative 32-split address bound before choosing a schedule.
    # The wrapper checks the selected launch configuration again.
    if _use_grouped(q, k, False) and not _wide_offsets(q, k, v, slots, 32, 32):
        if rows <= 2:
            return 32, 16
        if rows >= 32:
            return 32, 1
        if rows >= 8:
            return 32, 4
    return 32, 8


def _wide_offsets(q, k, v, slots, block, splits):
    # Host metadata only: bound logical N, padded slot offsets, physical byte
    # spans. A true result disables three-head
    # reuse; the generic Triton path uses int64 offsets in that case.
    limit = 2**31 - 1
    return (
        k.shape[0] > limit
        or slots.shape[1] + block * splits > limit
        or (slots.shape[1] + block * splits) * slots.stride(1) * slots.element_size()
        > limit
        or any(
            sum(max(0, n - 1) * s for n, s in zip(x.shape, x.stride()))
            * x.element_size()
            > limit
            for x in (q, k, v, slots)
        )
    )


def _use_torch(q, k, slots):
    # All accepted widths exceed one tile; only the local head pair matters.
    return (q.shape[1], k.shape[1]) != (3, 1)


def _use_grouped(q, k, wide_offsets):
    return (
        not wide_offsets
        and q.dtype == torch.bfloat16
        and q.shape[1:] == (3, 256)
        and k.shape[1] == 1
    )


def sparse_attention(q, k, v, slots, softmax_scale=None):
    """Compute sparse attention over physical KV rows.

    Q is BF16 [R,Hq,256], K/V are contiguous BF16 [N,Hkv,256], and slots
    is int32 [R,S]. All tensors share one NPU. Q and slots may have row gaps.
    Local Hq:Hkv pairs are 24:2, 12:1, 6:1 and 3:1 for attention TP1/2/4/8.
    S=(B+1)*c-1+t, where B is 512 or 2048, c>=2, and t is 0 or 3..c.
    For B=512,c=4, ordinary calls use S=2051; MTP uses S=2054 or 2055.

    Each query head attends to its selected slots using its shared KV head.
    Negative slots are padding; live slots must be below N. Duplicates retain
    separate softmax contributions. Live Q/K/V must be finite and must not
    overflow. All-padding rows return zero. None or zero scale uses 1/16.
    The output is a new contiguous BF16 tensor with Q's shape.

    For 3:1 heads, Triton reuses each KV tile across three query heads when
    address spans fit int32; otherwise it uses per-head int64 addressing.
    Other head pairs use masked gathers and FP32 matrix operations.
    Triton processes 32 slots per tile. Safe grouped inputs use 16/8/4/1
    splits for R=1..2/3..7/8..31/>=32; wide addresses use 8 splits.
    Few rows need more KV parallelism; more rows need less merging.
    Split softmax states are rescaled to their shared maximum before merging.
    Launches are chunked to stay within the 65535-program grid limit.

    Inputs are unchanged. Warm up before graph capture; keep shapes, strides,
    addresses and scale fixed, and update tensor contents in place for replay.
    """
    if softmax_scale is not None and not isinstance(softmax_scale, (int, float)):
        raise TypeError("softmax_scale must be a Python number or None")
    if not can_run_sparse_attention(q, k, v, slots):
        raise ValueError("Unsupported NPU sparse attention tensor configuration")
    rows, heads, dim = q.shape
    width = slots.shape[1]
    if rows == 0 or k.shape[0] == 0:
        return torch.zeros(q.shape, dtype=q.dtype, device=q.device)
    block, splits = _configuration(q, k, v, slots)
    wide_offsets = _wide_offsets(q, k, v, slots, block, splits)
    grouped = _use_grouped(q, k, wide_offsets)
    if _use_torch(q, k, slots):
        return torch_attention(q, k, v, slots, softmax_scale or dim**-0.5)
    items = rows * heads
    chunk_items = min(items, _MAX_GRID_PROGRAMS // splits)
    if grouped:
        chunk_items = chunk_items // 3 * 3
    output = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    if splits == 1:
        partial = maxima = sums = output
    else:
        partial = torch.empty(
            (chunk_items, splits, dim), device=q.device, dtype=torch.float32
        )
        maxima = torch.empty(
            (chunk_items, splits), device=q.device, dtype=torch.float32
        )
        sums = torch.empty_like(maxima)
    for start in range(0, items, chunk_items):
        count = min(chunk_items, items - start)
        if grouped:
            _grouped_partials[(count // 3, splits)](
                q,
                k,
                v,
                slots,
                partial,
                maxima,
                sums,
                output,
                *q.stride(),
                k.stride(0),
                v.stride(0),
                *slots.stride(),
                dim,
                width,
                splits,
                triton.cdiv(width, splits * block),
                softmax_scale or dim**-0.5,
                block,
                start // 3,
                enable_fp_fusion=False,
                # Direct prefill favors lower buffer pressure; split schedules keep pipelining.
                multibuffer=splits != 1,
            )
        else:
            _sparse_partials[(count, splits)](
                q,
                k,
                v,
                slots,
                partial,
                maxima,
                sums,
                output,
                *q.stride(),
                *k.stride(),
                *v.stride(),
                *slots.stride(),
                heads,
                heads // k.shape[1],
                dim,
                width,
                splits,
                triton.cdiv(width, splits * block),
                softmax_scale or dim**-0.5,
                block,
                start,
                wide_offsets,
                enable_fp_fusion=False,
            )
        if splits != 1:
            _merge_partials[(count,)](
                partial,
                maxima,
                sums,
                output,
                dim,
                splits,
                start,
                enable_fp_fusion=False,
            )
    return output
