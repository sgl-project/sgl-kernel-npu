"""Tensor-only Triton row metadata for sparse KV and split MemFabric copies."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _mtp_rows_kernel(
    requests,
    seq_lens,
    positions,
    locations,
    counts,
    slots_out,
    positions_out,
    valid_out,
    destination_out,
    N: tl.constexpr,
    OUT_N: tl.constexpr,
    Q: tl.constexpr,
    REQUEST_ROWS: tl.constexpr,
    CONTEXT: tl.constexpr,
    HAS_LOC: tl.constexpr,
    DRAFT: tl.constexpr,
    REQ_STRIDE: tl.constexpr,
    SEQ_STRIDE: tl.constexpr,
    POS_STRIDE: tl.constexpr,
    LOC_STRIDE: tl.constexpr,
    COUNT_STRIDE: tl.constexpr,
    PUBLISH: tl.constexpr,
    HOST_CAPACITY: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = i // Q
    req = tl.load(requests + row * REQ_STRIDE, i < N, other=0).to(tl.int64)
    length = tl.load(seq_lens + row * SEQ_STRIDE, i < N, other=0)
    pos = tl.load(positions + i * POS_STRIDE, i < N, other=-1).to(tl.int64)
    live = (req > 0) & (req < REQUEST_ROWS) & (length > 0)
    live = live & (pos >= 0) & (pos < CONTEXT)
    if DRAFT:
        count = tl.load(counts + row * COUNT_STRIDE, i < N, other=0)
        live = live & (i % Q < count)
    if HAS_LOC:
        loc = tl.load(locations + i * LOC_STRIDE, i < OUT_N, other=-1).to(tl.int64)
        live = live & (loc >= 0)
    if PUBLISH:
        live = live & (loc < HOST_CAPACITY) & (i < N)
        tl.store(slots_out + i, i, i < OUT_N)
        tl.store(destination_out + i, loc, i < OUT_N)
    else:
        tl.store(slots_out + i, req, i < OUT_N)
    tl.store(positions_out + i, pos, i < OUT_N)
    tl.store(valid_out + i, live, i < OUT_N)


@triton.jit
def _prefill_plan_kernel(
    requests,
    lengths,
    table,
    source,
    destination,
    valid,
    B: tl.constexpr,
    ROWS: tl.constexpr,
    CONTEXT: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    REQ_STRIDE: tl.constexpr,
    LEN_STRIDE: tl.constexpr,
    BATCH_BLOCK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    r = tl.program_id(0)
    ids = tl.arange(0, BATCH_BLOCK)
    reqs = tl.load(requests + ids * REQ_STRIDE, ids < B, other=-1)
    lens = tl.load(lengths + ids * LEN_STRIDE, ids < B, other=0).to(tl.int64)
    lens = tl.where((reqs >= 0) & (lens > 0), lens, 0)
    start = tl.sum(tl.where(ids < r, lens, 0), 0)
    req = tl.load(requests + r * REQ_STRIDE).to(tl.int64)
    length = tl.load(lengths + r * LEN_STRIDE)
    pos = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    live = (req >= 0) & (length > 0) & (pos < length)
    safe_req = tl.minimum(tl.maximum(req, 0), ROWS - 1)
    safe_pos = tl.minimum(pos, CONTEXT - 1)
    physical = tl.load(table + safe_req * S0 + safe_pos * S1, live, other=0)
    out = start + pos
    tl.store(source + out, physical.to(tl.int64), live)
    tl.store(destination + out, out, live)
    tl.store(valid + out, True, live)


@triton.jit
def _mapping_init_kernel(
    counts, host_mask, CAPACITY: tl.constexpr, TOTAL: tl.constexpr, BLOCK: tl.constexpr
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(counts + i, 0, i < CAPACITY)
    tl.store(host_mask + i, True, i < TOTAL)


@triton.jit
def _mapping_count_kernel(
    metadata,
    locations,
    current_valid,
    counts,
    src_out,
    dst_out,
    valid_out,
    N: tl.constexpr,
    B: tl.constexpr,
    CAPACITY: tl.constexpr,
    BATCH_BLOCK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    # Metadata is O(B): current-row end, full-KV start, prefix, live-row end.
    rows = tl.arange(0, BATCH_BLOCK)
    ends = tl.load(metadata + rows, rows < B, other=0)
    owner = tl.sum(
        ((i[:, None] >= ends[None, :]) & (rows[None, :] < B)).to(tl.int32), 1
    )
    previous = tl.load(metadata + owner - 1, (i < N) & (owner > 0), other=0)
    start = tl.load(metadata + B + owner, i < N, other=0)
    prefix = tl.load(metadata + 2 * B + owner, i < N, other=0)
    live_end = tl.load(metadata + 3 * B + owner, i < N, other=0)
    eligible = i < live_end
    dst = tl.where(eligible, start + prefix + i - previous, 0).to(tl.int64)
    loc = tl.load(locations + i, i < N, other=-1).to(tl.int64)
    live = tl.load(current_valid + i, i < N, other=0) != 0
    live = live & eligible & (loc >= 0) & (loc < CAPACITY) & (i < N)
    tl.atomic_add(counts + tl.minimum(tl.maximum(loc, 0), CAPACITY - 1), 1, live)
    tl.store(src_out + i, i, i < N)
    tl.store(dst_out + i, dst, i < N)
    tl.store(valid_out + i, live, i < N)


@triton.jit
def _mapping_finish_kernel(
    source,
    locations,
    counts,
    destination,
    valid,
    host_mask,
    N: tl.constexpr,
    CAPACITY: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = tl.load(valid + i, i < N, other=0) != 0
    loc = tl.load(locations + i, i < N, other=0).to(tl.int64)
    safe = tl.minimum(tl.maximum(loc, 0), CAPACITY - 1)
    count = tl.load(counts + safe, i < N, other=0)
    dst = tl.load(destination + i, i < N, other=0)
    physical = tl.load(source + dst, i < N, other=-1)
    live = live & (count == 1) & (physical == loc)
    tl.store(valid + i, live, i < N)
    # CPU layout validation guarantees unique destinations for eligible rows.
    tl.store(host_mask + dst, False, (i < N) & live)


@triton.jit
def _copy_rows_metadata_kernel(
    source_rows,
    destination_rows,
    valid,
    src,
    dst,
    lengths,
    count,
    source_base_0,
    source_base_1,
    destination_base_0,
    destination_base_1,
    WIDTH_0: tl.constexpr,
    WIDTH_1: tl.constexpr,
    N: tl.constexpr,
    SRC_STRIDE: tl.constexpr,
    DST_STRIDE: tl.constexpr,
    VALID_STRIDE: tl.constexpr,
    COMPONENT_GRID: tl.constexpr,
    BLOCK: tl.constexpr,
):
    if COMPONENT_GRID:
        component = tl.program_id(1)
        row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        i = component * N + row
        mask = row < N
        first = component == 0
    else:
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        row = i % N
        mask = i < 2 * N
        first = i < N
    live = tl.load(valid + row * VALID_STRIDE, mask, other=0) != 0
    s = tl.load(source_rows + row * SRC_STRIDE, mask, other=0).to(tl.int64)
    d = tl.load(destination_rows + row * DST_STRIDE, mask, other=0).to(tl.int64)
    s, d = tl.where(live, s, 0), tl.where(live, d, 0)
    width = tl.where(first, WIDTH_0, WIDTH_1)
    tl.store(src + i, tl.where(first, source_base_0, source_base_1) + s * width, mask)
    tl.store(
        dst + i,
        tl.where(first, destination_base_0, destination_base_1) + d * width,
        mask,
    )
    tl.store(lengths + i, tl.where(live, width, 0), mask)
    if tl.program_id(0) == 0:
        if COMPONENT_GRID:
            if tl.program_id(1) == 0:
                tl.store(count, N * 2)
        else:
            tl.store(count, N * 2)


def mtp_rows(
    requests,
    seq_lens,
    input_positions,
    locations,
    counts,
    query_width,
    request_rows,
    context_len,
    *,
    publication_rows=None,
    host_capacity=0,
):
    """Materialize request IDs, positions and masks in a single device launch."""
    if not isinstance(query_width, int) or query_width <= 0 or requests.ndim != 1:
        raise ValueError("MTP requires a positive query width and 1-D request IDs")
    if seq_lens.shape != requests.shape or requests.numel() == 0:
        raise ValueError("MTP request IDs and sequence lengths must match")
    logical_tokens = requests.numel() * query_width
    if input_positions.numel() < logical_tokens:
        raise ValueError("MTP positions cannot truncate the logical query window")
    if locations is not None and locations.numel() < logical_tokens:
        raise ValueError("MTP locations cannot truncate the logical query window")
    if counts is not None and counts.shape != requests.shape:
        raise ValueError("MTP acceptance counts must match the request batch")
    if publication_rows is not None and (
        publication_rows < logical_tokens
        or locations is None
        or locations.numel() != publication_rows
        or host_capacity <= 0
    ):
        raise ValueError("MTP publication requires complete physical rows and capacity")
    shape = (
        (requests.numel(), query_width)
        if publication_rows is None
        else (publication_rows,)
    )
    slots = torch.empty(shape, dtype=torch.long, device=input_positions.device)
    positions = torch.empty_like(slots)
    valid = torch.empty(shape, dtype=torch.bool, device=slots.device)
    destination = torch.empty_like(slots) if publication_rows is not None else None
    draft = counts is not None
    input_positions = input_positions.reshape(-1)
    locations = locations.reshape(-1) if locations is not None else None
    _mtp_rows_kernel[(triton.cdiv(slots.numel(), 32),)](
        requests,
        seq_lens,
        input_positions,
        locations,
        counts,
        slots,
        positions,
        valid,
        destination,
        logical_tokens,
        slots.numel(),
        query_width,
        request_rows,
        context_len,
        locations is not None,
        draft,
        requests.stride(0),
        seq_lens.stride(0),
        input_positions.stride(0),
        locations.stride(0) if locations is not None else 0,
        counts.stride(0) if counts is not None else 0,
        publication_rows is not None,
        host_capacity,
        32,
    )
    if publication_rows is not None:
        return slots, destination, positions, valid
    return slots, positions, valid


def prefill_plan(
    requests, lengths, table, total, max_length, request_rows, context_len
):
    """Expand ragged request rows directly through the physical token map."""
    source = torch.empty(total, dtype=torch.long, device=table.device)
    destination = torch.empty_like(source)
    valid = torch.empty(total, dtype=torch.bool, device=table.device)
    if total:
        b = requests.numel()
        _prefill_plan_kernel[(b, triton.cdiv(max_length, 256))](
            requests,
            lengths,
            table,
            source,
            destination,
            valid,
            b,
            request_rows,
            context_len,
            table.stride(0),
            table.stride(1),
            requests.stride(0),
            lengths.stride(0),
            triton.next_power_of_2(b),
            256,
        )
    return source, destination, valid


def prefill_current_mapping(
    source, locations, current_valid, metadata, capacity, total, batch_size
):
    """Resolve unique current rows after CPU layout metadata has been uploaded."""
    n, b = current_valid.numel(), batch_size
    device = source.device
    src = torch.empty(n, device=device, dtype=torch.long)
    dst = torch.empty_like(src)
    valid = torch.empty(n, device=device, dtype=torch.bool)
    host_mask = torch.empty(total, device=device, dtype=torch.bool)
    counts = torch.empty(capacity, device=device, dtype=torch.int32)
    _mapping_init_kernel[(triton.cdiv(max(capacity, total), 512),)](
        counts,
        host_mask,
        capacity,
        total,
        512,
    )
    _mapping_count_kernel[(triton.cdiv(n, 32),)](
        metadata,
        locations,
        current_valid,
        counts,
        src,
        dst,
        valid,
        n,
        b,
        capacity,
        triton.next_power_of_2(b),
        32,
    )
    _mapping_finish_kernel[(triton.cdiv(n, 32),)](
        source,
        locations,
        counts,
        dst,
        valid,
        host_mask,
        n,
        capacity,
        32,
    )
    return src, dst, valid, host_mask


def copy_rows_metadata(sources, destinations, source_rows, destination_rows, valid):
    """Build split CKV/KPE byte descriptors without touching host KV."""
    if len(sources) != 2 or len(destinations) != 2:
        raise ValueError("Fused MF descriptors require two split KV components")
    n = source_rows.numel()
    device = source_rows.device
    src = torch.empty(n * len(sources), dtype=torch.int64, device=device)
    dst = torch.empty_like(src)
    lengths = torch.empty(src.shape, dtype=torch.int32, device=device)
    count = torch.empty(1, dtype=torch.int32, device=device)
    # Keep component boundaries aligned to the 32-byte int32 output cache line.
    component_grid = n % 8 == 0
    grid = (triton.cdiv(n, 32), 2) if component_grid else (triton.cdiv(2 * n, 32),)
    _copy_rows_metadata_kernel[grid](
        source_rows,
        destination_rows,
        valid,
        src,
        dst,
        lengths,
        count,
        sources[0].data_ptr(),
        sources[1].data_ptr(),
        destinations[0].data_ptr(),
        destinations[1].data_ptr(),
        sources[0].shape[-1] * sources[0].element_size(),
        sources[1].shape[-1] * sources[1].element_size(),
        n,
        source_rows.stride(0),
        destination_rows.stride(0),
        valid.stride(0),
        component_grid,
        32,
    )
    return src, dst, lengths, count
