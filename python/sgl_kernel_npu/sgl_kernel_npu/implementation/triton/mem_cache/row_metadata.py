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
    metadata,
    current_valid,
    current_src,
    current_dst,
    current_out_valid,
    host_mask,
    HAS_CURRENT: tl.constexpr,
    CURRENT_VALID_STRIDE: tl.constexpr,
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
    if HAS_CURRENT:
        current_start = tl.load(metadata + r)
        prefix = tl.load(metadata + B + r)
        extend = tl.load(metadata + 2 * B + r)
        width = tl.load(metadata + 3 * B + r)
        block_start = tl.program_id(1) * BLOCK
        use_device = tl.full((BLOCK,), False, tl.int1)
        if (block_start < prefix + extend) & (block_start + BLOCK > prefix):
            in_current = (pos >= prefix) & (pos < prefix + extend)
            idx = current_start + pos - prefix
            current_live = (
                tl.load(
                    current_valid + idx * CURRENT_VALID_STRIDE,
                    live & in_current,
                    other=0,
                )
                != 0
            )
            use_device = live & in_current & (req > 0) & current_live
        tl.store(host_mask + start + pos, ~use_device, live)
        if block_start < width:
            local_valid = (
                tl.load(
                    current_valid + (current_start + pos) * CURRENT_VALID_STRIDE,
                    pos < width,
                    other=0,
                )
                != 0
            )
            local_live = (pos < extend) & (req > 0) & local_valid
            tl.store(
                current_src + current_start + pos, current_start + pos, pos < width
            )
            tl.store(
                current_dst + current_start + pos,
                tl.where(local_live, start + prefix + pos, 0),
                pos < width,
            )
            tl.store(current_out_valid + current_start + pos, local_live, pos < width)
    physical = tl.load(table + safe_req * S0 + safe_pos * S1, live, other=0)
    out = start + pos
    tl.store(source + out, physical.to(tl.int64), live)
    tl.store(destination + out, out, live)
    tl.store(valid + out, True, live)


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
    requests,
    lengths,
    table,
    total,
    max_length,
    request_rows,
    context_len,
    *,
    current_metadata=None,
    current_valid=None,
    max_current_width=0,
):
    """Build host and optional current-device row mappings in one launch."""
    source = torch.empty(total, dtype=torch.long, device=table.device)
    destination = torch.empty_like(source)
    valid = torch.empty(total, dtype=torch.bool, device=table.device)
    has_current = current_metadata is not None
    if has_current != (current_valid is not None):
        raise ValueError("current metadata and validity must be supplied together")
    mapping = None
    if has_current:
        n = current_valid.numel()
        current_src = torch.empty(n, dtype=torch.long, device=table.device)
        current_dst = torch.empty_like(current_src)
        current_out_valid = torch.empty(n, dtype=torch.bool, device=table.device)
        host_mask = torch.empty(total, dtype=torch.bool, device=table.device)
        mapping = (current_src, current_dst, current_out_valid, host_mask)
    else:
        current_src = current_dst = current_out_valid = host_mask = None
    width = max(max_length, max_current_width)
    if width and requests.numel():
        b = requests.numel()
        _prefill_plan_kernel[(b, triton.cdiv(width, 256))](
            requests,
            lengths,
            table,
            source,
            destination,
            valid,
            current_metadata,
            current_valid,
            current_src,
            current_dst,
            current_out_valid,
            host_mask,
            has_current,
            current_valid.stride(0) if has_current else 0,
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
    return source, destination, valid, mapping


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
