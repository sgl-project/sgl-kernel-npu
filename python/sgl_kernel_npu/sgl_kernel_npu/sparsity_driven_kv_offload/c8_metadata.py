"""Tensor-only A5 C8 metadata and causal ring staging operators."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _prepare_kernel(
    seq_ptr,
    positions_ptr,
    locations_ptr,
    slot_ptr,
    last_ptr,
    src_table_ptr,
    out_table_ptr,
    safe_slot_ptr,
    length_ptr,
    prefix_ptr,
    sparse_ptr,
    valid_ptr,
    state_ptr,
    next_prefix_ptr,
    cache_size_ptr,
    ROWS: tl.constexpr,
    CAPACITY: tl.constexpr,
    QUERY_WIDTH: tl.constexpr,
    HAS_POSITIONS: tl.constexpr,
    FULL_VISIBLE: tl.constexpr,
    HAS_LOCATIONS: tl.constexpr,
    CONTEXT: tl.constexpr,
    HOST_CAPACITY: tl.constexpr,
    SRC_COLS: tl.constexpr,
    DST_COLS: tl.constexpr,
    SRC_S0: tl.constexpr,
    SRC_S1: tl.constexpr,
    DST_S0: tl.constexpr,
    DST_S1: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tile = tl.program_id(0).to(tl.int32)
    row = tl.program_id(1).to(tl.int32)
    cols = tile * BLOCK + tl.arange(0, BLOCK).to(tl.int32)

    # Every destination column is assigned once: preserve every source column,
    # including PAD columns, and zero only the destination's trailing columns.
    if tile * BLOCK < DST_COLS:
        value = tl.load(
            src_table_ptr + row * SRC_S0 + cols * SRC_S1,
            mask=(cols < SRC_COLS) & (cols < DST_COLS),
            other=0,
        ).to(tl.int32)
        tl.store(
            out_table_ptr + row * DST_S0 + cols * DST_S1, value, mask=cols < DST_COLS
        )

    if tile == 0:
        raw_slot = tl.load(slot_ptr + row).to(tl.int64)
        active_length = tl.load(seq_ptr + row).to(tl.int64)
        raw_length = active_length
        if HAS_POSITIONS:
            raw_length = (
                tl.load(positions_ptr + row * QUERY_WIDTH + QUERY_WIDTH - 1).to(
                    tl.int64
                )
                + 1
            )
        valid = (
            (raw_slot > 0) & (raw_slot < ROWS) & (active_length > 0) & (raw_length > 0)
        )
        if not FULL_VISIBLE:
            first = raw_length - QUERY_WIDTH
            valid = valid & (first >= 0) & (raw_length <= CONTEXT)
            if HAS_POSITIONS:
                for q in tl.static_range(QUERY_WIDTH):
                    pos = tl.load(positions_ptr + row * QUERY_WIDTH + q)
                    valid = valid & (pos == first + q)
            if HAS_LOCATIONS:
                for q in tl.static_range(QUERY_WIDTH):
                    loc = tl.load(locations_ptr + row * QUERY_WIDTH + q).to(tl.int64)
                    valid = valid & (loc >= 0) & (loc < HOST_CAPACITY)
        safe_slot = tl.where(valid, raw_slot, ROWS).to(tl.int32)
        length = tl.where(valid, raw_length, 0).to(tl.int32)
        prefix = length
        if not FULL_VISIBLE:
            prefix = tl.maximum(length - QUERY_WIDTH, 0) // 128 * 128
        sparse = valid & (prefix >= CAPACITY)
        previous = tl.load(last_ptr + safe_slot).to(tl.int32)
        warm = sparse & (previous >= 0) & (prefix >= previous)
        state = tl.where(sparse, tl.where(warm, -1, -2), -3).to(tl.int32)
        next_prefix = tl.where(sparse, prefix, -1).to(tl.int32)
        cache_size = tl.where(sparse, CAPACITY, 0).to(tl.int32)
        tl.store(safe_slot_ptr + row, safe_slot)
        tl.store(length_ptr + row, length)
        tl.store(prefix_ptr + row, prefix)
        tl.store(sparse_ptr + row, sparse)
        tl.store(valid_ptr + row, valid)
        tl.store(state_ptr + row, state)
        tl.store(next_prefix_ptr + row, next_prefix)
        tl.store(cache_size_ptr + row, cache_size)


def fused_prepare_lim_metadata(
    *,
    seq_lens,
    request_slots,
    last_prefix,
    block_table,
    out_block_table,
    safe_slots,
    lengths,
    prefixes,
    sparse_rows,
    valid_rows,
    request_states,
    next_prefixes,
    cache_sizes,
    capacity: int,
    request_rows: int,
    query_width: int = 1,
    positions=None,
    full_visible: bool = True,
    locations=None,
    context_len: int = 2**21,
    host_capacity: int = 0,
) -> None:
    """Prepare C8 positions, physical bounds and ring-prefix state in one launch."""
    batch = seq_lens.numel()
    cols = out_block_table.shape[1]
    block = 128
    _prepare_kernel[(triton.cdiv(cols, block), batch)](
        seq_lens,
        positions if positions is not None else seq_lens,
        locations if locations is not None else seq_lens,
        request_slots,
        last_prefix,
        block_table,
        out_block_table,
        safe_slots,
        lengths,
        prefixes,
        sparse_rows,
        valid_rows,
        request_states,
        next_prefixes,
        cache_sizes,
        request_rows,
        capacity,
        query_width,
        positions is not None,
        full_visible,
        locations is not None,
        context_len,
        host_capacity,
        block_table.shape[1],
        cols,
        block_table.stride(0),
        block_table.stride(1),
        out_block_table.stride(0),
        out_block_table.stride(1),
        block,
    )


@triton.jit
def _lim_copy_sfa_metadata_kernel(
    source_ids_ptr,
    device_slots_ptr,
    topk_misses_ptr,
    miss_counts_ptr,
    valid_rows_ptr,
    sparse_rows_ptr,
    request_slots_ptr,
    actual_seq_lengths_kv_ptr,
    num_cache_tokens_ptr,
    hbm_block_table_ptr,
    dram_block_table_ptr,
    host_block_table_ptr,
    seq_lens_ptr,
    prefix_lens_ptr,
    K: tl.constexpr,
    Q: tl.constexpr,
    C: tl.constexpr,
    HBM_PAGES: tl.constexpr,
    DRAM_PAGES: tl.constexpr,
    BLOCK: tl.constexpr,
    PAGED_HOST: tl.constexpr,
    HOST_COLS: tl.constexpr,
    HOST_S0: tl.constexpr,
    HOST_S1: tl.constexpr,
    PACKED_C8: tl.constexpr,
    HAS_DEVICE_SLOTS: tl.constexpr,
):
    tile_id = tl.program_id(0).to(tl.int32)
    row_id = tl.program_id(1).to(tl.int32)

    # Triton Ascend exposes torch.bool loads as int8; compare explicitly so
    # masks passed to tl.store and tl.where have boolean scalar type.
    is_valid = tl.load(valid_rows_ptr + row_id) != 0
    is_sparse = tl.load(sparse_rows_ptr + row_id) != 0
    request_slot = tl.load(request_slots_ptr + row_id).to(tl.int32)
    active = is_valid & is_sparse
    base_request = tl.where(is_valid, request_slot, 0).to(tl.int32)

    col = tile_id * BLOCK + tl.arange(0, BLOCK).to(tl.int32)
    if tile_id * BLOCK < K:
        for route in range(Q):
            source_base = (row_id.to(tl.int64) * Q + route) * K
            tl.store(
                source_ids_ptr + source_base + col.to(tl.int64),
                -1,
                mask=(col < K) & ~is_valid,
            )
            if HAS_DEVICE_SLOTS:
                tl.store(
                    device_slots_ptr + source_base + col.to(tl.int64),
                    -1,
                    mask=(col < K) & ~is_valid,
                )

    if tile_id * BLOCK < HBM_PAGES:
        hbm_base = row_id.to(tl.int64) * HBM_PAGES
        physical_col = col
        if PACKED_C8:
            prefix = tl.load(prefix_lens_ptr + row_id)
            cache_pages = C // 128
            physical_col = tl.where(
                col < cache_pages,
                col,
                cache_pages + (prefix // 128 + col - cache_pages) % 2,
            )
        hbm_values = base_request * HBM_PAGES + physical_col
        tl.store(
            hbm_block_table_ptr + hbm_base + col.to(tl.int64),
            hbm_values,
            mask=col < HBM_PAGES,
        )

    if tile_id * BLOCK < DRAM_PAGES:
        dram_base = row_id.to(tl.int64) * DRAM_PAGES
        if PAGED_HOST:
            pages = tl.load(
                host_block_table_ptr + row_id * HOST_S0 + col * HOST_S1,
                mask=col < HOST_COLS,
                other=0,
            ).to(tl.int32)
            dram_values = tl.where(is_valid, pages, 0)
        else:
            dram_values = base_request * DRAM_PAGES + col
        tl.store(
            dram_block_table_ptr + dram_base + col.to(tl.int64),
            dram_values,
            mask=col < DRAM_PAGES,
        )

    if tile_id == 0:
        for route in range(Q):
            tl.store(topk_misses_ptr + row_id * Q + route, 0, mask=~active)
        tl.store(miss_counts_ptr + row_id, 0, mask=~active)
        cache_length = tl.where(active, C, 0).to(tl.int32)
        seq_len = tl.load(seq_lens_ptr + row_id)
        sparse_length = C + Q - 1
        logical_length = tl.where(is_sparse, sparse_length, seq_len)
        if PACKED_C8:
            prefix = tl.load(prefix_lens_ptr + row_id)
            cache_length = tl.where(active, C + seq_len - prefix, 0).to(tl.int32)
            logical_length = seq_len
        tl.store(
            actual_seq_lengths_kv_ptr + row_id,
            tl.where(is_valid, logical_length, 0).to(tl.int32),
        )
        tl.store(num_cache_tokens_ptr + row_id, cache_length)


def fused_lim_copy_sfa_metadata(
    *,
    source_ids: torch.Tensor,
    topk_misses: torch.Tensor,
    miss_counts: torch.Tensor,
    valid_rows: torch.Tensor,
    sparse_rows: torch.Tensor,
    request_slots: torch.Tensor,
    actual_seq_lengths_kv: torch.Tensor,
    num_cache_tokens: torch.Tensor,
    hbm_block_table: torch.Tensor,
    dram_block_table: torch.Tensor,
    capacity: int,
    host_context_len: int,
    request_rows: int,
    seq_lens: torch.Tensor,
    query_width: int = 1,
    host_block_table: torch.Tensor | None = None,
    packed_c8: bool = False,
    prefix_lens: torch.Tensor | None = None,
    device_slots: torch.Tensor | None = None,
) -> None:
    """Resolve C8 lengths, rotated HBM pages and invalid rows in one launch."""
    batch = source_ids.shape[0] // query_width
    hbm_pages = hbm_block_table.shape[1]
    dram_pages = host_context_len // 128
    tiles = triton.cdiv(max(2048, hbm_pages, dram_pages), 2048)
    _lim_copy_sfa_metadata_kernel[(tiles, batch)](
        source_ids,
        device_slots if device_slots is not None else source_ids,
        topk_misses,
        miss_counts,
        valid_rows,
        sparse_rows,
        request_slots,
        actual_seq_lengths_kv,
        num_cache_tokens,
        hbm_block_table,
        dram_block_table,
        host_block_table if host_block_table is not None else dram_block_table,
        seq_lens,
        prefix_lens if prefix_lens is not None else seq_lens,
        2048,
        query_width,
        capacity,
        hbm_pages,
        dram_pages,
        2048,
        host_block_table is not None,
        host_block_table.shape[1] if host_block_table is not None else 0,
        host_block_table.stride(0) if host_block_table is not None else 0,
        host_block_table.stride(1) if host_block_table is not None else 0,
        packed_c8,
        device_slots is not None,
    )


@triton.jit
def _tail_metadata_kernel(
    req_to_token,
    requests,
    positions,
    publication_valid,
    starts,
    ends,
    history_src,
    history_dst,
    history_valid,
    current_dst,
    current_valid,
    REQUEST_ROWS: tl.constexpr,
    CONTEXT: tl.constexpr,
    REQUEST_STRIDE: tl.constexpr,
    C: tl.constexpr,
    Q: tl.constexpr,
    HAS_VALID: tl.constexpr,
):
    batch = tl.program_id(0)
    request = tl.load(requests + batch).to(tl.int32)
    first = tl.load(positions + batch * Q).to(tl.int32)
    last = tl.load(positions + batch * Q + Q - 1).to(tl.int32) + 1
    active = (request > 0) & (request < REQUEST_ROWS) & (first >= 0) & (last <= CONTEXT)
    if HAS_VALID:
        for route in tl.static_range(Q):
            active = active & (tl.load(publication_valid + batch * Q + route) != 0)
    safe = tl.where(active, request, 0)
    prefix = first // 128 * 128
    old_start = tl.load(starts + safe)
    old_end = tl.load(ends + safe)
    offset = tl.arange(0, 128)
    pos = prefix + offset
    restore = active & (pos < first) & ((pos < old_start) | (pos >= old_end))
    physical = tl.load(
        req_to_token + safe * REQUEST_STRIDE + pos, mask=restore, other=0
    ).to(tl.int64)
    dst = safe.to(tl.int64) * (C + 256) + C + pos % 256
    tl.store(history_src + batch * 128 + offset, physical)
    tl.store(history_dst + batch * 128 + offset, tl.where(restore, dst, 0))
    tl.store(history_valid + batch * 128 + offset, restore)
    q = tl.arange(0, Q)
    qpos = first + q
    qdst = safe.to(tl.int64) * (C + 256) + C + qpos % 256
    tl.store(current_dst + batch * Q + q, tl.where(active, qdst, 0))
    tl.store(current_valid + batch * Q + q, active)
    # The following data copies and consumers all use this ordered stream.
    tl.store(starts + safe, prefix, mask=active)
    tl.store(ends + safe, last, mask=active)


def stage_mtp_tail(
    *,
    current,
    cache,
    host,
    host_address,
    req_to_token,
    requests,
    positions,
    starts,
    ends,
    workspace,
    capacity,
    context_len,
    query_width=4,
    current_valid=None,
):
    """Write current candidates locally; restore only missing stable tail rows."""
    from sgl_kernel_npu.sparsity_driven_kv_offload.ops import unidex_copy_inplace

    batch = requests.numel()
    if query_width not in (1, 4):
        raise ValueError("Tail query width must be 1 or 4")
    if (
        positions.numel() < batch * query_width
        or current.shape[0] < batch * query_width
    ):
        raise ValueError(
            "Separate tail requires query_width current KV rows per request"
        )
    hs, hd, hv, cs, cd, cv = workspace
    _tail_metadata_kernel[(batch,)](
        req_to_token,
        requests,
        positions,
        current_valid if current_valid is not None else requests,
        starts,
        ends,
        hs,
        hd,
        hv,
        cd,
        cv,
        starts.numel() - 1,
        context_len,
        req_to_token.stride(0),
        capacity,
        query_width,
        current_valid is not None,
    )
    unidex_copy_inplace(
        host,
        cache,
        hs[: batch * 128],
        hd[: batch * 128],
        hv[: batch * 128],
        2,
        2,
        block_dim=48,
        src_ptr=host_address,
    )
    unidex_copy_inplace(
        current.view(-1, 1, current.shape[-1]),
        cache,
        cs[: batch * query_width],
        cd[: batch * query_width],
        cv[: batch * query_width],
        1,
        2,
        block_dim=48,
    )
