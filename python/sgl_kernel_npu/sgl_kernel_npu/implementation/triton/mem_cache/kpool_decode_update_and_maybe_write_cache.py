"""Triton implementation of kpool_decode_update_and_maybe_write_cache."""

import torch
import triton
import triton.language as tl


@triton.jit
def _kpool_decode_update_and_maybe_write_cache_kernel(
    buf_bf16_ptr,  # [num_pages, SLOTS_PER_PAGE * HEAD_DIM] bfloat16
    tail_k_ptr,  # [REQ_POOL_SIZE, TAIL_SIZE, HEAD_DIM] bfloat16
    tail_score_ptr,  # [REQ_POOL_SIZE, TAIL_SIZE, HEAD_DIM] same as score dtype
    key_ptr,  # [batch, HEAD_DIM] bfloat16
    slot_score_ptr,  # [batch, HEAD_DIM] same as score dtype
    ape_ptr,  # [POOL_SIZE, HEAD_DIM] float32
    block_tables_ptr,  # [batch, BLOCK_TABLE_COLS] int32
    req_pool_indices_ptr,  # [>=batch] int
    positions_ptr,  # [>=batch] int
    seq_lens_ptr,  # [>=batch] int
    out_cache_loc_ptr,  # [>=batch] int
    tail_k_stride_0,
    tail_k_stride_1,
    tail_score_stride_0,
    tail_score_stride_1,
    key_stride_0,
    slot_score_stride_0,
    ape_stride_0,
    block_tables_stride_0,
    block_tables_stride_1,
    REQ_POOL_SIZE: tl.constexpr,
    BUF_NUMEL_PER_PAGE: tl.constexpr,
    POOL_SIZE: tl.constexpr,
    TAIL_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_TABLE_COLS: tl.constexpr,
    SLOTS_PER_PAGE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)
    dim_mask = offs < HEAD_DIM

    # Load metadata
    req_raw = tl.load(req_pool_indices_ptr + row)
    req_valid = (req_raw >= 0) & (req_raw < REQ_POOL_SIZE)
    req = tl.minimum(tl.maximum(req_raw, 0), REQ_POOL_SIZE - 1)

    pos = tl.load(positions_ptr + row)
    safe_pos = tl.maximum(pos, 0)
    seq_len = tl.load(seq_lens_ptr + row)
    cache_loc = tl.load(out_cache_loc_ptr + row)
    pos_valid = req_valid & (cache_loc != 0) & (pos >= 0) & (pos < seq_len)

    slot = safe_pos % POOL_SIZE
    phys_slot = safe_pos % TAIL_SIZE

    # Load current key & score
    key = tl.load(
        key_ptr + row * key_stride_0 + offs,
        mask=dim_mask,
        other=0.0,
    ).to(tl.float32)
    score_current = tl.load(
        slot_score_ptr + row * slot_score_stride_0 + offs,
        mask=dim_mask,
        other=0.0,
    ).to(tl.float32)

    # (B) Pool compression (only when pool is full)
    if pos_valid & (slot == POOL_SIZE - 1):
        pool_logical_start = safe_pos - slot

        # Pass 1: find max score
        max_score = tl.full((BLOCK_D,), -float("inf"), tl.float32)
        for pool_slot in tl.static_range(0, POOL_SIZE):
            is_current = pool_slot == slot
            phys = (pool_logical_start + pool_slot) % TAIL_SIZE
            score_buf = tl.load(
                tail_score_ptr
                + req * tail_score_stride_0
                + phys * tail_score_stride_1
                + offs,
                mask=dim_mask,
                other=0.0,
            ).to(tl.float32)
            score = tl.where(is_current, score_current, score_buf)
            score += tl.load(
                ape_ptr + pool_slot * ape_stride_0 + offs,
                mask=dim_mask,
                other=0.0,
            ).to(tl.float32)
            max_score = tl.maximum(max_score, score)

        # Pass 2: softmax weighted average
        acc = tl.full((BLOCK_D,), 0.0, tl.float32)
        denom = tl.full((BLOCK_D,), 0.0, tl.float32)
        for pool_slot in tl.static_range(0, POOL_SIZE):
            is_current = pool_slot == slot
            phys = (pool_logical_start + pool_slot) % TAIL_SIZE
            score_buf = tl.load(
                tail_score_ptr
                + req * tail_score_stride_0
                + phys * tail_score_stride_1
                + offs,
                mask=dim_mask,
                other=0.0,
            ).to(tl.float32)
            score = tl.where(is_current, score_current, score_buf)
            score += tl.load(
                ape_ptr + pool_slot * ape_stride_0 + offs,
                mask=dim_mask,
                other=0.0,
            ).to(tl.float32)
            prob = tl.exp(score - max_score)
            denom += prob
            k_buf = tl.load(
                tail_k_ptr + req * tail_k_stride_0 + phys * tail_k_stride_1 + offs,
                mask=dim_mask,
                other=0.0,
            ).to(tl.float32)
            k = tl.where(is_current, key, k_buf)
            acc += k * prob

        # BF16: no Hadamard, no FP8 quantize — just write directly
        compressed = (acc / denom).to(tl.bfloat16)

        # Compute write location in buf
        pool_id = safe_pos // POOL_SIZE
        pool_page_group = pool_id // SLOTS_PER_PAGE
        token_page_row = pool_page_group * POOL_SIZE
        token_page_row = tl.minimum(tl.maximum(token_page_row, 0), BLOCK_TABLE_COLS - 1)
        packed_page = tl.load(
            block_tables_ptr
            + row * block_tables_stride_0
            + token_page_row * block_tables_stride_1,
        )
        loc_page_index = packed_page.to(tl.int64)
        loc_token_offset_in_page = pool_id % SLOTS_PER_PAGE
        out_k_offsets = (
            loc_page_index * BUF_NUMEL_PER_PAGE
            + loc_token_offset_in_page * HEAD_DIM
            + offs
        )

        tl.store(buf_bf16_ptr + out_k_offsets, compressed, mask=dim_mask)

    # (A) Tail buffer update (ALWAYS)
    tail_k_offset = req * tail_k_stride_0 + phys_slot * tail_k_stride_1 + offs
    tail_score_offset = (
        req * tail_score_stride_0 + phys_slot * tail_score_stride_1 + offs
    )
    update_mask = dim_mask & pos_valid
    tl.store(tail_k_ptr + tail_k_offset, key, mask=update_mask)
    tl.store(tail_score_ptr + tail_score_offset, score_current, mask=update_mask)


def kpool_decode_update_and_maybe_write_cache(
    buf: torch.Tensor,  # contiguous [num_pages, ..., HEAD_DIM] bfloat16 cache
    tail_k: torch.Tensor,  # [REQ_POOL_SIZE, TAIL_SIZE, HEAD_DIM] bfloat16
    tail_score: torch.Tensor,  # [REQ_POOL_SIZE, TAIL_SIZE, HEAD_DIM] float32/bfloat16
    key: torch.Tensor,  # [batch, HEAD_DIM] bfloat16
    slot_score: torch.Tensor,  # [batch, HEAD_DIM] float32/bfloat16
    ape: torch.Tensor,  # [POOL_SIZE, HEAD_DIM] float32
    block_tables: torch.Tensor,  # [batch, BLOCK_TABLE_COLS] int32
    req_pool_indices: torch.Tensor,
    positions: torch.Tensor,
    seq_lens: torch.Tensor,
    out_cache_loc: torch.Tensor,
    *,
    pool_size: int,
    slots_per_page: int,
    head_dim: int = 128,
) -> None:
    """BF16 variant of kpool_decode_update_and_maybe_write_cache.

    Differences from FP8 original:
    - buf is bfloat16 (no FP8 quantization, no per-slot scale)
    - No Hadamard128 rotation
    - Softmax-weighted average written directly as BF16
    """
    batch = key.shape[0]
    if batch == 0:
        return

    assert buf.dtype == torch.bfloat16
    assert tail_k.dtype == torch.bfloat16
    assert key.dtype == torch.bfloat16
    assert tail_k.ndim == 3
    assert tail_score.shape == tail_k.shape
    assert tail_k.shape[2] == head_dim
    assert ape.shape == (pool_size, head_dim)

    tail_size = tail_k.shape[1]
    req_pool_size = tail_k.shape[0]
    block_table_cols = block_tables.shape[1]

    # In-place outputs must not be silently replaced by contiguous copies.
    assert buf.is_contiguous(), "buf must be contiguous"
    assert tail_k.is_contiguous(), "tail_k must be contiguous"
    assert tail_score.is_contiguous(), "tail_score must be contiguous"
    buf_numel_per_page = buf.stride(0)
    assert buf_numel_per_page == slots_per_page * head_dim
    key = key.contiguous()
    slot_score = slot_score.contiguous()
    ape = ape.contiguous()
    block_tables = block_tables.contiguous()
    req_pool_indices = req_pool_indices.contiguous()
    positions = positions.contiguous()
    seq_lens = seq_lens.contiguous()
    out_cache_loc = out_cache_loc.contiguous()

    _kpool_decode_update_and_maybe_write_cache_kernel[(batch,)](
        buf,
        tail_k,
        tail_score,
        key,
        slot_score,
        ape,
        block_tables,
        req_pool_indices,
        positions,
        seq_lens,
        out_cache_loc,
        tail_k.stride(0),
        tail_k.stride(1),
        tail_score.stride(0),
        tail_score.stride(1),
        key.stride(0),
        slot_score.stride(0),
        ape.stride(0),
        block_tables.stride(0),
        block_tables.stride(1),
        REQ_POOL_SIZE=req_pool_size,
        BUF_NUMEL_PER_PAGE=buf_numel_per_page,
        POOL_SIZE=pool_size,
        TAIL_SIZE=tail_size,
        HEAD_DIM=head_dim,
        BLOCK_TABLE_COLS=block_table_cols,
        SLOTS_PER_PAGE=slots_per_page,
        BLOCK_D=triton.next_power_of_2(head_dim),
    )
