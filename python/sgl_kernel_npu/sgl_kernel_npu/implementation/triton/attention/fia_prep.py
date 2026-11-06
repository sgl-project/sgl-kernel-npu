"""Triton page-table preparation for FIA sparse prefill."""

import torch
import triton
import triton.language as tl


@triton.jit
def _fia_prep_kernel(
    topk_idx_ptr,  # [T, TOPK1] int32, -1-padded logical blocks (kvh==1)
    seq_lens_ptr,  # [T] int32 per-query CAUSAL KV len (abs_pos+1)
    req_pool_ptr,  # [T] int32 request index per query
    req_to_token_ptr,  # [num_req, max_ctx] int32 page slot table
    block_table_ptr,  # [T, TOPK1] int32 OUT (physical pages; own last, 0 pads)
    actual_kvlen_ptr,  # [T] int32 OUT
    max_req_to_token_cols,
    # strides
    stride_ti_t,
    stride_ti_k,
    stride_sl,
    stride_req,
    stride_rtt_r,
    stride_rtt_t,
    stride_bt_t,
    stride_bt_k,
    # meta
    TOPK1: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Per-query FIA prep: reorder (own last) + physical-page gather + actual_kvlen.
    Grid: (total_q,). One program per query; TOPK1 = topk+1 slots (small, ~18)."""
    pid = tl.program_id(0)
    seq_len = tl.load(seq_lens_ptr + pid * stride_sl).to(tl.int32)
    abs_pos = tl.maximum(seq_len - 1, 0)
    own = abs_pos // BLOCK_SIZE  # own logical block id (contains abs_pos)
    own_offset = abs_pos % BLOCK_SIZE  # causal offset within own block
    req = tl.load(req_pool_ptr + pid * stride_req).to(tl.int64)

    off = tl.arange(0, TOPK1)
    log_blk = tl.load(topk_idx_ptr + pid * stride_ti_t + off * stride_ti_k).to(tl.int32)
    is_own = (log_blk == own) & (log_blk >= 0)
    real_nonown = (log_blk >= 0) & (log_blk != own)
    rn = real_nonown.to(tl.int32)
    # exclusive scan -> output rank for each non-own real slot (0..count_real-1)
    rank = tl.cumsum(rn, axis=0) - rn  # [TOPK1]
    count_real = tl.sum(rn, axis=0)  # scalar
    own_present = tl.sum(is_own.to(tl.int32), axis=0) > 0

    # physical-page gather (in-kernel, vector-core): phys[i] = req_to_token[req,
    # max(log_blk[i],0)*bs] // bs; 0 for -1 pads (beyond actual_kvlen, unattended).
    safe_blk = tl.maximum(log_blk, 0)
    token_col = tl.minimum(safe_blk * BLOCK_SIZE, max_req_to_token_cols - 1)
    slots = tl.load(
        req_to_token_ptr + req * stride_rtt_r + token_col * stride_rtt_t,
        mask=log_blk >= 0,
        other=0,
    ).to(tl.int32)
    phys = slots // BLOCK_SIZE  # [TOPK1]

    # own block's physical page (scalar gather)
    own_col = tl.minimum(own * BLOCK_SIZE, max_req_to_token_cols - 1)
    own_slot = tl.load(
        req_to_token_ptr + req * stride_rtt_r + own_col * stride_rtt_t
    ).to(tl.int32)
    own_phys = own_slot // BLOCK_SIZE

    # zero-fill pads, then scatter non-own real -> [rank], own -> [count_real].
    # Own store is unconditional (empty query: own_phys at [0], kvlen==0 -> unattended).
    bt_base = block_table_ptr + pid * stride_bt_t
    tl.store(bt_base + off * stride_bt_k, tl.zeros((TOPK1,), tl.int32))
    tl.store(bt_base + rank * stride_bt_k, phys, mask=real_nonown)
    tl.store(bt_base + count_real * stride_bt_k, own_phys)

    # actual_kvlen: real blocks full + own causal (offset+1)
    kvl = tl.where(
        own_present, count_real * BLOCK_SIZE + own_offset + 1, count_real * BLOCK_SIZE
    )
    tl.store(actual_kvlen_ptr + pid, kvl.to(tl.int32))


def fia_prep(
    ti,
    seq_lens,
    per_query_req,
    req_to_token,
    block_size,
    block_table_out=None,
    actual_kvlen_out=None,
):
    total_q, topk1 = ti.shape
    block_table = (
        block_table_out
        if block_table_out is not None
        else torch.empty((total_q, topk1), dtype=torch.int32, device=ti.device)
    )
    actual_kvlen = (
        actual_kvlen_out
        if actual_kvlen_out is not None
        else torch.empty((total_q,), dtype=torch.int32, device=ti.device)
    )
    _fia_prep_kernel[(total_q,)](
        ti,
        seq_lens,
        per_query_req,
        req_to_token,
        block_table,
        actual_kvlen,
        req_to_token.shape[1],
        ti.stride(0),
        ti.stride(1),
        seq_lens.stride(0) if seq_lens.dim() else 0,
        per_query_req.stride(0) if per_query_req.dim() else 0,
        req_to_token.stride(0),
        req_to_token.stride(1),
        block_table.stride(0),
        block_table.stride(1),
        TOPK1=topk1,
        BLOCK_SIZE=block_size,
        num_warps=1,
        num_stages=1,
    )
    return block_table, actual_kvlen
