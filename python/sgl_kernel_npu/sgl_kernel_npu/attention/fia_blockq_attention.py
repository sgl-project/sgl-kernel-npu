"""NPU PREFILL sparse-attention: native FA (FIA) with a per-query custom block_table.
Per-query prep (own-block reorder + page gather + actual_kvlen) is fused into
one triton kernel ``_fia_prep_kernel``."""

from __future__ import annotations

from typing import Optional

import torch
import torch_npu  # noqa: F401  -- registers the npu backend
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


def flash_prefill_bnsd_blockq_sparse_fia(
    q: torch.Tensor,  # [total_q, num_q_heads, head_dim]
    k_cache_bnsd: torch.Tensor,  # [num_pages, block_size, num_kv_heads, head_dim]
    v_cache_bnsd: torch.Tensor,  # same shape
    topk_idx: torch.Tensor,  # [num_kv_heads, total_q, topk+1] int32, -1-padded
    seq_lens: torch.Tensor,  # [total_q] int32 -- per-query CAUSAL KV length (abs_pos+1)
    per_query_req: torch.Tensor,  # [total_q] int32/int64 -- request index per query
    req_to_token: torch.Tensor,  # [num_requests, max_ctx] int32 -- page slot table
    block_size: int,
    sm_scale: Optional[float],
    num_pages: int,
    max_num_blocks: int,
    block_table_out: Optional[
        torch.Tensor
    ] = None,  # reuse buffer (skip per-call empty)
    actual_kvlen_out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Single-pass FIA sparse+causal prefill attention. Returns [total_q, QH, D]."""
    assert q.dtype in (torch.float16, torch.bfloat16)
    total_q, num_q_heads, head_dim = q.shape
    num_pages_c, block_size_c, num_kv_heads, cache_head_dim = k_cache_bnsd.shape
    assert block_size_c == block_size
    assert cache_head_dim == head_dim
    assert k_cache_bnsd.shape == v_cache_bnsd.shape
    assert num_kv_heads == 1, (
        "FIA sparse path supports num_kv_heads==1 (MiniMax-M3 gqa=16); "
        f"got {num_kv_heads} -- use the triton blockq path instead."
    )
    assert topk_idx.shape[0] == num_kv_heads
    assert topk_idx.shape[1] == total_q
    assert topk_idx.dtype == torch.int32
    device = q.device
    if sm_scale is None:
        sm_scale = head_dim**-0.5
    topk1 = topk_idx.shape[2]  # topk + 1 (last slot = appended local/own block)
    max_req_to_token_cols = req_to_token.shape[1]

    # FUSED prep: one triton kernel (reorder + page gather + actual_kvlen).
    # Reuse caller-provided buffers (hoisted to _build_prefill_meta).
    block_table = (
        block_table_out
        if block_table_out is not None
        else torch.empty((total_q, topk1), dtype=torch.int32, device=device)
    )
    actual_kvlen = (
        actual_kvlen_out
        if actual_kvlen_out is not None
        else torch.empty((total_q,), dtype=torch.int32, device=device)
    )
    ti = topk_idx[0].contiguous()  # [T, topk1]
    # seq_lens / per_query_req arrive pre-cast to int32 (hoisted to
    # _build_prefill_meta, shared across sparse layers) -- no per-layer cast here.
    _fia_prep_kernel[(total_q,)](
        ti,
        seq_lens,
        per_query_req,
        req_to_token,
        block_table,
        actual_kvlen,
        max_req_to_token_cols,
        ti.stride(0),
        ti.stride(1),
        seq_lens.stride(0) if seq_lens.dim() > 0 else 0,
        per_query_req.stride(0) if per_query_req.dim() > 0 else 0,
        req_to_token.stride(0),
        req_to_token.stride(1),
        block_table.stride(0),
        block_table.stride(1),
        TOPK1=topk1,
        BLOCK_SIZE=block_size,
        num_warps=1,
        num_stages=1,
    )
    # KV 3D [num_pages, block_size, kvh*D]; layout TND.
    k_paged = k_cache_bnsd.view(num_pages, block_size, num_kv_heads * head_dim)
    v_paged = v_cache_bnsd.view(num_pages, block_size, num_kv_heads * head_dim)

    # FIA sparse_mode=0: own block length-limited via actual_kvlen (causal);
    # past score blocks are fully past, so no causal mask is needed.
    out, _ = torch.ops.npu.npu_fused_infer_attention_score(
        q,
        k_paged,
        v_paged,
        block_table=block_table,
        block_size=block_size,
        num_heads=num_q_heads,
        num_key_value_heads=num_kv_heads,
        input_layout="TND",
        sparse_mode=0,
        scale=sm_scale,
        actual_seq_lengths=list(range(1, total_q + 1)),
        actual_seq_lengths_kv=actual_kvlen.tolist(),
    )
    return out
