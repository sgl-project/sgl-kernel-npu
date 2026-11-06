"""Triton implementation of the prefill score + index-head attention operator.

Standalone copy of ``flash_prefill_bnsd_score_attn`` from
``indexer/flash_block_score_prefill.py``: the kernel is byte-identical, the
wrapper only gains a ``score_type`` assertion, and the two helpers it needs are
copied here instead of importing the legacy module.
"""

from typing import Optional

import torch
import triton
import triton.language as tl

# Tunable launch configs for the score kernels (deterministic, cuda-graph safe).
_SCORE_ATTN_NW = 8
_SCORE_ATTN_NS = 2


# Copied from indexer/flash_block_score_decode.py; keep byte-identical.
def _next_power_of_2(x: int) -> int:
    return 1 << (int(x) - 1).bit_length()


@triton.jit
def _prefill_bnsd_score_attn_kernel(
    q_ptr,  # [total_q, num_idx_heads, head_dim]  (index Q)
    k_cache_ptr,  # [num_pages, page_size, num_kv_heads, head_dim] (index K)
    v_cache_ptr,  # [num_pages, page_size, num_kv_heads, head_dim] (index V)
    block_table_ptr,  # [all_seqblock_q, max_num_blocks]
    qb_to_qstart_ptr,  # [all_seqblock_q]
    qb_to_qblock_ptr,  # [all_seqblock_q]
    qb_seq_lens_ptr,  # [all_seqblock_q]
    qb_qend_ptr,  # [all_seqblock_q] exclusive q upper bound (cu_seqlens[r+1])
    score_ptr,  # [num_idx_heads, total_q, max_seqblock_k]
    idx_o_ptr,  # [total_q, num_idx_heads, head_dim]
    # scalars
    total_q,
    num_kv_heads,
    gqa_group_size,
    head_dim,
    all_seqblock_q,
    sm_scale,
    # strides
    stride_q_n,
    stride_q_h,
    stride_q_d,
    stride_k_block,
    stride_k_offset,
    stride_k_h,
    stride_k_d,
    stride_v_block,
    stride_v_offset,
    stride_v_h,
    stride_v_d,
    stride_bt_q,
    stride_bt_n,
    stride_s_h,
    stride_s_q,
    stride_s_n,
    stride_o_n,
    stride_o_h,
    stride_o_d,
    # constexpr meta
    block_size: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    SCORE_TYPE: tl.constexpr,
):
    """Fused block-score + index-head dense attention (query-block tiled)."""
    tl.static_assert(SCORE_TYPE == "max" or SCORE_TYPE == "lse")
    tl.static_assert(BLOCK_SIZE_N >= block_size)

    pid_qb = tl.program_id(0)
    pid_kh = tl.program_id(1)

    seq_len = tl.load(qb_seq_lens_ptr + pid_qb).to(tl.int32)
    num_blocks = tl.cdiv(seq_len, block_size)

    q_start = tl.load(qb_to_qstart_ptr + pid_qb).to(tl.int32)
    q_block_local = tl.load(qb_to_qblock_ptr + pid_qb).to(tl.int32)
    q_end = tl.load(qb_qend_ptr + pid_qb).to(tl.int32)

    off_d = tl.arange(0, BLOCK_SIZE_D)
    off_n = tl.arange(0, BLOCK_SIZE_N)
    off_qh = tl.arange(0, BLOCK_SIZE_Q * BLOCK_SIZE_H)
    qi = off_qh // BLOCK_SIZE_H
    hh = off_qh % BLOCK_SIZE_H
    pid_h_base = pid_kh * gqa_group_size
    # Clamp q_token to [q_start, q_end-1] so phantom tail rows stay in-request.
    q_token_raw = q_start + q_block_local * BLOCK_SIZE_Q + qi
    row_valid = (q_token_raw < q_end) & (hh < gqa_group_size)
    q_token_flat = tl.maximum(q_start, tl.minimum(q_token_raw, q_end - 1))
    head_flat = pid_h_base + hh

    q_offsets = (
        q_token_flat[:, None] * stride_q_n
        + head_flat[:, None] * stride_q_h
        + off_d[None, :] * stride_q_d
    )
    q = tl.load(
        q_ptr + q_offsets,
        mask=row_valid[:, None] & (off_d[None, :] < head_dim),
        other=0.0,
    )

    sm_scale_log2e = sm_scale * 1.4426950409
    # Finite floor (not -inf): all-masked blocks are a no-op, no NaN/guard needed.
    m_i = tl.full((BLOCK_SIZE_Q * BLOCK_SIZE_H,), -1.0e30, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_SIZE_Q * BLOCK_SIZE_H,), dtype=tl.float32)
    acc_o = tl.zeros((BLOCK_SIZE_Q * BLOCK_SIZE_H, BLOCK_SIZE_D), dtype=tl.float32)

    for logical_block in tl.range(num_blocks):
        physical_block = tl.load(
            block_table_ptr + pid_qb * stride_bt_q + logical_block * stride_bt_n
        ).to(tl.int64)
        key_pos = logical_block * block_size + off_n
        pos_mask = key_pos < seq_len

        k_offsets = (
            physical_block * stride_k_block
            + off_n[None, :] * stride_k_offset
            + pid_kh * stride_k_h
            + off_d[:, None] * stride_k_d
        )
        k = tl.load(
            k_cache_ptr + k_offsets,
            mask=(off_d[:, None] < head_dim) & pos_mask[None, :],
            other=0.0,
        )
        qk = tl.dot(q, k) * sm_scale_log2e  # [M, N]
        causal = q_token_raw[:, None] >= key_pos[None, :]
        qk = tl.where(causal & pos_mask[None, :], qk, float("-inf"))

        # per-block score
        sub_max = tl.max(qk, axis=1)
        if SCORE_TYPE == "max":
            score = sub_max
        else:
            score = sub_max + tl.log2(tl.sum(tl.exp2(qk - sub_max[:, None]), axis=1))
            score = tl.where(score != score, float("-inf"), score)
        s_offsets = (
            head_flat * stride_s_h
            + q_token_raw * stride_s_q
            + logical_block * stride_s_n
        )
        tl.store(
            score_ptr + s_offsets, score.to(score_ptr.dtype.element_ty), mask=row_valid
        )

        # online softmax -> idx_o accumulation
        m_new = tl.maximum(m_i, sub_max)
        p = tl.exp2(qk - m_new[:, None])
        l_new = tl.sum(p, axis=1)
        acc_o = acc_o * tl.exp2(m_i - m_new)[:, None]
        v_offsets = (
            physical_block * stride_v_block
            + off_n[:, None] * stride_v_offset
            + pid_kh * stride_v_h
            + off_d[None, :] * stride_v_d
        )
        v = tl.load(
            v_cache_ptr + v_offsets,
            mask=pos_mask[:, None] & (off_d[None, :] < head_dim),
            other=0.0,
        )
        acc_o = acc_o + tl.dot(p.to(v.dtype), v)
        l_i = l_i * tl.exp2(m_i - m_new) + l_new
        m_i = m_new

    idx_o = acc_o / l_i[:, None]
    o_offsets = (
        q_token_raw[:, None] * stride_o_n
        + head_flat[:, None] * stride_o_h
        + off_d[None, :] * stride_o_d
    )
    tl.store(
        idx_o_ptr + o_offsets,
        idx_o.to(idx_o_ptr.dtype.element_ty),
        mask=row_valid[:, None] & (off_d[None, :] < head_dim),
    )


def _build_qblock_mappings(
    cu_seqlens: torch.Tensor,
    seq_lens: torch.Tensor,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    block_size_q: int,
    page_size: int,
    max_blocks: int,
    device,
):
    """Precompute per-query-block varlen mappings (cheap PyTorch)."""
    seq_lens_l = seq_lens.to(device=device, dtype=torch.long)
    cu_q = cu_seqlens.to(device=device, dtype=torch.long)
    reqs = req_pool_indices.to(device=device, dtype=torch.long)

    q_lens = cu_q[1:] - cu_q[:-1]  # [bs]
    qb_per_req = (q_lens + block_size_q - 1) // block_size_q  # [bs]
    all_seqblock_q = int(qb_per_req.sum().item())

    # Owning request + q-start (token) per query-block.
    qb_to_req = reqs.repeat_interleave(qb_per_req)  # [all_seqblock_q]
    qb_to_qstart = cu_q[:-1].repeat_interleave(qb_per_req)
    # Per-q-block exclusive upper bound (cu_seqlens[r+1]); masks partial tail rows.
    qb_qend = cu_q[1:].repeat_interleave(qb_per_req)
    qb_seq_lens = seq_lens_l.repeat_interleave(qb_per_req)
    # Local q-block index within its request.
    cu_blocks = torch.zeros_like(qb_per_req)
    cu_blocks[1:] = qb_per_req[:-1].cumsum(0)
    arange_all = torch.arange(all_seqblock_q, device=device, dtype=torch.long)
    qb_to_qblock = arange_all - cu_blocks.repeat_interleave(qb_per_req)

    # block_table[qb, blk] = physical page of logical block blk of qb's request.
    blk_cols = torch.arange(max_blocks, device=device, dtype=torch.long) * page_size
    max_cols = req_to_token.shape[1]
    blk_cols = blk_cols.clamp(max=max_cols - 1)
    # Broadcast row/column indices to avoid materializing the full [Q, ctx] slab.
    token_slots = req_to_token[
        qb_to_req[:, None], blk_cols
    ]  # [all_seqblock_q, max_blocks]
    block_table = (token_slots // page_size).to(torch.int32)

    return (
        qb_to_qstart.to(torch.int32),
        qb_to_qblock.to(torch.int32),
        qb_seq_lens.to(torch.int32),
        qb_qend.to(torch.int32),
        block_table,
        all_seqblock_q,
    )


def flash_prefill_bnsd_score_attn(
    q: torch.Tensor,  # [total_q, num_idx_heads, head_dim]  (index Q)
    k_cache_bnsd: torch.Tensor,  # [num_pages, page_size, num_kv_heads, head_dim]
    v_cache_bnsd: torch.Tensor,  # [num_pages, page_size, num_kv_heads, head_dim]
    cu_seqlens: torch.Tensor,
    seq_lens: torch.Tensor,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    block_size_q: int,
    block_size_k: int,
    sm_scale: float,
    score_type: str = "max",
    qblock_mappings: Optional[tuple] = None,
):
    """Fused block-score + index-head dense attention (query-block tiled)."""
    total_q, num_idx_heads, head_dim = q.shape
    num_kv_heads = k_cache_bnsd.shape[2]
    assert num_idx_heads % num_kv_heads == 0
    assert score_type in ("max", "lse"), f"unsupported score_type: {score_type!r}"
    gqa_group_size = num_idx_heads // num_kv_heads
    page_size = block_size_k
    max_seqblock_k = (int(seq_lens.max().item()) + page_size - 1) // page_size
    max_blocks = max_seqblock_k
    device = q.device

    if qblock_mappings is None:
        (
            qb_to_qstart,
            qb_to_qblock,
            qb_seq_lens,
            qb_qend,
            block_table,
            all_seqblock_q,
        ) = _build_qblock_mappings(
            cu_seqlens,
            seq_lens,
            req_to_token,
            req_pool_indices,
            block_size_q,
            page_size,
            max_blocks,
            device,
        )
    else:
        (
            qb_to_qstart,
            qb_to_qblock,
            qb_seq_lens,
            qb_qend,
            block_table,
            all_seqblock_q,
        ) = qblock_mappings

    BLOCK_SIZE_Q = _next_power_of_2(block_size_q)
    score = torch.full(
        (num_idx_heads, total_q, max_seqblock_k),
        float("-inf"),
        device=device,
        dtype=torch.float32,
    )
    idx_o = torch.zeros(
        (total_q, num_idx_heads, head_dim), device=device, dtype=q.dtype
    )

    if all_seqblock_q > 0:
        grid = (all_seqblock_q, num_kv_heads)
        _prefill_bnsd_score_attn_kernel[grid](
            q,
            k_cache_bnsd,
            v_cache_bnsd,
            block_table,
            qb_to_qstart,
            qb_to_qblock,
            qb_seq_lens,
            qb_qend,
            score,
            idx_o,
            total_q,
            num_kv_heads,
            gqa_group_size,
            head_dim,
            all_seqblock_q,
            sm_scale,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k_cache_bnsd.stride(0),
            k_cache_bnsd.stride(1),
            k_cache_bnsd.stride(2),
            k_cache_bnsd.stride(3),
            v_cache_bnsd.stride(0),
            v_cache_bnsd.stride(1),
            v_cache_bnsd.stride(2),
            v_cache_bnsd.stride(3),
            block_table.stride(0),
            block_table.stride(1),
            score.stride(0),
            score.stride(1),
            score.stride(2),
            idx_o.stride(0),
            idx_o.stride(1),
            idx_o.stride(2),
            block_size_k,
            BLOCK_SIZE_Q,
            triton.next_power_of_2(gqa_group_size),
            triton.next_power_of_2(head_dim),
            triton.next_power_of_2(page_size),
            score_type,
            num_warps=_SCORE_ATTN_NW,
            num_stages=_SCORE_ATTN_NS,
        )
    return score, idx_o
