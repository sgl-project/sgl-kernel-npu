"""Triton implementation of the prefill block-topk operator."""

import torch
import triton
import triton.language as tl


@triton.jit
def _prefill_topk_from_score_kernel(
    score_ptr,  # [num_kv_heads, total_q, max_seqblock_k] fp32 (-inf for invalid)
    seq_lens_ptr,  # [total_q] int32 -- per-query CAUSAL KV length (prefix+within+1)
    out_ptr,  # [num_kv_heads, total_q, topk + 1] int32 -- topk + appended local block
    total_q,
    max_seqblock_k,
    block_size,
    topk,
    stride_s_h,
    stride_s_q,
    stride_s_k,
    stride_o_h,
    stride_o_q,
    stride_o_t,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    TOPK: tl.constexpr,
):
    """Fused block-TopK + causal-local-block append from the score tensor."""
    pid_qb = tl.program_id(0)
    pid_h = tl.program_id(1)

    off_q = tl.arange(0, BLOCK_SIZE_Q)  # [BSQ]
    q_tok = pid_qb * BLOCK_SIZE_Q + off_q  # [BSQ]
    q_valid = q_tok < total_q

    # Per-query causal KV length -> query position + local block.
    seq_len_q = tl.load(seq_lens_ptr + q_tok, mask=q_valid, other=1).to(tl.int32)
    query_pos = tl.maximum(seq_len_q - 1, 0)  # [BSQ]
    local_blk = tl.minimum(query_pos // block_size, max_seqblock_k - 1)  # [BSQ]

    off_k = tl.arange(0, BLOCK_SIZE_K)  # [K]
    k_valid = off_k < max_seqblock_k  # [K]

    # Score slab [BSQ, K]; out-of-range/future blocks are -inf.
    s_offsets = (
        pid_h * stride_s_h + q_tok[:, None] * stride_s_q + off_k[None, :] * stride_s_k
    )
    scores = tl.load(
        score_ptr + s_offsets,
        mask=q_valid[:, None] & k_valid[None, :],
        other=float("-inf"),
    )  # [BSQ, K] fp32

    # TopK via repeated max+argmax (lowest-index tie-break); selected block removed each iter.
    for rank in tl.range(0, TOPK):
        best = tl.max(scores, axis=1)  # [BSQ]
        is_best = scores == best[:, None]  # [BSQ, K]
        pos = tl.min(
            tl.where(is_best, off_k[None, :], BLOCK_SIZE_K), axis=1
        )  # [BSQ] argmax (min index on ties)
        # Store selected block only if finite + causally valid; else -1.
        valid_blk = (
            (best > float("-inf"))
            & (pos < max_seqblock_k)
            & (pos * block_size <= query_pos)
        )
        out_blk = tl.where(valid_blk, pos, -1)
        tl.store(
            out_ptr + pid_h * stride_o_h + q_tok * stride_o_q + rank * stride_o_t,
            tl.where(q_valid, out_blk, -1),
            mask=q_valid,
        )
        # Blank the selected position so the next iter picks the next best.
        scores = tl.where(off_k[None, :] == pos[:, None], float("-inf"), scores)

    # Append the causal local block at slot TOPK (-1 if already selected: dedup).
    rank_off = tl.arange(0, TOPK)  # [TOPK]
    sel_offsets = (
        pid_h * stride_o_h
        + q_tok[:, None] * stride_o_q
        + rank_off[None, :] * stride_o_t
    )
    selected = tl.load(
        out_ptr + sel_offsets, mask=q_valid[:, None], other=-1
    )  # [BSQ, TOPK]
    local_present = (
        tl.sum((selected == local_blk[:, None]).to(tl.int32), axis=1) > 0
    )  # [BSQ]
    out_local = tl.where(local_present, -1, local_blk)
    tl.store(
        out_ptr + pid_h * stride_o_h + q_tok * stride_o_q + TOPK * stride_o_t,
        tl.where(q_valid, out_local, -1),
        mask=q_valid,
    )


def flash_prefill_bnsd_topk_from_score(
    score: torch.Tensor,  # [num_kv_heads, total_q, max_seqblock_k] fp32
    per_query_seq_lens: torch.Tensor,  # [total_q] int32 -- per-query causal KV len
    topk: int,
    block_size_k: int,
) -> torch.Tensor:
    """Fused TopK + causal-local-block append -> topk_idx [num_kv_heads, total_q, topk+1]."""
    num_kv_heads, total_q, max_seqblock_k = score.shape
    assert score.dtype == torch.float32, "score must be fp32"
    assert per_query_seq_lens.dtype == torch.int32
    assert per_query_seq_lens.shape[0] == total_q
    # The kernel builds arange(0, TOPK); reject an empty selection early.
    assert topk > 0, f"topk must be positive, got {topk!r}"
    device = score.device

    if total_q == 0:
        return torch.empty(
            (num_kv_heads, 0, topk + 1), dtype=torch.int32, device=device
        )

    out = torch.empty(
        (num_kv_heads, total_q, topk + 1), dtype=torch.int32, device=device
    )

    BLOCK_SIZE_K = triton.next_power_of_2(max_seqblock_k)
    # BSQ capped at 16 (UB-safe guard: bsq*2*K*4 <= 32KB).
    bsq = 1
    while bsq < 16 and bsq * 2 * BLOCK_SIZE_K * 4 <= 32768:
        bsq *= 2
    BLOCK_SIZE_Q = max(1, bsq)

    grid = (triton.cdiv(total_q, BLOCK_SIZE_Q), num_kv_heads)
    _prefill_topk_from_score_kernel[grid](
        score,
        per_query_seq_lens,
        out,
        total_q,
        max_seqblock_k,
        block_size_k,
        topk,
        score.stride(0),
        score.stride(1),
        score.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        BLOCK_SIZE_Q=BLOCK_SIZE_Q,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        TOPK=topk,
        num_warps=4,
        num_stages=1,
    )
    return out
