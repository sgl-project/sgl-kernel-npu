"""Triton implementation of the query-block append."""

import torch
import triton
import triton.language as tl


@triton.heuristics(
    {
        "BLOCK_SIZE_T": lambda args: triton.next_power_of_2(args["topk"]),
    }
)
@triton.jit
def _append_local_block_to_topk_idx_kernel(
    topk_idx_ptr,  # [num_kv_heads, batch_size, topk]
    seq_lens_ptr,  # [batch_size]
    out_ptr,  # [num_kv_heads, batch_size, topk + 1]
    batch_size,
    topk,
    num_blocks,
    block_size: tl.constexpr,
    stride_topk_h,
    stride_topk_b,
    stride_topk_t,
    stride_out_h,
    stride_out_b,
    stride_out_t,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
):
    """Append the causal local block (query-block-tiled, BSQ queries/program).
    Validates candidates and appends the local block (deduped to -1)."""
    pid_qb = tl.program_id(0)
    pid_h = tl.program_id(1)
    off_q = tl.arange(0, BLOCK_SIZE_Q)  # [BSQ]
    off_t = tl.arange(0, BLOCK_SIZE_T)  # [topk_pow2]
    q_tok = pid_qb * BLOCK_SIZE_Q + off_q  # [BSQ]
    q_valid = q_tok < batch_size

    # Per-query causal position + local block.
    seq_len = tl.load(seq_lens_ptr + q_tok, mask=q_valid, other=1).to(tl.int32)
    query_pos = tl.maximum(seq_len - 1, 0)
    local_blk = tl.minimum(query_pos // block_size, num_blocks - 1)

    # Load candidates [BSQ, topk] and validate.
    in_off = (
        pid_h * stride_topk_h
        + q_tok[:, None] * stride_topk_b
        + off_t[None, :] * stride_topk_t
    )
    cand = tl.load(
        topk_idx_ptr + in_off,
        mask=q_valid[:, None] & (off_t[None, :] < topk),
        other=-1,
    ).to(tl.int32)
    valid = (
        (cand >= 0) & (cand < num_blocks) & (cand * block_size <= query_pos[:, None])
    )
    cand_out = tl.where(valid, cand, -1)

    # Store validated candidates [BSQ, topk].
    out_off = (
        pid_h * stride_out_h
        + q_tok[:, None] * stride_out_b
        + off_t[None, :] * stride_out_t
    )
    tl.store(
        out_ptr + out_off, cand_out, mask=q_valid[:, None] & (off_t[None, :] < topk)
    )

    # Append local block at slot topk: -1 if already present (dedup).
    local_present = tl.sum((cand_out == local_blk[:, None]).to(tl.int32), axis=1) > 0
    out_local = tl.where(local_present, -1, local_blk)
    tl.store(
        out_ptr + pid_h * stride_out_h + q_tok * stride_out_b + topk * stride_out_t,
        tl.where(q_valid, out_local, -1),
        mask=q_valid,
    )


@torch.no_grad()
def append_local_block_to_topk_idx(topk_idx, seq_lens, block_size, num_blocks):
    """Append the query's own block after validating the topk candidates."""
    assert topk_idx.ndim == 3
    assert topk_idx.dtype == torch.int32
    assert topk_idx.is_contiguous()
    assert topk_idx.shape[2] > 0
    assert seq_lens.ndim == 1
    assert seq_lens.shape[0] == topk_idx.shape[1]
    assert seq_lens.is_contiguous()
    assert block_size > 0
    assert num_blocks > 0

    num_kv_heads, batch_size, topk = topk_idx.shape
    out = torch.empty(
        (num_kv_heads, batch_size, topk + 1),
        dtype=topk_idx.dtype,
        device=topk_idx.device,
    )
    BSQ = 16
    grid = (triton.cdiv(batch_size, BSQ), num_kv_heads)
    _append_local_block_to_topk_idx_kernel[grid](
        topk_idx,
        seq_lens,
        out,
        batch_size,
        topk,
        num_blocks,
        block_size=block_size,
        stride_topk_h=topk_idx.stride(0),
        stride_topk_b=topk_idx.stride(1),
        stride_topk_t=topk_idx.stride(2),
        stride_out_h=out.stride(0),
        stride_out_b=out.stride(1),
        stride_out_t=out.stride(2),
        BLOCK_SIZE_Q=BSQ,
        num_warps=4,
        num_stages=1,
    )
    return out
