"""PyTorch reference for the prefill score + index-head attention operator.

Independent of the Triton implementation: it never imports Triton and computes
both outputs with plain tensor operations, one request and one KV block at a
time. ``_build_qblock_mappings`` is copied from
``indexer/flash_block_score_prefill.py`` so both backends derive the page table
the same way without importing the legacy module.
"""

import torch


# Copied from indexer/flash_block_score_prefill.py; keep byte-identical.
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


def _block_score(qk, score_type):
    """Reduce one block's masked log2-domain logits to one score per query row."""
    best = qk.max(dim=-1).values
    if score_type == "max":
        return best
    # A fully masked block gives -inf - -inf = nan; the kernel rewrites that to -inf.
    values = best + torch.log2(torch.exp2(qk - best.unsqueeze(-1)).sum(dim=-1))
    return torch.where(torch.isnan(values), float("-inf"), values)


def flash_prefill_bnsd_score_attn(
    q,
    k_cache_bnsd,
    v_cache_bnsd,
    cu_seqlens,
    seq_lens,
    req_to_token,
    req_pool_indices,
    block_size_q,
    block_size_k,
    sm_scale,
    score_type="max",
    qblock_mappings=None,
):
    """Score every (query, KV block) pair and run causal index-head attention.

    Mirrors the kernel's mask rule: the causal comparison uses the packed global
    query-row index against the request-local KV position, so only a request
    whose ``cu_seqlens[r] == 0`` is truly causal. That is the legacy behavior and
    is intentionally reproduced.
    """
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

    score = torch.full(
        (num_idx_heads, total_q, max_seqblock_k),
        float("-inf"),
        device=device,
        dtype=torch.float32,
    )
    idx_o = torch.zeros(
        (total_q, num_idx_heads, head_dim), device=device, dtype=q.dtype
    )

    if all_seqblock_q == 0:
        return score, idx_o

    cu = cu_seqlens.to(device=device, dtype=torch.long).tolist()
    lens = seq_lens.to(device=device, dtype=torch.long).tolist()
    # Query-block rows of one active request are consecutive and start at
    # qb_to_qblock == 0.
    first_row = [i for i, local in enumerate(qb_to_qblock.tolist()) if local == 0]
    active = [r for r in range(len(lens)) if cu[r + 1] > cu[r]]
    scale = sm_scale * 1.4426950409

    for r, row in zip(active, first_row):
        q_start, q_end, seq_len = cu[r], cu[r + 1], lens[r]
        q_len = q_end - q_start
        if q_len == 0:
            continue
        num_blocks = (seq_len + page_size - 1) // page_size
        q_rows = q[q_start:q_end].to(torch.float32)
        q_rows = q_rows.reshape(q_len, num_idx_heads, head_dim)
        # Global packed row index: this reproduces the kernel's causal mask.
        q_pos = torch.arange(q_start, q_end, device=device, dtype=torch.long)

        for kv_head in range(num_kv_heads):
            heads = slice(kv_head * gqa_group_size, (kv_head + 1) * gqa_group_size)
            m_i = torch.full(
                (q_len, gqa_group_size),
                -1.0e30,
                device=device,
                dtype=torch.float32,
            )
            l_i = torch.zeros(
                (q_len, gqa_group_size), device=device, dtype=torch.float32
            )
            acc_o = torch.zeros(
                (q_len, gqa_group_size, head_dim),
                device=device,
                dtype=torch.float32,
            )

            for b in range(num_blocks):
                page_id = block_table[row, b].to(torch.long)
                keys = k_cache_bnsd[page_id].to(torch.float32)
                values = v_cache_bnsd[page_id]
                key_pos = b * page_size + torch.arange(
                    page_size, device=device, dtype=torch.long
                )
                col_mask = (q_pos[:, None, None] >= key_pos[None, None, :]) & (
                    key_pos[None, None, :] < seq_len
                )

                qk = torch.einsum(
                    "qgd,cd->qgc", q_rows[:, heads, :], keys[:, kv_head, :]
                )
                qk = (qk * scale).masked_fill(~col_mask, float("-inf"))

                val = _block_score(qk, score_type)
                score[heads, q_start:q_end, b] = val.permute(1, 0)

                # Online softmax in the same block order as the kernel. The p
                # cast mirrors ``p.to(v.dtype)`` before the V dot.
                m_new = torch.maximum(m_i, val)
                p = torch.exp2(qk - m_new.unsqueeze(-1))
                pv = torch.einsum(
                    "qgp,pd->qgd",
                    p.to(values.dtype).to(torch.float32),
                    values[:, kv_head, :].to(torch.float32),
                )
                acc_o = acc_o * torch.exp2(m_i - m_new).unsqueeze(-1) + pv
                l_i = l_i * torch.exp2(m_i - m_new) + p.sum(dim=-1)
                m_i = m_new

            idx_o[q_start:q_end, heads, :] = (acc_o / l_i.unsqueeze(-1)).to(q.dtype)

    return score, idx_o
