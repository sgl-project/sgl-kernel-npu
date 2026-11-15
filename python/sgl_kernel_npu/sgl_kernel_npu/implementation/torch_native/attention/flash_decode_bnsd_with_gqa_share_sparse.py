"""Independent PyTorch reference for GQA-shared sparse decode attention.

Only the selected blocks are gathered, so the cost is ``B * topk * block_size``
regardless of the context length. The reference keeps the accumulators in
fp32 and casts to ``q.dtype`` at the end; it does not share code with the
Triton implementation and never calls the native operator.
"""

import torch


@torch.no_grad()
def flash_decode_bnsd_with_gqa_share_sparse(
    q,
    sink,
    k_cache_bnsd,
    v_cache_bnsd,
    block_table,
    seq_lens,
    block_size,
    topk_idx,
    sm_scale=None,
    req_to_token=None,
    req_pool_indices=None,
    max_num_blocks=None,
    num_pages=None,
    sanitize_page_ids=False,
):
    """Return ``[B, QH, D]`` sparse attention over the selected blocks."""
    batch_size, num_q_heads, head_dim = q.shape
    num_kv_heads = k_cache_bnsd.shape[2]
    gqa_group_size = num_q_heads // num_kv_heads

    if sm_scale is None:
        sm_scale = head_dim**-0.5

    if topk_idx.shape[0] == num_kv_heads:
        kv_topk = topk_idx.to(torch.int64)
    else:
        # Per-query-head input keeps the first head of each GQA group.
        kv_topk = topk_idx.reshape(
            num_kv_heads, gqa_group_size, batch_size, topk_idx.shape[2]
        )[:, 0].to(torch.int64)

    if req_to_token is not None:
        max_kv_len = int(max_num_blocks) * block_size
    else:
        max_kv_len = block_table.shape[1] * block_size
    seq_lens_i64 = seq_lens.to(torch.int64).clamp(max=max_kv_len)

    token_offsets = torch.arange(block_size, device=q.device, dtype=torch.int64)
    q_view = q.reshape(batch_size, num_kv_heads, gqa_group_size, head_dim)
    sink_view = (
        None if sink is None else sink.reshape(num_kv_heads, gqa_group_size, head_dim)
    )
    out = torch.empty(
        (batch_size, num_kv_heads, gqa_group_size, head_dim),
        dtype=q.dtype,
        device=q.device,
    )

    for batch in range(batch_size):
        for kv_head in range(num_kv_heads):
            logical = kv_topk[kv_head, batch]
            safe_logical = logical.clamp(min=0)
            if req_to_token is not None:
                row = int(req_pool_indices[batch])
                columns = (safe_logical * block_size).clamp(
                    max=req_to_token.shape[1] - 1
                )
                slots = req_to_token[row].to(torch.int64)[columns]
                physical = slots // block_size
                if sanitize_page_ids:
                    physical = physical.clamp(min=0, max=int(num_pages) - 1)
            else:
                physical = block_table[batch].to(torch.int64)[safe_logical]
            physical = physical.clamp(min=0)

            positions = logical[:, None] * block_size + token_offsets[None, :]
            valid = (logical[:, None] >= 0) & (
                positions < seq_lens_i64[batch]
            )  # [topk, block_size]

            k_blocks = k_cache_bnsd[physical, :, kv_head, :].float()
            v_blocks = v_cache_bnsd[physical, :, kv_head, :].float()
            # Triton loads K/V under the same mask with ``other=0.0``. Apply it
            # here too: a NaN or Inf left in an unused cache slot must not
            # survive as ``0 * NaN`` in the weighted sums below.
            keep = valid.unsqueeze(-1)
            k_blocks = k_blocks.masked_fill(~keep, 0.0)
            v_blocks = v_blocks.masked_fill(~keep, 0.0)

            q_head = q_view[batch, kv_head].float()  # [gqa_group_size, D]
            scores = torch.einsum("gd,tnd->gtn", q_head, k_blocks) * sm_scale
            scores = scores.masked_fill(~valid.unsqueeze(0), float("-inf"))

            row_max = scores.reshape(gqa_group_size, -1).max(dim=-1).values
            finite_max = torch.where(
                torch.isfinite(row_max), row_max, torch.zeros_like(row_max)
            )
            # exp(-inf - finite_max) is 0, so invalid slots contribute nothing.
            weights = torch.exp(scores - finite_max[:, None, None])
            numerator = torch.einsum("gtn,tnd->gd", weights, v_blocks)
            denominator = weights.reshape(gqa_group_size, -1).sum(dim=-1)

            if sink_view is not None:
                sink_scores = (q_head * sink_view[kv_head].float()).sum(
                    dim=-1
                ) * sm_scale
                merged_max = torch.maximum(finite_max, sink_scores)
                rescale = torch.exp(finite_max - merged_max)
                denominator = denominator * rescale + torch.exp(
                    sink_scores - merged_max
                )
                numerator = numerator * rescale[:, None]

            normalizer = torch.where(
                denominator > 0,
                denominator,
                torch.ones_like(denominator),
            )
            attention = numerator / normalizer[:, None]
            attention = torch.where(
                denominator[:, None] > 0,
                attention,
                torch.zeros_like(attention),
            )
            out[batch, kv_head] = attention.to(q.dtype)

    return out.view(batch_size, num_q_heads, head_dim)
