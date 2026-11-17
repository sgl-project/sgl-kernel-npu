"""PyTorch reference for the prefill block-topk operator."""

import torch


def flash_prefill_bnsd_topk_from_score(score, per_query_seq_lens, topk, block_size_k):
    """Select the top-k blocks per query and append the query's local block."""
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

    # Per-query causal position and the block it belongs to (inclusive length).
    q_pos = (per_query_seq_lens.to(torch.long) - 1).clamp(min=0)
    local = (q_pos // block_size_k).clamp(max=max_seqblock_k - 1)

    out = torch.full(
        (num_kv_heads, total_q, topk + 1), -1, dtype=torch.int32, device=device
    )
    # Repeated max: same order as the kernel, invalid picks keep their -1 slot.
    work = score.clone()
    cols = torch.arange(max_seqblock_k, device=device)
    for rank in range(topk):
        best = work.max(dim=-1, keepdim=True).values
        pos = torch.where(work == best, cols, max_seqblock_k).min(dim=-1).values
        valid = (best.squeeze(-1) > float("-inf")) & (pos * block_size_k <= q_pos)
        out[..., rank] = torch.where(valid, pos, -1).to(torch.int32)
        work.scatter_(-1, pos.unsqueeze(-1), float("-inf"))

    # Append the causal local block unless the selected slots already contain it.
    present = (out[..., :topk] == local.unsqueeze(-1)).any(dim=-1)
    out[..., topk] = torch.where(present, torch.full_like(local, -1), local).to(
        torch.int32
    )
    return out
