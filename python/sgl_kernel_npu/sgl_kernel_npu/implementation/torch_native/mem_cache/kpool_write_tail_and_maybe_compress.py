"""Independent PyTorch backend for kpool_write_tail_and_maybe_compress."""

from typing import Optional

import torch


@torch.no_grad()
def kpool_write_tail_and_maybe_compress(
    index_k_cache: torch.Tensor,
    key: torch.Tensor,
    score: torch.Tensor,
    tail_k: torch.Tensor,
    tail_score: torch.Tensor,
    ape: torch.Tensor,
    req_pool_indices: torch.Tensor,
    write_start: torch.Tensor,
    tail_logical_start: torch.Tensor,
    write_loc: torch.Tensor,
    out_cache_loc: torch.Tensor,
    num_draft_tokens: int,
    effective_n_per_batch: Optional[torch.Tensor] = None,
    *,
    num_programs: Optional[int] = None
) -> None:
    """Torch reference matching the original target-verify semantics."""
    total_rows = key.shape[0]
    if total_rows == 0:
        return
    batch_size = total_rows // num_draft_tokens
    pool_size = ape.shape[0]
    tail_size = tail_k.shape[1]
    cache_2d = index_k_cache.view(-1, 128)
    for batch in range(batch_size):
        if out_cache_loc[batch * num_draft_tokens].item() == 0:
            continue
        req = int(req_pool_indices[batch].item())
        start = int(write_start[batch].item())
        for draft_offset in range(num_draft_tokens):
            row = batch * num_draft_tokens + draft_offset
            tail_slot = (start + draft_offset) % tail_size
            tail_k[req, tail_slot] = key[row]
            tail_score[req, tail_slot] = score[row]
        effective_n = (
            num_draft_tokens
            if effective_n_per_batch is None
            else int(effective_n_per_batch[batch].item())
        )
        effective_n = max(0, min(effective_n, num_draft_tokens))
        base_pool = start // pool_size
        completed_pools = (start + effective_n) // pool_size - base_pool
        if completed_pools == 0:
            continue
        first_logical_start = int(tail_logical_start[batch].item())
        slots = torch.arange(pool_size, dtype=torch.int64, device=key.device)
        for pool_offset in range(completed_pools):
            logical_start = first_logical_start + pool_offset * pool_size
            tail_slots = torch.remainder(logical_start + slots, tail_size)
            pool_keys = tail_k[req].index_select(0, tail_slots).float()
            pool_scores = tail_score[req].index_select(0, tail_slots).float()
            probs = torch.softmax(pool_scores + ape.float(), dim=0)
            pooled = torch.sum(pool_keys * probs, dim=0).to(torch.bfloat16)
            cache_2d[int(write_loc[batch, pool_offset].item())] = pooled
