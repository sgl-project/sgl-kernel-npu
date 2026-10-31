"""Independent PyTorch backend for kpool_assemble_softmax_rotate_write_cache."""

from typing import Optional

import torch


@torch.no_grad()
def kpool_assemble_softmax_rotate_write_cache(
    index_k_cache: torch.Tensor,
    chunk_k: torch.Tensor,
    chunk_score: torch.Tensor,
    tail_k: torch.Tensor,
    tail_score: torch.Tensor,
    req_pool_idx: torch.Tensor,
    n_from_tail: torch.Tensor,
    chunk_src_start: torch.Tensor,
    tail_logical_base: torch.Tensor,
    ape: torch.Tensor,
    loc: torch.Tensor,
    write_mask: Optional[torch.Tensor] = None,
    *,
    num_programs: Optional[int] = None
) -> None:
    """Torch numerical reference for BF16 KPool assembly and cache write."""
    n_pools = req_pool_idx.numel()
    if n_pools == 0:
        return
    pool_size = ape.shape[0]
    tail_size = tail_k.shape[1]
    if write_mask is None:
        active_rows = torch.arange(n_pools, dtype=torch.int64, device=chunk_k.device)
    else:
        active_rows = torch.nonzero(write_mask, as_tuple=False).flatten()
    req = req_pool_idx.index_select(0, active_rows)
    n_tail = n_from_tail.index_select(0, active_rows).to(torch.int64)
    chunk_src = chunk_src_start.index_select(0, active_rows)
    tail_base = tail_logical_base.index_select(0, active_rows).to(torch.int64)
    slots = torch.arange(pool_size, dtype=torch.int64, device=chunk_k.device).unsqueeze(
        0
    )
    use_tail = slots < n_tail.unsqueeze(1)
    tail_phys = torch.remainder(tail_base.unsqueeze(1) + slots, tail_size)
    tail_req = req.unsqueeze(1).expand_as(tail_phys)
    chunk_offsets = torch.clamp(slots - n_tail.unsqueeze(1), min=0)
    chunk_indices = chunk_src.unsqueeze(1) + chunk_offsets
    assembled_k = torch.empty(
        (*use_tail.shape, chunk_k.shape[-1]), dtype=chunk_k.dtype, device=chunk_k.device
    )
    assembled_score = torch.empty(
        assembled_k.shape, dtype=torch.float32, device=chunk_k.device
    )
    # Gather only selected branches; full-tail rows may point past an empty chunk.
    assembled_k[use_tail] = tail_k[tail_req[use_tail], tail_phys[use_tail]]
    assembled_score[use_tail] = tail_score[
        tail_req[use_tail], tail_phys[use_tail]
    ].float()
    use_chunk = ~use_tail
    assembled_k[use_chunk] = chunk_k[chunk_indices[use_chunk]]
    assembled_score[use_chunk] = chunk_score[chunk_indices[use_chunk]].float()
    logits = assembled_score.float() + ape.float().unsqueeze(0)
    probs = torch.softmax(logits, dim=1)
    pooled = torch.sum(assembled_k.float() * probs, dim=1).to(torch.bfloat16)
    cache_2d = index_k_cache.view(-1, 128)
    active_locs = loc.index_select(0, active_rows)
    cache_2d.index_copy_(0, active_locs, pooled)
