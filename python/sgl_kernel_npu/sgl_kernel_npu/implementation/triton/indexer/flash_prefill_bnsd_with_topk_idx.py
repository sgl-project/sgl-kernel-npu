"""Triton implementation of the prefill score + topk operator.

Composition of the two migrated Triton leaves, preserved line for line from the
legacy ``flash_prefill_bnsd_with_topk_idx``: only the imports point at the new
leaf modules.
"""

from typing import Optional

import torch

from .flash_prefill_bnsd_score import flash_prefill_bnsd_score
from .flash_prefill_bnsd_topk_from_score import flash_prefill_bnsd_topk_from_score


def flash_prefill_bnsd_with_topk_idx(
    q: torch.Tensor,  # idx_q [total_q, num_idx_heads, idx_dim]
    k_cache_bnsd: torch.Tensor,  # idx_k [num_pages, page_size, num_kv_heads, idx_dim]
    cu_seqlens: torch.Tensor,
    seq_lens: torch.Tensor,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    block_size_q: int,
    block_size_k: int,
    topk: int,
    sm_scale: float,
    score_type: str = "max",
    qblock_mappings: Optional[tuple] = None,
    per_query_seq_lens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Prefill indexer: score (varlen, batched over query-blocks) + topk."""
    max_seqblock_k = (int(seq_lens.max().item()) + block_size_k - 1) // block_size_k
    score = flash_prefill_bnsd_score(
        q,
        k_cache_bnsd,
        cu_seqlens,
        seq_lens,
        req_to_token,
        req_pool_indices,
        block_size_q,
        block_size_k,
        sm_scale,
        score_type,
        qblock_mappings=qblock_mappings,
    )
    if per_query_seq_lens is not None and q.shape[1] == k_cache_bnsd.shape[2]:
        return flash_prefill_bnsd_topk_from_score(
            score, per_query_seq_lens, topk, block_size_k
        )
    actual_topk = min(topk, max_seqblock_k)
    idx = torch.topk(score, k=actual_topk, dim=-1).indices.to(torch.int32)
    if actual_topk < topk:
        pad = torch.full(
            (idx.shape[0], idx.shape[1], topk - actual_topk),
            -1,
            device=idx.device,
            dtype=idx.dtype,
        )
        idx = torch.cat([idx, pad], dim=-1)
    return idx.contiguous()
