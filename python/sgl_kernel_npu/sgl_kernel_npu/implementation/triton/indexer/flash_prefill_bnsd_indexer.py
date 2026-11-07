"""Triton implementation of the fused prefill indexer (score + attention + topk).

Composition of the two migrated Triton leaves, preserved line for line from the
legacy ``flash_prefill_bnsd_indexer``: only the imports point at the new leaf
modules.
"""

from typing import Optional

import torch

from .flash_prefill_bnsd_score_attn import flash_prefill_bnsd_score_attn
from .flash_prefill_bnsd_topk_from_score import flash_prefill_bnsd_topk_from_score


def flash_prefill_bnsd_indexer(
    q: torch.Tensor,  # idx_q [total_q, num_idx_heads, idx_dim]
    k_cache_bnsd: torch.Tensor,  # idx_k
    v_cache_bnsd: torch.Tensor,  # idx_v
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
):
    """Prefill indexer (fused): returns (idx_o, topk_idx)."""
    score, idx_o = flash_prefill_bnsd_score_attn(
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
        score_type,
        qblock_mappings,
    )
    if per_query_seq_lens is not None and q.shape[1] == k_cache_bnsd.shape[2]:
        # Fused TopK + local-block append -> [num_kv_heads, total_q, topk + 1].
        return idx_o, flash_prefill_bnsd_topk_from_score(
            score, per_query_seq_lens, topk, block_size_k
        )
    max_seqblock_k = score.shape[-1]
    actual_topk = min(topk, max_seqblock_k)
    _, idx = torch.topk(score, k=actual_topk, dim=-1)  # [num_idx_heads, total_q, k]
    idx = idx.to(torch.int32)
    if actual_topk < topk:
        pad = torch.full(
            (idx.shape[0], idx.shape[1], topk - actual_topk),
            -1,
            device=idx.device,
            dtype=idx.dtype,
        )
        idx = torch.cat([idx, pad], dim=-1)
    return idx_o, idx.contiguous()
