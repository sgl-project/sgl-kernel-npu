from typing import Optional

import torch

from .fia_prep import fia_prep


def flash_prefill_bnsd_blockq_sparse_fia(
    q: torch.Tensor,  # [total_q, num_q_heads, head_dim]
    k_cache_bnsd: torch.Tensor,  # [num_pages, block_size, num_kv_heads, head_dim]
    v_cache_bnsd: torch.Tensor,  # same shape
    topk_idx: torch.Tensor,  # [num_kv_heads, total_q, topk+1] int32, -1-padded
    seq_lens: torch.Tensor,  # [total_q] int32 -- per-query CAUSAL KV length (abs_pos+1)
    per_query_req: torch.Tensor,  # [total_q] int32/int64 -- request index per query
    req_to_token: torch.Tensor,  # [num_requests, max_ctx] int32 -- page slot table
    block_size: int,
    sm_scale: Optional[float],
    num_pages: int,
    max_num_blocks: int,
    block_table_out: Optional[
        torch.Tensor
    ] = None,  # reuse buffer (skip per-call empty)
    actual_kvlen_out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Single-pass FIA sparse+causal prefill attention. Returns [total_q, QH, D]."""
    assert q.dtype in (torch.float16, torch.bfloat16)
    total_q, num_q_heads, head_dim = q.shape
    num_pages_c, block_size_c, num_kv_heads, cache_head_dim = k_cache_bnsd.shape
    assert block_size_c == block_size
    assert cache_head_dim == head_dim
    assert k_cache_bnsd.shape == v_cache_bnsd.shape
    assert num_kv_heads == 1, (
        "FIA sparse path supports num_kv_heads==1 (MiniMax-M3 gqa=16); "
        f"got {num_kv_heads} -- use the triton blockq path instead."
    )
    assert topk_idx.shape[0] == num_kv_heads
    assert topk_idx.shape[1] == total_q
    assert topk_idx.dtype == torch.int32
    table, lengths = fia_prep(
        topk_idx[0].contiguous(),
        seq_lens,
        per_query_req,
        req_to_token,
        block_size,
        block_table_out,
        actual_kvlen_out,
    )
    scale = head_dim**-0.5 if sm_scale is None else sm_scale
    out = torch.empty_like(q)
    for row in range(total_q):
        length = int(lengths[row])
        pages = table[row, : (length + block_size - 1) // block_size].long()
        keys = k_cache_bnsd[pages].reshape(-1, head_dim)[:length].float()
        values = v_cache_bnsd[pages].reshape(-1, head_dim)[:length].float()
        score = q[row].float() @ keys.T * scale
        out[row] = (score.softmax(-1) @ values).to(q.dtype)
    return out
