"""Independent PyTorch backend for expand_kpool_group_and_tail_to_topk."""

import torch


def expand_kpool_group_and_tail_to_topk(
    pool_indices: torch.Tensor,
    seq_lens_per_token: torch.Tensor,
    n_real: int,
    num_q_padded: int,
    pool_size: int,
    index_topk: int,
):
    """Independent PyTorch expansion and uncompressed-tail reference."""
    n_pool_topk = pool_indices.shape[1]
    assert index_topk == n_pool_topk * pool_size
    tail_pool = pool_size - 1
    n_real = min(n_real, pool_indices.shape[0], seq_lens_per_token.shape[0])
    offsets = torch.arange(pool_size, dtype=torch.int32, device=pool_indices.device)
    token_indices = (
        pool_indices[:n_real].to(torch.int32).unsqueeze(-1) * pool_size + offsets
    )
    token_indices = token_indices.reshape(n_real, index_topk)
    out_cols = index_topk + tail_pool
    out = torch.full(
        (num_q_padded, out_cols), -1, dtype=torch.int32, device=pool_indices.device
    )
    out[:n_real, :index_topk] = token_indices
    seq_lens_used = seq_lens_per_token[:n_real]
    pool_lens_per_token = torch.div(seq_lens_used, pool_size, rounding_mode="floor").to(
        torch.int32
    )
    tail_starts = pool_lens_per_token * pool_size
    tail_counts = (seq_lens_used - tail_starts).to(torch.int32)
    for index in range(tail_pool):
        mask = tail_counts > index
        values = (tail_starts + index).to(torch.int32)
        out[:n_real, index_topk + index] = torch.where(
            mask, values, torch.full_like(values, -1)
        )
    return out
