"""PyTorch reference for the query-block append."""

import torch


def append_local_block_to_topk_idx(topk_idx, seq_lens, block_size, num_blocks):
    """PyTorch reference: validate the candidates and append the query's block."""
    assert topk_idx.ndim == 3
    assert topk_idx.dtype == torch.int32
    assert topk_idx.is_contiguous()
    assert topk_idx.shape[2] > 0
    assert seq_lens.ndim == 1
    assert seq_lens.shape[0] == topk_idx.shape[1]
    assert seq_lens.is_contiguous()
    assert block_size > 0
    assert num_blocks > 0

    query_pos = (seq_lens.to(torch.int64) - 1).clamp(min=0)  # [B]
    local = torch.clamp(query_pos // block_size, max=num_blocks - 1)  # [B]
    # int64 keeps the validation arithmetic from overflowing; the Triton
    # kernel multiplies int32 candidates by block_size instead.
    candidates = topk_idx.to(torch.int64)
    valid = (
        (candidates >= 0)
        & (candidates < num_blocks)
        & (candidates * block_size <= query_pos.view(1, -1, 1))
    )
    validated = torch.where(valid, candidates, torch.full_like(candidates, -1))
    # ``local`` always passes validation, so checking the kept candidates is
    # equivalent to checking the raw input and matches the kernel's dedup.
    local_present = (validated == local.view(1, -1, 1)).any(dim=-1)
    appended = torch.where(local_present, torch.full_like(local, -1), local)
    return torch.cat([validated, appended.unsqueeze(-1)], dim=-1).to(torch.int32)
