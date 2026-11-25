"""Shared attention and paged-cache reference helpers."""

import torch


def dense_attention(query, key, value, scale, *, mask=None, bias=None):
    """Compute FP32 attention for [B, H, S, D] tensors."""
    repeats = query.size(1) // key.size(1)
    if query.size(1) % key.size(1) or key.size(1) != value.size(1):
        raise ValueError("Query heads must be a multiple of the key/value heads")
    key = key.float().repeat_interleave(repeats, dim=1)
    value = value.float().repeat_interleave(repeats, dim=1)
    scores = (query.float() @ key.transpose(-1, -2)) * scale
    if bias is not None:
        scores = scores + bias.float()
    if mask is not None:
        if mask.dtype in (torch.bool, torch.uint8):
            scores = scores.masked_fill(mask.bool(), -torch.inf)
        else:
            scores = scores + mask.float()
    # Softmax of a fully masked row is NaN in PyTorch. FIA returns zero there.
    probabilities = scores.softmax(dim=-1)
    probabilities = probabilities.masked_fill(
        scores.isneginf().all(dim=-1, keepdim=True), 0
    )
    return (probabilities @ value).to(query.dtype), scores.logsumexp(
        dim=-1, keepdim=True
    )


def _paged_sequence(
    cache: torch.Tensor, block_table: torch.Tensor, sequence_length: int
) -> torch.Tensor:
    """Gather logical [S, N2, D] rows from [blocks, block_size, N2, D] pages."""
    block_size = cache.shape[1]
    if sequence_length == 0:
        return cache.new_empty((0, *cache.shape[2:]))
    blocks = (sequence_length + block_size - 1) // block_size
    pages = [
        cache[int(block_table[i])].reshape(block_size, cache.shape[2], cache.shape[3])
        for i in range(blocks)
    ]
    return torch.cat(pages, dim=0)[:sequence_length]
