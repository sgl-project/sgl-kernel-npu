"""Independent PyTorch expansion of packed-query causal metadata."""

import torch


def expand_causal_seqlens(
    cache_seqlens,
    query_lens,
    device,
    total_q,
    max_query_len,
    *,
    seqlens_out=None,
    token_to_batch_out=None,
):
    """Expand request IDs and compute local offsets using prefix sums."""
    requests = torch.arange(cache_seqlens.numel(), dtype=torch.int64, device=device)
    counts = query_lens.to(torch.int64)
    token_to_batch = torch.repeat_interleave(requests, counts, output_size=total_q)
    packed_starts = torch.cumsum(query_lens, dim=0, dtype=torch.int32) - query_lens
    causal_starts = (cache_seqlens - query_lens + 1).clamp_min(0)
    seqlens = torch.arange(total_q, dtype=torch.int32, device=device) + (
        causal_starts - packed_starts
    ).index_select(0, token_to_batch)
    if seqlens_out is not None:
        seqlens_out.copy_(seqlens)
        seqlens = seqlens_out
    if token_to_batch_out is not None:
        token_to_batch_out.copy_(token_to_batch)
        token_to_batch = token_to_batch_out
    return seqlens, token_to_batch
