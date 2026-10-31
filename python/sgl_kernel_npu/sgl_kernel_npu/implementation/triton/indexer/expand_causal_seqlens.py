"""Expand per-request causal lengths and packed-token request IDs."""

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _expand_causal_seqlens_kernel(
    cache_seqlens_ptr,
    query_lens_ptr,
    seqlens_expanded_ptr,
    token_to_batch_ptr,
    total_q,
    BLOCK_BATCH: tl.constexpr,
    BLOCK_QUERY: tl.constexpr,
):
    request_idx = tl.program_id(0)
    query_chunk_idx = tl.program_id(1)

    # Compute this request's packed-token offset without materializing a cumsum.
    # Batch sizes are small on this path, so loading one short query-length vector
    # per request is cheaper than launching cumsum + repeat_interleave kernels.
    request_offsets = tl.arange(0, BLOCK_BATCH)
    prior_query_lens = tl.load(
        query_lens_ptr + request_offsets,
        mask=request_offsets < request_idx,
        other=0,
    ).to(tl.int32)
    request_output_start = tl.sum(prior_query_lens, axis=0)

    query_len = tl.load(query_lens_ptr + request_idx).to(tl.int32)
    cache_seqlen = tl.load(cache_seqlens_ptr + request_idx).to(tl.int32)
    causal_start = tl.maximum(cache_seqlen - query_len + 1, 0)

    query_offsets = (query_chunk_idx * BLOCK_QUERY + tl.arange(0, BLOCK_QUERY)).to(
        tl.int32
    )
    output_offsets = request_output_start + query_offsets
    mask = (query_offsets < query_len) & (output_offsets < total_q)

    tl.store(
        seqlens_expanded_ptr + output_offsets,
        causal_start + query_offsets,
        mask=mask,
    )
    tl.store(token_to_batch_ptr + output_offsets, request_idx, mask=mask)


def expand_causal_seqlens(
    cache_seqlens: torch.Tensor,
    query_lens: torch.Tensor,
    device: torch.device,
    total_q: int,
    max_query_len: int,
    *,
    seqlens_out: Optional[torch.Tensor] = None,
    token_to_batch_out: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand per-request seq_lens into per-token values (all on device).

    For request *b* with total seq_len *S* and query_len *Q*, each token's
    value ranges from ``S - Q + 1`` to ``S``. The fused kernel also emits the
    packed-token-to-request mapping used by the DSA indexer.
    """
    batch_size = cache_seqlens.shape[0]
    if cache_seqlens.ndim != 1 or query_lens.ndim != 1:
        raise ValueError("cache_seqlens and query_lens must be one-dimensional")
    if query_lens.shape[0] != batch_size:
        raise ValueError(
            "cache_seqlens and query_lens must have the same batch dimension"
        )
    if cache_seqlens.stride(0) != 1 or query_lens.stride(0) != 1:
        raise ValueError("cache_seqlens and query_lens must be contiguous")
    if cache_seqlens.dtype != torch.int32 or query_lens.dtype != torch.int32:
        raise ValueError("cache_seqlens and query_lens must be int32")
    if cache_seqlens.device != device or query_lens.device != device:
        raise ValueError("cache_seqlens and query_lens must be on device")
    if total_q < 0 or max_query_len < 0:
        raise ValueError("total_q and max_query_len must be non-negative")
    if total_q > 0 and max_query_len == 0:
        raise ValueError("max_query_len must be positive when total_q is positive")

    if seqlens_out is None:
        seqlens_out = torch.empty(total_q, dtype=torch.int32, device=device)
    elif (
        seqlens_out.ndim != 1
        or seqlens_out.shape[0] != total_q
        or seqlens_out.dtype != torch.int32
        or seqlens_out.device != device
        or seqlens_out.stride(0) != 1
    ):
        raise ValueError(
            "seqlens_out must be contiguous int32 with shape [total_q] on device"
        )

    if token_to_batch_out is None:
        token_to_batch_out = torch.empty(total_q, dtype=torch.int64, device=device)
    elif (
        token_to_batch_out.ndim != 1
        or token_to_batch_out.shape[0] != total_q
        or token_to_batch_out.dtype != torch.int64
        or token_to_batch_out.device != device
        or token_to_batch_out.stride(0) != 1
    ):
        raise ValueError(
            "token_to_batch_out must be contiguous int64 with shape [total_q] on device"
        )

    if batch_size == 0 or total_q == 0:
        return seqlens_out, token_to_batch_out

    block_batch = triton.next_power_of_2(batch_size)
    block_query = max(32, min(256, triton.next_power_of_2(max_query_len)))
    grid = (batch_size, triton.cdiv(max_query_len, block_query))
    _expand_causal_seqlens_kernel[grid](
        cache_seqlens,
        query_lens,
        seqlens_out,
        token_to_batch_out,
        total_q,
        BLOCK_BATCH=block_batch,
        BLOCK_QUERY=block_query,
    )
    return seqlens_out, token_to_batch_out
