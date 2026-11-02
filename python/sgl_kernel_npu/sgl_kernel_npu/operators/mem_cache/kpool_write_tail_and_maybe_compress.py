"""Public tail-ring write and completed-pool compression operator."""

from typing import Optional

import torch

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_triton_fn = None
_torch_native_fn = None


@register_impl("mem_cache.kpool_write_tail_and_maybe_compress", "auto")
@register_impl("mem_cache.kpool_write_tail_and_maybe_compress", "triton")
def _triton_impl(
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
    num_programs: Optional[int] = None,
) -> None:
    global _triton_fn
    if _triton_fn is None:
        from ...implementation.triton.mem_cache.kpool_write_tail_and_maybe_compress import (
            kpool_write_tail_and_maybe_compress,
        )

        _triton_fn = kpool_write_tail_and_maybe_compress

    return _triton_fn(
        index_k_cache,
        key,
        score,
        tail_k,
        tail_score,
        ape,
        req_pool_indices,
        write_start,
        tail_logical_start,
        write_loc,
        out_cache_loc,
        num_draft_tokens,
        effective_n_per_batch,
        num_programs=num_programs,
    )


@register_impl("mem_cache.kpool_write_tail_and_maybe_compress", "torch_native")
def _torch_native_impl(
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
    num_programs: Optional[int] = None,
) -> None:
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.mem_cache.kpool_write_tail_and_maybe_compress import (
            kpool_write_tail_and_maybe_compress,
        )

        _torch_native_fn = kpool_write_tail_and_maybe_compress
    return _torch_native_fn(
        index_k_cache,
        key,
        score,
        tail_k,
        tail_score,
        ape,
        req_pool_indices,
        write_start,
        tail_logical_start,
        write_loc,
        out_cache_loc,
        num_draft_tokens,
        effective_n_per_batch,
        num_programs=num_programs,
    )


@register_operator("mem_cache.kpool_write_tail_and_maybe_compress")
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
    num_programs: Optional[int] = None,
    impl="auto",
) -> None:
    """Write draft tokens to tail rings and compress completed pools in place.

    ``auto`` selects Triton; ``torch_native`` selects independent PyTorch.

    Args:
        index_k_cache: torch.Tensor, shape Contiguous storage flattenable to [cache_rows, 128], dtype torch.bfloat16. Cache updated in place for completed pools.
        key: torch.Tensor, shape [B * N, 128], dtype torch.bfloat16. Draft keys.
        score: torch.Tensor, shape Same as key, dtype torch.float16, torch.bfloat16, or torch.float32. Scores paired with key.
        tail_k: torch.Tensor, shape [R, pool_size + N, 128], dtype torch.bfloat16. Tail rings updated in place.
        tail_score: torch.Tensor, shape Same as tail_k, dtype torch.float16, torch.bfloat16, or torch.float32. Score rings updated in place.
        ape: torch.Tensor, shape [pool_size, 128], dtype torch.float32. Additive pooling-position scores; pool_size > 0.
        req_pool_indices: torch.Tensor, shape [>= B], dtype torch.int64. Request/ring row for each batch item.
        write_start: torch.Tensor, shape [>= B], dtype torch.int32 or torch.int64. Logical tail position at which each batch starts.
        tail_logical_start: torch.Tensor, shape [>= B], dtype torch.int32 or torch.int64. Logical start of the candidate completed pool.
        write_loc: torch.Tensor, shape [B, ceil(N / pool_size)], dtype torch.int64. Candidate cache rows; read only and must be contiguous.
        out_cache_loc: torch.Tensor, shape [>= B * N], dtype torch.int32 or torch.int64. Token cache locations; a zero first entry for a batch skips that whole batch. Read only.
        num_draft_tokens: int, shape —, dtype —. Static N; must be positive and divide key.shape[0].
        effective_n_per_batch: torch.Tensor or None, shape [>= B] or —, dtype torch.int32 or torch.int64. Optional effective draft count used only for pool-completion decisions.
        num_programs: int or None, shape —, dtype —. Keyword-only Triton grid size; positive when supplied and capped at B. Defaults to the available Vector Core count capped at B.
        impl: Backend name: auto, triton or torch_native.

    Returns:
        None. Updates index_k_cache, tail_k and tail_score in place.
    """
    impl_fn = get_impl_by_name("mem_cache.kpool_write_tail_and_maybe_compress", impl)
    return impl_fn(
        index_k_cache,
        key,
        score,
        tail_k,
        tail_score,
        ape,
        req_pool_indices,
        write_start,
        tail_logical_start,
        write_loc,
        out_cache_loc,
        num_draft_tokens,
        effective_n_per_batch,
        num_programs=num_programs,
    )
