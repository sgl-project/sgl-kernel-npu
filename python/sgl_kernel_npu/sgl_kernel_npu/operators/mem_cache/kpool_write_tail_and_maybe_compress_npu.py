"""Public tail-ring write and completed-pool compression operator."""

from typing import Optional

import torch

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_triton_fn = None


@register_impl("mem_cache.kpool_write_tail_and_maybe_compress_npu", "auto")
@register_impl("mem_cache.kpool_write_tail_and_maybe_compress_npu", "triton")
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
        from ...implementation.triton.mem_cache.kpool_index_npu import (
            kpool_write_tail_and_maybe_compress_npu,
        )

        _triton_fn = kpool_write_tail_and_maybe_compress_npu

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


@register_operator("mem_cache.kpool_write_tail_and_maybe_compress_npu")
def kpool_write_tail_and_maybe_compress_npu(
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

    ``impl`` accepts ``auto`` or ``triton``; both currently select the same
    Triton implementation.
    """
    impl_fn = get_impl_by_name(
        "mem_cache.kpool_write_tail_and_maybe_compress_npu", impl
    )
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
