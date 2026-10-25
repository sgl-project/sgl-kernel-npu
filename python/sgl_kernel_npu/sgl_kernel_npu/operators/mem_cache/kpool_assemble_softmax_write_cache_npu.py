"""Public pooled-key assembly and index-cache write operator."""

from typing import Optional

import torch

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_triton_fn = None


@register_impl("mem_cache.kpool_assemble_softmax_write_cache_npu", "auto")
@register_impl("mem_cache.kpool_assemble_softmax_write_cache_npu", "triton")
def _triton_impl(
    index_k_cache: torch.Tensor,
    chunk_k: torch.Tensor,
    chunk_score: torch.Tensor,
    tail_k: torch.Tensor,
    tail_score: torch.Tensor,
    req_pool_idx: torch.Tensor,
    n_from_tail: torch.Tensor,
    chunk_src_start: torch.Tensor,
    tail_logical_base: torch.Tensor,
    ape: torch.Tensor,
    loc: torch.Tensor,
    write_mask: Optional[torch.Tensor] = None,
    *,
    num_programs: Optional[int] = None,
) -> None:
    global _triton_fn
    if _triton_fn is None:
        from ...implementation.triton.mem_cache.kpool_index_npu import (
            kpool_assemble_softmax_write_cache_npu,
        )

        _triton_fn = kpool_assemble_softmax_write_cache_npu

    return _triton_fn(
        index_k_cache,
        chunk_k,
        chunk_score,
        tail_k,
        tail_score,
        req_pool_idx,
        n_from_tail,
        chunk_src_start,
        tail_logical_base,
        ape,
        loc,
        write_mask,
        num_programs=num_programs,
    )


@register_operator("mem_cache.kpool_assemble_softmax_write_cache_npu")
def kpool_assemble_softmax_write_cache_npu(
    index_k_cache: torch.Tensor,
    chunk_k: torch.Tensor,
    chunk_score: torch.Tensor,
    tail_k: torch.Tensor,
    tail_score: torch.Tensor,
    req_pool_idx: torch.Tensor,
    n_from_tail: torch.Tensor,
    chunk_src_start: torch.Tensor,
    tail_logical_base: torch.Tensor,
    ape: torch.Tensor,
    loc: torch.Tensor,
    write_mask: Optional[torch.Tensor] = None,
    *,
    num_programs: Optional[int] = None,
    impl="auto",
) -> None:
    """Assemble pooled BF16 keys and write index-cache rows in place.

    ``impl`` accepts ``auto`` or ``triton``; both currently select the same
    Triton implementation.
    """
    impl_fn = get_impl_by_name("mem_cache.kpool_assemble_softmax_write_cache_npu", impl)
    return impl_fn(
        index_k_cache,
        chunk_k,
        chunk_score,
        tail_k,
        tail_score,
        req_pool_idx,
        n_from_tail,
        chunk_src_start,
        tail_logical_base,
        ape,
        loc,
        write_mask,
        num_programs=num_programs,
    )
