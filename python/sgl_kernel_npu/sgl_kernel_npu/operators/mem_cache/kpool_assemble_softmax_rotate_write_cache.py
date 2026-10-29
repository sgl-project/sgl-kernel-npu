"""Public pooled-key assembly and index-cache write operator."""

from typing import Optional

import torch

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_triton_fn = None
_torch_native_fn = None


@register_impl("mem_cache.kpool_assemble_softmax_rotate_write_cache", "auto")
@register_impl("mem_cache.kpool_assemble_softmax_rotate_write_cache", "triton")
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
        from ...implementation.triton.mem_cache.kpool_assemble_softmax_rotate_write_cache import (
            kpool_assemble_softmax_rotate_write_cache,
        )

        _triton_fn = kpool_assemble_softmax_rotate_write_cache

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


@register_impl("mem_cache.kpool_assemble_softmax_rotate_write_cache", "torch_native")
def _torch_native_impl(
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
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.mem_cache.kpool_assemble_softmax_rotate_write_cache import (
            kpool_assemble_softmax_rotate_write_cache,
        )

        _torch_native_fn = kpool_assemble_softmax_rotate_write_cache
    return _torch_native_fn(
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


@register_operator("mem_cache.kpool_assemble_softmax_rotate_write_cache")
def kpool_assemble_softmax_rotate_write_cache(
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

    ``auto`` selects Triton; ``torch_native`` selects independent PyTorch.

    Args:
        index_k_cache: torch.Tensor, shape Contiguous storage flattenable to [cache_rows, 128], dtype torch.bfloat16. Index cache updated in place.
        chunk_k: torch.Tensor, shape [N, 128], dtype torch.bfloat16. Current-chunk keys.
        chunk_score: torch.Tensor, shape Same as chunk_k, dtype torch.float16, torch.bfloat16, or torch.float32. Scores paired with chunk_k.
        tail_k: torch.Tensor, shape [R, T, 128], dtype torch.bfloat16. Request-local key rings; read only.
        tail_score: torch.Tensor, shape Same as tail_k, dtype torch.float16, torch.bfloat16, or torch.float32. Scores paired with tail_k.
        req_pool_idx: torch.Tensor, shape [P], dtype torch.int64. Request/ring row for each of the P pools.
        n_from_tail: torch.Tensor, shape [P], dtype torch.int32 or torch.int64. Number of leading slots read from the tail ring.
        chunk_src_start: torch.Tensor, shape [P], dtype torch.int64. First chunk row used by each pool's suffix.
        tail_logical_base: torch.Tensor, shape [P], dtype torch.int32 or torch.int64. Logical first tail slot; physical positions wrap by T.
        ape: torch.Tensor, shape [pool_size, 128], dtype torch.float32. Additive pooling-position scores; pool_size > 0.
        loc: torch.Tensor, shape [P], dtype torch.int64. Flattened cache rows to write.
        write_mask: torch.Tensor or None, shape [P] or —, dtype torch.bool. Optional per-pool write enable.
        num_programs: int or None, shape —, dtype —. Keyword-only Triton grid size. Must be positive when supplied; it is capped at P. By default, the backend uses the available Vector Core count capped at P.
        impl: Backend name: auto, triton or torch_native.

    Returns:
        None. Updates index_k_cache in place.
    """
    impl_fn = get_impl_by_name(
        "mem_cache.kpool_assemble_softmax_rotate_write_cache", impl
    )
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
