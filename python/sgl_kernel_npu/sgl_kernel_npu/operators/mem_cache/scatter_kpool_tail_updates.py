"""Public scatter update operator for request-local kpool tail rings."""

import torch

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_triton_fn = None
_torch_native_fn = None


@register_impl("mem_cache.scatter_kpool_tail_updates", "auto")
@register_impl("mem_cache.scatter_kpool_tail_updates", "triton")
def _triton_impl(
    pool,
    chunk_k: torch.Tensor,
    chunk_score: torch.Tensor,
    tail_k: torch.Tensor,
    tail_score: torch.Tensor,
    req_pool_idx: torch.Tensor,
    dst_logical_start: torch.Tensor,
    chunk_src_start: torch.Tensor,
    n_write: torch.Tensor,
) -> None:
    global _triton_fn
    if _triton_fn is None:
        from ...implementation.triton.mem_cache.scatter_kpool_tail_updates import (
            scatter_kpool_tail_updates,
        )

        _triton_fn = scatter_kpool_tail_updates

    return _triton_fn(
        pool,
        chunk_k,
        chunk_score,
        tail_k,
        tail_score,
        req_pool_idx,
        dst_logical_start,
        chunk_src_start,
        n_write,
    )


@register_impl("mem_cache.scatter_kpool_tail_updates", "torch_native")
def _torch_native_impl(
    pool,
    chunk_k: torch.Tensor,
    chunk_score: torch.Tensor,
    tail_k: torch.Tensor,
    tail_score: torch.Tensor,
    req_pool_idx: torch.Tensor,
    dst_logical_start: torch.Tensor,
    chunk_src_start: torch.Tensor,
    n_write: torch.Tensor,
) -> None:
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.mem_cache.scatter_kpool_tail_updates import (
            scatter_kpool_tail_updates,
        )

        _torch_native_fn = scatter_kpool_tail_updates
    return _torch_native_fn(
        pool,
        chunk_k,
        chunk_score,
        tail_k,
        tail_score,
        req_pool_idx,
        dst_logical_start,
        chunk_src_start,
        n_write,
    )


@register_operator("mem_cache.scatter_kpool_tail_updates")
def scatter_kpool_tail_updates(
    pool,
    chunk_k: torch.Tensor,
    chunk_score: torch.Tensor,
    tail_k: torch.Tensor,
    tail_score: torch.Tensor,
    req_pool_idx: torch.Tensor,
    dst_logical_start: torch.Tensor,
    chunk_src_start: torch.Tensor,
    n_write: torch.Tensor,
    *,
    impl="auto",
) -> None:
    """Scatter chunk keys and scores into tail rings in place.

    ``auto`` selects Triton; ``torch_native`` selects independent PyTorch.

    Args:
        pool: Object, shape —, dtype Must expose an integer index_kpool attribute. Pool configuration. The wrapper reads pool.index_kpool as pool_size.
        chunk_k: torch.Tensor, shape [N, 128], dtype Floating point; no explicit wrapper check. Source keys; use the same dtype as tail_k.
        chunk_score: torch.Tensor, shape Same as chunk_k, dtype Floating point; no explicit wrapper check. Source scores; use the same dtype as tail_score.
        tail_k: torch.Tensor, shape [R, T, 128], dtype Floating point; must be compatible with chunk_k. Destination key rings; updated in place.
        tail_score: torch.Tensor, shape Same as tail_k, dtype Floating point; must be compatible with chunk_score. Destination score rings; updated in place.
        req_pool_idx: torch.Tensor, shape [P], dtype Integer; no explicit wrapper check. Destination request/ring row for each plan row.
        dst_logical_start: torch.Tensor, shape [P], dtype Integer; no explicit wrapper check. First logical destination slot for each plan row.
        chunk_src_start: torch.Tensor, shape [P], dtype Integer; no explicit wrapper check. First source chunk row for each plan row.
        n_write: torch.Tensor, shape [P], dtype Integer; no explicit wrapper check. Number of rows to copy for each plan row.
        impl: Backend name: auto, triton or torch_native.

    Returns:
        None. Updates tail_k and tail_score in place.
    """
    impl_fn = get_impl_by_name("mem_cache.scatter_kpool_tail_updates", impl)
    return impl_fn(
        pool,
        chunk_k,
        chunk_score,
        tail_k,
        tail_score,
        req_pool_idx,
        dst_logical_start,
        chunk_src_start,
        n_write,
    )
