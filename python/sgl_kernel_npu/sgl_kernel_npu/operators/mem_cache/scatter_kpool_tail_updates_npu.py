"""Public scatter update operator for request-local kpool tail rings."""

import torch

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_triton_fn = None


@register_impl("mem_cache.scatter_kpool_tail_updates_npu", "auto")
@register_impl("mem_cache.scatter_kpool_tail_updates_npu", "triton")
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
        from ...implementation.triton.mem_cache.kpool_index_npu import (
            scatter_kpool_tail_updates_npu,
        )

        _triton_fn = scatter_kpool_tail_updates_npu

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


@register_operator("mem_cache.scatter_kpool_tail_updates_npu")
def scatter_kpool_tail_updates_npu(
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

    ``pool.index_kpool`` supplies the pool size. ``impl`` accepts ``auto`` or
    ``triton``; both currently select the same Triton implementation.
    """
    impl_fn = get_impl_by_name("mem_cache.scatter_kpool_tail_updates_npu", impl)
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
