"""Public BF16 decode-update and conditional cache-write operator."""

import torch

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_triton_fn = None


@register_impl("mem_cache.kpool_decode_update_and_maybe_write_cache_bf16", "auto")
@register_impl("mem_cache.kpool_decode_update_and_maybe_write_cache_bf16", "triton")
def _triton_impl(
    buf: torch.Tensor,
    tail_k: torch.Tensor,
    tail_score: torch.Tensor,
    key: torch.Tensor,
    slot_score: torch.Tensor,
    ape: torch.Tensor,
    block_tables: torch.Tensor,
    req_pool_indices: torch.Tensor,
    positions: torch.Tensor,
    seq_lens: torch.Tensor,
    out_cache_loc: torch.Tensor,
    *,
    pool_size: int,
    slots_per_page: int,
    head_dim: int = 128,
) -> None:
    global _triton_fn
    if _triton_fn is None:
        from ...implementation.triton.mem_cache.kpool_index_npu import (
            kpool_decode_update_and_maybe_write_cache_bf16,
        )

        _triton_fn = kpool_decode_update_and_maybe_write_cache_bf16

    return _triton_fn(
        buf,
        tail_k,
        tail_score,
        key,
        slot_score,
        ape,
        block_tables,
        req_pool_indices,
        positions,
        seq_lens,
        out_cache_loc,
        pool_size=pool_size,
        slots_per_page=slots_per_page,
        head_dim=head_dim,
    )


@register_operator("mem_cache.kpool_decode_update_and_maybe_write_cache_bf16")
def kpool_decode_update_and_maybe_write_cache_bf16(
    buf: torch.Tensor,
    tail_k: torch.Tensor,
    tail_score: torch.Tensor,
    key: torch.Tensor,
    slot_score: torch.Tensor,
    ape: torch.Tensor,
    block_tables: torch.Tensor,
    req_pool_indices: torch.Tensor,
    positions: torch.Tensor,
    seq_lens: torch.Tensor,
    out_cache_loc: torch.Tensor,
    *,
    pool_size: int,
    slots_per_page: int,
    head_dim: int = 128,
    impl="auto",
) -> None:
    """Update BF16 decode tails and conditionally write completed pools.

    The cache and tail buffers are updated in place. ``impl`` accepts ``auto``
    or ``triton``; both currently select the same Triton implementation.
    """
    impl_fn = get_impl_by_name(
        "mem_cache.kpool_decode_update_and_maybe_write_cache_bf16", impl
    )
    return impl_fn(
        buf,
        tail_k,
        tail_score,
        key,
        slot_score,
        ape,
        block_tables,
        req_pool_indices,
        positions,
        seq_lens,
        out_cache_loc,
        pool_size=pool_size,
        slots_per_page=slots_per_page,
        head_dim=head_dim,
    )
