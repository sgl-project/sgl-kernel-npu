"""Public BF16 decode-update and conditional cache-write operator."""

import torch

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_triton_fn = None
_torch_native_fn = None


@register_impl("mem_cache.kpool_decode_update_and_maybe_write_cache", "auto")
@register_impl("mem_cache.kpool_decode_update_and_maybe_write_cache", "triton")
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
        from ...implementation.triton.mem_cache.kpool_decode_update_and_maybe_write_cache import (
            kpool_decode_update_and_maybe_write_cache,
        )

        _triton_fn = kpool_decode_update_and_maybe_write_cache

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


@register_impl("mem_cache.kpool_decode_update_and_maybe_write_cache", "torch_native")
def _torch_native_impl(
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
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.mem_cache.kpool_decode_update_and_maybe_write_cache import (
            kpool_decode_update_and_maybe_write_cache,
        )

        _torch_native_fn = kpool_decode_update_and_maybe_write_cache
    return _torch_native_fn(
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


@register_operator("mem_cache.kpool_decode_update_and_maybe_write_cache")
def kpool_decode_update_and_maybe_write_cache(
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

    ``auto`` selects Triton; ``torch_native`` selects independent PyTorch.

    Args:
        buf: torch.Tensor, shape [num_pages, slots_per_page * head_dim], dtype torch.bfloat16. Flattened index cache; updated in place when a pool completes.
        tail_k: torch.Tensor, shape [R, T, head_dim], dtype torch.bfloat16. Request-local key rings; updated in place for valid rows.
        tail_score: torch.Tensor, shape Same as tail_k, dtype torch.float32 per the backend contract. Scores paired with tail_k; updated in place for valid rows.
        key: torch.Tensor, shape [B, head_dim], dtype torch.bfloat16. Current decode keys.
        slot_score: torch.Tensor, shape [B, head_dim], dtype torch.float32 per the backend contract. Scores paired with key.
        ape: torch.Tensor, shape [pool_size, head_dim], dtype torch.float32. Additive pooling-position scores.
        block_tables: torch.Tensor, shape [B, BLOCK_TABLE_COLS], dtype torch.int32. Token-page table used to locate a completed pool's cache page.
        req_pool_indices: torch.Tensor, shape [>= B], dtype Integer; not explicitly checked by the wrapper. Request/ring row for each batch item.
        positions: torch.Tensor, shape [>= B], dtype Integer; not explicitly checked by the wrapper. Current token position for each batch item.
        seq_lens: torch.Tensor, shape [>= B], dtype Integer; not explicitly checked by the wrapper. Sequence length for each batch item.
        out_cache_loc: torch.Tensor, shape [>= B], dtype Integer; not explicitly checked by the wrapper. Token cache locations; a zero entry suppresses that row's updates. Read only.
        pool_size: int, shape —, dtype —. Keyword-only number of tokens in one pool.
        slots_per_page: int, shape —, dtype —. Keyword-only number of pooled slots per flattened cache page.
        head_dim: int, shape —, dtype —. Keyword-only channel width; must match the tail's last dimension.
        impl: Backend name: auto, triton or torch_native.

    Returns:
        None. Updates buf, tail_k and tail_score in place.
    """
    impl_fn = get_impl_by_name(
        "mem_cache.kpool_decode_update_and_maybe_write_cache", impl
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
