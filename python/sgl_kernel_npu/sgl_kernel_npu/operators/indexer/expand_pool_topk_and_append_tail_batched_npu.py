"""Public pool-topk expansion and uncompressed-tail append operator."""

import torch

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_triton_fn = None


@register_impl("indexer.expand_pool_topk_and_append_tail_batched_npu", "auto")
@register_impl("indexer.expand_pool_topk_and_append_tail_batched_npu", "triton")
def _triton_impl(
    pool_indices: torch.Tensor,
    seq_lens_per_token: torch.Tensor,
    n_real: int,
    num_q_padded: int,
    pool_size: int,
    index_topk: int,
) -> torch.Tensor:
    global _triton_fn
    if _triton_fn is None:
        from ...implementation.triton.indexer.kpool_topk_npu import (
            _expand_pool_topk_and_append_tail_batched_npu,
        )

        _triton_fn = _expand_pool_topk_and_append_tail_batched_npu

    return _triton_fn(
        pool_indices,
        seq_lens_per_token,
        n_real,
        num_q_padded,
        pool_size,
        index_topk,
    )


@register_operator("indexer.expand_pool_topk_and_append_tail_batched_npu")
def expand_pool_topk_and_append_tail_batched_npu(
    pool_indices: torch.Tensor,
    seq_lens_per_token: torch.Tensor,
    n_real: int,
    num_q_padded: int,
    pool_size: int,
    index_topk: int,
    *,
    impl="auto",
) -> torch.Tensor:
    """Expand pool-level top-k indices and append valid tail-token indices.

    Returns an int32 tensor with padded rows and invalid slots set to ``-1``.
    ``impl`` accepts ``auto`` or ``triton``.
    """
    impl_fn = get_impl_by_name(
        "indexer.expand_pool_topk_and_append_tail_batched_npu", impl
    )
    return impl_fn(
        pool_indices,
        seq_lens_per_token,
        n_real,
        num_q_padded,
        pool_size,
        index_topk,
    )
