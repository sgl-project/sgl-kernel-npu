"""Public pool-topk expansion and uncompressed-tail append operator."""

import torch

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_triton_fn = None
_torch_native_fn = None


@register_impl("indexer.expand_kpool_group_and_tail_to_topk", "auto")
@register_impl("indexer.expand_kpool_group_and_tail_to_topk", "triton")
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
        from ...implementation.triton.indexer.expand_kpool_group_and_tail_to_topk import (
            _expand_kpool_group_and_tail_to_topk,
        )

        _triton_fn = _expand_kpool_group_and_tail_to_topk

    return _triton_fn(
        pool_indices,
        seq_lens_per_token,
        n_real,
        num_q_padded,
        pool_size,
        index_topk,
    )


@register_impl("indexer.expand_kpool_group_and_tail_to_topk", "torch_native")
def _torch_native_impl(
    pool_indices: torch.Tensor,
    seq_lens_per_token: torch.Tensor,
    n_real: int,
    num_q_padded: int,
    pool_size: int,
    index_topk: int,
) -> torch.Tensor:
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.indexer.expand_kpool_group_and_tail_to_topk import (
            expand_kpool_group_and_tail_to_topk,
        )

        _torch_native_fn = expand_kpool_group_and_tail_to_topk
    return _torch_native_fn(
        pool_indices, seq_lens_per_token, n_real, num_q_padded, pool_size, index_topk
    )


@register_operator("indexer.expand_kpool_group_and_tail_to_topk")
def expand_kpool_group_and_tail_to_topk(
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

    ``auto`` selects Triton; ``torch_native`` selects independent PyTorch.

    Args:
        pool_indices: torch.Tensor, shape [input_rows, N_POOL_TOPK], with enough rows for the effective n_real, dtype torch.int32. Selected pool positions. Row strides are passed to the kernel.
        seq_lens_per_token: torch.Tensor, shape [input_rows], with enough rows for the effective n_real, dtype torch.int32. Request length for each query row; use contiguous one-dimensional storage.
        n_real: int, shape —, dtype —. Requested valid row count. The backend clamps it to the smaller input row count before launching.
        num_q_padded: int, shape —, dtype —. Output row count; must be at least the effective n_real.
        pool_size: int, shape —, dtype —. Number of tokens represented by one pool; Triton requires pool_size >= 4.
        index_topk: int, shape —, dtype —. Token-level top-k; must equal N_POOL_TOPK * pool_size.
        impl: Backend name: auto, triton or torch_native.

    Returns:
        Int32 [num_q_padded, index_topk + pool_size - 1] token indices.
    """
    impl_fn = get_impl_by_name("indexer.expand_kpool_group_and_tail_to_topk", impl)
    return impl_fn(
        pool_indices,
        seq_lens_per_token,
        n_real,
        num_q_padded,
        pool_size,
        index_topk,
    )
