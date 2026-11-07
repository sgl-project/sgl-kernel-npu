"""Public pooled-page-table construction operator."""

import torch

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_torch_native_fn = None


@register_impl("indexer.build_pooled_page_table", "auto")
@register_impl("indexer.build_pooled_page_table", "torch_native")
def _torch_native_impl(page_table_64: torch.Tensor, pool_size: int) -> torch.Tensor:
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.indexer.build_pooled_page_table import (
            build_pooled_page_table,
        )

        _torch_native_fn = build_pooled_page_table

    return _torch_native_fn(page_table_64, pool_size)


@register_operator("indexer.build_pooled_page_table")
def build_pooled_page_table(
    page_table_64: torch.Tensor,
    pool_size: int,
    *,
    impl="auto",
) -> torch.Tensor:
    """Pack each logical pool page into its first token page.

    ``auto`` selects Torch native.

    Args:
        page_table_64: torch.Tensor, shape [..., NUM_BLOCKS], dtype The backend does not restrict it; use an integer page-ID dtype. Token-page table. The last dimension is sampled.
        pool_size: int, shape —, dtype —. Number of token pages represented by one pool page. Must be positive and divide the backend constant BLOCK_SIZE_K = 128.
        impl: Backend name: auto or torch_native.

    Returns:
        Page table with last axis selected at pool_size intervals.
    """
    impl_fn = get_impl_by_name("indexer.build_pooled_page_table", impl)
    return impl_fn(page_table_64, pool_size)
