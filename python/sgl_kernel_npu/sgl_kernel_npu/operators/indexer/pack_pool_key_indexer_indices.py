"""Public pooled-index packing with per-query tails and padding masks."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator


@register_impl("indexer.pack_pool_key_indexer_indices", "auto")
@register_impl("indexer.pack_pool_key_indexer_indices", "triton")
def _triton_impl(*args, **kwargs):
    from ...implementation.triton.indexer.pack_pool_key_indexer_indices import (
        pack_pool_key_indexer_indices,
    )

    return pack_pool_key_indexer_indices(*args, **kwargs)


@register_operator("indexer.pack_pool_key_indexer_indices")
def pack_pool_key_indexer_indices(
    indices,
    positions,
    valid_mask,
    topk,
    pool_size,
    *,
    output_buffer=None,
    impl="auto",
):
    """Pack history and per-query tails into int32 [T,topk+pool_size-1].

    The default output aliases indices; an optional output buffer may be wider.
    Invalid query rows become -1. Positions and the boolean valid mask are
    contiguous token-aligned vectors on the same device as indices.
    """
    return get_impl_by_name("indexer.pack_pool_key_indexer_indices", impl)(
        indices,
        positions,
        valid_mask,
        topk,
        pool_size,
        output_buffer=output_buffer,
    )
