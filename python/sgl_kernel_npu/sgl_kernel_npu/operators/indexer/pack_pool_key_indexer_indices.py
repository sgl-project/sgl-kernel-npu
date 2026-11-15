"""Public pooled-index packing with per-query tails and real query boundaries."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_triton_fn = None
_torch_native_fn = None


@register_impl("indexer.pack_pool_key_indexer_indices", "auto")
@register_impl("indexer.pack_pool_key_indexer_indices", "triton")
def _triton_impl(*args, **kwargs):
    global _triton_fn
    if _triton_fn is None:
        from ...implementation.triton.indexer.pack_pool_key_indexer_indices import (
            pack_pool_key_indexer_indices,
        )

        _triton_fn = pack_pool_key_indexer_indices
    return _triton_fn(*args, **kwargs)


@register_impl("indexer.pack_pool_key_indexer_indices", "torch_native")
def _torch_native_impl(*args, **kwargs):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.indexer.pack_pool_key_indexer_indices import (
            pack_pool_key_indexer_indices,
        )

        _torch_native_fn = pack_pool_key_indexer_indices
    return _torch_native_fn(*args, **kwargs)


@register_operator("indexer.pack_pool_key_indexer_indices")
def pack_pool_key_indexer_indices(
    indices,
    positions,
    query_ends,
    topk,
    pool_size,
    *,
    output_buffer=None,
    impl="auto",
):
    """Pack history and per-query tails into int32 [T,topk+pool_size-1].

    The default output aliases indices; an optional output buffer may be wider.
    Rows after query_ends[-1] become -1. Positions are token-aligned int64 [T];
    query_ends is the original int64 [B] cumulative real-query boundary vector.
    Both are contiguous and on the same device as indices. History candidates
    are consumed as returned by causal CANN; artificial history is discarded
    using the original positions. auto selects Triton; torch_native is a reference.
    """
    return get_impl_by_name("indexer.pack_pool_key_indexer_indices", impl)(
        indices,
        positions,
        query_ends,
        topk,
        pool_size,
        output_buffer=output_buffer,
    )
