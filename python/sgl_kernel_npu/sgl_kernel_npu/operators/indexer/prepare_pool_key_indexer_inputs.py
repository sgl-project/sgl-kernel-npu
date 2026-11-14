"""Public sentinel preparation for the causal paged pool-key indexer."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_triton_fn = None
_torch_native_fn = None


@register_impl("indexer.prepare_pool_key_indexer_inputs", "auto")
@register_impl("indexer.prepare_pool_key_indexer_inputs", "triton")
def _triton_impl(*args, **kwargs):
    global _triton_fn
    if _triton_fn is None:
        from ...implementation.triton.indexer.prepare_pool_key_indexer_inputs import (
            prepare_pool_key_indexer_inputs,
        )

        _triton_fn = prepare_pool_key_indexer_inputs
    return _triton_fn(*args, **kwargs)


@register_impl("indexer.prepare_pool_key_indexer_inputs", "torch_native")
def _torch_native_impl(*args, **kwargs):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.indexer.prepare_pool_key_indexer_inputs import (
            prepare_pool_key_indexer_inputs,
        )

        _torch_native_fn = prepare_pool_key_indexer_inputs
    return _torch_native_fn(*args, **kwargs)


@register_operator("indexer.prepare_pool_key_indexer_inputs")
def prepare_pool_key_indexer_inputs(
    query, weights, ends, lengths, tails, table, *, impl="auto"
):
    """Append zero query/weight rows and safe request metadata for CANN.

    Inputs are [T,H,D], [T,H], three int64 [R] vectors and an int32
    [M,C] page table with M >= R and C >= 1. All tensors share the query device.
    Returns six independent contiguous tensors with one extra token/request.
    Pooled lengths are clamped to at least one; the sentinel end is T + 1.
    Cache block zero and each empty history's first block must be readable.
    Extra table rows, including row R used by the sentinel, must be valid.
    Crop the sentinel result and pack using the original query metadata.
    """
    return get_impl_by_name("indexer.prepare_pool_key_indexer_inputs", impl)(
        query, weights, ends, lengths, tails, table
    )
