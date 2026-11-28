"""Public metadata preparation operator."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_triton_fn = None
_torch_native_fn = None


@register_impl("indexer.fused_prepare_lim_metadata", "auto")
@register_impl("indexer.fused_prepare_lim_metadata", "triton")
def _triton_impl(**kwargs):
    global _triton_fn
    if _triton_fn is None:
        from ...implementation.triton.indexer.fused_prepare_lim_metadata import (
            fused_prepare_lim_metadata,
        )

        _triton_fn = fused_prepare_lim_metadata
    return _triton_fn(**kwargs)


@register_impl("indexer.fused_prepare_lim_metadata", "torch_native")
def _torch_native_impl(**kwargs):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.indexer.fused_prepare_lim_metadata import (
            fused_prepare_lim_metadata,
        )

        _torch_native_fn = fused_prepare_lim_metadata
    return _torch_native_fn(**kwargs)


@register_operator("indexer.fused_prepare_lim_metadata")
def fused_prepare_lim_metadata(
    *,
    seq_lens,
    request_slots,
    last_prefix,
    block_table,
    out_block_table,
    safe_slots,
    lengths,
    sparse_rows,
    valid_rows,
    request_states,
    next_prefixes,
    cache_sizes,
    capacity: int,
    request_rows: int,
    query_width: int = 1,
    positions=None,
    resident_safe_slots=None,
    request_to_resident=None,
    resident_rows=None,
    impl="auto",
) -> None:
    """Prepare LIM metadata in place; auto selects Triton.

    Tensor vectors have shape [B]. Integer tensors use int32 and flags use
    bool. Tables use int32; block_table is [B, S] and out_block_table [B, D].
    seq_lens and request_slots are input lengths and request IDs. last_prefix
    contains resident prefix history with a sentinel row. The output vectors
    safe_slots, lengths, sparse_rows, valid_rows, request_states, next_prefixes
    and cache_sizes receive the derived metadata. capacity is the cache size;
    request_rows excludes its sentinel. positions optionally supplies [B,
    query_width] visible positions. resident_safe_slots optionally receives
    resident IDs; request_to_resident maps request IDs to resident IDs and
    resident_rows specifies its row count, excluding the sentinel. impl selects
    auto, triton or torch_native. Returns None.
    """
    return get_impl_by_name("indexer.fused_prepare_lim_metadata", impl)(
        seq_lens=seq_lens,
        request_slots=request_slots,
        last_prefix=last_prefix,
        block_table=block_table,
        out_block_table=out_block_table,
        safe_slots=safe_slots,
        lengths=lengths,
        sparse_rows=sparse_rows,
        valid_rows=valid_rows,
        request_states=request_states,
        next_prefixes=next_prefixes,
        cache_sizes=cache_sizes,
        capacity=capacity,
        request_rows=request_rows,
        query_width=query_width,
        positions=positions,
        resident_safe_slots=resident_safe_slots,
        request_to_resident=request_to_resident,
        resident_rows=resident_rows,
    )
