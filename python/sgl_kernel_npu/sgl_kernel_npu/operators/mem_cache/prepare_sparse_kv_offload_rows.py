"""Sparse KV offload row metadata dispatch."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_triton_fn = None
_torch_native_fn = None


@register_impl("mem_cache.prepare_sparse_kv_offload_rows", "auto")
@register_impl("mem_cache.prepare_sparse_kv_offload_rows", "triton")
def _triton_impl(*args, **kwargs):
    global _triton_fn
    if _triton_fn is None:
        from ...implementation.triton.mem_cache.prepare_sparse_kv_offload_rows import (
            prepare_sparse_kv_offload_rows,
        )

        _triton_fn = prepare_sparse_kv_offload_rows
    return _triton_fn(*args, **kwargs)


@register_impl("mem_cache.prepare_sparse_kv_offload_rows", "torch_native")
def _torch_native_impl(*args, **kwargs):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.mem_cache.prepare_sparse_kv_offload_rows import (
            prepare_sparse_kv_offload_rows,
        )

        _torch_native_fn = prepare_sparse_kv_offload_rows
    return _torch_native_fn(*args, **kwargs)


@register_operator("mem_cache.prepare_sparse_kv_offload_rows")
def prepare_sparse_kv_offload_rows(
    tokens,
    device,
    request_ids,
    seq_lens,
    positions,
    out_cache_loc,
    *,
    request_rows,
    context_len,
    host_capacity,
    query_width=1,
    extend_seq_lens=None,
    extend_prefix_lens=None,
    compact_prefill=False,
    impl="auto",
):
    """Prepare sparse KV offload rows; auto selects Triton.

    Inputs are one-dimensional integer tensors: request_ids and seq_lens have
    batch length; positions and out_cache_loc have token length. Extend lengths
    and prefixes describe prefill rows. tokens and device specify output size
    and device. request_rows, context_len and host_capacity bound validity;
    query_width is the decode width and compact_prefill selects compact layout.
    impl selects auto, triton or torch_native. Returns source, destination,
    token positions (int64) and validity (bool), each of shape [tokens].
    """
    fn = get_impl_by_name("mem_cache.prepare_sparse_kv_offload_rows", impl)
    return fn(
        tokens,
        device,
        request_ids,
        seq_lens,
        positions,
        out_cache_loc,
        request_rows=request_rows,
        context_len=context_len,
        host_capacity=host_capacity,
        query_width=query_width,
        extend_seq_lens=extend_seq_lens,
        extend_prefix_lens=extend_prefix_lens,
        compact_prefill=compact_prefill,
    )
