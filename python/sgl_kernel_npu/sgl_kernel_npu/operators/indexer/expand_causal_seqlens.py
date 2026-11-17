"""Public packed-query causal length expansion operator."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_triton_fn = None
_torch_native_fn = None


@register_impl("indexer.expand_causal_seqlens", "auto")
@register_impl("indexer.expand_causal_seqlens", "triton")
def _triton_impl(
    cache_seqlens,
    query_lens,
    device,
    total_q,
    max_query_len,
    *,
    seqlens_out=None,
    token_to_batch_out=None,
):
    global _triton_fn
    if _triton_fn is None:
        from ...implementation.triton.indexer.expand_causal_seqlens import (
            expand_causal_seqlens,
        )

        _triton_fn = expand_causal_seqlens
    return _triton_fn(
        cache_seqlens,
        query_lens,
        device,
        total_q,
        max_query_len,
        seqlens_out=seqlens_out,
        token_to_batch_out=token_to_batch_out,
    )


@register_impl("indexer.expand_causal_seqlens", "torch_native")
def _torch_native_impl(
    cache_seqlens,
    query_lens,
    device,
    total_q,
    max_query_len,
    *,
    seqlens_out=None,
    token_to_batch_out=None,
):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.indexer.expand_causal_seqlens import (
            expand_causal_seqlens,
        )

        _torch_native_fn = expand_causal_seqlens
    return _torch_native_fn(
        cache_seqlens,
        query_lens,
        device,
        total_q,
        max_query_len,
        seqlens_out=seqlens_out,
        token_to_batch_out=token_to_batch_out,
    )


@register_operator("indexer.expand_causal_seqlens")
def expand_causal_seqlens(
    cache_seqlens,
    query_lens,
    device,
    total_q,
    max_query_len,
    *,
    seqlens_out=None,
    token_to_batch_out=None,
    impl="auto",
):
    """Expand causal lengths and token-to-request IDs for packed queries.

    ``auto`` selects Triton; ``torch_native`` selects independent PyTorch.

    Args:
        cache_seqlens: Contiguous int32 Tensor [B], total KV lengths including queries.
        query_lens: Contiguous int32 Tensor [B], non-negative query counts.
        device: torch.device shared by inputs and outputs.
        total_q: int, sum of query_lens, supplied by the caller without a device sync.
        max_query_len: int, non-negative upper bound on query_lens; positive if total_q > 0.
        seqlens_out: Optional contiguous int32 Tensor [total_q], overwritten in place.
        token_to_batch_out: Optional contiguous int64 Tensor [total_q], overwritten in place.
        impl: Backend name: auto, triton or torch_native.

    Returns:
        int32 causal lengths and int64 request IDs, both [total_q]. Request b
        emits max(cache_seqlens[b] - query_lens[b] + 1, 0) + arange(query_lens[b]).
        Empty queries emit no tokens. Supplied output buffers are returned by
        identity; otherwise outputs are allocated. Inputs remain read-only.
        Outputs must not overlap the inputs or each other.
    """
    impl_fn = get_impl_by_name("indexer.expand_causal_seqlens", impl)
    return impl_fn(
        cache_seqlens,
        query_lens,
        device,
        total_q,
        max_query_len,
        seqlens_out=seqlens_out,
        token_to_batch_out=token_to_batch_out,
    )
