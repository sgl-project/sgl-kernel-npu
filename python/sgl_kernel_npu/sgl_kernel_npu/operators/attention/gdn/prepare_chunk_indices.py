"""Public chunk index preparation for packed GDN sequences."""

from ...utils.registry import get_impl_by_name, register_impl, register_operator

_torch_native_fn = None


@register_impl("attention.gdn.prepare_chunk_indices", "auto")
@register_impl("attention.gdn.prepare_chunk_indices", "torch_native")
def _torch_native_impl(cu_seqlens, chunk_size):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ....implementation.torch_native.attention.gdn.prepare_chunk_indices import (
            prepare_chunk_indices,
        )

        _torch_native_fn = prepare_chunk_indices
    return _torch_native_fn(cu_seqlens, chunk_size)


@register_operator("attention.gdn.prepare_chunk_indices")
def prepare_chunk_indices(cu_seqlens, chunk_size, *, impl="auto"):
    """Return packed chunk coordinates for nonempty sequences.

    Args:
        cu_seqlens: Nondecreasing int32/int64 cumulative boundaries, shape [B+1],
            starting at zero, with B >= 1. Reused tensor objects must stay unchanged.
        chunk_size: Positive integer giving the maximum number of tokens per chunk.
        impl: "auto" or "torch_native". Auto selects torch_native.

    Returns:
        Tensor [total_chunks, 2] on the input device and with its dtype.
        The columns are the sequence index among nonempty sequences and the
        zero-based index within that sequence.
    """
    return get_impl_by_name("attention.gdn.prepare_chunk_indices", impl)(
        cu_seqlens, chunk_size
    )
