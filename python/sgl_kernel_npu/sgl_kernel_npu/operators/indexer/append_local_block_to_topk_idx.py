"""Public operator that appends the query's block to the selected topk blocks."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator

OP = "indexer.append_local_block_to_topk_idx"

_triton_fn = None
_torch_native_fn = None


# Importing this module registers the three names without importing Triton.
@register_impl(OP, "auto")
@register_impl(OP, "triton")
def _triton_impl(topk_idx, seq_lens, block_size, num_blocks):
    global _triton_fn
    if _triton_fn is None:
        from ...implementation.triton.indexer.append_local_block_to_topk_idx import (
            append_local_block_to_topk_idx,
        )

        _triton_fn = append_local_block_to_topk_idx

    return _triton_fn(topk_idx, seq_lens, block_size, num_blocks)


@register_impl(OP, "torch_native")
def _torch_native_impl(topk_idx, seq_lens, block_size, num_blocks):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.indexer.append_local_block_to_topk_idx import (
            append_local_block_to_topk_idx,
        )

        _torch_native_fn = append_local_block_to_topk_idx

    return _torch_native_fn(topk_idx, seq_lens, block_size, num_blocks)


@register_operator(OP)
def append_local_block_to_topk_idx(
    topk_idx,
    seq_lens,
    block_size,
    num_blocks,
    *,
    impl="auto",
):
    """Append the block each query belongs to after its selected topk blocks.

    The default ``impl="auto"`` resolves to the Triton entry, and ``"triton"``
    is the same function; ``"torch_native"`` is an independent reference used
    for accuracy comparison. ``topk_idx`` is contiguous ``int32``
    ``[N, B, topk]`` with ``topk > 0``, and ``seq_lens`` is contiguous
    ``int32``/``int64`` ``[B]``.
    Returns a new ``int32`` ``[N, B, topk + 1]`` tensor; inputs are not modified
    and there is no fallback.
    """
    return get_impl_by_name(OP, impl)(topk_idx, seq_lens, block_size, num_blocks)
