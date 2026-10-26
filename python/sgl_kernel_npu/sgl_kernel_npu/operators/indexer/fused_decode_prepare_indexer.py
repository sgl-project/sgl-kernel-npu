"""Public fused decode-indexer input preparation operator."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_triton_fn = None
_torch_native_fn = None


@register_impl("indexer.fused_decode_prepare_indexer", "auto")
@register_impl("indexer.fused_decode_prepare_indexer", "triton")
def _triton_impl(
    x,
    weights_proj_weight,
    seqlens_32,
    block_tables,
    pool_size=4,
    n_heads_pow=32**-0.5,
    softmax_scale=128**-0.5,
):
    global _triton_fn
    if _triton_fn is None:
        from ...implementation.triton.indexer.kpool_decode_prepare_indexer import (
            fused_decode_prepare_indexer,
        )

        _triton_fn = fused_decode_prepare_indexer

    return _triton_fn(
        x,
        weights_proj_weight,
        seqlens_32,
        block_tables,
        pool_size,
        n_heads_pow,
        softmax_scale,
    )


@register_impl("indexer.fused_decode_prepare_indexer", "torch_native")
def _torch_native_impl(
    x,
    weights_proj_weight,
    seqlens_32,
    block_tables,
    pool_size=4,
    n_heads_pow=32**-0.5,
    softmax_scale=128**-0.5,
):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.indexer.fused_decode_prepare_indexer import (
            fused_decode_prepare_indexer,
        )

        _torch_native_fn = fused_decode_prepare_indexer

    return _torch_native_fn(
        x,
        weights_proj_weight,
        seqlens_32,
        block_tables,
        pool_size,
        n_heads_pow,
        softmax_scale,
    )


@register_operator("indexer.fused_decode_prepare_indexer")
def fused_decode_prepare_indexer(
    x,
    weights_proj_weight,
    seqlens_32,
    block_tables,
    pool_size=4,
    n_heads_pow=32**-0.5,
    softmax_scale=128**-0.5,
    *,
    impl="auto",
):
    """Prepare decode weights, pooled lengths and pooled page tables.

    Returns ``(w_bf16, pool_seqlens, pool_block_tables)``. ``impl`` accepts
    ``auto``, ``triton`` or ``torch_native``; ``auto`` selects Triton.
    """
    impl_fn = get_impl_by_name("indexer.fused_decode_prepare_indexer", impl)
    return impl_fn(
        x,
        weights_proj_weight,
        seqlens_32,
        block_tables,
        pool_size,
        n_heads_pow,
        softmax_scale,
    )
