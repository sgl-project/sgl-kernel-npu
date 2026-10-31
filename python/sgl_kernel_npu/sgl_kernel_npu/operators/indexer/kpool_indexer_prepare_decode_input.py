"""Public fused decode-indexer input preparation operator."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_triton_fn = None
_torch_native_fn = None


@register_impl("indexer.kpool_indexer_prepare_decode_input", "auto")
@register_impl("indexer.kpool_indexer_prepare_decode_input", "triton")
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
        from ...implementation.triton.indexer.kpool_indexer_prepare_decode_input import (
            kpool_indexer_prepare_decode_input,
        )

        _triton_fn = kpool_indexer_prepare_decode_input

    return _triton_fn(
        x,
        weights_proj_weight,
        seqlens_32,
        block_tables,
        pool_size,
        n_heads_pow,
        softmax_scale,
    )


@register_impl("indexer.kpool_indexer_prepare_decode_input", "torch_native")
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
        from ...implementation.torch_native.indexer.kpool_indexer_prepare_decode_input import (
            kpool_indexer_prepare_decode_input,
        )

        _torch_native_fn = kpool_indexer_prepare_decode_input

    return _torch_native_fn(
        x,
        weights_proj_weight,
        seqlens_32,
        block_tables,
        pool_size,
        n_heads_pow,
        softmax_scale,
    )


@register_operator("indexer.kpool_indexer_prepare_decode_input")
def kpool_indexer_prepare_decode_input(
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

    ``auto`` selects Triton; ``torch_native`` selects independent PyTorch.
    Triton accepts power-of-two head counts up to 128 and masks K tails.

    Args:
        x: torch.Tensor, shape [N, HIDDEN_SIZE], dtype torch.bfloat16. Decode hidden states.
        weights_proj_weight: torch.Tensor, shape [N_HEADS, HIDDEN_SIZE], dtype torch.float32. ReplicatedLinear projection weight.
        seqlens_32: torch.Tensor, shape [N], dtype torch.int32. Real KV length for each request.
        block_tables: torch.Tensor, shape [N, NUM_BLOCKS], dtype torch.int32. Contiguous token-page table.
        pool_size: int, shape —, dtype —. Pooling width; the current kernel requires exactly 4.
        n_heads_pow: float, shape —, dtype —. First factor in the combined projection scale.
        softmax_scale: float, shape —, dtype —. Second factor in the combined projection scale.
        impl: Backend name: auto, triton or torch_native.

    Returns:
        BF16 weights [B, N], int32 pooled lengths [B], and int32 pooled
        page table [B, ceil(M / pool_size)]. Inputs remain read-only.
    """
    impl_fn = get_impl_by_name("indexer.kpool_indexer_prepare_decode_input", impl)
    return impl_fn(
        x,
        weights_proj_weight,
        seqlens_32,
        block_tables,
        pool_size,
        n_heads_pow,
        softmax_scale,
    )
