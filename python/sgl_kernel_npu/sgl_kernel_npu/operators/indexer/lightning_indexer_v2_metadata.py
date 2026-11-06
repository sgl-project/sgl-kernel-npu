"""CANN 9.2.0 ops-transformer interface for arch22 (A3)."""

from ..utils.cann import detect_cann_version
from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator

_cann_fn = None


def _binding():
    global _cann_fn
    if _cann_fn is None:
        from ..._third_party.sgl_ops_transformer.ops import lightning_indexer_metadata

        _cann_fn = lightning_indexer_metadata
    return _cann_fn


@register_impl(
    "indexer.lightning_indexer_v2_metadata",
    "auto",
    supported_generations={Generation.ARCH22},
)
@register_impl(
    "indexer.lightning_indexer_v2_metadata",
    "cann_ops_transformers_cann920",
    supported_generations={Generation.ARCH22},
)
def _cann_impl(*args, **kwargs):
    if detect_cann_version() != "9.2.0":
        raise RuntimeError("lightning_indexer_v2_metadata requires CANN 9.2.0")
    return _binding()(*args, **kwargs)


@register_operator("indexer.lightning_indexer_v2_metadata")
def lightning_indexer_v2_metadata(
    num_heads_q,
    num_heads_k,
    head_dim,
    topk,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_q=None,
    seqused_k=None,
    cmp_residual_k=None,
    batch_size=None,
    max_seqlen_q=None,
    max_seqlen_k=None,
    layout_q=None,
    layout_k=None,
    mask_mode=None,
    cmp_ratio=None,
    *,
    impl="auto"
):
    """Build opaque int32[2048] scheduling data; A3 kernels ignore this data.

    The default auto implementation uses bundled CANN 9.2.0 on A3.

    Args:
        num_heads_q (int): Number of query heads, in 1..64.
        num_heads_k (int): Number of key heads; must be 1.
        head_dim (int): Query/key head dimension; must be 128.
        topk (int): Selected key count, matching the main operator.
        cu_seqlens_q (Tensor | None): Int32 [B+1] query length metadata.
        cu_seqlens_k (Tensor | None): Int32 [B+1] cumulative TND key lengths.
        seqused_q (Tensor | None): Int32 [B] valid query lengths.
        seqused_k (Tensor | None): Int32 [B] valid key lengths.
        cmp_residual_k (Tensor | None): Int32 [B] compression remainders.
        batch_size (int | None): Batch count; None uses upstream default 0.
        max_seqlen_q (int | None): Maximum query length; None defaults to -1.
        max_seqlen_k (int | None): Maximum key length; None defaults to -1.
        layout_q (str | None): Query layout; None defaults to BSND.
        layout_k (str | None): Key layout; None defaults to BSND.
        mask_mode (int | None): Mask mode; None defaults to 0.
        cmp_ratio (int | None): Compression ratio; None defaults to 1.
        impl (str): auto or cann_ops_transformers_cann920.

    Tensor arguments use the input device of the corresponding main call.
    """
    return get_impl_by_name("indexer.lightning_indexer_v2_metadata", impl)(
        num_heads_q=num_heads_q,
        num_heads_k=num_heads_k,
        head_dim=head_dim,
        topk=topk,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        cmp_residual_k=cmp_residual_k,
        batch_size=batch_size,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        layout_q=layout_q,
        layout_k=layout_k,
        mask_mode=mask_mode,
        cmp_ratio=cmp_ratio,
    )
