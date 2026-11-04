"""CANN 9.2.0 ops-transformer interface for arch22 (A3)."""

from functools import lru_cache

from ..utils.cann import detect_cann_version
from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator


@lru_cache(maxsize=1)
def _binding():
    from ..._third_party.sgl_ops_transformer.ops import lightning_indexer_metadata

    return lightning_indexer_metadata


@register_impl(
    "attention.lightning_indexer_v2_metadata",
    "auto",
    supported_generations={Generation.ARCH22},
)
@register_impl(
    "attention.lightning_indexer_v2_metadata",
    "cann_ops_transformers_cann920",
    supported_generations={Generation.ARCH22},
)
def _cann_impl(*args, **kwargs):
    if detect_cann_version() != "9.2.0":
        raise RuntimeError("lightning_indexer_v2_metadata requires CANN 9.2.0")
    return _binding()(*args, **kwargs)


@register_operator("attention.lightning_indexer_v2_metadata")
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
    """See docs/api/{en,zh}/operators/attention/lightning_indexer_v2_metadata.md for the contract."""
    return get_impl_by_name("attention.lightning_indexer_v2_metadata", impl)(
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
