"""Public quant_sparse_lightning_indexer_metadata operator backed by CANNBot-DSL on arch35."""

from __future__ import annotations

from ..utils.cannbot_dsl import ascend950_device, load_cannbot_dsl
from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator


@register_impl(
    "indexer.quant_sparse_lightning_indexer_metadata",
    "auto",
    supported_generations={Generation.ARCH35},
)
@register_impl(
    "indexer.quant_sparse_lightning_indexer_metadata",
    "cannbot_dsl",
    supported_generations={Generation.ARCH35},
)
def _cannbot_dsl_impl(*args, **kwargs):
    with ascend950_device(*args, **kwargs):
        return load_cannbot_dsl("quant_sparse_lightning_indexer_metadata")(
            *args, **kwargs
        )


@register_operator("indexer.quant_sparse_lightning_indexer_metadata")
def quant_sparse_lightning_indexer_metadata(
    candidate_block_length,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_q=None,
    seqused_k=None,
    cmp_residual_k=None,
    *,
    batch_size=None,
    max_seqlen_q=-1,
    max_seqlen_k=-1,
    num_heads_q,
    num_heads_k,
    head_dim,
    topk,
    quant_mode,
    candidate_block_size,
    mask_mode=0,
    cmp_ratio=1,
    layout_q="TND",
    layout_k="TND",
    impl="auto",
):
    """Invoke the upstream DeepSeek V4.1 operator with its original ABI.

    ``auto`` selects CANNBot-DSL on arch35. ``cannbot_dsl`` is explicit.
    The implementation is imported on first execution.
    """
    return get_impl_by_name("indexer.quant_sparse_lightning_indexer_metadata", impl)(
        candidate_block_length,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        cmp_residual_k,
        batch_size=batch_size,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        num_heads_q=num_heads_q,
        num_heads_k=num_heads_k,
        head_dim=head_dim,
        topk=topk,
        quant_mode=quant_mode,
        candidate_block_size=candidate_block_size,
        mask_mode=mask_mode,
        cmp_ratio=cmp_ratio,
        layout_q=layout_q,
        layout_k=layout_k,
    )
