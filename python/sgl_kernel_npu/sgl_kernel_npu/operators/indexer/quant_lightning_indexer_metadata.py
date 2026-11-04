"""Public quant_lightning_indexer_metadata operator backed by CANNBot-DSL on arch35."""

from __future__ import annotations

from ..utils.cannbot_dsl import ascend950_device, load_cannbot_dsl, require_ascend950
from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator


@register_impl(
    "indexer.quant_lightning_indexer_metadata",
    "auto",
    supported_generations={Generation.ARCH35},
)
@register_impl(
    "indexer.quant_lightning_indexer_metadata",
    "cannbot_dsl",
    supported_generations={Generation.ARCH35},
)
def _cannbot_dsl_impl(*args, **kwargs):
    with ascend950_device(*args, **kwargs):
        return load_cannbot_dsl("quant_lightning_indexer_metadata")(*args, **kwargs)


@register_operator("indexer.quant_lightning_indexer_metadata")
def quant_lightning_indexer_metadata(
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
    mask_mode=0,
    cmp_ratio=1,
    layout_q="TND",
    layout_k="TND",
    candidate_topk_blocks=-1,
    candidate_block_size=-1,
    impl="auto",
):
    """Invoke the upstream DeepSeek V4.1 operator with its original ABI.

    ``auto`` selects CANNBot-DSL on arch35. ``cannbot_dsl`` is explicit.
    The implementation is imported on first execution.
    """
    require_ascend950(cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, cmp_residual_k)
    return get_impl_by_name("indexer.quant_lightning_indexer_metadata", impl)(
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
        mask_mode=mask_mode,
        cmp_ratio=cmp_ratio,
        layout_q=layout_q,
        layout_k=layout_k,
        candidate_topk_blocks=candidate_topk_blocks,
        candidate_block_size=candidate_block_size,
    )
