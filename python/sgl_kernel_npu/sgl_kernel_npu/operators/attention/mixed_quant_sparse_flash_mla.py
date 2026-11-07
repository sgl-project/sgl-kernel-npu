"""Public mixed_quant_sparse_flash_mla operator backed by CANNBot-DSL on arch35."""

from __future__ import annotations

from ..utils.cannbot_dsl import ascend950_device, load_cannbot_dsl
from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator


@register_impl(
    "attention.mixed_quant_sparse_flash_mla",
    "auto",
    supported_generations={Generation.ARCH35},
)
@register_impl(
    "attention.mixed_quant_sparse_flash_mla",
    "cannbot_dsl",
    supported_generations={Generation.ARCH35},
)
def _cannbot_dsl_impl(*args, **kwargs):
    with ascend950_device(*args, **kwargs):
        operator = load_cannbot_dsl("mixed_quant_sparse_flash_mla")
        return operator(*args, **kwargs)


@register_operator("attention.mixed_quant_sparse_flash_mla")
def mixed_quant_sparse_flash_mla(
    q,
    *,
    ori_kv=None,
    cmp_kv=None,
    ori_sparse_indices=None,
    cmp_sparse_indices=None,
    ori_block_table=None,
    cmp_block_table=None,
    cu_seqlens_q=None,
    seqused_q=None,
    seqused_ori_kv=None,
    seqused_cmp_kv=None,
    ori_topk_length=None,
    cmp_topk_length=None,
    sinks=None,
    metadata=None,
    quant_mode,
    softmax_scale=None,
    layout_q="TND",
    layout_kv="PA_BBND",
    return_softmax_lse=False,
    out,
    lse,
    impl="auto",
):
    """Compute sparse attention with caller-owned ``out`` and ``lse`` buffers.

    ``auto`` and ``cannbot_dsl`` select the same lazy-loaded A5 implementation.
    """
    return get_impl_by_name("attention.mixed_quant_sparse_flash_mla", impl)(
        q,
        ori_kv=ori_kv,
        cmp_kv=cmp_kv,
        ori_sparse_indices=ori_sparse_indices,
        cmp_sparse_indices=cmp_sparse_indices,
        ori_block_table=ori_block_table,
        cmp_block_table=cmp_block_table,
        cu_seqlens_q=cu_seqlens_q,
        seqused_q=seqused_q,
        seqused_ori_kv=seqused_ori_kv,
        seqused_cmp_kv=seqused_cmp_kv,
        ori_topk_length=ori_topk_length,
        cmp_topk_length=cmp_topk_length,
        sinks=sinks,
        metadata=metadata,
        quant_mode=quant_mode,
        softmax_scale=softmax_scale,
        layout_q=layout_q,
        layout_kv=layout_kv,
        return_softmax_lse=return_softmax_lse,
        out=out,
        lse=lse,
    )
