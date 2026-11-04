"""Public mixed_quant_sparse_flash_mla_metadata operator backed by CANNBot-DSL on arch35."""

from __future__ import annotations

from ..utils.cannbot_dsl import ascend950_device, load_cannbot_dsl, require_ascend950
from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator


@register_impl(
    "attention.mixed_quant_sparse_flash_mla_metadata",
    "auto",
    supported_generations={Generation.ARCH35},
)
@register_impl(
    "attention.mixed_quant_sparse_flash_mla_metadata",
    "cannbot_dsl",
    supported_generations={Generation.ARCH35},
)
def _cannbot_dsl_impl(*args, **kwargs):
    with ascend950_device(*args, **kwargs):
        return load_cannbot_dsl("mixed_quant_sparse_flash_mla_metadata")(
            *args, **kwargs
        )


@register_operator("attention.mixed_quant_sparse_flash_mla_metadata")
def mixed_quant_sparse_flash_mla_metadata(
    ori_topk_length,
    cmp_topk_length,
    *,
    cu_seqlens_q=None,
    seqused_q=None,
    seqused_ori_kv=None,
    seqused_cmp_kv=None,
    batch_size=None,
    max_seqlen_q=None,
    max_seqlen_ori_kv=None,
    max_seqlen_cmp_kv=None,
    num_heads_q,
    num_heads_kv,
    head_dim,
    quant_mode,
    layout_q="TND",
    layout_kv="PA_BBND",
    has_ori_kv=True,
    has_cmp_kv=True,
    impl="auto",
):
    """Invoke the upstream DeepSeek V4.1 operator with its original ABI.

    ``auto`` selects CANNBot-DSL on arch35. ``cannbot_dsl`` is explicit.
    The implementation is imported on first execution.
    """
    require_ascend950(ori_topk_length, cmp_topk_length)
    return get_impl_by_name("attention.mixed_quant_sparse_flash_mla_metadata", impl)(
        ori_topk_length,
        cmp_topk_length,
        cu_seqlens_q=cu_seqlens_q,
        seqused_q=seqused_q,
        seqused_ori_kv=seqused_ori_kv,
        seqused_cmp_kv=seqused_cmp_kv,
        batch_size=batch_size,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_ori_kv=max_seqlen_ori_kv,
        max_seqlen_cmp_kv=max_seqlen_cmp_kv,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        head_dim=head_dim,
        quant_mode=quant_mode,
        layout_q=layout_q,
        layout_kv=layout_kv,
        has_ori_kv=has_ori_kv,
        has_cmp_kv=has_cmp_kv,
    )
