"""Public A5 quantized shared-KV attention APIs."""

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator

_ATTN_OP = "attention.kv_quant_sparse_attn_sharedkv"
_META_OP = "attention.kv_quant_sparse_attn_sharedkv_metadata_host"
_reference = None


@register_impl(_ATTN_OP, "auto", supported_generations={Generation.ARCH35})
@register_impl(_ATTN_OP, "ascendc", supported_generations={Generation.ARCH35})
def _ascendc_attention(q, **kwargs):
    import torch

    return torch.ops.npu.kv_quant_sparse_attn_sharedkv(q, **kwargs)


@register_impl(
    _ATTN_OP, "torch_native", supported_generations={Generation.ARCH35}
)
def _torch_native_attention(q, **kwargs):
    global _reference
    if _reference is None:
        from ...implementation.torch_native.attention import (
            kv_quant_sparse_attn_sharedkv as reference_module,
        )

        _reference = reference_module.kv_quant_sparse_attn_sharedkv
    return _reference(q, **kwargs)


@register_operator(_ATTN_OP)
def kv_quant_sparse_attn_sharedkv(
    q,
    *,
    ori_kv=None,
    cmp_kv=None,
    ori_sparse_indices=None,
    cmp_sparse_indices=None,
    ori_block_table=None,
    cmp_block_table=None,
    cu_seqlens_q=None,
    cu_seqlens_ori_kv=None,
    cu_seqlens_cmp_kv=None,
    seqused_q=None,
    seqused_kv=None,
    sinks=None,
    metadata=None,
    kv_quant_mode=1,
    tile_size=64,
    rope_head_dim=64,
    softmax_scale=1.0,
    cmp_ratio=1,
    ori_mask_mode=4,
    cmp_mask_mode=3,
    ori_win_left=127,
    ori_win_right=0,
    layout_q="BSND",
    layout_kv="PA_ND",
    return_softmax_lse=False,
    impl="auto",
):
    """Compute A5 quantized shared-KV attention; return (output, LSE).

    auto selects the existing Ascend C binding. torch_native is an independent
    reference for uncompressed BSND paged SWA accuracy tests.
    """
    kwargs = dict(
        ori_kv=ori_kv,
        cmp_kv=cmp_kv,
        ori_sparse_indices=ori_sparse_indices,
        cmp_sparse_indices=cmp_sparse_indices,
        ori_block_table=ori_block_table,
        cmp_block_table=cmp_block_table,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_ori_kv=cu_seqlens_ori_kv,
        cu_seqlens_cmp_kv=cu_seqlens_cmp_kv,
        seqused_q=seqused_q,
        seqused_kv=seqused_kv,
        sinks=sinks,
        metadata=metadata,
        kv_quant_mode=kv_quant_mode,
        tile_size=tile_size,
        rope_head_dim=rope_head_dim,
        softmax_scale=softmax_scale,
        cmp_ratio=cmp_ratio,
        ori_mask_mode=ori_mask_mode,
        cmp_mask_mode=cmp_mask_mode,
        ori_win_left=ori_win_left,
        ori_win_right=ori_win_right,
        layout_q=layout_q,
        layout_kv=layout_kv,
        return_softmax_lse=return_softmax_lse,
    )
    return get_impl_by_name(_ATTN_OP, impl)(q, **kwargs)


@register_impl(_META_OP, "auto", supported_generations={Generation.ARCH35})
@register_impl(_META_OP, "host", supported_generations={Generation.ARCH35})
def _host_metadata(**kwargs):
    import torch

    return torch.ops.npu.kv_quant_sparse_attn_sharedkv_metadata_host(**kwargs)


@register_operator(_META_OP)
def kv_quant_sparse_attn_sharedkv_metadata_host(
    num_heads_q,
    num_heads_kv,
    head_dim,
    layout_q,
    layout_kv,
    *,
    cu_seqlens_q=None,
    seqused_kv=None,
    batch_size=0,
    max_seqlen_q=0,
    max_seqlen_kv=0,
    cmp_topk=0,
    cmp_ratio=1,
    ori_mask_mode=4,
    cmp_mask_mode=3,
    ori_win_left=127,
    ori_win_right=0,
    has_ori_kv=True,
    has_cmp_kv=True,
    kv_quant_mode=1,
    tile_size=64,
    rope_head_dim=64,
    impl="auto",
):
    """Build A5 metadata from CPU int32 lengths via the existing host binding."""
    kwargs = dict(
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        head_dim=head_dim,
        layout_q=layout_q,
        layout_kv=layout_kv,
        cu_seqlens_q=cu_seqlens_q,
        seqused_kv=seqused_kv,
        batch_size=batch_size,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_kv=max_seqlen_kv,
        cmp_topk=cmp_topk,
        cmp_ratio=cmp_ratio,
        ori_mask_mode=ori_mask_mode,
        cmp_mask_mode=cmp_mask_mode,
        ori_win_left=ori_win_left,
        ori_win_right=ori_win_right,
        has_ori_kv=has_ori_kv,
        has_cmp_kv=has_cmp_kv,
        kv_quant_mode=kv_quant_mode,
        tile_size=tile_size,
        rope_head_dim=rope_head_dim,
    )
    return get_impl_by_name(_META_OP, impl)(**kwargs)
