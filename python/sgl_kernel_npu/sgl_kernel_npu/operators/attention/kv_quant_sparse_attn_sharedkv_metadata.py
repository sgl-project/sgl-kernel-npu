"""Host metadata API for A5 quantized shared-KV attention."""

import torch

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator

_metadata_reference = None


@register_impl(
    "attention.kv_quant_sparse_attn_sharedkv_metadata_host",
    "auto",
    supported_generations={Generation.ARCH35},
)
@register_impl(
    "attention.kv_quant_sparse_attn_sharedkv_metadata_host",
    "host",
    supported_generations={Generation.ARCH35},
)
def _host_metadata(**kwargs):
    return torch.ops.npu.kv_quant_sparse_attn_sharedkv_metadata_host(**kwargs)


@register_impl(
    "attention.kv_quant_sparse_attn_sharedkv_metadata_host",
    "torch_native",
    supported_generations={Generation.ARCH35},
)
def _torch_native_metadata(**kwargs):
    global _metadata_reference
    if _metadata_reference is None:
        from ...implementation.torch_native.attention import (
            kv_quant_sparse_attn_sharedkv_metadata as reference_module,
        )

        _metadata_reference = reference_module.kv_quant_sparse_attn_sharedkv_metadata
    return _metadata_reference(**kwargs)


@register_operator("attention.kv_quant_sparse_attn_sharedkv_metadata_host")
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
    """Build A5 metadata from CPU int32 lengths.

    torch_native independently constructs the single-request decode schedule;
    other schedules require impl="host" or impl="auto".
    """
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
    impl_fn = get_impl_by_name(
        "attention.kv_quant_sparse_attn_sharedkv_metadata_host", impl
    )
    return impl_fn(**kwargs)
