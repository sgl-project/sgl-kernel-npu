"""Public mixed_quant_sparse_flash_mla_metadata operator backed by CANNBot-DSL on arch35."""

from __future__ import annotations

from importlib import import_module

import torch

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator

_cannbot_dsl_fn = None


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
    global _cannbot_dsl_fn
    # Positional arguments are optional sequence tensors.
    device = next(
        (value.device for value in args if value is not None),
        torch.npu.current_device(),
    )
    with torch.npu.device(device):
        if _cannbot_dsl_fn is None:
            _cannbot_dsl_fn = import_module(
                "sgl_kernel_npu.implementation.cannbot-dsl.attention.mixed_quant_sparse_flash_mla_metadata"
            ).mixed_quant_sparse_flash_mla_metadata
        return _cannbot_dsl_fn(*args, **kwargs)


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
    """Build scheduling metadata for ``mixed_quant_sparse_flash_mla``.

    ``auto`` selects the lazy A5 CANNBot-DSL backend.

    Args:
        ori_topk_length: Required contiguous INT32 [T,1], valid original sparse prefix per query.
        cmp_topk_length: Required contiguous INT32 [T,1]; pass zeros when has_cmp_kv=False.
        cu_seqlens_q: Optional INT32 [B+1], cumulative query row boundaries.
        seqused_q: Optional INT32 [B], compatibility argument; scheduling uses cu_seqlens_q.
        seqused_ori_kv: Optional INT32 [B], original KV lengths; retained for API compatibility.
        seqused_cmp_kv: Optional INT32 [B], compressed KV lengths; retained for API compatibility.
        batch_size: Optional int, compatibility argument; B is derived from cu_seqlens_q (one batch if absent).
        max_seqlen_q: Optional int, compatibility argument; query rows come from the length tensor shape.
        max_seqlen_ori_kv: Optional int, original KV capacity (compatibility argument).
        max_seqlen_cmp_kv: Optional int, compressed KV capacity (compatibility argument).
        num_heads_q: int, query heads; use 64.
        num_heads_kv: int, shared MLA KV heads; use 1.
        head_dim: int, logical head dimension; use 512.
        quant_mode: int, use 1 for the supported quantized profile.
        layout_q: str, use TND (flattened query tokens).
        layout_kv: str, use PA_BBND packed paged KV.
        has_ori_kv: bool, must be True for this implementation.
        has_cmp_kv: bool, whether the consuming attention call has compressed KV.
        impl: str, auto and cannbot_dsl select the same AICPU schedule implementation.
    """
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
