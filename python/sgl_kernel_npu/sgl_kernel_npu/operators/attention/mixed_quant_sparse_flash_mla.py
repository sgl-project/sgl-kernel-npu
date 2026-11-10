"""Public mixed_quant_sparse_flash_mla operator backed by CANNBot-DSL on arch35."""

from __future__ import annotations

from importlib import import_module

import torch

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator

_torch_native_fn = None
_cannbot_dsl_fn = None


@register_impl(
    "attention.mixed_quant_sparse_flash_mla",
    "torch_native",
    supported_generations={Generation.ARCH35},
)
def _torch_native_impl(*args, **kwargs):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.attention.mixed_quant_sparse_flash_mla import (
            mixed_quant_sparse_flash_mla,
        )

        _torch_native_fn = mixed_quant_sparse_flash_mla
    return _torch_native_fn(*args, **kwargs)


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
def _cannbot_dsl_impl(q, *args, **kwargs):
    global _cannbot_dsl_fn
    with torch.npu.device(q.device):
        if _cannbot_dsl_fn is None:
            _cannbot_dsl_fn = import_module(
                "sgl_kernel_npu.implementation.cannbot-dsl.attention.mixed_quant_sparse_flash_mla"
            ).mixed_quant_sparse_flash_mla
        return _cannbot_dsl_fn(q, *args, **kwargs)


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

    ``auto`` selects the lazy A5 CANNBot-DSL backend.

    Args:
        q: BF16 [T,64,512], attention queries.
        ori_kv: UINT8 [P,S,1,544], FP8 KV bytes plus BF16 scales; required by cannbot_dsl.
        cmp_kv: Optional UINT8 [P,S,1,320], packed FP4 KV bytes plus BF16 group scales.
        ori_sparse_indices: INT32 [T,1,Kori], required original logical token IDs; -1 is padding.
        cmp_sparse_indices: Optional INT32 [T,1,Kcmp], compressed logical token IDs; -1 is padding.
        ori_block_table: Required INT32 [B,max_ori_blocks], original KV page map.
        cmp_block_table: Optional INT32 [B,max_cmp_blocks], compressed KV page map.
        cu_seqlens_q: INT32 [B+1], required by cannbot_dsl; cumulative boundaries ending at T.
        seqused_q: Optional INT32 [B]; when supplied, must equal cu_seqlens_q spans (no padded rows).
        seqused_ori_kv: Optional INT32 [B], original KV lengths; retained for API compatibility.
        seqused_cmp_kv: Optional INT32 [B], compressed KV lengths; retained for API compatibility.
        ori_topk_length: Optional INT32 [T,1], valid sparse prefix; None uses all original index columns.
        cmp_topk_length: Optional INT32 [T,1], valid sparse prefix; None uses all compressed index columns. Omit when cmp_kv is absent.
        sinks: FP32 [64], attention-sink logits (zero value contribution).
        metadata: INT32 [1024] device schedule required by cannbot_dsl; torch_native ignores it.
        quant_mode: int, use 1 for the supported quantized profile.
        softmax_scale: Optional float, defaults to 1/sqrt(512).
        layout_q: str, use TND (flattened query tokens).
        layout_kv: str, use PA_BBND packed paged KV.
        return_softmax_lse: bool, write lse when True.
        out: Writable BF16 [T,64,512] output tensor.
        lse: Writable FP32 [1,T,64] when return_softmax_lse=True; otherwise pass an empty FP32 [0] tensor.
        impl: str, auto selects cannbot_dsl; torch_native is an independent compute reference.
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
