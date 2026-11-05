"""CANN quantized flash attention and an independent MXFP8 reference."""

from typing import Optional, Tuple

import torch

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator

_cann_fn = None
_torch_native_fn = None


@register_impl(
    "attention.quant_flash_attn", "auto", supported_generations={Generation.ARCH35}
)
@register_impl(
    "attention.quant_flash_attn",
    "cann_ops_transformer",
    supported_generations={Generation.ARCH35},
)
def _cann_impl(*args, **kwargs):
    global _cann_fn
    if _cann_fn is None:
        from cann_ops_transformer.ops import quant_flash_attn

        _cann_fn = quant_flash_attn
    return _cann_fn(*args, **kwargs)


@register_impl(
    "attention.quant_flash_attn",
    "torch_native",
    supported_generations={Generation.ARCH35},
)
def _torch_native_impl(*args, **kwargs):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.attention.quant_flash_attn import (
            quant_flash_attn,
        )

        _torch_native_fn = quant_flash_attn
    return _torch_native_fn(*args, **kwargs)


@register_operator("attention.quant_flash_attn")
def quant_flash_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_descale: torch.Tensor,
    k_descale: torch.Tensor,
    v_descale: torch.Tensor,
    quant_mode: int,
    *,
    block_table: Optional[torch.Tensor] = None,
    p_scale: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_kv: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_kv: Optional[torch.Tensor] = None,
    sinks: Optional[torch.Tensor] = None,
    attn_mask: Optional[torch.Tensor] = None,
    metadata: Optional[torch.Tensor] = None,
    softmax_scale: float = 1.0,
    mask_mode: int = 0,
    win_left: int = -1,
    win_right: int = -1,
    max_seqlen_q: int = -1,
    max_seqlen_kv: int = -1,
    layout_q: str = "BSND",
    layout_q_descale: str = "BSND",
    layout_kv: str = "BSND",
    layout_out: str = "BSND",
    return_softmax_lse: bool = False,
    impl: str = "auto",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute quantized attention; auto selects cann_ops_transformer on arch35.

    CANN arguments and return values are forwarded unchanged. The independent
    torch_native reference covers quant_mode=1, TND Q/output, PA_BNBD KV,
    TND/N2TGD Q descales and mask modes 0/3. It is an eager accuracy reference.

    Args:
        q: Quantized queries; MXFP8 uses FP8 E4M3 [T, Nq, D].
        k, v: Quantized keys/values; PA_BNBD uses [P, Nkv, S, D].
        q_descale: E8M0 [T, Nq, ceil(D/64), 2] for TND, or
            [Nkv, T, Nq/Nkv, ceil(D/64), 2] for N2TGD.
        k_descale: E8M0 [P, Nkv, S, ceil(D/64), 2] for PA_BNBD.
        v_descale: E8M0 [P, Nkv, S/64, D, 2] for PA_BNBD.
        quant_mode: CANN quantization mode; 1 selects MXFP8 with FP8 P.
        block_table: Int32 [B, max_pages] physical page indices.
        p_scale: Optional FP32 [1] scale for FP8 attention probabilities.
        cu_seqlens_q, cu_seqlens_kv: Optional int32 [B+1] cumulative lengths.
        seqused_q, seqused_kv: Optional int32 [B] used sequence lengths.
        sinks, attn_mask: Optional CANN sinks and attention mask tensors.
        metadata: Device schedule from quant_flash_attn_metadata.
        softmax_scale: Multiplier applied to QK scores.
        mask_mode, win_left, win_right: CANN masking attributes.
        max_seqlen_q, max_seqlen_kv: Maximum sequence lengths; -1 is unspecified.
        layout_q, layout_q_descale, layout_kv, layout_out: CANN layout names.
        return_softmax_lse: Whether to return FP32 LSE; otherwise it is empty.
        impl: auto (cann_ops_transformer), cann_ops_transformer, or torch_native.

    Returns:
        BF16 attention output and FP32 softmax LSE. For TND, their shapes are
        [T, Nq, Dv] and [Nq, T], respectively. Inputs are not modified.
    """
    impl_fn = get_impl_by_name("attention.quant_flash_attn", impl)
    return impl_fn(
        q,
        k,
        v,
        q_descale,
        k_descale,
        v_descale,
        quant_mode,
        block_table=block_table,
        p_scale=p_scale,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_kv,
        seqused_q=seqused_q,
        seqused_kv=seqused_kv,
        sinks=sinks,
        attn_mask=attn_mask,
        metadata=metadata,
        softmax_scale=softmax_scale,
        mask_mode=mask_mode,
        win_left=win_left,
        win_right=win_right,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_kv=max_seqlen_kv,
        layout_q=layout_q,
        layout_q_descale=layout_q_descale,
        layout_kv=layout_kv,
        layout_out=layout_out,
        return_softmax_lse=return_softmax_lse,
    )
