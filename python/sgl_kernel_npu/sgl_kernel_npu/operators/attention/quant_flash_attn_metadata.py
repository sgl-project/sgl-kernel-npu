"""CANN device scheduling metadata for quantized flash attention."""

from typing import Optional

import torch

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator

_cann_fn = None


@register_impl(
    "attention.quant_flash_attn_metadata",
    "auto",
    supported_generations={Generation.ARCH35},
)
@register_impl(
    "attention.quant_flash_attn_metadata",
    "cann_ops_transformer",
    supported_generations={Generation.ARCH35},
)
def _cann_impl(*args, **kwargs):
    global _cann_fn
    if _cann_fn is None:
        from cann_ops_transformer.ops import quant_flash_attn_metadata

        _cann_fn = quant_flash_attn_metadata
    return _cann_fn(*args, **kwargs)


@register_operator("attention.quant_flash_attn_metadata")
def quant_flash_attn_metadata(
    num_heads_q: int,
    num_heads_kv: int,
    head_dim: int,
    quant_mode: int,
    *,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_kv: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_kv: Optional[torch.Tensor] = None,
    batch_size: Optional[int] = None,
    max_seqlen_q: int = -1,
    max_seqlen_kv: int = -1,
    head_dim_v: Optional[int] = None,
    mask_mode: int = 0,
    win_left: int = -1,
    win_right: int = -1,
    layout_q: str = "BSND",
    layout_q_descale: str = "BSND",
    layout_kv: str = "BSND",
    layout_out: str = "BSND",
    is_grad_enabled: bool = False,
    impl: str = "auto",
) -> torch.Tensor:
    """Build a device QFA schedule; auto selects cann_ops_transformer on arch35.

    The schedule is CANN-specific, with no independent torch_native scheduler.
    Its correctness is checked through the consuming attention computation.
    Length tensors remain on device; this wrapper does not prepare lengths or
    page tables. All attributes must agree with the consuming QFA call.

    Args:
        num_heads_q, num_heads_kv: Local query and KV head counts.
        head_dim: Query/key head dimension.
        quant_mode: CANN quantization mode; 1 selects MXFP8.
        cu_seqlens_q, cu_seqlens_kv: Optional int32 [B+1] cumulative lengths.
        seqused_q, seqused_kv: Optional int32 [B] used sequence lengths.
        batch_size: Optional batch size; for TND, infer it from length shapes.
        max_seqlen_q, max_seqlen_kv: Maximum lengths; -1 is unspecified.
        head_dim_v: Value head dimension; None uses head_dim.
        mask_mode, win_left, win_right: CANN masking attributes.
        layout_q, layout_q_descale, layout_kv, layout_out: CANN layout names.
        is_grad_enabled: CANN training-schedule flag; inference uses False.
        impl: auto (cann_ops_transformer) or cann_ops_transformer.

    Returns:
        Int32 device tensor [2, max_schedule_size]. CANN determines the schedule
        size and contents. Input tensors are not modified.
    """
    impl_fn = get_impl_by_name("attention.quant_flash_attn_metadata", impl)
    return impl_fn(
        num_heads_q,
        num_heads_kv,
        head_dim,
        quant_mode,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_kv,
        seqused_q=seqused_q,
        seqused_kv=seqused_kv,
        batch_size=batch_size,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_kv=max_seqlen_kv,
        head_dim_v=head_dim_v,
        mask_mode=mask_mode,
        win_left=win_left,
        win_right=win_right,
        layout_q=layout_q,
        layout_q_descale=layout_q_descale,
        layout_kv=layout_kv,
        layout_out=layout_out,
        is_grad_enabled=is_grad_enabled,
    )
