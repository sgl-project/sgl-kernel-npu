"""Public entry for the legacy decomposed GDN forward pipeline."""

from typing import Optional

import torch

from ...utils.registry import get_impl_by_name, register_impl, register_operator

_triton_fn = None
_torch_native_fn = None


@register_impl("attention.gdn.chunk_gated_delta_rule_npu", "auto")
@register_impl("attention.gdn.chunk_gated_delta_rule_npu", "triton")
def _triton_impl(*args, **kwargs):
    global _triton_fn
    if _triton_fn is None:
        from .chunk import chunk_gated_delta_rule_npu

        _triton_fn = chunk_gated_delta_rule_npu
    return _triton_fn(*args, **kwargs)


@register_impl("attention.gdn.chunk_gated_delta_rule_npu", "torch_native")
def _torch_native_impl(*args, **kwargs):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ....implementation.torch_native.attention.gdn.chunk_gated_delta_rule_npu import (
            chunk_gated_delta_rule_npu,
        )

        _torch_native_fn = chunk_gated_delta_rule_npu
    return _torch_native_fn(*args, **kwargs)


@register_operator("attention.gdn.chunk_gated_delta_rule_npu")
def chunk_gated_delta_rule_npu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = True,
    cu_seqlens: Optional[torch.LongTensor] = None,
    head_first: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    *,
    impl="auto",
):
    """Return (o, final_state, h); auto selects the copied Triton composition.

    q/k: [B,T,Hg,K], v: [B,T,H,V], g/beta: [B,T,H].
    initial_state/final_state: [N,H,K,V]; final_state is optional FP32.
    h contains entering chunk states in the input dtype. Packed inputs use
    B=1 and cu_seqlens=[N+1]. head_first=True raises DeprecationWarning.
    All legacy defaults, input preparation and environment switches apply.
    This entry is separate from the fused CANN chunk_gated_delta_rule API.
    """
    return get_impl_by_name("attention.gdn.chunk_gated_delta_rule_npu", impl)(
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        output_final_state,
        cu_seqlens,
        head_first,
        use_qk_l2norm_in_kernel,
    )
