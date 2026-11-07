from typing import Optional

import torch

from ...utils.registry import get_impl_by_name, register_impl, register_operator

_torch_native_fn = None


@register_impl("attention.kda.chunk_kda_fwd", "auto")
@register_impl("attention.kda.chunk_kda_fwd", "ascendc")
def _ascendc_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    a_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    chunk_indices: Optional[torch.Tensor] = None,
    layout: str = "BSND",
    scale: float = 1.0,
    chunk_size: int = 64,
    safe_gate: bool = False,
    lower_bound: float = -5.0,
    use_gate_in_kernel: bool = False,
    state_v_first: bool = False,
    output_final_state: bool = True,
    output_gk: bool = False,
    output_w: bool = False,
    output_u: bool = False,
    output_qg: bool = False,
    output_kg: bool = False,
    output_v_new: bool = False,
    output_h: bool = False
):
    return torch.ops.npu.chunk_kda_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        a_log=a_log,
        dt_bias=dt_bias,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        layout=layout,
        scale=scale,
        chunk_size=chunk_size,
        safe_gate=safe_gate,
        lower_bound=lower_bound,
        use_gate_in_kernel=use_gate_in_kernel,
        state_v_first=state_v_first,
        output_final_state=output_final_state,
        output_gk=output_gk,
        output_w=output_w,
        output_u=output_u,
        output_qg=output_qg,
        output_kg=output_kg,
        output_v_new=output_v_new,
        output_h=output_h,
    )


@register_impl("attention.kda.chunk_kda_fwd", "torch_native")
def _torch_native_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    a_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    chunk_indices: Optional[torch.Tensor] = None,
    layout: str = "BSND",
    scale: float = 1.0,
    chunk_size: int = 64,
    safe_gate: bool = False,
    lower_bound: float = -5.0,
    use_gate_in_kernel: bool = False,
    state_v_first: bool = False,
    output_final_state: bool = True,
    output_gk: bool = False,
    output_w: bool = False,
    output_u: bool = False,
    output_qg: bool = False,
    output_kg: bool = False,
    output_v_new: bool = False,
    output_h: bool = False
):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ....implementation.torch_native.attention.kda.chunk_kda_fwd import (
            chunk_kda_fwd,
        )

        _torch_native_fn = chunk_kda_fwd
    return _torch_native_fn(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        a_log=a_log,
        dt_bias=dt_bias,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        layout=layout,
        scale=scale,
        chunk_size=chunk_size,
        safe_gate=safe_gate,
        lower_bound=lower_bound,
        use_gate_in_kernel=use_gate_in_kernel,
        state_v_first=state_v_first,
        output_final_state=output_final_state,
        output_gk=output_gk,
        output_w=output_w,
        output_u=output_u,
        output_qg=output_qg,
        output_kg=output_kg,
        output_v_new=output_v_new,
        output_h=output_h,
    )


@register_operator("attention.kda.chunk_kda_fwd")
def chunk_kda_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    a_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    chunk_indices: Optional[torch.Tensor] = None,
    layout: str = "BSND",
    scale: float = 1.0,
    chunk_size: int = 64,
    safe_gate: bool = False,
    lower_bound: float = -5.0,
    use_gate_in_kernel: bool = False,
    state_v_first: bool = False,
    output_final_state: bool = True,
    output_gk: bool = False,
    output_w: bool = False,
    output_u: bool = False,
    output_qg: bool = False,
    output_kg: bool = False,
    output_v_new: bool = False,
    output_h: bool = False,
    impl="auto"
):
    """Run fused per-channel KDA prefill; auto selects AscendC.

    Args:
        q: (B, S, H, K) bfloat16 (layout-dependent, see ``layout``).
        k: (B, S, H, K) bfloat16, same shape as q.
        v: (B, S, HV, V) bfloat16.
        g: (B, S, HV, K) float32 or bfloat16 log decay, or raw gate when activated internally.
        beta: (B, S, HV) float32 or bfloat16 delta coefficient.
        a_log: (HV,) float32 gate decay parameter, required when use_gate_in_kernel.
        dt_bias: (HV*K,) float32 gate bias.
        initial_state: (N, HV, K, V) [or (N, HV, V, K) when state_v_first] float32.
        cu_seqlens: (N+1,) int64 varlen sequence boundaries.
        chunk_indices: (2*NC,) int64 canonical (seq_id, chunk_id) pairs.
        layout: one of "BSND", "BNSD", "TND", "NTD".
        scale: query scaling factor, usually K^(-0.5).
        chunk_size: 64 or 128.
        safe_gate: whether to use a bounded gate.
        lower_bound: safe gate lower bound in [-5, 0).
        use_gate_in_kernel: whether to compute the activated gate in-kernel.
        state_v_first: whether the state tensors have (V, K) last two dims.
        output_final_state: return the FP32 final recurrent state; default True.
        output_gk: return chunk-local cumulative log2 decay; default False.
        output_w: return transformed keys; default False.
        output_u: return transformed values; default False.
        output_qg: return gated queries; default False.
        output_kg: return keys decayed to each chunk end; default False.
        output_v_new: return values corrected by the entering state; default False.
        output_h: return entering chunk states; default False.
        impl: "auto" (AscendC), "ascendc", or "torch_native".

    Returns:
        A tuple of length 11:
        (attn_out, final_state?, gk?, aqk_out, akk_out, w?, u?, qg?, kg?, v_new?, h?).
        attn_out: (B, S, HV, V) [or (T, HV, V) for rank-3 layouts].
        final_state: (N, HV, K, V) [or (N, HV, V, K) when state_v_first] float32.
        gk: (B, HV, S, K) [or (HV, S, K)] float32.
        aqk_out/akk_out: (B, HV, S, C) [or (HV, S, C)].
        w/qg/kg: (B, HV, S, K) [or (HV, S, K)].
        u/v_new: (B, HV, S, V) [or (HV, S, V)].
        h: (B, NC, HV, K, V) [or (NC, HV, K, V)]; K/V swap when
            state_v_first=True. Unrequested optional outputs are None.

    Example:
        >>> from sgl_kernel_npu.operators.attention.kda import chunk_kda_fwd
        >>> q = torch.randn(1, 64, 1, 128, device="npu", dtype=torch.bfloat16)
        >>> k, v = torch.randn_like(q), torch.randn_like(q)
        >>> g = torch.full(q.shape, -0.01, device=q.device, dtype=torch.float32)
        >>> beta = torch.full(q.shape[:-1], 0.5, device=q.device, dtype=torch.float32)
        >>> outputs = chunk_kda_fwd(q, k, v, g, beta, scale=128**-0.5)
        >>> outputs[0].shape
        torch.Size([1, 64, 1, 128])
    """
    return get_impl_by_name("attention.kda.chunk_kda_fwd", impl)(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        a_log=a_log,
        dt_bias=dt_bias,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        layout=layout,
        scale=scale,
        chunk_size=chunk_size,
        safe_gate=safe_gate,
        lower_bound=lower_bound,
        use_gate_in_kernel=use_gate_in_kernel,
        state_v_first=state_v_first,
        output_final_state=output_final_state,
        output_gk=output_gk,
        output_w=output_w,
        output_u=output_u,
        output_qg=output_qg,
        output_kg=output_kg,
        output_v_new=output_v_new,
        output_h=output_h,
    )
