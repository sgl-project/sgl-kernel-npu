from typing import Optional

import torch


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
) -> tuple:
    r"""Fused chunk KDA forward kernel (direct-launch NPU implementation).

    Args:
        q: (B, S, H, K) bfloat16 (layout-dependent, see ``layout``).
        k: (B, S, H, K) bfloat16, same shape as q.
        v: (B, S, HV, V) bfloat16.
        g: (B, S, HV, K) float32 or bfloat16 raw gate.
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
        output_*: whether to materialize the corresponding intermediate output.

    Returns:
        A tuple of length 11:
        (attn_out, final_state?, gk?, aqk_out, akk_out, w?, u?, qg?, kg?, v_new?, h?).
        attn_out: (B, S, HV, V) [or (T, HV, V) for rank-3 layouts].
        final_state: (N, HV, K, V) [or (N, HV, V, K) when state_v_first] float32.
        gk: (B, HV, S, K) [or (HV, S, K)] float32.
        aqk_out/akk_out: (B, HV, S, C) [or (HV, S, C)].
        w/qg/kg: (B, HV, S, K) [or (HV, S, K)].
        u/v_new: (B, HV, S, V) [or (HV, S, V)].
        h: (B, NC, HV, K, V) [or (NC, HV, K, V)].
    """
    return torch.ops.npu.chunk_kda_fwd(
        q,
        k,
        v,
        g,
        beta,
        a_log,
        dt_bias,
        initial_state,
        cu_seqlens,
        chunk_indices,
        layout,
        scale,
        chunk_size,
        safe_gate,
        lower_bound,
        use_gate_in_kernel,
        state_v_first,
        output_final_state,
        output_gk,
        output_w,
        output_u,
        output_qg,
        output_kg,
        output_v_new,
        output_h,
    )
