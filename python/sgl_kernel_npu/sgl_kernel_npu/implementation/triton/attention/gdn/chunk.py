# Adapted from fla/chunk.py; preserve its forward-only chunk pipeline.
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
import os
from typing import Optional

import torch
from einops import rearrange

from ...norm.l2norm_fwd import l2norm_fwd
from .chunk_delta_h import (
    chunk_gated_delta_rule_fwd_h_npu as chunk_gated_delta_rule_fwd_h,
)
from .chunk_local_cumsum import chunk_local_cumsum
from .chunk_o import chunk_fwd_o_npu as chunk_fwd_o
from .chunk_scaled_dot_kkt import (
    chunk_scaled_dot_kkt_fwd_npu as chunk_scaled_dot_kkt_fwd,
)
from .gdn_utils import SUPPRESS_LEVEL, input_guard
from .mega_chunk_gdn import run_mega_chunk_gdn
from .recompute_w_u_fwd import recompute_w_u_fwd
from .solve_tril import solve_tril


def _use_mega_gdn():
    return os.getenv("GDN_USE_MEGA_GDN", "0") == "1"


def chunk_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: Optional[torch.LongTensor] = None,
):
    if _use_mega_gdn():
        g, o, A, final_state, w, h, v_new = run_mega_chunk_gdn(
            q, k, v, g, beta, scale, initial_state, output_final_state, cu_seqlens
        )
        if SUPPRESS_LEVEL < 3:
            return g, o, A, final_state, None, h, None
        return g, o, A, final_state, w, h, v_new

    g = chunk_local_cumsum(g, chunk_size=64, cu_seqlens=cu_seqlens)
    # obtain WY representation. u is actually the new v.
    A = chunk_scaled_dot_kkt_fwd(
        k=k, beta=beta, g_cumsum=g, cu_seqlens=cu_seqlens, output_dtype=torch.float32
    )
    A = solve_tril(A=A, cu_seqlens=cu_seqlens, output_dtype=k.dtype)
    w, u = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        g_cumsum=g,
        cu_seqlens=cu_seqlens,
    )
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
    )
    o = chunk_fwd_o(
        q=q,
        k=k,
        v=v_new,
        h=h,
        g=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
    )
    if SUPPRESS_LEVEL < 3:
        return g, o, A, final_state, None, h, None
    elif SUPPRESS_LEVEL >= 3:
        return g, o, A, final_state, w, h, v_new


@input_guard
@torch.compiler.disable
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
):
    """Return (output, final_state, entering_chunk_states) for chunked GDN.

    Inputs use sequence-first [B, T, H, D] storage. Packed sequences use
    B=1 and cu_seqlens. initial_state is [N, H, K, V]; final_state is FP32
    when requested, and h is [B, total_chunks, H, K, V] in the key dtype.
    """

    assert q.dtype == k.dtype == v.dtype
    assert (
        q.dtype != torch.float32
    ), "ChunkGatedDeltaRuleFunction does not support float32. Please use bfloat16."
    assert (
        len(beta.shape) == 3
    ), "beta must be of shape [B, T, H] if head_first=False, or [B, H, T] otherwise."

    if head_first:
        raise DeprecationWarning(
            "head_first is deprecated and will be removed in a future version. "
            "Please use head_first=False for now instead."
        )
        q, k, v, beta, g = map(
            lambda x: rearrange(x, "b h t ... -> b t h ..."), (q, k, v, beta, g)
        )
    # if not head_first and q.shape[1] < q.shape[2]:
    #     warnings.warn(
    #         f"Input tensor shape suggests potential format mismatch: seq_len ({q.shape[1]}) < num_heads ({q.shape[2]}). "
    #         "This may indicate the inputs were passed in head-first format [B, H, T, ...] "
    #         "when head_first=False was specified. "
    #         "Please verify your input tensor format matches the expected shape [B, T, H, ...]."
    #     )
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing."
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}."
            )
    if scale is None:
        scale = k.shape[-1] ** -0.5

    if use_qk_l2norm_in_kernel:
        q = l2norm_fwd(q)
        k = l2norm_fwd(k)

    _, o, _, final_state, _, h, _ = chunk_gated_delta_rule_fwd(
        q, k, v, g, beta, scale, initial_state, output_final_state, cu_seqlens
    )
    o = o.to(q.dtype)
    if head_first:
        o = rearrange(o, "b t h ... -> b h t ...")
    return o, final_state, h
