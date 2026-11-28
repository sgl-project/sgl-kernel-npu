"""Independent FP32 PyTorch reference implementations for fused GDN verify and commit."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def _recurrent_gdn_verify_ref(
    q_raw: torch.Tensor,  # [B, K, NK, D]
    k_raw: torch.Tensor,  # [B, K, NK, D]
    v_raw: torch.Tensor,  # [B, K, NV, D]
    a: torch.Tensor,  # [B, K, NV]
    b: torch.Tensor,  # [B, K, NV]
    A_log: torch.Tensor,  # [NV]
    dt_bias: torch.Tensor,  # [NV]
    ssm_states: torch.Tensor,  # [NUM_SLOTS, NV, D, D]
    cache_indices: torch.Tensor,  # [B]
    eps: float = 1e-6,
    out_dtype: torch.dtype = torch.bfloat16,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Step-by-step FP32 recurrent GDN verify (independent of WY chunking)."""
    B, K, NK, D = q_raw.shape
    NV = v_raw.shape[2]
    gqa = NV // NK
    scale = D**-0.5

    qf = q_raw.float()
    kf = k_raw.float()
    vf = v_raw.float()
    q_norm = qf * torch.rsqrt(qf.pow(2).sum(dim=-1, keepdim=True) + eps)
    k_norm = kf * torch.rsqrt(kf.pow(2).sum(dim=-1, keepdim=True) + eps)

    # Quantize normalized Q/K to bf16 matching Triton prep boundary
    q_n = q_norm.to(torch.bfloat16).float()
    k_n = k_norm.to(torch.bfloat16).float()

    g_step = -torch.exp(A_log.float()).view(1, 1, NV) * F.softplus(
        a.float() + dt_bias.float().view(1, 1, NV)
    )
    beta = torch.sigmoid(b.float())
    rec_g = torch.cumsum(g_step, dim=1)

    state = torch.zeros((B, NV, D, D), dtype=torch.float32, device=q_raw.device)
    slots = cache_indices[:B].to(torch.int64)
    valid = slots >= 0
    if valid.any():
        state[valid] = ssm_states[slots[valid]].float()

    out = torch.empty((B, K, NV, D), dtype=torch.float32, device=q_raw.device)
    rec_v = torch.empty((B, K, NV, D), dtype=torch.float32, device=q_raw.device)

    for t in range(K):
        qt = q_n[:, t].repeat_interleave(gqa, dim=1)  # [B, NV, D]
        kt = k_n[:, t].repeat_interleave(gqa, dim=1)  # [B, NV, D]
        vt = vf[:, t]  # [B, NV, D]
        gt = torch.exp(g_step[:, t])[:, :, None, None]  # [B, NV, 1, 1]
        bt = beta[:, t][:, :, None]  # [B, NV, 1]

        state = state * gt
        # Exact Delta-rule innovation vector == WY (U - W @ S_0^T) at step t
        v_pred = torch.einsum("bhvk,bhk->bhv", state, kt)
        delta_t = bt * (vt - v_pred)
        rec_v[:, t] = delta_t
        state = state + torch.einsum("bhv,bhk->bhvk", delta_t, kt)
        out[:, t] = torch.einsum("bhvk,bhk->bhv", state, qt) * scale

    return (
        out.to(out_dtype),
        k_n.to(torch.bfloat16),
        rec_v.to(torch.bfloat16),
        rec_g.to(torch.float32),
    )


def fused_gdn_verify(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    ssm_states: torch.Tensor,
    cache_indices: torch.Tensor,
    num_k_heads: int,
    num_v_heads: int,
    head_dim: int = 128,
    eps: float = 1e-6,
    workspace=None,
    linear_layer_idx: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, K, _ = mixed_qkv.shape
    if B == 0:
        dev = mixed_qkv.device
        return (
            torch.empty(
                (0, K, num_v_heads, head_dim), dtype=mixed_qkv.dtype, device=dev
            ),
            torch.empty(
                (0, K, num_k_heads, head_dim), dtype=torch.bfloat16, device=dev
            ),
            torch.empty(
                (0, K, num_v_heads, head_dim), dtype=torch.bfloat16, device=dev
            ),
            torch.empty((0, K, num_v_heads), dtype=torch.float32, device=dev),
        )
    q_dim = num_k_heads * head_dim
    k_dim = num_k_heads * head_dim
    v_dim = num_v_heads * head_dim
    q_raw, k_raw, v_raw = torch.split(mixed_qkv, [q_dim, k_dim, v_dim], dim=-1)
    out, rec_k, rec_v, rec_g = _recurrent_gdn_verify_ref(
        q_raw.view(B, K, num_k_heads, head_dim),
        k_raw.view(B, K, num_k_heads, head_dim),
        v_raw.view(B, K, num_v_heads, head_dim),
        a,
        b,
        A_log,
        dt_bias,
        ssm_states,
        cache_indices,
        eps=eps,
        out_dtype=mixed_qkv.dtype,
    )
    if workspace is not None and workspace.rec_k_all is not None:
        workspace.rec_k_all[linear_layer_idx, :B].copy_(rec_k)
        workspace.rec_v_all[linear_layer_idx, :B].copy_(rec_v)
        workspace.rec_g_all[linear_layer_idx, :B].copy_(rec_g)
    return out, rec_k, rec_v, rec_g


def fused_conv1d_gdn_verify(
    qkvz: torch.Tensor,
    ba: torch.Tensor,
    conv_weights_t: torch.Tensor,
    conv_states: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    ssm_states: torch.Tensor,
    cache_indices: torch.Tensor,
    batch_size: int,
    k_ver: int,
    num_k_heads: int,
    num_v_heads: int,
    head_dim: int = 128,
    eps: float = 1e-6,
    query_start_loc: Optional[torch.Tensor] = None,
    workspace=None,
    linear_layer_idx: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if batch_size == 0:
        dev = qkvz.device
        return (
            torch.empty(
                (0, k_ver, num_v_heads, head_dim), dtype=qkvz.dtype, device=dev
            ),
            torch.empty(
                (0, k_ver, num_k_heads, head_dim), dtype=torch.bfloat16, device=dev
            ),
            torch.empty(
                (0, k_ver, num_v_heads, head_dim), dtype=torch.bfloat16, device=dev
            ),
            torch.empty((0, k_ver, num_v_heads), dtype=torch.float32, device=dev),
        )
    qkv_dim = (2 * num_k_heads + num_v_heads) * head_dim
    win_len = conv_states.shape[1]
    x_in = qkvz[:, :qkv_dim].view(batch_size, k_ver, qkv_dim)
    w = conv_weights_t.float()  # [4, qkv_dim]

    mixed_qkv = torch.empty(
        (batch_size, k_ver, qkv_dim), dtype=torch.bfloat16, device=qkvz.device
    )
    has_qsl = (
        query_start_loc is not None
        and query_start_loc.ndim == 1
        and query_start_loc.shape[0] >= batch_size + 1
    )
    slots_list = cache_indices[:batch_size].to(torch.int64).tolist()
    qsl_list = query_start_loc.to(torch.int64).tolist() if has_qsl else None

    for i_b in range(batch_size):
        slot = slots_list[i_b]
        valid = slot >= 0 and (not has_qsl or qsl_list[i_b + 1] > qsl_list[i_b])
        if valid:
            old_tail = conv_states[slot, win_len - 3 : win_len].clone()
            conv_states[slot, 0].copy_(old_tail[1])
            conv_states[slot, 1].copy_(old_tail[2])
            s0, s1, s2 = old_tail[0].float(), old_tail[1].float(), old_tail[2].float()
        else:
            zeros = torch.zeros((qkv_dim,), dtype=torch.float32, device=qkvz.device)
            s0, s1, s2 = zeros, zeros, zeros
        for t in range(k_ver):
            xt = x_in[i_b, t]
            if valid:
                conv_states[slot, 2 + t].copy_(xt)
            xf = xt.float()
            acc = s0 * w[0] + s1 * w[1] + s2 * w[2] + xf * w[3]
            mixed_qkv[i_b, t] = F.silu(acc).to(torch.bfloat16)
            s0, s1, s2 = s1, s2, xf

    ba_3d = ba.view(batch_size, k_ver, 2 * num_v_heads)
    b_in = ba_3d[:, :, :num_v_heads].contiguous()
    a_in = ba_3d[:, :, num_v_heads:].contiguous()
    return fused_gdn_verify(
        mixed_qkv,
        a_in,
        b_in,
        A_log,
        dt_bias,
        ssm_states,
        cache_indices,
        num_k_heads,
        num_v_heads,
        head_dim=head_dim,
        eps=eps,
        workspace=workspace,
        linear_layer_idx=linear_layer_idx,
    )


def fused_gdn_commit(
    ssm_states_all: torch.Tensor,
    rec_k_all: torch.Tensor,
    rec_v_all: torch.Tensor,
    rec_g_all: torch.Tensor,
    dst_slots: torch.Tensor,
    last_steps: torch.Tensor,
    track_slots: Optional[torch.Tensor] = None,
    track_steps: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Independent FP32 PyTorch reference for all-layer in-place SSM commit."""
    num_layers, _, nv, _, _ = ssm_states_all.shape
    nk = rec_k_all.shape[3]
    k_ver = rec_k_all.shape[2]
    gqa = nv // nk
    B = dst_slots.shape[0]
    if B == 0:
        return ssm_states_all

    has_track = track_slots is not None and track_steps is not None
    dst_list = dst_slots.to(torch.int64).tolist()
    last_list = last_steps.to(torch.int64).tolist()
    trk_slot_list = track_slots.to(torch.int64).tolist() if has_track else None
    trk_step_list = track_steps.to(torch.int64).tolist() if has_track else None

    def _compute_committed(h0_f, k_f, v_f, g_f, step_idx):
        # h0_f: [NV, D, D], k_f: [K, NV, D], v_f: [K, NV, D], g_f: [K, NV]
        g_end = g_f[step_idx]  # [NV]
        steps = torch.arange(step_idx + 1, device=h0_f.device)
        dec = torch.exp(g_end.unsqueeze(0) - g_f[steps])  # [step+1, NV]
        v_dec = v_f[steps] * dec.unsqueeze(-1)  # [step+1, NV, D]
        k_sub = k_f[steps]  # [step+1, NV, D]
        upd = torch.einsum("thv,thk->hvk", v_dec, k_sub)
        return h0_f * torch.exp(g_end)[:, None, None] + upd

    for i_l in range(num_layers):
        for i_b in range(B):
            dst = dst_list[i_b]
            last = last_list[i_b]
            if dst < 0 or last < 0 or last >= k_ver:
                continue
            h0_f = ssm_states_all[i_l, dst].float().clone()
            k_f = rec_k_all[i_l, i_b].float().repeat_interleave(gqa, dim=1)
            v_f = rec_v_all[i_l, i_b].float()
            g_f = rec_g_all[i_l, i_b].float()

            if has_track:
                tslot = trk_slot_list[i_b]
                tstep = trk_step_list[i_b]
                if tslot >= 0 and 0 <= tstep < k_ver:
                    h_trk = _compute_committed(h0_f, k_f, v_f, g_f, tstep)
                    ssm_states_all[i_l, tslot].copy_(h_trk.to(ssm_states_all.dtype))

            h_new = _compute_committed(h0_f, k_f, v_f, g_f, last)
            ssm_states_all[i_l, dst].copy_(h_new.to(ssm_states_all.dtype))

    return ssm_states_all
