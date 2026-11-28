from typing import Optional

import torch
import torch.nn.functional as F


def fused_gdn_prefill(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    ssm_states: torch.Tensor,
    cache_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor,
    chunk_offsets: torch.Tensor,
    num_k_heads: int,
    num_v_heads: int,
    total_valid_tokens: Optional[int] = None,
    ckpt_chunk_idx: Optional[torch.Tensor] = None,
    ckpt_dst_slots: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Independent FP32 PyTorch recurrent reference for fused_gdn_prefill."""
    D = 128
    BT = 64
    NK = int(num_k_heads)
    NV = int(num_v_heads)
    GQA = NV // NK
    T = int(mixed_qkv.shape[0])
    B = int(cu_seqlens.shape[0]) - 1
    dev = mixed_qkv.device
    scale = D**-0.5

    if chunk_indices.shape[0] == 0:
        return torch.zeros((T, NV, D), dtype=mixed_qkv.dtype, device=dev)

    Q_DIM = NK * D
    K_DIM = NK * D
    q_raw = mixed_qkv[:, :Q_DIM].view(T, NK, D).float()
    k_raw = mixed_qkv[:, Q_DIM : Q_DIM + K_DIM].view(T, NK, D).float()
    v_raw = mixed_qkv[:, Q_DIM + K_DIM :].view(T, NV, D).float()

    q_norm = F.normalize(q_raw, p=2, dim=-1, eps=1e-6)
    k_norm = F.normalize(k_raw, p=2, dim=-1, eps=1e-6)
    if GQA > 1:
        q_exp = q_norm.repeat_interleave(GQA, dim=1)
        k_exp = k_norm.repeat_interleave(GQA, dim=1)
    else:
        q_exp = q_norm
        k_exp = k_norm

    g_all = -torch.exp(A_log.float())[None, :] * F.softplus(
        a.float() + dt_bias.float()[None, :]
    )
    beta_all = torch.sigmoid(b.float())

    o = torch.zeros((T, NV, D), dtype=mixed_qkv.dtype, device=dev)
    cu_list = cu_seqlens.cpu().tolist()
    cache_list = cache_indices.cpu().tolist()
    save_ckpt = ckpt_chunk_idx is not None and ckpt_dst_slots is not None
    ckpt_idx_list = ckpt_chunk_idx.cpu().tolist() if save_ckpt else None
    ckpt_dst_list = ckpt_dst_slots.cpu().tolist() if save_ckpt else None

    for bi in range(B):
        bos = int(cu_list[bi])
        eos = int(cu_list[bi + 1])
        seq_len = eos - bos
        slot = int(cache_list[bi])
        h = ssm_states[slot].float().clone()  # [NV, DV, DK]
        ckpt_tok = -1
        ckpt_dst = -1
        if save_ckpt:
            ci = int(ckpt_idx_list[bi])
            dst = int(ckpt_dst_list[bi])
            if ci >= 0 and dst >= 0:
                ckpt_tok = ci * BT
                ckpt_dst = dst

        for local_t in range(seq_len):
            if local_t == ckpt_tok and ckpt_dst >= 0:
                ssm_states[ckpt_dst].copy_(h.to(ssm_states.dtype))
            t = bos + local_t
            qt = q_exp[t]  # [NV, D]
            kt = k_exp[t]  # [NV, D]
            vt = v_raw[t]  # [NV, D]
            gt = torch.exp(g_all[t])[:, None, None]  # [NV, 1, 1]
            bt = beta_all[t][:, None]  # [NV, 1]

            h = h * gt
            v_pred = torch.bmm(h, kt.unsqueeze(-1)).squeeze(-1)  # [NV, D]
            v_delta = (vt - v_pred) * bt
            h = h + torch.bmm(v_delta.unsqueeze(-1), kt.unsqueeze(1))
            o[t] = (torch.bmm(h, qt.unsqueeze(-1)).squeeze(-1) * scale).to(o.dtype)

        ssm_states[slot].copy_(h.to(ssm_states.dtype))

    return o
