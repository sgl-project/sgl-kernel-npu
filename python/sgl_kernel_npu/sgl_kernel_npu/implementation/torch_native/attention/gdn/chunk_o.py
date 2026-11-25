# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import torch


def chunk_fwd_o(q, k, v, h, g=None, scale=None, cu_seqlens=None, chunk_size=64):
    B, T, Hg, K = q.shape
    H, V = v.shape[-2:]
    gates = g.transpose(1, 2).contiguous().flatten()
    scale = K**-0.5 if scale is None else scale
    if cu_seqlens is None:
        bounds = [(b * T, (b + 1) * T) for b in range(B)]
    else:
        offsets = cu_seqlens.tolist()
        bounds = list(zip(offsets[:-1], offsets[1:]))
    qs, ks = q.reshape(-1, Hg, K), k.reshape(-1, Hg, K)
    vs = v.reshape(-1, H, V)
    states = h.reshape(-1, H, K, V)
    out = torch.empty_like(v)
    flat_out = out.view(-1, H, V)
    state_id = 0
    for bos, eos in bounds:
        for start in range(bos, eos, chunk_size):
            end = min(start + chunk_size, eos)
            size = end - start
            query = (
                qs[start:end].repeat_interleave(H // Hg, dim=1).transpose(0, 1).float()
            )
            key = (
                ks[start:end].repeat_interleave(H // Hg, dim=1).transpose(0, 1).float()
            )
            values = vs[start:end].transpose(0, 1).float()
            decay = gates.as_strided(
                (H, size), (T, 1), storage_offset=gates.storage_offset() + start
            ).float()
            diff = decay[:, :, None] - decay[:, None, :]
            weights = (query @ key.transpose(-1, -2)) * torch.where(
                diff <= 0, diff, float("-inf")
            ).exp()
            weights = weights.tril().to(v.dtype).float()
            inter = (query @ states[state_id].float()) * decay.exp()[..., None]
            result = inter * scale + (weights @ values) * scale
            flat_out[start:end] = result.transpose(0, 1).to(v.dtype)
            state_id += 1
    return out
