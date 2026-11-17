"""Independent scalar-step reference for the fused GDN decode kernel."""

import torch
import torch.nn.functional as F


def fused_gdn_decode(
    A_log,
    a,
    dt_bias,
    softplus_beta,
    softplus_threshold,
    q,
    k,
    v,
    b,
    initial_state_source,
    initial_state_indices,
    scale=None,
    use_qk_l2norm_in_kernel=False,
    cu_seqlens=None,
):
    q, k, v, a, b = (x.contiguous() for x in (q, k, v, a, b))
    state = initial_state_source.contiguous()
    indices = initial_state_indices.contiguous()
    batch, length, heads, key_dim = q.shape
    value_heads, value_dim = v.shape[-2:]
    sequences = batch if cu_seqlens is None else len(cu_seqlens) - 1
    if cu_seqlens is None:
        if length != 1:
            raise ValueError("decode kernel only supports T=1")
        starts = list(range(batch))
    else:
        if length != sequences:
            raise ValueError("decode kernel only supports 1 token per sequence")
        starts = cu_seqlens.tolist()[:-1]
    if scale is None:
        scale = key_dim**-0.5
    elif scale <= 0:
        raise ValueError("scale must be positive")
    qflat = q.reshape(-1, heads, key_dim)
    kflat = k.reshape(-1, heads, key_dim)
    vflat = v.reshape(-1, value_heads, value_dim)
    aflat = a.reshape(-1, value_heads)
    bflat = b.reshape(-1, value_heads)
    result = torch.empty_like(v)
    out = result.reshape(-1, value_heads, value_dim)
    # The Triton kernel addresses each physical state slot as [V, K].
    states = state.reshape(-1, value_heads, value_dim, key_dim)
    for n, pos in enumerate(starts):
        slot = int(indices[n].item())
        for hv in range(value_heads):
            h = hv // (value_heads // heads)
            query, key = qflat[pos, h].float(), kflat[pos, h].float()
            if use_qk_l2norm_in_kernel:
                query = query / (query.square().sum().sqrt() + 1e-6)
                key = key / (key.square().sum().sqrt() + 1e-6)
            gate = -A_log[hv].float().exp() * F.softplus(
                aflat[pos, hv].float() + dt_bias[hv].float(),
                beta=softplus_beta,
                threshold=softplus_threshold,
            )
            beta = bflat[pos, hv].float().sigmoid()
            if slot >= 0:
                current = states[slot, hv].float().clone()
            else:
                current = torch.zeros((value_dim, key_dim), device=q.device)
            current *= gate.exp()
            delta = (vflat[pos, hv].float() - (current * key).sum(-1)) * beta
            current += delta[:, None] * key[None, :]
            out[pos, hv] = (current * (query * scale)).sum(-1).to(out.dtype)
            if slot >= 0:
                states[slot, hv] = current.to(state.dtype)
    return result
