"""Independent eager reference for mhc_pre."""

import torch
import torch.nn.functional as F


def mhc_pre_dpskv4(x, fn, scale, base, hc_mult, sinkhorn_iters, hc_eps, rms_norm_eps):
    if x.ndim != 3 or x.shape[1] != hc_mult:
        raise ValueError("x must be [tokens, hc_mult, hidden]")
    if sinkhorn_iters < 1:
        raise ValueError("sinkhorn_iters must be positive")
    flat = x.flatten(1).float()
    mixes = F.linear(flat, fn.float()) * torch.rsqrt(
        flat.square().mean(-1, keepdim=True) + rms_norm_eps
    )
    n = hc_mult
    pre = torch.sigmoid(mixes[:, :n] * scale[0] + base[:n]) + hc_eps
    post = 2 * torch.sigmoid(mixes[:, n : 2 * n] * scale[1] + base[n : 2 * n])
    comb = (mixes[:, 2 * n :] * scale[2] + base[2 * n :]).reshape(-1, n, n)
    comb = comb.softmax(-1) + hc_eps
    comb = comb / (comb.sum(-2, keepdim=True) + hc_eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + hc_eps)
        comb = comb / (comb.sum(-2, keepdim=True) + hc_eps)
    y = (x.float() * pre.unsqueeze(-1)).sum(1).to(x.dtype)
    return y, post.float(), comb.float()
