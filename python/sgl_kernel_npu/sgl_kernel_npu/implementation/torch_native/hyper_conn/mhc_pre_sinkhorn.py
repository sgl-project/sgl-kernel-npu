"""Independent FP32 mHC projection and Sinkhorn reference.

Semantics: CANN ops-transformer mhc/mhc_pre_sinkhorn and SGLang's
_mhc_pre_torch; no vendor kernel is called.
"""

import torch


def mhc_pre_sinkhorn(x, phi, alpha, bias, hc_mult, num_iters, hc_eps, norm_eps):
    flat = x.float().flatten(1)
    normalized = flat * torch.rsqrt(flat.square().mean(-1, keepdim=True) + norm_eps)
    mixes = torch.nn.functional.linear(normalized, phi)
    pre_raw, post_raw, res_raw = mixes.split([hc_mult, hc_mult, hc_mult**2], dim=-1)
    pre_bias, post_bias, res_bias = bias.split([hc_mult, hc_mult, hc_mult**2])
    pre = torch.sigmoid(pre_raw * alpha[0] + pre_bias) + hc_eps
    post = 2 * torch.sigmoid(post_raw * alpha[1] + post_bias)
    comb = (res_raw * alpha[2] + res_bias).view(-1, hc_mult, hc_mult)
    comb = comb.softmax(-1) + hc_eps
    comb = comb / (comb.sum(-2, keepdim=True) + hc_eps)
    for _ in range(num_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + hc_eps)
        comb = comb / (comb.sum(-2, keepdim=True) + hc_eps)
    hin = (pre.unsqueeze(-1) * x.float()).sum(1).to(x.dtype)
    return hin, post, comb
