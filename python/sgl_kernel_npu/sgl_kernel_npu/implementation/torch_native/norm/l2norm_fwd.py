"""L2 forward uses epsilon under the square root, matching the native kernels."""

import torch


def l2norm_fwd(x, out, eps):
    values = x.float()
    out.copy_(values * torch.rsqrt(values.square().sum(-1, keepdim=True) + eps))
