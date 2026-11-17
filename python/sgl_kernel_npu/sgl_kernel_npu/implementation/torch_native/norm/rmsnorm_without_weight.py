"""PyTorch weightless RMSNorm over the last dimension."""

import torch


def fused_rmsnorm_without_weight(x, eps):
    values = x.float()
    variance = values.square().mean(dim=-1, keepdim=True)
    output = values * torch.rsqrt(variance + eps)
    return output.to(x.dtype)
