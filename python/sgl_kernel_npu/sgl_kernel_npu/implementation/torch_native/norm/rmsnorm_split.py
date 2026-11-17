"""PyTorch reference for the two phases of a tensor-parallel RMSNorm."""

import torch


def fused_rsqrt_mul(x, variance, weight, eps=1e-6):
    # The Triton kernel keeps the input dtype and reads one variance per row;
    # a flat [B*L] variance is the same layout as [B, L, 1].
    if variance.dim() == 1:
        variance = variance.view(x.shape[0], x.shape[1], 1)
    return x * torch.rsqrt(variance + eps) * weight


def fused_variance(x):
    # The Triton kernel accumulates in the input dtype and returns [B, L, 1].
    return x.square().sum(dim=-1, keepdim=True) / x.shape[-1]
