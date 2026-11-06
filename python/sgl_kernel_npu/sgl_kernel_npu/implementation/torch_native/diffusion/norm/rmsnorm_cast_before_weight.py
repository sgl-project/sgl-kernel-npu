"""Eager reference for RMSNorm with the cast-before-weight rounding."""

import torch


def rmsnorm_cast_before_weight(x, weight, eps):
    """Compute ``weight * (x * rsqrt(mean(x**2) + eps))`` with a BF16 cast first."""
    values = x.float()
    variance = values.square().mean(dim=-1, keepdim=True)
    normalized = values * torch.rsqrt(variance + eps)
    return weight * normalized.to(x.dtype)
