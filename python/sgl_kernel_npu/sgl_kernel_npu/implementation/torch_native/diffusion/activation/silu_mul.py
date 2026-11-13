"""Eager reference for the fused silu-multiply."""

import torch


def silu_mul(a, b):
    """Compute ``silu(a) * b`` with PyTorch's BF16 rounding boundaries."""
    return torch.nn.functional.silu(a) * b
