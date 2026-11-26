"""PyTorch reference for fast GELU."""

import torch


def fast_gelu(input):
    x = input.float()
    return (x * torch.sigmoid(1.702 * x)).to(input.dtype)
