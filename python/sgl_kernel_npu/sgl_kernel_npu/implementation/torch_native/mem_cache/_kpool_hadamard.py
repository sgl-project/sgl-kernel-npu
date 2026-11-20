"""PyTorch reference for the normalized KPool Hadamard128 transform."""

import torch


def hadamard128(x: torch.Tensor) -> torch.Tensor:
    shape = x.shape
    x = x.float()
    for stride in (1, 2, 4, 8, 16, 32, 64):
        pairs = x.reshape(*shape[:-1], 128 // (2 * stride), 2, stride)
        a, b = pairs.unbind(dim=-2)
        x = torch.stack((a + b, a - b), dim=-2).reshape(shape)
    return x * (128**-0.5)
