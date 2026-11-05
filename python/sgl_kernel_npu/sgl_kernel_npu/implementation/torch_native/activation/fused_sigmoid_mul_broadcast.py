"""Independent PyTorch row-wise sigmoid gating reference."""

import torch


def fused_sigmoid_mul_broadcast(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Use a column gate explicitly: a one-dimensional gate is per row."""
    gate = gate.reshape(x.shape[0], 1)
    return (x.float() * torch.sigmoid(gate.float())).to(x.dtype)
