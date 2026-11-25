"""Independent PyTorch reference for same-shape sigmoid gating."""

import torch


def fused_sigmoid_mul(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Compute in FP32 and cast once to the input dtype.

    This implementation uses only PyTorch operations.
    """
    return (x.float() * torch.sigmoid(gate.float())).to(x.dtype)
