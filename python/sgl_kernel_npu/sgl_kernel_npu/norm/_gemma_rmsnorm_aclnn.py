from __future__ import annotations

import torch
import torch_npu


def npu_gemma_rms_norm(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply standard RMSNorm with Gemma weight semantics on Ascend 950."""
    return torch_npu.npu_rms_norm(input, 1.0 + weight, eps)
