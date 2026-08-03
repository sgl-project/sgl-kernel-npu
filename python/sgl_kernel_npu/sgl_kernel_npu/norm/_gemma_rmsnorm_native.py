"""Native Gemma RMSNorm provider, staged as ``norm/gemma_rmsnorm.py`` on 910.

Exactly one provider module is staged into the wheel (see ``setup.py``), so this
file is never importable under its own name from an installed package.
"""

from __future__ import annotations

import torch
import torch_npu


def npu_gemma_rms_norm(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply native Gemma RMSNorm on Ascend 910."""
    return torch_npu.npu_gemma_rms_norm(input, weight, eps)
