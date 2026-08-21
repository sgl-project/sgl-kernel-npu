"""Ascend 910 Gemma RMSNorm provider.

Staged into the wheel as ``norm/gemma_rmsnorm.py`` when building for
``Ascend910`` (see ``build_tools/target_provider.py``). The provider tree is
build input, never a runtime package: the module only exists under its stable
path inside a built wheel.
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
