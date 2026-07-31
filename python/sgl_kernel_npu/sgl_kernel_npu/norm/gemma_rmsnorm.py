from __future__ import annotations

from typing import TYPE_CHECKING

from sgl_kernel_npu._build_target import GEMMA_RMS_NORM_PROVIDER

if TYPE_CHECKING:
    import torch

import torch_npu

if GEMMA_RMS_NORM_PROVIDER == "native":

    def npu_gemma_rms_norm(
        input: torch.Tensor,
        weight: torch.Tensor,
        eps: float = 1e-6,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply native Gemma RMSNorm on Ascend 910B/910C."""
        return torch_npu.npu_gemma_rms_norm(input, weight, eps)

elif GEMMA_RMS_NORM_PROVIDER == "aclnn":

    def npu_gemma_rms_norm(
        input: torch.Tensor,
        weight: torch.Tensor,
        eps: float = 1e-6,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply standard RMSNorm with Gemma weight semantics on Ascend 950."""
        return torch_npu.npu_rms_norm(input, 1.0 + weight, eps)

else:
    raise RuntimeError(
        f"Unsupported Gemma RMSNorm provider: {GEMMA_RMS_NORM_PROVIDER!r}"
    )
