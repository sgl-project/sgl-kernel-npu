from __future__ import annotations

import torch
import torch_npu

if torch.ops.npu.sgl_kernel_npu_use_native_gemma_rms_norm():

    def npu_gemma_rms_norm(
        input: torch.Tensor,
        weight: torch.Tensor,
        eps: float = 1e-6,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply native Gemma RMSNorm on Ascend 910."""
        return torch_npu.npu_gemma_rms_norm(input, weight, eps)

else:

    def npu_gemma_rms_norm(
        input: torch.Tensor,
        weight: torch.Tensor,
        eps: float = 1e-6,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply standard RMSNorm with Gemma weight semantics on Ascend 950."""
        return torch_npu.npu_rms_norm(input, 1.0 + weight, eps)
