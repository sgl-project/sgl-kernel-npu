from __future__ import annotations

from typing import TYPE_CHECKING

from sgl_kernel_npu._build_target import GEMMA_RMS_NORM_PROVIDER

if TYPE_CHECKING:
    import torch


if GEMMA_RMS_NORM_PROVIDER == "native":
    import torch_npu

    def gemma_rms_norm(
        input: torch.Tensor,
        weight: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """Apply native Gemma RMSNorm on Ascend 910B/910C."""
        return torch_npu.npu_gemma_rms_norm(input, weight, eps)[0]

    def add_gemma_rms_norm(
        input: torch.Tensor,
        weight: torch.Tensor,
        residual: torch.Tensor,
        eps: float = 1e-6,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Add a residual and apply native RMSNorm with Gemma weight semantics."""
        norm_output, _, residual_sum = torch_npu.npu_add_rms_norm(
            residual, input, 1.0 + weight, eps
        )
        return norm_output, residual_sum

elif GEMMA_RMS_NORM_PROVIDER == "triton":
    from sgl_kernel_npu.norm._gemma_rmsnorm_triton import launch_gemma_rms_norm

    def gemma_rms_norm(
        input: torch.Tensor,
        weight: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """Apply the Ascend 950 Triton Gemma RMSNorm implementation."""
        return launch_gemma_rms_norm(input, weight, None, eps)[0]

    def add_gemma_rms_norm(
        input: torch.Tensor,
        weight: torch.Tensor,
        residual: torch.Tensor,
        eps: float = 1e-6,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Add a residual and apply the Ascend 950 Triton implementation."""
        return launch_gemma_rms_norm(input, weight, residual, eps)

else:
    raise RuntimeError(
        f"Unsupported Gemma RMSNorm provider: {GEMMA_RMS_NORM_PROVIDER!r}"
    )
