from typing import Optional, Tuple

import torch
from sgl_kernel_npu.norm._gemma_rmsnorm_triton import launch_gemma_rms_norm


def _validate_inputs(
    input: torch.Tensor,
    weight: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
) -> None:
    if input.ndim == 0:
        raise ValueError("input must have at least one dimension")
    if input.shape[-1] == 0:
        raise ValueError("input.shape[-1] must be greater than zero")
    if input.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("input must have dtype torch.float16 or torch.bfloat16")
    if input.device.type != "npu":
        raise ValueError("input must be on an NPU device")
    if weight.ndim != 1 or weight.numel() != input.shape[-1]:
        raise ValueError("weight must be one-dimensional and match input.shape[-1]")
    if weight.device != input.device:
        raise ValueError("weight must be on the same device as input")
    if weight.dtype != input.dtype:
        raise TypeError("weight must have the same dtype as input")
    if residual is not None:
        if residual.shape != input.shape:
            raise ValueError("residual must have the same shape as input")
        if residual.device != input.device:
            raise ValueError("residual must be on the same device as input")
        if residual.dtype != input.dtype:
            raise TypeError("residual must have the same dtype as input")


def gemma_rms_norm(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Apply the Ascend 950 Triton Gemma RMSNorm implementation."""
    _validate_inputs(input, weight)
    return launch_gemma_rms_norm(input, weight, None, eps)[0]


def add_gemma_rms_norm(
    input: torch.Tensor,
    weight: torch.Tensor,
    residual: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Add a residual and apply the Ascend 950 Triton Gemma RMSNorm."""
    _validate_inputs(input, weight, residual)
    return launch_gemma_rms_norm(input, weight, residual, eps)
