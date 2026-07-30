from typing import Callable, Dict, Optional, Tuple

import torch
import torch_npu
import triton
import triton.language as tl
from sgl_kernel_npu.utils.npu_device import NpuDeviceFamily, get_npu_device_family
from sgl_kernel_npu.utils.triton_utils import get_device_properties


@triton.jit
def _a5_gemma_rms_norm_kernel(
    hidden_state_ptr,
    hidden_state_stride_bs,
    weight_ptr,
    residual_ptr,
    add_output_ptr,
    norm_output_ptr,
    variance_epsilon,
    batch,
    dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
):
    core_id = tl.program_id(0)
    core_num = tl.num_programs(0)
    batch_per_core = tl.cdiv(batch, core_num)
    start_batch = core_id * batch_per_core
    end_batch = tl.minimum(start_batch + batch_per_core, batch)
    offset_d = tl.arange(0, BLOCK_SIZE)
    mask_d = offset_d < dim

    for row_start in tl.range(start_batch, end_batch, BLOCK_M):
        offset_row = row_start + tl.arange(0, BLOCK_M)
        offset_hidden = offset_row[:, None] * hidden_state_stride_bs + offset_d[None, :]
        mask_bs = (offset_row < batch)[:, None] & mask_d[None, :]

        hidden_state = tl.load(
            hidden_state_ptr + offset_hidden, mask=mask_bs, other=0.0
        )
        if HAS_RESIDUAL:
            residual = tl.load(residual_ptr + offset_hidden, mask=mask_bs, other=0.0)
            add_output = hidden_state + residual
            tl.store(add_output_ptr + offset_hidden, add_output, mask=mask_bs)
        else:
            add_output = hidden_state

        add_output_fp32 = add_output.to(tl.float32)
        weight = tl.load(weight_ptr + offset_d, mask=mask_d, other=0.0).to(tl.float32)
        variance = tl.sum(add_output_fp32 * add_output_fp32, axis=-1) / dim
        norm_output = add_output_fp32 * tl.rsqrt(variance[:, None] + variance_epsilon)
        norm_output = norm_output * (weight + 1.0)
        tl.store(
            norm_output_ptr + offset_hidden,
            norm_output.to(hidden_state.dtype),
            mask=mask_bs,
        )


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


def _launch_a5_triton(
    input: torch.Tensor,
    weight: torch.Tensor,
    residual: Optional[torch.Tensor],
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    original_shape = input.shape
    if input.numel() == 0:
        residual_sum = input if residual is None else input + residual
        return torch.empty_like(input), residual_sum

    dim = input.shape[-1]
    input_2d = input.contiguous().view(-1, dim)
    weight = weight.contiguous()
    batch = input_2d.shape[0]
    block_size = triton.next_power_of_2(dim)
    block_m = 1 if block_size > 4096 else min(2, batch)

    if residual is None:
        residual_2d = input_2d
        residual_sum_2d = input_2d
        residual_sum = input
    else:
        residual_2d = residual.contiguous().view(-1, dim)
        residual_sum_2d = torch.empty_like(input_2d)
        residual_sum = residual_sum_2d.view(original_shape)

    norm_output_2d = torch.empty_like(input_2d)
    norm_output = norm_output_2d.view(original_shape)
    _, num_vectorcore = get_device_properties()
    grid = (min(num_vectorcore, batch),)

    _a5_gemma_rms_norm_kernel[grid](
        input_2d,
        input_2d.stride(0),
        weight,
        residual_2d,
        residual_sum_2d,
        norm_output_2d,
        eps,
        batch,
        dim,
        block_size,
        block_m,
        HAS_RESIDUAL=residual is not None,
    )
    return norm_output, residual_sum


def _native_gemma_rms_norm(
    input: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    return torch_npu.npu_gemma_rms_norm(input, weight, eps)[0]


def _native_add_gemma_rms_norm(
    input: torch.Tensor,
    weight: torch.Tensor,
    residual: torch.Tensor,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    norm_output, _, residual_sum = torch_npu.npu_add_rms_norm(
        residual, input, 1.0 + weight, eps
    )
    return norm_output, residual_sum


def _triton_gemma_rms_norm(
    input: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    norm_output, _ = _launch_a5_triton(input, weight, None, eps)
    return norm_output


def _triton_add_gemma_rms_norm(
    input: torch.Tensor,
    weight: torch.Tensor,
    residual: torch.Tensor,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return _launch_a5_triton(input, weight, residual, eps)


def _fallback_gemma_rms_norm(
    input: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    return torch_npu.npu_rms_norm(input, 1.0 + weight, eps)[0]


def _fallback_add_gemma_rms_norm(
    input: torch.Tensor,
    weight: torch.Tensor,
    residual: torch.Tensor,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    residual_sum = input + residual
    norm_output = torch_npu.npu_rms_norm(residual_sum, 1.0 + weight, eps)[0]
    return norm_output, residual_sum


_GemmaRMSNormProvider = Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor]
_AddGemmaRMSNormProvider = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, float],
    Tuple[torch.Tensor, torch.Tensor],
]

_GEMMA_RMS_NORM_PROVIDERS: Dict[NpuDeviceFamily, _GemmaRMSNormProvider] = {
    NpuDeviceFamily.ASCEND_310P: _fallback_gemma_rms_norm,
    NpuDeviceFamily.A2: _native_gemma_rms_norm,
    NpuDeviceFamily.A3: _native_gemma_rms_norm,
    NpuDeviceFamily.A5: _triton_gemma_rms_norm,
    NpuDeviceFamily.UNKNOWN: _fallback_gemma_rms_norm,
}

_ADD_GEMMA_RMS_NORM_PROVIDERS: Dict[NpuDeviceFamily, _AddGemmaRMSNormProvider] = {
    NpuDeviceFamily.ASCEND_310P: _fallback_add_gemma_rms_norm,
    NpuDeviceFamily.A2: _native_add_gemma_rms_norm,
    NpuDeviceFamily.A3: _native_add_gemma_rms_norm,
    NpuDeviceFamily.A5: _triton_add_gemma_rms_norm,
    NpuDeviceFamily.UNKNOWN: _fallback_add_gemma_rms_norm,
}


def gemma_rms_norm(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Apply Gemma RMSNorm with checkpoint offset-weight semantics."""

    _validate_inputs(input, weight)
    if input.numel() == 0:
        return torch.empty_like(input)
    provider = _GEMMA_RMS_NORM_PROVIDERS[get_npu_device_family()]
    return provider(input.contiguous(), weight.contiguous(), eps)


def add_gemma_rms_norm(
    input: torch.Tensor,
    weight: torch.Tensor,
    residual: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Add a residual and apply Gemma RMSNorm without mutating the inputs."""

    _validate_inputs(input, weight, residual)
    if input.numel() == 0:
        residual_sum = input + residual
        return torch.empty_like(input), residual_sum
    provider = _ADD_GEMMA_RMS_NORM_PROVIDERS[get_npu_device_family()]
    return provider(input.contiguous(), weight.contiguous(), residual.contiguous(), eps)
