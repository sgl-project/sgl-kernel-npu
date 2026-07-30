from typing import Optional, Tuple

import torch
import triton
import triton.language as tl
from sgl_kernel_npu.utils.triton_utils import get_device_properties


@triton.jit
def _gemma_rms_norm_kernel(
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


def launch_gemma_rms_norm(
    input: torch.Tensor,
    weight: torch.Tensor,
    residual: Optional[torch.Tensor],
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Launch the Triton Gemma RMSNorm implementation selected at build time."""
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

    _gemma_rms_norm_kernel[grid](
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
