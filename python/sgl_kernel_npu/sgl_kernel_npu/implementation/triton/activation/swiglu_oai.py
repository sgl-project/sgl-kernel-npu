import torch
import triton
import triton.language as tl
from sgl_kernel_npu.implementation.triton.utils.triton_utils import (
    get_device_properties,
)

# Bound each gate/up vector to 4 KiB even for FP32, independently of row width.
_TILE_SIZE = 1024


@triton.jit
def swiglu_oai_kernel(
    hidden_states,
    gated_output,
    gemm1_alpha,
    gemm1_clamp_limit,
    num_elements,
    BLOCK_SIZE: tl.constexpr,
):
    start = tl.program_id(0).to(tl.int64) * BLOCK_SIZE
    stride = tl.num_programs(0) * BLOCK_SIZE
    lanes = tl.arange(0, BLOCK_SIZE)

    for base in tl.range(start, num_elements, stride):
        offsets = base + lanes
        valid = offsets < num_elements
        gate = tl.load(hidden_states + 2 * offsets, mask=valid, other=0)
        up = tl.load(hidden_states + 2 * offsets + 1, mask=valid, other=0)

        gate = tl.where(gate > gemm1_clamp_limit, gemm1_clamp_limit, gate)
        up = tl.where(up > gemm1_clamp_limit, gemm1_clamp_limit, up)
        up = tl.where(up < -gemm1_clamp_limit, -gemm1_clamp_limit, up)
        sig = 1.0 / (1.0 + tl.exp(-gate * gemm1_alpha))
        glu = gate * sig
        out = (up + 1) * glu

        tl.store(gated_output + offsets, out, mask=valid)


def swiglu_oai(
    hidden_states,
    dim,
    gemm1_alpha,
    gemm1_clamp_limit,
):
    hidden_states = hidden_states.view(-1, dim)
    rows = hidden_states.shape[0]
    output_dim = dim // 2
    gated_output = torch.empty(
        (rows, output_dim),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )

    num_elements = gated_output.numel()
    if num_elements == 0:
        return gated_output
    _, num_vector_cores = get_device_properties()
    grid = (min(triton.cdiv(num_elements, _TILE_SIZE), num_vector_cores),)

    swiglu_oai_kernel[grid](
        hidden_states,
        gated_output,
        gemm1_alpha,
        gemm1_clamp_limit,
        num_elements,
        _TILE_SIZE,
        multibuffer=False,
    )
    return gated_output
