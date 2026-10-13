import torch
import triton
import triton.language as tl
from sgl_kernel_npu.utils.triton_utils import get_device_properties


@triton.jit
def swiglu_oai_kernel(
    hidden_states,
    gated_output,
    gemm1_alpha,
    gemm1_clamp_limit,
    output_dim: tl.constexpr,
    stride_row: tl.constexpr,
    stride_col: tl.constexpr,
    num_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Bound the live activation tensor independently of the intermediate width.
    for start in range(
        tl.program_id(0) * BLOCK_SIZE,
        num_elements,
        tl.num_programs(0) * BLOCK_SIZE,
    ):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_elements
        rows = offsets // output_dim
        cols = offsets % output_dim
        input_offsets = rows * stride_row + 2 * cols * stride_col
        gate = tl.load(hidden_states + input_offsets, mask=mask, other=0)
        up = tl.load(hidden_states + input_offsets + stride_col, mask=mask, other=0)

        gate = tl.where(gate > gemm1_clamp_limit, gemm1_clamp_limit, gate)
        up = tl.where(up > gemm1_clamp_limit, gemm1_clamp_limit, up)
        up = tl.where(up < -gemm1_clamp_limit, -gemm1_clamp_limit, up)
        sig = 1.0 / (1.0 + tl.exp(-gate * gemm1_alpha))
        glu = gate * sig
        out = (up + 1) * glu
        tl.store(gated_output + offsets, out, mask=mask)


def swiglu_oai_triton(
    hidden_states,
    dim,
    gemm1_alpha,
    gemm1_clamp_limit,
):
    if dim <= 0 or dim % 2:
        raise ValueError("dim must be a positive even integer")
    hidden_states = hidden_states.view(-1, dim)
    output_dim = dim // 2
    gated_output = torch.empty(
        (hidden_states.shape[0], output_dim),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    num_elements = gated_output.numel()
    if num_elements == 0:
        return gated_output

    _, num_vectorcore = get_device_properties()
    block_size = 1024
    grid = min(triton.cdiv(num_elements, block_size), num_vectorcore)
    swiglu_oai_kernel[(grid,)](
        hidden_states,
        gated_output,
        gemm1_alpha,
        gemm1_clamp_limit,
        output_dim,
        hidden_states.stride(0),
        hidden_states.stride(1),
        num_elements,
        block_size,
    )
    return gated_output


def swiglu_oai_native(layer, hidden_states):
    E, _, N = layer.w13_weight.size()
    gate_up = hidden_states.view(-1, N)
    alpha = layer.moe_runner_config.gemm1_alpha
    limit = layer.moe_runner_config.gemm1_clamp_limit
    gate, up = gate_up[..., ::2], gate_up[..., 1::2]
    gate = gate.clamp(min=None, max=limit)
    up = up.clamp(min=-limit, max=limit)
    glu = gate * torch.sigmoid(gate * alpha)
    gated_output = (up + 1) * glu
    return gated_output


def swiglu_oai(layer, hidden_states):
    return swiglu_oai_triton(
        hidden_states,
        layer.w13_weight.shape[2],
        layer.moe_runner_config.gemm1_alpha,
        layer.moe_runner_config.gemm1_clamp_limit,
    )
