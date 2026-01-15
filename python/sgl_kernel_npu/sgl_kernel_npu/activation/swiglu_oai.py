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
    BLOCK_SIZE: tl.constexpr,
    MINIBLOCK_SIZE: tl.constexpr,
    BS: tl.constexpr,
):
    i_block = tl.program_id(0)

    for i_miniblock in range(0, BLOCK_SIZE, MINIBLOCK_SIZE):
        offset_bs = i_block * BLOCK_SIZE + i_miniblock + tl.arange(0, MINIBLOCK_SIZE)
        mask_bs = offset_bs < BS

        offset_gate = tl.arange(0, output_dim) * 2
        offset_up = tl.arange(0, output_dim) * 2 + 1

        gate = tl.load(
            hidden_states + offset_bs[:, None] * output_dim * 2 + offset_gate[None, :],
            mask=mask_bs[:, None],
        )
        up = tl.load(
            hidden_states + offset_bs[:, None] * output_dim * 2 + offset_up[None, :],
            mask=mask_bs[:, None],
        )

        gate = tl.where(gate > gemm1_clamp_limit, gemm1_clamp_limit, gate)
        up = tl.where(up > gemm1_clamp_limit, gemm1_clamp_limit, up)
        up = tl.where(up < -gemm1_clamp_limit, -gemm1_clamp_limit, up)
        sig = 1.0 / (1.0 + tl.exp(-gate * gemm1_alpha))
        glu = gate * sig
        out = (up + 1) * glu

        tl.store(
            gated_output
            + offset_bs[:, None] * output_dim
            + tl.arange(0, output_dim)[None, :],
            out,
            mask=mask_bs[:, None],
        )


def swiglu_oai_triton(
    hidden_states,
    dim,
    gemm1_alpha,
    gemm1_clamp_limit,
):
    hidden_states = hidden_states.view(-1, dim)
    BS = hidden_states.shape[0]
    output_dim = dim // 2
    gated_output = torch.empty(
        (BS, output_dim),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )

    kernel_num = get_device_properties()[0]
    MINIBLOCK_SIZE = 16
    BLOCK_SIZE = triton.cdiv(BS, MINIBLOCK_SIZE * kernel_num) * MINIBLOCK_SIZE
    BLOCK_NUM = triton.cdiv(BS, BLOCK_SIZE)

    swiglu_oai_kernel[(BLOCK_NUM,)](
        hidden_states,
        gated_output,
        gemm1_alpha,
        gemm1_clamp_limit,
        output_dim,
        BLOCK_SIZE,
        MINIBLOCK_SIZE,
        BS,
    )
    return gated_output


def swiglu_oai_native(layer, hidden_states):
    E, N, _ = layer.w13_weight.size()
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
        layer.w13_weight.shape[1],
        layer.moe_runner_config.gemm1_alpha,
        layer.moe_runner_config.gemm1_clamp_limit,
    )
