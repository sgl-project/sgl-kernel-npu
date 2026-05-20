import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from sgl_kernel_npu.utils.triton_utils import get_device_properties


@triton.jit
def swiglu_silu_clamp_mul_kernel(
    hidden_states,
    gated_output,
    limit,
    output_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    MINIBLOCK_SIZE: tl.constexpr,
    BS: tl.constexpr,
):
    i_block = tl.program_id(0)

    for i_miniblock in range(0, BLOCK_SIZE, MINIBLOCK_SIZE):
        offset_bs = i_block * BLOCK_SIZE + i_miniblock + tl.arange(0, MINIBLOCK_SIZE)
        mask_bs = offset_bs < BS

        gate_offsets = (
            offset_bs[:, None] * (output_dim * 2) + tl.arange(0, output_dim)[None, :]
        )
        up_offsets = (
            offset_bs[:, None] * (output_dim * 2)
            + output_dim
            + tl.arange(0, output_dim)[None, :]
        )

        gate = tl.load(hidden_states + gate_offsets, mask=mask_bs[:, None])
        up = tl.load(hidden_states + up_offsets, mask=mask_bs[:, None])

        silu_gate = gate * tl.sigmoid(gate)
        silu_gate = min(silu_gate, limit)
        up_clamped = tl.clamp(up, -limit, limit)
        out = silu_gate * up_clamped

        tl.store(
            gated_output
            + offset_bs[:, None] * output_dim
            + tl.arange(0, output_dim)[None, :],
            out,
            mask=mask_bs[:, None],
        )


def swiglu_silu_clamp_mul_triton(x: torch.Tensor, limit: float = 7.0) -> torch.Tensor:

    original_shape = x.shape
    dim = original_shape[-1]
    hidden_states = x.view(-1, dim)

    BS = hidden_states.shape[0]
    output_dim = dim // 2

    gated_output = torch.empty(
        (*original_shape[:-1], output_dim),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )

    kernel_num = get_device_properties()[0]
    MINIBLOCK_SIZE = 4
    BLOCK_SIZE = triton.cdiv(BS, MINIBLOCK_SIZE * kernel_num) * MINIBLOCK_SIZE
    BLOCK_NUM = triton.cdiv(BS, BLOCK_SIZE)

    swiglu_silu_clamp_mul_kernel[(BLOCK_NUM,)](
        hidden_states,
        gated_output,
        limit,
        output_dim,
        BLOCK_SIZE,
        MINIBLOCK_SIZE,
        BS,
    )

    return gated_output


def swiglu_silu_clamp_mul_native(x: torch.Tensor, limit: float = 7.0) -> torch.Tensor:
    """Out-variant of swiglustep activation.

    Writes into `out`:
      silu(x[:d]).clamp(max=limit) * x[d:].clamp(-limit, limit)
    """
    gate, up = x.chunk(2, dim=-1)
    gate = F.silu(gate)
    gate = gate.clamp(max=limit)
    up = up.clamp(min=-limit, max=limit)
    out = gate * up
    return out


def swiglu_silu_clamp_mul(hidden_states, limit: float = 7.0) -> torch.Tensor:
    return swiglu_silu_clamp_mul_native(
        hidden_states,
        limit,
    )
