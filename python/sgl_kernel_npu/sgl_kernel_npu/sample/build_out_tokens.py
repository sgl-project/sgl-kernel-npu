import torch
import triton
import triton.language as tl
from sgl_kernel_npu.utils.triton_utils import get_device_properties


@triton.jit
def _build_out_tokens_kernel(
    draft,
    correct_len,
    bonus,
    out,
    N: tl.constexpr,
    GAMMA: tl.constexpr,
    DRAFT_ROW: tl.constexpr,
    DRAFT_COL: tl.constexpr,
    LEN_STRIDE: tl.constexpr,
    BONUS_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    for start in range(tl.program_id(0) * BLOCK, N, tl.num_programs(0) * BLOCK):
        offsets = start + tl.arange(0, BLOCK)
        valid = offsets < N
        row = offsets // (GAMMA + 1)
        col = offsets % (GAMMA + 1)
        length = tl.load(correct_len + row * LEN_STRIDE, mask=valid, other=0)
        token = tl.load(bonus + row * BONUS_STRIDE, mask=valid, other=0).to(tl.int64)
        value = tl.load(
            draft + row * DRAFT_ROW + col * DRAFT_COL,
            mask=valid & (col < GAMMA),
            other=0,
        ).to(tl.int64)
        # Bitwise selection preserves all int64 bits without the int64 select
        # lowering issue in the CANN 9.1 Triton compiler.
        select = -(col == length).to(tl.int64)
        value = (token & select) | (value & ~select)
        tl.store(out + offsets, value, mask=valid)


def build_out_tokens(
    *,
    draft_tokens: torch.Tensor,
    correct_len: torch.Tensor,
    bonus: torch.Tensor,
    verify_num_draft_tokens: int,
    gamma: int,
) -> torch.Tensor:
    """Build DSpark's full output buffer, including the uncommitted suffix.

    Inputs are int32/int64 NPU tensors of shapes [batch, gamma], [batch],
    and [batch]. Each correct_len must be in [0, gamma]. The int64 output
    copies the draft, zeroes the last column, then inserts bonus at correct_len.
    Consumers must use the acceptance lengths to identify valid output tokens.
    """
    if gamma < 0 or verify_num_draft_tokens != gamma + 1:
        raise ValueError("verify_num_draft_tokens must equal gamma + 1")
    if draft_tokens.ndim != 2 or draft_tokens.shape[1] != gamma:
        raise ValueError("draft_tokens must have shape [batch, gamma]")
    bs = draft_tokens.shape[0]
    if correct_len.shape != (bs,) or bonus.shape != (bs,):
        raise ValueError("correct_len and bonus must have shape [batch]")
    for tensor in (draft_tokens, correct_len, bonus):
        if tensor.dtype not in (torch.int32, torch.int64):
            raise ValueError("inputs must have dtype int32 or int64")
        if tensor.device != draft_tokens.device or tensor.device.type != "npu":
            raise ValueError("inputs must be on the same NPU device")
    out = torch.empty(
        (bs, verify_num_draft_tokens), dtype=torch.int64, device=draft_tokens.device
    )
    if bs == 0:
        return out
    _, vector_cores = get_device_properties()
    _build_out_tokens_kernel[(vector_cores,)](
        draft_tokens,
        correct_len,
        bonus,
        out,
        out.numel(),
        gamma,
        *draft_tokens.stride(),
        correct_len.stride(0),
        bonus.stride(0),
        BLOCK=256,
    )
    return out
