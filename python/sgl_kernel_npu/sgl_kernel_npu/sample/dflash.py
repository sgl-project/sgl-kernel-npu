"""Fused greedy draft acceptance and bonus selection for Triton-Ascend."""

import torch
import triton
import triton.language as tl
from sgl_kernel_npu.utils.triton_utils import get_device_properties


@triton.jit
def _fused_correct_drafts_and_bonus_kernel(
    candidates_ptr,
    target_predict_ptr,
    num_correct_drafts_ptr,
    bonus_tokens_ptr,
    block_size,
    bs,
    BLOCK: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    for tile in range(tl.program_id(0), tl.cdiv(bs, BLOCK_ROWS), tl.num_programs(0)):
        b = (tile * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)).to(tl.int64)
        offs = tl.arange(0, BLOCK)
        in_row = (b[:, None] < bs) & (offs[None, :] < block_size - 1)
        drafts = tl.load(
            candidates_ptr + b[:, None] * block_size + 1 + offs[None, :],
            mask=in_row,
            other=-1,
        )
        targets = tl.load(
            target_predict_ptr + b[:, None] * block_size + offs[None, :],
            mask=in_row,
            other=-2,
        )
        eq = (drafts == targets) & in_row
        num_correct = tl.min(tl.where(eq, BLOCK, offs[None, :]), 1)
        bonus_token = tl.load(
            target_predict_ptr + b * block_size + num_correct, mask=b < bs, other=0
        )
        # Eight int32 results fill a 32-byte store block. Adjacent cores must
        # not share scalar-store padding when processing multiple rows.
        tl.store(num_correct_drafts_ptr + b, num_correct.to(tl.int32), mask=b < bs)
        tl.store(bonus_tokens_ptr + b, bonus_token.to(tl.int64), mask=b < bs)


def compute_dflash_correct_drafts_and_bonus(
    *,
    candidates: torch.Tensor,
    target_predict: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute DFlash accept lengths and bonus tokens (greedy verify rule).

    Args:
        candidates: Token ids proposed by the DFlash draft, including the current token.
            Shape: [bs, block_size]. candidates[:, 0] is the current token.
        target_predict: Token ids predicted by the target model for each position in the block.
            Shape: [bs, block_size]. target_predict[:, t] corresponds to argmax at position t.

    Returns:
        correct_len: int32 tensor [bs], number of accepted *draft* tokens (excluding current token and bonus token).
        bonus: int64 tensor [bs], the target-predicted token at index correct_len (the "bonus" token to append).

    Notes:
        Matches the reference implementation rule:
          accept while candidates[:, 1:] == target_predict[:, :-1] consecutively.
    """
    if candidates.ndim != 2:
        raise ValueError(f"candidates must be 2D, got shape={tuple(candidates.shape)}")
    if target_predict.shape != candidates.shape:
        raise ValueError(
            "target_predict must have the same shape as candidates. "
            f"candidates.shape={tuple(candidates.shape)}, target_predict.shape={tuple(target_predict.shape)}"
        )

    bs, block_size = candidates.shape
    if bs <= 0:
        raise ValueError(f"batch size must be positive, got {bs}.")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}.")

    assert candidates.device == target_predict.device
    num_correct_drafts = torch.empty(bs, dtype=torch.int32, device=candidates.device)
    bonus_tokens = torch.empty(bs, dtype=torch.int64, device=candidates.device)
    block_rows = 8
    grid = (min(get_device_properties()[1], triton.cdiv(bs, block_rows)),)
    _fused_correct_drafts_and_bonus_kernel[grid](
        candidates.contiguous(),
        target_predict.contiguous(),
        num_correct_drafts,
        bonus_tokens,
        block_size,
        bs,
        BLOCK=max(8, triton.next_power_of_2(max(block_size - 1, 1))),
        BLOCK_ROWS=block_rows,
    )
    return num_correct_drafts, bonus_tokens
