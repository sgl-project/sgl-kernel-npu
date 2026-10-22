"""Torch oracles copied from SGLang 7036f4b0ad for migrated CUDA cases.

Reference code is test-only and never imports or calls NPU implementations.
"""

from __future__ import annotations

from typing import Optional, Tuple

import msgspec
import torch


def ceil_align(n, a):
    return (n + a - 1) // a * a


class SoftmaxTemp:

    @classmethod
    def torch(
        cls, *, logits: torch.Tensor, temperatures: torch.Tensor, rows_per_request: int
    ) -> torch.Tensor:
        return softmax_temp(
            logits=logits, temperatures=temperatures, rows_per_request=rows_per_request
        )


def softmax_temp(
    *, logits: torch.Tensor, temperatures: torch.Tensor, rows_per_request: int
) -> torch.Tensor:
    num_rows = logits.shape[0]
    bs = num_rows // rows_per_request
    assert (
        bs * rows_per_request == num_rows
    ), f"num_rows {num_rows} not divisible by rows_per_request {rows_per_request}"
    temp_per_row = torch.repeat_interleave(
        temperatures.reshape(bs).to(torch.float32), rows_per_request, dim=0
    )
    scaled = logits.to(torch.float32) / temp_per_row[:, None]
    return torch.softmax(scaled, dim=-1)


class MixedAcceptSelectResult(msgspec.Struct):
    correct_len: torch.Tensor
    bonus: torch.Tensor
    cap_trim_lens: torch.Tensor


class SelectMixedAccept:

    @classmethod
    def torch(
        cls,
        *,
        greedy_mask: torch.Tensor,
        greedy_len: torch.Tensor,
        greedy_bonus: torch.Tensor,
        greedy_trim: torch.Tensor,
        sampling_len: torch.Tensor,
        sampling_bonus: torch.Tensor,
        sampling_trim: torch.Tensor,
    ) -> MixedAcceptSelectResult:
        return select_mixed_accept(
            greedy_mask=greedy_mask,
            greedy_len=greedy_len,
            greedy_bonus=greedy_bonus,
            greedy_trim=greedy_trim,
            sampling_len=sampling_len,
            sampling_bonus=sampling_bonus,
            sampling_trim=sampling_trim,
        )


def select_mixed_accept(
    *,
    greedy_mask: torch.Tensor,
    greedy_len: torch.Tensor,
    greedy_bonus: torch.Tensor,
    greedy_trim: torch.Tensor,
    sampling_len: torch.Tensor,
    sampling_bonus: torch.Tensor,
    sampling_trim: torch.Tensor,
) -> MixedAcceptSelectResult:
    correct_len = torch.where(
        greedy_mask, greedy_len.to(sampling_len.dtype), sampling_len
    )
    bonus = torch.where(greedy_mask, greedy_bonus, sampling_bonus)
    cap_trim_lens = torch.where(
        greedy_mask, greedy_trim.to(sampling_trim.dtype), sampling_trim
    )
    return MixedAcceptSelectResult(
        correct_len=correct_len, bonus=bonus, cap_trim_lens=cap_trim_lens
    )


class AcceptGreedy:

    @classmethod
    def torch(
        cls,
        *,
        candidates: torch.Tensor,
        target_logits: torch.Tensor,
        verify_num_draft_tokens: int,
        cutoff_verify_lens: Optional[torch.Tensor] = None,
        fused_argmax: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return accept_greedy(
            candidates=candidates,
            target_logits=target_logits,
            verify_num_draft_tokens=verify_num_draft_tokens,
            cutoff_verify_lens=cutoff_verify_lens,
            fused_argmax=fused_argmax,
        )


def accept_greedy(
    *,
    candidates: torch.Tensor,
    target_logits: torch.Tensor,
    verify_num_draft_tokens: int,
    cutoff_verify_lens: Optional[torch.Tensor] = None,
    fused_argmax: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bs = candidates.shape[0]
    target_predict = _row_argmax(target_logits, fused=fused_argmax).view(
        bs, verify_num_draft_tokens
    )
    correct_len, bonus = compute_dflash_correct_drafts_and_bonus(
        candidates=candidates, target_predict=target_predict
    )
    cap_trim_lens = torch.zeros_like(correct_len)
    if cutoff_verify_lens is not None:
        correct_len, cap_trim_lens = CapCorrectLen.torch(
            correct_len=correct_len, verify_lens=cutoff_verify_lens
        )
        row_ids = torch.arange(bs, device=target_predict.device)
        bonus = target_predict[row_ids, correct_len.to(torch.long)].to(torch.int64)
    return (correct_len, bonus, cap_trim_lens)


def _row_argmax(logits: torch.Tensor, fused: bool = False) -> torch.Tensor:
    if (
        fused
        and logits.is_cuda
        and (logits.dim() == 2)
        and (logits.dtype == torch.float32)
        and (logits.stride(1) == 1)
        and (
            logits.shape[0] <= 64
            or (logits.shape[0] <= 384 and logits.shape[1] >= 65536)
        )
        and (logits.shape[1] >= 4096)
    ):
        from sglang.kernels.ops.speculative.row_argmax import row_argmax

        return row_argmax(logits)
    return torch.argmax(logits, dim=-1)


class FinalizeAcceptLensResult(msgspec.Struct):
    commit_lens: torch.Tensor
    new_seq_lens: torch.Tensor
    cap_trim_lens: torch.Tensor


class FinalizeAcceptLens:

    @classmethod
    def torch(
        cls,
        *,
        correct_len: torch.Tensor,
        cap_trim_lens: torch.Tensor,
        prefix_lens: torch.Tensor,
    ) -> FinalizeAcceptLensResult:
        return finalize_accept_lens(
            correct_len=correct_len,
            cap_trim_lens=cap_trim_lens,
            prefix_lens=prefix_lens,
        )


def finalize_accept_lens(
    *, correct_len: torch.Tensor, cap_trim_lens: torch.Tensor, prefix_lens: torch.Tensor
) -> FinalizeAcceptLensResult:
    commit_lens = correct_len.to(torch.int32) + 1
    new_seq_lens = prefix_lens + commit_lens.to(prefix_lens.dtype)
    return FinalizeAcceptLensResult(
        commit_lens=commit_lens,
        new_seq_lens=new_seq_lens,
        cap_trim_lens=cap_trim_lens.to(torch.int32),
    )


class CapCorrectLen:

    @classmethod
    def torch(
        cls, *, correct_len: torch.Tensor, verify_lens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return cap_correct_len(correct_len=correct_len, verify_lens=verify_lens)


def cap_correct_len(
    *, correct_len: torch.Tensor, verify_lens: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    ell_r = (verify_lens.to(device=correct_len.device) - 1).to(correct_len.dtype)
    capped = torch.minimum(correct_len, ell_r)
    cap_trim_lens = correct_len - capped
    return (capped, cap_trim_lens)


def compute_dflash_correct_drafts_and_bonus(
    *, candidates: torch.Tensor, target_predict: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
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
            f"target_predict must have the same shape as candidates. candidates.shape={tuple(candidates.shape)}, target_predict.shape={tuple(target_predict.shape)}"
        )
    bs, block_size = candidates.shape
    if bs <= 0:
        raise ValueError(f"batch size must be positive, got {bs}.")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}.")
    matches = candidates[:, 1:] == target_predict[:, :-1]
    correct_len = matches.to(torch.int32).cumprod(dim=1).sum(dim=1)
    bonus = target_predict[torch.arange(bs, device=target_predict.device), correct_len]
    return (correct_len.to(torch.int32), bonus.to(torch.int64))
