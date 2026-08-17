import torch
import triton
import triton.language as tl


@triton.jit
def _select_global_top1_kernel(
    candidates_ptr,
    output_ptr,
    TP_SIZE: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_TP: tl.constexpr,
):
    row = tl.program_id(0)
    rank_offsets = tl.arange(0, BLOCK_TP)
    rank_mask = rank_offsets < TP_SIZE
    row_base = row * TP_SIZE * 2
    values = tl.load(
        candidates_ptr + row_base + rank_offsets * 2,
        mask=rank_mask,
        other=float("-inf"),
    )
    token_ids = tl.load(
        candidates_ptr + row_base + rank_offsets * 2 + 1,
        mask=rank_mask,
        other=VOCAB_SIZE,
    ).to(tl.int32)
    best_value = tl.max(values, axis=0)
    best_token = tl.min(
        tl.where(values == best_value, token_ids, VOCAB_SIZE), axis=0
    )
    tl.store(output_ptr + row, best_token.to(tl.int64))


def select_global_top1_npu(
    candidates: torch.Tensor, *, vocab_size: int
) -> torch.Tensor:
    batch_size, tp_size, pair_size = candidates.shape
    assert pair_size == 2 and candidates.is_contiguous()
    output = torch.empty(batch_size, dtype=torch.long, device=candidates.device)
    _select_global_top1_kernel[(batch_size,)](
        candidates,
        output,
        TP_SIZE=tp_size,
        VOCAB_SIZE=vocab_size,
        BLOCK_TP=triton.next_power_of_2(tp_size),
    )
    return output
