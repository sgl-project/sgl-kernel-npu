import torch
import triton
import triton.language as tl

_BLOCK_H = 32
_MAX_ROWS = 16


@triton.jit
def _score_kernel(
    prefix_ptr,
    bank_ptr,
    cw_ptr,
    scores_ptr,
    NVB,
    eps,
    stride_pm,
    stride_bm,
    stride_bb,
    stride_sm,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_t = tl.program_id(0)
    row = tl.program_id(1)
    if row > NVB:
        return
    sumsq = 0.0
    dot = 0.0
    for h_start in tl.static_range(0, H, BLOCK_H):
        offsets = h_start + tl.arange(0, BLOCK_H)
        if row < NVB:
            value = tl.load(
                bank_ptr + pid_t * stride_bm + row * stride_bb + offsets
            ).to(tl.float32)
        else:
            value = tl.load(prefix_ptr + pid_t * stride_pm + offsets).to(tl.float32)
        weight = tl.load(cw_ptr + offsets)
        sumsq += tl.sum(value * value)
        dot += tl.sum(value * weight)
    rrms = 1.0 / tl.sqrt(sumsq / H + eps)
    tl.store(scores_ptr + pid_t * stride_sm + row, dot * rrms)


@triton.jit
def _combine_kernel(
    prefix_ptr,
    bank_ptr,
    scores_ptr,
    out_ptr,
    NVB,
    stride_pm,
    stride_bm,
    stride_bb,
    stride_sm,
    stride_om,
    BLOCK_H: tl.constexpr,
    MAX_ROWS: tl.constexpr,
):
    pid_t = tl.program_id(0)
    offsets_h = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    offsets_b = tl.arange(0, MAX_ROWS)
    mask_b = offsets_b <= NVB
    logits = tl.load(
        scores_ptr + pid_t * stride_sm + offsets_b,
        mask=mask_b,
        other=float("-inf"),
    )
    logits_max = tl.max(logits, axis=0)
    probs = tl.where(mask_b, tl.exp(logits - logits_max), 0.0)
    probs /= tl.sum(probs, axis=0)

    acc = tl.zeros([BLOCK_H], tl.float32)
    for row in range(0, NVB + 1):
        if row < NVB:
            value = tl.load(
                bank_ptr + pid_t * stride_bm + row * stride_bb + offsets_h
            ).to(tl.float32)
        else:
            value = tl.load(prefix_ptr + pid_t * stride_pm + offsets_h).to(tl.float32)
        probability = tl.sum(tl.where(offsets_b == row, probs, 0.0), axis=0)
        acc += probability * value
    tl.store(out_ptr + pid_t * stride_om + offsets_h, acc.to(out_ptr.dtype.element_ty))


def mix_fused(
    prefix_sum: torch.Tensor,
    bank: torch.Tensor,
    num_valid_blocks: int,
    combined_weight: torch.Tensor,
    variance_epsilon: float,
) -> torch.Tensor:
    """Ascend Kimi-K3 attention-residual score and combine pipeline."""
    num_tokens, hidden_size = prefix_sum.shape
    if num_tokens == 0:
        return prefix_sum
    if hidden_size % _BLOCK_H:
        raise ValueError(f"hidden size {hidden_size} must be divisible by {_BLOCK_H}")

    scores = torch.empty(
        (num_tokens, _MAX_ROWS), dtype=torch.float32, device=prefix_sum.device
    )
    _score_kernel[(num_tokens, num_valid_blocks + 1)](
        prefix_sum,
        bank,
        combined_weight,
        scores,
        num_valid_blocks,
        variance_epsilon,
        prefix_sum.stride(0),
        bank.stride(0),
        bank.stride(1),
        scores.stride(0),
        H=hidden_size,
        BLOCK_H=_BLOCK_H,
        num_warps=4,
    )

    out = torch.empty_like(prefix_sum)
    _combine_kernel[(num_tokens, hidden_size // _BLOCK_H)](
        prefix_sum,
        bank,
        scores,
        out,
        num_valid_blocks,
        prefix_sum.stride(0),
        bank.stride(0),
        bank.stride(1),
        scores.stride(0),
        out.stride(0),
        BLOCK_H=_BLOCK_H,
        MAX_ROWS=_MAX_ROWS,
        num_warps=4,
    )
    return out
