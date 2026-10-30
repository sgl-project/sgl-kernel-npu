"""Persistent attention-residual scoring and mixing, including hidden tails."""

import math

import torch
import triton
import triton.language as tl

from ..utils.triton_utils import get_device_properties


def validate_mix_inputs(
    prefix_sum, bank, num_valid_blocks, combined_weight, variance_epsilon
):
    """Validate Triton input metadata without inspecting device tensor values."""
    supported_dtypes = (torch.float16, torch.bfloat16, torch.float32)
    for name, tensor, ndim in (
        ("prefix_sum", prefix_sum, 2),
        ("bank", bank, 3),
        ("combined_weight", combined_weight, 1),
    ):
        assert isinstance(tensor, torch.Tensor), f"{name} must be a torch.Tensor"
        assert tensor.layout == torch.strided, f"{name} must have strided layout"
        assert (
            tensor.ndim == ndim
        ), f"{name} must have {ndim} dimensions, got shape {tuple(tensor.shape)}"
        assert (
            tensor.dtype in supported_dtypes
        ), f"{name} must have float16, bfloat16 or float32 dtype, got {tensor.dtype}"

    num_tokens, hidden_size = prefix_sum.shape
    assert hidden_size > 0, f"hidden size must be positive, got {hidden_size}"
    assert bank.shape[0] == num_tokens and bank.shape[2] == hidden_size, (
        "bank must match prefix_sum's token count and hidden size, "
        f"got prefix_sum={tuple(prefix_sum.shape)}, bank={tuple(bank.shape)}"
    )
    assert combined_weight.shape[0] == hidden_size, (
        "combined_weight must have one entry per hidden feature, "
        f"got shape {tuple(combined_weight.shape)}, hidden size={hidden_size}"
    )
    assert prefix_sum.device == bank.device == combined_weight.device, (
        "prefix_sum, bank and combined_weight must share a device, "
        f"got {prefix_sum.device}, {bank.device} and {combined_weight.device}"
    )
    assert isinstance(num_valid_blocks, int) and not isinstance(
        num_valid_blocks, bool
    ), f"num_valid_blocks must be a Python integer, got {num_valid_blocks!r}"
    if not 0 <= num_valid_blocks <= bank.shape[1]:
        raise ValueError("num_valid_blocks must fit within the residual bank")

    assert isinstance(variance_epsilon, (int, float)) and not isinstance(
        variance_epsilon, bool
    ), "variance_epsilon must be a Python int or float"
    assert (
        math.isfinite(variance_epsilon) and variance_epsilon > 0
    ), f"variance_epsilon must be finite and positive, got {variance_epsilon!r}"


@triton.jit(do_not_specialize=["N", "B"])
def _attn_res_mix_fused_kernel(
    prefix_ptr,
    bank_ptr,
    cw_ptr,
    out_ptr,
    N,
    B,
    stride_pm,
    stride_bm,
    stride_bb,
    stride_om,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    EPS: tl.constexpr,
    NUM_CORES: tl.constexpr,
    NB: tl.constexpr,
):
    block_size = (N - 1) // NUM_CORES + 1
    pid = tl.program_id(0)
    token_start = pid * block_size
    if token_start >= N:
        return
    token_end = tl.minimum(token_start + block_size, N)

    hidden_offsets = tl.arange(0, BLOCK_H)
    hidden_mask = hidden_offsets < H
    row_offsets = tl.arange(0, NB)
    combined_weight = tl.load(cw_ptr + hidden_offsets, mask=hidden_mask, other=0.0).to(
        tl.float32
    )

    for token in range(token_start, token_end):
        scores = tl.full([NB], -float("inf"), dtype=tl.float32)
        for row in range(B + 1):
            if row < B:
                value = tl.load(
                    bank_ptr + token * stride_bm + row * stride_bb + hidden_offsets,
                    mask=hidden_mask,
                    other=0.0,
                ).to(tl.float32)
            else:
                value = tl.load(
                    prefix_ptr + token * stride_pm + hidden_offsets,
                    mask=hidden_mask,
                    other=0.0,
                ).to(tl.float32)
            inverse_rms = tl.rsqrt(tl.sum(value * value) / H + EPS)
            score = tl.sum(value * inverse_rms * combined_weight)
            scores = tl.where(row_offsets == row, score, scores)

        scores_max = tl.max(scores)
        exp_scores = tl.exp(scores - scores_max)
        probabilities = exp_scores / tl.sum(exp_scores)

        output = tl.zeros([BLOCK_H], dtype=tl.float32)
        for row in range(B + 1):
            if row < B:
                value = tl.load(
                    bank_ptr + token * stride_bm + row * stride_bb + hidden_offsets,
                    mask=hidden_mask,
                    other=0.0,
                ).to(tl.float32)
            else:
                value = tl.load(
                    prefix_ptr + token * stride_pm + hidden_offsets,
                    mask=hidden_mask,
                    other=0.0,
                ).to(tl.float32)
            probability = tl.sum(tl.where(row_offsets == row, probabilities, 0.0))
            output += probability * value

        tl.store(
            out_ptr + token * stride_om + hidden_offsets,
            output.to(out_ptr.dtype.element_ty),
            mask=hidden_mask,
        )


def attn_res_mix_fused(
    prefix_sum: torch.Tensor,
    bank: torch.Tensor,
    num_valid_blocks: int,
    combined_weight: torch.Tensor,
    variance_epsilon: float,
) -> torch.Tensor:
    """Ascend Kimi-K3 attention-residual score and combine pipeline.

    Keep scoring, softmax, and mixing in one persistent vector-core kernel to
    avoid materializing the score matrix and launching a second kernel.  N and
    B stay dynamic so prefill/decode shapes reuse the same compilation.
    """
    validate_mix_inputs(
        prefix_sum, bank, num_valid_blocks, combined_weight, variance_epsilon
    )
    num_tokens, hidden_size = prefix_sum.shape
    if num_tokens == 0:
        return prefix_sum
    assert (
        prefix_sum.device.type == "npu"
    ), f"Triton attention-residual mixing requires NPU tensors, got {prefix_sum.device}"
    assert prefix_sum.stride(-1) == 1, (
        "prefix_sum must have unit hidden stride, " f"got strides {prefix_sum.stride()}"
    )
    assert (
        bank.stride(-1) == 1
    ), f"bank must have unit hidden stride, got strides {bank.stride()}"
    assert combined_weight.stride(0) == 1, (
        "combined_weight must have unit stride, "
        f"got strides {combined_weight.stride()}"
    )

    output = torch.empty_like(prefix_sum)
    _, num_vector_cores = get_device_properties()
    _attn_res_mix_fused_kernel[(num_vector_cores,)](
        prefix_sum,
        bank,
        combined_weight,
        output,
        num_tokens,
        num_valid_blocks,
        prefix_sum.stride(0),
        bank.stride(0),
        bank.stride(1),
        output.stride(0),
        H=hidden_size,
        BLOCK_H=triton.next_power_of_2(hidden_size),
        EPS=variance_epsilon,
        NUM_CORES=num_vector_cores,
        NB=triton.next_power_of_2(num_valid_blocks + 1),
        multibuffer=True,
    )
    return output
