import torch
import triton
import triton.language as tl
from sgl_kernel_npu.utils.triton_utils import get_device_properties


@triton.jit(do_not_specialize=["N", "B"])
def _mix_fused_kernel(
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

    hidden_offsets = tl.arange(0, H)
    row_offsets = tl.arange(0, NB)
    combined_weight = tl.load(cw_ptr + hidden_offsets).to(tl.float32)

    for token in range(token_start, token_end):
        scores = tl.full([NB], -float("inf"), dtype=tl.float32)
        for row in range(B + 1):
            if row < B:
                value = tl.load(
                    bank_ptr + token * stride_bm + row * stride_bb + hidden_offsets
                ).to(tl.float32)
            else:
                value = tl.load(prefix_ptr + token * stride_pm + hidden_offsets).to(
                    tl.float32
                )
            inverse_rms = tl.rsqrt(tl.sum(value * value) / H + EPS)
            score = tl.sum(value * inverse_rms * combined_weight)
            scores = tl.where(row_offsets == row, score, scores)

        scores_max = tl.max(scores)
        exp_scores = tl.exp(scores - scores_max)
        probabilities = exp_scores / tl.sum(exp_scores)

        output = tl.zeros([H], dtype=tl.float32)
        for row in range(B + 1):
            if row < B:
                value = tl.load(
                    bank_ptr + token * stride_bm + row * stride_bb + hidden_offsets
                ).to(tl.float32)
            else:
                value = tl.load(prefix_ptr + token * stride_pm + hidden_offsets).to(
                    tl.float32
                )
            probability = tl.sum(tl.where(row_offsets == row, probabilities, 0.0))
            output += probability * value

        tl.store(
            out_ptr + token * stride_om + hidden_offsets,
            output.to(out_ptr.dtype.element_ty),
        )


def mix_fused(
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
    num_tokens, hidden_size = prefix_sum.shape
    if num_tokens == 0:
        return prefix_sum
    if not 0 <= num_valid_blocks <= bank.shape[1]:
        raise ValueError("num_valid_blocks must fit within the residual bank")

    output = torch.empty_like(prefix_sum)
    _, num_vector_cores = get_device_properties()
    _mix_fused_kernel[(num_vector_cores,)](
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
        EPS=variance_epsilon,
        NUM_CORES=num_vector_cores,
        NB=triton.next_power_of_2(num_valid_blocks + 1),
        multibuffer=True,
    )
    return output
