"""Independent FP32 attention-residual score, softmax and mix reference."""

import torch

from ...triton.hyper_conn.attn_residual_utils import validate_mix_inputs


def attn_residual_mix(
    prefix_sum,
    bank,
    num_valid_blocks,
    combined_weight,
    variance_epsilon,
):
    """Return the mixture of original residual rows, without an output norm."""
    validate_mix_inputs(
        prefix_sum, bank, num_valid_blocks, combined_weight, variance_epsilon
    )
    if prefix_sum.shape[0] == 0:
        return prefix_sum

    rows = torch.cat(
        (bank[:, :num_valid_blocks, :].float(), prefix_sum.float().unsqueeze(1)),
        dim=1,
    )
    inverse_rms = torch.rsqrt(
        rows.square().sum(dim=-1, keepdim=True) / prefix_sum.shape[1] + variance_epsilon
    )
    scores = (rows * inverse_rms * combined_weight.float()).sum(dim=-1)
    probabilities = torch.softmax(scores, dim=1)
    output = (probabilities.unsqueeze(-1) * rows).sum(dim=1)
    return output.to(prefix_sum.dtype)
