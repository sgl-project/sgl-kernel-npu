"""Independent PyTorch hyper-connection combine; no custom native operator calls."""

import torch
import torch.nn.functional as F


def _sigmoid(values):
    """Evaluate sigmoid as 1 / (1 + exp(-x)), the form the kernel expands to."""
    return 1.0 / (1.0 + torch.exp(-values))


def hc_combine(
    block_output,
    residual,
    normed_residual,
    inject_weight,
    hc_count,
    hidden_size,
):
    """Project injection logits and broadcast-add gated block output per branch.

    Intermediate arithmetic after ``F.linear`` stays in float32, rounding to
    BF16 once on the final combined output.
    """
    rows = residual.shape[0]
    if rows == 0:
        return torch.empty_like(residual)
    gate_logits = F.linear(normed_residual, inject_weight)
    gate = 2.0 * _sigmoid(gate_logits.float() / hc_count)
    injection = block_output.float().unsqueeze(1) * gate.unsqueeze(-1)
    combined = residual.reshape(rows, hc_count, hidden_size).float() + injection
    return combined.reshape(rows, hc_count * hidden_size).to(torch.bfloat16)
