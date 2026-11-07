"""Independent PyTorch hyper-connection branch mix; no custom native operator calls."""

import torch
import torch.nn.functional as F


def _sigmoid(values):
    """Evaluate sigmoid as 1 / (1 + exp(-x)), the form the kernel expands to.

    Both tails saturate correctly: exp(-x) overflowing to inf gives 0, and
    underflowing to 0 gives 1.
    """
    return 1.0 / (1.0 + torch.exp(-values))


def hc_mix(normed, down_weight, up_weight, hc_count, hidden_size):
    """Project, activate and gate-average hyper-connection branches into one row.

    Intermediate arithmetic inside each vector stage stays in float32, rounding
    to BF16 only at global-memory boundaries (``activated`` written before the
    ``up_weight`` projection, and the final ``mixed`` output).
    """
    rows = normed.shape[0]
    if rows == 0:
        return normed.new_empty((0, hidden_size))
    low_rank = F.linear(normed, down_weight)
    scaled = low_rank.float() / hc_count
    activated = (scaled * _sigmoid(scaled)).to(torch.bfloat16)
    gate_logits = F.linear(activated, up_weight).reshape(rows, hc_count, hidden_size)
    gate = _sigmoid(gate_logits.float())
    product = gate * normed.reshape(rows, hc_count, hidden_size).float()
    mixed = product.sum(dim=1) / hc_count
    return mixed.to(torch.bfloat16)
