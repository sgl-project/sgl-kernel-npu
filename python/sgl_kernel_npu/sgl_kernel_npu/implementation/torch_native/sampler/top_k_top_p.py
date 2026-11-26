"""PyTorch reference for the CANN top-k/top-p logits filter."""

import torch


def top_k_top_p(logits, p=None, k=None):
    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch, vocabulary]")
    if logits.numel() == 0:
        return logits.clone()

    values, indices = logits.sort(dim=-1, stable=True)
    if k is not None:
        cutoff = values.gather(-1, logits.size(-1) - k.long().unsqueeze(-1))
        values = values.masked_fill(values < cutoff, -torch.inf)

    if p is not None:
        cumulative = values.float().softmax(dim=-1).cumsum(dim=-1)
        discard = cumulative <= (1.0 - p.float()).unsqueeze(-1)
        # Retain the highest logit even when p is zero.
        discard[:, -1] = False
        values = values.masked_fill(discard, -torch.inf)

    return torch.empty_like(logits).scatter(-1, indices, values)
