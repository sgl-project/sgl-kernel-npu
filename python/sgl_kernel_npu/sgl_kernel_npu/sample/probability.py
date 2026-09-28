from typing import Union

import torch


def _renorm_from_sorted_probs(
    probs: torch.Tensor,
    sorted_probs: torch.Tensor,
    sorted_indices: torch.Tensor,
) -> torch.Tensor:
    sorted_probs.div_(sorted_probs.sum(dim=-1, keepdim=True).clamp_min_(1e-20))
    return torch.zeros_like(probs).scatter_(
        dim=-1, index=sorted_indices, src=sorted_probs
    )


def _as_batch_threshold(
    value: Union[torch.Tensor, int, float],
    probs: torch.Tensor,
    dtype: torch.dtype,
    name: str,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        return torch.full((probs.shape[0],), value, device=probs.device, dtype=dtype)
    if value.ndim == 0:
        return value.to(device=probs.device, dtype=dtype).expand(probs.shape[0])
    if value.ndim != 1 or value.shape[0] != probs.shape[0]:
        raise ValueError(f"{name} must be a scalar or have shape [batch]")
    return value.to(device=probs.device, dtype=dtype)


def top_k_renorm_prob(
    probs: torch.Tensor,
    top_ks: Union[torch.Tensor, int],
) -> torch.Tensor:
    """Keep each row's top-k probabilities and renormalize the row."""
    if probs.ndim != 2:
        raise ValueError("probs must have shape [batch, vocab]")

    vocab_size = probs.shape[-1]
    sorted_probs, sorted_indices = probs.sort(dim=-1, descending=True)
    top_ks = _as_batch_threshold(top_ks, probs, torch.long, "top_ks").clamp(
        min=1, max=vocab_size
    )
    positions = torch.arange(vocab_size, device=probs.device).view(1, -1)
    sorted_probs.masked_fill_(positions >= top_ks.view(-1, 1), 0.0)
    return _renorm_from_sorted_probs(probs, sorted_probs, sorted_indices)


def top_p_renorm_prob(
    probs: torch.Tensor,
    top_ps: Union[torch.Tensor, float],
) -> torch.Tensor:
    """Keep each row's nucleus probabilities and renormalize the row."""
    if probs.ndim != 2:
        raise ValueError("probs must have shape [batch, vocab]")

    sorted_probs, sorted_indices = probs.sort(dim=-1, descending=True)
    top_ps = _as_batch_threshold(top_ps, probs, probs.dtype, "top_ps").clamp(
        min=0.0, max=1.0
    )
    cumulative_probs = sorted_probs.cumsum(dim=-1)
    sorted_probs.masked_fill_(cumulative_probs - sorted_probs > top_ps.view(-1, 1), 0.0)
    return _renorm_from_sorted_probs(probs, sorted_probs, sorted_indices)
