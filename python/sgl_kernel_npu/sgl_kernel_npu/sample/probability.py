from typing import Union

import torch


def _encode_keep_keys(sorted_probs: torch.Tensor, sorted_indices: torch.Tensor):
    """Pack each token ID and its nonzero keep bit into one exact FP32 integer.

    Contract: indices come from a full row sort; masked sorted_probs contains
    either that token's original probability or zero. This is not a general
    scatter replacement. No float probability values are packed into the key.
    """
    if sorted_indices.shape[-1] > 2**23:
        raise ValueError("Packed keep keys require vocab_size <= 2**23")
    keys = sorted_indices.to(torch.float32).mul_(2.0)
    keys.add_(sorted_probs.ne(0).to(torch.float32))
    return keys


def _write_from_ordered_keys(probs, ordered_keys, denominator):
    # Keys are now 2*j or 2*j+1 at vocabulary position j. Recover the mask
    # with contiguous elementwise operations; no gather/scatter/modulo needed.
    base_keys = (
        torch.arange(probs.shape[-1], device=probs.device, dtype=torch.float32)
        .mul_(2.0)
        .view(1, -1)
    )
    output = probs.masked_fill(ordered_keys == base_keys, 0.0)
    return output.div_(denominator)


def _renorm_from_sorted_probs(
    probs: torch.Tensor,
    sorted_probs: torch.Tensor,
    sorted_indices: torch.Tensor,
) -> torch.Tensor:
    # Keep the original reduction order and clamp to preserve normalization.
    denominator = sorted_probs.sum(dim=-1, keepdim=True).clamp_min_(1e-20)
    keys = _encode_keep_keys(sorted_probs, sorted_indices)
    # Sorting VALUES carries the keep bit into vocabulary order. The returned
    # sort indices are unused, so no full-vocabulary indexed read/write follows.
    ordered_keys = keys.sort(dim=-1).values
    return _write_from_ordered_keys(probs, ordered_keys, denominator)


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
