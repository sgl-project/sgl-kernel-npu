"""Top-k/top-p renormalization using fused filtering and RN restoration.

The only implementation is the packed-key Triton path for NPU FP32
probabilities. There is no environment-controlled dispatch or scatter fallback.
"""

from typing import Union

import torch


def _prepare_probs(probs: torch.Tensor) -> torch.Tensor:
    if probs.ndim != 2:
        raise ValueError("probs must have shape [batch, vocab]")
    if probs.dtype != torch.float32:
        raise ValueError("probs must use torch.float32")
    if probs.device.type != "npu":
        raise ValueError("probs must be on an NPU")
    if not 1 <= probs.shape[1] <= 2**23:
        raise ValueError("probs must have vocab_size in [1, 2**23] for exact FP32 keys")
    return probs.contiguous()


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
    """Keep top-k tokens and renormalize NPU FP32 probabilities with RN division."""
    probs = _prepare_probs(probs)
    top_ks = _as_batch_threshold(top_ks, probs, torch.long, "top_ks").clamp(
        min=1, max=probs.shape[1]
    )
    if probs.shape[0] == 0:
        return torch.empty_like(probs)
    from .probability_triton import filter_and_renorm

    sorted_probs, sorted_indices = probs.sort(dim=-1, descending=True)
    return filter_and_renorm(probs, sorted_probs, sorted_indices, top_ks)


def top_p_renorm_prob(
    probs: torch.Tensor,
    top_ps: Union[torch.Tensor, float],
) -> torch.Tensor:
    """Keep the native sorted nucleus and renormalize NPU FP32 probabilities.

    Selection uses inclusive cumsum minus the current value, preserving the
    original rounding and tie behavior. Both endpoints of top_p are supported.
    """
    probs = _prepare_probs(probs)
    top_ps = _as_batch_threshold(top_ps, probs, probs.dtype, "top_ps").clamp(
        min=0.0, max=1.0
    )
    if probs.shape[0] == 0:
        return torch.empty_like(probs)
    from .probability_triton import filter_and_renorm

    sorted_probs, sorted_indices = probs.sort(dim=-1, descending=True)
    cumulative_probs = sorted_probs.cumsum(dim=-1)
    return filter_and_renorm(
        probs, sorted_probs, sorted_indices, top_ps, cumulative_probs
    )
