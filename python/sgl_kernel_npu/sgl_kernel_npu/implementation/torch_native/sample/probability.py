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


def _reference(probs, threshold, top_p):
    values, indices = probs.sort(dim=-1, descending=True)
    if top_p:
        values.masked_fill_(values.cumsum(-1) - values > threshold[:, None], 0.0)
    else:
        positions = torch.arange(probs.shape[1], device=probs.device)
        values.masked_fill_(positions >= threshold[:, None], 0.0)
    values.div_(values.sum(-1, keepdim=True).clamp_min_(1e-20))
    return torch.zeros_like(probs).scatter_(-1, indices, values)


def top_k_renorm_prob(probs, top_ks):
    probs = _prepare_probs(probs)
    top_ks = _as_batch_threshold(top_ks, probs, torch.long, "top_ks").clamp(
        1, probs.shape[1]
    )
    return _reference(probs, top_ks, False)


def top_p_renorm_prob(probs, top_ps):
    probs = _prepare_probs(probs)
    top_ps = _as_batch_threshold(top_ps, probs, probs.dtype, "top_ps").clamp(0, 1)
    return _reference(probs, top_ps, True)
