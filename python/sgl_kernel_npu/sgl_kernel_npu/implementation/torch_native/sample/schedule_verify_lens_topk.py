"""Torch reference for DSpark verify-length scheduling."""

import torch


def schedule_verify_lens_topk_triton(*, confidence, budget, cfg):
    num_requests, _ = confidence.shape
    max_len = cfg.resolved_max_verify_len()
    survival = torch.cumprod(confidence.to(torch.float32), dim=1)[:, :max_len]
    extra = torch.zeros(num_requests, dtype=torch.int64, device=confidence.device)
    if budget > 0 and survival.numel() > 0:
        cols = survival.shape[1]
        probs = survival.reshape(-1)
        valid = probs >= cfg.survival_eps
        indices = torch.arange(probs.numel(), device=confidence.device)
        requests = indices // cols
        positions = indices % cols
        # Stable sorts establish request as the last tie-breaker, then position.
        order = indices[torch.argsort(positions, stable=True)]
        masked = torch.where(valid, probs, torch.full_like(probs, float("-inf")))
        order = order[torch.argsort(-masked[order], stable=True)]
        chosen = order[: min(int(budget), probs.numel())]
        extra.scatter_add_(0, requests[chosen], valid[chosen].to(torch.int64))
    return (
        (extra + cfg.min_verify_len)
        .clamp(min=max(cfg.min_verify_len, 1), max=max_len)
        .to(torch.int32)
    )
