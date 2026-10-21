"""Torch oracles copied from SGLang 7036f4b0ad for migrated CUDA cases.

Reference code is test-only and never imports or calls NPU implementations.
"""

from __future__ import annotations

from typing import Any

import torch


def ceil_align(n, a):
    return (n + a - 1) // a * a


class ScheduleVerifyLensTopk:

    @classmethod
    def torch(cls, *, confidence: torch.Tensor, budget: int, cfg: Any) -> torch.Tensor:
        return schedule_verify_lens_topk(confidence=confidence, budget=budget, cfg=cfg)


def compute_sort_survival(confidence: torch.Tensor) -> torch.Tensor:
    return torch.cumprod(confidence.to(torch.float32), dim=1)


def schedule_verify_lens_topk(
    *, confidence: torch.Tensor, budget: int, cfg: Any
) -> torch.Tensor:
    return schedule_verify_lens_topk_from_survival(
        survival_probs=compute_sort_survival(confidence), budget=budget, cfg=cfg
    )


def schedule_verify_lens_topk_from_survival(
    *, survival_probs: torch.Tensor, budget: int, cfg: Any
) -> torch.Tensor:
    num_requests, _gamma = survival_probs.shape
    max_len = cfg.resolved_max_verify_len()
    device = survival_probs.device
    selected_extra = torch.zeros(num_requests, dtype=torch.int64, device=device)
    if budget > 0:
        candidate_window = survival_probs[:, :max_len]
        num_candidates = candidate_window.numel()
        if num_candidates > 0:
            request_index = (
                torch.arange(num_requests, device=device)
                .view(num_requests, 1)
                .expand_as(candidate_window)
            )
            position_index = (
                torch.arange(candidate_window.shape[1], device=device)
                .view(1, candidate_window.shape[1])
                .expand_as(candidate_window)
            )
            valid = candidate_window >= cfg.survival_eps
            flat_prob = candidate_window.reshape(-1).to(torch.float64)
            flat_request = request_index.reshape(-1)
            flat_position = position_index.reshape(-1)
            flat_valid = valid.reshape(-1)
            order = _value_independent_descending_order(
                probs=flat_prob,
                positions=flat_position,
                requests=flat_request,
                valid=flat_valid,
            )
            take = min(int(budget), num_candidates)
            chosen = order[:take]
            chosen_requests = flat_request[chosen]
            chosen_valid = flat_valid[chosen].to(torch.int64)
            selected_extra.scatter_add_(0, chosen_requests, chosen_valid)
    min_len = torch.full(
        (num_requests,), cfg.min_verify_len, dtype=torch.int64, device=device
    )
    verify_lens = min_len + selected_extra
    lower_bound = max(cfg.min_verify_len, 1)
    verify_lens = torch.clamp(verify_lens, min=lower_bound, max=max_len)
    return verify_lens.to(torch.int32)


def _value_independent_descending_order(
    *,
    probs: torch.Tensor,
    positions: torch.Tensor,
    requests: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    masked_prob = torch.where(valid, probs, torch.full_like(probs, float("-inf")))
    num_candidates = masked_prob.numel()
    order = torch.arange(num_candidates, device=probs.device)
    order = order[torch.argsort(requests[order], stable=True)]
    order = order[torch.argsort(positions[order], stable=True)]
    order = order[torch.argsort(-masked_prob[order], stable=True)]
    return order
