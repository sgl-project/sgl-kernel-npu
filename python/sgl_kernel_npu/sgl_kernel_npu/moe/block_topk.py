"""Block-constrained MoE routing for Ascend NPU."""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import torch_npu


def block_topk_cann_hybrid(
    router_logits: torch.Tensor,
    correction_bias: torch.Tensor,
    block_size: int,
    expert_capacity: int,
    top_k: int,
    expert_tie_break: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Select token experts from a block-level expert capacity."""
    if router_logits.dim() != 2:
        raise ValueError(f"router_logits must be 2D, got {router_logits.dim()}D")
    if correction_bias.shape != (router_logits.shape[1],):
        raise ValueError(
            "correction_bias must have one value per expert, got "
            f"{tuple(correction_bias.shape)} for {router_logits.shape[1]} experts"
        )

    num_tokens, num_experts = router_logits.shape
    device = router_logits.device
    if num_tokens == 0:
        return (
            torch.empty((0, top_k), dtype=router_logits.dtype, device=device),
            torch.empty((0, top_k), dtype=torch.int32, device=device),
        )

    base_scores = torch.sigmoid(router_logits.float())
    routing_scores = base_scores + correction_bias.float()

    num_blocks = (num_tokens + block_size - 1) // block_size
    padded_num_tokens = num_blocks * block_size
    pad_tokens = padded_num_tokens - num_tokens
    if pad_tokens:
        base_scores = F.pad(base_scores, (0, 0, 0, pad_tokens), value=0.0)
        routing_scores = F.pad(
            routing_scores,
            (0, 0, 0, pad_tokens),
            value=float("-inf"),
        )

    routing_scores_blocked = routing_scores.view(num_blocks, block_size, num_experts)
    block_expert_scores = routing_scores_blocked.max(dim=1).values

    if expert_tie_break is None:
        expert_idx = torch.arange(num_experts, device=device, dtype=torch.float32)
        expert_tie_break = -expert_idx * 3e-7
    combined = block_expert_scores + expert_tie_break
    _, capacity_ids = combined.topk(expert_capacity, dim=-1)

    capacity_ids_per_token = capacity_ids.unsqueeze(1).expand(-1, block_size, -1)
    capacity_scores = routing_scores_blocked.gather(2, capacity_ids_per_token)
    capacity_scores = capacity_scores - capacity_ids_per_token.to(torch.float32) * 3e-7
    _, local_ids, _ = torch_npu.npu_moe_gating_top_k(
        capacity_scores.view(padded_num_tokens, expert_capacity),
        top_k,
        norm_type=0,
    )
    local_ids = local_ids.to(torch.int64).view(num_blocks, block_size, top_k)
    ids = capacity_ids_per_token.gather(2, local_ids).view(padded_num_tokens, top_k)

    selected_base_scores = base_scores.gather(1, ids)
    row_max = selected_base_scores.max(dim=-1, keepdim=True).values
    has_nonzero_score = row_max > 1e-30
    scaled = selected_base_scores / torch.where(
        has_nonzero_score, row_max, torch.ones_like(row_max)
    )
    normalized = scaled / scaled.sum(dim=-1, keepdim=True).clamp_min(1e-30)
    weights = torch.where(
        has_nonzero_score,
        normalized,
        torch.full_like(selected_base_scores, 1.0 / top_k),
    )

    return (
        weights[:num_tokens].to(router_logits.dtype),
        ids[:num_tokens].to(torch.int32),
    )
