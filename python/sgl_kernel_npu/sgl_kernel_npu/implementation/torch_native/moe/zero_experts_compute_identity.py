import torch


def zero_experts_compute_identity(
    expert_indices,
    expert_scales,
    num_experts,
    zero_expert_type,
    hidden_states,
    identity_mask_value=0,
):
    mask = expert_indices >= num_experts
    total = expert_scales.masked_fill(~mask, 0).sum(-1)
    result = (hidden_states * total[:, None]).to(hidden_states.dtype)
    replacement = torch.full_like(expert_indices, identity_mask_value)
    replacement[mask.all(-1), 0] = 0
    expert_indices.copy_(torch.where(mask, replacement, expert_indices))
    expert_scales.masked_fill_(mask, 0)
    return result
