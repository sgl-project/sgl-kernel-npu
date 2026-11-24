"""Independent torch reference for the LPLB physical-expert dispatch."""

import torch


def dispatch_probability(topk_ids, log2phy_prob, log2phy_map, random_vals=None):
    """Sample one physical expert per ``(token, slot)``.

    Inverse-CDF sampling over the per-row LP probabilities, falling back to a
    uniform choice over valid replicas when a row sums to zero. Equivalent to
    the Triton kernel modulo float32 rounding in the cumulative sum.
    """
    original_shape = topk_ids.shape
    flat_ids = topk_ids.reshape(-1).long()
    n = flat_ids.numel()
    num_logical, max_copies = log2phy_prob.shape
    if random_vals is None:
        random_vals = torch.rand(n, dtype=torch.float32, device=log2phy_prob.device)

    probs = log2phy_prob.index_select(0, flat_ids)  # (N, max_copies)
    maps = log2phy_map.index_select(0, flat_ids)  # (N, max_copies)

    row_sum = probs.sum(dim=-1, keepdim=True)
    fallback = (maps >= 0).to(torch.float32)
    probs = torch.where(row_sum > 0, probs, fallback)
    row_sum = probs.sum(dim=-1)

    # Inverse CDF: count the slots whose prefix sum is still <= u. Equal keys
    # advance the choice, matching the kernel's ``u >= cum`` comparison.
    threshold = (random_vals * row_sum).unsqueeze(-1)
    cumulative = probs.cumsum(dim=-1)
    chosen = (cumulative <= threshold).sum(dim=-1).clamp(max=max_copies - 1)

    out = maps.gather(1, chosen.unsqueeze(-1)).squeeze(-1)
    return out.reshape(original_shape).to(topk_ids.dtype)
