import torch


def top_k_top_p_renorm_probs(
    probs: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    need_top_k_sampling: bool,
    need_top_p_sampling: bool,
) -> torch.Tensor:
    """Apply the same sequential top-k then top-p policy used by SGLang GPU."""
    if not need_top_k_sampling and not need_top_p_sampling:
        return probs

    vocab_size = probs.shape[-1]
    sorted_probs, sorted_indices = probs.sort(dim=-1, descending=True)

    if need_top_k_sampling:
        top_ks = top_ks.to(device=probs.device, dtype=torch.long).clamp(
            min=1, max=vocab_size
        )
        positions = torch.arange(vocab_size, device=probs.device).view(1, -1)
        sorted_probs.masked_fill_(positions >= top_ks.view(-1, 1), 0.0)
        sorted_probs.div_(
            sorted_probs.sum(dim=-1, keepdim=True).clamp_min_(1e-20)
        )

    if need_top_p_sampling:
        top_ps = top_ps.to(device=probs.device, dtype=probs.dtype)
        cumulative_probs = sorted_probs.cumsum(dim=-1)
        sorted_probs.masked_fill_(
            cumulative_probs - sorted_probs > top_ps.view(-1, 1), 0.0
        )
        sorted_probs.div_(
            sorted_probs.sum(dim=-1, keepdim=True).clamp_min_(1e-20)
        )

    return torch.zeros_like(probs).scatter_(
        dim=-1, index=sorted_indices, src=sorted_probs
    )
