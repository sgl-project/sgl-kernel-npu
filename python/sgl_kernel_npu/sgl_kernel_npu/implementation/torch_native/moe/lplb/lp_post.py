"""Independent torch reference for the LPLB ``lp_post`` operator."""

import torch


def lp_post(
    log2phy_prob,
    x,
    t1,
    phy_single,
    phy_replicated,
    log2phy,
):
    """Overwrite ``log2phy_prob`` from the IPM output and return it.

    Matches ``sglang/kernels/jit/csrc/lplb/lp_post.cuh``: zero-init the physical
    table, scatter the clamped replica ratios and the single-copy shares, then
    gather through ``log2phy`` (``-1`` wraps to the zero sink slot).
    """
    phy_prob_size = phy_single.numel() + phy_replicated.numel() + 1
    phy_prob = torch.zeros(
        phy_prob_size, dtype=torch.float32, device=log2phy_prob.device
    )
    # Replicated first, single second: the scatter order of the reference.
    phy_prob[phy_replicated.long()] = torch.clamp(x, min=0)
    phy_prob[phy_single.long()] = t1

    index = log2phy.long()
    index = torch.where(index < 0, index + phy_prob_size, index)
    log2phy_prob.copy_(phy_prob[index])
    return log2phy_prob
