"""Independent eager reference for mhc_post."""

import torch


def mhc_post_dpskv4(x, residual, post, comb):
    return (
        post.float().unsqueeze(-1) * x.float().unsqueeze(-2)
        + torch.einsum("...ji,...jd->...id", comb.float(), residual.float())
    ).to(x.dtype)
