"""Independent PyTorch reference for the GDN gate preprocessing."""

import torch


def fused_gdn_gating(A_log, a, b, dt_bias, beta=1.0, threshold=20.0):
    """Compute g = -exp(A_log) * softplus_beta(a + dt_bias) and sigmoid(b).

    The thresholded softplus follows F.softplus: the log branch is taken when
    beta * x <= threshold, otherwise the value passes through unchanged.
    """
    x = a.to(torch.float32) + dt_bias.to(torch.float32)
    scaled = beta * x
    softplus_x = torch.where(
        scaled <= threshold,
        torch.log1p(torch.exp(scaled)) / beta,
        x,
    )
    return -torch.exp(A_log.to(torch.float32)) * softplus_x, torch.sigmoid(b)
