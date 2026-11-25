"""PyTorch reference for residual addition followed by RMSNorm."""

import torch

from ..utils.normalization import normalize


def add_rms_norm(x1, x2, gamma, epsilon=1e-6, *, round_intermediates=False):
    residual_sum = x1.float() + x2.float()
    residual = residual_sum.to(x1.dtype)
    normalized_dtype = None
    if round_intermediates and x1.dtype == torch.bfloat16:
        # The arch22 BF16 kernel rounds the sum before reduction and the
        # normalized values before applying gamma.
        residual_sum = residual.float()
        normalized_dtype = x1.dtype
    output, rstd = normalize(
        residual_sum, gamma, epsilon, normalized_dtype=normalized_dtype
    )
    return output.to(x1.dtype), rstd, residual
