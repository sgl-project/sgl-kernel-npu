"""PyTorch reference for RMSNorm over the dimensions spanned by gamma."""

import torch


def normalize(input_fp32, gamma, epsilon, *, normalized_dtype=None):
    dims = tuple(range(-gamma.ndim, 0))
    rstd = torch.rsqrt(input_fp32.square().mean(dim=dims, keepdim=True) + epsilon)
    normalized = input_fp32 * rstd
    if normalized_dtype is not None:
        normalized = normalized.to(normalized_dtype).float()
    return normalized * gamma.float(), rstd


def rms_norm(input, gamma, epsilon=1e-6):
    output, rstd = normalize(input.float(), gamma, epsilon)
    return output.to(input.dtype), rstd
