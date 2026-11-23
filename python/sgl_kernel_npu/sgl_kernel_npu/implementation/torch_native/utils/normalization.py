"""PyTorch reference for RMSNorm over the dimensions spanned by gamma."""

import torch


def normalize(input_fp32, gamma, epsilon):
    dims = tuple(range(-gamma.ndim, 0))
    rstd = torch.rsqrt(input_fp32.square().mean(dim=dims, keepdim=True) + epsilon)
    return input_fp32 * rstd * gamma.float(), rstd


def rms_norm(input, gamma, epsilon=1e-6):
    output, rstd = normalize(input.float(), gamma, epsilon)
    return output.to(input.dtype), rstd
