"""PyTorch reference for residual addition followed by RMSNorm."""

from .rms_norm import normalize


def add_rms_norm(x1, x2, gamma, epsilon=1e-6):
    residual_sum = x1.float() + x2.float()
    output, rstd = normalize(residual_sum, gamma, epsilon)
    return output.to(x1.dtype), rstd, residual_sum.to(x1.dtype)
