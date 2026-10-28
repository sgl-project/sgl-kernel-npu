"""PyTorch reference implementation of ELU used for accuracy comparison."""

import torch


def elu(x, alpha=1.0):
    """Reference ELU: ``x`` if ``x > 0`` else ``alpha * (exp(x) - 1)``.

    Computed in float32 so the reference stays independent of the kernel
    dtype, then cast back to the input dtype.
    """
    values = x.float()
    # expm1 avoids the catastrophic cancellation of exp(x) - 1 near zero.
    result = torch.where(values > 0, values, alpha * torch.expm1(values))
    return result.to(x.dtype)
