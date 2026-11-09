"""Independent PyTorch reference for ELU."""

import torch


def elu(x, alpha=1.0):
    """Evaluate the ELU definition with PyTorch, in float32.

    The piecewise definition is evaluated literally: the negative branch is
    selected with a boolean mask and computed with torch.expm1, which keeps the
    relative accuracy of exp(x) - 1 near zero. Nothing here calls the Ascend C
    operator, so the reference stays independent of the code under test.
    """
    compute_dtype = (
        torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
    )
    xf = x.to(compute_dtype)
    out = xf.clone()
    negative = xf <= 0
    out[negative] = alpha * torch.expm1(xf[negative])
    return out.to(x.dtype)
