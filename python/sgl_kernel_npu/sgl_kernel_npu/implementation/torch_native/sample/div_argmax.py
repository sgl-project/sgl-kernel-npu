import torch


def div_argmax(x, divisor, *, divisor_min=None, ignore_nan=False):
    """Row-wise ``(x / divisor).argmax(-1)``.

    ``torch.clamp_min`` propagates NaN like the kernel's clamp; with
    ``ignore_nan`` a NaN quotient becomes -inf before ``torch.argmax``.
    """
    d = divisor.clamp_min(divisor_min) if divisor_min is not None else divisor
    v = x / d
    if ignore_nan:
        v = torch.where(torch.isnan(v), float("-inf"), v)
    return v.argmax(dim=-1)
