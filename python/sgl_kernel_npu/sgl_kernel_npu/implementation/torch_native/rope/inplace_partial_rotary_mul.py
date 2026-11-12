"""Independent eager reference for inplace_partial_rotary_mul."""

import torch


def inplace_partial_rotary_mul(
    x, cos, sin, *, rotary_mode="interleave", partial_slice=None
):
    start, end = partial_slice if partial_slice is not None else (0, x.shape[-1])
    if (end - start) % 2 or start < 0 or end > x.shape[-1]:
        raise ValueError(
            "partial_slice must select an even, in-bounds rotary dimension"
        )
    part = x[..., start:end].float()
    if rotary_mode == "interleave":
        rotated = torch.stack((-part[..., 1::2], part[..., ::2]), -1).flatten(-2)
    elif rotary_mode == "half":
        a, b = part.chunk(2, -1)
        rotated = torch.cat((-b, a), -1)
    else:
        raise NotImplementedError(f"unsupported rotary_mode={rotary_mode}")
    x[..., start:end].copy_((part * cos.float() + rotated * sin.float()).to(x.dtype))
    return x
