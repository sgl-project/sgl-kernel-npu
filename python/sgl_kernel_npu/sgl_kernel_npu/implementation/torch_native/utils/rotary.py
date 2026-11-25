"""PyTorch reference for the standard half and adjacent-pair RoPE modes."""

import torch


def rotary_mul(input, r1, r2, rotary_mode="half", rotate=None):
    if rotate is not None:
        raise ValueError("torch_native rotary_mul requires rotate=None")
    if input.size(-1) % 2:
        raise ValueError("rotary_mul requires an even last dimension")

    x = input.float()
    if rotary_mode == "half":
        left, right = x.chunk(2, dim=-1)
        rotated = torch.cat((-right, left), dim=-1)
    elif rotary_mode == "interleave":
        rotated = torch.stack((-x[..., 1::2], x[..., ::2]), dim=-1).flatten(-2)
    else:
        raise ValueError("torch_native rotary_mul supports half and interleave")

    cos = r1.to(input.dtype).float()
    sin = r2.to(input.dtype).float()
    return (x * cos + rotated * sin).to(input.dtype)
