"""PyTorch reference for deinterleaving followed by half-rotation RoPE."""

import torch

from .rotary_mul import rotary_mul


def interleave_rope(x, cos, sin):
    deinterleaved = torch.cat((x[..., ::2], x[..., 1::2]), dim=-1)
    return rotary_mul(deinterleaved, cos, sin)
