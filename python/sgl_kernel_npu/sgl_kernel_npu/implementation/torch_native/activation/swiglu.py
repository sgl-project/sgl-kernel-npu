"""PyTorch reference for SwiGLU."""

from torch.nn import functional as F


def swiglu(input, dim=-1):
    if input.size(dim) % 2:
        raise ValueError("SwiGLU requires an even split dimension")
    gate, up = input.chunk(2, dim)
    return (F.silu(gate.float()) * up.float()).to(input.dtype)
