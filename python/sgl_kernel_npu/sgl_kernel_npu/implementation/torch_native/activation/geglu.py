"""PyTorch reference for the GeGLU output and its GELU intermediate."""

from torch.nn import functional as F


def geglu(input, dim=-1, approximate=1, activate_left=False):
    if input.size(dim) % 2:
        raise ValueError("GeGLU requires an even split dimension")
    if approximate not in (0, 1):
        raise ValueError("approximate must be 0 (exact) or 1 (tanh)")
    left, right = input.chunk(2, dim)
    gate, other = (left, right) if activate_left else (right, left)
    activated = F.gelu(gate.float(), approximate="tanh" if approximate == 1 else "none")
    return (activated * other.float()).to(input.dtype), activated.to(input.dtype)
