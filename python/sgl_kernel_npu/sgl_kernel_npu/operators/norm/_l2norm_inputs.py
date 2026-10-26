"""Shared forward metadata checks and row-major preparation for L2 normalization."""


def prepare_input(x):
    if x.ndim < 1 or x.shape[-1] <= 0:
        raise ValueError("x must have a positive last dimension")
    maximum = 65536 // x.element_size()
    if x.shape[-1] > min(maximum, 1 << (x.shape[-1] - 1).bit_length()):
        raise RuntimeError("This layer doesn't support feature dim >= 64KB.")
    return x.contiguous()
