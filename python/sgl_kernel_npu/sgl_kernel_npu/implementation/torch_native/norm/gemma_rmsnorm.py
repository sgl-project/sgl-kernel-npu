import torch

from ....operators.utils.generation import Generation, get_hardware


def gemma_rmsnorm(input, weight, eps=1e-6):
    rstd = torch.rsqrt(input.float().square().mean(-1, keepdim=True) + eps)
    gamma = (
        (1.0 + weight).float()
        if get_hardware().generation == Generation.ARCH35
        else 1.0 + weight.float()
    )
    return (input.float() * rstd * gamma).to(input.dtype), rstd
