import torch


def gemma_rmsnorm(input, weight, eps=1e-6, *, round_gamma_in_weight_dtype):
    rstd = torch.rsqrt(input.float().square().mean(-1, keepdim=True) + eps)
    gamma = (
        (1.0 + weight).float() if round_gamma_in_weight_dtype else 1.0 + weight.float()
    )
    return (input.float() * rstd * gamma).to(input.dtype), rstd
