import torch


def fused_gdn_gating_without_sigmoid(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
):
    x = a.to(torch.float32) + dt_bias.to(torch.float32)
    beta_x = beta * x
    softplus_x = torch.where(
        beta_x <= threshold, (1.0 / beta) * torch.log1p(torch.exp(beta_x)), x
    )
    g = -torch.exp(A_log.to(torch.float32)) * softplus_x
    g = g.unsqueeze(0)
    return g, b
