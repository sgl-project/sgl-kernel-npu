import torch
import torch.nn.functional as F


def fused_kda_gate(
    gate, A_log, head_dim, *, gate_bias=None, beta=1.0, threshold=20.0, lower_bound=None
):
    heads = A_log.numel()
    if heads * head_dim != gate.shape[-1]:
        raise ValueError("KDA gate shape does not match A_log and head_dim")
    if gate_bias is not None and gate_bias.numel() != gate.shape[-1]:
        raise ValueError("KDA gate bias shape does not match the gate")
    x = gate.float().reshape(*gate.shape[:-1], heads, head_dim)
    if gate_bias is not None:
        x = x + gate_bias.float().reshape(heads, head_dim)
    exp_a = A_log.float().reshape(heads, 1).exp()
    if lower_bound is not None:
        return lower_bound * torch.sigmoid(exp_a * x)
    return -exp_a * F.softplus(x, beta=beta, threshold=threshold)
