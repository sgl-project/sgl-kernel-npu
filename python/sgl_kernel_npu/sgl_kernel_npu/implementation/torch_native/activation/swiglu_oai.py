import torch


def swiglu_oai(hidden_states, dim, gemm1_alpha, gemm1_clamp_limit):
    gate_up = hidden_states.view(-1, dim)
    gate, up = gate_up[..., ::2], gate_up[..., 1::2]
    gate = gate.clamp(min=None, max=gemm1_clamp_limit)
    up = up.clamp(min=-gemm1_clamp_limit, max=gemm1_clamp_limit)
    glu = gate * torch.sigmoid(gate * gemm1_alpha)
    return (up + 1) * glu
