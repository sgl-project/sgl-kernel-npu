# This file contains swiglu for OpenAI models.
# It will be optimized using Triton in the future.
import torch


def swiglu_oai(layer, hidden_states):
    E, N, _ = layer.w13_weight.size()
    gate_up = hidden_states.view(-1, N)
    alpha = layer.moe_runner_config.gemm1_alpha
    limit = layer.moe_runner_config.gemm1_clamp_limit
    gate, up = gate_up[..., ::2], gate_up[..., 1::2]
    gate = gate.clamp(min=None, max=limit)
    up = up.clamp(min=-limit, max=limit)
    glu = gate * torch.sigmoid(gate * alpha)
    gated_output = (up + 1) * glu
    return gated_output
