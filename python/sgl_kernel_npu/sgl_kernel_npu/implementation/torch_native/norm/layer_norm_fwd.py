"""Grouped LayerNorm/RMSNorm and SiLU gating on prepared tensor views."""

import torch


def layer_norm_fwd(
    x, weight, bias, z, out, mean, rstd, eps, norm_before_gate, is_rms_norm, sigmoid_gate=False
):
    values = x.float()
    gate = None
    if z is not None:
        gate = torch.sigmoid(z.float())
        if not sigmoid_gate:
            gate = z.float() * gate
    if gate is not None and not norm_before_gate:
        values = values * gate
        if sigmoid_gate:
            values = values.to(x.dtype).float()
    if is_rms_norm:
        centered = values
    else:
        averages = values.mean(-1)
        mean.copy_(averages)
        centered = values - averages.unsqueeze(-1)
    inverse = 1.0 / torch.sqrt(centered.square().mean(-1) + eps)
    rstd.copy_(inverse)
    output = centered * inverse.unsqueeze(-1) * weight.float()
    if bias is not None:
        output = output + bias.float()
    if gate is not None and norm_before_gate:
        if sigmoid_gate:
            output = output.to(out.dtype).float()
        output = output * gate
    out.copy_(output)
