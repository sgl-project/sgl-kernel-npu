"""Independent CPU FP64 selected-slot attention with direct BF16 rounding."""

import torch


def sparse_attention(q, k, v, slots, softmax_scale=None):
    device = q.device
    q, k, v = (x.detach().cpu().double() for x in (q, k, v))
    slots = slots.cpu().long()
    out = torch.zeros_like(q)
    group = q.shape[1] // k.shape[1]
    for row in range(q.shape[0]):
        ids = slots[row][slots[row] >= 0]
        if not ids.numel():
            continue
        for head in range(q.shape[1]):
            scores = (k[ids, head // group] @ q[row, head]) * (softmax_scale or 1 / 16)
            out[row, head] = scores.softmax(0) @ v[ids, head // group]
    # Direct BF16 lattice rounding in FP64, avoiding an FP32 midpoint.
    _, exponent = torch.frexp(out)
    step = torch.ldexp(torch.ones_like(out), (exponent - 8).clamp(min=-133))
    return (torch.round(out / step) * step).to(torch.bfloat16).to(device)
