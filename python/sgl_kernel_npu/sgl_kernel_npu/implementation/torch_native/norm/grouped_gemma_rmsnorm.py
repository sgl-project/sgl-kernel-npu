"""Independent CPU FP64 reference for grouped Gemma RMSNorm."""

import torch


def grouped_gemma_rmsnorm(x, weight, group_size, eps):
    values = x.detach().cpu()
    scale = 1 + weight.detach().cpu().double()
    result = torch.empty_like(values)
    # Bound CPU FP64 temporaries while preserving each group's reduction.
    for start in range(0, values.shape[0], 64):
        stop = min(values.shape[0], start + 64)
        grouped = values[start:stop].double().reshape(-1, 4, group_size)
        square_mean = (grouped * grouped).sum(-1, keepdim=True) / group_size
        normalized = (grouped / torch.sqrt(square_mean + eps)).flatten(1)
        result[start:stop] = (normalized * scale).to(torch.bfloat16)
    return result.to(x.device)
