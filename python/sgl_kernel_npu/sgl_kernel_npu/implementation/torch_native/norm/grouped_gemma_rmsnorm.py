"""Independent CPU FP64 reference for grouped Gemma RMSNorm."""

import torch


def grouped_gemma_rmsnorm(x, weight, group_size=None, eps=1e-6):
    values = x.detach().cpu()
    width = values.shape[-1] if group_size is None else group_size
    slices = weight.numel() // width
    scale = (1.0 + weight.detach().cpu().double()).reshape(slices, width)
    flat = values.reshape(-1, slices, width)
    result = torch.empty_like(flat)
    # Bound CPU FP64 temporaries while preserving each group's reduction.
    for start in range(0, flat.shape[0], 64):
        stop = min(flat.shape[0], start + 64)
        grouped = flat[start:stop].double()
        square_mean = (grouped * grouped).sum(-1, keepdim=True) / width
        normalized = grouped / torch.sqrt(square_mean + eps)
        result[start:stop] = (normalized * scale).to(values.dtype)
    return result.reshape(values.shape).to(x.device)
