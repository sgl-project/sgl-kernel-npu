"""Shared eager packed-KV quantization and decoding."""

import torch


def _quantize(x, dtype, *, round_scale=False):
    # Reference-only host conversion: older CANN lacks generic NPU FP8 casts.
    device = x.device
    host_fp8 = str(dtype).startswith("torch.float8_") and device.type != "cpu"
    if host_fp8:
        x = x.cpu()
    limit = torch.finfo(dtype).max
    scale = x.float().abs().amax(-1, keepdim=True) / limit
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    if round_scale:
        scale = torch.pow(2.0, torch.ceil(torch.log2(scale)))
    quant = (x.float() / scale).clamp(-limit, limit).to(dtype)
    if host_fp8:
        quant = quant.view(torch.uint8).to(device).view(dtype)
        scale = scale.to(device)
    return quant, scale
