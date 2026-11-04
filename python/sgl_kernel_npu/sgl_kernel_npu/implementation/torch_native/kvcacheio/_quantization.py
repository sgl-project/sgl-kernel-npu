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


def _fp8_to_float(value):
    if str(value.dtype).startswith("torch.float8_") and value.device.type != "cpu":
        return value.view(torch.uint8).cpu().view(value.dtype).float().to(value.device)
    return value.float()


def unpack_kv(cache):
    rows = cache.view(torch.uint8).reshape(-1, cache.shape[-1])
    if rows.shape[-1] not in (608, 640):
        raise NotImplementedError("reference supports 608B and legacy 640B KV rows")
    rope = rows[:, :128].contiguous().view(torch.bfloat16).float()
    nope = _fp8_to_float(
        rows[:, 128:576].contiguous().view(torch.float8_e4m3fn)
    ).reshape(-1, 7, 64)
    if rows.shape[-1] == 640:
        exponent = rows[:, 576:583].to(torch.int32) - 127
        scales = torch.pow(2.0, exponent.float()).reshape(-1, 7, 1)
    else:
        scales = (
            rows[:, 576:590].contiguous().view(torch.bfloat16).float().reshape(-1, 7, 1)
        )
    return torch.cat(((nope * scales).flatten(1), rope), -1)
