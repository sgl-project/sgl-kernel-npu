"""Shared eager packed-KV quantization and decoding."""

import torch


def _quantize(x, dtype, *, round_scale=False):
    limit = torch.finfo(dtype).max
    scale = x.float().abs().amax(-1, keepdim=True) / limit
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    if round_scale:
        scale = torch.pow(2.0, torch.ceil(torch.log2(scale)))
    return (x.float() / scale).clamp(-limit, limit).to(dtype), scale


def unpack_kv(cache):
    rows = cache.view(torch.uint8).reshape(-1, cache.shape[-1])
    if rows.shape[-1] != 608:
        raise NotImplementedError("reference supports 608-byte KV rows")
    rope = rows[:, :128].contiguous().view(torch.bfloat16).float()
    nope = (
        rows[:, 128:576]
        .contiguous()
        .view(torch.float8_e4m3fn)
        .float()
        .reshape(-1, 7, 64)
    )
    scales = (
        rows[:, 576:590].contiguous().view(torch.bfloat16).float().reshape(-1, 7, 1)
    )
    return torch.cat(((nope * scales).flatten(1), rope), -1)
