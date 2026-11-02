"""Eager FP8 cache references for the CANN fp8_bf16 row format."""

import torch


def _quantize(x, dtype, *, round_scale=False):
    limit = torch.finfo(dtype).max
    scale = x.float().abs().amax(-1, keepdim=True) / limit
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    if round_scale:
        scale = torch.pow(2.0, torch.ceil(torch.log2(scale)))
    return (x.float() / scale).clamp(-limit, limit).to(dtype), scale


def kv_compress_epilog(
    cache,
    x,
    slot_mapping,
    *,
    quant_group_size=64,
    quant_mode="fp8_bf16",
    round_scale=True,
    x_scale=1.0,
):
    if quant_mode != "fp8_bf16" or quant_group_size != 64:
        raise NotImplementedError("reference supports fp8_bf16 with 64-element groups")
    # DSV4 stores 64 bf16 RoPE features followed by group-quantized NoPE and bf16 scales.
    if x.shape[-1] != 512 or cache.shape[-1] != 608:
        raise NotImplementedError(
            "reference supports DSV4's 448 NoPE + 64 RoPE, 608-byte rows"
        )
    rows = cache.view(torch.uint8).reshape(-1, cache.shape[-1])
    values = x.reshape(-1, 512).float() * x_scale
    for i, slot in enumerate(slot_mapping.tolist()):
        if slot < 0:
            continue
        quant, scales = _quantize(
            values[i, :448].reshape(7, 64), torch.float8_e4m3fn, round_scale=round_scale
        )
        rows[slot, :128].copy_(
            values[i, 448:].to(torch.bfloat16).contiguous().view(torch.uint8)
        )
        rows[slot, 128:576].copy_(quant.contiguous().view(torch.uint8).flatten())
        rows[slot, 576:590].copy_(
            scales.to(torch.bfloat16).contiguous().view(torch.uint8).flatten()
        )
    return cache


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


def indexer_quant_cache(cache, scale_cache, x, slot_mapping, *, quant_mode="fp8"):
    if quant_mode != "fp8":
        raise NotImplementedError("reference supports fp8 per-token quantization")
    quant, scales = _quantize(x.reshape(-1, x.shape[-1]), torch.float8_e4m3fn)
    keys = cache.reshape(-1, cache.shape[-1])
    scale_rows = scale_cache.reshape(-1)
    for i, slot in enumerate(slot_mapping.tolist()):
        if slot >= 0:
            keys[slot].copy_(quant[i])
            scale_rows[slot].copy_(scales[i, 0])
    return cache, scale_cache
