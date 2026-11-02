"""Independent eager reference for kv_compress_epilog."""

import torch

from ._quantization import _quantize


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
