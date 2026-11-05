"""Independent eager reference for kv_compress_epilog."""

import torch

from .quantization import _quantize


def _legacy_e8m0_quantize(x):
    """Independent reference for the deployed A5 9.1 writer's E8M0 rule.

    The original writer/tests use FP32 max*reciprocal after amax>=1e-4,
    then the next binary exponent even when that value is already 2**n.
    This is not the released writer's ceil(log2(scale)) convention.
    """
    values = x.float().cpu()
    amax = values.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    raw_scale = amax * torch.tensor(1.0 / 448.0, dtype=torch.float32, device="cpu")
    _, exponent = torch.frexp(raw_scale)
    scale = torch.ldexp(torch.ones_like(raw_scale), exponent)
    quant = (values / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return quant, scale


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
    if quant_mode not in ("fp8_bf16", "fp8_e8m0") or quant_group_size != 64:
        raise NotImplementedError(
            "reference supports 64-element BF16/E8M0 scale groups"
        )
    legacy = quant_mode == "fp8_e8m0"
    if legacy and not round_scale:
        raise NotImplementedError("legacy E8M0 scales require power-of-two rounding")
    # DSV4 stores 64 bf16 RoPE features followed by group-quantized NoPE and bf16 scales.
    if x.shape[-1] != 512 or cache.shape[-1] != (640 if legacy else 608):
        raise NotImplementedError(
            "reference requires matching 608B BF16-scale or 640B E8M0-scale rows"
        )
    rows = cache.view(torch.uint8).reshape(-1, cache.shape[-1])
    values = x.reshape(-1, 512).float() * x_scale
    for i, slot in enumerate(slot_mapping.tolist()):
        if slot < 0:
            continue
        groups = values[i, :448].reshape(7, 64)
        if legacy:
            quant, scales = _legacy_e8m0_quantize(groups)
        else:
            quant, scales = _quantize(
                groups, torch.float8_e4m3fn, round_scale=round_scale
            )
        rows[slot, :128].copy_(
            values[i, 448:].to(torch.bfloat16).contiguous().view(torch.uint8)
        )
        rows[slot, 128:576].copy_(quant.contiguous().view(torch.uint8).flatten())
        if legacy:
            rows[slot, 576:583].copy_(
                (torch.log2(scales).flatten() + 127).to(torch.uint8)
            )
        else:
            rows[slot, 576:590].copy_(
                scales.to(torch.bfloat16).contiguous().view(torch.uint8).flatten()
            )
    return cache
