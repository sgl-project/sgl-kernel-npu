"""Independent normal-FP8 reference for the A5 indexer cache writer."""

import torch


def indexer_quant_cache(
    cache,
    scale_cache,
    x,
    slot_mapping,
    *,
    quant_mode="fp8",
    round_scale=True,
    x_scale=1.0,
    _cann_9_1=False,
):
    if quant_mode != "fp8":
        raise NotImplementedError("reference supports normal fp8 quantization only")
    # The public wrapper supplies the SDK contract without calling a DUT op.
    # 9.1 normal FP8 ignores round_scale; 9.2 applies it to FP32 scales too.
    # x_scale is HiFloat8-only in both versions.
    values = x.reshape(-1, x.shape[-1]).float().cpu()
    if values.shape[-1] > 128:
        raise NotImplementedError("reference supports one <=128-value block per row")
    if not torch.isfinite(values).all():
        raise NotImplementedError("reference supports finite activations only")
    scales = values.abs().amax(-1, keepdim=True) * torch.tensor(
        1.0 / 448.0, dtype=torch.float32, device="cpu"
    )
    if ((scales > 0) & (scales < torch.finfo(torch.float32).tiny)).any():
        raise NotImplementedError(
            "subnormal FP32 scales require device-specific validation"
        )
    if round_scale and not _cann_9_1:
        # frexp avoids log2 rounding across adjacent FP32 exponent boundaries.
        mantissa, exponent = torch.frexp(scales)
        exponent = exponent - (mantissa == 0.5).to(exponent.dtype)
        rounded = torch.ldexp(torch.ones_like(scales), exponent)
        scales = torch.where(scales > 0, rounded, scales)
    # Zero rows store scale=0. Use a safe divisor only for reference arithmetic.
    divisor = torch.where(scales > 0, scales, torch.ones_like(scales))
    if _cann_9_1:
        normalized = values / divisor
    else:
        # The released 9.2 kernel computes FP32 reciprocal, then multiplies.
        normalized = values * divisor.reciprocal()
    quant = normalized.clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    keys = cache.view(torch.uint8).reshape(-1, cache.shape[-1])
    scale_rows = scale_cache.reshape(-1)
    for i, slot in enumerate(slot_mapping.tolist()):
        if slot >= 0:
            keys[slot].copy_(quant[i].view(torch.uint8))
            scale_rows[slot].copy_(scales[i, 0])
    return cache, scale_cache
