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
):
    if quant_mode != "fp8":
        raise NotImplementedError("reference supports normal fp8 quantization only")
    # round_scale belongs to MX-FP8, and x_scale to HiFloat8. Neither changes
    # normal FP8 (mode 1): the per-128-value scale is stored as FP32.
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
    # An all-zero row stores scale=0 and zero data; avoid a 0/0 reference NaN.
    divisor = torch.where(scales > 0, scales, torch.ones_like(scales))
    quant = (values / divisor).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    keys = cache.view(torch.uint8).reshape(-1, cache.shape[-1])
    scale_rows = scale_cache.reshape(-1)
    for i, slot in enumerate(slot_mapping.tolist()):
        if slot >= 0:
            keys[slot].copy_(quant[i].view(torch.uint8))
            scale_rows[slot].copy_(scales[i, 0])
    return cache, scale_cache
