"""Independent eager reference for indexer_quant_cache."""

import torch

from ._quantization import _quantize


def indexer_quant_cache(
    cache, scale_cache, x, slot_mapping, *, quant_mode="fp8", round_scale=True
):
    if quant_mode != "fp8":
        raise NotImplementedError("reference supports fp8 per-token quantization")
    quant, scales = _quantize(
        x.reshape(-1, x.shape[-1]), torch.float8_e4m3fn, round_scale=round_scale
    )
    keys = cache.reshape(-1, cache.shape[-1])
    scale_rows = scale_cache.reshape(-1)
    for i, slot in enumerate(slot_mapping.tolist()):
        if slot >= 0:
            keys[slot].copy_(quant[i])
            scale_rows[slot].copy_(scales[i, 0])
    return cache, scale_cache
