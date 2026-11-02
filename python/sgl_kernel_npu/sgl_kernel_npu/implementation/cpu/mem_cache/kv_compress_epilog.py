"""CPU packing implementation with explicit output writeback."""

import torch

from ...torch_native.mem_cache.kv_compress_epilog import kv_compress_epilog as pack


def kv_compress_epilog(
    cache, x, slots, quant_group_size, quant_mode, round_scale_flag, layout
):
    # Preserve the cache strides because the packed block layout uses stride(0).
    cpu_bytes = torch.empty_strided(
        cache.shape, cache.stride(), dtype=torch.uint8, device="cpu"
    )
    cpu_bytes.copy_(cache.view(torch.uint8).cpu())
    cpu_cache = cpu_bytes.view(cache.dtype)
    pack(
        cpu_cache,
        x.cpu(),
        slots.cpu(),
        quant_group_size,
        quant_mode,
        round_scale_flag,
        layout,
    )
    cache.view(torch.uint8).copy_(cpu_bytes)
