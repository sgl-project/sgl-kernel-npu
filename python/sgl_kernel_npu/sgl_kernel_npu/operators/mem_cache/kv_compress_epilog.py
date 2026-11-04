"""Pack quantized KV cache rows through a common in-place interface."""

import torch

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator

_torch_native_fn = None


@register_impl(
    "mem_cache.kv_compress_epilog", "auto", supported_generations=[Generation.ARCH35]
)
@register_impl(
    "mem_cache.kv_compress_epilog", "ascendc", supported_generations=[Generation.ARCH35]
)
def _ascendc_impl(
    cache, x, slot_mapping, quant_group_size, quant_mode, round_scale_flag, layout
):
    return torch.ops.npu.kv_compress_epilog(
        cache, x, slot_mapping, quant_group_size, quant_mode, round_scale_flag, layout
    )


@register_impl("mem_cache.kv_compress_epilog", "torch_native")
def _torch_native_impl(
    cache, x, slot_mapping, quant_group_size, quant_mode, round_scale_flag, layout
):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.mem_cache.kv_compress_epilog import (
            kv_compress_epilog,
        )

        _torch_native_fn = kv_compress_epilog
    return _torch_native_fn(
        cache, x, slot_mapping, quant_group_size, quant_mode, round_scale_flag, layout
    )


@register_operator("mem_cache.kv_compress_epilog")
def kv_compress_epilog(
    kv_compress_cache,
    x,
    slot_mapping,
    quant_group_size,
    quant_mode,
    round_scale_flag,
    layout,
    *,
    impl="auto"
):
    """Quantize input rows and write them into the selected cache slots.

    Args:
        kv_compress_cache: float8_e4m3fn / float8_e5m2 cache tensor with layout-dependent shape.
            Caller-owned packed cache. Layout 1 accepts slot rows, including [slots, 1, width].
            Layout 2 uses [blocks, block_size, width] or [blocks, block_size, 1, width].
        x: bfloat16 tensor of shape [N, D]. Dense input rows with N > 0 and a 64-column RoPE
            suffix.
        slot_mapping: int32 / int64 tensor of shape [N]. Destination slot for each input row. A
            value of -1 skips that row.
        quant_group_size: MXFP8 group width, either 64 or 128. FP8 uses fixed groups of 128.
        quant_mode: 1 selects FP8 with FP32 scales. 2 selects MXFP8 with E8M0 exponent bytes.
        round_scale_flag: In MXFP8 mode, increment each nonzero biased scale exponent by one
            when true.
        layout: 1 stores one slot per row using the cache row stride. FP8 packs quantized
            values, RoPE, then scales. MXFP8 packs RoPE, quantized values, then scales. 2 is
            MXFP8 only and packs all value rows before all scale rows in each block.
        impl: Keyword-only. Choices are "auto", "ascendc", "torch_native". Defaults to
            "ascendc" via "auto".

    Returns:
        None. Mutates the supplied cache. Unwritten slots retain their contents.
    """
    return get_impl_by_name("mem_cache.kv_compress_epilog", impl)(
        kv_compress_cache,
        x,
        slot_mapping,
        quant_group_size,
        quant_mode,
        round_scale_flag,
        layout,
    )
