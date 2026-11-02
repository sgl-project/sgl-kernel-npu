"""Pack quantized KV cache rows through a common in-place interface."""

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator

_torch_native_fn = None
_cpu_fn = None


@register_impl(
    "mem_cache.kv_compress_epilog", "auto", supported_generations=[Generation.ARCH35]
)
@register_impl(
    "mem_cache.kv_compress_epilog", "ascendc", supported_generations=[Generation.ARCH35]
)
def _ascendc_impl(
    cache, x, slot_mapping, quant_group_size, quant_mode, round_scale_flag, layout
):
    import torch

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


@register_impl("mem_cache.kv_compress_epilog", "cpu")
def _cpu_impl(
    cache, x, slot_mapping, quant_group_size, quant_mode, round_scale_flag, layout
):
    global _cpu_fn
    if _cpu_fn is None:
        from ...implementation.cpu.mem_cache.kv_compress_epilog import (
            kv_compress_epilog,
        )

        _cpu_fn = kv_compress_epilog
    return _cpu_fn(
        cache, x, slot_mapping, quant_group_size, quant_mode, round_scale_flag, layout
    )


def _check_inputs(cache, x, slots, group, mode, layout):
    import torch

    def check(ok, message):
        if not ok:
            raise RuntimeError(message)

    check(x.ndim == 2, "x must be 2D tensor")
    check(x.shape[0] > 0 and x.shape[1] > 0, "x dimensions must be positive")
    check(x.dtype == torch.bfloat16, "x must be BF16")
    check(slots.ndim == 1, "slot_mapping must be 1D tensor")
    check(
        slots.shape[0] == x.shape[0], "slot_mapping size must equal x's first dimension"
    )
    check(
        slots.dtype in (torch.int32, torch.int64), "slot_mapping must be INT32 or INT64"
    )
    check(
        cache.dtype in (torch.float8_e5m2, torch.float8_e4m3fn),
        "kv_compress_cache must be FP8_E5M2 or FP8_E4M3FN",
    )
    if cache.ndim == 4:
        check(
            cache.shape[2] == 1,
            "kv_compress_cache 4D tensor requires headnum (dim 2) == 1",
        )
        cache = cache.squeeze(2)
    if mode == 2:
        check(group in (64, 128), "MXFP8 quant_group_size must be 64 or 128")
    qcols = x.shape[1] - 64
    scales = (qcols + group - 1) // group if mode == 2 else (qcols + 127) // 128
    if layout == 2:
        check(mode == 2, "layout=2 only supports MXFP8")
        check(cache.ndim >= 2, "layout=2 requires kv_compress_cache to be at least 2D")
        check(cache.shape[1] > 0, "blockSize must be positive")
        minimum = cache.shape[1] * (qcols + 128 + (scales + 7) // 8 * 8)
    else:
        used = qcols + 128 + scales * (1 if mode == 2 else 4)
        minimum = (used + 127) // 128 * 128
    check(
        cache.stride(0) <= 0 or cache.stride(0) >= minimum,
        "block_stride is smaller than the packed cache width",
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
    """Write packed cache storage and return None; -1 slots are skipped."""
    _check_inputs(
        kv_compress_cache, x, slot_mapping, quant_group_size, quant_mode, layout
    )
    return get_impl_by_name("mem_cache.kv_compress_epilog", impl)(
        kv_compress_cache,
        x,
        slot_mapping,
        quant_group_size,
        quant_mode,
        round_scale_flag,
        layout,
    )
