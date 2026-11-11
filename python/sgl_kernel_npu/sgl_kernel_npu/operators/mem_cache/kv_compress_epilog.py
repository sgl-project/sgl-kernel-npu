"""Public kv_compress_epilog operator with registered implementations."""

import torch

from ..utils.cann import detect_cann_version
from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator


@register_impl(
    "mem_cache.kv_compress_epilog", "auto", supported_generations={Generation.ARCH35}
)
def _auto_kv_compress_epilog(cache, x, slot_mapping, **kwargs):
    if isinstance(kwargs["quant_mode"], int):
        return _cann_9_1_kv_compress_epilog(cache, x, slot_mapping, **kwargs)
    cann_version = detect_cann_version()
    if cann_version.startswith("9.1."):
        backend = "ascendc"
    elif cann_version.startswith("9.2."):
        backend = "cann_ops_transformers_cann920"
    else:
        raise RuntimeError(
            f"Unsupported CANN release {cann_version}; expected CANN 9.1.x or 9.2.x"
        )
    return get_impl_by_name("mem_cache.kv_compress_epilog", backend)(
        cache, x, slot_mapping, **kwargs
    )


@register_impl(
    "mem_cache.kv_compress_epilog",
    "cann_ops_transformers_cann920",
    supported_generations={Generation.ARCH35},
)
def _cann_kv_compress_epilog(cache, x, slot_mapping, **kwargs):
    if isinstance(kwargs["quant_mode"], int):
        raise ValueError("The released KV writer requires a packed quant_mode string")
    cann_version = detect_cann_version()
    if not cann_version.startswith("9.2."):
        raise RuntimeError(
            f"This implementation requires CANN 9.2.x, got {cann_version}"
        )
    from ..._third_party.sgl_ops_transformer_dsv4 import ops

    ops.kv_compress_epilog(cache.view(torch.uint8), x, slot_mapping, **kwargs)
    return cache


@register_impl(
    "mem_cache.kv_compress_epilog",
    "ascendc",
    supported_generations={Generation.ARCH35},
)
def _cann_9_1_kv_compress_epilog(cache, x, slot_mapping, **kwargs):
    if isinstance(kwargs["quant_mode"], int):
        return torch.ops.npu.kv_compress_epilog(cache, x, slot_mapping, **kwargs)
    cann_version = detect_cann_version()
    if not cann_version.startswith("9.1."):
        raise RuntimeError(
            f"This implementation requires CANN 9.1.x, got {cann_version}"
        )
    op = getattr(torch.ops.npu, "kv_compress_epilog", None)
    if op is None:
        raise RuntimeError(
            f"Missing operator npu.kv_compress_epilog in the installed sgl-kernel-npu wheel for CANN {cann_version}"
        )
    if (
        kwargs["quant_mode"] != "fp8_e8m0"
        or not kwargs["round_scale"]
        or kwargs["x_scale"] != 1.0
    ):
        raise RuntimeError(
            "cann_9_1 640B KV writer requires fp8_e8m0, round_scale=True and x_scale=1"
        )
    op(
        cache.view(torch.float8_e4m3fn).reshape(-1, 1, cache.shape[-1]),
        x,
        slot_mapping,
        quant_group_size=kwargs["quant_group_size"],
        quant_mode=2,
        round_scale_flag=True,
        layout=1,
    )
    return cache


@register_impl(
    "mem_cache.kv_compress_epilog",
    "torch_native",
)
def _torch_kv_compress_epilog(cache, x, slot_mapping, **kwargs):
    from ...implementation.torch_native.mem_cache.kv_compress_epilog import (
        kv_compress_epilog,
    )

    return kv_compress_epilog(cache, x, slot_mapping, **kwargs)


@register_operator("mem_cache.kv_compress_epilog")
def kv_compress_epilog(
    kv_compress_cache,
    x,
    slot_mapping,
    quant_group_size=64,
    quant_mode=None,
    round_scale_flag=None,
    layout=None,
    *,
    round_scale=True,
    x_scale=1.0,
    impl="auto",
):
    """Quantize and scatter KV rows in place; -1 slots are skipped.

    Integer quant_mode=1/2 with round_scale_flag and layout retains the native
    FP8/MXFP8 interface and returns None. Auto uses ascendc for that interface.
    Omitted/string quant_mode selects the DSV4 SDK-specific 608B/640B encoding
    with 64-element groups and 448+64 input width, returning the mutated cache.

    impl: auto (SDK-selected), ascendc, cann_ops_transformers_cann920, or torch_native.
    torch_native is an eager correctness reference, not a graph fallback.
    """
    cache = kv_compress_cache
    if isinstance(quant_mode, int):
        if round_scale_flag is None or layout is None:
            raise TypeError("Integer quant_mode requires round_scale_flag and layout")
        if not round_scale or x_scale != 1.0:
            raise ValueError(
                "round_scale and x_scale apply only to the packed interface"
            )
        return get_impl_by_name("mem_cache.kv_compress_epilog", impl)(
            cache,
            x,
            slot_mapping,
            quant_group_size=quant_group_size,
            quant_mode=quant_mode,
            round_scale_flag=round_scale_flag,
            layout=layout,
        )
    if round_scale_flag is not None or layout is not None:
        raise ValueError("round_scale_flag and layout require an integer quant_mode")
    from .get_npu_kv_quant_path import (
        CANN_9_1_NPU_KV_QUANT_PATH,
        NPU_KV_QUANT_PATH,
        get_npu_kv_quant_path,
    )

    if quant_group_size != 64:
        raise ValueError(
            "The migrated DSV4 KV contract requires 64-element quantization groups"
        )
    if impl == "torch_native":
        path = (
            CANN_9_1_NPU_KV_QUANT_PATH if cache.shape[-1] == 640 else NPU_KV_QUANT_PATH
        )
    else:
        path = get_npu_kv_quant_path()
    mode = path.writer_quant_mode if quant_mode is None else quant_mode
    if mode != path.writer_quant_mode or cache.shape[-1] != path.packed_dim(448, 64):
        raise ValueError(
            "KV cache encoding does not match the selected SDK reader/writer contract"
        )
    return get_impl_by_name("mem_cache.kv_compress_epilog", impl)(
        cache,
        x,
        slot_mapping,
        quant_group_size=quant_group_size,
        quant_mode=mode,
        round_scale=round_scale,
        x_scale=x_scale,
    )
