"""Public kv_compress_epilog operator with registered implementations."""

import torch

from ..utils.cann import detect_cann_version
from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator


@register_impl(
    "mem_cache.kv_compress_epilog", "auto", supported_generations={Generation.ARCH35}
)
def _auto_kv_compress_epilog(cache, x, slot_mapping, **kwargs):
    cann_version = detect_cann_version()
    if cann_version.startswith("9.1."):
        backend = "ascendc"
    elif cann_version.startswith("9.2."):
        backend = "cann_ops_transformer_cann920"
    else:
        raise RuntimeError(
            f"Unsupported CANN release {cann_version}; expected CANN 9.1.x or 9.2.x"
        )
    return get_impl_by_name("mem_cache.kv_compress_epilog", backend)(
        cache, x, slot_mapping, **kwargs
    )


@register_impl(
    "mem_cache.kv_compress_epilog",
    "cann_ops_transformer_cann920",
    supported_generations={Generation.ARCH35},
)
def _cann_kv_compress_epilog(cache, x, slot_mapping, **kwargs):
    cann_version = detect_cann_version()
    if not cann_version.startswith("9.2."):
        raise RuntimeError(
            f"This implementation requires CANN 9.2.x, got {cann_version}"
        )
    from ..._third_party.sgl_ops_transformer import ops  # noqa: F401

    op = getattr(torch.ops.cann_ops_transformer, "kv_compress_epilog", None)
    if op is None:
        raise RuntimeError(
            f"Missing operator cann_ops_transformer.kv_compress_epilog in the installed sgl-kernel-npu wheel for CANN {cann_version}"
        )
    op(cache.view(torch.uint8), x, slot_mapping, **kwargs)
    return cache


@register_impl(
    "mem_cache.kv_compress_epilog",
    "ascendc",
    supported_generations={Generation.ARCH35},
)
def _cann_9_1_kv_compress_epilog(cache, x, slot_mapping, **kwargs):
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
    supported_generations={Generation.ARCH35},
)
def _torch_kv_compress_epilog(cache, x, slot_mapping, **kwargs):
    from ...implementation.torch_native.mem_cache.kv_compress_epilog import (
        kv_compress_epilog,
    )

    return kv_compress_epilog(cache, x, slot_mapping, **kwargs)


@register_operator("mem_cache.kv_compress_epilog")
def kv_compress_epilog(
    cache,
    x,
    slot_mapping,
    *,
    quant_group_size=64,
    quant_mode=None,
    round_scale=True,
    x_scale=1.0,
    impl="auto",
):
    """Quantize and scatter A5 KV rows in place; -1 slots are skipped.

    impl: auto (SDK-selected), ascendc, cann_ops_transformer_cann920, or torch_native.
    torch_native is an eager correctness reference, not a graph fallback.
    """
    from .mem_cache_utils import (
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
