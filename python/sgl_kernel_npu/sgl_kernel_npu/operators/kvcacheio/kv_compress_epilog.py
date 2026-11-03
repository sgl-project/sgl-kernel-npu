"""Public kv_compress_epilog operator with registered implementations."""

import torch

from ..utils._transformer_compat import OperatorUnavailableError, attention_family
from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator


@register_impl(
    "kvcacheio.kv_compress_epilog", "auto", supported_generations={Generation.ARCH35}
)
@register_impl(
    "kvcacheio.kv_compress_epilog",
    "cann_ops_transformer",
    supported_generations={Generation.ARCH35},
)
def _cann_kv_compress_epilog(cache, x, slot_mapping, **kwargs):
    family = attention_family(True)
    if family.legacy:
        if (
            kwargs["quant_mode"] != "fp8_e8m0"
            or not kwargs["round_scale"]
            or kwargs["x_scale"] != 1.0
        ):
            raise OperatorUnavailableError(
                "Legacy 640B KV writer requires fp8_e8m0, round_scale=True and x_scale=1"
            )
        family.ops[2](
            cache.view(torch.float8_e4m3fn).reshape(-1, 1, cache.shape[-1]),
            x,
            slot_mapping,
            quant_group_size=kwargs["quant_group_size"],
            quant_mode=2,
            round_scale_flag=True,
            layout=1,
        )
    else:
        family.ops[2](cache.view(torch.uint8), x, slot_mapping, **kwargs)
    return cache


@register_impl(
    "kvcacheio.kv_compress_epilog",
    "torch_native",
    supported_generations={Generation.ARCH35},
)
def _torch_kv_compress_epilog(cache, x, slot_mapping, **kwargs):
    from ...implementation.torch_native.kvcacheio.kv_compress_epilog import (
        kv_compress_epilog,
    )

    return kv_compress_epilog(cache, x, slot_mapping, **kwargs)


@register_operator("kvcacheio.kv_compress_epilog")
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

    impl: auto (released/legacy compatibility), cann_ops_transformer, or torch_native.
    torch_native is an eager correctness reference, not a graph fallback.
    """
    from .layout import (
        LEGACY_NPU_KV_QUANT_PATH,
        NPU_KV_QUANT_PATH,
        get_npu_kv_quant_path,
    )

    if quant_group_size != 64:
        raise ValueError(
            "The migrated DSV4 KV contract requires 64-element quantization groups"
        )
    if impl == "torch_native":
        path = LEGACY_NPU_KV_QUANT_PATH if cache.shape[-1] == 640 else NPU_KV_QUANT_PATH
    else:
        path = get_npu_kv_quant_path()
    mode = path.writer_quant_mode if quant_mode is None else quant_mode
    if mode != path.writer_quant_mode or cache.shape[-1] != path.packed_dim(448, 64):
        raise ValueError(
            "KV cache encoding does not match the selected reader/writer family"
        )
    return get_impl_by_name("kvcacheio.kv_compress_epilog", impl)(
        cache,
        x,
        slot_mapping,
        quant_group_size=quant_group_size,
        quant_mode=mode,
        round_scale=round_scale,
        x_scale=x_scale,
    )
