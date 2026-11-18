"""Public kv_compress_epilog_v2 operator with registered implementations."""

from ..utils.cann import detect_cann_version
from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator


@register_impl(
    "mem_cache.kv_compress_epilog_v2",
    "auto",
    supported_generations={Generation.ARCH35},
)
@register_impl(
    "mem_cache.kv_compress_epilog_v2",
    "cann_recipes_infer",
    supported_generations={Generation.ARCH35},
)
def _cann_kv_compress_epilog_v2(cache, x, slot_mapping, **kwargs):
    cann_version = detect_cann_version()
    if not cann_version.startswith("9.2."):
        raise RuntimeError(
            f"This implementation requires CANN 9.2.x, got {cann_version}"
        )
    from ..._third_party.sgl_cann_recipes_infer import kv_compress_epilog_v2 as op

    op(cache, x, slot_mapping, **kwargs)


@register_impl("mem_cache.kv_compress_epilog_v2", "torch_native")
def _torch_kv_compress_epilog_v2(cache, x, slot_mapping, **kwargs):
    from ...implementation.torch_native.mem_cache.kv_compress_epilog_v2 import (
        kv_compress_epilog_v2,
    )

    kv_compress_epilog_v2(cache, x, slot_mapping, **kwargs)


@register_operator("mem_cache.kv_compress_epilog_v2")
def kv_compress_epilog_v2(
    cache,
    x,
    slot_mapping,
    *,
    quant_group_size=32,
    quant_mode="mxfp8_bf16",
    round_scale=True,
    x_scale=1.0,
    impl="auto",
):
    """Quantize BF16 KV rows and scatter them into the cache in place.

    Signature, in-place mutation and the empty return value match the released
    ``custom.kv_compress_epilog_v2`` binding, which also owns input validation.

    auto selects the bundled CANN 9.2.0 recipes kernel on Ascend 950. x is
    contiguous BF16 [T, D]; slot_mapping is int32/int64 [T] and rows with -1
    (or out-of-range) slots are skipped. cache is uint8/FP8 [N, C] (layout 1)
    or [B, S, 1, C] (layout 2); written rows hold [values | uint16 scales].
    MXFP8 requires quant_group_size=32 with an FP8 cache; MXFP4 accepts 16 or
    32 with a uint8 cache. round_scale applies to MXFP8; x_scale is reserved
    and must be 1.0.

    impl: auto/cann_recipes_infer or torch_native. torch_native is an eager
    correctness reference, not a graph fallback.
    """
    get_impl_by_name("mem_cache.kv_compress_epilog_v2", impl)(
        cache,
        x,
        slot_mapping,
        quant_group_size=quant_group_size,
        quant_mode=quant_mode,
        round_scale=round_scale,
        x_scale=x_scale,
    )
