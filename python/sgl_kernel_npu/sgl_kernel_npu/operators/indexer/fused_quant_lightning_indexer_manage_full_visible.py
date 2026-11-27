"""Public Ascend950 HiSparse operator."""

import torch

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator

_OP = "indexer.fused_quant_lightning_indexer_manage_full_visible"
_SUPPORTED_GENERATIONS = (Generation.ARCH35,)


@register_impl(_OP, "auto", supported_generations=_SUPPORTED_GENERATIONS)
@register_impl(_OP, "ascendc", supported_generations=_SUPPORTED_GENERATIONS)
def _ascendc_impl(*args, **kwargs) -> None:
    torch.ops.npu.fused_quant_lightning_indexer_manage_full_visible.default(
        *args, **kwargs
    )


@register_operator(_OP)
def fused_quant_lightning_indexer_manage_full_visible(
    query,
    weights,
    key,
    query_scale,
    key_scale,
    block_table,
    query_ends,
    lengths,
    cache_tokens,
    request_state,
    pool_entries,
    slot_pool,
    topk_sources,
    topk_slots,
    topk_miss_counts,
    miss_sources,
    miss_slots,
    miss_counts,
    *,
    impl="auto",
) -> None:
    """Compute causal TopK from FP8 scores and update resident slots in place.

    Both auto and ascendc select the native 3510 operator.
    See the operator README for the complete tensor contract.
    """
    get_impl_by_name(_OP, impl)(
        query,
        weights,
        key,
        query_scale,
        key_scale,
        block_table,
        query_ends,
        lengths,
        cache_tokens,
        request_state,
        pool_entries,
        slot_pool,
        topk_sources,
        topk_slots,
        topk_miss_counts,
        miss_sources,
        miss_slots,
        miss_counts,
    )
