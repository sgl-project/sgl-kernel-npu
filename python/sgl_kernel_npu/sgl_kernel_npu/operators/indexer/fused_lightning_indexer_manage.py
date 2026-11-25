"""Public Lightning Indexer and resident-cache planning operator."""

import torch

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator

_OP = "indexer.fused_lightning_indexer_manage"
_SUPPORTED_GENERATIONS = (Generation.ARCH22,)


@register_impl(_OP, "auto", supported_generations=_SUPPORTED_GENERATIONS)
@register_impl(_OP, "ascendc", supported_generations=_SUPPORTED_GENERATIONS)
def _ascendc_impl(*args, **kwargs) -> None:
    torch.ops.npu.fused_lightning_indexer_manage.default(*args, **kwargs)


@register_operator(_OP)
def fused_lightning_indexer_manage(
    index_weights,
    query,
    index_key_cache,
    index_block_table,
    actual_seq_lengths_query,
    actual_seq_lengths_key,
    offload_seq_lengths_key,
    num_cache_tokens,
    request_state,
    req_pool_entries,
    cache_slots_pool,
    topk_src_ids,
    topk_dst_slots,
    topk_miss_counts,
    miss_src_ids,
    miss_dst_slots,
    miss_counts,
    full_visible=False,
    *,
    impl="auto",
) -> None:
    """Select TopK and update the resident-cache plan in place.

    Auto selects the wheel's Ascend C implementation. The index weights,
    query and key cache use FP16 or BF16 with shapes [T, H], [T, H, 128] and
    [P, 128, 1, 128]. The other tensors use int32. The [B, L] block table
    indexes key pages. The [B] inputs specify cumulative query ends, visible
    and previous offload lengths, cache capacities, request states and slot
    map rows. The operation mutates the [R, S] cache_slots_pool, [T, K] TopK
    sources and destinations, [T] TopK miss counts, [B, M] miss sources and
    destinations, and [B] miss counts. Full-visible mode includes all
    causally visible positions in selection. See the native contract for
    supported dimensions and metadata bounds.
    """
    get_impl_by_name(_OP, impl)(
        index_weights,
        query,
        index_key_cache,
        index_block_table,
        actual_seq_lengths_query,
        actual_seq_lengths_key,
        offload_seq_lengths_key,
        num_cache_tokens,
        request_state,
        req_pool_entries,
        cache_slots_pool,
        topk_src_ids,
        topk_dst_slots,
        topk_miss_counts,
        miss_src_ids,
        miss_dst_slots,
        miss_counts,
        full_visible=full_visible,
    )
