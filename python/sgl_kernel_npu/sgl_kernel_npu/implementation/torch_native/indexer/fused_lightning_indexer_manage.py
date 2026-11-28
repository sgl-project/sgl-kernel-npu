"""Eager PyTorch reference for causal TopK and resident-slot planning."""

from itertools import pairwise

import torch

from .lightning_indexer import lightning_indexer

_PAGE_SIZE = 128
_TOPK = 2048
_MAX_CACHE = 32640
_INVALID_SLOT = -(1 << 31)


def _offload_request_valid(
    length, actual, width, capacity, source_capacity, full_visible
):
    full_length = (width == 1 or full_visible) and length == actual
    stable_prefix = (
        length % _PAGE_SIZE == 0
        and length <= (actual - width) // _PAGE_SIZE * _PAGE_SIZE
    )
    minimum_cache = min(length, width * _TOPK)
    return (
        _TOPK <= length <= min(actual, source_capacity)
        and (full_length or stable_prefix)
        and minimum_cache <= capacity <= min(length, _MAX_CACHE)
        and capacity % _PAGE_SIZE == 0
    )


def _resident_plan(pool, route_sources, length, actual, capacity, cold, full_visible):
    selected = set().union(*route_sources)
    if cold:
        copied = sorted(selected)
        for source in range(length):
            if len(copied) == capacity:
                break
            if source not in selected:
                copied.append(source)
        residents = {source: slot for slot, source in enumerate(copied)}
        pool.fill_(_INVALID_SLOT)
        route_misses = route_sources
    else:
        width = len(route_sources)
        if (
            length % _PAGE_SIZE == 0
            and length <= (actual - width) // _PAGE_SIZE * _PAGE_SIZE
            and capacity < pool.numel()
            and int(pool[capacity]) >= capacity
        ):
            # Normalize the identity map produced by the non-offload route.
            pool[capacity:].fill_(_INVALID_SLOT)
        residents = {
            source: slot
            for source, slot in enumerate(pool[:length].tolist())
            if 0 <= slot < capacity
        }
        missing = selected.difference(residents)
        copied = sorted(missing)
        victims = [source for source in residents if source not in selected]
        if len(copied) > len(victims):
            raise ValueError(
                "A steady LIM request requires an initialized resident cache"
            )
        route_misses = [
            [source for source in row if source in missing] for row in route_sources
        ]
        for victim, source in zip(victims, copied):
            residents[source] = residents.pop(victim)
            pool[victim] = _INVALID_SLOT
        if full_visible and length == actual:
            # Retained current positions must be refreshed without counting as
            # per-route misses. New current positions are already in the prefix.
            copied.extend(
                source
                for source in range(actual - width, actual)
                if source in residents and source not in missing
            )
    copied_slots = [residents[source] for source in copied]
    if copied:
        pool[copied] = torch.tensor(copied_slots, dtype=pool.dtype, device="cpu")
    return residents, route_misses, copied, copied_slots


@torch.no_grad()
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
):
    """Fill the caller's state and output buffers using the eager CPU oracle."""
    if (
        query.dtype not in (torch.float16, torch.bfloat16)
        or index_weights.dtype != query.dtype
        or index_key_cache.dtype != query.dtype
    ):
        raise ValueError(
            "query, index_weights and index_key_cache must share FP16 or BF16 dtype"
        )
    spans = list(pairwise([0, *actual_seq_lengths_query.tolist()]))
    metadata = zip(
        spans,
        actual_seq_lengths_key.tolist(),
        offload_seq_lengths_key.tolist(),
        num_cache_tokens.tolist(),
        request_state.tolist(),
        req_pool_entries.tolist(),
        strict=True,
    )
    query, keys, weights, pages = (
        tensor.detach().cpu()
        for tensor in (query, index_key_cache, index_weights, index_block_table)
    )
    source_capacity = cache_slots_pool.size(1)
    for batch, ((start, end), actual, offload, capacity, state, row) in enumerate(
        metadata
    ):
        width = end - start
        length = actual if state == -3 else offload
        miss_counts[batch] = 0
        if not 0 <= start <= end <= query.size(0):
            continue
        topk_miss_counts[start:end].zero_()
        topk_src_ids[start:end].fill_(-1)
        topk_dst_slots[start:end].fill_(-1)
        if not (1 <= width <= 14 and width <= length <= source_capacity):
            continue
        row_valid = 0 <= row < cache_slots_pool.size(0)
        if state != -3 and (
            state not in (-2, -1)
            or not row_valid
            or not _offload_request_valid(
                length, actual, width, capacity, source_capacity, full_visible
            )
        ):
            continue
        page_ids = pages[batch, : (length + _PAGE_SIZE - 1) // _PAGE_SIZE]
        if ((page_ids < 0) | (page_ids >= keys.size(0))).any():
            raise ValueError("LIM index page is outside index_key_cache")
        selected, _ = lightning_indexer(
            query[start:end],
            keys,
            weights[start:end],
            actual_seq_lengths_query=torch.tensor([width], device="cpu"),
            actual_seq_lengths_key=torch.tensor([length], device="cpu"),
            block_table=pages[batch : batch + 1],
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=_TOPK,
            sparse_mode=3 if state == -3 or full_visible else 0,
        )
        if state == -3:
            if row_valid:
                cache_slots_pool[row].copy_(
                    torch.arange(
                        source_capacity, dtype=cache_slots_pool.dtype, device="cpu"
                    )
                )
            topk_src_ids[start:end].copy_(selected[:, 0])
            topk_dst_slots[start:end].copy_(selected[:, 0])
            continue
        route_sources = [
            sorted(source for source in ids if source >= 0)
            for ids in selected[:, 0].tolist()
        ]
        pool = cache_slots_pool[row].detach().cpu()
        residents, route_misses, copied, copied_slots = _resident_plan(
            pool, route_sources, length, actual, capacity, state == -2, full_visible
        )
        cache_slots_pool[row].copy_(pool)
        miss_counts[batch] = len(copied)
        miss_src_ids[batch, : len(copied)].copy_(
            torch.tensor(copied, dtype=torch.int32, device="cpu")
        )
        miss_dst_slots[batch, : len(copied)].copy_(
            torch.tensor(copied_slots, dtype=torch.int32, device="cpu")
        )
        for token, (sources, misses) in enumerate(
            zip(route_sources, route_misses), start
        ):
            miss_set = set(misses)
            ordered = misses + [source for source in sources if source not in miss_set]
            topk_miss_counts[token] = len(misses)
            topk_src_ids[token, : len(ordered)].copy_(
                torch.tensor(ordered, dtype=torch.int32, device="cpu")
            )
            topk_dst_slots[token, : len(ordered)].copy_(
                torch.tensor(
                    [residents[source] for source in ordered],
                    dtype=torch.int32,
                    device="cpu",
                )
            )
