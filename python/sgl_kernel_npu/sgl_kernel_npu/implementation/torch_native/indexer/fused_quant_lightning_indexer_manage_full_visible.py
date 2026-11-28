"""Eager PyTorch reference for Ascend950 FP8 LIM."""

from itertools import islice, pairwise

import torch

_PAGE_SIZE = 128
_TOPK = 2048
_MAX_CACHE = 32640
_INVALID_SLOT = -(1 << 31)


def _quant_scores(query, weights, keys, key_scale):
    scores = ((keys @ query.T).relu() @ weights) * key_scale
    # The indexer sorts BF16 scores with CAST_ROUND, including halfway values.
    return ((scores.contiguous().view(torch.int32) + 0x8000) & -65536).view(
        torch.float32
    )


def _plan_residency(pool, routes, length, capacity, cold):
    selected = set().union(*routes)
    if cold:
        copied = sorted(selected) + list(
            islice(
                (source for source in range(length) if source not in selected),
                capacity - len(selected),
            )
        )
        residents = {source: slot for slot, source in enumerate(copied)}
        pool.fill_(_INVALID_SLOT)
        missing = selected
    else:
        residents = {
            source: slot
            for source, slot in enumerate(pool[:length].tolist())
            if 0 <= slot < capacity
        }
        if len(residents) != capacity or len(set(residents.values())) != capacity:
            raise ValueError("Steady LIM requires a valid cold-filled resident map")
        missing = selected.difference(residents)
        copied = sorted(missing)
        victims = [source for source in residents if source not in selected]
        for victim, source in zip(victims, copied):
            residents[source] = residents.pop(victim)
            pool[victim] = _INVALID_SLOT
        copied.extend(
            source
            for source in range(length - len(routes), length)
            if source in residents and source not in missing
        )
    return residents, missing, copied


@torch.no_grad()
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
):
    """Publish causal TopK and one copy plan per request."""
    query = query.cpu().float()
    weights = weights.cpu().float() * query_scale.cpu()
    keys = key.cpu().view(torch.uint8)
    scales = key_scale.cpu()
    pages = block_table.cpu().long()
    ends = [0, *query_ends.tolist()]
    if ends[-1] != query.size(0) or any(
        not 1 <= end - start <= 14 for start, end in pairwise(ends)
    ):
        raise ValueError("LIM requires 1–14 rows per request in cumulative query_ends")

    topk_sources.fill_(-1)
    topk_slots.fill_(-1)
    topk_miss_counts.zero_()
    miss_counts.zero_()
    for batch, ((start, end), length, capacity, state, row) in enumerate(
        zip(
            pairwise(ends),
            lengths.tolist(),
            cache_tokens.tolist(),
            request_state.tolist(),
            pool_entries.tolist(),
            strict=True,
        )
    ):
        width = end - start
        if (
            state not in (-3, -2, -1)
            or not width <= length <= min(pages.size(1) * _PAGE_SIZE, slot_pool.size(1))
            or not 0 < row < slot_pool.size(0) - 1
        ):
            continue
        if state != -3 and (
            not width * _TOPK <= capacity <= min(length, _MAX_CACHE)
            or capacity % _PAGE_SIZE
        ):
            continue
        page_ids = pages[batch, : (length + _PAGE_SIZE - 1) // _PAGE_SIZE]
        if ((page_ids < 0) | (page_ids >= keys.size(0))).any():
            raise ValueError("An active LIM block table entry is outside the key cache")
        key_rows = (
            keys.index_select(0, page_ids)
            .view(key.dtype)
            .reshape(-1, key.size(-1))[:length]
            .float()
        )
        scale_rows = scales.index_select(0, page_ids).reshape(-1)[:length]
        routes = []
        for token in range(start, end):
            visible = length - (end - token - 1)
            if visible <= _TOPK:
                routes.append(list(range(visible)))
            else:
                scores = _quant_scores(
                    query[token],
                    weights[token],
                    key_rows[:visible],
                    scale_rows[:visible],
                )
                routes.append(
                    scores.argsort(descending=True, stable=True)[:_TOPK].tolist()
                )

        if state == -3:
            for token, sources in enumerate(routes, start):
                ids = torch.tensor(sources, dtype=torch.int32, device="cpu")
                topk_sources[token, : len(sources)].copy_(ids)
                topk_slots[token, : len(sources)].copy_(ids)
            continue

        pool = slot_pool[row].cpu()
        residents, missing, copied = _plan_residency(
            pool, routes, length, capacity, cold=state == -2
        )
        destinations = [residents[source] for source in copied]
        if copied:
            pool[copied] = torch.tensor(destinations, dtype=pool.dtype, device="cpu")
            slot_pool[row].copy_(pool)
            miss_sources[batch, : len(copied)].copy_(
                torch.tensor(copied, dtype=torch.int32, device="cpu")
            )
            miss_slots[batch, : len(copied)].copy_(
                torch.tensor(destinations, dtype=torch.int32, device="cpu")
            )
            miss_counts[batch] = len(copied)

        for token, sources in enumerate(routes, start):
            misses = [source for source in sources if source in missing]
            ordered = misses + [source for source in sources if source not in missing]
            topk_sources[token, : len(ordered)].copy_(
                torch.tensor(ordered, dtype=torch.int32, device="cpu")
            )
            topk_slots[token, : len(ordered)].copy_(
                torch.tensor(
                    [residents[source] for source in ordered],
                    dtype=torch.int32,
                    device="cpu",
                )
            )
            topk_miss_counts[token] = len(misses)
