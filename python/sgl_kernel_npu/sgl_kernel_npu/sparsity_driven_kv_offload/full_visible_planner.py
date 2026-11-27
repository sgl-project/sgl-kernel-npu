"""Map causal A5 Quant Lightning Indexer TopK into the shared C pool."""

import torch


def plan_full_visible_cache(
    topk: torch.Tensor,
    lengths: torch.Tensor,
    resident_rows: torch.Tensor,
    sparse: torch.Tensor,
    warm: torch.Tensor,
    source_to_slot: torch.Tensor,
    slot_to_source: torch.Tensor,
    source_ids: torch.Tensor,
    device_slots: torch.Tensor,
    topk_misses: torch.Tensor,
    miss_sources: torch.Tensor,
    miss_slots: torch.Tensor,
    miss_counts: torch.Tensor,
) -> None:
    """Plan a fixed-width batch without synchronizing counts back to the host.

    TopK is [B, Q, K]. Sparse rows share one allocation across their Q queries.
    Short rows use identity destinations. The forward map may contain stale
    entries; only a matching inverse owner is a hit. Each table reserves its
    final column as a write sink and the final row for padded graph requests.
    """
    batch, width, top_k = topk.shape
    context = source_to_slot.shape[1] - 1
    capacity = slot_to_source.shape[1] - 1
    device = topk.device
    routes = torch.arange(width, device=device, dtype=torch.int32)
    visible = lengths[:, None] - (width - 1 - routes)
    legal = (topk >= 0) & (topk < visible[:, :, None]) & (topk < context)
    ids = torch.where(legal, topk, context)
    sparse_ids = torch.where(sparse[:, None, None], ids, context)
    ordered = sparse_ids.reshape(batch, -1).sort(dim=1).values
    unique = (ordered != context) & torch.cat(
        [
            torch.ones((batch, 1), dtype=torch.bool, device=device),
            ordered[:, 1:] != ordered[:, :-1],
        ],
        dim=1,
    )

    rows = resident_rows.long()
    inverse = slot_to_source.index_select(0, rows)
    inverse = torch.where(warm[:, None], inverse, -1)

    def lookup(keys):
        slots = source_to_slot[rows[:, None], keys.long()]
        safe = slots.clamp(0, capacity).long()
        hits = (slots >= 0) & (slots < capacity) & (inverse.gather(1, safe) == keys)
        return slots, hits

    old_slots, hits = lookup(ordered)
    missing = unique & ~hits
    protected = torch.zeros_like(inverse)
    protected.scatter_(1, torch.where(unique & hits, old_slots, capacity).long(), 1)
    candidates = torch.arange(capacity, dtype=torch.int32, device=device)[None, :]
    # Prefer unused slots. Protect every route's hits from eviction.
    cost = candidates + capacity * (
        (inverse[:, :capacity] >= 0).int() + 2 * protected[:, :capacity]
    )
    available = cost.argsort(dim=1)
    ranks = missing.cumsum(1, dtype=torch.int32) - 1
    assigned = available.gather(1, ranks.clamp(0, capacity - 1).long()).int()
    new_sources = torch.where(missing, ordered, -1)
    new_slots = torch.where(missing, assigned, -1)

    _, previous_hits = lookup(sparse_ids.reshape(batch, -1))
    topk_misses.copy_(
        (legal & sparse[:, None, None] & ~previous_hits.reshape_as(topk))
        .sum(dim=-1, dtype=torch.int32)
        .reshape(-1)
    )
    inverse.scatter_(1, torch.where(missing, assigned, capacity).long(), new_sources)
    slot_to_source.index_copy_(0, rows, inverse)
    source_to_slot[rows[:, None], torch.where(missing, ordered, context).long()] = (
        new_slots
    )

    miss_counts.copy_(missing.sum(dim=1, dtype=torch.int32))
    miss_out = torch.where(missing, ranks, miss_sources.shape[1] - 1).long()
    miss_sources.scatter_(1, miss_out, new_sources)
    miss_slots.scatter_(1, miss_out, new_slots)

    destinations = source_to_slot[rows[:, None], ids.reshape(batch, -1).long()]
    destinations = torch.where(sparse[:, None, None], destinations.reshape_as(ids), ids)
    # Pack invalid entries at the end, including any out-of-window index
    # returned for an early verify query or a graph padding row.
    compact = torch.where(legal, legal.cumsum(-1, dtype=torch.int32) - 1, top_k)
    compact = compact.reshape(batch * width, top_k).long()
    for output, values in ((source_ids, ids), (device_slots, destinations)):
        packed = output.new_full((batch * width, top_k + 1), -1)
        packed.scatter_(1, compact, torch.where(legal, values, -1).reshape_as(output))
        output.copy_(packed[:, :top_k])
