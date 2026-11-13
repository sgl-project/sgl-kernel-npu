"""Reference semantics and dispatch for sparse-KV cache planning."""

from __future__ import annotations

from typing import Sequence

import torch


def plan_sparse_kv_cache_reference(
    source_to_slot: list[list[int]],
    slot_to_source: list[list[int]],
    req_ids: Sequence[int],
    topk_src_ids: Sequence[Sequence[int]],
    *,
    max_context_len: int,
    cache_capacity: int = 2048,
) -> tuple[
    list[list[int]],
    list[list[int]],
    list[int],
    list[list[int]],
    list[list[int]],
    list[int],
]:
    """Plan one Q1 batch, mutating the two maps in place.

    Misses are assigned to the lowest free slot first, then to the lowest slot
    not protected by a hit from this TopK. Output rows contain sorted misses
    followed by sorted hits. Invalid request IDs produce all-sentinel outputs
    and do not touch either map.
    """
    if len(req_ids) != len(topk_src_ids):
        raise ValueError("req_ids and topk_src_ids batch dimensions must match")
    if any(len(row) != cache_capacity for row in topk_src_ids):
        raise ValueError(f"topk rows must have length {cache_capacity}")

    topk_out = [[-1] * cache_capacity for _ in req_ids]
    dst_out = [[-1] * cache_capacity for _ in req_ids]
    miss_counts = [0] * len(req_ids)
    miss_src_out = [[-1] * cache_capacity for _ in req_ids]
    miss_dst_out = [[-1] * cache_capacity for _ in req_ids]
    request_miss_counts = [0] * len(req_ids)

    real_req_count = len(source_to_slot)
    for batch_idx, req_id in enumerate(req_ids):
        if req_id < 0 or req_id >= real_req_count:
            continue
        src_map = source_to_slot[req_id]
        inverse = slot_to_source[req_id]
        sources = sorted(
            {src for src in topk_src_ids[batch_idx] if 0 <= src < max_context_len}
        )
        hits: list[tuple[int, int]] = []
        misses: list[int] = []
        protected: set[int] = set()
        for src in sources:
            slot = src_map[src] if src < len(src_map) else -1
            if 0 <= slot < cache_capacity and inverse[slot] == src:
                hits.append((src, slot))
                protected.add(slot)
            else:
                if src < len(src_map):
                    src_map[src] = -1
                misses.append(src)

        assigned: list[tuple[int, int]] = []
        for src in misses:
            slot = next((i for i, owner in enumerate(inverse) if owner < 0), None)
            if slot is None:
                slot = next(
                    (i for i in range(cache_capacity) if i not in protected), None
                )
            if slot is None:
                # All slots are protected by requested hits: keep this source
                # absent from the device cache and emit a sentinel destination.
                assigned.append((src, -1))
                continue
            old_src = inverse[slot]
            if old_src >= 0 and old_src < len(src_map) and src_map[old_src] == slot:
                src_map[old_src] = -1
            inverse[slot] = src
            src_map[src] = slot
            protected.add(slot)
            assigned.append((src, slot))

        miss_counts[batch_idx] = len(misses)
        request_miss_counts[batch_idx] = len(misses)
        for i, (src, slot) in enumerate(assigned):
            topk_out[batch_idx][i] = src
            dst_out[batch_idx][i] = slot
            miss_src_out[batch_idx][i] = src
            miss_dst_out[batch_idx][i] = slot
        offset = len(assigned)
        for i, (src, slot) in enumerate(hits):
            topk_out[batch_idx][offset + i] = src
            dst_out[batch_idx][offset + i] = slot
    return (
        topk_out,
        dst_out,
        miss_counts,
        miss_src_out,
        miss_dst_out,
        request_miss_counts,
    )


def plan_sparse_kv_cache(
    source_to_slot: torch.Tensor,
    slot_to_source: torch.Tensor,
    req_ids: torch.Tensor,
    topk_input: torch.Tensor,
    topk_src_ids: torch.Tensor,
    topk_dst_slots: torch.Tensor,
    topk_miss_counts: torch.Tensor,
    miss_src_ids: torch.Tensor,
    miss_dst_slots: torch.Tensor,
    miss_counts: torch.Tensor,
    *,
    max_context_len: int,
    cache_capacity: int = 2048,
) -> None:
    """Dispatch the in-place NPU cache planner into preallocated buffers."""
    torch.ops.npu.plan_sparse_kv_cache(
        source_to_slot,
        slot_to_source,
        req_ids,
        topk_input,
        topk_src_ids,
        topk_dst_slots,
        topk_miss_counts,
        miss_src_ids,
        miss_dst_slots,
        miss_counts,
        max_context_len,
        cache_capacity,
    )
