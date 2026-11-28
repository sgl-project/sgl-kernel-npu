"""Eager PyTorch reference for split-host KV staging and sparse attention."""

from itertools import pairwise

import torch

_PAGE_SIZE = 128
_CKV_DIM = 512
_ROPE_DIM = 64
_TOPK = 2048


def _physical_rows(pages, logical_rows, capacity):
    if ((logical_rows < 0) | (logical_rows >= pages.numel() * _PAGE_SIZE)).any():
        raise ValueError("KV row is outside its page table")
    rows = pages[logical_rows // _PAGE_SIZE] * _PAGE_SIZE + logical_rows % _PAGE_SIZE
    if ((rows < 0) | (rows >= capacity)).any():
        raise ValueError("KV page table points outside storage")
    return rows


def _copy_rows(target, destination, host, host_pages, sources, current, first):
    if not sources.numel():
        return
    from_current = (
        (first <= sources) & (sources < first + current.size(0))
        if current is not None
        else torch.zeros_like(sources, dtype=torch.bool)
    )
    ckv, rope = target
    if from_current.any():
        rows = current[sources[from_current] - first]
        ckv[destination[from_current]] = rows[:, :_CKV_DIM]
        rope[destination[from_current]] = rows[:, _CKV_DIM:]
    from_host = ~from_current
    if from_host.any():
        rows = _physical_rows(host_pages, sources[from_host], host[0].size(0))
        ckv[destination[from_host]] = host[0][rows]
        rope[destination[from_host]] = host[1][rows]


@torch.no_grad()
def fused_scatter_copy_sparse_flash_attention_split(
    query,
    query_rope,
    hbm_ckv,
    hbm_rope,
    actual_seq_lengths_query,
    actual_seq_lengths_kv,
    num_cache_tokens,
    topk_src_ids,
    topk_dst_slots,
    topk_miss_counts,
    miss_src_ids,
    miss_dst_slots,
    miss_counts,
    hbm_block_table,
    dram_block_table,
    dram_kv_dev_ptr,
    attention_out,
    scale_value,
    *,
    full_visible=False,
    resident_refresh_tail=False,
    current=None,
    positions=None,
    valid=None,
    request_slots=None,
    req_to_token=None,
    resident_end=None,
    host_token_capacity=0,
    dram_rope_dev_ptr=0,
    dram_ckv=None,
    dram_rope=None,
):
    """Apply the plan in eager mode using explicit host CKV and RoPE tensors."""
    if dram_ckv is None or dram_rope is None:
        raise ValueError("torch_native requires dram_ckv and dram_rope tensors")
    if (
        dram_ckv.size(-1) != _CKV_DIM
        or dram_rope.size(-1) != _ROPE_DIM
        or dram_ckv.dtype != query.dtype
        or dram_rope.dtype != query.dtype
    ):
        raise ValueError(
            "Host CKV and RoPE must have widths 512 and 64 and match the query dtype"
        )
    host_ckv = dram_ckv.detach().cpu().reshape(-1, _CKV_DIM)
    host_rope = dram_rope.detach().cpu().reshape(-1, _ROPE_DIM)
    if host_ckv.size(0) != host_rope.size(
        0
    ) or not 0 <= host_token_capacity <= host_ckv.size(0):
        raise ValueError(
            "Host KV tensors must cover the same number of rows and host_token_capacity"
        )
    host_rows = host_token_capacity or host_ckv.size(0)
    host = host_ckv[:host_rows], host_rope[:host_rows]

    staging = (current, positions, valid, request_slots, req_to_token, resident_end)
    stage_current = current is not None
    if any((tensor is not None) != stage_current for tensor in staging):
        raise ValueError("The resident staging tensors must be supplied together")
    if stage_current:
        current, positions, valid, request_slots, req_to_token, ends = (
            tensor.detach().cpu() for tensor in staging
        )
        current = current.reshape(-1, _CKV_DIM + _ROPE_DIM)

    query, query_rope = (
        tensor.detach().cpu().float() for tensor in (query, query_rope)
    )
    ckv = hbm_ckv.detach().cpu().view(-1, _CKV_DIM)
    rope = hbm_rope.detach().cpu().view(-1, _ROPE_DIM)
    hbm = ckv, rope
    hbm_pages, host_pages, sources, slots, miss_sources, miss_slots = (
        tensor.detach().cpu().long()
        for tensor in (
            hbm_block_table,
            dram_block_table,
            topk_src_ids,
            topk_dst_slots,
            miss_src_ids,
            miss_dst_slots,
        )
    )
    per_query_misses = topk_miss_counts.tolist()
    spans = list(pairwise([0, *actual_seq_lengths_query.tolist()]))
    lengths = actual_seq_lengths_kv.tolist()
    capacities = num_cache_tokens.tolist()
    request_misses = miss_counts.tolist()
    if any(capacity < 0 for capacity in capacities):
        raise ValueError("num_cache_tokens must be non-negative")
    if any(count < 0 or count > miss_sources.size(1) for count in request_misses):
        raise ValueError("miss_counts exceeds the request copy plan")
    # Complete the union before attention when the native path cold-fills
    # or consumes a full-visible plan, including any refresh suffix.
    copy_union = any(
        cap > 0 and count > 0 and (full_visible or count >= cap)
        for cap, count in zip(capacities, request_misses, strict=True)
    )
    result = torch.zeros_like(query)
    for batch, ((start, end), length, capacity, count) in enumerate(
        zip(spans, lengths, capacities, request_misses, strict=True)
    ):
        if length <= 0:
            continue
        if not (0 <= start < end <= query.size(0)):
            raise ValueError("Cumulative query ends are outside the query tensor")
        pages, dram = hbm_pages[batch], host_pages[batch]
        current_rows, first = None, -1
        if stage_current:
            req = int(request_slots[batch])
            width = end - start
            first = int(positions[start])
            if (
                not 0 < req < req_to_token.size(0)
                or not 0 <= first <= req_to_token.size(1) - width
                or not valid[start:end].all()
                or positions[start:end].tolist() != list(range(first, first + width))
            ):
                raise ValueError("Invalid active resident staging metadata")
            current_rows = current[start:end]
            if capacity == 0:
                if first + width != length:
                    raise ValueError(
                        "Current rows must end at the short request's KV length"
                    )
                previous = max(0, min(int(ends[req]), first))
                history = req_to_token[req, previous:first].long()
                if ((history < 0) | (history >= host_rows)).any():
                    raise ValueError("Short resident history is outside host storage")
                restored = _physical_rows(
                    pages, torch.arange(previous, first, device="cpu"), ckv.size(0)
                )
                staged = _physical_rows(
                    pages, torch.arange(first, length, device="cpu"), ckv.size(0)
                )
                ckv[restored], rope[restored] = host[0][history], host[1][history]
                ckv[staged], rope[staged] = (
                    current_rows[:, :_CKV_DIM],
                    current_rows[:, _CKV_DIM:],
                )
                ends[req] = length
            else:
                ends[req] = -1

        if copy_union and count:
            destination = _physical_rows(pages, miss_slots[batch, :count], ckv.size(0))
            _copy_rows(
                hbm,
                destination,
                host,
                dram,
                miss_sources[batch, :count],
                current_rows,
                first,
            )

        for token in range(start, end):
            visible = max(0, length - (end - token - 1))
            if visible == 0:
                continue
            if capacity == 0:
                selected = torch.arange(visible, device="cpu")
            elif capacity >= _TOPK and (full_visible or visible >= capacity):
                if not copy_union:
                    n = per_query_misses[token]
                    if not 0 <= n <= _TOPK:
                        raise ValueError("topk_miss_counts exceeds the TopK width")
                    if n:
                        destination = _physical_rows(
                            pages, slots[token, :n], ckv.size(0)
                        )
                        _copy_rows(
                            hbm,
                            destination,
                            host,
                            dram,
                            sources[token, :n],
                            current_rows,
                            first,
                        )
                selected = slots[token]
                selected = selected[(selected >= 0) & (selected < capacity)]
                if not full_visible:
                    selected = torch.cat(
                        (selected, torch.arange(capacity, visible, device="cpu"))
                    )
            else:
                continue
            rows = _physical_rows(pages, selected, ckv.size(0))
            if rows.numel():
                kv = ckv[rows].float()
                kpe = rope[rows].float()
                scores = query[token] @ kv.T + query_rope[token] @ kpe.T
                result[token] = (scores * scale_value).softmax(dim=-1) @ kv

    hbm_ckv.copy_(ckv.view_as(hbm_ckv))
    hbm_rope.copy_(rope.view_as(hbm_rope))
    attention_out.copy_(result)
    if stage_current:
        resident_end.copy_(ends)
