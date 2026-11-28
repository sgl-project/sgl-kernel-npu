"""Eager PyTorch reference for Ascend950 packed C8 CopySFA."""

from itertools import pairwise

import torch

_PAGE = 128
_LATENT = 512
_ROPE_BYTES_END = 640
_ROW_BYTES = 656


def _physical_rows(pages, logical, rows):
    blocks = logical // _PAGE
    if ((logical < 0) | (blocks >= pages.numel())).any():
        raise ValueError("A logical KV row is outside its block table")
    physical = pages[blocks] * _PAGE + logical % _PAGE
    if ((physical < 0) | (physical >= rows)).any():
        raise ValueError("A KV block table entry points outside packed storage")
    return physical


def _packed_rows(tensor):
    if tensor.dtype != torch.int8 or tensor.size(-1) != _ROW_BYTES:
        raise ValueError("C8 KV requires INT8 views of complete 656-byte records")
    return tensor.cpu().view(-1, _ROW_BYTES)


def _dequantize(packed, dtype):
    latent = packed[:, :_LATENT].contiguous().view(torch.float8_e4m3fn)
    rope = packed[:, _LATENT:_ROPE_BYTES_END].contiguous().view(torch.bfloat16)
    scales = packed[:, _ROPE_BYTES_END:].contiguous().view(torch.float32)
    latent = (latent.float().reshape(-1, 4, 128) * scales.unsqueeze(-1)).flatten(1)
    return latent.to(dtype).float(), rope.to(dtype).float()


def _stage_short(hbm, host, pages, length, positions, valid, current, history, extent):
    first = length - len(positions)
    if first < 0 or length > history.numel():
        return -1
    begin = max(0, min(extent, first))
    source = history[begin:first]
    if ((source < 0) | (source >= host.size(0))).any():
        raise ValueError("Short resident history is outside packed Host storage")
    destination = _physical_rows(
        pages, torch.arange(begin, first, device="cpu"), hbm.size(0)
    )
    hbm[destination] = host[source]
    live = valid & (positions >= first) & (positions < length)
    hbm[_physical_rows(pages, positions[live], hbm.size(0))] = current[live]
    contiguous = valid.all() and positions.tolist() == list(range(first, length))
    return length if contiguous else -1


@torch.no_grad()
def fused_quant_scatter_copy_sparse_flash_attention_v2(
    query_nope,
    query_rope,
    hbm_kv_bytes,
    dram_kv_bytes,
    dram_kv_dev_ptr,
    hbm_block_table,
    dram_block_table,
    topk_dst_slots,
    topk_src_ids,
    topk_miss_counts,
    actual_seq_lengths_query,
    actual_seq_lengths_kv,
    num_cache_tokens,
    miss_src_ids,
    miss_dst_slots,
    miss_counts,
    scale_value,
    attention_out,
    *,
    current=None,
    positions=None,
    current_valid=None,
    request_slots=None,
    req_to_token=None,
    resident_end=None,
    native_lim_plan=False,
):
    """Apply the complete request copy plan and attend to the supplied slots."""
    host = _packed_rows(dram_kv_bytes)
    hbm = _packed_rows(hbm_kv_bytes)
    staging = (
        current,
        positions,
        current_valid,
        request_slots,
        req_to_token,
        resident_end,
    )
    if any((tensor is None) != (current is None) for tensor in staging):
        raise ValueError("The six resident staging tensors must be supplied together")
    staged = current is not None
    if staged:
        current = _packed_rows(current)
        pos, valid, requests, history, extents = (
            tensor.cpu() for tensor in staging[1:]
        )

    query_dtype = query_nope.dtype
    query = query_nope.cpu().float()
    rope = query_rope.cpu().float()
    output = torch.zeros_like(query)
    hbm_pages, host_pages, slots, sources, destinations = (
        tensor.cpu().long()
        for tensor in (
            hbm_block_table,
            dram_block_table,
            topk_dst_slots,
            miss_src_ids,
            miss_dst_slots,
        )
    )
    ends = [0, *actual_seq_lengths_query.tolist()]
    if ends[-1] != query.size(0) or any(
        not 1 <= end - start <= 16 for start, end in pairwise(ends)
    ):
        raise ValueError(
            "CopySFA requires 1–16 rows per request in cumulative query ends"
        )

    for batch, ((start, end), length, cache_size, count) in enumerate(
        zip(
            pairwise(ends),
            actual_seq_lengths_kv.tolist(),
            num_cache_tokens.tolist(),
            miss_counts.tolist(),
            strict=True,
        )
    ):
        if length == 0:
            continue
        if not (length > 0 and cache_size >= 0 and 0 <= count <= sources.size(1)):
            raise ValueError("Invalid CopySFA length or request miss count")
        pages = hbm_pages[batch]
        use_current = staged and 0 < int(requests[batch]) < history.size(0)
        if use_current:
            request = int(requests[batch])
            if cache_size == 0:
                extent = _stage_short(
                    hbm,
                    host,
                    pages,
                    length,
                    pos[start:end],
                    valid[start:end],
                    current[start:end],
                    history[request].long(),
                    int(extents[request]),
                )
            else:
                extent = -1
            extents[request] = extent
        if count:
            logical = sources[batch, :count]
            target = _physical_rows(pages, destinations[batch, :count], hbm.size(0))
            from_current = torch.zeros_like(logical, dtype=torch.bool)
            if use_current:
                first = int(pos[start])
                delta = logical - first
                within = (first >= 0) & (0 <= delta) & (delta < end - start)
                rows = start + delta.clamp(0, end - start - 1)
                from_current = within & valid[rows] & (pos[rows] == logical)
                hbm[target[from_current]] = current[rows[from_current]]
            from_host = ~from_current
            physical = _physical_rows(
                host_pages[batch], logical[from_host], host.size(0)
            )
            hbm[target[from_host]] = host[physical]

        for token in range(start, end):
            visible = length - (end - token - 1)
            selected = slots[token].flatten()
            selected = selected[(0 <= selected) & (selected < visible)]
            if not selected.numel():
                continue
            physical = _physical_rows(pages, selected, hbm.size(0))
            latent_kv, rope_k = _dequantize(hbm[physical], query_dtype)
            score = query[token] @ latent_kv.T + rope[token] @ rope_k.T
            output[token] = (score * scale_value).softmax(dim=-1) @ latent_kv

    hbm_kv_bytes.copy_(hbm.view_as(hbm_kv_bytes))
    attention_out.copy_(output)
    if staged:
        resident_end.copy_(extents)
