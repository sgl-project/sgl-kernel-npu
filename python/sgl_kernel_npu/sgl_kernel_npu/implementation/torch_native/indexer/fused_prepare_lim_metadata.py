"""Independent PyTorch reference for LIM metadata."""

import torch


def fused_prepare_lim_metadata(
    *,
    seq_lens,
    request_slots,
    last_prefix,
    block_table,
    out_block_table,
    safe_slots,
    lengths,
    sparse_rows,
    valid_rows,
    request_states,
    next_prefixes,
    cache_sizes,
    capacity,
    request_rows,
    query_width=1,
    positions=None,
    resident_safe_slots=None,
    request_to_resident=None,
    resident_rows=None,
):
    rows = request_rows if resident_rows is None else resident_rows
    out_block_table.zero_()
    cols = min(block_table.shape[1], out_block_table.shape[1])
    out_block_table[:, :cols].copy_(block_table[:, :cols])
    raw = request_slots.to(torch.int64)
    active = seq_lens.to(torch.int64)
    length = (
        active
        if positions is None
        else positions.reshape(-1, query_width)[:, -1].to(torch.int64) + 1
    )
    valid_request = (raw > 0) & (raw < request_rows)
    safe_index = torch.where(valid_request, raw, 0)
    resident = (
        raw
        if request_to_resident is None
        else request_to_resident[safe_index].to(torch.int64)
    )
    valid = (
        valid_request & (resident > 0) & (resident < rows) & (active > 0) & (length > 0)
    )
    safe = torch.where(valid, raw, request_rows)
    safe_resident = torch.where(valid, resident, rows)
    length = torch.where(valid, length, 0).to(torch.int32)
    sparse = valid & (length >= capacity)
    previous = last_prefix[safe_resident].to(torch.int32)
    warm = (previous >= 0) & (length >= previous)
    safe_slots.copy_(safe)
    if resident_safe_slots is not None:
        resident_safe_slots.copy_(safe_resident)
    else:
        safe_slots.copy_(safe_resident)
    lengths.copy_(length)
    sparse_rows.copy_(sparse)
    valid_rows.copy_(valid)
    request_states.copy_(torch.where(sparse, torch.where(warm, -1, -2), -3))
    next_prefixes.copy_(torch.where(sparse, length, -1))
    cache_sizes.copy_(torch.where(sparse, capacity, 0))
