"""Independent PyTorch reference for sparse KV offload row metadata."""

import torch


def prepare_sparse_kv_offload_rows(
    tokens,
    device,
    request_ids,
    seq_lens,
    positions,
    out_cache_loc,
    *,
    request_rows,
    context_len,
    host_capacity,
    query_width=1,
    extend_seq_lens=None,
    extend_prefix_lens=None,
    compact_prefill=False,
):
    source = torch.arange(tokens, dtype=torch.int64, device=device)
    destination = out_cache_loc[:tokens].to(torch.int64)
    batch = request_ids.numel()
    if not tokens:
        return (
            source,
            destination,
            source.clone(),
            torch.empty(0, dtype=torch.bool, device=device),
        )
    if extend_seq_lens is not None:
        if compact_prefill:
            ends = extend_seq_lens.cumsum(0)
            row = (source[:, None] >= ends[None, :]).sum(dim=1)
            starts = torch.cat((ends.new_zeros(1), ends))
            offset = source - starts[row.clamp(max=batch)]
        else:
            width = tokens // batch
            row = source // width
            offset = source % width
        in_batch = row < batch
        safe_row = row.clamp(max=batch - 1)
        req = torch.where(in_batch, request_ids[safe_row], 0)
        length = torch.where(in_batch, extend_seq_lens[safe_row], 0)
        prefix_ok = row < extend_prefix_lens.numel()
        safe_prefix = row.clamp(max=max(extend_prefix_lens.numel() - 1, 0))
        prefix = (
            torch.where(prefix_ok, extend_prefix_lens[safe_prefix], 0)
            if extend_prefix_lens.numel()
            else torch.zeros_like(row)
        )
        token_positions = (prefix + offset).to(torch.int64)
        valid = (
            (req > 0)
            & (offset < length)
            & (token_positions >= 0)
            & (token_positions < context_len)
        )
    else:
        row = source // query_width
        inside = source < batch * query_width
        safe_row = row.clamp(max=max(batch - 1, 0))
        req = (
            torch.where(inside, request_ids[safe_row], 0)
            if batch
            else torch.zeros_like(row)
        )
        length = (
            torch.where(inside, seq_lens[safe_row], 0)
            if batch
            else torch.zeros_like(row)
        )
        token_positions = torch.where(inside, positions[:tokens], -1).to(torch.int64)
        valid = (
            inside
            & (req > 0)
            & (req < request_rows)
            & (length > 0)
            & (token_positions >= 0)
            & (token_positions < context_len)
        )
    valid &= (destination >= 0) & (destination < host_capacity)
    return source, destination, token_positions, valid
