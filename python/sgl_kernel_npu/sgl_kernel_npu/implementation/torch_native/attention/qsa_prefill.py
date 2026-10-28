"""Independent PyTorch reference for ratio-4 D256 QSA prefill main attention.

This implementation expands the selected logical blocks, maps them through the
request table and evaluates dense softmax attention over the resulting token
list with float32 accumulation. It shares no code with the Ascend C pipeline:
there is no query packing, no sharing-group detection, no run descriptor and no
separate causal-tail merge, so it is usable as an accuracy reference.

Rows are processed in chunks because the gathered key/value tensors are
proportional to rows x 2052 x KV x 256.
"""

import torch

# Bound the gathered key and value tensors per chunk to roughly 256 MiB.
_CHUNK_BYTES = 256 << 20


def _chunk_rows(columns, dim, kv_heads):
    """Rows per chunk, from the tensors held at once per (column, dim) pair.

    Gathering keys and values keeps every KV head in bfloat16, 2 * kv_heads * 2
    bytes, and the KV head being processed is also held as a float32 copy of
    both, 2 * 4 bytes. The float32 logits and weights add group / dim of that,
    which is under 5% at these shapes and is ignored.
    """
    return max(1, _CHUNK_BYTES // (columns * dim * (4 * kv_heads + 8)))


def _selected_positions(blocks, length, base, rows_begin, device):
    """Return logical token positions and their validity for one row chunk.

    Each nonnegative selected block expands into four consecutive positions
    bounded by ``length``. Every row then appends its own causal remainder of
    0--3 tokens. Selection order and duplicates are preserved; invalid entries
    stay in place and are masked instead of being compacted away.
    """
    rows = blocks.shape[0]
    offsets = torch.arange(4, device=device, dtype=torch.int64)
    selected = blocks.to(torch.int64)
    positions = selected[:, :, None] * 4 + offsets
    valid = (selected[:, :, None] >= 0) & (positions < length)
    positions = positions.reshape(rows, -1)
    valid = valid.reshape(rows, -1)
    visible = (
        base + rows_begin + 1 + torch.arange(rows, device=device, dtype=torch.int64)
    )
    remainder = visible % 4
    tail = (visible - remainder)[:, None] + offsets
    tail_valid = (offsets < remainder[:, None]) & (tail >= 0) & (tail < length)
    return (
        torch.cat((positions, tail), dim=1),
        torch.cat((valid, tail_valid), dim=1),
    )


def qsa_prefill(q, k, v, blocks, request_table, request_row, length, base, scale):
    """Dense float32 attention over the same selected tokens as the kernel.

    Inputs follow the public operator's contract and are not modified. The
    result is a newly allocated tensor with q's shape, dtype and device.
    """
    rows, heads, dim = q.shape
    kv_heads = k.shape[1]
    group = heads // kv_heads
    out = torch.zeros_like(q)
    if rows == 0 or length == 0:
        return out
    capacity = k.shape[0]
    device = q.device
    # Reading one table row keeps the operator free of device-to-host copies
    # and accepts a request table whose rows are not unit-strided.
    table_row = request_table.index_select(0, request_row.reshape(1)).reshape(-1)
    columns = 4 * blocks.shape[1] + 4
    chunk = _chunk_rows(columns, dim, kv_heads)
    for begin in range(0, rows, chunk):
        end = min(begin + chunk, rows)
        positions, valid = _selected_positions(
            blocks[begin:end], length, base, begin, device
        )
        safe = torch.where(valid, positions, torch.zeros_like(positions))
        slots = table_row.index_select(0, safe.reshape(-1)).reshape(safe.shape)
        slots = slots.to(torch.int64)
        valid = valid & (slots >= 0) & (slots < capacity)
        slots = torch.where(valid, slots, torch.zeros_like(slots))
        gathered = slots.reshape(-1)
        keys = k.index_select(0, gathered).reshape(end - begin, -1, kv_heads, dim)
        values = v.index_select(0, gathered).reshape(end - begin, -1, kv_heads, dim)
        for kv in range(kv_heads):
            queries = q[begin:end, kv * group : (kv + 1) * group, :].float()
            logits = torch.matmul(
                queries, keys[:, :, kv, :].float().transpose(1, 2)
            ) * float(scale)
            logits = logits.masked_fill(~valid[:, None, :], -float("inf"))
            weights = torch.softmax(logits, dim=-1)
            # A row without any valid token produces an all -inf row; its
            # attention output is zero rather than NaN.
            weights = torch.nan_to_num(weights, nan=0.0)
            result = torch.matmul(weights, values[:, :, kv, :].float())
            out[begin:end, kv * group : (kv + 1) * group, :] = result.to(q.dtype)
    return out
