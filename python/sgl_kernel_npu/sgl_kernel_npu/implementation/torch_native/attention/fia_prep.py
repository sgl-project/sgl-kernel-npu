"""Independent page table preparation for sparse FIA attention."""

import torch


def fia_prep(
    ti,
    seq_lens,
    per_query_req,
    req_to_token,
    block_size,
    block_table_out=None,
    actual_kvlen_out=None,
):
    rows, width = ti.shape
    table = (
        block_table_out
        if block_table_out is not None
        else torch.empty((rows, width), dtype=torch.int32, device=ti.device)
    )
    lengths = (
        actual_kvlen_out
        if actual_kvlen_out is not None
        else torch.empty((rows,), dtype=torch.int32, device=ti.device)
    )
    table.zero_()
    for row in range(rows):
        seq_len = int(seq_lens[row].item())
        absolute = max(seq_len - 1, 0)
        own = absolute // block_size
        request = int(per_query_req[row].item())
        blocks = ti[row].tolist()
        nonown = [block for block in blocks if block >= 0 and block != own]
        columns = req_to_token.shape[1]
        for column, block in enumerate(nonown):
            table[row, column] = (
                req_to_token[request, min(block * block_size, columns - 1)]
                // block_size
            )
        table[row, len(nonown)] = (
            req_to_token[request, min(own * block_size, columns - 1)] // block_size
        )
        lengths[row] = len(nonown) * block_size + (
            absolute % block_size + 1 if own in blocks else 0
        )
    return table, lengths
