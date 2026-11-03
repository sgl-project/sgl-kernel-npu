"""Independent PyTorch scatter reference for in-place token-table updates."""

import torch


def update_oe_token_table(
    tokens,
    req_lens,
    row_indices,
    column_starts,
    ignore_tokens,
    batch_size,
    max_context_len,
    oe_token_table,
):
    """Filter the packed input tokens and write only the requested slices."""
    values = tokens
    if ignore_tokens.numel():
        ignored = (tokens[:, None] == ignore_tokens[None, :]).any(dim=-1)
        values = torch.where(ignored, -1, tokens)

    lengths = req_lens[:batch_size].to(torch.int64)
    requests = torch.repeat_interleave(
        torch.arange(batch_size, device=tokens.device),
        lengths,
        output_size=tokens.numel(),
    )
    packed_starts = lengths.cumsum(0) - lengths
    local_positions = (
        torch.arange(tokens.numel(), device=tokens.device) - packed_starts[requests]
    )
    destinations = (
        row_indices[requests] * max_context_len
        + column_starts[requests]
        + local_positions
    )
    oe_token_table.view(-1)[destinations] = values
    return oe_token_table
