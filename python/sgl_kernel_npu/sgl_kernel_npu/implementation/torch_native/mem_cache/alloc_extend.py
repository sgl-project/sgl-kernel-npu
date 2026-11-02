"""Vectorized paged allocation on the input Tensor device."""

import torch


def alloc_extend(
    pre_lens, seq_lens, last_loc, free_pages, page_size, out_indices, values
):
    lengths = seq_lens - pre_lens
    row_ids = torch.repeat_interleave(
        torch.arange(pre_lens.numel(), device=pre_lens.device), lengths
    )
    output_starts = torch.cumsum(lengths, 0) - lengths
    within = (
        torch.arange(row_ids.numel(), device=pre_lens.device) - output_starts[row_ids]
    )
    prefix_pages = (pre_lens + page_size - 1) // page_size
    boundary = prefix_pages * page_size
    reuse_lengths = torch.minimum(seq_lens, boundary) - pre_lens
    reuse = within < reuse_lengths[row_ids]
    result = torch.empty_like(row_ids, dtype=out_indices.dtype)
    result[reuse] = last_loc[row_ids[reuse]] + 1 + within[reuse]
    new_pages = (seq_lens + page_size - 1) // page_size - prefix_pages
    page_starts = torch.cumsum(new_pages, 0) - new_pages
    new_rows = row_ids[~reuse]
    new_offsets = within[~reuse] - reuse_lengths[new_rows]
    result[~reuse] = (
        free_pages[page_starts[new_rows] + new_offsets // page_size] * page_size
        + new_offsets % page_size
    )
    out_indices[: result.numel()].copy_(result)
    # Match the final-request partial-page write condition.
    if pre_lens.numel() and bool(
        (seq_lens[-1] > boundary[-1]) & (seq_lens[-1] % page_size != 0)
    ):
        values[0].copy_(new_pages.sum())
