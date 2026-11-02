"""CPU page allocation with caller-owned output buffers."""

import torch


def alloc_extend(
    pre_lens, seq_lens, last_loc, free_pages, page_size, out_indices, values
):
    output_offset = 0
    page_offset = 0
    prefixes = pre_lens.tolist()
    sequences = seq_lens.tolist()
    last_locations = last_loc.tolist()
    for row, (prefix, sequence, last) in enumerate(
        zip(prefixes, sequences, last_locations)
    ):
        count = sequence - prefix
        boundary = (prefix + page_size - 1) // page_size * page_size
        reuse = min(sequence, boundary) - prefix
        if reuse:
            out_indices[output_offset : output_offset + reuse].copy_(
                last + 1 + torch.arange(reuse, dtype=out_indices.dtype, device="cpu")
            )
        new_count = count - reuse
        if new_count:
            offsets = torch.arange(new_count, dtype=torch.int64, device="cpu")
            slots = (
                free_pages[page_offset + offsets // page_size] * page_size
                + offsets % page_size
            )
            out_indices[output_offset + reuse : output_offset + count].copy_(slots)
        pages = (sequence + page_size - 1) // page_size - (
            prefix + page_size - 1
        ) // page_size
        page_offset += pages
        output_offset += count
        # The page count is written only when the final request reaches a new partial page.
        if row == len(prefixes) - 1 and sequence > boundary and sequence % page_size:
            values[0] = page_offset
