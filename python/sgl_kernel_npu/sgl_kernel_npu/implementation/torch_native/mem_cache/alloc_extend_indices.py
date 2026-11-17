import torch


def alloc_extend_indices(
    pre_lens,
    seq_lens,
    last_loc,
    free_pages,
    page_size,
    out_indices,
    max_num_extend_tokens,
):
    cursor = 0
    page_cursor = 0
    for prefix, total, last in zip(
        pre_lens.tolist(), seq_lens.tolist(), last_loc.tolist()
    ):
        count = total - prefix
        old = min(total, (prefix + page_size - 1) // page_size * page_size) - prefix
        out_indices[cursor : cursor + old] = torch.arange(
            last + 1, last + 1 + old, device=out_indices.device, dtype=out_indices.dtype
        )
        remaining = count - old
        offsets = torch.arange(remaining, device=out_indices.device)
        out_indices[cursor + old : cursor + count] = (
            free_pages[page_cursor + offsets // page_size] * page_size
            + offsets % page_size
        )
        page_cursor += (remaining + page_size - 1) // page_size
        cursor += count
