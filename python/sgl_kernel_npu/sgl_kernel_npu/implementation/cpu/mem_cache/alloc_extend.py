"""CPU page allocation with caller-owned output buffers."""

import torch


def alloc_extend(
    pre_lens, seq_lens, last_loc, free_pages, page_size, out_indices, values
):
    extend_lens = seq_lens - pre_lens
    total_extend_tokens = extend_lens.sum().item()

    part1_ends = torch.min(
        seq_lens, ((pre_lens + page_size - 1) // page_size) * page_size
    )
    part1_lens = (part1_ends - pre_lens).to(device="cpu")

    part2_starts = ((pre_lens + page_size - 1) // page_size) * page_size
    part2_ends = (seq_lens // page_size) * page_size
    part2_lens = (part2_ends - part2_starts).to(device="cpu")

    part3_lens = (seq_lens % page_size).to(device="cpu")

    # Step 3: Allocate and fill
    global_token_idx = 0
    global_page_idx = 0
    page_size_range = torch.arange(page_size, device="cpu")

    extend_lens = extend_lens.to(device="cpu")
    last_loc = last_loc.to(device="cpu")
    free_pages = free_pages.to(device="cpu")

    for i in range(len(pre_lens)):
        current_token_idx = 0
        if extend_lens[i] == 0:
            continue

        # Part 1: Fill the partial page at the end of pre_len
        if part1_lens[i] > 0:
            indices = last_loc[i] + 1 + torch.arange(part1_lens[i], device="cpu")
            out_indices[global_token_idx : global_token_idx + part1_lens[i]] = indices
            global_token_idx += part1_lens[i]
            current_token_idx += part1_lens[i]
        if (
            global_token_idx >= total_extend_tokens
            or current_token_idx >= extend_lens[i]
        ):
            continue

        # Part 2:Fill full new pages
        if part2_lens[i] > 0:
            num_full_pages = part2_lens[i] // page_size
            page_id = free_pages[global_page_idx : global_page_idx + num_full_pages]
            slots = (page_id[:, None] * page_size + page_size_range).flatten()
            valid_slots = slots[
                : min(len(slots), total_extend_tokens - global_token_idx)
            ]
            out_indices[global_token_idx : global_token_idx + len(valid_slots)] = (
                valid_slots
            )
            global_token_idx += len(valid_slots)
            current_token_idx += len(valid_slots)
            global_page_idx += num_full_pages

        # Part 3: Fill the final partial page
        if (
            part3_lens[i] > 0
            and global_token_idx < total_extend_tokens
            and current_token_idx < extend_lens[i]
        ):
            page = free_pages[global_page_idx]
            slots = page * page_size + torch.arange(part3_lens[i], device="cpu")
            out_indices[global_token_idx : global_token_idx + part3_lens[i]] = slots
            global_token_idx += part3_lens[i]
            current_token_idx += part3_lens[i]
            global_page_idx += 1
    boundary = (pre_lens + page_size - 1) // page_size * page_size
    if pre_lens.numel() and bool(
        (seq_lens[-1] > boundary[-1]) & (seq_lens[-1] % page_size != 0)
    ):
        values[0].copy_(
            (
                (seq_lens + page_size - 1) // page_size
                - (pre_lens + page_size - 1) // page_size
            ).sum()
        )
