"""
Copyright 2025 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch


def alloc_extend(
    pre_lens, seq_lens, last_loc, free_pages, page_size, out_indices, values
):
    if any(
        t.dtype != torch.int64
        for t in (pre_lens, seq_lens, last_loc, free_pages, out_indices)
    ):
        raise ValueError("Only support int64 input dtype")
    extend_lens = seq_lens - pre_lens
    end_pos = torch.cumsum(extend_lens, 0)
    start_pos = end_pos - extend_lens

    extend_num_tokens = out_indices.shape[0]
    if extend_num_tokens == 0:
        return

    j = torch.arange(extend_num_tokens, device=pre_lens.device, dtype=torch.int64)
    owner = torch.searchsorted(end_pos, j, right=True)
    local = j - start_pos[owner]
    last_loc_g = last_loc[owner]

    ceil_prefix = (pre_lens + page_size - 1) // page_size * page_size
    floor_seq = seq_lens // page_size * page_size

    if free_pages.numel() == 0:
        # Only valid when no request needs a new page; nothing below indexes
        # the empty pool, so a short pool would silently return garbage.
        ceil_seq = (seq_lens + page_size - 1) // page_size * page_size
        assert torch.all(ceil_seq == ceil_prefix), (
            "alloc_extend: free_pages is empty but the batch requires "
            "new pages; caller must ensure pool >= demand"
        )
        out_indices.copy_(last_loc_g + 1 + local)
        return

    num1 = torch.clamp(seq_lens, max=ceil_prefix) - pre_lens
    done_after_1 = (pre_lens + num1) == seq_lens
    num2 = torch.where(done_after_1, torch.zeros_like(num1), floor_seq - ceil_prefix)
    num3 = torch.where(done_after_1, torch.zeros_like(num1), seq_lens - floor_seq)

    full_pages = num2 // page_size
    need_extra_page = (num3 > 0).to(torch.int64)
    pages_per_req = full_pages + need_extra_page
    end_new_pages = torch.cumsum(pages_per_req, 0)
    start_new_pages = end_new_pages - pages_per_req

    num1_g = num1[owner]
    num2_g = num2[owner]
    start_new_pages_g = start_new_pages[owner]
    end_new_pages_g = end_new_pages[owner]

    is_phase1 = local < num1_g
    is_phase2 = (~is_phase1) & (local < num1_g + num2_g)

    val_phase1 = last_loc_g + 1 + local

    rel2 = torch.clamp(local - num1_g, min=0)
    # torch.where below evaluates both branches per slot, so dead-lane page
    # indices must stay in range even where their phase is never selected.
    page_idx2 = torch.clamp(
        start_new_pages_g + rel2 // page_size, min=0, max=free_pages.numel() - 1
    )
    pos_in_page2 = rel2 % page_size
    val_phase2 = free_pages[page_idx2] * page_size + pos_in_page2

    rel3 = torch.clamp(local - num1_g - num2_g, min=0)
    page_idx3 = torch.clamp(end_new_pages_g - 1, min=0, max=free_pages.numel() - 1)
    val_phase3 = free_pages[page_idx3] * page_size + rel3

    out = torch.where(
        is_phase1, val_phase1, torch.where(is_phase2, val_phase2, val_phase3)
    )
    out_indices.copy_(out)

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
