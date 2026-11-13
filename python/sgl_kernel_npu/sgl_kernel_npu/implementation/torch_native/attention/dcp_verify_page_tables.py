"""PyTorch reference for in-place DCP Verify page tables."""

import torch


def dcp_verify_page_tables(
    slots, requests, ordinary, local, active_cols, query_len, page_size, dcp_size
):
    ordinary.zero_()
    local.zero_()
    if active_cols == 0 or slots.shape[0] == 0 or requests.numel() == 0:
        return

    valid = (requests >= 0) & (requests < slots.shape[0])
    safe_requests = torch.where(valid, requests, 0).to(torch.int64)
    page_starts = slots[:, : active_cols * page_size : page_size]
    pages = page_starts.index_select(0, safe_requests) // page_size
    pages = torch.where(valid[:, None], pages, 0)
    ordinary[:, :active_cols].copy_(pages)

    local_pages = pages[:, ::dcp_size] // dcp_size
    local.view(requests.numel(), query_len, local.shape[1])[
        :, :, : local_pages.shape[1]
    ].copy_(local_pages[:, None, :])
