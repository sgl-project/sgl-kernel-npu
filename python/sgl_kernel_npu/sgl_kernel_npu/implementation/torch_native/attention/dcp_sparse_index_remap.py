"""PyTorch reference for stable DCP sparse index compaction."""

import torch


def dcp_sparse_index_remap(indices, dcp_size, dcp_rank, page_size):
    pages = indices // page_size
    keep = (indices >= 0) & (pages % dcp_size == dcp_rank)
    local = (pages // dcp_size) * page_size + indices % page_size

    # Assign retained entries to the front and rejected entries to the tail.
    # Every destination appears exactly once per row, so scatter is deterministic.
    kept = keep.cumsum(dim=1)
    columns = torch.arange(indices.shape[1], device=indices.device)
    destination = torch.where(keep, kept - 1, kept[:, -1:] + columns - kept)
    values = torch.where(keep, local, -1)
    return torch.empty_like(indices).scatter_(1, destination, values)
