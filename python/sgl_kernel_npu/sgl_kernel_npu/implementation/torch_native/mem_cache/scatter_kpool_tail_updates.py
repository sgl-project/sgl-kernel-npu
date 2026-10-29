"""Independent PyTorch backend for scatter_kpool_tail_updates."""

import torch


def scatter_kpool_tail_updates(
    pool,
    chunk_k: torch.Tensor,
    chunk_score: torch.Tensor,
    tail_k: torch.Tensor,
    tail_score: torch.Tensor,
    req_pool_idx: torch.Tensor,
    dst_logical_start: torch.Tensor,
    chunk_src_start: torch.Tensor,
    n_write: torch.Tensor,
):
    """Pure Torch reference for the scatter kernel, without GPU Triton."""
    pool_size = pool.index_kpool
    slots = torch.arange(pool_size, dtype=torch.int64, device=chunk_k.device).unsqueeze(
        0
    )
    write_slots = slots < n_write.to(torch.int64).unsqueeze(1)
    requests = req_pool_idx.to(torch.int64).unsqueeze(1).expand(-1, pool_size)
    destinations = (
        dst_logical_start.to(torch.int64).unsqueeze(1) + slots
    ) % tail_k.shape[1]
    sources = chunk_src_start.to(torch.int64).unsqueeze(1) + slots
    selected_requests = requests[write_slots]
    selected_destinations = destinations[write_slots]
    selected_sources = sources[write_slots]
    tail_k[selected_requests, selected_destinations] = chunk_k[selected_sources]
    tail_score[selected_requests, selected_destinations] = chunk_score[selected_sources]
