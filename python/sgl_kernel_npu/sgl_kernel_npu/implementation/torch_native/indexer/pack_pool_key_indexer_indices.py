# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PyTorch reference for causal pooled-index packing."""

import torch


def pack_pool_key_indexer_indices(
    indices, positions, query_ends, topk, pool_size, *, output_buffer=None
):
    """Discard artificial history candidates and restore each query's causal tail."""
    output = indices if output_buffer is None else output_buffer[: indices.shape[0]]
    width = topk + pool_size - 1
    if indices.shape[0] == 0:
        return output[:, :width]
    valid = torch.arange(indices.shape[0], device=indices.device) < query_ends[-1]
    visible = torch.where(valid, positions, 0).to(torch.int64).unsqueeze(1) + 1
    tail_start = torch.div(visible, pool_size, rounding_mode="floor") * pool_size
    history = tail_start.clamp_max(topk)
    cols = torch.arange(width, device=indices.device).unsqueeze(0)
    # Causal CANN supplies valid history indices.
    result = torch.where(cols < history, indices[:, :width], -1)
    tail_col = cols - history
    tail_valid = (tail_col >= 0) & (tail_col < visible - tail_start)
    result = torch.where(tail_valid, (tail_start + tail_col).to(torch.int32), result)
    result = torch.where(valid.unsqueeze(1), result, -1)
    output.fill_(-1)
    output[:, :width].copy_(result)
    return output[:, :width]
