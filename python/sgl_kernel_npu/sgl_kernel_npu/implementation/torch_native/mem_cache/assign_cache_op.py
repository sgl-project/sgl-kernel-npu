"""Copy packed source segments into destination rows."""

import torch


def assign_cache_op(dst, src, dst_start, dst_end, src_start, src_end):
    carrier = {1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}.get(
        dst.element_size()
    )
    if carrier is None:
        return True
    # Reinterpret equal-width payloads without numeric conversion or allocating a copy.
    dst_bits = dst.view(carrier)
    src_bits = src.view(carrier)
    lengths = src_end[: dst.shape[0]] - src_start[: dst.shape[0]]
    rows = torch.repeat_interleave(
        torch.arange(dst.shape[0], device=dst.device), lengths
    )
    begin = torch.cumsum(lengths, 0) - lengths
    offsets = torch.arange(rows.numel(), device=dst.device) - begin[rows]
    dst_bits[rows, dst_start[rows] + offsets] = src_bits[src_start[rows] + offsets]
    return True
