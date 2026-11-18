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
    # The public API already receives selected destination rows and source offsets.
    for i in range(dst.shape[0]):
        dst_bits[i][dst_start[i] : dst_end[i]] = src_bits[src_start[i] : src_end[i]]
    return True
