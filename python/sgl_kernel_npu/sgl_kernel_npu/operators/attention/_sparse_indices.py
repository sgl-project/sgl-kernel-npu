"""Tensor-only logical-to-physical paged sparse index conversion."""

import torch

from ..utils._transformer_compat import OperatorUnavailableError


def logical_to_physical(indices, lengths, q, table, cache, cu_q, layout):
    """Resolve original-KV indices for the native DSpark ABI using tensor ops."""
    if table is None:
        raise ValueError("Legacy sparse original KV requires ori_block_table")
    if layout == "TND":
        if cu_q is None:
            raise ValueError("TND sparse original KV requires cu_seqlens_q")
        token = torch.arange(q.shape[0], device=q.device)
        batch = (token[:, None] >= cu_q[None, 1:]).sum(-1).clamp(max=table.shape[0] - 1)
    elif layout == "BSND":
        batch = torch.arange(q.shape[0], device=q.device).repeat_interleave(q.shape[1])
    else:
        raise OperatorUnavailableError(
            "Legacy sparse indices support TND or BSND queries"
        )
    flat = indices.reshape(-1, indices.shape[-1]).long()
    valid = flat >= 0
    if lengths is not None:
        column = torch.arange(flat.shape[-1], device=q.device)
        valid = valid & (column < lengths.reshape(-1, 1))
    safe = flat.clamp(min=0, max=table.shape[1] * cache.shape[1] - 1)
    pages = table[batch[:, None], safe // cache.shape[1]].long()
    slots = pages * cache.shape[1] + safe % cache.shape[1]
    return torch.where(valid, slots, -1).to(torch.int32).reshape_as(indices)
