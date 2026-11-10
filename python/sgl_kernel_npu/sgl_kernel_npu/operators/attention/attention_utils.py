"""Shared helpers for sparse-attention operator adapters."""

import torch


def cann_9_1_dspark_kv_indices_to_physical_slots(
    indices, lengths, q, table, cache, cu_q, layout
):
    """Map CANN 9.1 DSpark original-KV indices to physical cache slots.

    Used by the explicit sparse paths of sparse_flash_mla (A3) and
    mixed_quant_sparse_flash_mla (A5), whose native kernels read physical slots.

    Physical slot = table[request, index // page_size] * page_size
    + index % page_size. Negative and length-masked entries become -1.
    """
    if table is None:
        raise ValueError("cann_9_1 sparse original KV requires ori_block_table")
    if layout == "TND":
        if cu_q is None:
            raise ValueError("TND sparse original KV requires cu_seqlens_q")
        token = torch.arange(q.shape[0], device=q.device)
        batch = (token[:, None] >= cu_q[None, 1:]).sum(-1).clamp(max=table.shape[0] - 1)
    elif layout == "BSND":
        batch = torch.arange(q.shape[0], device=q.device).repeat_interleave(q.shape[1])
    else:
        raise RuntimeError("cann_9_1 sparse indices support TND or BSND queries")
    flat = indices.reshape(-1, indices.shape[-1]).long()
    valid = flat >= 0
    if lengths is not None:
        column = torch.arange(flat.shape[-1], device=q.device)
        valid = valid & (column < lengths.reshape(-1, 1))
    safe = flat.clamp(min=0, max=table.shape[1] * cache.shape[1] - 1)
    pages = table[batch[:, None], safe // cache.shape[1]].long()
    slots = pages * cache.shape[1] + safe % cache.shape[1]
    return torch.where(valid, slots, -1).to(torch.int32).reshape_as(indices)
