"""Minimal single-token routing metadata for single-node GLM decode."""

import torch
import triton
import triton.language as tl


@triton.jit
def _decode_dispatch_layout(topk_ids, counts, send_indices):
    slots = tl.arange(0, 8)
    experts = tl.arange(0, 512)
    ids = tl.load(topk_ids + slots)
    valid = (ids >= 0) & (ids < 288)
    hits = (ids[:, None] == experts[None, :]) & valid[:, None]
    histogram = tl.sum(hits.to(tl.int32), axis=0)
    tl.store(counts + experts, histogram, experts < 288)
    # Stable expert-local positions. Unique TopK IDs make these all zero,
    # but computing the tiny prefix also handles repeated valid IDs.
    previous = (ids[:, None] == ids[None, :]) & (slots[None, :] < slots[:, None])
    positions = tl.sum(previous.to(tl.int32), axis=1)
    tl.store(send_indices + slots, tl.where(valid, positions, 0))


def decode_dispatch_layout(topk_ids: torch.Tensor):
    """Return only the expert counts and send indices consumed by dispatch.

    Full writes replace the five zeros and generic multi-node metadata. Inputs
    and outputs remain on the current device stream and support graph replay.
    """
    if (
        topk_ids.shape != (1, 8)
        or topk_ids.dtype != torch.int32
        or not topk_ids.is_contiguous()
    ):
        raise ValueError(
            "compact decode layout requires contiguous INT32 indices [1, 8]"
        )
    counts = torch.empty(288, device=topk_ids.device, dtype=torch.int32)
    send_indices = torch.empty_like(topk_ids)
    _decode_dispatch_layout[(1,)](topk_ids, counts, send_indices)
    return counts, send_indices
