"""Reference for merging per-partition attention and log-sum-exp."""

import torch


def attention_update(lse, local_out, update_type):
    if update_type not in (0, 1):
        raise ValueError("update_type must be 0 or 1")
    lse = torch.stack(tuple(lse)).float()
    partial = torch.stack(tuple(local_out))
    weights = lse.softmax(dim=0).masked_fill(lse.isneginf().all(dim=0), 0)
    merged = (weights.unsqueeze(-1) * partial.float()).sum(dim=0).to(partial.dtype)
    if update_type == 1:
        return merged, lse.logsumexp(dim=0)
    return merged, None
