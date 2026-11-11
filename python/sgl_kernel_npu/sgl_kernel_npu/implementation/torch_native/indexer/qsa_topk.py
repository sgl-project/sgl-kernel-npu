"""Independent exact CPU selection; ties may choose different valid indices."""

import torch


def fast_topk(score, lengths, topk, row_starts=None):
    cpu, lens = score.detach().cpu(), lengths.cpu().tolist()
    starts = [0] * len(lens) if row_starts is None else row_starts.cpu().tolist()
    out = torch.full((score.shape[0], topk), -1, dtype=torch.int32, device="cpu")
    for row, (start, length) in enumerate(zip(starts, lens)):
        count = min(length, topk)
        if length <= topk:
            out[row, :count] = torch.arange(count, dtype=torch.int32, device="cpu")
        else:
            out[row] = (
                cpu[row, start : start + length].topk(topk).indices.to(torch.int32)
            )
    return out.to(score.device)
