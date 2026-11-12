"""Independent CPU FP64 per-row/head MQA reference; never a graph fallback."""

import math

import torch


def _ref_mqa(kind, args):
    q, keys, first, second, *extra = [
        x.detach().cpu() if isinstance(x, torch.Tensor) else x for x in args
    ]
    width = keys.shape[0] if kind == "packed" else extra[0]
    out = torch.full(
        (q.shape[0], width), -float("inf"), dtype=torch.float64, device="cpu"
    )
    for row in range(q.shape[0]):
        if kind == "packed":
            start, end = int(first[row]), int(second[row])
            selected = keys[start:end, 0].double()
        else:
            start, end = 0, int(second[row])
            positions = torch.arange(end, device="cpu")
            pages = first[row, positions // keys.shape[1]].long()
            selected = keys[pages, positions % keys.shape[1], 0].double()
        values = torch.zeros(end - start, dtype=torch.float64, device="cpu")
        for head in range(q.shape[1]):
            values += (selected @ q[row, head].double()).clamp_min(0)
        out[row, start:end] = values / math.sqrt(q.shape[-1])
    return out.float()


def packed(q, k, starts, ends):
    return _ref_mqa("packed", (q, k, starts, ends)).to(q.device)


def paged(q, cache, table, lengths, width):
    return _ref_mqa("paged", (q, cache, table, lengths, width)).to(q.device)
