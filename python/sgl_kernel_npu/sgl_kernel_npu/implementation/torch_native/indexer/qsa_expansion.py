"""Independent CPU block expansion, including compact tail and padding."""

import torch


def expand_blocks(blocks, positions, lengths, ratio, topk):
    device = blocks.device
    blocks, positions, lengths = blocks.cpu(), positions.cpu(), lengths.cpu()
    result = []
    for row, position, length in zip(
        blocks.tolist(), positions.tolist(), lengths.tolist()
    ):
        raw = []
        for block in row:
            raw.extend(
                [
                    block * ratio + offset if block >= 0 else -1
                    for offset in range(ratio)
                ]
            )
        raw = [v if 0 <= v < length else -1 for v in raw[:topk]]
        start = ((position + 1) // ratio) * ratio
        count = position + 1 - start
        raw += [
            start + j if j < count and start + j < length else -1
            for j in range(ratio - 1)
        ]
        raw = [v for v in raw if v >= 0] + [v for v in raw if v < 0]
        result.append(raw)
    return (
        torch.tensor(result, dtype=torch.int64, device="cpu")
        .to(torch.int32)
        .reshape(blocks.shape[0], topk + ratio - 1)
        .to(device)
    )
