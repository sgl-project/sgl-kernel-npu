"""Independent PyTorch reference for scalar chunk-local cumulative sums."""

import torch


def chunk_local_cumsum(
    g,
    chunk_size,
    reverse=False,
    scale=None,
    cu_seqlens=None,
    head_first=False,
    output_dtype=torch.float32,
):
    if g.ndim != 3:
        raise ValueError("scalar cumulative sum requires a 3-D input")
    if chunk_size <= 0 or chunk_size & (chunk_size - 1):
        raise ValueError("chunk_size must be a positive power of two")
    if cu_seqlens is not None and g.shape[0] != 1:
        raise ValueError("variable-length input requires batch size 1")
    x = g.transpose(1, 2) if head_first else g
    result = torch.empty_like(x, dtype=output_dtype or g.dtype)
    seqs = [(b, 0, x.shape[1]) for b in range(x.shape[0])]
    if cu_seqlens is not None:
        offsets = cu_seqlens.tolist()
        seqs = [(0, start, end) for start, end in zip(offsets, offsets[1:])]
    for batch, start, end in seqs:
        for pos in range(start, end, chunk_size):
            stop = min(pos + chunk_size, end)
            block = x[batch, pos:stop].float()
            if reverse:
                block = block.flip(0)
            block = block.cumsum(0)
            if reverse:
                block = block.flip(0)
            if scale is not None:
                block = block * scale
            # The existing Triton kernel casts to the input type before store.
            result[batch, pos:stop] = block.to(g.dtype)
    return result.transpose(1, 2).contiguous() if head_first else result
