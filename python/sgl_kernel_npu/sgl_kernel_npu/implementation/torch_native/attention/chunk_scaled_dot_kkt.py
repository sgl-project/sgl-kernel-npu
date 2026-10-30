"""Independent reference for the chunked strictly lower-triangular KKT."""

import torch


def chunk_scaled_dot_kkt(
    k, beta, g_cumsum=None, cu_seqlens=None, chunk_size=64, output_dtype=torch.float32
):
    batch, length, groups, _ = k.shape
    heads = beta.shape[-1]
    if g_cumsum is None:
        raise ValueError("g_cumsum is required by the current Triton wrapper")
    if heads % groups:
        raise ValueError("the number of heads must be a multiple of key heads")
    if cu_seqlens is not None and batch != 1:
        raise ValueError("variable-length input requires batch size 1")
    out = torch.zeros(
        (batch, length, heads, chunk_size), device=k.device, dtype=output_dtype
    )
    sequences = [(b, 0, length) for b in range(batch)]
    if cu_seqlens is not None:
        offsets = cu_seqlens.tolist()
        sequences = [(0, start, end) for start, end in zip(offsets, offsets[1:])]
    for b, start, end in sequences:
        for pos in range(start, end, chunk_size):
            stop = min(pos + chunk_size, end)
            for h in range(heads):
                keys = k[b, pos:stop, h // (heads // groups)].float()
                gate = g_cumsum[b, pos:stop, h].float()
                diff = gate[:, None] - gate[None, :]
                decay = torch.where(diff <= 0, diff.exp(), 0.0)
                matrix = (keys @ keys.transpose(0, 1)) * decay
                matrix = matrix * beta[b, pos:stop, h].float()[:, None]
                out[b, pos:stop, h, : stop - pos] = matrix.tril(-1).to(output_dtype)
    return out
