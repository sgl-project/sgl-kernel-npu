"""Independent CPU FP32 reference for circular-cache CompressorV2.

    kv = X @ Wkv.T; gate = X @ Wgate.T
    compressed[d] = sum_t softmax(gate[:,d])[t] * kv[t,d]

The reference supports BSH/TH and requires cache capacity >= cmp_ratio.
It copies data to CPU and reads metadata there; it is not a performance backend.
Output padding is initialized to zero for convenience, but is not part of the
upstream numerical contract. The caller's cache is updated in place.
"""

import torch


def compressor_v2(
    x,
    wkv,
    wgate,
    state_cache,
    cmp_ratio=4,
    *,
    state_block_table=None,
    cu_seqlens=None,
    seqused=None,
    start_pos=None,
):
    capacity, dim = state_cache.shape[1], wkv.shape[0]
    if capacity < cmp_ratio:
        raise NotImplementedError("torch_native requires cache capacity >= cmp_ratio")
    packed = x.ndim == 2
    boundaries = cu_seqlens.detach().cpu().tolist() if packed else None
    batch = len(boundaries) - 1 if packed else x.shape[0]
    lengths = (
        [boundaries[b + 1] - boundaries[b] for b in range(batch)]
        if packed
        else [x.shape[1]] * batch
    )
    used = seqused.detach().cpu().tolist() if seqused is not None else lengths
    starts = start_pos.detach().cpu().tolist() if start_pos is not None else [0] * batch
    blocks = state_block_table.detach().cpu().tolist()
    if len(set(blocks)) != batch or any(
        b < 0 or b >= state_cache.shape[0] for b in blocks
    ):
        raise ValueError("state_block_table IDs must be unique and in range")
    if packed and (boundaries[0] != 0 or boundaries[-1] != x.shape[0]):
        raise ValueError("cu_seqlens must start at zero and end at T")
    if any(n < 0 or n > length for n, length in zip(used, lengths)) or any(
        s < 0 for s in starts
    ):
        raise ValueError("Invalid sequence length or start position")
    state = state_cache.detach().cpu().clone()
    flat = x.detach().cpu().float().reshape(-1, x.shape[-1])
    kv = flat @ wkv.detach().cpu().float().T
    gate = flat @ wgate.detach().cpu().float().T
    if packed:
        shape = (min(x.shape[0], x.shape[0] // cmp_ratio + batch), dim)
    else:
        shape = (batch, (x.shape[1] + cmp_ratio - 1) // cmp_ratio, dim)
    result = torch.zeros(shape, dtype=x.dtype, device="cpu")
    output_offset = 0
    for b, (start, count, physical) in enumerate(zip(starts, used, blocks)):
        offset = boundaries[b] if packed else b * x.shape[1]
        complete = (start % cmp_ratio + count) // cmp_ratio
        history = start % cmp_ratio
        prior = torch.arange(start - history, start, device="cpu") % capacity
        group_kv = torch.cat(
            (state[physical, prior, :dim], kv[offset : offset + count]), dim=0
        )
        group_gate = torch.cat(
            (state[physical, prior, dim:], gate[offset : offset + count]), dim=0
        )
        if complete:
            values = group_kv[: complete * cmp_ratio].reshape(complete, cmp_ratio, dim)
            scores = group_gate[: complete * cmp_ratio].reshape(
                complete, cmp_ratio, dim
            )
            compressed = (scores.softmax(dim=1) * values).sum(dim=1).to(x.dtype)
            if packed:
                result[output_offset : output_offset + complete] = compressed
            else:
                result[b, :complete] = compressed
            output_offset += complete
        # Write just the final capacity tokens; untouched cache positions persist.
        begin = max(0, count - capacity)
        positions = torch.arange(start + begin, start + count, device="cpu") % capacity
        state[physical, positions, :dim] = kv[offset + begin : offset + count]
        state[physical, positions, dim:] = gate[offset + begin : offset + count]
    state_cache.copy_(state.to(state_cache.device))
    return result.to(x.device)
