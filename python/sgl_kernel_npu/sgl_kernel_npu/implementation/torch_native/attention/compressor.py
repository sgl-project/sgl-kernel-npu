"""Independent eager reference for compressor."""

import torch
import torch.nn.functional as F


def compressor(
    x,
    wkv,
    wgate,
    state_cache,
    ape,
    norm_weight,
    *,
    rope_sin,
    rope_cos,
    rope_head_dim,
    cmp_ratio,
    state_block_table,
    cu_seqlens=None,
    seqused=None,
    start_pos=None,
    coff=1,
    norm_eps=1e-6,
    rotary_mode=2,
    cache_mode=2,
):
    if coff not in (1, 2) or cache_mode not in (1, 2) or rotary_mode not in (1, 2):
        raise NotImplementedError(
            "reference supports coff=1/2, paged/cycle cache and half/interleaved RoPE"
        )
    if state_cache.ndim != 3 or state_cache.dtype != torch.float32:
        raise ValueError("state_cache must be FP32 [pages, slots, 2*coff*head_dim]")
    d = wkv.shape[0] // coff
    batch = len(start_pos)
    tnd = cu_seqlens is not None
    flat = x.reshape(-1, x.shape[-1])
    kv = F.linear(flat.float(), wkv.float())
    score = F.linear(flat.float(), wgate.float())
    out_shape = (
        (min(flat.shape[0], flat.shape[0] // cmp_ratio + batch), d)
        if tnd
        else (batch, (x.shape[1] + cmp_ratio - 1) // cmp_ratio, d)
    )
    out = x.new_zeros(out_shape)
    rows = state_cache.reshape(-1, state_cache.shape[-1])
    ring = state_cache.shape[1]
    history = coff * cmp_ratio

    def slot(b, p, start):
        if cache_mode == 1:
            return int(state_block_table[b, p // ring]) * ring + p % ring
        if state_block_table.ndim == 1:
            return int(state_block_table[b]) * ring + p % ring
        return int(state_block_table[b, history + p - start])

    offset = 0
    for b in range(batch):
        start = int(start_pos[b])
        qstart = int(cu_seqlens[b]) if tnd else b * x.shape[1]
        count = (
            int(seqused[b])
            if seqused is not None
            else int(cu_seqlens[b + 1]) - qstart
            if tnd
            else x.shape[1]
        )
        end = start + count
        boundary = end // cmp_ratio * cmp_ratio
        # Snapshot before updates so long prefills cannot overwrite ring history.
        previous = {}
        for p in range(max(0, start - history), start):
            previous[p] = rows[slot(b, p, start)].clone()
        for i in range(count):
            p = start + i
            previous[p] = torch.cat(
                (kv[qstart + i], score[qstart + i] + ape[p % cmp_ratio].float())
            )
        batch_out = 0
        for stop in range((start // cmp_ratio + 1) * cmp_ratio, end + 1, cmp_ratio):
            keys, logits = [], []
            for j in range(coff):
                lo = stop - (coff - j) * cmp_ratio
                for p in range(lo, lo + cmp_ratio):
                    if p < 0:
                        keys.append(torch.zeros(d, device=x.device))
                        logits.append(torch.full((d,), -torch.inf, device=x.device))
                    else:
                        value = previous[p]
                        keys.append(value[j * d : (j + 1) * d])
                        logits.append(value[coff * d + j * d : coff * d + (j + 1) * d])
            compressed = (torch.stack(logits).softmax(0) * torch.stack(keys)).sum(0)
            compressed = (
                compressed
                * torch.rsqrt(compressed.square().mean() + norm_eps)
                * norm_weight.float()
            )
            index = offset if tnd else b * out.shape[1] + batch_out
            if rope_head_dim:
                part = compressed[-rope_head_dim:]
                if rotary_mode == 2:
                    rotated = torch.stack((-part[1::2], part[::2]), -1).flatten()
                else:
                    a, c = part.chunk(2)
                    rotated = torch.cat((-c, a))
                cos = rope_cos.reshape(-1, rope_head_dim)[index].float()
                sin = rope_sin.reshape(-1, rope_head_dim)[index].float()
                compressed[-rope_head_dim:] = part * cos + rotated * sin
            out.reshape(-1, d)[index] = compressed.to(x.dtype)
            offset += 1
            batch_out += 1
        # Match the native retention rule, including extra verification tail rows.
        window = coff * cmp_ratio
        pad = ring - window + 2 if ring > window else 0
        keep = boundary - (coff - 1) * cmp_ratio
        if pad:
            keep = min(keep, end - pad if end > pad else 0)
        for p in range(start, end):
            chunk_start = max(start, p // cmp_ratio * cmp_ratio)
            if cache_mode == 1 or chunk_start >= keep:
                destination = slot(b, p, start)
                if cache_mode != 1 or destination // ring != 0:
                    rows[destination].copy_(previous[p])
    return out
