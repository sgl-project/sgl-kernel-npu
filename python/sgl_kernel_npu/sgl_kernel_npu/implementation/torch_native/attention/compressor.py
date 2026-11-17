# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0.
"""Independent PyTorch reference for the stateful KV compressor."""

import torch


@torch.no_grad()
def compressor(
    x,
    wkv,
    wgate,
    state_cache,
    ape,
    norm_weight,
    rope_sin,
    rope_cos,
    state_block_table=None,
    cu_seqlens=None,
    seqused=None,
    start_pos=None,
    rope_head_dim=64,
    cmp_ratio=4,
    coff=1,
    norm_eps=1e-6,
    rotary_mode=2,
    cache_mode=1,
    state_cache_stride_dim0=0,
    *,
    arch35=False,
):
    # Check the semantics used by this reference, not NPU tiling limits.
    if x.ndim not in (2, 3):
        raise ValueError("compressor reference: x must be TH or BSH")
    if cmp_ratio <= 0 or coff not in (1, 2) or cache_mode not in (1, 2):
        raise ValueError("compressor reference: invalid compression or cache mode")
    if rotary_mode != 2 or rope_head_dim != 64:
        raise ValueError(
            "compressor reference: only interleaved RoPE with width 64 is implemented"
        )
    if state_block_table is None or (x.ndim == 2) != (cu_seqlens is not None):
        raise ValueError(
            "compressor reference: cache table is required; cu_seqlens is required only for TH"
        )
    if cache_mode == 2 and state_block_table.ndim != 2:
        raise ValueError("compressor reference: mode 2 requires a 2D state_block_table")
    if cu_seqlens is not None and (cu_seqlens.ndim != 1 or cu_seqlens.numel() < 1):
        raise ValueError("compressor reference: cu_seqlens must be a nonempty vector")

    # CPU FP32 math is deliberately independent of the Ascend C matmul,
    # softmax and normalization. Only metadata/indices cross to Python lists.
    merged = x.ndim == 2
    lengths = cu_seqlens.cpu().tolist() if merged else None
    batch = len(lengths) - 1 if merged else x.shape[0]
    dim = norm_weight.numel()
    ratio = cmp_ratio
    out_rows = (
        min(x.shape[0], x.shape[0] // ratio + batch)
        if merged
        else (x.shape[1] + ratio - 1) // ratio
    )
    out_shape = (out_rows, dim) if merged else (batch, out_rows, dim)
    output = torch.empty(out_shape, dtype=torch.float32, device="cpu")
    if (merged and x.shape[0] == 0) or (not merged and (batch == 0 or x.shape[1] == 0)):
        return output.to(device=x.device, dtype=x.dtype)
    starts = start_pos.cpu().tolist() if start_pos is not None else [0] * batch
    used = (
        seqused.cpu().tolist()
        if seqused is not None
        else [
            lengths[b + 1] - lengths[b] if merged else x.shape[1] for b in range(batch)
        ]
    )
    tokens = x.detach().to(device="cpu", dtype=torch.float32).reshape(-1, x.shape[-1])
    keys = tokens @ wkv.detach().to(device="cpu", dtype=torch.float32).T
    scores = tokens @ wgate.detach().to(device="cpu", dtype=torch.float32).T
    position_weight = ape.detach().cpu()
    norm = norm_weight.detach().cpu()
    sine, cosine = rope_sin.detach().cpu(), rope_cos.detach().cpu()
    table = state_block_table.detach().cpu()
    state = state_cache.detach().cpu().clone()
    initial = state.clone()
    width = coff * dim
    block_size = state.shape[1]
    flat_state = state.reshape(-1, 2 * width)
    flat_initial = initial.reshape(-1, 2 * width)

    def location(b, position):
        if cache_mode == 1:
            block = int(table[b, position // block_size])
            return block * block_size + position % block_size, block != 0
        return int(table[b, coff * ratio + position - starts[b]]), True

    full_load = arch35 and not merged and x.shape[1] <= 4 and tokens.shape[0] <= 256
    preserve_page_tails = arch35 and cache_mode == 2 and coff == 2 and not full_load
    packed_row = 0
    for b in range(batch):
        start, count = starts[b], used[b]
        base = lengths[b] if merged else b * x.shape[1]
        projected_kv = keys[base : base + count]
        projected_score = scores[base : base + count].clone()
        for t in range(count):
            projected_score[t] += position_weight[(start + t) % ratio]
        boundary = (start + count) // ratio * ratio
        row = 0
        for end in range((start // ratio + 1) * ratio, boundary + 1, ratio):
            group_keys = torch.zeros(
                (coff * ratio, dim), dtype=torch.float32, device="cpu"
            )
            group_scores = torch.full_like(group_keys, -float("inf"))
            for c in range(coff):
                feature = slice(c * dim, (c + 1) * dim)
                for offset in range(ratio):
                    pos = end - (coff - c) * ratio + offset
                    target = c * ratio + offset
                    if pos < 0:
                        continue
                    if pos >= start:
                        group_keys[target] = projected_kv[pos - start, feature]
                        group_scores[target] = projected_score[pos - start, feature]
                    else:
                        loc, _ = location(b, pos)
                        group_keys[target] = flat_initial[loc, :width][feature]
                        group_scores[target] = flat_initial[loc, width:][feature]
            compressed = (torch.softmax(group_scores, dim=0) * group_keys).sum(dim=0)
            compressed = (
                compressed * torch.rsqrt(compressed.square().mean() + norm_eps) * norm
            )
            sin = sine[packed_row] if merged else sine[b, row]
            cos = cosine[packed_row] if merged else cosine[b, row]
            rotary = compressed[-rope_head_dim:].clone()
            # The migrated kernel compiles interleaved RoPE only (mode 2).
            compressed[-rope_head_dim::2] = (
                rotary[0::2] * cos[0::2] - rotary[1::2] * sin[0::2]
            )
            compressed[-rope_head_dim + 1 :: 2] = (
                rotary[0::2] * sin[1::2] + rotary[1::2] * cos[1::2]
            )
            if merged:
                output[packed_row] = compressed
            else:
                output[b, row] = compressed
            row += 1
            packed_row += 1

        # Match the copied native SaveState implementations, which clip by
        # absolute token position, including a partial compression chunk.
        keep = max(boundary - (coff - 1) * ratio, 0)
        # arch35 FULL_LOAD preserves the original no-pad policy. The arch22
        # PERF and arch35 NORMAL implementations also retain MTP tail rows.
        if not full_load and block_size > coff * ratio:
            pad = block_size - coff * ratio + 2
            keep = min(keep, max(start + count - pad, 0))
        first = 0 if cache_mode == 1 else max(keep - start, 0)
        for t in range(count):
            pos = start + t
            loc, writable = location(b, pos)
            # Different rows a ring apart indicate a page tail worth retaining.
            if t < first and (
                not preserve_page_tails
                or t + block_size >= count
                or loc == location(b, pos + block_size)[0]
            ):
                continue
            if writable:
                flat_state[loc, :width] = projected_kv[t]
                flat_state[loc, width:] = projected_score[t]
    state_cache.copy_(state.to(state_cache.device))
    return output.to(device=x.device, dtype=x.dtype)
