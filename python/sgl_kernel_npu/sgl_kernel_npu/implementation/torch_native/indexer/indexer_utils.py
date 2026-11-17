# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE.cannbot-dsl under implementation/cannbot-dsl for the full text of the License.

"""CPU Torch arithmetic shared by the two quantized indexer references."""

import torch

HEADS = 32


def _decode_mxfp4(packed, scales):
    """Low nibble precedes high nibble; one E8M0 scale covers 32 values."""
    magnitudes = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float32)
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2).long()
    values = magnitudes[codes & 7] * torch.where(codes < 8, 1.0, -1.0)
    exponents = scales.flatten(-2).int() - 127
    return values * torch.exp2(exponents.float()).repeat_interleave(32, dim=-1)


def _bf16_fma(lhs, rhs, acc):
    """Round the exact BF16 product-plus-addend directly to BF16, ties to even."""
    product, addend = lhs.double() * rhs.double(), acc.double()
    summed = product + addend
    recovered = summed - product
    # TwoSum retains the direction of bits lost even by the FP64 addition.
    residual = (product - (summed - recovered)) + (addend - recovered)
    _, exponent = torch.frexp(summed.abs())
    step = torch.ldexp(torch.ones_like(summed), (exponent - 8).clamp_min(-133))
    scaled = summed.abs() / step
    lower = torch.floor(scaled)
    rounded = torch.round(scaled)
    direction = torch.sign(summed) * residual
    rounded = torch.where(
        (scaled - lower == 0.5) & (direction != 0),
        lower + (direction > 0),
        rounded,
    )
    return (torch.sign(summed) * rounded * step).float()


def _scores_reference(query, keys, weights, query_scale, key_scale):
    query_float = _decode_mxfp4(query, query_scale)
    key_float = _decode_mxfp4(keys, key_scale).squeeze(1)
    # Cube QK, Vector1 weights and each fused multiply-add result are BF16.
    qk = (query_float @ key_float.T).to(torch.bfloat16).float()
    bf16_weights = weights.to(torch.bfloat16).float()
    scores = torch.zeros((query.shape[0], keys.shape[0]), dtype=torch.float32)
    for head in range(HEADS):
        scores = _bf16_fma(
            qk[:, head].clamp_min(0), bf16_weights[:, head, None], scores
        )
    return scores


def _visible_tokens(key_length, used_query, row, mask, ratio, residual):
    if row >= used_query:
        return 0
    if mask == 0:
        return key_length
    return min(
        key_length,
        max(0, (key_length * ratio + residual - used_query + row + 1) // ratio),
    )


def _indexer_reference(
    q,
    k,
    w,
    descale_q,
    descale_k=None,
    *,
    topk,
    quant_mode,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_q=None,
    seqused_k=None,
    cmp_residual_k=None,
    block_table=None,
    output_idx_offset=None,
    metadata=None,
    max_seqlen_q=-1,
    mask_mode=0,
    cmp_ratio=1,
    layout_q="TND",
    layout_k="TND",
    return_value=False,
    candidate_topk_blocks=-1,
    candidate_block_size=-1,
    candidate_block_indices=None,
    candidate_block_length=None
):
    """Score on CPU; metadata is a device schedule, not part of the mathematics.

    Equal scores may produce different valid TopK permutations on each backend.
    CPU transfers intentionally keep FP64 FMA emulation off the NPU.
    """
    if quant_mode != 1 or layout_q != "TND" or layout_k not in ("TND", "PA_BBND"):
        raise NotImplementedError(
            "Reference supports MXFP4 TND queries and TND/PA_BBND keys"
        )
    if mask_mode not in (0, 3) or cmp_ratio < 1 or topk < 1:
        raise ValueError("Invalid topk, mask_mode or cmp_ratio")
    device = q.device

    def cpu(x):
        return None if x is None else x.detach().cpu()

    q, k, w, qs, ks = map(cpu, (q, k, w, descale_q, descale_k))
    sparse = candidate_block_indices is not None
    candidates, candidate_lengths = map(
        cpu, (candidate_block_indices, candidate_block_length)
    )
    if (sparse or candidate_topk_blocks > 0) and candidate_block_size != 8:
        raise ValueError("Candidate blocks contain eight tokens")
    if sparse and layout_k == "PA_BBND":
        if ks is not None or k.ndim != 3 or k.shape[-1] != 544:
            raise ValueError("QSLI PA keys require combined [pages,groups,544] bytes")
        pages, groups, _ = k.shape
        ks = k[..., 512:].reshape(pages, groups * 8, 1, 2, 2)
        k = k[..., :512].reshape(pages, groups * 8, 1, 64)
    if ks is None:
        raise ValueError("Separate keys require E8M0 descale_k")
    bounds = [0, q.shape[0]] if cu_seqlens_q is None else cpu(cu_seqlens_q).tolist()
    key_bounds = None if cu_seqlens_k is None else cpu(cu_seqlens_k).tolist()
    if layout_k == "TND" and len(bounds) > 2 and key_bounds is None:
        raise ValueError("Batched TND keys require cu_seqlens_k")
    table, offsets = cpu(block_table), cpu(output_idx_offset)
    used_q, used_k, residuals = map(cpu, (seqused_q, seqused_k, cmp_residual_k))
    ids = torch.full((q.shape[0], 1, topk), -1, dtype=torch.int32, device="cpu")
    values = torch.full(ids.shape, -float("inf"), dtype=torch.bfloat16, device="cpu")
    count = max(0, candidate_topk_blocks)
    candidate_ids = torch.full(
        (q.shape[0], 1, count), -1, dtype=torch.int32, device="cpu"
    )
    lengths = torch.zeros((q.shape[0], 1), dtype=torch.int32, device="cpu")
    for batch, (start, end) in enumerate(zip(bounds, bounds[1:])):
        if layout_k == "TND":
            lo, hi = (
                (0, k.shape[0]) if key_bounds is None else key_bounds[batch : batch + 2]
            )
            keys, scales = k[lo:hi], ks[lo:hi]
        else:
            pages = torch.arange(k.shape[0]) if table is None else table[batch].long()
            # Never gather unused padding entries from the block table.
            capacity = pages.numel() * k.shape[1]
            length = capacity if used_k is None else int(used_k[batch])
            pages = pages[: (length + k.shape[1] - 1) // k.shape[1]]
            keys = k[pages].reshape(-1, 1, 64)
            scales = ks[pages].reshape(-1, 1, 2, 2)
        length = keys.shape[0] if used_k is None else int(used_k[batch])
        keys, scales = keys[:length], scales[:length]
        scores = _scores_reference(
            q[start:end], keys, w[start:end], qs[start:end], scales
        )
        used = end - start if used_q is None else int(used_q[batch])
        residual = 0 if residuals is None else int(residuals[batch])
        for local, row in enumerate(range(start, end)):
            visible = _visible_tokens(
                length, used, local, mask_mode, cmp_ratio, residual
            )
            allowed = torch.arange(length, device="cpu") < visible
            if sparse:
                n = int(candidate_lengths.reshape(-1)[row])
                blocks = candidates[row].reshape(-1)[:n].long()
                allowed &= torch.isin(torch.arange(length, device="cpu") // 8, blocks)
            positions = allowed.nonzero().flatten()
            if positions.numel():
                n = min(topk, positions.numel())
                selected = positions[scores[local, positions].topk(n).indices]
                offset = 0 if offsets is None else int(offsets.reshape(-1)[row])
                ids[row, 0, :n] = (selected + offset).int()
                values[row, 0, :n] = scores[local, selected].bfloat16()
            if count and visible:
                padded = torch.full(
                    ((visible + 7) // 8 * 8,), -float("inf"), device="cpu"
                )
                padded[:visible] = scores[local, :visible]
                block_scores = padded.reshape(-1, 8).amax(-1)
                if visible % 8:
                    block_scores[-1] = float(
                        "inf"
                    )  # Always retain the partial causal block.
                n = min(count, block_scores.numel())
                candidate_ids[row, 0, :n] = block_scores.topk(n).indices.int()
                lengths[row, 0] = n
    result = (
        ids.to(device),
        (values if return_value else values.flatten()[:0]).to(device),
    )
    if sparse:
        return result
    return (
        *result,
        (candidate_ids if count else candidate_ids.flatten()).to(device),
        (lengths if count else lengths.flatten()[:0]).to(device),
    )


def indexer_reference(*args, **kwargs):
    with torch.device("cpu"):
        return _indexer_reference(*args, **kwargs)
