# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE.cannbot-dsl in the root of the software repository for the full text of the License.

"""Shared independent CPU references and inputs for QLI and QSLI tests."""

from collections import Counter, defaultdict
from itertools import accumulate

import torch

HEADS = 32
BLOCK_SIZE = 8
CANDIDATE_CAPACITY = 2048
TOPK = 512
SCORE_RTOL = 1 / 128
SCORE_ATOL = 2.5e-5


def _prefix(lengths):
    return [0, *accumulate(lengths)]


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


def _check_topk(indices, values, scores, allowed, offset=0):
    """Check values, completeness and the TopK boundary without ordering ties."""
    valid = indices != -1
    if values is not None:
        assert torch.isneginf(values[~valid]).all(), "Invalid TopK values must be -inf"
    selected = indices[valid].long() - offset
    assert selected.numel() == min(indices.numel(), int(allowed.sum()))
    assert selected.unique().numel() == selected.numel()
    assert ((selected >= 0) & (selected < scores.numel())).all()
    assert allowed[selected].all()
    if selected.numel() == 0:
        return
    selected_scores = scores[selected]
    if values is not None:
        torch.testing.assert_close(
            values[valid].float(), selected_scores, rtol=SCORE_RTOL, atol=SCORE_ATOL
        )
    boundary = scores[allowed].topk(selected.numel()).values[-1]
    if torch.isposinf(boundary):
        assert torch.isposinf(selected_scores).all()
    else:
        assert (
            selected_scores.min() >= boundary - SCORE_ATOL - SCORE_RTOL * boundary.abs()
        )


def _make_inputs(
    layout,
    page_size,
    mask,
    *,
    sparse,
    candidates=False,
    candidate_capacity=CANDIDATE_CAPACITY,
    key_lengths=(513, 777),
    topk=TOPK,
    query_lengths=None,
    signed_weights=False,
    uniform_scale=False,
):
    generator = torch.Generator().manual_seed(20260929)
    typical = query_lengths is not None
    query_lengths = list(query_lengths) if typical else [3, 4]
    used_queries = query_lengths if typical else [3, 3]
    assert len(query_lengths) == len(key_lengths)
    total_queries = sum(query_lengths)
    ratio = 2 if mask == 3 else 1
    residual = [0] * len(query_lengths) if typical else [1, 0]

    def random_bytes(shape, low=0, high=256):
        return torch.randint(low, high, shape, generator=generator, dtype=torch.uint8)

    q = random_bytes((total_queries, HEADS, 64))
    qs = (
        torch.full((total_queries, HEADS, 2, 2), 127, dtype=torch.uint8)
        if uniform_scale
        else random_bytes((total_queries, HEADS, 2, 2), 124, 128)
    )
    weights = torch.rand((total_queries, HEADS), generator=generator)
    weights = weights * 2 - 1 if signed_weights else weights + 0.1
    logical_keys = [random_bytes((length, 1, 64)) for length in key_lengths]
    logical_scales = [
        (
            torch.full((length, 1, 2, 2), 127, dtype=torch.uint8)
            if uniform_scale
            else random_bytes((length, 1, 2, 2), 124, 128)
        )
        for length in key_lengths
    ]
    table = None
    if layout == "TND":
        k, ks = torch.cat(logical_keys), torch.cat(logical_scales)
    else:
        max_pages = (max(key_lengths) + page_size - 1) // page_size
        pages = len(key_lengths) * max_pages
        table = (
            torch.randperm(pages, generator=generator)
            .int()
            .reshape(len(key_lengths), max_pages)
        )
        k = torch.zeros((pages, page_size, 1, 64), dtype=torch.uint8)
        ks = torch.full((pages, page_size, 1, 2, 2), 127, dtype=torch.uint8)
        for batch, length in enumerate(key_lengths):
            tokens = torch.arange(length)
            physical_rows = (
                table[batch, tokens // page_size].long() * page_size
                + tokens % page_size
            )
            k.view(-1, 1, 64).index_copy_(0, physical_rows, logical_keys[batch])
            ks.view(-1, 1, 2, 2).index_copy_(0, physical_rows, logical_scales[batch])
        if sparse:
            k = torch.cat(
                (
                    k.reshape(pages, page_size // 8, 512),
                    ks.reshape(pages, page_size // 8, 32),
                ),
                -1,
            )
            ks = None
    candidate_ids = torch.full(
        (total_queries, 1, CANDIDATE_CAPACITY), -1, dtype=torch.int32
    )
    candidate_lengths = torch.zeros((total_queries, 1), dtype=torch.int32)
    references, permitted = [], []
    for batch, (start, length) in enumerate(zip(_prefix(query_lengths), query_lengths)):
        reference = _scores_reference(
            q[start : start + length],
            logical_keys[batch],
            weights[start : start + length],
            qs[start : start + length],
            logical_scales[batch],
        )
        for row in range(length):
            visible = _visible_tokens(
                key_lengths[batch],
                used_queries[batch],
                row,
                mask,
                ratio,
                residual[batch],
            )
            allowed = torch.arange(key_lengths[batch]) < visible
            count = (key_lengths[batch] + 7) // 8
            chosen = torch.randperm(count, generator=generator)[
                : min(CANDIDATE_CAPACITY, max(1, count - 5))
            ].int()
            if sparse and not typical and start + row == 1:
                chosen = chosen[:0]
            candidate_ids[start + row, 0, : chosen.numel()] = chosen
            candidate_lengths[start + row] = chosen.numel()
            if sparse:
                allowed &= torch.isin(torch.arange(key_lengths[batch]) // 8, chosen)
            references.append(reference[row])
            permitted.append(allowed)
    device = torch.device("npu", torch.npu.current_device())

    def to_device(tensor):
        return tensor.to(device) if tensor is not None else None

    sequence = dict(
        cu_seqlens_q=torch.tensor(
            _prefix(query_lengths), dtype=torch.int32, device=device
        ),
        seqused_q=torch.tensor(used_queries, dtype=torch.int32, device=device),
        seqused_k=torch.tensor(key_lengths, dtype=torch.int32, device=device),
    )
    if layout == "TND":
        sequence["cu_seqlens_k"] = torch.tensor(
            _prefix(key_lengths), dtype=torch.int32, device=device
        )
    if mask == 3:
        sequence["cmp_residual_k"] = torch.tensor(
            residual, dtype=torch.int32, device=device
        )
    attributes = dict(
        topk=topk,
        max_seqlen_q=max(query_lengths),
        layout_q="TND",
        layout_k=layout,
        mask_mode=mask,
        cmp_ratio=ratio,
    )
    metadata_attributes = dict(
        attributes,
        max_seqlen_k=max(key_lengths),
        num_heads_q=HEADS,
        num_heads_k=1,
        head_dim=128,
    )
    if not sparse:
        attributes.update(
            candidate_topk_blocks=candidate_capacity if candidates else -1,
            candidate_block_size=8 if candidates else -1,
        )
        metadata_attributes.update(
            candidate_topk_blocks=attributes["candidate_topk_blocks"],
            candidate_block_size=attributes["candidate_block_size"],
        )
    else:
        attributes["candidate_block_size"] = 8
        metadata_attributes.update(candidate_block_size=8, quant_mode=1)
    return dict(
        q=to_device(q),
        k=to_device(k),
        w=to_device(weights),
        descale_q=to_device(qs),
        descale_k=to_device(ks),
        block_table=to_device(table),
        sequence=sequence,
        attributes=attributes,
        metadata_attributes=metadata_attributes,
        candidate_ids=to_device(candidate_ids),
        candidate_lengths=to_device(candidate_lengths),
        reference=references,
        allowed=permitted,
        offsets=to_device(
            torch.arange(total_queries, dtype=torch.int32).reshape(-1, 1) * 10000
        ),
    )


def _assert_schedule(metadata, tasks, group_rows, workers=32):
    """Check coverage and LD fan-in independently of the load-balancing policy."""
    assert metadata.dtype == torch.int32 and metadata.shape == (1024,)
    records = metadata.reshape(128, 8).tolist()
    assert not metadata.reshape(128, 8)[workers:36].any()
    assert not metadata.reshape(128, 8)[36 + 2 * workers : 108].any()
    assert not metadata.reshape(128, 8)[108:].any()
    coverage = Counter()
    owners = defaultdict(set)
    for core, record in enumerate(records[:workers]):
        enabled, *fields = record
        assert enabled in (0, 1)
        if not enabled:
            assert not any(fields)
            continue
        task_begin, task_end = tuple(record[1:4]), tuple(record[4:7])
        assert task_begin <= task_end
        for task in tasks:
            if task_begin <= task < task_end:
                coverage[task] += 1
                owners[task[:2]].add(core)
    assert coverage == Counter({task: 1 for task in tasks})
    merged_rows = defaultdict(list)
    slots = {}
    for enabled, batch, group, base, parts, row_start, row_count, reserved in records[
        36:108
    ]:
        assert enabled in (0, 1)
        if not enabled:
            assert (batch, group, base, parts, row_start, row_count, reserved) == (
                0,
            ) * 7
            continue
        key = (batch, group)
        assert parts == len(owners[key]) and parts > 1
        assert reserved == 0 and base >= 0 and row_count > 0
        assert slots.setdefault(key, (base, parts)) == (base, parts)
        merged_rows[key].extend(range(row_start, row_start + row_count))
    assert set(merged_rows) == {key for key, cores in owners.items() if len(cores) > 1}
    for key, rows in merged_rows.items():
        assert sorted(rows) == list(range(group_rows[key]))
    allocated = [
        slot for base, parts in slots.values() for slot in range(base, base + parts)
    ]
    assert len(allocated) == len(set(allocated))
