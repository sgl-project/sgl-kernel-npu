# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE.cannbot-dsl in the root of the software repository for the full text of the License.

"""Independent reference and precision tests for quant_lightning_indexer."""

import os
from itertools import accumulate

import pytest

pytest.importorskip("torch_npu")
pytest.importorskip("cannbotdsl")
import torch
from sgl_kernel_npu.operators.indexer import (
    quant_lightning_indexer,
    quant_lightning_indexer_metadata,
)

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


def _npu_device():
    pytest.importorskip("torch_npu")
    device = torch.device("npu", int(os.environ.get("NPU_DEVICE_ID", "0")))
    torch.npu.set_device(device)
    return device


def _load_operators():
    return quant_lightning_indexer, quant_lightning_indexer_metadata


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
    device = _npu_device()

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


def _check_kernel(
    layout,
    page_size,
    mask,
    *,
    sparse,
    candidates=False,
    candidate_capacity=CANDIDATE_CAPACITY,
    key_lengths=(513, 777),
    topk=TOPK,
    empty=None,
    query_lengths=None,
    signed_weights=False,
    uniform_scale=False,
):
    data = _make_inputs(
        layout,
        page_size,
        mask,
        sparse=sparse,
        candidates=candidates,
        candidate_capacity=candidate_capacity,
        key_lengths=key_lengths,
        topk=topk,
        query_lengths=query_lengths,
        signed_weights=signed_weights,
        uniform_scale=uniform_scale,
    )
    if empty is not None:
        if empty == "query":
            data["sequence"]["seqused_q"].zero_()
        elif empty == "key":
            data["sequence"]["seqused_k"].zero_()
        elif empty == "candidate" and sparse:
            data["candidate_lengths"].zero_()
        else:
            raise ValueError(f"Unsupported empty input: {empty}")
        for allowed in data["allowed"]:
            allowed.zero_()
    if not sparse and query_lengths is not None and empty is None:
        # 典型 TND 场景由 cu_seqlens_q 提供每个 batch 的实际 Q 长度。
        data["sequence"]["seqused_q"] = None
    kernel, make_metadata = _load_operators()
    metadata_args = (data["candidate_lengths"],) if sparse else ()
    metadata = make_metadata(
        *metadata_args, **data["sequence"], **data["metadata_attributes"]
    )
    args = [data[key] for key in ("q", "k", "w", "descale_q")]
    if sparse:
        args.extend((data["candidate_ids"], data["candidate_lengths"]))
    else:
        args.append(data["descale_k"])
    kwargs = dict(
        data["sequence"],
        **data["attributes"],
        quant_mode=1,
        block_table=data["block_table"],
        output_idx_offset=data["offsets"],
        metadata=metadata,
    )
    if sparse:
        kwargs["descale_k"] = data["descale_k"]
    for return_value in (True, False):
        outputs = kernel(*args, **kwargs, return_value=return_value)
        torch.npu.synchronize()
        indices, values = (tensor.cpu() for tensor in outputs[:2])
        assert indices.dtype == torch.int32
        assert indices.shape == (len(data["reference"]), 1, topk)
        assert values.dtype == torch.bfloat16
        assert values.shape == indices.shape if return_value else values.numel() == 0
        for row, (scores, allowed) in enumerate(
            zip(data["reference"], data["allowed"])
        ):
            _check_topk(
                indices[row, 0],
                values[row, 0] if return_value else None,
                scores,
                allowed,
                row * 10000,
            )
        if not sparse:
            candidate_ids, candidate_lengths = (tensor.cpu() for tensor in outputs[2:])
            if not candidates:
                assert candidate_ids.numel() == candidate_lengths.numel() == 0
                continue
            assert candidate_ids.shape == (indices.shape[0], 1, candidate_capacity)
            for row, (scores, allowed) in enumerate(
                zip(data["reference"], data["allowed"])
            ):
                padded = torch.full((((scores.numel() + 7) // 8) * 8,), -float("inf"))
                padded[: scores.numel()] = scores.masked_fill(~allowed, -float("inf"))
                block_scores = padded.reshape(-1, 8).amax(-1)
                valid_blocks = torch.isfinite(block_scores)
                visible = int(allowed.sum())
                # The public candidate contract always retains the partial block.
                if visible % BLOCK_SIZE:
                    block_scores[visible // BLOCK_SIZE] = float("inf")
                count = min(candidate_capacity, int(valid_blocks.sum()))
                assert candidate_lengths[row, 0] == count
                _check_topk(
                    candidate_ids[row, 0, :count], None, block_scores, valid_blocks
                )


def test_bf16_fma_avoids_intermediate_rounding():
    lhs = torch.tensor([7971459301376.0, 7971459301376.0, 1.0 + 3 / 128])
    rhs = torch.tensor([-0.6875, -0.6875, 1.5])
    acc = torch.tensor([1120.0, -1120.0, 2.0**-100])
    expected = torch.tensor([-5463198400512.0, -5497558138880.0, 1.5390625])
    torch.testing.assert_close(_bf16_fma(lhs, rhs, acc), expected, rtol=0, atol=0)


def test_mxfp4_decode():
    packed = torch.tensor([[0x21, 0xF8] * 32], dtype=torch.uint8)
    scales = torch.tensor([[[127, 128], [126, 127]]], dtype=torch.uint8)
    expected = torch.tensor([0.5, 1.0, 0.0, -6.0] * 32)
    expected *= torch.tensor([1.0, 2.0, 0.5, 1.0]).repeat_interleave(32)
    torch.testing.assert_close(_decode_mxfp4(packed, scales)[0], expected)


def test_topk_reference_rejects_non_topk():
    with pytest.raises(AssertionError):
        _check_topk(
            torch.tensor([0, 1]),
            None,
            torch.tensor([1.0, 2.0, 9.0]),
            torch.ones(3, dtype=torch.bool),
        )


def test_topk_reference_rejects_zero_padding():
    with pytest.raises(AssertionError, match="Invalid TopK values"):
        _check_topk(
            torch.tensor([-1]),
            torch.tensor([0.0]),
            torch.tensor([1.0]),
            torch.tensor([False]),
        )


@pytest.mark.npu
@pytest.mark.parametrize(
    "layout,page_size,mask,candidates",
    [
        pytest.param("PA_BBND", 128, 0, False, id="pa128_ragged"),
        pytest.param("PA_BBND", 64, 3, True, id="pa64_causal_candidates"),
        pytest.param("TND", 128, 3, False, id="tnd_causal"),
    ],
)
def test_quant_lightning_indexer(layout, page_size, mask, candidates):
    _check_kernel(layout, page_size, mask, sparse=False, candidates=candidates)


@pytest.mark.npu
def test_candidate16_ld_tail():
    _check_kernel(
        "PA_BBND", 64, 3, sparse=False, candidates=True, candidate_capacity=16
    )


@pytest.mark.npu
@pytest.mark.parametrize("layout", ["PA_BBND", "TND"])
def test_invalid_values_are_negative_infinity(layout):
    _check_kernel(layout, 128, 0, sparse=False, key_lengths=(19, 73))


@pytest.mark.npu
@pytest.mark.parametrize("layout", ["PA_BBND", "TND"])
@pytest.mark.parametrize("empty", ["query", "key"])
def test_all_rows_empty(layout, empty):
    _check_kernel(layout, 128, 0, sparse=False, key_lengths=(19, 73), empty=empty)


@pytest.mark.npu
@pytest.mark.slow
@pytest.mark.parametrize(
    "batch_size,query_length,key_length",
    [
        pytest.param(12, 6, 65536, id="b12_s1_6_n32_s2_64k"),
        pytest.param(12, 6, 131072, id="b12_s1_6_n32_s2_128k"),
        pytest.param(32, 6, 65536, id="b32_s1_6_n32_s2_64k"),
        pytest.param(32, 6, 131072, id="b32_s1_6_n32_s2_128k"),
        pytest.param(32, 1, 65536, id="b32_s1_1_n32_s2_64k"),
        pytest.param(32, 1, 131072, id="b32_s1_1_n32_s2_128k"),
    ],
)
@pytest.mark.parametrize(
    "candidate_enabled", [True, False], ids=["candidate_on", "candidate_off"]
)
def test_typical_case(batch_size, query_length, key_length, candidate_enabled):
    _check_kernel(
        "PA_BBND",
        128,
        3,
        sparse=False,
        candidates=candidate_enabled,
        candidate_capacity=CANDIDATE_CAPACITY,
        query_lengths=(query_length,) * batch_size,
        key_lengths=(key_length,) * batch_size,
        signed_weights=True,
        uniform_scale=True,
    )


@pytest.mark.npu
@pytest.mark.parametrize(
    "has_cu_k,has_used_k,batches",
    [
        (False, False, 1),
        (False, True, 1),
        (True, False, 1),
        (True, True, 1),
        (True, False, 2),
        (True, True, 2),
    ],
)
def test_tnd_optional_key_lengths(has_cu_k, has_used_k, batches):
    data = _make_inputs(
        "TND",
        128,
        3,
        sparse=False,
        query_lengths=(3, 7)[:batches],
        key_lengths=(513, 1031)[:batches],
        signed_weights=True,
    )
    sequence = dict(data["sequence"], seqused_q=None)
    if not has_cu_k:
        sequence["cu_seqlens_k"] = None
    if not has_used_k:
        sequence["seqused_k"] = None
    kernel, make_metadata = _load_operators()
    metadata = make_metadata(**sequence, **data["metadata_attributes"])
    args = [data[name] for name in ("q", "k", "w", "descale_q", "descale_k")]
    for return_value in (False, True):
        outputs = kernel(
            *args,
            **sequence,
            **data["attributes"],
            quant_mode=1,
            metadata=metadata,
            return_value=return_value,
        )
        torch.npu.synchronize()
        indices = outputs[0].cpu()
        values = outputs[1].cpu() if return_value else None
        for row, (scores, allowed) in enumerate(
            zip(data["reference"], data["allowed"])
        ):
            _check_topk(
                indices[row, 0],
                values[row, 0] if return_value else None,
                scores,
                allowed,
            )


@pytest.mark.npu
def test_metadata_rejects_noncontiguous_sequence():
    data = _make_inputs("PA_BBND", 128, 3, sparse=False)
    _, make_metadata = _load_operators()
    sequence = dict(data["sequence"])
    lengths = sequence["seqused_k"]
    storage = torch.empty(
        lengths.numel() * 2, dtype=lengths.dtype, device=lengths.device
    )
    storage[::2] = lengths
    sequence["seqused_k"] = storage[::2]
    with pytest.raises(ValueError):
        make_metadata(**sequence, **data["metadata_attributes"])


@pytest.mark.parametrize("entry", ["operator", "metadata"])
def test_public_contract_rejects_64_heads(entry):
    kernel, make_metadata = _load_operators()
    with pytest.raises(ValueError, match="N1=32"):
        if entry == "metadata":
            make_metadata(
                num_heads_q=64,
                num_heads_k=1,
                head_dim=128,
                topk=512,
                layout_k="PA_BBND",
            )
        else:
            kernel(
                torch.empty((1, 64, 64), dtype=torch.uint8, device="npu"),
                None,
                None,
                None,
                None,
                topk=512,
                quant_mode=1,
            )


@pytest.mark.npu
@pytest.mark.parametrize(
    "layout,page_size,mask,query_lengths,key_lengths,topk,candidates,candidate_capacity",
    [
        pytest.param(
            "PA_BBND",
            16,
            3,
            (1, 5, 7),
            (255, 256, 257),
            17,
            True,
            17,
            id="pa16_ragged_topk17_candidate17",
        ),
        pytest.param(
            "PA_BBND",
            1024,
            0,
            (7,),
            (1025,),
            513,
            False,
            2048,
            id="pa1024_page_tail_topk513",
        ),
        pytest.param(
            "TND",
            128,
            3,
            (1, 6, 9),
            (511, 512, 513),
            512,
            True,
            16,
            id="tnd_query_tile_tail_candidates",
        ),
        pytest.param(
            "PA_BBND",
            128,
            3,
            (5, 7),
            (25599, 25601),
            512,
            False,
            2048,
            id="pa128_cache_boundary",
        ),
        pytest.param(
            "PA_BBND",
            128,
            3,
            (5, 6, 7),
            (65535, 65536, 65537),
            512,
            False,
            2048,
            id="pa128_64k_boundary_query_tail",
        ),
        pytest.param(
            "PA_BBND",
            128,
            3,
            (6, 7),
            (131071, 131073),
            512,
            False,
            2048,
            id="pa128_128k_boundary_query_tail",
        ),
        pytest.param(
            "PA_BBND",
            128,
            3,
            (0, 7),
            (0, 513),
            512,
            True,
            2048,
            id="pa128_mixed_empty_batch",
        ),
    ],
)
def test_generalized_case(
    layout,
    page_size,
    mask,
    query_lengths,
    key_lengths,
    topk,
    candidates,
    candidate_capacity,
):
    _check_kernel(
        layout,
        page_size,
        mask,
        sparse=False,
        candidates=candidates,
        candidate_capacity=candidate_capacity,
        query_lengths=query_lengths,
        key_lengths=key_lengths,
        topk=topk,
        signed_weights=True,
    )


@pytest.mark.npu
@pytest.mark.parametrize("topk", [2048, 8192])
def test_large_topk_ld_workspace(topk):
    _check_kernel(
        "TND",
        128,
        0,
        sparse=False,
        query_lengths=(1,),
        key_lengths=(16384,),
        topk=topk,
        uniform_scale=True,
    )
