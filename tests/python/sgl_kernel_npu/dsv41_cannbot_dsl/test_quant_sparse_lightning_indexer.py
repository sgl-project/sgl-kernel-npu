# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE.cannbot-dsl in the root of the software repository for the full text of the License.

"""Independent reference and precision tests for quant_sparse_lightning_indexer."""

import pytest

pytest.importorskip("torch_npu")
pytest.importorskip("cannbotdsl")
import torch
from _indexer_reference import (
    BLOCK_SIZE,
    CANDIDATE_CAPACITY,
    TOPK,
    _check_topk,
    _make_inputs,
)
from sgl_kernel_npu.operators.indexer import (
    quant_sparse_lightning_indexer,
    quant_sparse_lightning_indexer_metadata,
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
    kernel, make_metadata = (
        quant_sparse_lightning_indexer,
        quant_sparse_lightning_indexer_metadata,
    )
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


def test_sparse_topk_ignores_unselected_blocks():
    _check_topk(
        torch.tensor([0, 2]),
        torch.tensor([2.0, 1.0]),
        torch.tensor([2.0, 9.0, 1.0]),
        torch.tensor([True, False, True]),
    )


@pytest.mark.npu
@pytest.mark.parametrize("page_size", [48, 128])
def test_pa_page_table_reload_across_empty_batches(page_size):
    data = _make_inputs(
        "PA_BBND",
        page_size,
        0,
        sparse=True,
        query_lengths=[1] * 96,
        key_lengths=[129] * 96,
        uniform_scale=True,
    )
    data["candidate_lengths"][1::2].zero_()
    for row in range(1, 96, 2):
        data["allowed"][row].zero_()
    kernel, make_metadata = (
        quant_sparse_lightning_indexer,
        quant_sparse_lightning_indexer_metadata,
    )
    metadata = make_metadata(
        data["candidate_lengths"], **data["sequence"], **data["metadata_attributes"]
    )
    for _ in range(50):
        indices, values = kernel(
            data["q"],
            data["k"],
            data["w"],
            data["descale_q"],
            data["candidate_ids"],
            data["candidate_lengths"],
            descale_k=data["descale_k"],
            **data["sequence"],
            **data["attributes"],
            quant_mode=1,
            block_table=data["block_table"],
            output_idx_offset=data["offsets"],
            metadata=metadata,
            return_value=True,
        )
        indices, values = indices.cpu(), values.cpu()
        for row, (scores, allowed) in enumerate(
            zip(data["reference"], data["allowed"])
        ):
            _check_topk(indices[row, 0], values[row, 0], scores, allowed, row * 10000)


@pytest.mark.npu
@pytest.mark.parametrize(
    "layout,page_size,mask",
    [
        pytest.param("PA_BBND", 128, 0, id="pa128_zero_candidate_tail"),
        pytest.param("PA_BBND", 64, 3, id="pa64_causal_zero_candidate_tail"),
        pytest.param("TND", 128, 3, id="tnd_causal_zero_candidate_tail"),
    ],
)
def test_quant_sparse_lightning_indexer(layout, page_size, mask):
    _check_kernel(layout, page_size, mask, sparse=True)


@pytest.mark.npu
def test_topk16_ld_tail():
    _check_kernel("PA_BBND", 128, 0, sparse=True, topk=16)


@pytest.mark.npu
@pytest.mark.parametrize("layout", ["PA_BBND", "TND"])
def test_invalid_values_are_negative_infinity(layout):
    _check_kernel(layout, 128, 0, sparse=True, key_lengths=(19, 73))


@pytest.mark.npu
@pytest.mark.parametrize("layout", ["PA_BBND", "TND"])
@pytest.mark.parametrize("empty", ["query", "candidate"])
def test_all_rows_empty(layout, empty):
    _check_kernel(layout, 128, 0, sparse=True, key_lengths=(19, 73), empty=empty)


@pytest.mark.npu
@pytest.mark.slow
@pytest.mark.parametrize(
    "batch_size,query_length,key_length",
    [
        pytest.param(12, 6, 131072, id="b12_s1_6_n32_s2_128k_c2048"),
    ],
)
def test_typical_case(batch_size, query_length, key_length):
    _check_kernel(
        "PA_BBND",
        128,
        3,
        sparse=True,
        query_lengths=(query_length,) * batch_size,
        key_lengths=(key_length,) * batch_size,
        signed_weights=True,
        uniform_scale=True,
    )


@pytest.mark.npu
@pytest.mark.parametrize(
    "layout,page_size,mask,query_lengths,key_lengths,topk",
    [
        pytest.param(
            "PA_BBND", 16, 3, (1, 5, 7), (255, 256, 257), 17, id="pa16_ragged_topk17"
        ),
        pytest.param(
            "PA_BBND", 1024, 0, (7,), (1025,), 513, id="pa1024_page_tail_topk513"
        ),
        pytest.param(
            "TND", 128, 3, (1, 6, 9), (511, 512, 513), 512, id="tnd_query_tile_tail"
        ),
        pytest.param(
            "PA_BBND",
            128,
            3,
            (6, 7),
            (65535, 65537),
            512,
            id="pa128_64k_boundary_query_tail",
        ),
        pytest.param(
            "PA_BBND", 128, 3, (0, 7), (0, 513), 512, id="pa128_mixed_empty_batch"
        ),
        pytest.param(
            "PA_BBND",
            128,
            0,
            (1,),
            (8192,),
            512,
            id="pa128_single_query_multi_round_ld",
        ),
    ],
)
def test_generalized_case(layout, page_size, mask, query_lengths, key_lengths, topk):
    _check_kernel(
        layout,
        page_size,
        mask,
        sparse=True,
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
        sparse=True,
        query_lengths=(1,),
        key_lengths=(16384,),
        topk=topk,
        uniform_scale=True,
    )
