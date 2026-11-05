# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE.cannbot-dsl in the root of the software repository for the full text of the License.

"""Independent reference and precision tests for quant_sparse_lightning_indexer_metadata."""

import pytest

pytest.importorskip("torch_npu")
pytest.importorskip("cannbotdsl")
import torch
from _indexer_reference import _assert_schedule, _prefix, _visible_tokens
from sgl_kernel_npu.operators.indexer import quant_sparse_lightning_indexer_metadata

TOPK = 512
QUERY_GROUP_SIZE = 1
KEY_TILE_SIZE = 512
BLOCK_SIZE = 8
CANDIDATE_CAPACITY = 2048


def _check_metadata(
    *, layout, query_lengths, key_lengths, mask=0, zero=False, cmp_ratio=1
):
    device = torch.device("npu", torch.npu.current_device())
    make_metadata = quant_sparse_lightning_indexer_metadata
    used_queries = [0 if zero else length for length in query_lengths]
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
    if mask == 3 and cmp_ratio != 1:
        sequence["cmp_residual_k"] = torch.zeros(
            len(query_lengths), dtype=torch.int32, device=device
        )
    attributes = dict(
        max_seqlen_q=max(query_lengths),
        max_seqlen_k=max(key_lengths),
        num_heads_q=32,
        num_heads_k=1,
        head_dim=128,
        topk=TOPK,
        layout_q="TND",
        layout_k=layout,
        mask_mode=mask,
        cmp_ratio=cmp_ratio,
    )
    lengths = [
        min(CANDIDATE_CAPACITY, (key_lengths[batch] + BLOCK_SIZE - 1) // BLOCK_SIZE)
        for batch, count in enumerate(query_lengths)
        for _ in range(count)
    ]
    attributes.update(quant_mode=1, candidate_block_size=BLOCK_SIZE)
    candidate_lengths = torch.tensor(lengths, dtype=torch.int32, device=device).reshape(
        -1, 1
    )
    metadata = make_metadata(candidate_lengths, **sequence, **attributes).cpu()
    tasks, group_rows = [], {}
    group_size, tile_size = QUERY_GROUP_SIZE, KEY_TILE_SIZE
    for batch, used in enumerate(used_queries):
        for start in range(0, used, group_size):
            active = min(group_size, used - start)
            visible = _visible_tokens(
                key_lengths[batch], used, start + active - 1, mask, cmp_ratio, 0
            )
            length = (
                lengths[_prefix(query_lengths)[batch] + start] * BLOCK_SIZE
                if visible
                else 0
            )
            group = start // group_size
            group_rows[batch, group] = active
            tasks.extend(
                (batch, group, tile)
                for tile in range((length + tile_size - 1) // tile_size)
            )
    workers = int(torch.npu.get_device_properties(device).cube_core_num)
    _assert_schedule(metadata, tasks, group_rows, workers)


def test_schedule_reference_rejects_duplicate_tile():
    metadata = torch.zeros((128, 8), dtype=torch.int32)
    metadata[:2, 0] = 1
    metadata[:2, 6] = 1
    with pytest.raises(AssertionError):
        _assert_schedule(metadata.flatten(), [(0, 0, 0)], {(0, 0): 1})


@pytest.mark.npu
@pytest.mark.parametrize(
    "layout,query_lengths,key_lengths,mask,zero",
    [
        pytest.param("PA_BBND", [7, 3], [513, 777], 0, False, id="pa_ragged"),
        pytest.param("TND", [7, 3], [513, 777], 3, False, id="tnd_causal_ragged"),
        pytest.param("PA_BBND", [1], [131072], 0, False, id="cross_core_ld"),
        pytest.param("PA_BBND", [1], [513], 0, True, id="zero_work"),
    ],
)
def test_quant_sparse_lightning_indexer_metadata(
    layout, query_lengths, key_lengths, mask, zero
):
    _check_metadata(
        layout=layout,
        query_lengths=query_lengths,
        key_lengths=key_lengths,
        mask=mask,
        zero=zero,
    )


@pytest.mark.npu
@pytest.mark.slow
@pytest.mark.parametrize("key_length", [131072], ids=["b12_s1_6_s2_128k_c2048"])
def test_typical_case(key_length):
    _check_metadata(
        layout="PA_BBND",
        query_lengths=[6] * 12,
        key_lengths=[key_length] * 12,
        mask=3,
        cmp_ratio=2,
    )
