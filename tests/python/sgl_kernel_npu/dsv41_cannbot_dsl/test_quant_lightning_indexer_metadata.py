# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE.cannbot-dsl in the root of the software repository for the full text of the License.

"""Independent reference and precision tests for quant_lightning_indexer_metadata."""

import os
from collections import Counter, defaultdict
from itertools import accumulate

import pytest

pytest.importorskip("torch_npu")
pytest.importorskip("cannbotdsl")
import torch
from sgl_kernel_npu.operators.indexer import quant_lightning_indexer_metadata

TOPK = 512
QUERY_GROUP_SIZE = 6
KEY_TILE_SIZE = 256


def _prefix(lengths):
    return [0, *accumulate(lengths)]


def _visible_tokens(key_length, used_query, row, mask, ratio, residual):
    if row >= used_query:
        return 0
    if mask == 0:
        return key_length
    return min(
        key_length,
        max(0, (key_length * ratio + residual - used_query + row + 1) // ratio),
    )


def _npu_device():
    pytest.importorskip("torch_npu")
    device = torch.device("npu", int(os.environ.get("NPU_DEVICE_ID", "0")))
    torch.npu.set_device(device)
    return device


def _load_metadata():
    return quant_lightning_indexer_metadata


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


def _check_metadata(
    *, layout, query_lengths, key_lengths, mask=0, zero=False, cmp_ratio=1
):
    device = _npu_device()
    make_metadata = _load_metadata()
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
    metadata = make_metadata(**sequence, **attributes).cpu()
    tasks, group_rows = [], {}
    group_size, tile_size = QUERY_GROUP_SIZE, KEY_TILE_SIZE
    for batch, used in enumerate(used_queries):
        for start in range(0, used, group_size):
            active = min(group_size, used - start)
            visible = _visible_tokens(
                key_lengths[batch], used, start + active - 1, mask, cmp_ratio, 0
            )
            length = visible
            group = start // group_size
            group_rows[batch, group] = active
            tasks.extend(
                (batch, group, tile)
                for tile in range((length + tile_size - 1) // tile_size)
            )
    workers = int(torch.npu.get_device_properties(device).cube_core_num)
    _assert_schedule(metadata, tasks, group_rows, workers)


def test_schedule_reference_rejects_missing_tile():
    with pytest.raises(AssertionError):
        _assert_schedule(torch.zeros(1024, dtype=torch.int32), [(0, 0, 0)], {(0, 0): 1})


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
def test_quant_lightning_indexer_metadata(
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
@pytest.mark.parametrize(
    "key_length", [65536, 131072], ids=["b12_s1_6_s2_64k", "b12_s1_6_s2_128k"]
)
def test_typical_case(key_length):
    _check_metadata(
        layout="PA_BBND",
        query_lengths=[6] * 12,
        key_lengths=[key_length] * 12,
        mask=3,
        cmp_ratio=2,
    )
