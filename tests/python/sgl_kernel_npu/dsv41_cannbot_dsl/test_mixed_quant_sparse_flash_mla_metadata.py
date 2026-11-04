# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE.cannbot-dsl in the root of the software repository for the full text of the License.

"""Structural and argument tests for the local MQSMLA AICPU planner."""

import sys
from pathlib import Path

import pytest

pytest.importorskip("torch_npu")
pytest.importorskip("cannbotdsl")
import importlib

import torch
from sgl_kernel_npu.operators.attention import mixed_quant_sparse_flash_mla_metadata

METADATA = importlib.import_module(
    "sgl_kernel_npu.implementation.cannbot-dsl.attention.mixed_quant_sparse_flash_mla_metadata"
)


@pytest.mark.npu
@pytest.mark.parametrize(
    "spans,has_cmp", [([1], False), ([3, 0, 5], True), ([42, 23], True)]
)
def test_mixed_quant_sparse_flash_mla_metadata_structure(spans, has_cmp):
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("NPU unavailable")
    torch.npu.set_device(0)
    cu = torch.tensor([0, *torch.tensor(spans).cumsum(0).tolist()], dtype=torch.int32)
    rows = sum(spans)
    ori = torch.full((rows, 1), 128, dtype=torch.int32, device="npu")
    cmp = torch.full_like(ori, 129 if has_cmp else 0)
    ori[::4] = 0
    cmp[::4] = 0
    metadata = mixed_quant_sparse_flash_mla_metadata(
        ori,
        cmp,
        cu_seqlens_q=cu.npu(),
        num_heads_q=64,
        num_heads_kv=1,
        head_dim=512,
        quant_mode=1,
        has_cmp_kv=has_cmp,
    )
    assert metadata.dtype == torch.int32 and metadata.device.type == "npu"
    assert metadata.shape == (METADATA.MQSMLA_METADATA_TOTAL_SIZE,)
    host = metadata.cpu()
    fa = host[: METADATA.FD_METADATA_BASE].view(
        METADATA.AIC_CORE_MAX_NUM, METADATA.FA_METADATA_SIZE
    )
    enabled = fa[:, METADATA.FA_CORE_ENABLE_INDEX].ne(0)
    count = min(rows, torch.npu.get_device_properties(0).cube_core_num)
    assert enabled.tolist() == [True] * count + [False] * (len(fa) - count)
    assert torch.count_nonzero(fa[~enabled]) == 0
    active = fa[enabled]
    starts = (
        cu[active[:, METADATA.FA_BN2_START_INDEX].long()]
        + active[:, METADATA.FA_M_START_INDEX]
    )
    ends = (
        cu[active[:, METADATA.FA_BN2_END_INDEX].long()]
        + active[:, METADATA.FA_M_END_INDEX]
    )
    assert starts[0] == 0 and ends[-1] == rows
    assert torch.all(ends > starts)
    torch.testing.assert_close(starts[1:], ends[:-1], rtol=0, atol=0)
    assert (
        torch.count_nonzero(
            active[:, [METADATA.FA_S2_START_INDEX, METADATA.FA_S2_END_INDEX]]
        )
        == 0
    )
    # Only the documented FA/FD region is initialized; trailing storage is reserved.
    assert (
        torch.count_nonzero(
            host[METADATA.FD_METADATA_BASE : METADATA.FD_USED_VEC_NUM_WORD + 1]
        )
        == 0
    )


@pytest.mark.parametrize(
    "change,match",
    [
        ({"num_heads_q": 63}, "metadata supports"),
        ({"quant_mode": 0}, "metadata supports"),
        ({"layout_q": "BSND"}, "TND / PA_BBND"),
        ({"ori_topk_length": torch.ones(2, 1)}, "int32 tensor"),
        ({"cmp_topk_length": torch.ones(3, 1, dtype=torch.int32)}, "same shape"),
    ],
)
def test_mixed_quant_sparse_flash_mla_metadata_rejects_invalid_input(change, match):
    # Keep invalid inputs on the supported device so the public device gate
    # does not mask the upstream argument validation being exercised here.
    kwargs = dict(
        ori_topk_length=torch.ones(2, 1, dtype=torch.int32, device="npu"),
        cmp_topk_length=torch.ones(2, 1, dtype=torch.int32, device="npu"),
        num_heads_q=64,
        num_heads_kv=1,
        head_dim=512,
        quant_mode=1,
    )
    kwargs.update(
        {
            key: value.npu() if isinstance(value, torch.Tensor) else value
            for key, value in change.items()
        }
    )
    with pytest.raises(ValueError, match=match):
        mixed_quant_sparse_flash_mla_metadata(**kwargs)
