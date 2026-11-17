# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE.cannbot-dsl under implementation/cannbot-dsl for the full text of the License.

"""DSV4.1 decode/verify and empty-key regression, with selected device timing."""

import pytest
import torch

from tests.python.utils.common import (
    get_skip_acc_check,
    make_executor,
    require_distinct_impls,
    require_test_impl,
)
from tests.python.utils.performance import check_performance

SCORE_RTOL = 1 / 128
SCORE_ATOL = 2.5e-5


def _check_topk(indices, values, scores, allowed):
    """Check values, completeness and the TopK boundary without ordering ties."""
    valid = indices != -1
    if values is not None:
        assert torch.isneginf(values[~valid]).all(), "Invalid TopK values must be -inf"
    selected = indices[valid].long()
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


def generate_data(rows, length):
    # DSV4.1 index Q: 32 heads x 128 MXFP4 values. Reconstructed model shapes,
    # not a captured production tensor dump; T=6 models static DSPARK verify.
    torch.manual_seed(42)
    pages = (max(length, 1) + 63) // 64
    q = torch.randint(0, 256, (rows, 32, 64), dtype=torch.uint8)
    k = torch.randint(0, 256, (pages, 64, 1, 64), dtype=torch.uint8)
    qs = torch.full((rows, 32, 2, 2), 127, dtype=torch.uint8)
    ks = torch.full((pages, 64, 1, 2, 2), 127, dtype=torch.uint8)
    w = torch.rand(rows, 32)
    table = torch.arange(pages, dtype=torch.int32).reshape(1, -1)
    sequence = dict(
        cu_seqlens_q=torch.tensor([0, rows], dtype=torch.int32),
        seqused_k=torch.tensor([length], dtype=torch.int32),
    )
    attrs = dict(topk=512, mask_mode=3, cmp_ratio=1, layout_q="TND", layout_k="PA_BBND")
    meta = dict(
        sequence,
        **attrs,
        num_heads_q=32,
        num_heads_k=1,
        head_dim=128,
        max_seqlen_q=rows,
        max_seqlen_k=max(length, 1)
    )
    kwargs = dict(sequence, **attrs, block_table=table, quant_mode=1, return_value=True)
    k = torch.cat((k.reshape(pages, 8, 512), ks.reshape(pages, 8, 32)), -1)
    candidates = torch.full((rows, 1, 2048), -1, dtype=torch.int32)
    blocks = torch.arange(0, (length + 7) // 8, 2, dtype=torch.int32)
    count = blocks.numel()
    candidates[:, :, :count] = blocks
    lengths = torch.full((rows, 1), count, dtype=torch.int32)
    args = (q, k, w, qs, candidates, lengths)
    kwargs["candidate_block_size"] = 8
    meta.update(candidate_block_length=lengths, candidate_block_size=8, quant_mode=1)
    return args, kwargs, meta


def check_indexer(operator, args, kwargs, length):
    executor = make_executor(operator, args=args, kwargs=kwargs)
    if get_skip_acc_check():
        require_test_impl(operator, executor.impl)
        return executor
    require_distinct_impls(operator, executor.impl, executor.impl_ref)
    # Native full ranking provides scores for alternate valid choices at BF16 ties.
    ref = operator(*args, **dict(kwargs, topk=max(length, 1)), impl=executor.impl_ref)
    actual = executor()
    for row in range(args[0].shape[0]):
        scores = torch.full((length,), -float("inf"), device="cpu")
        ref_ids = ref[0][row, 0].cpu()
        ref_vals = ref[1][row, 0].cpu()
        valid = ref_ids >= 0
        scores[ref_ids[valid].long()] = ref_vals[valid].float()
        allowed = torch.isfinite(scores)
        _check_topk(actual[0][row, 0].cpu(), actual[1][row, 0].cpu(), scores, allowed)
    executor.accuracy_checked = True
    return executor


perf_shape = [(1, 4096), (6, 4096)]


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 4096), id="dsv41-decode"),
        pytest.param((6, 4096), id="dsv41-static-verify6"),
        pytest.param((1, 0), id="edge-empty-keys"),
    ],
)
def test_quant_sparse_lightning_indexer_dsl(shape):
    from sgl_kernel_npu.operators.indexer import (
        quant_sparse_lightning_indexer_dsl,
        quant_sparse_lightning_indexer_metadata_dsl,
    )

    require_test_impl(quant_sparse_lightning_indexer_metadata_dsl, "auto")
    args, kwargs, meta = generate_data(*shape)
    kwargs["metadata"] = quant_sparse_lightning_indexer_metadata_dsl(**meta)
    executor = check_indexer(quant_sparse_lightning_indexer_dsl, args, kwargs, shape[1])
    check_performance(executor, enable_perf=shape in perf_shape)
