# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE.cannbot-dsl under implementation/cannbot-dsl for the full text of the License.

"""DSV4.1 decode/verify attention shapes, generated values and an empty row."""

import pytest
import torch

from tests.python.utils.accuracy import assert_accuracy
from tests.python.utils.common import (
    get_skip_acc_check,
    make_executor,
    require_distinct_impls,
    require_test_impl,
)
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard

D = 512
perf_shape = [(1, 128, 512), (6, 128, 512)]


def _make_kv(batch, tokens, block_size, fp4, generator):
    """Pack KV bytes and scales, then shuffle physical page placement."""
    pages = (tokens + block_size - 1) // block_size
    shape = (batch, pages * block_size, D)
    group = 16 if fp4 else 32
    scales = (
        0.125 + torch.rand((*shape[:-1], D // group), generator=generator) * 0.375
    ).bfloat16()
    if fp4:
        codes = torch.randint(0, 16, shape, generator=generator, dtype=torch.uint8)
        encoded = codes[..., 0::2] | (codes[..., 1::2] << 4)
    else:
        values = (torch.randn(shape, generator=generator) * 2).to(torch.float8_e4m3fn)
        encoded = values.view(torch.uint8)
    packed = torch.cat((encoded, scales.view(torch.uint8)), dim=-1)
    table = (
        torch.randperm(batch * pages, generator=generator).reshape(batch, pages).int()
    )
    physical = torch.empty_like(packed.reshape(batch * pages, block_size, -1))
    physical[table.long().flatten()] = packed.reshape_as(physical)
    return physical, table


def generate_data(rows, ori_count, cmp_count):
    # Model geometry Q[T,64,512], FP8/FP4 KV; synthetic values, not a request dump.
    with torch.device("cpu"):
        generator = torch.Generator().manual_seed(42)
        q = (torch.randn(rows, 64, D, generator=generator) * 0.25).bfloat16()
        kwargs = dict(
            q=q,
            sinks=torch.randn(64, generator=generator),
            cu_seqlens_q=torch.tensor([0, rows], dtype=torch.int32),
            quant_mode=1,
            return_softmax_lse=cmp_count is not None,
        )
        for side, fp4, count in [("ori", False, ori_count), ("cmp", True, cmp_count)]:
            if count is None:
                continue
            physical, table = _make_kv(1, 1024, 128, fp4, generator)
            row_bytes = 320 if fp4 else 544
            physical = physical.reshape(-1, 128, 1, row_bytes)
            stride0 = (
                128 * row_bytes + 64
            )  # Serving caches can have padding between pages.
            backing = torch.full(
                (physical.shape[0] * stride0,), 0xA5, dtype=torch.uint8
            )
            view = backing.as_strided(
                physical.shape, (stride0, row_bytes, row_bytes, 1)
            )
            view.copy_(physical)
            kwargs[side + "_kv"] = backing.npu().as_strided(
                physical.shape, view.stride()
            )
            kwargs[side + "_block_table"] = table
            indices = torch.randperm(1024, generator=generator)[: max(count, 1)].int()
            kwargs[side + "_sparse_indices"] = indices.reshape(1, 1, -1).repeat(
                rows, 1, 1
            )
            kwargs[side + "_topk_length"] = torch.full(
                (rows, 1), count, dtype=torch.int32
            )
        kwargs = {
            k: v.npu() if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()
        }
    kwargs["out"] = torch.empty_like(kwargs["q"])
    lse_shape = (1, rows, 64) if kwargs["return_softmax_lse"] else (0,)
    kwargs["lse"] = torch.empty(lse_shape, dtype=torch.float32)
    return kwargs


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 128, 512), id="dsv41-decode-mixed-kv"),
        pytest.param((6, 128, 512), id="dsv41-static-verify6-mixed-kv"),
        pytest.param((1, 128, None), id="dsv41-decode-original-kv-no-lse"),
        pytest.param((1, 0, 0), id="edge-empty-sparse-row"),
    ],
)
def test_mixed_quant_sparse_flash_mla_dsl(shape):
    from sgl_kernel_npu.operators.attention import (
        mixed_quant_sparse_flash_mla_dsl,
        mixed_quant_sparse_flash_mla_metadata_dsl,
    )

    require_test_impl(mixed_quant_sparse_flash_mla_metadata_dsl, "auto")
    kwargs = generate_data(*shape)
    cmp_length = kwargs.get("cmp_topk_length")
    if cmp_length is None:
        # Metadata requires both length tensors even when compressed KV is absent.
        cmp_length = torch.zeros_like(kwargs["ori_topk_length"])
    kwargs["metadata"] = mixed_quant_sparse_flash_mla_metadata_dsl(
        kwargs["ori_topk_length"],
        cmp_length,
        cu_seqlens_q=kwargs["cu_seqlens_q"],
        num_heads_q=64,
        num_heads_kv=1,
        head_dim=512,
        quant_mode=1,
        has_ori_kv=True,
        has_cmp_kv="cmp_kv" in kwargs,
    )
    executor = make_executor(mixed_quant_sparse_flash_mla_dsl, kwargs=kwargs)
    if get_skip_acc_check():
        require_test_impl(mixed_quant_sparse_flash_mla_dsl, executor.impl)
    else:
        require_distinct_impls(
            mixed_quant_sparse_flash_mla_dsl, executor.impl, executor.impl_ref
        )
        ref_out = torch.empty_like(kwargs["out"])
        ref_lse = torch.empty_like(kwargs["lse"])
        result_ref = mixed_quant_sparse_flash_mla_dsl(
            **dict(kwargs, out=ref_out, lse=ref_lse), impl=executor.impl_ref
        )
        result = executor()
        assert result is result_ref is None
        assert_accuracy(kwargs["out"], ref_out)
        assert_accuracy(
            kwargs["lse"],
            ref_lse,
            {"torch.float32": AccuracyStandard(atol=2e-4, rtol=2e-4)},
        )
        executor.accuracy_checked = True
    # Caller outputs are overwritten, so repeated profiling does not accumulate state.
    check_performance(executor, enable_perf=shape in perf_shape)
