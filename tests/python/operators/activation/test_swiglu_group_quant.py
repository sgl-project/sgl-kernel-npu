# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import pytest
import torch

from tests.python.utils.accuracy import assert_accuracy
from tests.python.utils.common import (
    get_skip_acc_check,
    get_test_impls,
    make_executor,
    require_distinct_impls,
    require_test_impl,
)
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard


@pytest.fixture(autouse=True)
def select_backend(configure_public_interface_test):
    from sgl_kernel_npu.operators.activation import swiglu_group_quant

    require_test_impl(swiglu_group_quant, get_test_impls()[0])


# DeepSeek-V4 TP=1 dimensions (2*2048 channels, 256 experts); synthetic batches.
# (8, 65536) is a synthetic wide-row case for column tiling.
perf_shape = [(8, 4096), (8, 65536)]


def _make_inputs(rows, width, dtype=torch.bfloat16, seed=0):
    torch.manual_seed(seed)
    return torch.randn(rows, width, dtype=dtype, device="cpu").npu()


def generate_data(mode, seed, options):
    kwargs = options.copy()
    rows, width = kwargs.pop("rows", 32), kwargs.pop("width", 512)
    x = _make_inputs(rows, width, kwargs.pop("dtype", torch.bfloat16), seed)
    zeros = kwargs.pop("zero_rows", 0)
    x[:zeros].zero_()
    counts = kwargs.pop("counts", None)
    if counts is not None:
        kwargs["group_index"] = torch.tensor(counts, device="npu")
    weight = None
    if kwargs.pop("weight", False):
        torch.manual_seed(seed + 100)
        weight = torch.rand(rows, device="cpu")
        kwargs["topk_weight"] = weight.npu()
    invariant = kwargs.pop("invariant", None)
    kwargs.update(x=x, quant_mode=mode)
    return kwargs, zeros, counts, invariant


E5 = torch.float8_e5m2


@pytest.mark.parametrize(
    "mode,seed,options",
    [
        pytest.param(3, 0, dict(output_origin=True), id="origin-e4m3"),
        pytest.param(3, 1, dict(output_origin=True, dst_type=E5), id="origin-e5m2"),
        pytest.param(
            3,
            2,
            dict(output_origin=True, dtype=torch.float16, rows=16),
            id="origin-fp16",
        ),
        pytest.param(3, 3, dict(output_origin=True, weight=True), id="origin-weight"),
        pytest.param(
            3, 4, dict(output_origin=True, clamp_value=3.0), id="origin-clamp"
        ),
        pytest.param(
            3,
            42,
            dict(output_origin=True, clamp_value=1e-50),
            id="origin-clamp-underflow",
        ),
        pytest.param(
            3,
            5,
            dict(output_origin=True, dst_type=E5, clamp_value=1.5, weight=True),
            id="origin-weight-clamp",
        ),
        pytest.param(3, 6, dict(rows=16, invariant="origin"), id="origin-toggle"),
        pytest.param(1, 7, {}, id="group-e4m3"),
        pytest.param(1, 8, dict(dst_type=E5), id="group-e5m2"),
        pytest.param(1, 9, dict(weight=True), id="group-weight"),
        pytest.param(1, 10, dict(clamp_value=2.0), id="group-clamp"),
        pytest.param(
            1,
            11,
            dict(rows=2048, width=768, dtype=torch.float16),
            id="group-multicore-fp16",
        ),
        pytest.param(
            1, 12, dict(rows=16, invariant="group-size"), id="group-size-inert"
        ),
        pytest.param(1, 0, dict(rows=16, zero_rows=16), id="group-zero"),
        pytest.param(2, 13, {}, id="mx-e4m3"),
        pytest.param(2, 14, dict(dst_type=E5), id="mx-e5m2"),
        pytest.param(2, 15, dict(weight=True), id="mx-weight"),
        pytest.param(2, 16, dict(dst_type=E5, round_scale=True), id="mx-e5m2-round"),
        pytest.param(2, 33, dict(clamp_value=1.0), id="mx-clamp"),
        pytest.param(2, 34, dict(dst_type=E5, clamp_value=1.5), id="mx-e5m2-clamp"),
        pytest.param(2, 35, dict(clamp_value=1.0, weight=True), id="mx-weight-clamp"),
        pytest.param(2, 36, dict(rows=2048, width=768), id="mx-multicore"),
        pytest.param(2, 37, dict(zero_rows=8), id="mx-zero-block"),
        pytest.param(2, 17, dict(rows=16, dtype=torch.float16), id="mx-fp16"),
        pytest.param(1, 39, dict(rows=16, invariant="origin"), id="group-origin-inert"),
        pytest.param(2, 40, dict(rows=16, invariant="origin"), id="mx-origin-inert"),
        pytest.param(3, 18, {}, id="fp8"),
        pytest.param(3, 19, dict(round_scale=True), id="fp8-round"),
        pytest.param(3, 20, dict(round_scale=True, dst_type=E5), id="fp8-e5m2-round"),
        pytest.param(3, 21, dict(round_scale=True, ue8m0_scale=True), id="fp8-e8m0"),
        pytest.param(
            1, 22, dict(counts=[8, 16, 32], invariant="counts"), id="group-index-full"
        ),
        pytest.param(1, 23, dict(counts=[16]), id="group-index-partial"),
        pytest.param(1, 29, dict(rows=4, dst_type=None), id="default-dtype"),
        pytest.param(1, 0, dict(rows=0), id="empty", marks=pytest.mark.extra_checks),
        pytest.param(
            2,
            41,
            dict(rows=8, width=4096, counts=[3, 0, 1, 4] + [0] * 252, clamp_value=1.0),
            id="deepseek-v4-mx-tp1",
        ),
        pytest.param(1, 50, dict(rows=1, width=65536), id="wide-group-bf16-e4m3"),
        pytest.param(
            1,
            51,
            dict(
                rows=8,
                width=65792,
                dtype=torch.float16,
                dst_type=E5,
                weight=True,
                clamp_value=2.0,
            ),
            id="wide-group-fp16-e5m2-weight-clamp",
        ),
        pytest.param(2, 52, dict(rows=1, width=65536), id="wide-mx-bf16-e4m3"),
        pytest.param(
            2,
            53,
            dict(
                rows=8,
                width=65792,
                dtype=torch.float16,
                dst_type=E5,
                weight=True,
                clamp_value=1.0,
            ),
            id="wide-mx-fp16-e5m2-weight-clamp",
        ),
        pytest.param(
            3,
            54,
            dict(rows=1, width=65536, output_origin=True),
            id="wide-fp8-origin-bf16-e4m3",
        ),
        pytest.param(
            3,
            55,
            dict(
                rows=8, width=65792, dtype=torch.float16, dst_type=E5, round_scale=True
            ),
            id="wide-fp8-fp16-e5m2-round",
        ),
        pytest.param(
            3,
            56,
            dict(
                rows=4,
                width=65536,
                round_scale=True,
                ue8m0_scale=True,
                output_origin=True,
                weight=True,
            ),
            id="wide-fp8-e8m0-origin-weight",
        ),
        pytest.param(
            2, 57, dict(rows=8, width=65536, counts=[3, 0, 1, 4]), id="wide-mx-counts"
        ),
    ],
)
def test_swiglu_group_quant(mode, seed, options):
    from sgl_kernel_npu.implementation.torch_native.activation.swiglu_group_quant import (
        _ref_swiglu,
    )
    from sgl_kernel_npu.operators.activation import swiglu_group_quant

    kwargs, zeros, counts, invariant = generate_data(mode, seed, options)
    x = kwargs["x"]
    rows, width = x.shape
    valid = min(rows, sum(counts)) if counts is not None else rows
    weight = kwargs.get("topk_weight")
    impl, impl_ref = get_test_impls()

    def compare_outputs(actual, reference):
        y, scale, origin = actual
        ref_y, ref_scale, ref_origin = reference
        cols = width // 2
        assert y.shape == origin.shape == (rows, cols)
        assert y.dtype == (kwargs.get("dst_type") or torch.float8_e4m3fn)
        assert origin.dtype == x.dtype
        encoded = mode == 2 or kwargs.get("ue8m0_scale", False)
        assert scale.dtype == (torch.float8_e8m0fnu if encoded else torch.float32)
        assert scale.shape == (
            (rows, cols // 64, 2) if mode == 2 else (rows, cols // 128)
        )
        stored = scale[:valid].cpu()
        expected = ref_scale[:valid].cpu()
        if encoded:
            assert_accuracy(stored.view(torch.uint8), expected.view(torch.uint8))
        else:
            assert_accuracy(
                stored,
                expected,
                {"torch.float32": AccuracyStandard(1e-9 if mode == 1 else 0, 1e-6)},
            )
        if not valid:
            return
        quant = y[:valid].cpu().float()
        assert_accuracy(
            quant,
            ref_y[:valid].cpu().float(),
            {"torch.float32": AccuracyStandard(0.02, 0.15)},
        )
        clamp = kwargs.get("clamp_value", 0.0)
        activation = _ref_swiglu(x, clamp, weight)[:valid].cpu()
        if mode == 3 and kwargs.get("output_origin"):
            assert_accuracy(
                origin[:valid].cpu().float(),
                activation,
                {"torch.float32": AccuracyStandard(0.01, 0.01)},
            )
            assert_accuracy(
                origin[:valid].cpu().float(),
                ref_origin[:valid].cpu().float(),
                {"torch.float32": AccuracyStandard(0.01, 0.01)},
            )
        if mode == 2:
            activation = activation.to(x.dtype).float()
        factors = (
            (2.0 ** (stored.view(torch.uint8).float() - 127)) if encoded else stored
        )
        if mode == 2:
            assert torch.all(factors > 0)
        dequant = quant * factors.reshape(valid, -1).repeat_interleave(
            32 if mode == 2 else 128, dim=-1
        )
        assert_accuracy(
            dequant,
            activation,
            {"torch.float32": AccuracyStandard(0.02, 0.25 if mode == 2 else 0.15)},
        )
        if zeros:
            zero_scales = stored[:zeros].view(torch.uint8)
            assert_accuracy(zero_scales, torch.zeros_like(zero_scales))
            assert_accuracy(
                y[:zeros].cpu().view(torch.uint8),
                torch.zeros_like(y[:zeros].cpu().view(torch.uint8)),
            )
            if zeros < rows:
                assert stored[zeros:].view(torch.uint8).any()
        if mode == 1 and valid > zeros:
            assert stored.count_nonzero().item() > 0
        if mode == 2 and clamp and counts is None:
            _, unclamped, _ = swiglu_group_quant(
                **dict(kwargs, clamp_value=0.0), impl=impl_ref
            )
            assert (
                expected.view(torch.uint8) != unclamped[:valid].cpu().view(torch.uint8)
            ).float().mean() >= 0.10
        variants = []
        if mode == 2:
            variants.append(dict(round_scale=not kwargs.get("round_scale", False)))
        if invariant == "counts":
            variants.append(dict(group_index=None))
        if invariant == "origin":
            variants.append(dict(output_origin=True))
        if invariant == "group-size":
            variants.extend(dict(group_size=size) for size in (1, 64, 256))
        for changes in variants:
            other_y, other_scale, other_origin = swiglu_group_quant(
                **dict(kwargs, **changes), impl=impl
            )
            assert (
                other_origin.shape == origin.shape
                and other_origin.dtype == origin.dtype
            )
            assert_accuracy(y.cpu().view(torch.uint8), other_y.cpu().view(torch.uint8))
            assert_accuracy(
                scale.cpu().view(torch.uint8), other_scale.cpu().view(torch.uint8)
            )

    checked = not get_skip_acc_check()
    if checked:
        require_distinct_impls(swiglu_group_quant, impl, impl_ref)
        reference = swiglu_group_quant(**kwargs, impl=impl_ref)
        actual = swiglu_group_quant(**kwargs, impl=impl)
        compare_outputs(actual, reference)

    executor = make_executor(
        swiglu_group_quant,
        kwargs=kwargs,
        impl=impl,
        impl_ref=impl_ref,
        accuracy_checked=checked,
    )
    check_performance(executor, enable_perf=(rows, width) in perf_shape)


@pytest.mark.extra_checks
@pytest.mark.parametrize(
    "width,dtype,kwargs,message",
    [
        (512, torch.bfloat16, dict(quant_mode=7), "Unsupported quant mode"),
        (512, torch.bfloat16, dict(group_list_type=1), "tiling failed"),
        (384, torch.bfloat16, dict(quant_mode=1), "divisible by 256"),
        (192, torch.bfloat16, dict(quant_mode=2), "divisible by 128"),
        (512, torch.bfloat16, dict(dst_type=torch.float16), "dst_type must be"),
        (512, torch.float32, {}, "FLOAT16 or BFLOAT16"),
        (0, torch.bfloat16, dict(group_list_type=1), "tiling failed"),
    ],
)
def test_swiglu_rejected(width, dtype, kwargs, message):
    from sgl_kernel_npu.operators.activation import swiglu_group_quant

    with pytest.raises(RuntimeError, match=message):
        data, _, _, _ = generate_data(1, 0, dict(rows=4, width=width, dtype=dtype))
        swiglu_group_quant(**dict(data, **kwargs), impl=get_test_impls()[0])


@pytest.mark.extra_checks
def test_ascendc_zero_width_group_type_rejected():
    if get_test_impls()[0] not in ("ascendc", "auto"):
        pytest.skip("Tests the AscendC native binding")
    data, _, _, _ = generate_data(1, 0, dict(rows=4, width=0))
    with pytest.raises(RuntimeError, match="tiling failed"):
        torch.ops.npu.swiglu_group_quant(**dict(data, group_list_type=1))


@pytest.mark.extra_checks
def test_ascendc_graph_replay():
    from sgl_kernel_npu.operators.activation import swiglu_group_quant

    if get_test_impls()[0] not in ("ascendc", "auto"):
        pytest.skip("Tests AscendC graph capture")
    stream = torch.npu.Stream()
    captures = []
    for rows, limit in [(8, 1.0), (12, 2.0)]:
        kwargs, _, _, _ = generate_data(
            2,
            0,
            dict(rows=rows, counts=[rows // 2, rows - rows // 2], clamp_value=limit),
        )
        kwargs["impl"] = "ascendc"
        swiglu_group_quant(**kwargs)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream, auto_dispatch_capture=True):
            y, scale, _ = swiglu_group_quant(**kwargs)
        captures.append((graph, kwargs, y, scale))
    for graph, kwargs, y, scale in captures:
        with torch.npu.stream(stream):
            kwargs["x"].neg_()
            expected_y, expected_scale, _ = swiglu_group_quant(**kwargs)
            graph.replay()
        stream.synchronize()
        for actual, expected in ((y, expected_y), (scale, expected_scale)):
            assert_accuracy(
                actual.cpu().view(torch.uint8), expected.cpu().view(torch.uint8)
            )
