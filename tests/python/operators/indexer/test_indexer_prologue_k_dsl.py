# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE.cannbot-dsl under implementation/cannbot-dsl for the full text of the License.

"""K cache precision/performance and the fixed 16-row projection-tail guard."""

import importlib
import math

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

perf_shape = [(1, 0), (6, 1)]


@pytest.mark.parametrize(
    "t", [pytest.param(1, id="decode-tail"), pytest.param(17, id="edge-second-tile")]
)
def test_projection_tail_stays_inside_allocation(t, monkeypatch):
    """A full fixpipe tail tile must not corrupt storage after the allocation."""
    from sgl_kernel_npu.operators.indexer import indexer_prologue_k_dsl

    require_test_impl(indexer_prologue_k_dsl, "cannbot_dsl")
    module = importlib.import_module(
        "sgl_kernel_npu.implementation.cannbot-dsl.indexer.indexer_prologue_k"
    )
    args, kwargs = generate_data(t, 0, seed=20261001)
    d = args[1].shape[0]
    allocation = {}

    class GuardedTorch:
        def __getattr__(self, name):
            return getattr(torch, name)

        def empty(self, size, *args, **kwargs):
            if (
                len(size) == 2
                and size[1] == d
                and t <= size[0] <= t + 16
                and kwargs.get("dtype") == torch.bfloat16
            ):
                backing = torch.empty((size[0] + 16, d), *args, **kwargs)
                backing.view(torch.int16).fill_(0x7FFF)
                allocation.update(rows=size[0], backing=backing)
                return backing[: size[0]]
            return torch.empty(size, *args, **kwargs)

    monkeypatch.setattr(module, "torch", GuardedTorch())
    indexer_prologue_k_dsl(*args, **kwargs, impl="cannbot_dsl")
    torch.npu.synchronize()
    guard = allocation["backing"][allocation["rows"] :].view(torch.int16).cpu()
    assert bool((guard == 0x7FFF).all())


def _make_rope(t, dr, generator):
    positions = torch.arange(t, dtype=torch.float32).unsqueeze(1)
    frequencies = 1.0 / (40000.0 ** (torch.arange(0, dr, 2, dtype=torch.float32) / dr))
    base = (positions * frequencies.unsqueeze(0)).repeat_interleave(2, dim=-1)
    # Deliberately keep adjacent coefficients distinct: the current reference
    # applies sin/cos elementwise even though model-produced values commonly
    # repeat each pair.
    angles = base + torch.rand((t, dr), generator=generator) * 0.25
    return torch.sin(angles).contiguous(), torch.cos(angles).contiguous()


def _make_inputs(t, storage_mode, seed):
    with torch.device("cpu"):
        # DSV4.1 compressor latent 512 -> index head 128, with 64 RoPE features.
        generator = torch.Generator(device="cpu").manual_seed(seed)
        latent = torch.randn(t, 512, generator=generator).bfloat16()
        wk = (
            torch.randn(128, 512, generator=generator, dtype=torch.float32)
            / math.sqrt(512)
        ).bfloat16()
        norm_weight = torch.randn(128, generator=generator, dtype=torch.float32)
        rope_sin, rope_cos = _make_rope(t, 64, generator)
        # A token stores 64 K bytes and 4 scale bytes; mode 1 packs eight tokens.
        cache_shape = (2, 64, 1, 64) if storage_mode == 0 else (2, 8, 1, 8 * (64 + 4))
        k_cache = torch.randint(
            0, 256, cache_shape, generator=generator, dtype=torch.uint8
        )
        k_scale_cache = (
            torch.randint(0, 256, (2, 64, 1, 4), generator=generator, dtype=torch.uint8)
            if storage_mode == 0
            else None
        )
        cache_index = torch.randperm(2 * 64, generator=generator)[:t]
        if t > 1 and seed % 3 == 0:
            cache_index[-1] = -1
        return (
            latent,
            wk,
            norm_weight,
            rope_sin,
            rope_cos,
            k_cache,
            cache_index,
            k_scale_cache,
        )


def generate_data(rows, mode, *, seed=None):
    import torch_npu

    x, w, norm, sin, cos, cache, indices, scale = _make_inputs(
        rows, mode, 20260917 + mode if seed is None else seed
    )
    args = (
        x.npu(),
        torch_npu.npu_format_cast(w.npu(), torch_npu.Format.FRACTAL_NZ),
        norm.npu(),
        sin.npu(),
        cos.npu(),
        cache.npu(),
    )
    kwargs = dict(
        k_scale_cache=None if scale is None else scale.npu(),
        cache_index=indices.npu(),
        storage_mode=mode,
        norm_eps=1e-6,
        combined_block_size=8,
    )
    return args, kwargs


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 0), id="dsv41-decode-separate"),
        pytest.param((6, 1), id="dsv41-verify6-combined"),
        pytest.param((17, 0), id="edge-projection-tail"),
    ],
)
def test_indexer_prologue_k_dsl(shape):
    from sgl_kernel_npu.operators.indexer import indexer_prologue_k_dsl

    require_test_impl(indexer_prologue_k_dsl, "auto")
    args, kwargs = generate_data(*shape)
    executor = make_executor(indexer_prologue_k_dsl, args=args, kwargs=kwargs)
    if get_skip_acc_check():
        require_test_impl(indexer_prologue_k_dsl, executor.impl)
    else:
        require_distinct_impls(indexer_prologue_k_dsl, executor.impl, executor.impl_ref)
        ref_cache = args[-1].clone()
        ref_scale = (
            None if kwargs["k_scale_cache"] is None else kwargs["k_scale_cache"].clone()
        )
        reference = indexer_prologue_k_dsl(
            *args[:-1],
            ref_cache,
            **dict(kwargs, k_scale_cache=ref_scale),
            impl=executor.impl_ref
        )
        actual = executor()
        assert actual is args[-1] and reference is ref_cache
        assert_accuracy(actual, reference)
        assert_accuracy(kwargs["k_scale_cache"], ref_scale)
        executor.accuracy_checked = True
    # Replaying writes identical bytes to the same slots; no accumulating state.
    check_performance(executor, enable_perf=shape in perf_shape)
