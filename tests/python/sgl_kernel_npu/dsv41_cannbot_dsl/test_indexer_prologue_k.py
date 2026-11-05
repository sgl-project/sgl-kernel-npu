# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE.cannbot-dsl in the root of the software repository for the full text of the License.

import math

import pytest

pytest.importorskip("torch_npu")
pytest.importorskip("cannbotdsl")
import torch
from sgl_kernel_npu.operators.indexer import indexer_prologue_k


def _make_rope(t, dr, generator):
    positions = torch.arange(t, dtype=torch.float32).unsqueeze(1)
    frequencies = 1.0 / (40000.0 ** (torch.arange(0, dr, 2, dtype=torch.float32) / dr))
    base = (positions * frequencies.unsqueeze(0)).repeat_interleave(2, dim=-1)
    # Deliberately keep adjacent coefficients distinct: the current reference
    # applies sin/cos elementwise even though model-produced values commonly
    # repeat each pair.
    angles = base + torch.rand((t, dr), generator=generator) * 0.25
    return torch.sin(angles).contiguous(), torch.cos(angles).contiguous()


def _make_inputs(
    t,
    h,
    d,
    dr,
    block_num,
    block_size,
    storage_mode,
    combined_block_size,
    with_scale_cache,
    seed,
):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    latent = torch.randn(t, h, generator=generator).to(torch.bfloat16)
    wk = (
        torch.randn(d, h, generator=generator, dtype=torch.float32) / math.sqrt(h)
    ).to(torch.bfloat16)
    norm_weight = torch.randn(d, generator=generator, dtype=torch.float32)
    rope_sin, rope_cos = _make_rope(t, dr, generator)
    value_bytes = d // 2
    scale_bytes = d // 32
    if storage_mode == 0:
        cache_shape = (block_num, block_size, 1, value_bytes)
    else:
        cache_shape = (
            block_num,
            block_size // combined_block_size,
            1,
            combined_block_size * (value_bytes + scale_bytes),
        )
    k_cache = torch.randint(0, 256, cache_shape, generator=generator, dtype=torch.uint8)
    k_scale_cache = None
    if storage_mode == 0 and with_scale_cache:
        k_scale_cache = torch.randint(
            0,
            256,
            (block_num, block_size, 1, scale_bytes),
            generator=generator,
            dtype=torch.uint8,
        )
    slots = torch.randperm(block_num * block_size, generator=generator)[:t]
    cache_index = slots.to(torch.int64)
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


def _pack_mxfp4(key):
    """Return signed-E2M1 packed data and UE8M0 scale bytes."""
    t, d = key.shape
    groups = key.float().view(t, d // 32, 32)
    amax = groups.abs().amax(dim=-1).clamp_min(6.0 * (2.0**-126))
    scale = torch.exp2(torch.ceil(torch.log2(amax / 6.0)))
    magnitude = (groups.abs() / scale.unsqueeze(-1)).clamp_max(6.0)
    code = torch.full_like(magnitude, 7, dtype=torch.int64)
    for threshold, lower_code in (
        (5.0, 6),
        (3.5, 5),
        (2.5, 4),
        (1.75, 3),
        (1.25, 2),
        (0.75, 1),
        (0.25, 0),
    ):
        code = torch.where(magnitude <= threshold, lower_code, code)
    code = code + (groups < 0).to(torch.int64) * 8
    packed = code[..., 0::2] | (code[..., 1::2] << 4)
    packed_data = packed.flatten(1).to(torch.uint8)
    packed_scale = (torch.log2(scale).to(torch.int64) + 127).to(torch.uint8)
    return packed_data, packed_scale


def _golden(
    latent,
    wk,
    norm_weight,
    norm_eps,
    rope_sin,
    rope_cos,
    k_cache,
    cache_index,
    k_scale_cache,
    storage_mode,
    combined_block_size,
):
    # Match the projection's FP32 accumulation and BF16 output boundary
    # without relying on the host CPU's BF16 GEMM implementation.
    key_fp32 = torch.matmul(latent.float(), wk.float().T).to(torch.bfloat16).float()
    inverse_rms = torch.rsqrt(key_fp32.square().mean(dim=-1, keepdim=True) + norm_eps)
    key = (key_fp32 * inverse_rms * norm_weight).to(torch.bfloat16)

    dr = rope_sin.shape[-1]
    tail = key[:, -dr:].float()
    even = tail[:, 0::2]
    odd = tail[:, 1::2]
    rotated = torch.empty_like(tail)
    rotated[:, 0::2] = even * rope_cos[:, 0::2] - odd * rope_sin[:, 0::2]
    rotated[:, 1::2] = odd * rope_cos[:, 1::2] + even * rope_sin[:, 1::2]
    key[:, -dr:] = rotated.to(torch.bfloat16)

    packed_data, packed_scale = _pack_mxfp4(key)
    expected_cache = k_cache.clone()
    expected_scale = None if k_scale_cache is None else k_scale_cache.clone()
    if storage_mode == 0:
        cache_flat = expected_cache.view(-1, packed_data.shape[-1])
        scale_flat = (
            None
            if expected_scale is None
            else expected_scale.view(-1, packed_scale.shape[-1])
        )
        for row, slot in enumerate(cache_index.tolist()):
            if slot != -1:
                cache_flat[slot] = packed_data[row]
                if scale_flat is not None:
                    scale_flat[slot] = packed_scale[row]
    else:
        block_size = expected_cache.shape[1] * combined_block_size
        value_bytes = packed_data.shape[-1]
        scale_bytes = packed_scale.shape[-1]
        for row, flat_slot in enumerate(cache_index.tolist()):
            if flat_slot == -1:
                continue
            block_id = flat_slot // block_size
            slot = flat_slot % block_size
            group_id = slot // combined_block_size
            sub_slot = slot % combined_block_size
            data_start = sub_slot * value_bytes
            scale_start = combined_block_size * value_bytes + sub_slot * scale_bytes
            expected_cache[
                block_id, group_id, 0, data_start : data_start + value_bytes
            ] = packed_data[row]
            expected_cache[
                block_id, group_id, 0, scale_start : scale_start + scale_bytes
            ] = packed_scale[row]
    return expected_cache, expected_scale


def _run_case(
    t,
    h,
    d,
    dr,
    block_num,
    block_size,
    norm_eps,
    storage_mode,
    combined_block_size,
    with_scale_cache,
    seed,
    duplicate_indices=False,
    block_axis_step=1,
    block_axis_start=0,
):
    torch_npu = pytest.importorskip("torch_npu")
    inputs = _make_inputs(
        t,
        h,
        d,
        dr,
        block_num,
        block_size,
        storage_mode,
        combined_block_size,
        with_scale_cache,
        seed,
    )
    latent, wk, norm_weight, rope_sin, rope_cos, k_cache, cache_index, scale = inputs
    if duplicate_indices:
        cache_index[1] = cache_index[0]
        if storage_mode == 1:
            # Exercise two sub-slots in one combined row as well as an exact
            # duplicate.  Destination-row hashing must keep both ordered.
            cache_index[2] = 0
            cache_index[3] = 1
    expected_cache, expected_scale = _golden(
        latent,
        wk,
        norm_weight,
        norm_eps,
        rope_sin,
        rope_cos,
        k_cache,
        cache_index,
        scale,
        storage_mode,
        combined_block_size,
    )
    wk_npu = torch_npu.npu_format_cast(wk.npu(), torch_npu.Format.FRACTAL_NZ)

    def make_cache_view(tensor):
        if block_axis_step == 1 and block_axis_start == 0:
            return tensor.npu(), None
        stop = block_axis_start + int(tensor.shape[0]) * block_axis_step
        backing_shape = (stop + 1, *tensor.shape[1:])
        backing = torch.full(backing_shape, 0xA5, dtype=torch.uint8)
        view = backing[block_axis_start:stop:block_axis_step]
        view.copy_(tensor)
        before = backing.clone()
        backing_npu = backing.npu()
        view_npu = backing_npu[block_axis_start:stop:block_axis_step]
        selected = set(range(block_axis_start, stop, block_axis_step))
        return view_npu, (backing_npu, before, selected)

    cache_npu, cache_backing = make_cache_view(k_cache)
    if scale is None:
        scale_npu, scale_backing = None, None
    else:
        scale_npu, scale_backing = make_cache_view(scale)
    optional_args = {} if scale_npu is None else {"k_scale_cache": scale_npu}
    actual = indexer_prologue_k(
        latent.npu(),
        wk_npu,
        norm_weight.npu(),
        rope_sin.npu(),
        rope_cos.npu(),
        cache_npu,
        **optional_args,
        cache_index=cache_index.npu(),
        storage_mode=storage_mode,
        norm_eps=norm_eps,
        combined_block_size=combined_block_size,
    )
    torch.npu.synchronize()
    assert actual is cache_npu
    actual_cache = actual.cpu()
    if not torch.equal(actual_cache, expected_cache):
        mismatch = (actual_cache != expected_cache).nonzero()
        details = []
        for coordinate in mismatch[:8]:
            index = tuple(coordinate.tolist())
            details.append(
                (index, int(actual_cache[index]), int(expected_cache[index]))
            )
        pytest.fail(
            f"k_cache mismatch_count={mismatch.shape[0]}, "
            f"first_mismatches={details}"
        )
    if scale_npu is not None:
        actual_scale = scale_npu.cpu()
        if not torch.equal(actual_scale, expected_scale):
            mismatch = (actual_scale != expected_scale).nonzero()
            first = tuple(mismatch[0].tolist())
            pytest.fail(
                f"k_scale_cache mismatch_count={mismatch.shape[0]}, "
                f"first={first}, actual={int(actual_scale[first])}, "
                f"expected={int(expected_scale[first])}"
            )

    for backing_info in (cache_backing, scale_backing):
        if backing_info is None:
            continue
        backing_npu, before, selected = backing_info
        after = backing_npu.cpu()
        for block_index in range(after.shape[0]):
            if block_index not in selected:
                assert torch.equal(after[block_index], before[block_index])


@pytest.mark.npu
@pytest.mark.parametrize(
    "storage_mode,group_size,with_scale",
    [
        pytest.param(0, -1, True, id="mode0_with_scale"),
        pytest.param(1, 8, False, id="mode1_g8"),
    ],
)
def test_indexer_prologue_k_npu(storage_mode, group_size, with_scale):
    _run_case(
        5,
        512,
        128,
        64,
        2,
        64,
        1e-6,
        storage_mode,
        group_size,
        with_scale,
        20260917 + storage_mode + int(with_scale),
    )


@pytest.mark.npu
@pytest.mark.parametrize(
    "storage_mode,group_size,with_scale",
    [
        pytest.param(0, -1, True, id="mode0_with_scale"),
    ],
)
def test_indexer_prologue_k_duplicate_indices_npu(storage_mode, group_size, with_scale):
    _run_case(
        8,
        512,
        128,
        64,
        2,
        64,
        1e-6,
        storage_mode,
        group_size,
        with_scale,
        20260920 + storage_mode,
        duplicate_indices=True,
    )


@pytest.mark.npu
@pytest.mark.parametrize(
    "storage_mode,group_size,with_scale,start,duplicate_indices",
    [
        pytest.param(0, -1, True, 0, False, id="mode0_scale_step2"),
    ],
)
def test_indexer_prologue_k_strided_block_axis_npu(
    storage_mode,
    group_size,
    with_scale,
    start,
    duplicate_indices,
):
    _run_case(
        72,
        512,
        128,
        64,
        2,
        64,
        1e-6,
        storage_mode,
        group_size,
        with_scale,
        30300 + storage_mode + start,
        duplicate_indices=duplicate_indices,
        block_axis_step=2,
        block_axis_start=start,
    )
