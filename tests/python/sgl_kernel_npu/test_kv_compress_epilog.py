# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# kv_compress_epilog (A5-only): fuses per-128-block FP8 / per-group MXFP8 quantization of
# the first (d - 64) columns of x with an inline copy of the trailing 64 BF16 rope columns,
# writing the packed row into the in-place kv_compress_cache. FP8 layout1 and MXFP8
# layout2 store [quant | rope], while MXFP8 layout1 stores [rope | quant].
#
# The reference models are computed in plain PyTorch on CPU; the kernel is exercised with
# the same inputs on the NPU and its output bytes are checked against the reference.

import struct
import unittest

import numpy as np
import sgl_kernel_npu
import torch
import torch_npu
from utils import require_npu_op

pytestmark = require_npu_op("kv_compress_epilog")

DEVICE_ID = 0
torch_npu.npu.set_device(int(DEVICE_ID))

SLICE_SIZE = 64  # trailing BF16 rope columns kept verbatim
FP8_E5M2_MAX = 57344.0
FP8_E4M3FN_MAX = 448.0
# Reciprocal-of-max constants the kernel uses for the FP8 path (bit patterns, not 1.0/max).
INV_FP8_E5M2_BITS = 0x37924925
INV_FP8_E4M3_BITS = 0x3B124925
QUANT_MODE_FP8 = 1
QUANT_MODE_MXFP8 = 2
# Representative head dims cover both exact 128-wide groups and a partial final group.
FP8_VALID_DIMS = (192, 320, 512)


def _bits_to_float(bits):
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def _assert_equal(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def _layout1_dims(d, quant_mode, per_group_size):
    """Return (quant_col, scale_col, concat_col, kv_cache_col) for a layout=1 row."""
    quant_col = d - SLICE_SIZE
    if quant_mode == QUANT_MODE_MXFP8:
        scale_col = (quant_col + per_group_size - 1) // per_group_size
        scale_bytes = 1
    else:
        scale_col = (quant_col + 127) // 128
        scale_bytes = 4
    concat_col = quant_col + 2 * SLICE_SIZE + scale_col * scale_bytes
    kv_cache_col = ((concat_col + 127) // 128) * 128
    return quant_col, scale_col, concat_col, kv_cache_col


def _fp8_coeff(fp8_max):
    # The FP8 path multiplies the block max by this bit pattern, so the reference must
    # use the same constant rather than a correctly-rounded 1.0 / max.
    return _bits_to_float(
        INV_FP8_E4M3_BITS if fp8_max == FP8_E4M3FN_MAX else INV_FP8_E5M2_BITS
    )


def _mxfp8_coeff(fp8_max):
    # The MXFP8 path uses maxValue = (float)1.0 / FP8_*_MAX_VALUE on the host/kernel side.
    return np.float32(1.0) / np.float32(fp8_max)


def _ref_fp8_scales(xf, d, fp8_max):
    """Reference float32 scales for the FP8 path (one per 128-wide block)."""
    quant_col, scale_col, _, _ = _layout1_dims(d, QUANT_MODE_FP8, 0)
    coeff = _fp8_coeff(fp8_max)
    scales = torch.zeros((xf.shape[0], scale_col), dtype=torch.float32)
    for j in range(scale_col):
        blk = xf[:, j * 128 : min((j + 1) * 128, quant_col)]
        m = blk.abs().max(dim=1).values
        scales[:, j] = m * coeff
    return scales


def _ref_mxfp8_scales(xf, d, per_group_size, fp8_max, round_scale):
    """Reference E8M0 scale bytes for the MXFP8 path (one per per_group_size group)."""
    _, scale_col, _, _ = _layout1_dims(d, QUANT_MODE_MXFP8, per_group_size)
    coeff = float(_mxfp8_coeff(fp8_max))
    out = torch.zeros((xf.shape[0], scale_col), dtype=torch.uint8)
    for j in range(scale_col):
        blk = xf[:, j * per_group_size : (j + 1) * per_group_size]
        m = blk.abs().max(dim=1).values
        m = torch.clamp(m, min=1e-4)
        m2 = m * coeff  # float32 mul, mirrors Muls(max2, max2, coeff)
        e = (m2.view(torch.int32) >> 23) & 0xFF  # biased exponent of max * coeff
        if round_scale:
            e = e + (e != 0).to(
                torch.int32
            )  # kernel re-bias logic always adds one step
        out[:, j] = e.to(torch.uint8)
    return out


def _as_uint8(t):
    """Reinterpret fp8 bytes as uint8 (little-endian byte order)."""
    return t.cpu().contiguous().view(torch.uint8)


def _make_inputs(
    num_tokens,
    num_slots,
    d,
    slot_dtype=torch.int32,
    seed=0,
    drop_slot=False,
    slot_mapping_values=None,
    fill_value=None,
):
    torch.manual_seed(seed)
    if fill_value is None:
        x = torch.randn(num_tokens, d, dtype=torch.bfloat16)
    else:
        x = torch.full((num_tokens, d), fill_value, dtype=torch.bfloat16)
    if slot_mapping_values is None:
        slot_mapping = torch.arange(num_tokens) % num_slots
    else:
        slot_mapping = torch.as_tensor(slot_mapping_values)
    if drop_slot:
        slot_mapping[0] = -1
    return x, slot_mapping.to(slot_dtype)


def _check_untouched_slots(byte, slot_mapping, valid):
    """Rows no token maps to must remain all zero (kernel never touches them)."""
    written = set(slot_mapping[valid].tolist())
    for s in range(byte.shape[0]):
        if s not in written:
            _assert_equal(byte[s], torch.zeros_like(byte[s]))


def _check_fp8_layout1(
    num_tokens,
    num_slots,
    d,
    fp8_max,
    slot_dtype,
    drop_slot,
    seed,
    slot_mapping_values=None,
    fill_value=None,
):
    x, slot_mapping = _make_inputs(
        num_tokens,
        num_slots,
        d,
        slot_dtype,
        seed,
        drop_slot,
        slot_mapping_values,
        fill_value,
    )
    quant_col, scale_col, concat_col, kv_cache_col = _layout1_dims(d, QUANT_MODE_FP8, 0)
    fp8_dtype = torch.float8_e4m3fn if fp8_max == FP8_E4M3FN_MAX else torch.float8_e5m2

    cache = torch.zeros((num_slots, kv_cache_col), dtype=fp8_dtype).npu()
    torch.ops.npu.kv_compress_epilog(
        cache,
        x.npu(),
        slot_mapping.npu(),
        quant_group_size=0,
        quant_mode=QUANT_MODE_FP8,
        round_scale_flag=False,
        layout=1,
    )
    torch.npu.synchronize()

    xf = x.float()
    valid = slot_mapping != -1
    written_slots = slot_mapping[valid].long()
    byte = _as_uint8(cache)
    cache_f32 = cache.cpu().float()

    # 1. Stored per-block float32 scales must match the reference exactly.
    scale_off = quant_col + 2 * SLICE_SIZE
    scales_cached = (
        byte[:, scale_off : scale_off + scale_col * 4].contiguous().view(torch.float32)
    )
    ref_scales = _ref_fp8_scales(xf, d, fp8_max)
    torch.testing.assert_close(
        scales_cached[written_slots], ref_scales[valid], rtol=1e-5, atol=1e-7
    )

    # 2. Dequantized quant region must recover x (FP8 has a ~6% relative step).
    scales_exp = (
        scales_cached[:, :, None]
        .expand(-1, -1, 128)
        .reshape(num_slots, -1)[:, :quant_col]
    )
    deq = cache_f32[:, :quant_col] * scales_exp
    torch.testing.assert_close(
        deq[written_slots], xf[valid, :quant_col], rtol=0.15, atol=0.02
    )

    # 3. The trailing 64 BF16 rope columns are copied byte-exactly.
    rope_bytes = byte[:, quant_col : quant_col + 2 * SLICE_SIZE]
    rope_ref = x[:, -SLICE_SIZE:].contiguous().view(torch.uint8)
    _assert_equal(rope_bytes[written_slots], rope_ref[valid])

    # 4. Row padding is zero-filled.
    _assert_equal(byte[:, concat_col:], torch.zeros_like(byte[:, concat_col:]))

    _check_untouched_slots(byte, slot_mapping, valid)


def _check_mxfp8_layout1(
    num_tokens, num_slots, d, per_group_size, fp8_max, round_scale, seed
):
    x, slot_mapping = _make_inputs(num_tokens, num_slots, d, seed=seed, drop_slot=True)
    quant_col, scale_col, concat_col, kv_cache_col = _layout1_dims(
        d, QUANT_MODE_MXFP8, per_group_size
    )
    fp8_dtype = torch.float8_e4m3fn if fp8_max == FP8_E4M3FN_MAX else torch.float8_e5m2

    cache = torch.zeros((num_slots, kv_cache_col), dtype=fp8_dtype).npu()
    torch.ops.npu.kv_compress_epilog(
        cache,
        x.npu(),
        slot_mapping.npu(),
        quant_group_size=per_group_size,
        quant_mode=QUANT_MODE_MXFP8,
        round_scale_flag=round_scale,
        layout=1,
    )
    torch.npu.synchronize()

    xf = x.float()
    valid = slot_mapping != -1
    byte = _as_uint8(cache)
    cache_f32 = cache.cpu().float()

    # 1. Stored E8M0 scale bytes must match the reference exactly.
    scale_off = quant_col + 2 * SLICE_SIZE
    scale_cached = byte[:, scale_off : scale_off + scale_col]
    ref_scales = _ref_mxfp8_scales(xf, d, per_group_size, fp8_max, round_scale)
    _assert_equal(scale_cached[valid], ref_scales[valid])

    # 2. Quantization must use the power-of-two scale represented by the stored
    # E8M0 exponent. Values that exceed the FP8 range under that scale saturate.
    divisor = torch.pow(2.0, scale_cached.to(torch.float32) - 127.0)
    divisor_exp = (
        divisor[:, :, None]
        .expand(-1, -1, per_group_size)
        .reshape(num_slots, -1)[:, :quant_col]
    )
    value_off = 2 * SLICE_SIZE
    deq = cache_f32[:, value_off : value_off + quant_col] * divisor_exp
    expected = xf[valid, :quant_col]
    unsaturated = (expected.abs() / divisor_exp[valid]) <= fp8_max
    if not unsaturated.any():
        raise AssertionError("test input must contain unsaturated MXFP8 values")
    torch.testing.assert_close(
        deq[valid][unsaturated], expected[unsaturated], rtol=0.25, atol=0.02
    )

    # 3. Rope columns copied byte-exactly.
    rope_ref = x[:, -SLICE_SIZE:].contiguous().view(torch.uint8)
    _assert_equal(byte[:, : 2 * SLICE_SIZE][valid], rope_ref[valid])

    # 4. Row padding is zero-filled.
    _assert_equal(byte[:, concat_col:], torch.zeros_like(byte[:, concat_col:]))
    _check_untouched_slots(byte, slot_mapping, valid)


def _check_mxfp8_layout2(num_tokens, d, per_group_size, block_size, fp8_max, seed):
    quant_col, scale_col, _, _ = _layout1_dims(d, QUANT_MODE_MXFP8, per_group_size)
    value_per_token = quant_col + 2 * SLICE_SIZE
    scale_per_token = ((scale_col + 7) // 8) * 8
    block_stride = block_size * (value_per_token + scale_per_token)
    num_blocks = (num_tokens + block_size - 1) // block_size
    fp8_dtype = torch.float8_e4m3fn if fp8_max == FP8_E4M3FN_MAX else torch.float8_e5m2

    x, slot_mapping = _make_inputs(num_tokens, num_tokens, d, seed=seed, drop_slot=True)
    cache = torch.zeros(
        (num_blocks, block_size, value_per_token + scale_per_token), dtype=fp8_dtype
    ).npu()
    torch.ops.npu.kv_compress_epilog(
        cache,
        x.npu(),
        slot_mapping.npu(),
        quant_group_size=per_group_size,
        quant_mode=QUANT_MODE_MXFP8,
        round_scale_flag=True,
        layout=2,
    )
    torch.npu.synchronize()

    xf = x.float()
    cache_flat = cache.cpu().reshape(num_blocks, block_stride)  # fp8
    byte_flat = _as_uint8(cache).reshape(num_blocks, block_stride)
    ref_scales = _ref_mxfp8_scales(xf, d, per_group_size, fp8_max, True)
    rope_ref = x[:, -SLICE_SIZE:].contiguous().view(torch.uint8)

    for i in range(num_tokens):
        s = int(slot_mapping[i].item())
        if s == -1:
            continue
        b, si = divmod(s, block_size)
        val = cache_flat[
            b, si * value_per_token : si * value_per_token + value_per_token
        ]
        # value region = [fp8 quant][BF16 rope bytes]
        scales = torch.pow(2.0, ref_scales[i].to(torch.float32) - 127.0)
        scales_exp = scales[:, None].expand(-1, per_group_size).reshape(-1)[:quant_col]
        deq = val[:quant_col].float() * scales_exp
        torch.testing.assert_close(deq, xf[i, :quant_col], rtol=0.25, atol=0.02)
        _assert_equal(_as_uint8(val[quant_col:]), rope_ref[i])
        # All scale bytes for this slot (not just the first group).
        scale_off = block_size * value_per_token + si * scale_per_token
        _assert_equal(byte_flat[b, scale_off : scale_off + scale_col], ref_scales[i])

    written_slots = set(slot_mapping[slot_mapping != -1].tolist())
    for s in range(num_blocks * block_size):
        if s in written_slots:
            continue
        b, si = divmod(s, block_size)
        value_off = si * value_per_token
        scale_off = block_size * value_per_token + si * scale_per_token
        _assert_equal(
            byte_flat[b, value_off : value_off + value_per_token],
            torch.zeros(value_per_token, dtype=torch.uint8),
        )
        _assert_equal(
            byte_flat[b, scale_off : scale_off + scale_per_token],
            torch.zeros(scale_per_token, dtype=torch.uint8),
        )


class TestKvCompressEpilog(unittest.TestCase):

    def test_fp8_e4m3fn_layout1_single_block(self):
        # d=192: one 128-wide quant block, single slot per row.
        _check_fp8_layout1(
            16, 16, 192, FP8_E4M3FN_MAX, torch.int32, drop_slot=True, seed=0
        )

    def test_fp8_e4m3fn_layout1_multi_block_scales(self):
        # d=320: two 128-wide quant blocks -> two inline float32 scales.
        _check_fp8_layout1(
            16, 16, 320, FP8_E4M3FN_MAX, torch.int32, drop_slot=True, seed=1
        )

    def test_fp8_e5m2_layout1_int64_slots(self):
        # Exercises the e5m2 cache + int64 slot_mapping dtype combination.
        _check_fp8_layout1(
            16, 16, 192, FP8_E5M2_MAX, torch.int64, drop_slot=True, seed=2
        )

    def test_fp8_e4m3fn_layout1_multi_core(self):
        # Large batch to exercise multi-core tiling and the rowFactor UB loop.
        _check_fp8_layout1(
            4096, 4096, 320, FP8_E4M3FN_MAX, torch.int32, drop_slot=False, seed=3
        )

    def test_fp8_e5m2_layout1_d512_sparse_shuffled_slots(self):
        # Covers the upstream production shape, a partial final 128-wide group,
        # and non-contiguous scatter destinations in one focused case.
        slot_mapping = (
            torch.randperm(64, generator=torch.Generator().manual_seed(4)) * 2
        )
        _check_fp8_layout1(
            64,
            128,
            512,
            FP8_E5M2_MAX,
            torch.int32,
            drop_slot=False,
            seed=4,
            slot_mapping_values=slot_mapping,
        )

    def test_fp8_e4m3fn_layout1_zero_input(self):
        # Exercises the zero block-max path without accepting NaN quant results.
        _check_fp8_layout1(
            16,
            16,
            192,
            FP8_E4M3FN_MAX,
            torch.int32,
            drop_slot=False,
            seed=5,
            fill_value=0.0,
        )

    def test_fp8_layout2_rejected(self):
        d = 192
        block_size = 8
        x = torch.randn(1, d, dtype=torch.bfloat16).npu()
        slot_mapping = torch.zeros(1, dtype=torch.int32).npu()
        cache = torch.zeros(
            (1, block_size, d + SLICE_SIZE), dtype=torch.float8_e4m3fn
        ).npu()

        with self.assertRaisesRegex(RuntimeError, "layout=2 only supports MXFP8"):
            torch.ops.npu.kv_compress_epilog(
                cache,
                x,
                slot_mapping,
                quant_group_size=0,
                quant_mode=QUANT_MODE_FP8,
                round_scale_flag=False,
                layout=2,
            )

    def test_mxfp8_e4m3fn_layout1_group64_round_scale(self):
        _check_mxfp8_layout1(16, 16, 192, 64, FP8_E4M3FN_MAX, round_scale=True, seed=4)

    def test_mxfp8_e4m3fn_layout1_group128_no_round_scale(self):
        _check_mxfp8_layout1(
            16, 16, 192, 128, FP8_E4M3FN_MAX, round_scale=False, seed=5
        )

    def test_mxfp8_e5m2_layout1_group64_round_scale(self):
        _check_mxfp8_layout1(16, 16, 192, 64, FP8_E5M2_MAX, round_scale=True, seed=6)

    def test_mxfp8_e4m3fn_layout1_group64_multi_row(self):
        # Exercises rowFactor > 1 for the MXFP8 path at the upstream D=512 shape.
        _check_mxfp8_layout1(
            4096, 4096, 512, 64, FP8_E4M3FN_MAX, round_scale=True, seed=7
        )

    def test_mxfp8_e4m3fn_layout2(self):
        # Block-major layout: [num_blocks, block_size, value_per_token + scale_per_token].
        _check_mxfp8_layout2(16, 192, 128, 8, FP8_E4M3FN_MAX, seed=7)

    def test_mxfp8_e4m3fn_layout2_multi_scale_partial_block(self):
        # Four scales per token plus a partially occupied final cache block.
        _check_mxfp8_layout2(17, 320, 64, 8, FP8_E4M3FN_MAX, seed=9)


if __name__ == "__main__":
    unittest.main()
