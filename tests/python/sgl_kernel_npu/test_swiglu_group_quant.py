# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# swiglu_group_quant (A5-only): splits x's last dim into (gate, up), applies SwiGLU, and quantizes
# the result. Three quant modes:
#
#   quant_mode=1 (group) per-128-block scales, fp32;      scale = amax * (1 / fp8_max)
#   quant_mode=2 (mx)    per-32-block scales, e8m0;       power-of-two scale from the bf16 exponent
#   quant_mode=3 (fp8)   per-128-block scales, fp32 or e8m0; always scaled against 448.0
#
# The references below re-derive each mode from the kernel's own arithmetic rather than from a
# generic "quantize" formula, because the three modes disagree in ways that matter: mode 1 uses a
# truncated 1/448 bit pattern, mode 2 works entirely in the bf16 exponent domain, and mode 3 scales
# against 448.0 regardless of dst_type. Comparing against a textbook reference would fail.
#
# Two of the op's arguments are accepted but inert by the tiling and are covered as such:
# `group_size` never reaches the tiling math, and `round_scale` only affects mode 3.

import struct
import unittest

import sgl_kernel_npu
import torch
import torch_npu
from utils import require_npu_op

pytestmark = require_npu_op("swiglu_group_quant")

DEVICE_ID = 0
torch_npu.npu.set_device(int(DEVICE_ID))

GROUP_QUANT = 1
MX_QUANT = 2
FP8_QUANT = 3

FP8_E4M3FN_MAX = 448.0
FP8_E5M2_MAX = 57344.0
# The reciprocal bit patterns the kernel multiplies by (not a correctly-rounded 1.0 / max).
INV_FP8_E4M3_BITS = 0x3B124925
INV_FP8_E5M2_BITS = 0x37924925
GROUP_SCALE_RTOL = 1e-6
GROUP_SCALE_ATOL = 1e-9

# Mode 2 subtracts the fp8 format's own max exponent from the block's max bf16 exponent.
LOWER_BOUND_OF_MAX_EXP_FOR_E4M3 = 0x0400
LOWER_BOUND_OF_MAX_EXP_FOR_E5M2 = 0x0780
BF16_EXP_MASK = 0x7F80
E8M0_BIAS = 127
# For an fp16 input the kernel truncates fp16 -> bf16 before reading the exponent, which re-biases
# the 5-bit fp16 exponent (bias 15) into bf16's 8-bit field (bias 127): +112, shifted up by 7.
FP16_EXP_MASK = 0x7C00
BF16_EXP_REBIAS = 112 << 7


def _bits_to_float(bits):
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def _is_e5m2(dtype):
    return dtype == torch.float8_e5m2


def _coeff_for(dtype):
    """The 1/fp8_max multiplier the kernel uses for the given output fp8 dtype."""
    return _bits_to_float(INV_FP8_E5M2_BITS if _is_e5m2(dtype) else INV_FP8_E4M3_BITS)


def _exp_lower_bound_for(dtype):
    return (
        LOWER_BOUND_OF_MAX_EXP_FOR_E5M2
        if _is_e5m2(dtype)
        else LOWER_BOUND_OF_MAX_EXP_FOR_E4M3
    )


def _split(x):
    gate = x[..., : x.shape[-1] // 2]
    up = x[..., x.shape[-1] // 2 :]
    return gate, up


def _ref_swiglu(x, clamp_value=None, topk_weight=None):
    """The kernel's fused activation, in fp32, matching VFSwiGlu + the optional clamp/weight.

    The clamp is asymmetric: gate is only bounded above, while up is bounded on both sides.
    """
    gate, up = _split(x.float())
    if clamp_value is not None and clamp_value != 0.0:
        gate = torch.clamp(gate, max=clamp_value)
        up = torch.clamp(up, min=-clamp_value, max=clamp_value)
    out = gate / (1.0 + torch.exp(-gate)) * up
    if topk_weight is not None:
        out = out * topk_weight.float().unsqueeze(-1)
    return out


def _blocks(t, block):
    """Reshape the last dim into non-overlapping `block`-wide groups, padding the tail."""
    n, cols = t.shape
    nblocks = (cols + block - 1) // block
    padded = torch.zeros((n, nblocks * block), dtype=t.dtype)
    padded[:, :cols] = t
    return padded.view(n, nblocks, block)


def _ref_group_scales(t, dst_dtype):
    """Mode 1: one fp32 scale per 128-wide block, amax * coeff (0 for an all-zero block)."""
    amax = _blocks(t, 128).abs().amax(dim=-1)
    coeff = _coeff_for(dst_dtype)
    scale = amax * coeff
    return torch.where(amax == 0, torch.zeros_like(scale), scale)


def _mx_exp_field(t, src_dtype):
    """The exponent field the MX path reads, per element, in bf16's bit position.

    The kernel writes the fp32 SwiGLU result into a T0 (bf16/fp16) buffer with CAST_RINT -- i.e.
    round-to-nearest-even to the *input* dtype -- and only then takes exponents. So the value seen
    here is the input-dtype-rounded activation, not the fp32 one.
    """
    if src_dtype == torch.float16:
        # fp16 goes through a CAST_TRUNC fp16 -> bf16 first, which keeps the top 7 mantissa bits
        # (irrelevant to the exponent) and re-biases 15 -> 127. Masking with FP16_EXP_MASK drops the
        # sign, which must not leak into the field.
        field = (
            _blocks(t.to(torch.float16), 32).view(torch.int16).to(torch.int32)
            & FP16_EXP_MASK
        ) >> 10
        exp = (field << 7) + BF16_EXP_REBIAS
        # An exponent field of 0 (zero, or a denormal) has nothing to re-bias and casts to 0; this
        # also keeps the zero padding the block splitter adds from inventing an exponent. An all-ones
        # field is inf/NaN, which the kernel remaps to bf16's all-ones field before maximizing.
        exp = torch.where(field == 0, torch.zeros_like(exp), exp)
        return torch.where(field == 0x1F, torch.full_like(exp, BF16_EXP_MASK), exp)
    bits = _blocks(t.to(torch.bfloat16), 32).to(torch.float32).view(torch.int32) >> 16
    return bits & BF16_EXP_MASK


def _ref_mx_scale_bytes(t, dst_dtype, src_dtype):
    """Mode 2: one e8m0 byte per 32-wide block, derived from the block's max bf16 exponent.

    Mirrors VFComputeMaxExp + VFComputeScale: the bf16 exponent field of |t| is maximized over the
    block, clamped up to the fp8 format's own max exponent, and the difference (shifted by bf16's 7
    mantissa bits) is the e8m0 byte. An all-zero block yields byte 0.
    """
    lower = _exp_lower_bound_for(dst_dtype)
    max_exp = _mx_exp_field(t, src_dtype).amax(dim=-1)
    max_exp = torch.clamp(max_exp, min=lower)
    return (max_exp - lower) >> 7


def _ref_fp8_scales(t, dst_dtype, round_scale):
    """Mode 3: one scale per 128-wide block, always computed against 448.0, with a 1e-4 amax floor.

    Without round_scale the scale is the raw fp32 product. With it the scale is rounded up to a power
    of two by taking the exponent and adding one when any mantissa bit is set. dst_type does not
    enter the arithmetic here: the kernel hardcodes the e4m3 constants (448.0 / its reciprocal) for
    this mode, so an e5m2 output is deliberately over-scaled rather than saturation-bound.
    """
    amax = _blocks(t, 128).abs().amax(dim=-1)
    clamp_scale = torch.clamp(amax, min=1e-4)
    scale = clamp_scale * _coeff_for(torch.float8_e4m3fn)
    if not round_scale:
        return scale
    bits = scale.view(torch.int32)
    exp = ((bits >> 23) & 0xFF) - 127 + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    assert (
        int(exp.to(torch.int64).abs().max()) < 127
    ), "power-of-two scale out of fp32 range"
    return torch.pow(2.0, exp.to(torch.float32))


def _dequantize(y, scale_bytes_or_floats, block, split_d):
    """Rebuild the pre-quant values: y * scale, broadcasting each scale over its block."""
    scales = scale_bytes_or_floats.to(torch.float32)
    if scale_bytes_or_floats.dtype == torch.uint8:
        scales = torch.pow(2.0, scales - E8M0_BIAS)
    expanded = scales.repeat_interleave(block, dim=-1)[:, :split_d]
    return y.float() * expanded, expanded


def _call(
    x,
    topk_weight=None,
    group_index=None,
    dst_type=None,
    quant_mode=GROUP_QUANT,
    group_size=128,
    round_scale=False,
    ue8m0_scale=False,
    output_origin=False,
    group_list_type=0,
    clamp_value=0.0,
):
    return torch.ops.npu.swiglu_group_quant(
        x,
        topk_weight,
        group_index,
        dst_type,
        quant_mode,
        group_size,
        round_scale,
        ue8m0_scale,
        output_origin,
        group_list_type,
        clamp_value,
    )


def _make_inputs(num_tokens, d, dtype=torch.bfloat16, seed=0, device="npu"):
    torch.manual_seed(seed)
    x = torch.randn(num_tokens, d, dtype=dtype)
    return x.to(device)


def _check_swiglu_activation(d, dst_type, clamp_value, seed, use_weight):
    """Mode 3 with output_origin gives y_origin: the exact fused activation before quantization.

    This is the strongest available check of the split/clamp/weight math, because it does not depend
    on any of the mode-specific scale arithmetic.
    """
    num_tokens = 32
    x = _make_inputs(num_tokens, d, seed=seed)
    topk_weight = None
    if use_weight:
        torch.manual_seed(seed + 100)
        topk_weight = torch.rand(num_tokens, dtype=torch.float32).npu()

    y, scale, y_origin = _call(
        x,
        topk_weight=topk_weight,
        dst_type=dst_type,
        quant_mode=FP8_QUANT,
        output_origin=True,
        clamp_value=clamp_value,
    )
    torch.npu.synchronize()

    ref = _ref_swiglu(x.cpu(), clamp_value, topk_weight.cpu() if use_weight else None)
    torch.testing.assert_close(y_origin.cpu().float(), ref, rtol=0.01, atol=0.01)

    # y_origin is the source dtype; y is fp8 and must be consistent with the stored scales.
    assert y_origin.dtype == x.dtype
    assert y.dtype == dst_type
    # Mode 3 without ue8m0_scale stores fp32 scales, so they are used as values, not reinterpreted
    # as e8m0 bytes.
    assert scale.dtype == torch.float32
    deq, _ = _dequantize(y.cpu(), scale.cpu(), 128, ref.shape[-1])
    torch.testing.assert_close(deq, ref, rtol=0.15, atol=0.02)


def _check_group_mode(
    num_tokens, d, dst_type, seed, group_size, use_weight, clamp_value
):
    x = _make_inputs(num_tokens, d, seed=seed)
    topk_weight = None
    if use_weight:
        torch.manual_seed(seed + 100)
        topk_weight = torch.rand(num_tokens, dtype=torch.float32).npu()

    y, scale, _ = _call(
        x,
        topk_weight=topk_weight,
        dst_type=dst_type,
        quant_mode=GROUP_QUANT,
        group_size=group_size,
        clamp_value=clamp_value,
    )
    torch.npu.synchronize()

    ref = _ref_swiglu(x.cpu(), clamp_value, topk_weight.cpu() if use_weight else None)
    split_d = d // 2
    assert y.shape == (num_tokens, split_d)
    assert scale.shape == (num_tokens, (split_d + 127) // 128)
    assert scale.dtype == torch.float32

    # 1. Device MicroAPI activation/reduction can differ from the CPU reference by a few fp32 ULPs.
    ref_scales = _ref_group_scales(ref, dst_type)
    torch.testing.assert_close(
        scale.cpu(), ref_scales, rtol=GROUP_SCALE_RTOL, atol=GROUP_SCALE_ATOL
    )

    # 2. Dequantizing recovers the activation to within fp8 precision.
    deq, _ = _dequantize(y.cpu(), scale.cpu(), 128, split_d)
    nonzero = ref_scales.repeat_interleave(128, dim=-1)[:, :split_d] != 0
    assert nonzero.any(), "test input must contain a non-zero block"
    torch.testing.assert_close(deq[nonzero], ref[nonzero], rtol=0.15, atol=0.02)


def _check_mx_mode(
    num_tokens, d, dst_type, seed, use_weight, round_scale, clamp_value=0.0
):
    x = _make_inputs(num_tokens, d, seed=seed)
    topk_weight = None
    if use_weight:
        torch.manual_seed(seed + 100)
        topk_weight = torch.rand(num_tokens, dtype=torch.float32).npu()

    y, scale, _ = _call(
        x,
        topk_weight=topk_weight,
        dst_type=dst_type,
        quant_mode=MX_QUANT,
        round_scale=round_scale,
        clamp_value=clamp_value,
    )
    torch.npu.synchronize()

    ref = _ref_swiglu(x.cpu(), clamp_value, topk_weight.cpu() if use_weight else None)
    split_d = d // 2
    # MX scales are 2-aligned and carry a trailing factor-of-2 dimension.
    assert scale.shape == (num_tokens, (split_d + 31) // 32 // 2, 2)
    assert scale.dtype == torch.float8_e8m0fnu

    flat_scale = (
        scale.cpu().view(torch.uint8).reshape(num_tokens, -1)[:, : (split_d + 31) // 32]
    )

    # 1. The stored e8m0 byte is fully determined by the block's max bf16 exponent.
    ref_bytes = _ref_mx_scale_bytes(ref, dst_type, x.dtype).to(torch.uint8)

    # A loose clamp asserts nothing: silu(gate)*up stays inside the exponent bucket a randn block
    # already occupies, so clamp_value >= 3.0 moves no byte at all and the comparison below would
    # pass against a kernel that drops clamp_value entirely. Reject such test data here, where the
    # failure names the cause, rather than accepting a test with no power.
    if clamp_value != 0.0:
        unclamped = _ref_mx_scale_bytes(
            _ref_swiglu(x.cpu(), 0.0, topk_weight.cpu() if use_weight else None),
            dst_type,
            x.dtype,
        ).to(torch.uint8)
        moved = (ref_bytes != unclamped).float().mean().item()
        assert moved >= 0.10, (
            f"clamp_value={clamp_value} moves only {moved:.1%} of the scale bytes at this seed and"
            f" shape, so the clamp is untested -- tighten clamp_value or pick another seed"
        )

    torch.testing.assert_close(flat_scale, ref_bytes, rtol=0, atol=0)

    # 2. Each stored scale is a power of two and dequantizing recovers the activation. The scale
    #    targets 2^(E - 8) for e4m3 (2^(E - 15) for e5m2), i.e. fp8_max / 1.75 rather than fp8_max,
    #    so the top 12.5% of each octave saturates at fp8_max -- hence the loose tolerance. Both y
    #    and the scale come from the input-dtype-rounded activation, so compare against that.
    deq, divisor = _dequantize(y.cpu(), flat_scale, 32, split_d)
    rounded = ref.to(x.dtype).float()
    torch.testing.assert_close(deq, rounded, rtol=0.25, atol=0.02)
    assert torch.all(divisor > 0)

    # 3. round_scale is inert for MX: the scale is exponent-derived either way.
    y2, scale2, _ = _call(
        x,
        topk_weight=topk_weight,
        dst_type=dst_type,
        quant_mode=MX_QUANT,
        round_scale=not round_scale,
        clamp_value=clamp_value,
    )
    torch.npu.synchronize()
    torch.testing.assert_close(
        scale2.cpu().view(torch.uint8), scale.cpu().view(torch.uint8), rtol=0, atol=0
    )
    torch.testing.assert_close(
        y2.cpu().view(torch.uint8), y.cpu().view(torch.uint8), rtol=0, atol=0
    )


def _check_fp8_mode(num_tokens, d, dst_type, seed, round_scale, ue8m0_scale):
    x = _make_inputs(num_tokens, d, seed=seed)
    y, scale, _ = _call(
        x,
        dst_type=dst_type,
        quant_mode=FP8_QUANT,
        round_scale=round_scale,
        ue8m0_scale=ue8m0_scale,
    )
    torch.npu.synchronize()

    ref = _ref_swiglu(x.cpu())
    split_d = d // 2
    assert y.shape == (num_tokens, split_d)
    assert scale.shape == (num_tokens, (split_d + 127) // 128)
    assert scale.dtype == (torch.float8_e8m0fnu if ue8m0_scale else torch.float32)

    ref_scales = _ref_fp8_scales(ref, dst_type, round_scale)
    if ue8m0_scale:
        # The e8m0 byte is the power-of-two exponent plus the bias.
        assert (
            round_scale
        ), "ue8m0_scale requires round_scale for a coherent scale buffer"
        bits = ref_scales.view(torch.int32)
        ref_bytes = ((((bits >> 23) & 0xFF) - 127) + E8M0_BIAS).to(torch.uint8)
        torch.testing.assert_close(
            scale.cpu().view(torch.uint8), ref_bytes, rtol=0, atol=0
        )
        stored = scale.cpu().view(torch.uint8)
    else:
        torch.testing.assert_close(scale.cpu(), ref_scales, rtol=1e-6, atol=0)
        stored = scale.cpu()

    deq, _ = _dequantize(y.cpu(), stored, 128, split_d)
    torch.testing.assert_close(deq, ref, rtol=0.15, atol=0.02)


class TestSwigluGroupQuant(unittest.TestCase):

    # ---- the fused activation, read back through y_origin ----

    def test_fp8_output_origin_e4m3(self):
        _check_swiglu_activation(512, torch.float8_e4m3fn, 0.0, 0, False)

    def test_fp8_output_origin_e5m2(self):
        _check_swiglu_activation(512, torch.float8_e5m2, 0.0, 1, False)

    def test_fp8_output_origin_fp16_input(self):
        # fp16 input selects the other DTYPE_X instantiation.
        num_tokens, d = 16, 512
        x = _make_inputs(num_tokens, d, dtype=torch.float16, seed=2)
        y, _, y_origin = _call(
            x, dst_type=torch.float8_e4m3fn, quant_mode=FP8_QUANT, output_origin=True
        )
        torch.npu.synchronize()
        ref = _ref_swiglu(x.cpu())
        torch.testing.assert_close(y_origin.cpu().float(), ref, rtol=0.01, atol=0.01)

    def test_fp8_output_origin_with_topk_weight(self):
        _check_swiglu_activation(512, torch.float8_e4m3fn, 0.0, 3, True)

    def test_fp8_output_origin_with_clamp(self):
        # The clamp is asymmetric, so this also pins the gate/up bound difference.
        _check_swiglu_activation(512, torch.float8_e4m3fn, 3.0, 4, False)

    def test_fp8_output_origin_clamp_and_weight(self):
        _check_swiglu_activation(512, torch.float8_e5m2, 1.5, 5, True)

    def test_output_origin_false_leaves_y_origin_unwritten(self):
        # With output_origin unset the kernel takes the other tiling key and skips the y_origin
        # stores, so the output must be unaffected by the extra allocation.
        num_tokens, d = 16, 512
        x = _make_inputs(num_tokens, d, seed=6)
        y_a, scale_a, _ = _call(
            x, dst_type=torch.float8_e4m3fn, quant_mode=FP8_QUANT, output_origin=False
        )
        y_b, scale_b, _ = _call(
            x, dst_type=torch.float8_e4m3fn, quant_mode=FP8_QUANT, output_origin=True
        )
        torch.npu.synchronize()
        torch.testing.assert_close(
            y_a.cpu().view(torch.uint8), y_b.cpu().view(torch.uint8), rtol=0, atol=0
        )
        torch.testing.assert_close(scale_a.cpu(), scale_b.cpu(), rtol=0, atol=0)

    # ---- mode 1: per-128 fp32 group scales ----

    def test_group_quant_e4m3(self):
        _check_group_mode(32, 512, torch.float8_e4m3fn, 7, 128, False, 0.0)

    def test_group_quant_e5m2(self):
        _check_group_mode(32, 512, torch.float8_e5m2, 8, 128, False, 0.0)

    def test_group_quant_with_topk_weight(self):
        _check_group_mode(32, 512, torch.float8_e4m3fn, 9, 128, True, 0.0)

    def test_group_quant_with_clamp(self):
        _check_group_mode(32, 512, torch.float8_e4m3fn, 10, 128, False, 2.0)

    def test_group_quant_fp16_input_multi_core(self):
        # A large batch exercises the multi-core split and the rowFactor UB loop.
        num_tokens, d = 2048, 768
        x = _make_inputs(num_tokens, d, dtype=torch.float16, seed=11)
        y, scale, _ = _call(x, dst_type=torch.float8_e4m3fn, quant_mode=GROUP_QUANT)
        torch.npu.synchronize()
        ref = _ref_swiglu(x.cpu())
        torch.testing.assert_close(
            scale.cpu(),
            _ref_group_scales(ref, torch.float8_e4m3fn),
            rtol=GROUP_SCALE_RTOL,
            atol=GROUP_SCALE_ATOL,
        )
        deq, _ = _dequantize(y.cpu(), scale.cpu(), 128, d // 2)
        torch.testing.assert_close(deq, ref, rtol=0.15, atol=0.02)

    def test_group_quant_group_size_is_inert(self):
        # group_size is accepted for signature compatibility but never reaches the tiling math.
        num_tokens, d = 16, 512
        x = _make_inputs(num_tokens, d, seed=12)
        results = [
            _call(
                x, dst_type=torch.float8_e4m3fn, quant_mode=GROUP_QUANT, group_size=gs
            )
            for gs in (1, 64, 128, 256)
        ]
        torch.npu.synchronize()
        for y, scale, _ in results[1:]:
            torch.testing.assert_close(
                y.cpu().view(torch.uint8),
                results[0][0].cpu().view(torch.uint8),
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(scale.cpu(), results[0][1].cpu(), rtol=0, atol=0)

    def test_group_quant_zero_input_gives_zero_output(self):
        # An all-zero block has amax 0, so both the scale and y must be exactly 0.
        num_tokens, d = 16, 512
        x = torch.zeros(num_tokens, d, dtype=torch.bfloat16).npu()
        y, scale, _ = _call(x, dst_type=torch.float8_e4m3fn, quant_mode=GROUP_QUANT)
        torch.npu.synchronize()
        torch.testing.assert_close(
            scale.cpu(), torch.zeros_like(scale.cpu()), rtol=0, atol=0
        )
        torch.testing.assert_close(
            y.cpu().view(torch.uint8),
            torch.zeros((num_tokens, d // 2), dtype=torch.uint8),
            rtol=0,
            atol=0,
        )

    # ---- mode 2: per-32 e8m0 mx scales ----

    def test_mx_quant_e4m3(self):
        _check_mx_mode(32, 512, torch.float8_e4m3fn, 13, False, False)

    def test_mx_quant_e5m2(self):
        # e5m2 uses the other exponent lower bound (0x0780 vs 0x0400).
        _check_mx_mode(32, 512, torch.float8_e5m2, 14, False, False)

    def test_mx_quant_with_topk_weight(self):
        _check_mx_mode(32, 512, torch.float8_e4m3fn, 15, True, False)

    def test_mx_quant_e5m2_with_round_scale(self):
        _check_mx_mode(32, 512, torch.float8_e5m2, 16, False, True)

    def test_mx_quant_with_clamp(self):
        # The clamp is not a side detail here: it changes the activation, hence the bf16 exponent
        # field the scale is read from, hence the stored e8m0 byte. This is also the mode
        # production actually calls, with swiglu_limit as the clamp.
        _check_mx_mode(32, 512, torch.float8_e4m3fn, 33, False, False, 1.0)

    def test_mx_quant_e5m2_with_clamp(self):
        # e5m2's exponent lower bound (0x0780 vs 0x0400) lets a block fall less far, so this needs
        # a looser clamp than the e4m3 case to move a comparable share of the bytes.
        _check_mx_mode(32, 512, torch.float8_e5m2, 34, False, False, 1.5)

    def test_mx_quant_clamp_and_weight(self):
        # The weight rescales the activation before the exponent field is read, so it and the
        # clamp both land on the same byte.
        _check_mx_mode(32, 512, torch.float8_e4m3fn, 35, True, False, 1.0)

    def test_mx_quant_large_batch_multi_core(self):
        # The shape class this op runs on in production: a full token batch, which reaches the
        # multi-core split and the rowFactor UB loop rather than a single row block.
        num_tokens, d = 2048, 768
        x = _make_inputs(num_tokens, d, seed=36)
        y, scale, _ = _call(x, dst_type=torch.float8_e4m3fn, quant_mode=MX_QUANT)
        torch.npu.synchronize()

        split_d = d // 2
        nblocks = split_d // 32
        assert scale.shape == (num_tokens, nblocks // 2, 2)
        ref = _ref_swiglu(x.cpu())
        flat = scale.cpu().view(torch.uint8).reshape(num_tokens, -1)[:, :nblocks]
        torch.testing.assert_close(
            flat,
            _ref_mx_scale_bytes(ref, torch.float8_e4m3fn, x.dtype).to(torch.uint8),
            rtol=0,
            atol=0,
        )
        deq, _ = _dequantize(y.cpu(), flat, 32, split_d)
        torch.testing.assert_close(deq, ref.to(x.dtype).float(), rtol=0.25, atol=0.02)

    def test_mx_quant_zero_block_gives_zero_scale_byte(self):
        # A block whose max exponent is 0 (every element zero) must store byte 0 rather than the
        # format's clamped lower bound. Zero rows and live rows run in the same launch so both
        # branches of the scale computation are exercised together.
        num_tokens, d = 32, 512
        zero_rows = 8
        x = _make_inputs(num_tokens, d, seed=37)
        x[:zero_rows] = 0.0
        y, scale, _ = _call(x, dst_type=torch.float8_e4m3fn, quant_mode=MX_QUANT)
        torch.npu.synchronize()

        split_d = d // 2
        flat = scale.cpu().view(torch.uint8).reshape(num_tokens, -1)[:, : split_d // 32]
        ref_bytes = _ref_mx_scale_bytes(
            _ref_swiglu(x.cpu()), torch.float8_e4m3fn, x.dtype
        ).to(torch.uint8)
        # Guard the test's premise before trusting it: the input must actually exercise both paths.
        assert (
            ref_bytes[:zero_rows] == 0
        ).all(), "zero rows must give zero scale bytes"
        assert (
            ref_bytes[zero_rows:] != 0
        ).any(), "live rows must give non-zero scale bytes"
        torch.testing.assert_close(flat, ref_bytes, rtol=0, atol=0)
        # An all-zero block quantizes to an all-zero y.
        assert (y.cpu().view(torch.uint8)[:zero_rows] == 0).all()

    def test_mx_quant_fp16_input(self):
        num_tokens, d = 16, 512
        x = _make_inputs(num_tokens, d, dtype=torch.float16, seed=17)
        y, scale, _ = _call(x, dst_type=torch.float8_e4m3fn, quant_mode=MX_QUANT)
        torch.npu.synchronize()
        ref = _ref_swiglu(x.cpu())
        flat = scale.cpu().view(torch.uint8).reshape(num_tokens, -1)[:, : d // 2 // 32]
        torch.testing.assert_close(
            flat,
            _ref_mx_scale_bytes(ref, torch.float8_e4m3fn, torch.float16).to(
                torch.uint8
            ),
            rtol=0,
            atol=0,
        )

    def test_output_origin_true_is_inert_for_group_and_mx(self):
        # Only fp8 has tiling keys that store y_origin (31/32); group and mx take keys 1/2 and the
        # kernel never passes y_origin to their op classes. So output_origin must leave y and scale
        # bit-identical -- and y_origin comes back allocated but unwritten, which is why callers
        # must not read it in these two modes. The invariance is all this contract can assert.
        num_tokens, d = 16, 512
        for quant_mode in (GROUP_QUANT, MX_QUANT):
            x = _make_inputs(num_tokens, d, seed=38 + quant_mode)
            y_off, scale_off, _ = _call(
                x,
                dst_type=torch.float8_e4m3fn,
                quant_mode=quant_mode,
                output_origin=False,
            )
            y_on, scale_on, y_origin = _call(
                x,
                dst_type=torch.float8_e4m3fn,
                quant_mode=quant_mode,
                output_origin=True,
            )
            torch.npu.synchronize()
            assert y_origin.shape == (num_tokens, d // 2)
            assert y_origin.dtype == x.dtype
            # Compare as bytes: the scale dtype differs per mode (fp32 for group, e8m0 for mx).
            torch.testing.assert_close(
                y_on.cpu().view(torch.uint8),
                y_off.cpu().view(torch.uint8),
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                scale_on.cpu().view(torch.uint8),
                scale_off.cpu().view(torch.uint8),
                rtol=0,
                atol=0,
            )

    # ---- mode 3: per-128 fp8 scales, fp32 or e8m0 ----

    def test_fp8_quant_e4m3_no_round_scale(self):
        _check_fp8_mode(32, 512, torch.float8_e4m3fn, 18, False, False)

    def test_fp8_quant_e4m3_round_scale(self):
        _check_fp8_mode(32, 512, torch.float8_e4m3fn, 19, True, False)

    def test_fp8_quant_e5m2_round_scale(self):
        # dst_type does not enter mode 3's scale math: it is always scaled against 448.0.
        _check_fp8_mode(32, 512, torch.float8_e5m2, 20, True, False)

    def test_fp8_quant_ue8m0_scale(self):
        _check_fp8_mode(32, 512, torch.float8_e4m3fn, 21, True, True)

    # ---- optional group_index ----

    def test_group_index_selects_the_processed_rows(self):
        # With group_index the kernel derives the row count from the group list's total, so a list
        # summing to the full batch must reproduce the no-group-index result.
        num_tokens, d = 32, 512
        x = _make_inputs(num_tokens, d, seed=22)
        y_plain, scale_plain, _ = _call(
            x, dst_type=torch.float8_e4m3fn, quant_mode=GROUP_QUANT
        )
        counts = torch.tensor([8, 8, 16], dtype=torch.int64)
        torch.testing.assert_close(
            counts.sum(), torch.tensor(num_tokens, dtype=torch.int64)
        )
        y_gi, scale_gi, _ = _call(
            x,
            group_index=counts.cumsum(0).npu(),
            dst_type=torch.float8_e4m3fn,
            quant_mode=GROUP_QUANT,
        )
        torch.npu.synchronize()
        torch.testing.assert_close(
            y_gi.cpu().view(torch.uint8),
            y_plain.cpu().view(torch.uint8),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(scale_gi.cpu(), scale_plain.cpu(), rtol=0, atol=0)

    def test_group_index_partial_batch(self):
        # A group list totalling fewer rows than the batch leaves the remaining rows untouched.
        num_tokens, d = 32, 512
        processed = 16
        x = _make_inputs(num_tokens, d, seed=23)
        y, scale, _ = _call(
            x,
            group_index=torch.tensor([processed], dtype=torch.int64).npu(),
            dst_type=torch.float8_e4m3fn,
            quant_mode=GROUP_QUANT,
        )
        torch.npu.synchronize()
        ref = _ref_swiglu(x.cpu())
        torch.testing.assert_close(
            scale.cpu()[:processed],
            _ref_group_scales(ref, torch.float8_e4m3fn)[:processed],
            rtol=GROUP_SCALE_RTOL,
            atol=GROUP_SCALE_ATOL,
        )

    # ---- rejected arguments and shapes ----

    def test_rejects_unsupported_quant_mode(self):
        x = _make_inputs(4, 512, seed=24)
        with self.assertRaisesRegex(RuntimeError, "Unsupported quant mode"):
            _call(x, quant_mode=7)

    def test_rejects_unsupported_group_list_type(self):
        x = _make_inputs(4, 512, seed=25)
        with self.assertRaisesRegex(RuntimeError, "tiling failed"):
            _call(x, group_list_type=1)

    def test_rejects_bad_last_dim_for_group_quant(self):
        # group/fp8 modes require the last dim to be divisible by 256.
        x = _make_inputs(4, 384, seed=26)
        with self.assertRaisesRegex(RuntimeError, "divisible by 256"):
            _call(x, quant_mode=GROUP_QUANT)

    def test_rejects_bad_last_dim_for_mx_quant(self):
        # mx requires divisibility by 128 instead.
        x = _make_inputs(4, 192, seed=27)
        with self.assertRaisesRegex(RuntimeError, "divisible by 128"):
            _call(x, quant_mode=MX_QUANT)

    def test_rejects_non_fp8_dst_type(self):
        x = _make_inputs(4, 512, seed=28)
        with self.assertRaisesRegex(RuntimeError, "dst_type must be"):
            _call(x, dst_type=torch.float16, quant_mode=GROUP_QUANT)

    def test_rejects_fp32_input(self):
        x = torch.randn(4, 512, dtype=torch.float32).npu()
        with self.assertRaisesRegex(RuntimeError, "FLOAT16 or BFLOAT16"):
            _call(x, quant_mode=GROUP_QUANT)

    def test_default_dst_type_is_e4m3fn(self):
        x = _make_inputs(4, 512, seed=29)
        y, _, _ = _call(x, quant_mode=GROUP_QUANT, dst_type=None)
        torch.npu.synchronize()
        assert y.dtype == torch.float8_e4m3fn

    def test_empty_batch(self):
        # No rows to process: the outputs come back empty rather than launching a zero-block kernel.
        x = torch.randn(0, 512, dtype=torch.bfloat16).npu()
        y, scale, y_origin = _call(
            x, dst_type=torch.float8_e4m3fn, quant_mode=GROUP_QUANT
        )
        torch.npu.synchronize()
        assert y.shape == (0, 256)
        assert scale.shape == (0, 2)
        assert y_origin.shape == (0, 256)


if __name__ == "__main__":
    unittest.main()
