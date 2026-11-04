# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import math
import struct

import torch

# The reciprocal bit patterns the kernel multiplies by (not a correctly-rounded 1.0 / max).
INV_FP8_E4M3_BITS = 0x3B124925
INV_FP8_E5M2_BITS = 0x37924925

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
    padded = torch.zeros((n, nblocks * block), dtype=t.dtype, device=t.device)
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


def swiglu_group_quant(
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
):
    if x.ndim < 1 or not x.is_contiguous():
        raise RuntimeError("x must be contiguous with at least one dimension")
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError("x should be FLOAT16 or BFLOAT16")
    if quant_mode not in (1, 2, 3):
        raise RuntimeError("Unsupported quant mode")
    dst_type = torch.float8_e4m3fn if dst_type is None else dst_type
    if dst_type not in (torch.float8_e4m3fn, torch.float8_e5m2):
        raise RuntimeError("dst_type must be Float8_e4m3fn or Float8_e5m2")
    alignment = 128 if quant_mode == 2 else 256
    if x.size(-1) % alignment:
        raise RuntimeError(f"last dimension must be divisible by {alignment}")
    if math.prod(x.shape[:-1]) != 0 and group_list_type != 0:
        raise RuntimeError("swiglu_group_quant: tiling failed for group_list_type")
    rows = math.prod(x.shape[:-1])
    width = x.size(-1) // 2
    shape = (*x.shape[:-1], width)
    block = 32 if quant_mode == 2 else 128
    blocks = (width + block - 1) // block
    scale_shape = (
        (*shape[:-1], blocks // 2, 2) if quant_mode == 2 else (*shape[:-1], blocks)
    )
    scale_dtype = (
        torch.float8_e8m0fnu
        if quant_mode == 2 or (quant_mode == 3 and ue8m0_scale)
        else torch.float32
    )
    y = x.new_empty(shape, dtype=dst_type)
    origin = x.new_empty(shape)
    scale = x.new_empty(scale_shape, dtype=scale_dtype)
    count = rows
    if group_index is not None and group_index.numel():
        count = min(rows, int(group_index.sum().item()))
    if count <= 0 or width == 0:
        return y, scale, origin
    weight = (
        topk_weight.flatten()[:count]
        if topk_weight is not None and topk_weight.numel()
        else None
    )
    act = _ref_swiglu(x.reshape(rows, width * 2)[:count], clamp_value, weight)
    if quant_mode == 2:
        exponent = _ref_mx_scale_bytes(act, dst_type, x.dtype)
        divisor = torch.exp2(exponent.float() - E8M0_BIAS).masked_fill(
            exponent == 255, float("nan")
        )
        stored = exponent.to(torch.uint8).view(scale_dtype)
        act = act.to(x.dtype).float()
    elif quant_mode == 3:
        divisor = _ref_fp8_scales(act, dst_type, round_scale)
        stored = divisor
        if ue8m0_scale and round_scale:
            stored = (
                ((divisor.view(torch.int32) >> 23) & 255)
                .to(torch.uint8)
                .view(scale_dtype)
            )
        if output_origin:
            origin.view(rows, width)[:count].copy_(act)
    else:
        divisor = _ref_group_scales(act, dst_type)
        stored = divisor
    if quant_mode == 3 and not round_scale:
        inv = 448.0 / _blocks(act, 128).abs().amax(dim=-1).clamp_min(1e-4)
        normalized = act * inv.repeat_interleave(block, dim=-1)
    else:
        expanded = divisor.repeat_interleave(block, dim=-1)
        normalized = torch.where(expanded == 0, act, act / expanded)
    limit = torch.finfo(dst_type).max
    y.view(rows, width)[:count].copy_(normalized.clamp(-limit, limit).to(dst_type))
    scale.view(rows, blocks)[:count].copy_(stored)
    return y, scale, origin
