# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""CANNBot-DSL implementation of ``indexer_prologue_k``.

The operator executes the following pipeline and updates ``k_cache`` in place::

    latent -- BF16 GEMM --> projected -- RMSNorm --> tail RoPE
           --> block-32 MXFP4 packing --> indexed uint8 k_cache

Projection and fused post-processing/cache scatter use two kernels on the
same NPU stream, so the projected intermediate does not require a host sync.

The compiled contract keeps the model shape (H, D, Dr) and the storage
layout static, while the token count ``T`` and the whole cache geometry
(block count, slot rows, axis-0/1 strides) are dynamic ``Dim`` axes read
from the tensors at runtime.  One compiled binary therefore serves every
``T``/cache size for a given model shape and storage configuration, and
the per-launch core counts are runtime scalar arguments.
"""

import inspect
import json
import os

import torch

from cannbotdsl import (
    Dim,
    MemLoc,
    Tensor,
    TensorSpec,
    compile as dsl_compile,
    const_expr,
    dtypes,
    get_platform_info,
    make_layout,
)
from cannbotdsl.aot import export
from cannbotdsl.buffer import Buffer
from cannbotdsl.ring_buffer import RingBuffer
from cannbotdsl.lang.host import host
from cannbotdsl.lang.jit import jit
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops import reg as rr
from cannbotdsl.ops import scalar
from cannbotdsl.ops.arch import get_block_idx, get_block_num
from cannbotdsl.ops.matmul import matmul
from cannbotdsl.ops.memcpy import mem_copy
from cannbotdsl.ops.sync import vec_sync_all
from cannbotdsl.tensor import ceil_div, tile_slice

__all__ = [
    "IndexerPrologueKPostprocess",
    "IndexerPrologueKScatter",
    "indexer_prologue_k",
]


VL = 64
FP4_GROUP_SIZE = 32
FP4_MAX = 6.0
FP32_EXP_SHIFT = 23
FP32_MANTISSA_MASK = 0x7FFFFF
FP32_EXP_MASK = 0xFF
BASE_M = 16
BASE_N = 64
BASE_K = 128
NZ_FRACTAL_ALIGN = 16  # axis alignment required by raw FRACTAL_NZ copies
E2M1_VALUES_PER_BYTE = 2  # two 4-bit E2M1 codes packed per byte
ROPE_PAIR_SIZE = 2  # adjacent even/odd lanes form one rope pair
DMA_ALIGN_BYTES = 32  # UB<->GM fast-path DMA alignment
CACHE_ABI_DIM2 = 1  # the 4-D cache ABI keeps axis 2 fixed

# Keep the sample usable with both layout-based DSL builds and the 0.8.0
# CI wheel, whose constructors still expose the shape-based call convention.
_RING_BUFFER_USES_LAYOUT = "layout" in inspect.signature(RingBuffer).parameters
_BUFFER_POSITIONAL_PARAMETERS = tuple(
    name
    for name, parameter in inspect.signature(Buffer).parameters.items()
    if parameter.kind
    in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
)
_BUFFER_USES_LAYOUT = _BUFFER_POSITIONAL_PARAMETERS[1:3] == ("dtype", "layout")


def _make_ring_buffer(location, dtype, shape, *, depth):
    if _RING_BUFFER_USES_LAYOUT:
        return RingBuffer(location, dtype, make_layout(shape), depth=depth)
    return RingBuffer(location, shape, dtype, depth=depth)


def _make_buffer(location, dtype, shape):
    if _BUFFER_USES_LAYOUT:
        return Buffer(location, dtype, make_layout(shape))
    return Buffer(location, shape, dtype)


# Module-level AOT caches: one compiled ProviderCallable per key.
_FUSED_AOT_CALLABLE_CACHE = {}
_AOT_CALLABLE_CACHE = {}


def _ceil_div(value, divisor):
    return (value + divisor - 1) // divisor


def get_effective_core_counts():
    """Return the (AIC, AIV) core counts available to the launch stream.

    The NPU device properties provide the baseline; cannbotdsl's platform
    query overlays the effective counts, which also account for the current
    stream's core quota.  The device query is best-effort: on hosts without
    an NPU (AOT compilation, CPU CI) it raises and the static platform
    table still answers, keeping host-side default grids derivable.  A
    missing or invalid query is an error - the launch grid must never fall
    back to a hardcoded core count.
    """
    cube = 0
    vector = 0
    try:
        props = torch.npu.get_device_properties(torch.npu.current_device())
        cube = int(getattr(props, "cube_core_num", 0))
        vector = int(getattr(props, "vector_core_num", 0)) or 2 * cube
    except (
        AttributeError,
        AssertionError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
        pass
    try:
        info = get_platform_info()
        cube = int(info.cube_core_num) or cube
        vector = int(info.vector_core_num) or vector
    except (OSError, RuntimeError, TypeError, ValueError):
        pass
    if cube <= 0 or vector < cube:
        raise RuntimeError(
            f"invalid effective NPU core counts: AIC={cube}, AIV={vector}"
        )
    return cube, vector


class _ProjectionKernel:
    """BF16 ND ``[M,K]`` x NZ ``[N,K].T`` with FP32 accumulation.

    ``M`` (the token count) is read from ``gm_latent.shape[0]`` at runtime,
    so one compiled binary projects every ``T``; ``N``/``K`` stay static per
    model shape because they select the NZ tile bases below.
    """

    def __init__(self, m: int, n: int, k: int):
        self.m = int(m)
        self.n = int(n)
        self.k = int(k)
        # Raw FRACTAL_NZ GM->L1 copies must end on physical 16x16 fractal
        # boundaries.  Keep the wide fast path for aligned model shapes and
        # fall back to one fractal along a non-aligned axis.
        self.base_n = BASE_N if self.n % BASE_N == 0 else NZ_FRACTAL_ALIGN
        self.base_k = BASE_K if self.k % BASE_K == 0 else NZ_FRACTAL_ALIGN
        self.m_tiles = _ceil_div(self.m, BASE_M)
        self.n_tiles = _ceil_div(self.n, self.base_n)
        self.k_tiles = _ceil_div(self.k, self.base_k)
        self.total_tiles = self.m_tiles * self.n_tiles
        cube_cores, _ = get_effective_core_counts()
        self.block_num = min(cube_cores, self.total_tiles)

    @kernel
    def kernel(self, gm_latent: Tensor, gm_weight: Tensor, gm_projected: Tensor):
        base_n = self.base_n
        base_k = self.base_k
        l1_a = _make_ring_buffer(MemLoc.L1, dtypes.bfloat16, (BASE_M, base_k), depth=2)
        l1_b = _make_ring_buffer(MemLoc.L1, dtypes.bfloat16, (base_n, base_k), depth=2)
        l0a = _make_ring_buffer(MemLoc.L0A, dtypes.bfloat16, (BASE_M, base_k), depth=2)
        l0b = _make_ring_buffer(MemLoc.L0B, dtypes.bfloat16, (base_n, base_k), depth=2)
        l0c = _make_ring_buffer(MemLoc.L0C, dtypes.float32, (BASE_M, base_n), depth=1)

        block_idx = get_block_idx()
        block_num = get_block_num()
        # The M axis is dynamic: tile counts come from the runtime shape.
        m_tiles = ceil_div(gm_latent.shape[0], BASE_M)
        total_tiles = m_tiles * self.n_tiles
        for tile_idx in range(block_idx, total_tiles, block_num):
            m_idx = tile_idx // self.n_tiles
            n_idx = tile_idx % self.n_tiles
            for k_idx in range(self.k_tiles):
                mem_copy(
                    l1_a.next_write(),
                    tile_slice(gm_latent, (BASE_M, base_k), (m_idx, k_idx)),
                )
                mem_copy(
                    l1_b.next_write(),
                    tile_slice(gm_weight, (base_n, base_k), (n_idx, k_idx)),
                )
                mem_copy(l0a.next_write(), l1_a.next_read())
                mem_copy(l0b.next_write(), l1_b.next_read())
                matmul(
                    l0c.next_write(),
                    l0a.next_read(),
                    l0b.next_read(),
                    init=(k_idx == 0),
                )

            mem_copy(
                tile_slice(gm_projected, (BASE_M, base_n), (m_idx, n_idx)),
                l0c.next_read(),
            )

    @host
    def run(
        self,
        gm_latent: Tensor,
        gm_weight: Tensor,
        gm_projected: Tensor,
        block_num,
    ):
        self.kernel[block_num](gm_latent, gm_weight, gm_projected)


@kernel
class _FusedPostScatterKernel:
    """Destination-owner RMSNorm + RoPE + MXFP4 direct cache update."""

    def __init__(
        self,
        *,
        total_rows: int,
        index_head_dim: int,
        rope_head_dim: int,
        storage_mode: int,
        combined_block_size: int,
        raw_block_size: int,
        write_scale_cache: bool,
    ):
        self.total_rows = int(total_rows)
        self.index_head_dim = int(index_head_dim)
        self.rope_head_dim = int(rope_head_dim)
        self.storage_mode = int(storage_mode)
        self.combined_block_size = int(combined_block_size)
        # total_rows and raw_block_size are host-side defaults only: the
        # kernel reads the token count from gm_projected and the block
        # geometry from gm_cache at runtime.
        self.raw_block_size = int(raw_block_size)
        self.write_scale_cache = bool(write_scale_cache)
        self.padded_dim = _ceil_div(self.index_head_dim, VL) * VL

    @jit
    def _normalize(self, projected, gamma, normalized, epsilon: dtypes.float32):
        dim = self.index_head_dim
        with vf(mode="simd"):
            full_mask = rr.full_mask()
            lane0 = rr.update_mask(1, elem_bits=32)[0]
            square_sum = rr.vdups(0.0, dtypes.float32, mask=lane0)
            for segment in range(_ceil_div(dim, VL)):
                segment_offset = segment * VL
                valid = min(VL, dim - segment_offset)
                segment_mask = rr.update_mask(valid, elem_bits=32)[0]
                packed = rr.vload_unpack(
                    projected, segment_offset, unpack_mode=rr.UnpackMode.B16_TO_B32
                )
                elem = rr.vcast(packed, dtypes.float32, mask=segment_mask)
                partial = rr.vreduce_sum(
                    rr.vmul(elem, elem, mask=segment_mask), mask=segment_mask
                )
                square_sum = rr.vadd(square_sum, partial, mask=lane0)

            mean = rr.vmuls(square_sum, 1.0 / dim, mask=lane0)
            root = rr.vsqrt(rr.vadds(mean, epsilon, mask=lane0), mask=lane0)
            inverse_rms = rr.vdiv(
                rr.vdups(1.0, dtypes.float32, mask=lane0), root, mask=lane0
            )
            inverse_rms = rr.vdup(inverse_rms, mask=full_mask)

            for segment in range(_ceil_div(dim, VL)):
                segment_offset = segment * VL
                valid = min(VL, dim - segment_offset)
                segment_mask = rr.update_mask(valid, elem_bits=32)[0]
                packed = rr.vload_unpack(
                    projected, segment_offset, unpack_mode=rr.UnpackMode.B16_TO_B32
                )
                elem = rr.vcast(packed, dtypes.float32, mask=segment_mask)
                gamma_value = rr.vload(gamma, segment_offset)
                elem = rr.vmul(elem, inverse_rms, mask=segment_mask)
                elem = rr.vmul(elem, gamma_value, mask=segment_mask)
                elem = rr.vcast(elem, dtypes.bfloat16, mask=segment_mask)
                rr.vstore_pack(
                    normalized,
                    segment_offset,
                    elem,
                    segment_mask,
                    pack_mode=rr.PackMode.B32_TO_B16,
                )

            rr.vmem_bar("vst_vld")

    @jit
    def _load_bf16_rope_work(self, source, work):
        with vf(mode="simd"):
            dim = self.index_head_dim
            first_segment = (dim - self.rope_head_dim) // VL
            for segment in range(first_segment, _ceil_div(dim, VL)):
                segment_offset = segment * VL
                valid = min(VL, dim - segment_offset)
                segment_mask = rr.update_mask(valid, elem_bits=32)[0]
                packed = rr.vload_unpack(
                    source, segment_offset, unpack_mode=rr.UnpackMode.B16_TO_B32
                )
                elem = rr.vcast(packed, dtypes.float32, mask=segment_mask)
                rr.vstore(work, segment_offset, elem, segment_mask)

            rr.vmem_bar("vst_vld")

    @jit
    def _apply_rope(self, work, rope_sin, rope_cos, rotated):
        """Rotate adjacent pairs in the final ``rope_head_dim`` columns."""
        dim = self.index_head_dim
        rd = self.rope_head_dim
        rope_start = dim - rd

        with vf(mode="simd"):
            full_mask = rr.full_mask()
            for chunk in range(_ceil_div(rd, VL)):
                chunk_offset = chunk * VL
                valid = min(VL, rd - chunk_offset)
                pair_mask = rr.update_mask(valid // ROPE_PAIR_SIZE, elem_bits=32)[0]
                rope_mask = rr.update_mask(valid, elem_bits=32)[0]
                source_offset = rope_start + chunk_offset
                if const_expr(rope_start % 8 == 0):
                    x_even, x_odd = rr.vload_deinterleave(
                        work, source_offset, width="b32"
                    )
                else:
                    rr.vload_unalign_init(work, source_offset)
                    packed_rope = rr.vload_unalign(work, source_offset)
                    x_even, x_odd = rr.vdeinterleave(packed_rope, packed_rope)

                cos_even, cos_odd = rr.vload_deinterleave(
                    rope_cos, chunk_offset, width="b32"
                )
                sin_even, sin_odd = rr.vload_deinterleave(
                    rope_sin, chunk_offset, width="b32"
                )
                out_even = rr.vsub(
                    rr.vmul(x_even, cos_even, mask=pair_mask),
                    rr.vmul(x_odd, sin_even, mask=pair_mask),
                    mask=pair_mask,
                )
                out_odd = rr.vadd(
                    rr.vmul(x_odd, cos_odd, mask=pair_mask),
                    rr.vmul(x_even, sin_odd, mask=pair_mask),
                    mask=pair_mask,
                )
                interleaved, _ = rr.vinterleave(out_even, out_odd)
                rr.vstore(rotated, VL + chunk_offset, interleaved, rope_mask)
            rr.vmem_bar("vst_vld")

            # CANN 9.2 cannot lower the DSL unaligned-store helper.  Rebase
            # the compact rotated tail through an unaligned load, merge it
            # into aligned 64-lane work segments, and only issue aligned
            # stores.  The VL prefix guarantees every rebased load is in-bounds.
            first_segment = rope_start // VL
            for segment in range(first_segment, _ceil_div(dim, VL)):
                segment_start = segment * VL
                local_start = max(0, rope_start - segment_start)
                local_end = min(VL, dim - segment_start)
                end_mask = rr.update_mask(local_end, elem_bits=32)[0]
                prefix_mask = rr.update_mask(local_start, elem_bits=32)[0]
                range_mask = rr.mask_xor(end_mask, prefix_mask, exec_mask=full_mask)
                rotated_offset = VL + segment_start - rope_start
                rr.vload_unalign_init(rotated, rotated_offset)
                shifted = rr.vload_unalign(rotated, rotated_offset)
                old = rr.vload(work, segment_start)
                merged = rr.vmerge(range_mask, shifted, old)
                rr.vstore(work, segment_start, merged, end_mask)

            rr.vmem_bar("vst_vld")

    @jit
    def _pack_mxfp4(self, source, packed_output, code_scratch):
        """Pack signed E2M1 pairs and append one UE8M0 scale per 32 values."""
        dim = self.index_head_dim
        num_groups = dim // FP4_GROUP_SIZE
        value_bytes = dim // E2M1_VALUES_PER_BYTE
        with vf(mode="simd"):
            mask = rr.update_mask(FP4_GROUP_SIZE, elem_bits=32)[0]
            packed_mask = rr.update_mask(
                FP4_GROUP_SIZE // E2M1_VALUES_PER_BYTE, elem_bits=32
            )[0]
            byte_block_mask = rr.update_mask(FP4_GROUP_SIZE, elem_bits=32)[0]
            lane0 = rr.update_mask(1, elem_bits=32)[0]
            one_u32 = rr.vdups(1, dtypes.uint32, mask=mask)
            zero_u32 = rr.vdups(0, dtypes.uint32, mask=mask)
            sign_u32 = rr.vdups(8, dtypes.uint32, mask=mask)
            zero_f32 = rr.vdups(0.0, dtypes.float32, mask=mask)
            mantissa_mask = rr.vdups(FP32_MANTISSA_MASK, dtypes.uint32, mask=mask)
            exponent_mask = rr.vdups(FP32_EXP_MASK, dtypes.uint32, mask=mask)

            for group in range(num_groups):
                segment_offset = group * FP4_GROUP_SIZE
                unpacked = rr.vload_unpack(
                    source, segment_offset, unpack_mode=rr.UnpackMode.B16_TO_B32
                )
                elem = rr.vcast(unpacked, dtypes.float32, mask=mask)
                abs_value = rr.vabs(elem, mask=mask)
                amax = rr.vreduce_max(abs_value, mask=mask)
                safe_amax = rr.vmaxs(amax, FP4_MAX * (2.0**-126), mask=mask)
                raw_scale = rr.vmuls(safe_amax, 1.0 / FP4_MAX, mask=mask)

                scale_bits = rr.vreinterpret(raw_scale, dtypes.uint32)
                exponent = rr.vbitwise_and(
                    rr.vshr(scale_bits, FP32_EXP_SHIFT, mask=mask),
                    exponent_mask,
                    mask=mask,
                )
                mantissa = rr.vbitwise_and(scale_bits, mantissa_mask, mask=mask)
                has_mantissa = rr.vne(mantissa, zero_u32, mask=mask)
                increment = rr.vselect(one_u32, zero_u32, cond_mask=has_mantissa)
                exponent = rr.vadd(exponent, increment, mask=mask)
                pow2_bits = rr.vshl(exponent, FP32_EXP_SHIFT, mask=mask)
                scale = rr.vreinterpret(pow2_bits, dtypes.float32)
                scale = rr.vdup(scale, mask=mask)

                scaled_abs = rr.vdiv(abs_value, scale, mask=mask)
                code = rr.vdups(7, dtypes.uint32, mask=mask)
                for threshold, level_code in (
                    (5.0, 6),
                    (3.5, 5),
                    (2.5, 4),
                    (1.75, 3),
                    (1.25, 2),
                    (0.75, 1),
                    (0.25, 0),
                ):
                    lower = rr.vdups(level_code, dtypes.uint32, mask=mask)
                    choose_lower = rr.vles(scaled_abs, threshold, mask=mask)
                    code = rr.vselect(lower, code, cond_mask=choose_lower)

                is_negative = rr.vlt(elem, zero_f32, mask=mask)
                sign = rr.vselect(sign_u32, zero_u32, cond_mask=is_negative)
                code = rr.vadd(code, sign, mask=mask)
                even_code, odd_code = rr.vdeinterleave(code, code)
                high_nibble = rr.vshl(odd_code, 4, mask=packed_mask)
                packed_code = rr.vbitwise_or(even_code, high_nibble, mask=packed_mask)
                rr.vstore(
                    code_scratch,
                    group * VL,
                    packed_code,
                    packed_mask,
                )

                exponent_u8 = rr.vcast(
                    exponent,
                    dtypes.uint8,
                    mask=lane0,
                    reg_layout=rr.RegLayout.ZERO,
                )
                rr.vstore_first(
                    packed_output,
                    value_bytes + group,
                    exponent_u8,
                )

            # Two 32-elem groups produce 16 bytes each.  Join them into one
            # aligned 32-byte store to avoid non-aligned vector writes on 950.
            prefix = rr.update_mask(
                FP4_GROUP_SIZE // E2M1_VALUES_PER_BYTE, elem_bits=32
            )[0]
            tail = rr.mask_xor(byte_block_mask, prefix, exec_mask=rr.full_mask())
            rr.vmem_bar("vst_vld")

            for group_pair in range(num_groups // E2M1_VALUES_PER_BYTE):
                first_group = 2 * group_pair
                first = rr.vload(code_scratch, first_group * VL)
                second_shifted = rr.vload(
                    code_scratch,
                    (first_group + 1) * VL - FP4_GROUP_SIZE // E2M1_VALUES_PER_BYTE,
                )
                joined = rr.vmerge(tail, second_shifted, first)
                rr.vstore_pack(
                    packed_output,
                    group_pair * FP4_GROUP_SIZE,
                    joined,
                    byte_block_mask,
                    pack_mode=rr.PackMode.B32_TO_B8,
                )
            if const_expr(num_groups % E2M1_VALUES_PER_BYTE != 0):
                last_group = num_groups - 1
                last = rr.vload(code_scratch, last_group * VL)
                rr.vstore_pack(
                    packed_output,
                    (num_groups // E2M1_VALUES_PER_BYTE) * FP4_GROUP_SIZE,
                    last,
                    packed_mask,
                    pack_mode=rr.PackMode.B32_TO_B8,
                )

            rr.vmem_bar("vst_vld")

    @jit
    def _cast_output(self, work, output):
        with vf(mode="simd"):
            dim = self.index_head_dim
            first_segment = (dim - self.rope_head_dim) // VL
            for segment in range(first_segment, _ceil_div(dim, VL)):
                segment_offset = segment * VL
                valid = min(VL, dim - segment_offset)
                segment_mask = rr.update_mask(valid, elem_bits=32)[0]
                elem = rr.vload(work, segment_offset)
                packed = rr.vcast(elem, dtypes.bfloat16, mask=segment_mask)
                rr.vstore_pack(
                    output,
                    segment_offset,
                    packed,
                    segment_mask,
                    pack_mode=rr.PackMode.B32_TO_B16,
                )

    @jit
    def _store_bytes(
        self,
        packed_output,
        staging,
        gm_dest,
        dest_block,
        dest_slot,
        dest_col_tile,
        count,
        src_col,
        aligned,
    ):
        """Store one packed byte range via the DMA or the scalar fallback.

        dest_col_tile is a tile index along the cache width axis; src_col is
        the element offset inside packed_output.  Each next_write is colocated
        with its next_read so the staging FIFO ledger stays balanced.
        """
        if aligned:
            mem_copy(
                staging.next_write(),
                tile_slice(packed_output, (1, count), (0, src_col // count)),
            )
            mem_copy(
                tile_slice(
                    gm_dest,
                    (1, 1, 1, count),
                    (dest_block, dest_slot, 0, dest_col_tile),
                ).view(1, count),
                staging.next_read(),
            )
        else:
            for byte_idx in range(count):
                scalar.vec_store_bypass(
                    gm_dest.ptr(
                        (dest_block, dest_slot, 0, dest_col_tile * count + byte_idx)
                    ),
                    packed_output[0, src_col + byte_idx],
                )

    @jit
    def _store_packed_row(
        self,
        packed_output,
        packed_data,
        packed_scale,
        gm_cache,
        gm_scale_cache,
        cache_row,
    ):
        value_bytes = self.index_head_dim // E2M1_VALUES_PER_BYTE
        scale_bytes = self.index_head_dim // FP4_GROUP_SIZE
        # Partial-width UB vector copies are unsafe on 950 when D/32 is odd:
        # their 16-byte data tail (and odd-byte scale row) can raise a vector
        # core exception.  Unaligned shapes route through the scalar fallback
        # in _store_bytes; aligned/even-group shapes retain the DMA path.
        if const_expr(self.storage_mode == 0):
            # The cache slot axis is dynamic; read the block geometry from
            # the tensor instead of a compile-time constant.
            raw_block_size = gm_cache.shape[1]
            block_id = cache_row // raw_block_size
            slot = cache_row % raw_block_size
            aligned = value_bytes % DMA_ALIGN_BYTES == 0
            self._store_bytes(
                packed_output,
                packed_data,
                gm_cache,
                block_id,
                slot,
                0,
                value_bytes,
                0,
                aligned,
            )
            if const_expr(self.write_scale_cache):
                self._store_bytes(
                    packed_output,
                    packed_scale,
                    gm_scale_cache,
                    block_id,
                    slot,
                    0,
                    scale_bytes,
                    value_bytes,
                    aligned,
                )
        else:
            group_size = self.combined_block_size
            raw_block_size = gm_cache.shape[1] * group_size
            block_id = cache_row // raw_block_size
            slot = cache_row % raw_block_size
            cache_group = slot // group_size
            sub_slot = slot % group_size
            scale_offset = group_size * value_bytes + sub_slot * scale_bytes
            combined_width = group_size * (value_bytes + scale_bytes)
            self._store_bytes(
                packed_output,
                packed_data,
                gm_cache,
                block_id,
                cache_group,
                sub_slot,
                value_bytes,
                0,
                value_bytes % DMA_ALIGN_BYTES == 0
                and combined_width % DMA_ALIGN_BYTES == 0,
            )
            self._store_bytes(
                packed_output,
                packed_scale,
                gm_cache,
                block_id,
                cache_group,
                scale_offset // scale_bytes,
                scale_bytes,
                value_bytes,
                value_bytes % DMA_ALIGN_BYTES == 0,
            )

    @jit
    def _process_owned_row(
        self,
        gm_projected,
        gm_rope_sin,
        gm_rope_cos,
        gamma,
        gm_cache,
        gm_scale_cache,
        projected,
        output,
        packed_output,
        packed_data,
        packed_scale,
        code_scratch,
        work,
        rope_sin,
        rope_cos,
        rotated,
        row,
        cache_row,
        epsilon,
    ):
        dim = self.index_head_dim
        rd = self.rope_head_dim
        mem_copy(
            projected.next_write(),
            tile_slice(gm_projected, (1, dim), (row, 0)),
        )
        mem_copy(rope_sin, tile_slice(gm_rope_sin, (1, rd), (row, 0)))
        mem_copy(rope_cos, tile_slice(gm_rope_cos, (1, rd), (row, 0)))
        # Preserve both BF16 publication boundaries from the reference path.
        self._normalize(projected.next_read(), gamma, output, epsilon)
        self._load_bf16_rope_work(output, work)
        self._apply_rope(work, rope_sin, rope_cos, rotated)
        self._cast_output(work, output)
        self._pack_mxfp4(output, packed_output, code_scratch)
        value_bytes = self.index_head_dim // E2M1_VALUES_PER_BYTE
        scale_bytes = self.index_head_dim // FP4_GROUP_SIZE
        combined_width = self.combined_block_size * (value_bytes + scale_bytes)
        if const_expr(
            value_bytes % 32 != 0
            or (self.storage_mode == 1 and combined_width % 32 != 0)
        ):
            # The unaligned cache path consumes vector-produced packed bytes
            # from PIPE_S.  Make the V->S dependency explicit before scalar
            # loads from packed_output (not needed by the aligned MTE path).
            vec_sync_all()
        self._store_packed_row(
            packed_output,
            packed_data,
            packed_scale,
            gm_cache,
            gm_scale_cache,
            cache_row,
        )

    def __call__(
        self,
        gm_projected: Tensor,
        gm_rope_sin: Tensor,
        gm_rope_cos: Tensor,
        gm_gamma: Tensor,
        gm_cache_index: Tensor,
        gm_cache: Tensor,
        gm_scale_cache: Tensor,
        epsilon: dtypes.float32,
    ):
        dim = self.index_head_dim
        rd = self.rope_head_dim
        padded_dim = self.padded_dim
        padded_rd = _ceil_div(rd, VL) * VL
        projected = _make_ring_buffer(
            MemLoc.UB, dtypes.bfloat16, (1, padded_dim), depth=2
        )
        # This storage is reused by several V->V rounding boundaries before
        # the final MTE3 write, so it is a Buffer rather than a FIFO Channel.
        # One extra vector keeps the final masked BF16 unpack in bounds when
        # MXFP4 groups begin at a 32-element (half-vector) offset.
        output = _make_buffer(MemLoc.UB, dtypes.bfloat16, (1, padded_dim + VL))
        packed_output = _make_buffer(MemLoc.UB, dtypes.uint8, (1, padded_dim))
        packed_data = _make_ring_buffer(
            MemLoc.UB, dtypes.uint8, (1, dim // E2M1_VALUES_PER_BYTE), depth=1
        )
        packed_scale = _make_ring_buffer(
            MemLoc.UB, dtypes.uint8, (1, dim // FP4_GROUP_SIZE), depth=1
        )
        code_scratch = _make_buffer(
            MemLoc.UB, dtypes.uint32, (dim // FP4_GROUP_SIZE, VL)
        )
        gamma = _make_buffer(MemLoc.UB, dtypes.float32, (padded_dim,))
        work = _make_buffer(MemLoc.UB, dtypes.float32, (padded_dim + 2 * VL,))
        rope_sin = _make_buffer(MemLoc.UB, dtypes.float32, (1, padded_rd))
        rope_cos = _make_buffer(MemLoc.UB, dtypes.float32, (1, padded_rd))
        rotated = _make_buffer(MemLoc.UB, dtypes.float32, (padded_rd + 2 * VL,))

        mem_copy(gamma, gm_gamma)
        block_idx = get_block_idx()
        block_num = get_block_num()
        # The token count is a dynamic contract axis: every owner scans the
        # runtime row count.  Ownership is decided before RMSNorm/RoPE/packing,
        # so duplicate destinations remain deterministic while unowned rows
        # perform no vector work.
        total_rows = gm_projected.shape[0]
        for row in range(total_rows):
            cache_row = gm_cache_index[row]
            if cache_row != -1:
                if const_expr(self.storage_mode == 0):
                    owner_key = cache_row
                else:
                    group_size = self.combined_block_size
                    raw_block_size = gm_cache.shape[1] * group_size
                    block_id = cache_row // raw_block_size
                    slot = cache_row % raw_block_size
                    cache_group = slot // group_size
                    owner_key = block_id * (raw_block_size // group_size) + cache_group
                if owner_key % block_num == block_idx:
                    self._process_owned_row(
                        gm_projected,
                        gm_rope_sin,
                        gm_rope_cos,
                        gamma,
                        gm_cache,
                        gm_scale_cache,
                        projected,
                        output,
                        packed_output,
                        packed_data,
                        packed_scale,
                        code_scratch,
                        work,
                        rope_sin,
                        rope_cos,
                        rotated,
                        row,
                        cache_row,
                        epsilon,
                    )


class IndexerPrologueKPostprocess:
    """Host wrapper for the fused destination-owner vector stage."""

    def __init__(self, **kwargs):
        total_rows = int(kwargs["total_rows"])
        storage_mode = int(kwargs["storage_mode"])
        group_size = int(kwargs["combined_block_size"])
        _, vector_cores = get_effective_core_counts()
        self.block_num = (
            min(vector_cores, total_rows)
            if storage_mode == 0
            else min(vector_cores, _ceil_div(total_rows, group_size))
        )
        self.kernel = _FusedPostScatterKernel(**kwargs)

    @host
    def run(
        self,
        projected,
        rope_sin,
        rope_cos,
        gamma,
        cache_index,
        cache,
        scale_cache,
        epsilon: dtypes.float32,
        block_num,
    ):
        self.kernel[block_num](
            projected,
            rope_sin,
            rope_cos,
            gamma,
            cache_index,
            cache,
            scale_cache,
            epsilon,
        )


class IndexerPrologueKScatter:
    """Compatibility marker; scatter is fused into postprocess on this route."""


class _IndexerPrologueKPipeline:
    """Compile Projection and fused post-scatter into one host program."""

    def __init__(
        self,
        *,
        total_rows: int,
        head_dim: int,
        index_head_dim: int,
        rope_head_dim: int,
        storage_mode: int,
        combined_block_size: int,
        raw_block_size: int,
        write_scale_cache: bool,
    ):
        self.projection = _ProjectionKernel(total_rows, index_head_dim, head_dim)
        self.postprocess = IndexerPrologueKPostprocess(
            total_rows=total_rows,
            index_head_dim=index_head_dim,
            rope_head_dim=rope_head_dim,
            storage_mode=storage_mode,
            combined_block_size=combined_block_size,
            raw_block_size=raw_block_size,
            write_scale_cache=write_scale_cache,
        )

    @host
    def run(
        self,
        latent,
        weight,
        gamma,
        rope_sin,
        rope_cos,
        cache_index,
        cache,
        scale_cache,
        projected,
        _unused_packed_rows,
        epsilon: dtypes.float32,
        cube_block_num,
        vec_block_num,
    ):
        # Grid sizes stay runtime scalars so one binary adapts to every token
        # count and stream quota.  Kernel launches live in this @host source;
        # the per-stage @jit helpers only carry shared logic.
        self.projection.kernel[cube_block_num](latent, weight, projected)
        self.postprocess.kernel[vec_block_num](
            projected,
            rope_sin,
            rope_cos,
            gamma,
            cache_index,
            cache,
            scale_cache,
            epsilon,
        )


class _FusedIndexerPrologueKPipeline(_IndexerPrologueKPipeline):
    """Active two-kernel ABI with no temporary uint8 GM tensor."""

    @host
    def run_fused(
        self,
        latent,
        weight,
        gamma,
        rope_sin,
        rope_cos,
        cache_index,
        cache,
        scale_cache,
        projected,
        epsilon: dtypes.float32,
        cube_block_num,
        vec_block_num,
    ):
        self.projection.kernel[cube_block_num](latent, weight, projected)
        self.postprocess.kernel[vec_block_num](
            projected,
            rope_sin,
            rope_cos,
            gamma,
            cache_index,
            cache,
            scale_cache,
            epsilon,
        )


def _get_fused_pipeline_callable(
    *,
    head_dim: int,
    index_head_dim: int,
    rope_head_dim: int,
    storage_mode: int,
    combined_block_size: int,
    write_scale_cache: bool,
):
    """Compile/cache the dynamic two-kernel ABI without packed-row GM.

    The token count and the whole cache geometry are ``Dim`` axes, so the
    cache key is just the model shape plus the storage configuration: every
    ``T``/cache size reuses one binary, and the launch grids arrive as
    runtime scalar arguments on each call.
    """
    key = (
        head_dim,
        index_head_dim,
        rope_head_dim,
        storage_mode,
        combined_block_size,
        write_scale_cache,
    )
    compiled = _FUSED_AOT_CALLABLE_CACHE.get(key)
    if compiled is not None:
        return compiled
    # total_rows/raw_block_size are placeholders: the kernels derive both
    # from the tensors at runtime and never read the construction values.
    pipeline = _FusedIndexerPrologueKPipeline(
        total_rows=1,
        head_dim=head_dim,
        index_head_dim=index_head_dim,
        rope_head_dim=rope_head_dim,
        storage_mode=storage_mode,
        combined_block_size=combined_block_size,
        raw_block_size=1,
        write_scale_cache=write_scale_cache,
    )
    value_bytes = index_head_dim // E2M1_VALUES_PER_BYTE
    scale_bytes = index_head_dim // FP4_GROUP_SIZE
    cache_width = (
        value_bytes
        if storage_mode == 0
        else combined_block_size * (value_bytes + scale_bytes)
    )
    rows_dim = Dim("T", min=1)
    cache_blocks_dim = Dim("CACHE_BLOCKS", min=1)
    cache_slots_dim = Dim("CACHE_SLOTS", min=1)
    cache_stride0 = Dim("CACHE_S0")
    cache_stride1 = Dim("CACHE_S1")
    cache_spec = TensorSpec(
        (cache_blocks_dim, cache_slots_dim, 1, cache_width),
        dtypes.uint8,
        stride=(cache_stride0, cache_stride1, cache_width, 1),
    )
    if write_scale_cache:
        scale_spec = TensorSpec(
            (cache_blocks_dim, cache_slots_dim, 1, scale_bytes),
            dtypes.uint8,
            stride=(
                Dim("SCALE_S0"),
                Dim("SCALE_S1"),
                scale_bytes,
                1,
            ),
        )
    else:
        # Sentinel view produced by k_cache[:1, :1, :1, :1] when the public
        # function has no separate scale-cache output.  The view shares the
        # cache strides, so the spec reuses the cache stride Dims.
        scale_spec = TensorSpec(
            (1, 1, 1, 1),
            dtypes.uint8,
            stride=(cache_stride0, cache_stride1, cache_width, 1),
        )
    specs = (
        TensorSpec((rows_dim, head_dim), dtypes.bfloat16),
        TensorSpec(
            (index_head_dim, head_dim),
            dtypes.bfloat16,
            storage_format="nz",
        ),
        TensorSpec((index_head_dim,), dtypes.float32),
        TensorSpec((rows_dim, rope_head_dim), dtypes.float32),
        TensorSpec((rows_dim, rope_head_dim), dtypes.float32),
        TensorSpec((rows_dim,), dtypes.int64),
        cache_spec,
        scale_spec,
        TensorSpec((rows_dim, index_head_dim), dtypes.bfloat16),
        dtypes.float32,
        dtypes.int32,
        dtypes.int32,
    )
    compiled = dsl_compile(pipeline.run_fused, *specs)
    _FUSED_AOT_CALLABLE_CACHE[key] = compiled
    return compiled


def _get_pipeline_callable(
    *,
    total_rows,
    head_dim,
    index_head_dim,
    rope_head_dim,
    storage_mode,
    combined_block_size,
    raw_block_size,
    write_scale_cache,
    cache_shape,
    cache_stride,
    scale_cache_shape,
    scale_cache_stride,
    packed_stride,
):
    """Return a cached AOT callable for one fully static operator shape."""
    key = (
        total_rows,
        head_dim,
        index_head_dim,
        rope_head_dim,
        storage_mode,
        combined_block_size,
        raw_block_size,
        write_scale_cache,
        tuple(cache_shape),
        tuple(cache_stride),
        tuple(scale_cache_shape),
        tuple(scale_cache_stride),
        packed_stride,
    )
    compiled = _AOT_CALLABLE_CACHE.get(key)
    if compiled is not None:
        return compiled

    pipeline = _IndexerPrologueKPipeline(
        total_rows=total_rows,
        head_dim=head_dim,
        index_head_dim=index_head_dim,
        rope_head_dim=rope_head_dim,
        storage_mode=storage_mode,
        combined_block_size=combined_block_size,
        raw_block_size=raw_block_size,
        write_scale_cache=write_scale_cache,
    )
    specs = (
        TensorSpec((total_rows, head_dim), dtypes.bfloat16),
        TensorSpec(
            (index_head_dim, head_dim),
            dtypes.bfloat16,
            storage_format="nz",
        ),
        TensorSpec((index_head_dim,), dtypes.float32),
        TensorSpec((total_rows, rope_head_dim), dtypes.float32),
        TensorSpec((total_rows, rope_head_dim), dtypes.float32),
        TensorSpec((total_rows,), dtypes.int64),
        TensorSpec(tuple(cache_shape), dtypes.uint8, stride=tuple(cache_stride)),
        TensorSpec(
            tuple(scale_cache_shape),
            dtypes.uint8,
            stride=tuple(scale_cache_stride),
        ),
        TensorSpec((total_rows, index_head_dim), dtypes.bfloat16),
        TensorSpec((total_rows, packed_stride), dtypes.uint8),
        dtypes.float32,
        dtypes.int32,
        dtypes.int32,
    )
    compiled = dsl_compile(pipeline.run, *specs)
    _AOT_CALLABLE_CACHE[key] = compiled
    return compiled


def _validate_operator_geometry(latent, wk, rope_sin, storage_mode):
    """Layer 1: operator geometry (T/H/D/Dr and the storage mode)."""
    total_rows, head_dim = map(int, latent.shape)
    index_head_dim = int(wk.shape[0])
    rope_head_dim = int(rope_sin.shape[1])
    if total_rows <= 0:
        raise ValueError("T must be positive")
    if head_dim <= 0 or head_dim % NZ_FRACTAL_ALIGN != 0:
        raise ValueError(
            f"H must be a positive multiple of {NZ_FRACTAL_ALIGN} for FRACTAL_NZ wk"
        )
    if index_head_dim < FP4_GROUP_SIZE or index_head_dim % FP4_GROUP_SIZE != 0:
        raise ValueError(f"D must be a positive multiple of {FP4_GROUP_SIZE}")
    if rope_head_dim <= 0 or rope_head_dim % ROPE_PAIR_SIZE != 0:
        raise ValueError("Dr must be a positive even number")
    if rope_head_dim > index_head_dim:
        raise ValueError("Dr cannot exceed D")
    if storage_mode not in (0, 1):
        raise ValueError("storage_mode must be 0 or 1")
    return total_rows, head_dim, index_head_dim, rope_head_dim


def _validate_cache_layout(
    k_cache,
    k_scale_cache,
    storage_mode,
    combined_block_size,
    value_bytes,
    scale_bytes,
):
    """Layer 2: cache-side ABI contract for the requested storage mode."""
    if int(k_cache.shape[2]) != CACHE_ABI_DIM2:
        raise ValueError(f"k_cache dimension 2 must equal {CACHE_ABI_DIM2}")
    block_num = int(k_cache.shape[0])
    if storage_mode == 0:
        if int(k_cache.shape[3]) != value_bytes:
            raise ValueError(
                f"storage_mode=0 requires k_cache.shape[-1] == {value_bytes}"
            )
        if k_scale_cache is not None:
            expected_scale_shape = (
                block_num,
                int(k_cache.shape[1]),
                CACHE_ABI_DIM2,
                scale_bytes,
            )
            if tuple(k_scale_cache.shape) != expected_scale_shape:
                raise ValueError(
                    f"k_scale_cache must have shape {expected_scale_shape}"
                )
    else:
        if combined_block_size <= 0:
            raise ValueError(
                "storage_mode=1 requires combined_block_size to be positive"
            )
        expected_width = combined_block_size * (value_bytes + scale_bytes)
        if int(k_cache.shape[3]) != expected_width:
            raise ValueError(
                f"storage_mode=1 requires k_cache.shape[-1] == {expected_width}"
            )


def _resolve_launch_grids(total_rows, index_head_dim, storage_mode, group_size):
    """Return the (cube, vector) launch grid for one call.

    The compiled binary is shared across every T/cache geometry, so the grids
    are chosen per call from the runtime core quota and this call's workload,
    never baked into the compiled contract.  This function is the isolation
    seam for moving the elem-dependent scheduling onto a dedicated AICPU
    scheduler operator.
    """
    cube_cores, vector_cores = get_effective_core_counts()
    base_n = BASE_N if index_head_dim % BASE_N == 0 else NZ_FRACTAL_ALIGN
    cube_block_num = min(
        cube_cores,
        _ceil_div(total_rows, BASE_M) * _ceil_div(index_head_dim, base_n),
    )
    if storage_mode == 0:
        vec_block_num = min(vector_cores, total_rows)
    else:
        vec_block_num = min(vector_cores, _ceil_div(total_rows, group_size))
    return cube_block_num, vec_block_num


def indexer_prologue_k(
    latent: torch.Tensor,
    wk: torch.Tensor,
    norm_weight: torch.Tensor,
    rope_sin: torch.Tensor,
    rope_cos: torch.Tensor,
    k_cache: torch.Tensor,
    k_scale_cache: torch.Tensor | None = None,
    *,
    cache_index: torch.Tensor,
    storage_mode: int,
    norm_eps: float,
    combined_block_size: int = -1,
) -> torch.Tensor:
    """Project ``latent[T,H]`` and scatter packed MXFP4 rows into cache.

    ``storage_mode=0`` stores E2M1 bytes in ``k_cache`` and optionally stores
    UE8M0 scales in ``k_scale_cache``.  ``storage_mode=1`` combines ``G``
    tokens in each cache row, where ``G=combined_block_size``.  A cache index
    of ``-1`` skips the corresponding token.

    ``T`` and the cache geometry are dynamic contract axes: one compiled
    binary per model shape (H, D, Dr) and storage configuration serves every
    token count, cache size, and axis-0/1 cache stride.
    """
    total_rows, head_dim, index_head_dim, rope_head_dim = _validate_operator_geometry(
        latent, wk, rope_sin, storage_mode
    )
    value_bytes = index_head_dim // E2M1_VALUES_PER_BYTE
    scale_bytes = index_head_dim // FP4_GROUP_SIZE
    _validate_cache_layout(
        k_cache,
        k_scale_cache,
        storage_mode,
        combined_block_size,
        value_bytes,
        scale_bytes,
    )

    projected = torch.empty(
        (total_rows, index_head_dim), dtype=latent.dtype, device=latent.device
    )
    write_scale_cache = storage_mode == 0 and k_scale_cache is not None
    if write_scale_cache:
        scale_cache_arg = k_scale_cache
    else:
        # Keep the compiled pipeline ABI static without allocating a fake
        # output.  write_scale_cache=False is part of the compile key and
        # removes every gm_scale_cache access at compile time, so this alias
        # is never read or written by mode 0 without a scale cache (nor by
        # mode 1, whose scales are stored in the combined cache).
        scale_cache_arg = k_cache[:1, :1, :1, :1]
    compiled = _get_fused_pipeline_callable(
        head_dim=head_dim,
        index_head_dim=index_head_dim,
        rope_head_dim=rope_head_dim,
        storage_mode=storage_mode,
        combined_block_size=combined_block_size,
        write_scale_cache=write_scale_cache,
    )
    cube_block_num, vec_block_num = _resolve_launch_grids(
        total_rows, index_head_dim, storage_mode, combined_block_size
    )
    compiled(
        latent,
        wk,
        norm_weight,
        rope_sin,
        rope_cos,
        cache_index,
        k_cache,
        scale_cache_arg,
        projected,
        float(norm_eps),
        cube_block_num,
        vec_block_num,
    )
    return k_cache


def clear_caches():
    """Close cached executables."""
    for cache in (_FUSED_AOT_CALLABLE_CACHE, _AOT_CALLABLE_CACHE):
        for compiled in cache.values():
            compiled.close()
        cache.clear()


_EXPORT_DEFAULTS = {
    "head_dim": 5120,
    "index_head_dim": 128,
    "rope_head_dim": 64,
    "storage_mode": 0,
    "combined_block_size": -1,
    "write_scale_cache": True,
}


@export("indexer_prologue_k")
def export_indexer_prologue_k():
    """Collect one binary per model shape and storage configuration.

    T and the cache geometry are dynamic axes, so each profile is a single
    binary.  The defaults cover H=5120 (deployment) and H=512 (tests), each
    with mode 0 with/without a scale cache and mode 1 G=8.
    ``CANNBOTDSL_DS41_PROFILES`` may replace them through its
    ``indexer_prologue_k`` key; fields are those of ``_EXPORT_DEFAULTS``.
    """
    profiles = [
        {"head_dim": head_dim, **storage}
        for head_dim in (5120, 512)
        for storage in (
            {"storage_mode": 0, "write_scale_cache": True},
            {"storage_mode": 0, "write_scale_cache": False},
            {"storage_mode": 1, "combined_block_size": 8, "write_scale_cache": False},
        )
    ]
    profile_path = os.environ.get("CANNBOTDSL_DS41_PROFILES")
    if profile_path:
        with open(profile_path, encoding="utf-8") as stream:
            profiles = json.load(stream).get("indexer_prologue_k", profiles)
    clear_caches()
    try:
        for profile in profiles:
            unknown = set(profile) - set(_EXPORT_DEFAULTS)
            if unknown:
                raise ValueError(f"Unknown export profile fields: {sorted(unknown)}")
            _get_fused_pipeline_callable(**dict(_EXPORT_DEFAULTS, **profile))
    finally:
        clear_caches()
