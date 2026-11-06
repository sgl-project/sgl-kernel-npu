# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Public QLI MXFP4 implementation.

The public ABI is the one in the public interface specification: q is ``uint8[T1,32,64]`` and
k is ``uint8[block_num,block_size,1,64]`` (two packed E2M1 values per byte).
The Torch-facing host layer only validates the public ABI, builds launch
metadata/workspaces, and allocates public outputs.  It never creates
test inputs or computes a CPU reference.  DSL kernels load native FP4 E2M1
into L1/L0 and perform the Cube MX QK product into an FP32 accumulator.
Cube reads paged K and scales directly from GM. Two directed F322BF16
FIXP copies deliver the M halves to their AIVs. VEC1 applies BF16 ReLU,
weighted reduction and uint16 streaming TopK, with optional candidate TopK.
"""

from __future__ import annotations

__all__ = ["quant_lightning_indexer"]

import os
import threading
from math import gcd
from typing import Optional

import cannbotdsl
import torch
from cannbotdsl.aot import export

from cannbotdsl import MemLoc, Tensor, dtypes
from cannbotdsl.buffer import UB, make_buffer
from cannbotdsl.lang.jit import jit
from cannbotdsl.lang.host import host
from cannbotdsl.lang.kernel import kernel
from cannbotdsl import TensorSpec
from cannbotdsl.lang.vf import vf
from cannbotdsl.lang.control_flow import range as dsl_range
from cannbotdsl.lang.constexpr import range_constexpr
from cannbotdsl.ops.arch import get_block_idx, get_subblock_id
from cannbotdsl.ops.matmul import matmul as dsl_matmul
from cannbotdsl.ops.memcpy import mem_copy
from cannbotdsl.ops.sync import (
    PIPE,
    global_sync_all,
    vec_sync_all,
    cube_sync_intra_wait,
    cube_sync_intra_arrive,
    vec_sync_intra_wait,
    vec_sync_intra_arrive,
    vec_sync_notify,
    vec_sync_wait,
)
from cannbotdsl.tensor import (
    reinterpret,
    make_pointer,
    make_tensor,
    make_layout,
    tile_slice,
)
from cannbotdsl.ops import reg as rr

from cannbotdsl.ops.sync import cube_sync_all
from cannbotdsl.buffer import L0B, L0C
from cannbotdsl.ring_buffer import RingBuffer, make_ring_buffer
from cannbotdsl.lang.constexpr import const_expr
from cannbotdsl.ops.arch import is_aiv, get_squeeze_status
from cannbotdsl.ops.memcpy import copy_options, relu
from cannbotdsl.ops.sync import vec_sync_block_arrive, vec_sync_block_wait


# Group six N1=32 query rows into an M192 tile; N1=64 uses four rows.
QUERY_TILE_ROWS = 6
TILE_M = 32 * QUERY_TILE_ROWS
TILE_N = 256
QK_COLUMNS = TILE_N
CACHE_TOKENS = 25600
LOGICAL_D = 128
PACKED_D = LOGICAL_D // 2
N1 = 32
N2 = 1
VEC_TILE = 128
CANDIDATE_BLOCK_SIZE = 8
TOPK_TRUNK_LEN = 26624
MAX_CUBE_WORKERS = 36
# Metadata 的 LI 区占 36 行，LD/FD 归并记录从固定第 36 行开始。
LD_RECORD_START = 36
METADATA_ELEMENTS = 1024
METADATA_FIELDS = 8
METADATA_ROWS = METADATA_ELEMENTS // METADATA_FIELDS
# 该 TopK 容量采用缓存分数路径，其余容量使用通用流式路径。
CACHED_TOPK_COUNT = 512
BF16_NEG_INF_BITS = 0xFF80

BF16_SIGN_BIT = 0x8000


UINT16_ALL_BITS = 0xFFFF


OWNER_MIN_TOKENS = 32768


OWNER_MAX_TOKENS = 131072

STREAM_TOPK_TRUNK_LEN = 16384
MAX_MERGE_PARTS = 8
MAX_PA_BLOCK_SIZE = 1024
PA_BLOCK_ALIGNMENT = 16
QK_ROW_BYTES = TILE_N * 2
CACHED_QK_UB_BYTES = max(TILE_M * QK_ROW_BYTES, 98304)
CACHED_SCORE_UB_BYTES = (QUERY_TILE_ROWS // 2) * CACHE_TOKENS * 2
CACHED_HISTOGRAM_UB_OFFSET = CACHED_QK_UB_BYTES + CACHED_SCORE_UB_BYTES
HISTOGRAM_SLOT_OFFSET = QK_ROW_BYTES // 2
CACHED_VECTOR_UB_OFFSET = (
    CACHED_HISTOGRAM_UB_OFFSET + QUERY_TILE_ROWS * QK_ROW_BYTES + QK_ROW_BYTES
)
VECTOR_AFTER_QK_BYTES = 12288
# Keep histogram slots beyond the weights and both score-copy buffers.
HISTOGRAM_AFTER_QK_BYTES = 18432
# 缓存 TopK 的临时索引与输出使用已排空的 QK 区，不覆盖缓存分数。
CACHED_TOPK_INDEX_OFFSET = CACHE_TOKENS * 2 + 1024
KEY_STORAGE_ROW_ELEMENTS = N2 * PACKED_D
KEY_SCALE_STORAGE_ROW_ELEMENTS = N2 * 2 * 2


@jit
def _need_topk_values(metadata, return_values):
    return dtypes.int64(1 if metadata[LD_RECORD_START, 0] != 0 else return_values)


BYTE_CACHE_TOKENS = 49664


STREAM_ROWS = QUERY_TILE_ROWS // 2


STREAM_POOL_TOKENS = 8192


STREAM_PAGE_TOKENS = 4096


STREAM_ARENA_OFFSET = 98304


BYTE_ROW2_OFFSET = 2 * BYTE_CACHE_TOKENS


BYTE_OVERFLOW_OFFSET = STREAM_ROWS * BYTE_CACHE_TOKENS


BYTE_META_OFFSET = BYTE_OVERFLOW_OFFSET + STREAM_ROWS * CACHED_TOPK_COUNT * 2


STREAM_ARENA_BYTES = BYTE_META_OFFSET + STREAM_ROWS * 128 * 2


BYTE_FALLBACK_SCRATCH_OFFSET = 2 * BYTE_CACHE_TOKENS - STREAM_ARENA_OFFSET


BYTE_SEED_OFFSET = BYTE_CACHE_TOKENS - STREAM_ROWS * TILE_N * 2


STREAM_INDICES_OFFSET = STREAM_ROWS * STREAM_POOL_TOKENS * 2


STREAM_POSITIONS_OFFSET = 2 * STREAM_INDICES_OFFSET


STREAM_HISTOGRAM_OFFSET = STREAM_POSITIONS_OFFSET + STREAM_POOL_TOKENS * 2


STREAM_CARRIER_OFFSET = STREAM_HISTOGRAM_OFFSET + 256 * 2


STREAM_THRESHOLDS_OFFSET = STREAM_CARRIER_OFFSET + 64 * 4


STREAM_COUNTS_OFFSET = STREAM_THRESHOLDS_OFFSET + STREAM_ROWS * 16 * 2


STREAM_PACKED_OFFSET = STREAM_COUNTS_OFFSET + STREAM_ROWS * 8 * 4


STREAM_TMP_VALUES_OFFSET = STREAM_PACKED_OFFSET + STREAM_PAGE_TOKENS * 2


STREAM_TMP_INDICES_OFFSET = STREAM_TMP_VALUES_OFFSET + CACHED_TOPK_COUNT * 2


STREAM_STAGE_OFFSET = STREAM_TMP_INDICES_OFFSET + CACHED_TOPK_COUNT * 2

UB_CAPACITY_BYTES = 256 * 1024
UB_ALIGNMENT_BYTES = 32
# TopK 阶段的直方图与阈值暂存；避免覆盖仍待选择的缓存分数。
TOPK_HISTOGRAM_UB_OFFSET = CACHED_HISTOGRAM_UB_OFFSET + QUERY_TILE_ROWS * QK_ROW_BYTES
TOPK_TARGET_UB_OFFSET = CACHED_VECTOR_UB_OFFSET


TOPK_SELECT_TOPK = 512


TOPK_SELECT_POOL_CAPACITY = 8192


TOPK_SELECT_COMPACT_LIMIT = 4096


TOPK_SELECT_GROUP_SIZE = 4096


TOPK_SELECT_LOAD_TRUNK = 32768


DMA_TOPK_SELECT_TOPK = 512


DMA_TOPK_SELECT_POOL_CAPACITY = 18432


DMA_TOPK_SELECT_COMPACT_LIMIT = 2048


DMA_TOPK_SELECT_GROUP_SIZE = 16384


DMA_TOPK_SELECT_PAGE_SIZE = 4096


DMA_TOPK_SELECT_PAGE_BITS = DMA_TOPK_SELECT_PAGE_SIZE.bit_length() - 1


assert DMA_TOPK_SELECT_PAGE_SIZE == 1 << DMA_TOPK_SELECT_PAGE_BITS


POOL_ALIGNMENT_ELEMENTS = VEC_TILE


@jit
def _encode_bf16_sort_key(value, mask):
    bits = rr.vreinterpret(value, dtypes.uint16)
    sign = rr.vbitwise_and(
        bits, rr.vdups(BF16_SIGN_BIT, dtypes.uint16, mask=mask), mask=mask
    )
    negative = rr.veqs(sign, BF16_SIGN_BIT, mask=mask)
    positive_key = rr.vbitwise_xor(
        bits, rr.vdups(BF16_SIGN_BIT, dtypes.uint16, mask=mask), mask=mask
    )
    negative_key = rr.vbitwise_xor(
        bits, rr.vdups(UINT16_ALL_BITS, dtypes.uint16, mask=mask), mask=mask
    )
    return rr.vselect(negative_key, positive_key, cond_mask=negative)


@jit
def _decode_bf16_sort_key(keys, mask):
    positive = rr.veq(
        rr.vbitwise_and(
            keys, rr.vdups(BF16_SIGN_BIT, dtypes.uint16, mask=mask), mask=mask
        ),
        rr.vdups(BF16_SIGN_BIT, dtypes.uint16, mask=mask),
        mask=mask,
    )
    value_bits = rr.vselect(
        rr.vbitwise_xor(
            keys, rr.vdups(BF16_SIGN_BIT, dtypes.uint16, mask=mask), mask=mask
        ),
        rr.vbitwise_xor(
            keys, rr.vdups(UINT16_ALL_BITS, dtypes.uint16, mask=mask), mask=mask
        ),
        cond_mask=positive,
    )
    return value_bits


@jit
def _decode_cached_score_bits(keys, mask):
    return rr.vselect(
        rr.vbitwise_xor(
            keys, rr.vdups(BF16_SIGN_BIT, dtypes.uint16, mask=mask), mask=mask
        ),
        rr.vbitwise_xor(
            keys, rr.vdups(UINT16_ALL_BITS, dtypes.uint16, mask=mask), mask=mask
        ),
        cond_mask=rr.vges(keys, BF16_SIGN_BIT, mask=mask),
    )


class _UbPool:
    """编译期 UB 区间规划：保留显式别名，为其余暂存分配空闲区间。"""

    def __init__(self, start=0):
        self.start = start
        self.regions = []

    def slots(self, shape, dtype, depth=1, stride=None, addr=None):
        if stride is None:
            strides = []
            element_stride = 1
            for axis_size in reversed(shape):
                strides.insert(0, element_stride)
                element_stride *= axis_size
            stride = tuple(strides)
        span_bytes = (
            (sum((size - 1) * step for size, step in zip(shape, stride)) + 1)
            * dtype.bits
            // 8
        )
        alignment_mask = UB_ALIGNMENT_BYTES - 1
        slot_pitch = (
            (span_bytes + alignment_mask) // UB_ALIGNMENT_BYTES * UB_ALIGNMENT_BYTES
        )
        start = (
            (self.start + alignment_mask) // UB_ALIGNMENT_BYTES * UB_ALIGNMENT_BYTES
            if addr is None
            else addr
        )
        if addr is None:
            for ub_begin, ub_end in sorted(self.regions):
                if start + slot_pitch * depth <= ub_begin:
                    break
                start = max(start, ub_end)
        self.regions.append((start, start + slot_pitch * depth))
        if (
            start < 0
            or start + span_bytes + (depth - 1) * slot_pitch > UB_CAPACITY_BYTES
        ):
            raise ValueError("QLI UB allocation exceeds the device capacity")
        ub_root = UB.view(UB_CAPACITY_BYTES)
        return [
            reinterpret(
                ub_root,
                dtype=dtype,
                offset=start + slot * slot_pitch,
                layout=make_layout(shape, stride=stride),
            )
            for slot in range(depth)
        ]


def _ub_buffer(pool, memloc, shape, dtype, *, addr=None, stride=None):
    return make_buffer(pool.slots(shape, dtype, stride=stride, addr=addr)[0])


def _ub_ring_buffer(
    pool,
    memloc,
    shape,
    dtype,
    *,
    depth=1,
    addr=None,
    stride=None,
):
    slots = pool.slots(shape, dtype, depth=depth, stride=stride, addr=addr)
    return make_ring_buffer(slots)


def _offset_view(tensor, offsets):
    """Rebase an ND input with the current DSL's zero-copy interval slicing."""
    if all(isinstance(offset, int) and offset == 0 for offset in offsets):
        return tensor
    return tensor[tuple(slice(offset, None) for offset in offsets)]


def ceil_div(value: int, divisor: int) -> int:
    return (int(value) + int(divisor) - 1) // int(divisor)


def _pa_axis0_storage_rows(
    tensor: torch.Tensor,
    name: str,
    expected_inner_strides: tuple[int, ...],
    storage_row_elements: int,
) -> tuple[int, int]:
    """Validate ASC's PA stride contract and expose its storage-row span.

    Only axis 0 may contain padding. The returned extent includes that
    padding without copying or repacking public input data; the fused kernel
    still selects physical pages with the original axis-0 stride.
    """
    strides = tuple(int(step) for step in tensor.stride())
    if strides[1:] != expected_inner_strides:
        raise ValueError(
            f"{name} only supports non-contiguous storage on axis 0 in PA_BBND layout"
        )
    stride0 = strides[0]
    if stride0 < 0:
        raise ValueError(f"{name} axis-0 stride must be non-negative")
    if stride0 % storage_row_elements != 0:
        raise ValueError(f"{name} axis-0 stride must preserve complete PA storage rows")
    blocks = int(tensor.shape[0])
    if blocks <= 0:
        raise ValueError(f"{name} must contain at least one physical PA block")
    storage_rows = (blocks - 1) * (stride0 // storage_row_elements) + int(
        tensor.shape[1]
    )
    return stride0, storage_rows


class Owner64Vector:
    # 初始化M64/N512 单 Vector 评分的配置与缓冲区。
    def __init__(self, n1):
        self.n1 = int(n1)
        self.vec_rows = 6
        self.subblock_idx = 0
        self.ub_pool = _UbPool(196608)
        self.weight_ub = _ub_ring_buffer(
            self.ub_pool, MemLoc.UB, (6, self.n1), dtypes.float32, depth=1
        )
        self.packed_weight = _ub_buffer(
            self.ub_pool, MemLoc.UB, (6, 128), dtypes.bfloat16
        )
        self.key_ub = _ub_ring_buffer(
            self.ub_pool, MemLoc.UB, (2, 512), dtypes.uint16, depth=2
        )

    # 加载并打包 query 权重，供 VF 沿 head 维累加。
    @jit
    def load_weights(self, weights, public_row_base, task_span_count):
        weight_row = (
            dtypes.int64(public_row_base)
            + dtypes.int64(self.subblock_idx) * self.vec_rows
        )
        if dtypes.int32(self.subblock_idx) * self.vec_rows >= task_span_count:
            weight_row = dtypes.int64(public_row_base)
        mem_copy(
            self.weight_ub.next_write(),
            tile_slice(
                _offset_view(weights, (weight_row, 0)), (self.vec_rows, self.n1), (0, 0)
            ),
        )
        weight_read = self.weight_ub.next_read()
        with vf(mode="simd"):
            mask32 = rr.update_mask(self.n1, elem_bits=32)[0]
            mask16 = rr.update_mask(self.n1, elem_bits=16)[0]
            for row in range_constexpr(self.vec_rows):
                weight = rr.vcast(
                    rr.vload(weight_read, row * self.n1),
                    dtypes.bfloat16,
                    mask=mask32,
                    reg_layout=rr.RegLayout.ZERO,
                )
                packed = rr.vpack(
                    rr.vreinterpret_lanes(weight, dtypes.uint32),
                    dtypes.uint16,
                    part="lower",
                )
                rr.vstore(
                    self.packed_weight,
                    row * self.n1,
                    rr.vreinterpret(packed, dtypes.bfloat16),
                    mask16,
                )
            rr.vmem_bar("vst_vld")

    # 处理单 Vector 接收的 M64/N512 结果，完成两行加权评分。
    @jit
    def score(self, qk_handoff, group_count, query_start):
        key_write = self.key_ub.next_write()
        with vf(mode="simd"):
            mask16 = rr.update_mask(128, elem_bits=16)[0]
            acc0_0 = rr.vdups(0.0, dtypes.bfloat16, mask=mask16)
            acc0_1 = rr.vdups(0.0, dtypes.bfloat16, mask=mask16)
            acc0_2 = rr.vdups(0.0, dtypes.bfloat16, mask=mask16)
            acc0_3 = rr.vdups(0.0, dtypes.bfloat16, mask=mask16)
            acc1_0 = rr.vdups(0.0, dtypes.bfloat16, mask=mask16)
            acc1_1 = rr.vdups(0.0, dtypes.bfloat16, mask=mask16)
            acc1_2 = rr.vdups(0.0, dtypes.bfloat16, mask=mask16)
            acc1_3 = rr.vdups(0.0, dtypes.bfloat16, mask=mask16)
            for group in dsl_range(group_count, unroll=1):
                w0 = rr.vload_broadcast(
                    self.packed_weight, (query_start + 0) * self.n1 + group
                )
                x0_0 = rr.vload(qk_handoff, (0 * self.n1 + group) * 512 + 0)
                acc0_0 = rr.vmula(acc0_0, x0_0, w0, mask=mask16)
                x0_1 = rr.vload(qk_handoff, (0 * self.n1 + group) * 512 + 128)
                acc0_1 = rr.vmula(acc0_1, x0_1, w0, mask=mask16)
                x0_2 = rr.vload(qk_handoff, (0 * self.n1 + group) * 512 + 256)
                acc0_2 = rr.vmula(acc0_2, x0_2, w0, mask=mask16)
                x0_3 = rr.vload(qk_handoff, (0 * self.n1 + group) * 512 + 384)
                acc0_3 = rr.vmula(acc0_3, x0_3, w0, mask=mask16)
                w1 = rr.vload_broadcast(
                    self.packed_weight, (query_start + 1) * self.n1 + group
                )
                x1_0 = rr.vload(qk_handoff, (1 * self.n1 + group) * 512 + 0)
                acc1_0 = rr.vmula(acc1_0, x1_0, w1, mask=mask16)
                x1_1 = rr.vload(qk_handoff, (1 * self.n1 + group) * 512 + 128)
                acc1_1 = rr.vmula(acc1_1, x1_1, w1, mask=mask16)
                x1_2 = rr.vload(qk_handoff, (1 * self.n1 + group) * 512 + 256)
                acc1_2 = rr.vmula(acc1_2, x1_2, w1, mask=mask16)
                x1_3 = rr.vload(qk_handoff, (1 * self.n1 + group) * 512 + 384)
                acc1_3 = rr.vmula(acc1_3, x1_3, w1, mask=mask16)
            acc0_0 = rr.vadds(acc0_0, 0.0, mask=mask16)
            sortable_bits = _encode_bf16_sort_key(acc0_0, mask16)
            key = sortable_bits
            rr.vstore(key_write, 0 * 512 + 0, key, mask16)
            acc0_1 = rr.vadds(acc0_1, 0.0, mask=mask16)
            sortable_bits = _encode_bf16_sort_key(acc0_1, mask16)
            key = sortable_bits
            rr.vstore(key_write, 0 * 512 + 128, key, mask16)
            acc0_2 = rr.vadds(acc0_2, 0.0, mask=mask16)
            sortable_bits = _encode_bf16_sort_key(acc0_2, mask16)
            key = sortable_bits
            rr.vstore(key_write, 0 * 512 + 256, key, mask16)
            acc0_3 = rr.vadds(acc0_3, 0.0, mask=mask16)
            sortable_bits = _encode_bf16_sort_key(acc0_3, mask16)
            key = sortable_bits
            rr.vstore(key_write, 0 * 512 + 384, key, mask16)
            acc1_0 = rr.vadds(acc1_0, 0.0, mask=mask16)
            sortable_bits = _encode_bf16_sort_key(acc1_0, mask16)
            key = sortable_bits
            rr.vstore(key_write, 1 * 512 + 0, key, mask16)
            acc1_1 = rr.vadds(acc1_1, 0.0, mask=mask16)
            sortable_bits = _encode_bf16_sort_key(acc1_1, mask16)
            key = sortable_bits
            rr.vstore(key_write, 1 * 512 + 128, key, mask16)
            acc1_2 = rr.vadds(acc1_2, 0.0, mask=mask16)
            sortable_bits = _encode_bf16_sort_key(acc1_2, mask16)
            key = sortable_bits
            rr.vstore(key_write, 1 * 512 + 256, key, mask16)
            acc1_3 = rr.vadds(acc1_3, 0.0, mask=mask16)
            sortable_bits = _encode_bf16_sort_key(acc1_3, mask16)
            key = sortable_bits
            rr.vstore(key_write, 1 * 512 + 384, key, mask16)
            rr.vmem_bar("vst_vld")


class QliRawTopKWorkspace:
    """UB scratch shared by sequential Sparse and Candidate TopK stages."""

    def __init__(
        self,
        max_trunk_len: int,
        max_topk: int,
        pipelined: bool = False,
        live_histogram=None,
        merge_only=False,
    ):
        self.ub_pool = _UbPool()
        if live_histogram is not None:
            self.ub_pool.regions.append(live_histogram)
        self.max_trunk_len = int(max_trunk_len)
        self.max_topk = int(max_topk)
        self.pipelined = bool(pipelined)
        self.topk_pad = ceil_div(self.max_topk, 256) * 256
        self.merge_len_pad = ceil_div(self.topk_pad + self.max_trunk_len, 128) * 128
        if self.pipelined:
            self.current_trunk = _ub_ring_buffer(
                self.ub_pool,
                MemLoc.UB,
                (1, self.max_trunk_len),
                dtypes.uint16,
                depth=1,
                addr=4 * self.merge_len_pad,
            )
        if self.pipelined:
            self.merge_key = _ub_ring_buffer(
                self.ub_pool,
                MemLoc.UB,
                (1, self.merge_len_pad),
                dtypes.uint16,
                depth=2,
                addr=0,
            )
        else:
            self.merge_key = _ub_ring_buffer(
                self.ub_pool,
                MemLoc.UB,
                (1, self.merge_len_pad),
                dtypes.uint16,
                depth=1,
                addr=0,
            )
        self._allocate_index_scratch(merge_only)
        if not merge_only:
            self.history_idx = _ub_buffer(
                self.ub_pool,
                MemLoc.UB,
                (2, self.topk_pad),
                dtypes.int32,
            )
            self.next_idx = self.history_idx

    def _allocate_index_scratch(self, merge_only=False):
        self.tmp_idx = _ub_buffer(
            self.ub_pool, MemLoc.UB, (self.merge_len_pad,), dtypes.uint16
        )
        self.histogram = _ub_buffer(
            self.ub_pool,
            MemLoc.UB,
            (256,),
            dtypes.uint16,
            addr=TOPK_HISTOGRAM_UB_OFFSET,
        )
        self.high_target_carrier = _ub_buffer(
            self.ub_pool, MemLoc.UB, (64,), dtypes.int32, addr=TOPK_TARGET_UB_OFFSET
        )
        if not merge_only:
            self.history_key = _ub_buffer(
                self.ub_pool, MemLoc.UB, (2, self.topk_pad), dtypes.uint16
            )
            self.next_history_key = self.history_key


class QliRawTopKSelector:
    """ASC Arch35 uint16 radix TopK with in-AIV trunk/history merge.

    This is not LD/S2 core splitting.  One AIV processes a row sequentially in
    16K trunks.  The first trunk creates a TopK history; every later pass runs
    the same radix selector over ``history + current trunk`` and maps its local
    uint16 positions back to global int32 token indices.
    """

    def __init__(
        self,
        topk: int,
        workspace: QliRawTopKWorkspace,
    ):
        self.ub_pool = workspace.ub_pool
        self.topk = int(topk)
        if self.topk <= 0:
            raise ValueError("ASC raw TopK requires positive topk")
        # The ASC VF works in fixed 16K trunks.  The number of complete trunks
        # and the final tail length are runtime values so one compiled kernel
        # can serve every supported S2.
        self.trunk_len = workspace.max_trunk_len
        # ASC reserves history in 256-element units before the next trunk.
        self.topk_pad = ceil_div(self.topk, 256) * 256
        self.merge_len_pad = ceil_div(self.topk_pad + self.trunk_len, 128) * 128
        self.workspace = workspace
        self.pipelined = bool(getattr(workspace, "pipelined", False))
        if (
            self.trunk_len > workspace.max_trunk_len
            or self.topk > workspace.max_topk
            or self.merge_len_pad > workspace.merge_len_pad
        ):
            raise ValueError("TopK selector exceeds the shared UB workspace")
        # Final V->MTE3 stages are selector-owned so Sparse and Candidate can
        # share computational scratch without racing each other's GM writes.
        self.output_idx_stage = _ub_ring_buffer(
            self.ub_pool,
            MemLoc.UB,
            (1, self.topk),
            dtypes.int32,
            depth=1,
        )
        self.output_value_bits_stage = _ub_ring_buffer(
            self.ub_pool,
            MemLoc.UB,
            (1, self.topk),
            dtypes.uint16,
            depth=1,
        )
        self.hist_slot0 = workspace.histogram
        self.hist_slot1 = workspace.histogram

    @jit
    def _ranked_output_vf(
        self, keys, pos32, chunk, mask16, mask32, index_base, output_valid_count
    ):
        value_bits = _decode_bf16_sort_key(keys, mask16)
        valid32 = rr.vlts(
            rr.varange(chunk * 64, dtypes.uint32), output_valid_count, mask=mask32
        )
        valid16 = rr.vlts(
            rr.varange(chunk * 64, dtypes.uint16), output_valid_count, mask=mask16
        )
        indices = rr.vselect(
            rr.vadds(rr.vreinterpret(pos32, dtypes.int32), index_base, mask=mask32),
            rr.vdups(-1, dtypes.int32, mask=mask32),
            cond_mask=valid32,
        )
        value_bits = rr.vselect(
            value_bits,
            rr.vdups(BF16_NEG_INF_BITS, dtypes.uint16, mask=mask16),
            cond_mask=valid16,
        )
        return (indices, value_bits)

    @jit
    def _collect_threshold_indices_vf(self, sort_key, valid_len, tmp_idx, kth):
        ureg = rr.vstore_unalign_begin(tmp_idx)
        valid_chunk_count = (valid_len + 127) // 128
        for chunk in dsl_range(valid_chunk_count, unroll=1):
            valid_count = valid_len - chunk * 128
            mask16 = rr.update_mask(valid_count, elem_bits=16)[0]
            keys = rr.vload(sort_key, chunk * 128)
            idx = rr.varange(chunk * 128, dtypes.uint16)
            sq = rr.vsqueeze_and_storeunalign_init(
                idx, mask=rr.vgt(keys, kth, mask=mask16)
            )
            rr.vsqueeze_and_storeunalign(tmp_idx, 0, sq, ureg)
        for chunk in dsl_range(valid_chunk_count, unroll=1):
            valid_count = valid_len - chunk * 128
            mask16 = rr.update_mask(valid_count, elem_bits=16)[0]
            keys = rr.vload(sort_key, chunk * 128)
            idx = rr.varange(chunk * 128, dtypes.uint16)
            sq = rr.vsqueeze_and_storeunalign_init(
                idx, mask=rr.veq(keys, kth, mask=mask16)
            )
            rr.vsqueeze_and_storeunalign(tmp_idx, 0, sq, ureg)
        rr.vstore_unalign_post(tmp_idx, 0, ureg)

    @jit
    def _emit_indices_only_vf(self, tmp, out, output_valid_count, index_base):
        for chunk in dsl_range(ceil_div(self.topk, 64), unroll=1):
            mask = rr.update_mask(self.topk - chunk * 64, elem_bits=32)[0]
            positions = rr.vunpack(
                rr.vload(tmp, chunk * 64), dtypes.uint32, part="lower"
            )
            indices = rr.vadds(
                rr.vreinterpret(positions, dtypes.int32), index_base, mask=mask
            )
            valid = rr.vlts(
                rr.varange(chunk * 64, dtypes.uint32), output_valid_count, mask=mask
            )
            rr.vstore(
                out,
                chunk * 64,
                rr.vselect(
                    indices, rr.vdups(-1, dtypes.int32, mask=mask), cond_mask=valid
                ),
                mask,
            )

    @jit
    def _prepare_ld_merge(self, partial_indices, partial_bits, valid_length, preloaded):
        if not preloaded:
            mem_copy(
                reinterpret(
                    self.ld_indices.next_write(), layout=make_layout((1, valid_length))
                ),
                partial_indices[0:1, 0:valid_length],
            )
            mem_copy(
                reinterpret(
                    self.ld_bits.next_write(), layout=make_layout((1, valid_length))
                ),
                partial_bits[0:1, 0:valid_length],
            )
        current_trunk = self.workspace.merge_key
        trunk_write = current_trunk.next_write()
        ld_idx_read = self.ld_indices.next_read()
        ld_bits_read = self.ld_bits.next_read()
        with vf(mode="simd"):
            mask16 = rr.update_mask(64, elem_bits=16)[0]
            mask32 = rr.update_mask(64, elem_bits=32)[0]
            for chunk in dsl_range((valid_length + 63) // 64, unroll=1):
                count = min(64, valid_length - chunk * 64)
                mask16 = rr.update_mask(count, elem_bits=16)[0]
                mask32 = rr.update_mask(count, elem_bits=32)[0]
                bits = rr.vload(ld_bits_read, chunk * 64)
                indices = rr.vload(ld_idx_read, chunk * 64)
                positive_key = rr.vbitwise_xor(
                    bits,
                    rr.vdups(BF16_SIGN_BIT, dtypes.uint16, mask=mask16),
                    mask=mask16,
                )
                negative_key = rr.vbitwise_xor(
                    bits,
                    rr.vdups(UINT16_ALL_BITS, dtypes.uint16, mask=mask16),
                    mask=mask16,
                )
                negative = rr.veqs(
                    rr.vbitwise_and(
                        bits,
                        rr.vdups(BF16_SIGN_BIT, dtypes.uint16, mask=mask16),
                        mask=mask16,
                    ),
                    32768,
                    mask=mask16,
                )
                keys = rr.vselect(negative_key, positive_key, cond_mask=negative)
                valid32 = rr.vselect(
                    rr.vdups(1, dtypes.int32, mask=mask32),
                    rr.vdups(0, dtypes.int32, mask=mask32),
                    cond_mask=rr.vges(indices, 0, mask=mask32),
                )
                valid16 = rr.vpack(
                    rr.vreinterpret(valid32, dtypes.uint32), dtypes.uint16, part="lower"
                )
                keys = rr.vselect(
                    keys,
                    rr.vdups(0, dtypes.uint16, mask=mask16),
                    cond_mask=rr.veqs(valid16, 1, mask=mask16),
                )
                rr.vstore(trunk_write, chunk * 64, keys, mask16)
            rr.vmem_bar("vst_vld")
        trunk_read = current_trunk.next_read()
        self._find_indices(
            trunk_read,
            valid_length,
            self.workspace.tmp_idx,
            self.workspace.histogram,
            self.workspace.high_target_carrier,
        )
        return (trunk_read, ld_idx_read, ld_bits_read)

    @jit
    def _emit_ld_values(
        self,
        current_trunk,
        ld_indices,
        ld_bits,
        output_offset,
        gm_sparse_indices,
        gm_selected_value_bits,
        emit_gm=1,
    ):
        output_idx_stage = self.output_idx_stage.next_write()
        out_value_bits = self.output_value_bits_stage.next_write()
        output_valid_count = dtypes.int32(self.topk)
        with vf(mode="simd"):
            for chunk in dsl_range(ceil_div(self.topk, 64), unroll=1):
                count = self.topk - chunk * 64
                mask32 = rr.update_mask(count, elem_bits=32)[0]
                mask16 = rr.mask_and(
                    rr.update_mask(count, elem_bits=16)[0],
                    rr.update_mask(64, elem_bits=16)[0],
                    exec_mask=rr.update_mask(64, elem_bits=16)[0],
                )
                positions = rr.vunpack(
                    rr.vload(self.workspace.tmp_idx, chunk * 64),
                    dtypes.uint32,
                    part="lower",
                )
                indices = rr.vgather(
                    ld_indices, rr.vreinterpret(positions, dtypes.uint32), mask=mask32
                )
                positions16 = rr.vload(self.workspace.tmp_idx, chunk * 64)
                keys = rr.vgather(current_trunk, positions16, mask=mask16)
                value_bits = rr.vgather(ld_bits, positions16, mask=mask16)
                rr.vlts(
                    rr.varange(chunk * 64, dtypes.uint32),
                    output_valid_count,
                    mask=mask32,
                )
                rr.vlts(
                    rr.varange(chunk * 64, dtypes.uint16),
                    output_valid_count,
                    mask=mask16,
                )
                key_valid16 = rr.vselect(
                    rr.vdups(1, dtypes.uint16, mask=mask16),
                    rr.vdups(0, dtypes.uint16, mask=mask16),
                    cond_mask=rr.vges(keys, 1, mask=mask16),
                )
                key_valid32 = rr.vunpack(key_valid16, dtypes.uint32, part="lower")
                indices = rr.vselect(
                    indices,
                    rr.vdups(-1, dtypes.int32, mask=mask32),
                    cond_mask=rr.veqs(key_valid32, 1, mask=mask32),
                )
                valid_flags32 = rr.vselect(
                    rr.vdups(1, dtypes.int32, mask=mask32),
                    rr.vdups(0, dtypes.int32, mask=mask32),
                    cond_mask=rr.vges(indices, 0, mask=mask32),
                )
                valid_flags16 = rr.vpack(
                    rr.vreinterpret(valid_flags32, dtypes.uint32),
                    dtypes.uint16,
                    part="lower",
                )
                indices = rr.vselect(
                    indices,
                    rr.vdups(-1, dtypes.int32, mask=mask32),
                    cond_mask=rr.vges(indices, 0, mask=mask32),
                )
                value_bits = rr.vselect(
                    value_bits,
                    rr.vdups(BF16_NEG_INF_BITS, dtypes.uint16, mask=mask16),
                    cond_mask=rr.veqs(valid_flags16, 1, mask=mask16),
                )
                indices = rr.vselect(
                    rr.vadds(indices, output_offset, mask=mask32),
                    rr.vdups(-1, dtypes.int32, mask=mask32),
                    cond_mask=rr.veqs(valid_flags32, 1, mask=mask32),
                )
                rr.vstore(output_idx_stage, chunk * 64, indices, mask32)
                rr.vstore(out_value_bits, chunk * 64, value_bits, mask16)
            rr.vmem_bar("vst_vld")
        if emit_gm != 0:
            mem_copy(gm_sparse_indices, self.output_idx_stage.next_read())
            mem_copy(gm_selected_value_bits, self.output_value_bits_stage.next_read())

    def _init_ld_buffers(self, splits):
        self.ld_length = int(splits) * self.topk
        storage_length = ceil_div(self.ld_length, 128) * 128
        self.ld_indices = _ub_ring_buffer(
            self.ub_pool, MemLoc.UB, (1, storage_length), dtypes.int32, depth=1
        )
        self.ld_bits = _ub_ring_buffer(
            self.ub_pool, MemLoc.UB, (1, storage_length), dtypes.uint16, depth=1
        )

    @jit
    def _select_from_prefix_hist_vf(
        self,
        sort_key,
        valid_len,
        tmp_idx,
        histogram_low_reg,
        histogram_high_reg,
        threshold_out=None,
    ):
        full16 = rr.update_mask(128, elem_bits=16)[0]
        rr.update_mask(1, elem_bits=16)[0]
        bottom_k = valid_len - self.topk + 1
        all_ones = rr.vdups(UINT16_ALL_BITS, dtypes.uint16, mask=full16)
        hist0 = histogram_low_reg
        hist1 = histogram_high_reg
        candidate0 = rr.vselect(
            rr.varange(0, dtypes.uint16),
            all_ones,
            cond_mask=rr.vges(hist0, bottom_k, mask=full16),
        )
        candidate1 = rr.vselect(
            rr.varange(128, dtypes.uint16),
            all_ones,
            cond_mask=rr.vges(hist1, bottom_k, mask=full16),
        )
        high_target = rr.vreduce_min(
            rr.vmin(candidate0, candidate1, mask=full16), mask=full16
        )
        full16 = rr.update_mask(128, elem_bits=16)[0]
        rr.update_mask(1, elem_bits=16)[0]
        zero16 = rr.vdups(0, dtypes.uint16, mask=full16)
        one16 = rr.vdups(1, dtypes.uint16, mask=full16)
        bottom_k = valid_len - self.topk + 1
        high_target = rr.vdup(high_target, mask=full16)
        high_target8 = rr.vreinterpret_lanes(
            rr.vbitwise_or(
                high_target, rr.vshl(high_target, 8, mask=full16), mask=full16
            ),
            dtypes.uint8,
        )
        prev = rr.vsub(high_target, one16, mask=full16)
        is_zero = rr.veqs(high_target, 0, mask=full16)
        prev = rr.vselect(zero16, prev, cond_mask=is_zero)
        prev_lane = rr.vbitwise_and(
            prev, rr.vdups(127, dtypes.uint16, mask=full16), mask=full16
        )
        prev0 = rr.vgather_reg(histogram_low_reg, prev_lane)
        prev1 = rr.vgather_reg(histogram_high_reg, prev_lane)
        prev_count = rr.vselect(prev0, prev1, cond_mask=rr.vlts(prev, 128, mask=full16))
        prev_count = rr.vselect(zero16, prev_count, cond_mask=is_zero)
        next_k = rr.vsub(
            rr.vdups(bottom_k, dtypes.uint16, mask=full16), prev_count, mask=full16
        )
        next_k_register = rr.vdup(next_k, mask=full16)
        l0c0 = rr.vdups(0, dtypes.uint16, mask=full16)
        l1c0 = rr.vdups(0, dtypes.uint16, mask=full16)
        l0c1 = rr.vdups(0, dtypes.uint16, mask=full16)
        l1c1 = rr.vdups(0, dtypes.uint16, mask=full16)
        l0c2 = rr.vdups(0, dtypes.uint16, mask=full16)
        l1c2 = rr.vdups(0, dtypes.uint16, mask=full16)
        l0c3 = rr.vdups(0, dtypes.uint16, mask=full16)
        l1c3 = rr.vdups(0, dtypes.uint16, mask=full16)
        full8 = rr.update_mask(256, elem_bits=8)[0]
        for block in dsl_range(valid_len // 1024, unroll=1):
            low, high = rr.vload_deinterleave(sort_key, block * 1024 + 0, width="b8")
            eq_high = rr.veq(high, high_target8, mask=full8)
            l0c0 = rr.vhistogram_accumulate(l0c0, low, mask=eq_high, bin=0)
            l1c0 = rr.vhistogram_accumulate(l1c0, low, mask=eq_high, bin=1)
            low, high = rr.vload_deinterleave(sort_key, block * 1024 + 256, width="b8")
            eq_high = rr.veq(high, high_target8, mask=full8)
            l0c1 = rr.vhistogram_accumulate(l0c1, low, mask=eq_high, bin=0)
            l1c1 = rr.vhistogram_accumulate(l1c1, low, mask=eq_high, bin=1)
            low, high = rr.vload_deinterleave(sort_key, block * 1024 + 512, width="b8")
            eq_high = rr.veq(high, high_target8, mask=full8)
            l0c2 = rr.vhistogram_accumulate(l0c2, low, mask=eq_high, bin=0)
            l1c2 = rr.vhistogram_accumulate(l1c2, low, mask=eq_high, bin=1)
            low, high = rr.vload_deinterleave(sort_key, block * 1024 + 768, width="b8")
            eq_high = rr.veq(high, high_target8, mask=full8)
            l0c3 = rr.vhistogram_accumulate(l0c3, low, mask=eq_high, bin=0)
            l1c3 = rr.vhistogram_accumulate(l1c3, low, mask=eq_high, bin=1)
        for chunk in dsl_range(
            valid_len // 1024 * 4, (valid_len + 255) // 256, unroll=1
        ):
            mask8 = rr.update_mask(valid_len - chunk * 256, elem_bits=8)[0]
            low, high = rr.vload_deinterleave(sort_key, chunk * 256, width="b8")
            eq_high = rr.veq(high, high_target8, mask=mask8)
            l0c0 = rr.vhistogram_accumulate(l0c0, low, mask=eq_high, bin=0)
            l1c0 = rr.vhistogram_accumulate(l1c0, low, mask=eq_high, bin=1)
        l0 = rr.vadd(l0c0, l0c1, mask=full16)
        l0 = rr.vadd(l0, l0c2, mask=full16)
        l0 = rr.vadd(l0, l0c3, mask=full16)
        l1 = rr.vadd(l1c0, l1c1, mask=full16)
        l1 = rr.vadd(l1, l1c2, mask=full16)
        l1 = rr.vadd(l1, l1c3, mask=full16)
        full16 = rr.update_mask(128, elem_bits=16)[0]
        rr.update_mask(1, elem_bits=16)[0]
        next_k = next_k_register
        all_ones = rr.vdups(UINT16_ALL_BITS, dtypes.uint16, mask=full16)
        hist0 = l0
        hist1 = l1
        candidate0 = rr.vselect(
            rr.varange(0, dtypes.uint16),
            all_ones,
            cond_mask=rr.vge(hist0, next_k, mask=full16),
        )
        candidate1 = rr.vselect(
            rr.varange(128, dtypes.uint16),
            all_ones,
            cond_mask=rr.vge(hist1, next_k, mask=full16),
        )
        low_target = rr.vreduce_min(
            rr.vmin(candidate0, candidate1, mask=full16), mask=full16
        )
        full16 = rr.update_mask(128, elem_bits=16)[0]
        low_target = rr.vdup(low_target, mask=full16)
        kth = rr.vbitwise_or(
            rr.vshl(high_target, 8, mask=full16), low_target, mask=full16
        )
        if const_expr(threshold_out is not None):
            rr.vstore(threshold_out, 0, kth, rr.update_mask(1, elem_bits=16)[0])
        self._collect_threshold_indices_vf(sort_key, valid_len, tmp_idx, kth)
        rr.vmem_bar("vst_vld")

    @jit
    def _scan_high_histogram_vf(self, sort_key, valid_len):
        full16 = rr.update_mask(128, elem_bits=16)[0]
        h0c0 = rr.vdups(0, dtypes.uint16, mask=full16)
        h1c0 = rr.vdups(0, dtypes.uint16, mask=full16)
        h0c1 = rr.vdups(0, dtypes.uint16, mask=full16)
        h1c1 = rr.vdups(0, dtypes.uint16, mask=full16)
        h0c2 = rr.vdups(0, dtypes.uint16, mask=full16)
        h1c2 = rr.vdups(0, dtypes.uint16, mask=full16)
        h0c3 = rr.vdups(0, dtypes.uint16, mask=full16)
        h1c3 = rr.vdups(0, dtypes.uint16, mask=full16)
        full8 = rr.update_mask(256, elem_bits=8)[0]
        for block in dsl_range(valid_len // 1024, unroll=1):
            low, high = rr.vload_deinterleave(sort_key, block * 1024 + 0, width="b8")
            h0c0 = rr.vhistogram_accumulate(h0c0, high, mask=full8, bin=0)
            h1c0 = rr.vhistogram_accumulate(h1c0, high, mask=full8, bin=1)
            low, high = rr.vload_deinterleave(sort_key, block * 1024 + 256, width="b8")
            h0c1 = rr.vhistogram_accumulate(h0c1, high, mask=full8, bin=0)
            h1c1 = rr.vhistogram_accumulate(h1c1, high, mask=full8, bin=1)
            low, high = rr.vload_deinterleave(sort_key, block * 1024 + 512, width="b8")
            h0c2 = rr.vhistogram_accumulate(h0c2, high, mask=full8, bin=0)
            h1c2 = rr.vhistogram_accumulate(h1c2, high, mask=full8, bin=1)
            low, high = rr.vload_deinterleave(sort_key, block * 1024 + 768, width="b8")
            h0c3 = rr.vhistogram_accumulate(h0c3, high, mask=full8, bin=0)
            h1c3 = rr.vhistogram_accumulate(h1c3, high, mask=full8, bin=1)
        for chunk in dsl_range(
            valid_len // 1024 * 4, (valid_len + 255) // 256, unroll=1
        ):
            mask8 = rr.update_mask(valid_len - chunk * 256, elem_bits=8)[0]
            low, high = rr.vload_deinterleave(sort_key, chunk * 256, width="b8")
            h0c0 = rr.vhistogram_accumulate(h0c0, high, mask=mask8, bin=0)
            h1c0 = rr.vhistogram_accumulate(h1c0, high, mask=mask8, bin=1)
        histogram_low_reg = rr.vadd(h0c0, h0c1, mask=full16)
        histogram_low_reg = rr.vadd(histogram_low_reg, h0c2, mask=full16)
        histogram_low_reg = rr.vadd(histogram_low_reg, h0c3, mask=full16)
        histogram_high_reg = rr.vadd(h1c0, h1c1, mask=full16)
        histogram_high_reg = rr.vadd(histogram_high_reg, h1c2, mask=full16)
        histogram_high_reg = rr.vadd(histogram_high_reg, h1c3, mask=full16)
        return (histogram_low_reg, histogram_high_reg)

    @jit
    def _build_merge(
        self,
        current_trunk,
        history_key,
        merge_key,
        current_len,
        current_len_pad,
        history_prefix=None,
    ):
        """Assemble a trunk, optionally preceded by its retained TopK."""
        if history_prefix is None:
            history_prefix = self.topk_pad
        with vf(mode="simd"):
            mask16 = rr.update_mask(128, elem_bits=16)[0]
            for chunk in dsl_range((history_prefix + 127) // 128, unroll=1):
                offset = chunk * 128
                rr.vstore(
                    merge_key,
                    offset,
                    rr.vload(history_key, offset),
                    mask16,
                )
            current_chunk_count = (current_len + 127) // 128
            zero = rr.vdups(0, dtypes.uint16, mask=mask16)
            for chunk in dsl_range(current_chunk_count, unroll=1):
                source_offset = chunk * 128
                count = current_len - source_offset
                current_mask = rr.update_mask(count, elem_bits=16)[0]
                selected = rr.vselect(
                    rr.vload(current_trunk, source_offset), zero, cond_mask=current_mask
                )
                rr.vstore(merge_key, history_prefix + source_offset, selected, mask16)
            for chunk in dsl_range(
                current_chunk_count, (current_len_pad + 127) // 128, unroll=1
            ):
                rr.vstore(merge_key, history_prefix + chunk * 128, zero, mask16)
            rr.vmem_bar("vst_vld")

    @jit
    def _load_merge(
        self, gm_keys, history, merge, source_offset, history_prefix, length
    ):
        copied = ((length + 127) // 128) * 128
        if not self.pipelined:
            merge_slot = merge.next_write()
            mem_copy(
                reinterpret(
                    _offset_view(merge_slot, (0, history_prefix)),
                    layout=make_layout((1, copied)),
                ),
                gm_keys[0:1, source_offset : source_offset + copied],
            )
            with vf(mode="simd"):
                mask = rr.update_mask(128, elem_bits=16)[0]
                for chunk in dsl_range(history_prefix // 128, unroll=1):
                    rr.vstore(
                        merge_slot, chunk * 128, rr.vload(history, chunk * 128), mask
                    )
                rr.vmem_bar("vst_vld")
        elif history_prefix == 0:
            # Common case: no retained history -> load the trunk straight into
            # the double-buffered merge RingBuffer (async, no extra V copy).
            mem_copy(
                reinterpret(merge.next_write(), layout=make_layout((1, copied))),
                gm_keys[0:1, source_offset : source_offset + copied],
            )
        else:
            mem_copy(
                reinterpret(
                    self.workspace.current_trunk.next_write(),
                    layout=make_layout((1, copied)),
                ),
                gm_keys[0:1, source_offset : source_offset + copied],
            )
            self._build_merge(
                self.workspace.current_trunk.next_read(),
                history,
                merge.next_write(),
                length,
                copied,
                history_prefix,
            )

    @jit
    def _find_indices(
        self,
        sort_key,
        valid_len,
        tmp_idx,
        histogram,
        high_target_carrier,
        threshold_out=None,
    ):
        with vf(mode="simd"):
            self._find_indices_vf(
                sort_key,
                valid_len,
                tmp_idx,
                histogram,
                high_target_carrier,
                threshold_out,
            )

    @jit
    def _find_indices_vf(
        self,
        sort_key,
        valid_len,
        tmp_idx,
        histogram,
        high_target_carrier,
        threshold_out=None,
    ):
        histogram_low_reg, histogram_high_reg = self._scan_high_histogram_vf(
            sort_key, valid_len
        )
        self._select_from_prefix_hist_vf(
            sort_key,
            valid_len,
            tmp_idx,
            histogram_low_reg,
            histogram_high_reg,
            threshold_out,
        )

    @jit
    def _find_indices_cached(
        self, sort_key, valid_len, tmp_idx, histogram, high_target_carrier
    ):
        with vf(mode="simd"):
            rr.update_mask(128, elem_bits=16)[0]
            histogram_low_reg = rr.vload(histogram, 0)
            histogram_high_reg = rr.vload(histogram, 128)

            self._select_from_prefix_hist_vf(
                sort_key, valid_len, tmp_idx, histogram_low_reg, histogram_high_reg
            )

    @jit
    def _find_indices_dispatch(
        self,
        sort_key,
        valid_len,
        tmp_idx,
        histogram,
        high_target_carrier,
        cached_hist,
        hist_row,
    ):
        if cached_hist:
            self._find_indices_dispatch_cached(
                sort_key, valid_len, tmp_idx, histogram, high_target_carrier, hist_row
            )
        else:
            self._find_indices_dispatch_scanned(
                sort_key, valid_len, tmp_idx, histogram, high_target_carrier, hist_row
            )

    @jit
    def _find_indices_dispatch_cached(
        self, sort_key, valid_len, tmp_idx, histogram, high_target_carrier, hist_row
    ):
        with vf(mode="simd"):
            full16 = rr.update_mask(128, elem_bits=16)[0]
            histogram_low_reg = rr.vdups(0, dtypes.uint16, mask=full16)
            histogram_high_reg = rr.vdups(0, dtypes.uint16, mask=full16)
            histogram_low_reg = rr.vadd(
                rr.vload(self.hist_slot0, hist_row * 512),
                rr.vload(self.hist_slot1, hist_row * 512),
                mask=full16,
            )
            histogram_high_reg = rr.vadd(
                rr.vload(self.hist_slot0, hist_row * 512 + 256),
                rr.vload(self.hist_slot1, hist_row * 512 + 256),
                mask=full16,
            )

            self._select_from_prefix_hist_vf(
                sort_key, valid_len, tmp_idx, histogram_low_reg, histogram_high_reg
            )

    @jit
    def _find_indices_dispatch_scanned(
        self, sort_key, valid_len, tmp_idx, histogram, high_target_carrier, hist_row
    ):
        with vf(mode="simd"):
            histogram_low_reg, histogram_high_reg = self._scan_high_histogram_vf(
                sort_key, valid_len
            )

            self._select_from_prefix_hist_vf(
                sort_key, valid_len, tmp_idx, histogram_low_reg, histogram_high_reg
            )

    @jit
    def _select_pass(
        self,
        sort_key,
        valid_len,
        history_prefix,
        current_global_base,
        tmp_idx,
        histogram,
        high_target_carrier,
        history_idx_in,
        history_idx_out,
        next_history_key,
    ):
        """Run one ASC radix pass over one trunk or history+trunk."""
        self._find_indices(sort_key, valid_len, tmp_idx, histogram, high_target_carrier)

        with vf(mode="simd"):
            # ASC FindValueOutputVFImpl keeps the compacted positions as
            # uint16 and gathers 128 values per iteration.  Do not reuse the
            # uint16->uint32 unpack used by FindRealIndexVFImpl below: that
            # unpack exposes only 64 lanes and breaks value/index pairing
            # once history is merged with another trunk.
            for chunk in dsl_range(ceil_div(self.topk, 128), unroll=1):
                count = self.topk - chunk * 128
                mask16 = rr.update_mask(count, elem_bits=16)[0]
                pos16 = rr.vload(tmp_idx, chunk * 128)
                selected_key = rr.vgather(sort_key, pos16, mask=mask16)
                rr.vstore(
                    next_history_key,
                    chunk * 128,
                    selected_key,
                    mask16,
                )

            # ASC FindRealIndexVFImpl loads the same compacted positions with
            # DIST_UNPACK_B16, yielding 64 uint32 lanes for history/current
            # index mapping.  Keep this separate from the value gather above.
            rr.update_mask(64, elem_bits=32)[0]
            for chunk in dsl_range(ceil_div(self.topk, 64), unroll=1):
                count = self.topk - chunk * 64
                mask32 = rr.update_mask(count, elem_bits=32)[0]
                pos16 = rr.vload(tmp_idx, chunk * 64)
                pos32 = rr.vunpack(pos16, dtypes.uint32, part="lower")
                is_current = rr.vges(pos32, history_prefix, mask=mask32)
                is_history = rr.vlts(pos32, history_prefix, mask=mask32)
                old_idx = rr.vgather(history_idx_in, pos32, mask=is_history)
                current_idx = rr.vadds(
                    rr.vreinterpret(pos32, dtypes.int32),
                    current_global_base - history_prefix,
                    mask=mask32,
                )
                global_idx = rr.vselect(current_idx, old_idx, cond_mask=is_current)
                rr.vstore(history_idx_out, chunk * 64, global_idx, mask32)
            rr.vmem_bar("vst_vld")

    @jit
    def _commit_history(self, next_idx, history_idx, next_history_key, history_key):
        with vf(mode="simd"):
            copy_mask32 = rr.update_mask(64, elem_bits=32)[0]
            copy_mask16 = rr.update_mask(128, elem_bits=16)[0]
            for chunk in dsl_range(ceil_div(self.topk, 64), unroll=1):
                rr.vstore(
                    history_idx,
                    chunk * 64,
                    rr.vload(next_idx, chunk * 64),
                    copy_mask32,
                )
            for chunk in dsl_range(ceil_div(self.topk, 128), unroll=1):
                rr.vstore(
                    history_key,
                    chunk * 128,
                    rr.vload(next_history_key, chunk * 128),
                    copy_mask16,
                )
            rr.vmem_bar("vst_vld")

    @jit
    def store_empty_row(
        self,
        gm_sparse_indices: Tensor,
        gm_selected_value_bits: Tensor,
    ):
        """Store the public empty-TopK representation without scanning GM."""
        idx_write = self.output_idx_stage.next_write()
        val_write = self.output_value_bits_stage.next_write()
        with vf(mode="simd"):
            for chunk in dsl_range(ceil_div(self.topk, 64), unroll=1):
                count = self.topk - chunk * 64
                mask32 = rr.update_mask(count, elem_bits=32)[0]
                mask16 = rr.mask_and(
                    rr.update_mask(count, elem_bits=16)[0],
                    rr.update_mask(64, elem_bits=16)[0],
                    exec_mask=rr.update_mask(64, elem_bits=16)[0],
                )
                rr.vstore(
                    idx_write,
                    chunk * 64,
                    rr.vdups(-1, dtypes.int32, mask=mask32),
                    mask32,
                )
                rr.vstore(
                    val_write,
                    chunk * 64,
                    rr.vdups(BF16_NEG_INF_BITS, dtypes.uint16, mask=mask16),
                    mask16,
                )
            rr.vmem_bar("vst_vld")
        mem_copy(gm_sparse_indices, self.output_idx_stage.next_read())

        mem_copy(gm_selected_value_bits, self.output_value_bits_stage.next_read())

    @jit
    def select_candidate_row(
        self,
        gm_sort_key,
        total_tokens,
        valid_count,
        gm_indices,
        gm_values,
        index_base,
        need_values,
    ):
        if need_values == 0 and valid_count <= self.topk:
            out = self.output_idx_stage.next_write()
            with vf(mode="simd"):
                for chunk in dsl_range(ceil_div(self.topk, 64), unroll=1):
                    mask = rr.update_mask(self.topk - chunk * 64, elem_bits=32)[0]
                    rank = rr.varange(chunk * 64, dtypes.int32)
                    valid = rr.vlts(rank, valid_count, mask=mask)
                    indices = rr.vadds(rank, index_base, mask=mask)
                    rr.vstore(
                        out,
                        chunk * 64,
                        rr.vselect(
                            indices,
                            rr.vdups(-1, dtypes.int32, mask=mask),
                            cond_mask=valid,
                        ),
                        mask,
                    )
                rr.vmem_bar("vst_vld")
            mem_copy(gm_indices, self.output_idx_stage.next_read())
        else:
            self.select_row(
                gm_sort_key,
                total_tokens,
                valid_count,
                gm_indices,
                gm_values,
                index_base,
                need_values,
            )

    @jit
    def select_row(
        self,
        gm_sort_key: Tensor,
        total_tokens,
        output_valid_count,
        gm_sparse_indices: Tensor,
        gm_selected_value_bits: Tensor,
        index_base,
        need_values=1,
        cached_hist=0,
        hist_row=0,
    ):
        if (
            need_values == 0
            and total_tokens >= self.topk
            and total_tokens <= self.trunk_len
        ):
            merge_key = self.workspace.merge_key
            self._load_merge(
                gm_sort_key,
                self.workspace.history_key,
                merge_key,
                dtypes.int64(0),
                dtypes.int64(0),
                total_tokens,
            )
            merge_read = merge_key.next_read()
            self._find_indices_dispatch(
                merge_read,
                total_tokens,
                self.workspace.tmp_idx,
                self.workspace.histogram,
                self.workspace.high_target_carrier,
                cached_hist,
                hist_row,
            )
            out = self.output_idx_stage.next_write()
            tmp = self.workspace.tmp_idx
            with vf(mode="simd"):
                self._emit_indices_only_vf(tmp, out, output_valid_count, index_base)
                rr.vmem_bar("vst_vld")
            mem_copy(gm_sparse_indices, self.output_idx_stage.next_read())
        elif (
            need_values != 0
            and total_tokens >= self.topk
            and total_tokens <= self.trunk_len
        ):
            self._select_row_single_trunk(
                gm_sort_key,
                total_tokens,
                output_valid_count,
                gm_sparse_indices,
                gm_selected_value_bits,
                index_base,
                cached_hist,
                hist_row,
            )
        else:
            self._select_row_values(
                gm_sort_key,
                total_tokens,
                output_valid_count,
                gm_sparse_indices,
                gm_selected_value_bits,
                index_base,
            )

    @jit
    def _select_row_single_trunk(
        self,
        gm_sort_key: Tensor,
        total_tokens,
        output_valid_count,
        gm_sparse_indices: Tensor,
        gm_selected_value_bits: Tensor,
        index_base,
        cached_hist=0,
        hist_row=0,
    ):
        """PV-C8 port: single-trunk fast path WITH values.

        When the entire input fits in one trunk (total_tokens <= 16384) and
        this is the first (and only) pass, history stays empty: skip its
        zero-init, skip the next_history_key/next_idx round-trip, and gather
        value bits directly from merge_key at the squeeze positions.
        Mathematically identical to _select_row_values for this case because
        history_prefix == 0 makes the position mapping an identity.
        """
        merge_key = self.workspace.merge_key
        tmp_idx = self.workspace.tmp_idx
        histogram = self.workspace.histogram
        high_target_carrier = self.workspace.high_target_carrier
        self._load_merge(
            gm_sort_key,
            self.workspace.history_key,
            merge_key,
            dtypes.int64(0),
            dtypes.int64(0),
            total_tokens,
        )
        merge_read = merge_key.next_read()
        self._find_indices_dispatch(
            merge_read,
            total_tokens,
            tmp_idx,
            histogram,
            high_target_carrier,
            cached_hist,
            hist_row,
        )
        out_idx = self.output_idx_stage.next_write()
        out_val = self.output_value_bits_stage.next_write()
        self._emit_single_trunk(
            index_base, merge_read, out_idx, out_val, output_valid_count, tmp_idx
        )
        mem_copy(gm_sparse_indices, self.output_idx_stage.next_read())
        mem_copy(gm_selected_value_bits, self.output_value_bits_stage.next_read())

    @jit
    def _select_row_cached_ub(
        self,
        sort_key: Tensor,
        total_tokens,
        output_valid_count,
        gm_sparse_indices: Tensor,
        gm_selected_value_bits: Tensor,
        index_base,
        cached_hist=0,
        hist_row=0,
    ):
        """PV-C8 port: single-trunk fast path WITH values.

        When the entire input fits in one trunk (total_tokens <= 16384) and
        this is the first (and only) pass, history stays empty: skip its
        zero-init, skip the next_history_key/next_idx round-trip, and gather
        value bits directly from merge_key at the squeeze positions.
        Mathematically identical to _select_row_values for this case because
        history_prefix == 0 makes the position mapping an identity.
        """
        merge_key = sort_key
        tmp_idx = self.cache_tmp
        histogram = self.workspace.histogram
        high_target_carrier = self.workspace.high_target_carrier
        self._find_indices_dispatch(
            merge_key,
            total_tokens,
            tmp_idx,
            histogram,
            high_target_carrier,
            cached_hist,
            hist_row,
        )
        out_idx = self.cache_output_idx_stage.next_write()
        out_val = self.cache_output_value_bits_stage.next_write()
        with vf(mode="simd"):
            for chunk in dsl_range(ceil_div(self.topk, 64), unroll=1):
                count = self.topk - chunk * 64
                mask32 = rr.update_mask(count, elem_bits=32)[0]
                mask16 = rr.update_mask(count, elem_bits=16)[0]
                pos16 = rr.vload(tmp_idx, chunk * 64)
                pos32 = rr.vunpack(pos16, dtypes.uint32, part="lower")
                keys = rr.vgather(merge_key, pos16, mask=mask16)
                indices, value_bits = self._ranked_output_vf(
                    keys,
                    pos32,
                    chunk,
                    mask16,
                    mask32,
                    index_base,
                    output_valid_count,
                )
                rr.vstore(out_idx, chunk * 64, indices, mask32)
                rr.vstore(out_val, chunk * 64, value_bits, mask16)
            rr.vmem_bar("vst_vld")
        mem_copy(gm_sparse_indices, self.cache_output_idx_stage.next_read())
        mem_copy(gm_selected_value_bits, self.cache_output_value_bits_stage.next_read())

    @jit
    def _select_row_values(
        self,
        gm_sort_key: Tensor,
        total_tokens,
        output_valid_count,
        gm_sparse_indices: Tensor,
        gm_selected_value_bits: Tensor,
        index_base,
    ):
        # Rank masks use uint16 lanes; clamp the count before narrowing.
        if output_valid_count > self.topk:
            output_valid_count = self.topk
        merge_key = self.workspace.merge_key
        tmp_idx = self.workspace.tmp_idx
        histogram = self.workspace.histogram
        high_target_carrier = self.workspace.high_target_carrier
        history_key_storage = self.workspace.history_key
        history_key = history_key_storage
        next_history_key = self.workspace.next_history_key
        history_idx_storage = self.workspace.history_idx
        history_idx = history_idx_storage
        next_idx = self.workspace.next_idx
        output_idx_stage = self.output_idx_stage.next_write()
        out_value_bits = self.output_value_bits_stage.next_write()

        # Finite signed BF16 scores have positive sortable keys; zero marks
        # invalid tokens and empty history slots.
        with vf(mode="simd"):
            init_mask16 = rr.update_mask(128, elem_bits=16)[0]
            init_mask32 = rr.update_mask(64, elem_bits=32)[0]
            zero16 = rr.vdups(0, dtypes.uint16, mask=init_mask16)
            zero32 = rr.vdups(0, dtypes.int32, mask=init_mask32)
            for chunk in dsl_range(ceil_div(self.topk_pad * 2, 128), unroll=1):
                rr.vstore(history_key, chunk * 128, zero16, init_mask16)
            for chunk in dsl_range(ceil_div(self.topk_pad * 2, 64), unroll=1):
                rr.vstore(history_idx, chunk * 64, zero32, init_mask32)
            rr.vmem_bar("vst_vld")

        full_trunk_count = total_tokens // self.trunk_len
        tail_len = total_tokens % self.trunk_len
        for pass_idx in dsl_range(full_trunk_count, unroll=1):
            parity = pass_idx % 2
            history_key = tile_slice(
                history_key_storage, (1, self.topk_pad), (parity, 0)
            )
            next_history_key = tile_slice(
                history_key_storage, (1, self.topk_pad), (1 - parity, 0)
            )
            history_idx = tile_slice(
                history_idx_storage, (1, self.topk_pad), (parity, 0)
            )
            next_idx = tile_slice(
                history_idx_storage, (1, self.topk_pad), (1 - parity, 0)
            )
            trunk_offset = pass_idx * self.trunk_len
            history_prefix = dtypes.int64((0 if pass_idx == 0 else self.topk_pad))
            self._load_merge(
                gm_sort_key,
                history_key,
                merge_key,
                trunk_offset,
                history_prefix,
                self.trunk_len,
            )
            self._select_pass(
                merge_key.next_read(),
                history_prefix + self.trunk_len,
                history_prefix,
                trunk_offset,
                tmp_idx,
                histogram,
                high_target_carrier,
                history_idx,
                next_idx,
                next_history_key,
            )

        if tail_len > 0:
            parity = full_trunk_count % 2
            history_key = tile_slice(
                history_key_storage, (1, self.topk_pad), (parity, 0)
            )
            next_history_key = tile_slice(
                history_key_storage, (1, self.topk_pad), (1 - parity, 0)
            )
            history_idx = tile_slice(
                history_idx_storage, (1, self.topk_pad), (parity, 0)
            )
            next_idx = tile_slice(
                history_idx_storage, (1, self.topk_pad), (1 - parity, 0)
            )
            tail_offset = full_trunk_count * self.trunk_len
            history_prefix = dtypes.int64(self.topk_pad)
            if full_trunk_count == 0 and tail_len >= self.topk:
                history_prefix = dtypes.int64(0)
            ((tail_len + 255) // 256) * 256
            self._load_merge(
                gm_sort_key,
                history_key,
                merge_key,
                tail_offset,
                history_prefix,
                tail_len,
            )
            self._select_pass(
                merge_key.next_read(),
                history_prefix + tail_len,
                history_prefix,
                tail_offset,
                tmp_idx,
                histogram,
                high_target_carrier,
                history_idx,
                next_idx,
                next_history_key,
            )

        final_slot = (full_trunk_count + dtypes.int64((1 if tail_len > 0 else 0))) % 2
        final_idx = tile_slice(history_idx_storage, (1, self.topk_pad), (final_slot, 0))
        history_key = tile_slice(
            history_key_storage, (1, self.topk_pad), (final_slot, 0)
        )
        self._emit_selected_values(
            final_idx,
            history_key,
            index_base,
            out_value_bits,
            output_idx_stage,
            output_valid_count,
        )
        mem_copy(gm_sparse_indices, self.output_idx_stage.next_read())
        mem_copy(gm_selected_value_bits, self.output_value_bits_stage.next_read())

    @jit
    def _find_indices_dispatch_byte(
        self, sort_key, valid_len, tmp_idx, histogram, high_target_carrier, hist_row
    ):
        with vf(mode="simd"):
            m = rr.update_mask(128, elem_bits=16)[0]
            h0 = rr.vadd(
                rr.vload(self.hist_slot0, hist_row * 512),
                rr.vload(self.hist_slot1, hist_row * 512),
                mask=m,
            )
            h1 = rr.vadd(
                rr.vload(self.hist_slot0, hist_row * 512 + 256),
                rr.vload(self.hist_slot1, hist_row * 512 + 256),
                mask=m,
            )
            sentinel = rr.vdups(UINT16_ALL_BITS, dtypes.uint16, mask=m)
            c0 = rr.vselect(
                rr.varange(0, dtypes.uint16),
                sentinel,
                cond_mask=rr.vges(h0, valid_len - self.topk + 1, mask=m),
            )
            c1 = rr.vselect(
                rr.varange(128, dtypes.uint16),
                sentinel,
                cond_mask=rr.vges(h1, valid_len - self.topk + 1, mask=m),
            )
            high = rr.vdup(rr.vreduce_min(rr.vmin(c0, c1, mask=m), mask=m), mask=m)
            zero = rr.vdups(0, dtypes.uint16, mask=m)
            lane = rr.vbitwise_and(high, rr.vdups(127, dtypes.uint16, mask=m), mask=m)
            upto = rr.vselect(
                rr.vgather_reg(h0, lane),
                rr.vgather_reg(h1, lane),
                cond_mask=rr.vlts(high, 128, mask=m),
            )
            prev = rr.vselect(
                zero,
                rr.vsub(high, rr.vdups(1, dtypes.uint16, mask=m), mask=m),
                cond_mask=rr.veqs(high, 0, mask=m),
            )
            lane = rr.vbitwise_and(prev, rr.vdups(127, dtypes.uint16, mask=m), mask=m)
            before = rr.vselect(
                rr.vgather_reg(h0, lane),
                rr.vgather_reg(h1, lane),
                cond_mask=rr.vlts(prev, 128, mask=m),
            )
            before = rr.vselect(zero, before, cond_mask=rr.veqs(high, 0, mask=m))
            gt = rr.vsub(rr.vdups(valid_len, dtypes.uint16, mask=m), upto, mask=m)
            eq = rr.vsub(upto, before, mask=m)
            rr.vstore_first(
                high_target_carrier,
                0,
                rr.vreinterpret(
                    rr.vunpack(gt, dtypes.uint32, part="lower"), dtypes.int32
                ),
            )
            rr.vstore_first(
                high_target_carrier,
                1,
                rr.vreinterpret(
                    rr.vunpack(high, dtypes.uint32, part="lower"), dtypes.int32
                ),
            )
            rr.vstore_first(
                high_target_carrier,
                2,
                rr.vreinterpret(
                    rr.vunpack(eq, dtypes.uint32, part="lower"), dtypes.int32
                ),
            )
            upper = rr.vgather_reg(h1, rr.vdups(126, dtypes.uint16, mask=m))
            overflow = rr.vsub(
                rr.vdups(valid_len, dtypes.uint16, mask=m), upper, mask=m
            )
            rr.vstore_first(
                high_target_carrier,
                3,
                rr.vreinterpret(
                    rr.vunpack(overflow, dtypes.uint32, part="lower"), dtypes.int32
                ),
            )
            rr.vstore_first(
                high_target_carrier,
                31,
                rr.vdups(0, dtypes.int32, mask=rr.update_mask(1, elem_bits=32)[0]),
            )
            rr.vmem_bar("vst_vld")
        gt = dtypes.int32(high_target_carrier[0])
        high = dtypes.int32(high_target_carrier[1])
        eq = dtypes.int32(high_target_carrier[2])
        dtypes.int32(high_target_carrier[32 + hist_row * 8])
        if (
            high > 0
            and high < 255
            and valid_len >= self.topk
            and gt < self.topk
            and gt + eq <= (tmp_idx.shape[0] // 256) * 128
        ):
            self._byte_select(sort_key, valid_len, tmp_idx, high, gt + eq)
        else:
            with vf(mode="simd"):
                rr.vstore_first(
                    high_target_carrier,
                    31,
                    rr.vdups(1, dtypes.int32, mask=rr.update_mask(1, elem_bits=32)[0]),
                )
                rr.vmem_bar("vst_vld")

    @jit
    def _byte_compact(self, sort_key, valid_len, output, kth):
        with vf(mode="simd"):
            cursor = rr.vstore_unalign_begin(output)
            for chunk in dsl_range((valid_len + 127) // 128):
                m = rr.update_mask(valid_len - chunk * 128, elem_bits=16)[0]
                idx = rr.varange(chunk * 128, dtypes.uint16)
                keys = rr.vload_unpack(sort_key, chunk * 128, unpack_mode="unpack2")
                selected = rr.vsqueeze_and_storeunalign_init(
                    idx, mask=rr.vges(keys, kth, mask=m)
                )
                rr.vsqueeze_and_storeunalign(output, 0, selected, cursor)
            rr.vstore_unalign_post(output, 0, cursor)
            rr.vmem_bar("vst_vld")

    @jit
    def _byte_select(self, sort_key, valid_len, tmp_idx, kth, retained):
        candidates = reinterpret(
            tmp_idx,
            offset=(tmp_idx.shape[0] // 2) * 2,
            layout=make_layout((tmp_idx.shape[0] // 2,)),
        )
        if retained == self.topk:
            self._byte_compact(sort_key, valid_len, tmp_idx, kth)
        else:
            self._byte_compact(sort_key, valid_len, candidates, kth)
            with vf(mode="simd"):
                cursor = rr.vstore_unalign_begin(tmp_idx)
                for chunk in dsl_range((retained + 127) // 128):
                    m = rr.update_mask(retained - chunk * 128, elem_bits=16)[0]
                    idx = rr.vload(candidates, chunk * 128)
                    keys = rr.vreinterpret_lanes(
                        rr.vgather(sort_key, idx, mask=m), dtypes.uint16
                    )
                    selected = rr.vsqueeze_and_storeunalign_init(
                        idx, mask=rr.vgts(keys, kth, mask=m)
                    )
                    rr.vsqueeze_and_storeunalign(tmp_idx, 0, selected, cursor)
                for chunk in dsl_range((retained + 127) // 128):
                    m = rr.update_mask(retained - chunk * 128, elem_bits=16)[0]
                    idx = rr.vload(candidates, chunk * 128)
                    keys = rr.vreinterpret_lanes(
                        rr.vgather(sort_key, idx, mask=m), dtypes.uint16
                    )
                    selected = rr.vsqueeze_and_storeunalign_init(
                        idx, mask=rr.veqs(keys, kth, mask=m)
                    )
                    rr.vsqueeze_and_storeunalign(tmp_idx, 0, selected, cursor)
                rr.vstore_unalign_post(tmp_idx, 0, cursor)
                rr.vmem_bar("vst_vld")

    @jit
    def _select_row_byte(
        self,
        gm_keys,
        cached,
        overflow_cache,
        length,
        out_ids,
        out_bits,
        index_base,
        hist_row,
    ):
        if is_aiv():
            idx_write = self.cache_output_idx_stage.next_write()
            bits_write = self.cache_output_value_bits_stage.next_write()
            carrier = self.workspace.high_target_carrier
            failed = dtypes.int32(carrier[30])
            if failed == 0:
                self._find_indices_dispatch_byte(
                    cached,
                    length,
                    self.byte_tmp,
                    self.workspace.histogram,
                    carrier,
                    hist_row,
                )
                failed = dtypes.int32(carrier[31])
            if failed == 0:
                base = dtypes.int32(carrier[32 + hist_row * 8])
                with vf(mode="simd"):
                    all16 = rr.update_mask(128, elem_bits=16)[0]
                    offset = rr.vdups(0, dtypes.uint16, mask=all16)
                    for chunk in dsl_range(8):
                        m16 = rr.update_mask(64, elem_bits=16)[0]
                        m32 = rr.update_mask(64, elem_bits=32)[0]
                        idx = rr.vload(self.byte_tmp, chunk * 64)
                        code = rr.vreinterpret_lanes(
                            rr.vgather(cached, idx, mask=m16), dtypes.uint16
                        )
                        over = rr.veqs(code, 255, mask=m16)
                        prefix = rr.vsqueeze(over, dtypes.uint16)
                        position = rr.vadd(prefix, offset, mask=m16)
                        exact = rr.vgather(overflow_cache, position, mask=over)
                        keys16 = rr.vselect(
                            exact, rr.vadds(code, base, mask=m16), cond_mask=over
                        )
                        number = rr.vselect(
                            rr.vdups(1, dtypes.uint16, mask=m16),
                            rr.vdups(0, dtypes.uint16, mask=m16),
                            cond_mask=over,
                        )
                        offset = rr.vadd(
                            offset,
                            rr.vdup(
                                rr.vreduce_max(
                                    rr.vadd(prefix, number, mask=m16), mask=m16
                                ),
                                mask=all16,
                            ),
                            mask=all16,
                        )
                        values = _decode_cached_score_bits(keys16, m16)
                        indices = rr.vreinterpret(
                            rr.vunpack(idx, dtypes.uint32, part="lower"), dtypes.int32
                        )
                        rr.vstore(
                            idx_write,
                            chunk * 64,
                            rr.vadds(indices, index_base, mask=m32),
                            m32,
                        )
                        rr.vstore(bits_write, chunk * 64, values, m16)
                    rr.vmem_bar("vst_vld")
            else:
                with vf(mode="simd"):
                    mask = rr.update_mask(1, elem_bits=32)[0]
                    rr.vstore_first(carrier, 30, rr.vdups(1, dtypes.int32, mask=mask))
                    rr.vmem_bar("vst_vld")
                copied = ((length + 127) // 128) * 128
                mem_copy(
                    reinterpret(
                        self.byte_fallback_keys,
                        layout=make_layout((1, copied), stride=(BYTE_CACHE_TOKENS, 1)),
                    ),
                    gm_keys[0:1, 0:copied],
                )
                self._find_indices_dispatch_scanned(
                    self.byte_fallback_keys,
                    length,
                    self.byte_fallback_tmp,
                    self.workspace.histogram,
                    carrier,
                    hist_row,
                )
                with vf(mode="simd"):
                    for chunk in dsl_range(8):
                        m16 = rr.update_mask(64, elem_bits=16)[0]
                        m32 = rr.update_mask(64, elem_bits=32)[0]
                        idx = rr.vload(self.byte_fallback_tmp, chunk * 64)
                        keys16 = rr.vgather(self.byte_fallback_keys, idx, mask=m16)
                        values = _decode_cached_score_bits(keys16, m16)
                        indices = rr.vreinterpret(
                            rr.vunpack(idx, dtypes.uint32, part="lower"), dtypes.int32
                        )
                        rr.vstore(
                            idx_write,
                            chunk * 64,
                            rr.vadds(indices, index_base, mask=m32),
                            m32,
                        )
                        rr.vstore(bits_write, chunk * 64, values, m16)
                    rr.vmem_bar("vst_vld")
            mem_copy(out_ids, self.cache_output_idx_stage.next_read())
            mem_copy(out_bits, self.cache_output_value_bits_stage.next_read())
            if hist_row == 2:
                vec_sync_all()

    @jit
    def _emit_single_trunk(
        self, index_base, merge_read, out_idx, out_val, output_valid_count, tmp_idx
    ):
        with vf(mode="simd"):
            for chunk in dsl_range(ceil_div(self.topk, 64), unroll=1):
                count = self.topk - chunk * 64
                mask32 = rr.update_mask(count, elem_bits=32)[0]
                mask16 = rr.update_mask(count, elem_bits=16)[0]
                pos16 = rr.vload(tmp_idx, chunk * 64)
                pos32 = rr.vunpack(pos16, dtypes.uint32, part="lower")
                keys = rr.vgather(merge_read, pos16, mask=mask16)
                indices, value_bits = self._ranked_output_vf(
                    keys,
                    pos32,
                    chunk,
                    mask16,
                    mask32,
                    index_base,
                    output_valid_count,
                )
                rr.vstore(out_idx, chunk * 64, indices, mask32)
                rr.vstore(out_val, chunk * 64, value_bits, mask16)
            rr.vmem_bar("vst_vld")

    @jit
    def _emit_selected_values(
        self,
        final_idx,
        history_key,
        index_base,
        out_value_bits,
        output_idx_stage,
        output_valid_count,
    ):
        with vf(mode="simd"):
            for chunk in dsl_range(ceil_div(self.topk, 64), unroll=1):
                count = self.topk - chunk * 64
                mask32 = rr.update_mask(count, elem_bits=32)[0]
                mask16 = rr.mask_and(
                    rr.update_mask(count, elem_bits=16)[0],
                    rr.update_mask(64, elem_bits=16)[0],
                    exec_mask=rr.update_mask(64, elem_bits=16)[0],
                )
                indices = rr.vadds(
                    rr.vload(final_idx, chunk * 64), index_base, mask=mask32
                )
                keys = rr.vload(history_key, chunk * 64)
                value_bits = _decode_bf16_sort_key(keys, mask16)
                valid_rank32 = rr.vlts(
                    rr.varange(chunk * 64, dtypes.uint32),
                    output_valid_count,
                    mask=mask32,
                )
                valid_rank16 = rr.vlts(
                    rr.varange(chunk * 64, dtypes.uint16),
                    output_valid_count,
                    mask=mask16,
                )
                indices = rr.vselect(
                    indices,
                    rr.vdups(-1, dtypes.int32, mask=mask32),
                    cond_mask=valid_rank32,
                )
                value_bits = rr.vselect(
                    value_bits,
                    rr.vdups(BF16_NEG_INF_BITS, dtypes.uint16, mask=mask16),
                    cond_mask=valid_rank16,
                )
                rr.vstore(output_idx_stage, chunk * 64, indices, mask32)
                rr.vstore(out_value_bits, chunk * 64, value_bits, mask16)
            rr.vmem_bar("vst_vld")


class LdTopKSelector(QliRawTopKSelector):
    def __init__(self, topk, workspace, splits):
        self.ub_pool = workspace.ub_pool
        super().__init__(topk, workspace)
        self._init_ld_buffers(splits)

    @jit
    def merge_ld(
        self,
        partial_indices,
        partial_bits,
        gm_sparse_indices,
        gm_selected_value_bits,
        valid_length,
        output_offset=0,
        preloaded=False,
        need_values=1,
        emit_gm=1,
    ):
        trunk_read, ld_idx_read, ld_bits_read = self._prepare_ld_merge(
            partial_indices,
            partial_bits,
            valid_length,
            preloaded,
        )
        self.output_idx_stage.next_write()
        self.output_value_bits_stage.next_write()
        dtypes.int32(self.topk)
        if need_values != 0:
            self._emit_ld_values(
                trunk_read,
                ld_idx_read,
                ld_bits_read,
                output_offset,
                gm_sparse_indices,
                gm_selected_value_bits,
                emit_gm,
            )
        else:
            self._emit_ld_indices(
                trunk_read, ld_idx_read, output_offset, gm_sparse_indices, emit_gm
            )

    @jit
    def _emit_ld_indices(
        self, current_trunk, ld_indices, output_offset, gm_sparse_indices, emit_gm=1
    ):
        output_idx_stage = self.output_idx_stage.next_write()
        with vf(mode="simd"):
            for chunk in dsl_range(ceil_div(self.topk, 64), unroll=1):
                count = self.topk - chunk * 64
                mask32 = rr.update_mask(count, elem_bits=32)[0]
                mask16 = rr.mask_and(
                    rr.update_mask(count, elem_bits=16)[0],
                    rr.update_mask(64, elem_bits=16)[0],
                    exec_mask=rr.update_mask(64, elem_bits=16)[0],
                )
                positions16 = rr.vload(self.workspace.tmp_idx, chunk * 64)
                positions32 = rr.vunpack(positions16, dtypes.uint32, part="lower")
                indices = rr.vgather(ld_indices, positions32, mask=mask32)
                keys = rr.vgather(current_trunk, positions16, mask=mask16)
                valid16 = rr.vselect(
                    rr.vdups(1, dtypes.uint16, mask=mask16),
                    rr.vdups(0, dtypes.uint16, mask=mask16),
                    cond_mask=rr.vges(keys, 1, mask=mask16),
                )
                valid32 = rr.vunpack(valid16, dtypes.uint32, part="lower")
                indices = rr.vselect(
                    rr.vadds(indices, output_offset, mask=mask32),
                    rr.vdups(-1, dtypes.int32, mask=mask32),
                    cond_mask=rr.veqs(valid32, 1, mask=mask32),
                )
                rr.vstore(output_idx_stage, chunk * 64, indices, mask32)
            rr.vmem_bar("vst_vld")
        if emit_gm != 0:
            mem_copy(gm_sparse_indices, self.output_idx_stage.next_read())


class LdMergeStage:
    def __init__(self, topk, splits):
        self.topk = topk
        self.length = topk * splits
        self.workspace = QliRawTopKWorkspace(self.length, topk, merge_only=True)
        self.selector = LdTopKSelector(topk, self.workspace, splits)

    @jit
    def __call__(
        self,
        partial_indices: Tensor,
        partial_bits: Tensor,
        indices: Tensor,
        bits: Tensor,
        metadata: Tensor,
        rows,
        workers,
        cu_seqlens_q: Tensor,
        query_tile_rows,
        has_cu,
        output_offset_address=0,
        need_values=1,
    ):
        fd = dtypes.int64(LD_RECORD_START + get_block_idx() * 2 + get_subblock_id())
        if metadata[fd, 0] != 0:
            batch_idx = dtypes.int64(metadata[fd, 1])
            begin = dtypes.int64(0)
            if has_cu:
                begin = dtypes.int64(cu_seqlens_q[batch_idx])
            row_base = begin + dtypes.int64(metadata[fd, 2]) * query_tile_rows
            workspace_begin = dtypes.int64(metadata[fd, 3])
            part_count = dtypes.int64(metadata[fd, 4])
            m_begin = dtypes.int64(metadata[fd, 5])
            m_end = m_begin + dtypes.int64(metadata[fd, 6])
            for m in dsl_range(m_begin, m_end, unroll=1):
                row = row_base + m
                output_offset = dtypes.int32(0)
                if output_offset_address != 0:
                    offsets = make_tensor(
                        make_pointer(
                            dtypes.int32, dtypes.int64(output_offset_address), MemLoc.GM
                        ),
                        make_layout((rows, 1), stride=(1, 1)),
                    )
                    output_offset = offsets[row, 0]
                for first in dsl_range(
                    0, part_count, self.length // self.topk - 1, unroll=1
                ):
                    count = min(part_count - first, self.length // self.topk - 1)
                    carry = dtypes.int64((1 if first > 0 else 0))
                    ld_idx_write = self.selector.ld_indices.next_write()
                    ld_bits_write = self.selector.ld_bits.next_write()
                    if first > 0:
                        # Carry stays on-core: the previous round's output-stage
                        # slots are the sole producer of this round's carry.
                        mem_copy(
                            reinterpret(
                                ld_idx_write, layout=make_layout((1, self.topk))
                            ),
                            self.selector.output_idx_stage.next_read(),
                        )
                        mem_copy(
                            reinterpret(
                                ld_bits_write, layout=make_layout((1, self.topk))
                            ),
                            self.selector.output_value_bits_stage.next_read(),
                        )
                    for part in dsl_range(0, count, unroll=1):
                        source_row = (
                            workspace_begin + first + part
                        ) * query_tile_rows + m
                        if const_expr(self.topk % 16 != 0):
                            # Packed row offsets are not DMA aligned; keep these copies on the scalar unit.
                            for element in dsl_range(self.topk, unroll=1):
                                destination = (part + carry) * self.topk + element
                                ld_idx_write[0, destination] = partial_indices[
                                    source_row, element
                                ]
                                ld_bits_write[0, destination] = partial_bits[
                                    source_row, element
                                ]
                        else:
                            mem_copy(
                                tile_slice(
                                    ld_idx_write, (1, self.topk), (0, part + carry)
                                ),
                                tile_slice(
                                    partial_indices, (1, self.topk), (source_row, 0)
                                ),
                            )
                            mem_copy(
                                tile_slice(
                                    ld_bits_write, (1, self.topk), (0, part + carry)
                                ),
                                tile_slice(
                                    partial_bits, (1, self.topk), (source_row, 0)
                                ),
                            )
                    offset = dtypes.int32(
                        (output_offset if first + count == part_count else 0)
                    )
                    # Intermediate rounds emit into the UB output stages only;
                    # the final round drains to GM, so no GM write-read ordering
                    # is needed between rounds.
                    self.selector.merge_ld(
                        partial_indices,
                        partial_bits,
                        tile_slice(indices, (1, self.topk), (row, 0)),
                        tile_slice(bits, (1, self.topk), (row, 0)),
                        (count + carry) * self.topk,
                        offset,
                        True,
                        dtypes.int64(
                            (need_values if first + count == part_count else 1)
                        ),
                        dtypes.int64((1 if first + count == part_count else 0)),
                    )


# Candidate LD selectors keep their independent 16K merge scratch layout.
CANDIDATE_LD_TRUNK_LEN = 16384


class CandidateQliRawTopKWorkspace(QliRawTopKWorkspace):
    """UB scratch shared by sequential Sparse and Candidate TopK stages."""

    # 初始化候选块 TopK 工作区的配置与缓冲区。
    def __init__(
        self, max_trunk_len: int, max_topk: int, candidate_optimized: bool = False
    ):
        self.ub_pool = _UbPool()
        self.candidate_optimized = bool(candidate_optimized)
        self.max_trunk_len = int(max_trunk_len)
        self.max_topk = int(max_topk)
        self.topk_pad = ceil_div(self.max_topk, 256) * 256
        self.merge_len_pad = ceil_div(self.topk_pad + self.max_trunk_len, 128) * 128
        self.current_trunk = _ub_ring_buffer(
            self.ub_pool,
            MemLoc.UB,
            (1, self.max_trunk_len),
            dtypes.uint16,
            depth=1,
        )
        self.merge_key = _ub_ring_buffer(
            self.ub_pool,
            MemLoc.UB,
            (1, self.merge_len_pad),
            dtypes.uint16,
            depth=1,
        )
        self._allocate_index_scratch()
        self.history_idx = _ub_buffer(
            self.ub_pool,
            MemLoc.UB,
            (2, self.topk_pad),
            dtypes.int32,
        )
        self.next_idx = _ub_buffer(
            self.ub_pool,
            MemLoc.UB,
            (self.topk_pad,),
            dtypes.int32,
        )


class CandidateQliRawTopKSelector(QliRawTopKSelector):
    """ASC Arch35 uint16 radix TopK with in-AIV trunk/history merge.

    This is not LD/S2 core splitting.  One AIV processes a row sequentially in
    16K trunks.  The first trunk creates a TopK history; every later pass runs
    the same radix selector over ``history + current trunk`` and maps its local
    uint16 positions back to global int32 token indices.
    """

    # 初始化候选块 TopK 选择的配置与缓冲区。
    def __init__(
        self,
        topk: int,
        workspace: CandidateQliRawTopKWorkspace,
    ):
        self.ub_pool = workspace.ub_pool
        self.topk = int(topk)
        if self.topk <= 0:
            raise ValueError("ASC raw TopK requires positive topk")
        # The ASC VF works in fixed 16K trunks.  The number of complete trunks
        # and the final tail length are runtime values so one compiled kernel
        # can serve every supported S2.
        self.trunk_len = CANDIDATE_LD_TRUNK_LEN
        # ASC reserves history in 256-element units before the next trunk.
        self.topk_pad = ceil_div(self.topk, 256) * 256
        self.merge_len_pad = ceil_div(self.topk_pad + CANDIDATE_LD_TRUNK_LEN, 128) * 128
        self.workspace = workspace
        if (
            CANDIDATE_LD_TRUNK_LEN > workspace.max_trunk_len
            or self.topk > workspace.max_topk
            or self.merge_len_pad > workspace.merge_len_pad
        ):
            raise ValueError("TopK selector exceeds the shared UB workspace")
        # Final V->MTE3 stages are selector-owned so Sparse and Candidate can
        # share computational scratch without racing each other's GM writes.
        self.output_idx_stage = _ub_ring_buffer(
            self.ub_pool,
            MemLoc.UB,
            (1, self.topk),
            dtypes.int32,
            depth=1,
        )
        self.output_value_bits_stage = _ub_ring_buffer(
            self.ub_pool,
            MemLoc.UB,
            (1, self.topk),
            dtypes.uint16,
            depth=1,
        )

    # 读取最终阈值，按大于和等于阈值两轮收集候选块索引。
    @jit
    def _candidate_collect_indices(
        self, sort_key, valid_len, tmp_idx, histogram, high_target_carrier
    ):
        with vf(mode="simd"):
            full16 = rr.update_mask(128, elem_bits=16)[0]
            high_target = rr.vreinterpret_lanes(
                rr.vload_broadcast(high_target_carrier, 0, width="b16"), dtypes.uint16
            )
            low_target = rr.vload_broadcast(tmp_idx, 0)
            kth = rr.vbitwise_or(
                rr.vshl(high_target, 8, mask=full16), low_target, mask=full16
            )
            self._collect_threshold_indices_vf(sort_key, valid_len, tmp_idx, kth)
            rr.vmem_bar("vst_vld")

    # 在当前 VF 内读取高位阈值并计算桶内排名，准备低字节直方图扫描。
    @jit
    def _prepare_candidate_low_scan_vf(
        self, valid_len, histogram, tmp_idx, high_target_carrier
    ):
        full16 = rr.update_mask(128, elem_bits=16)[0]
        lane0 = rr.update_mask(1, elem_bits=16)[0]
        zero16 = rr.vdups(0, dtypes.uint16, mask=full16)
        one16 = rr.vdups(1, dtypes.uint16, mask=full16)
        bottom_k = valid_len - self.topk + 1
        high_target = rr.vreinterpret_lanes(
            rr.vload_broadcast(high_target_carrier, 0, width="b16"), dtypes.uint16
        )
        high_target8 = rr.vreinterpret_lanes(
            rr.vload_broadcast(high_target_carrier, 0, width="b8"), dtypes.uint8
        )
        prev = rr.vsub(high_target, one16, mask=full16)
        is_zero = rr.veqs(high_target, 0, mask=full16)
        prev = rr.vselect(zero16, prev, cond_mask=is_zero)
        prev_count = zero16
        for base in range_constexpr(0, 256, 128):
            hv = rr.vload(histogram, base)
            idx = rr.varange(base, dtypes.uint16)
            at_prev = rr.veq(idx, prev, mask=full16)
            picked = rr.vselect(hv, zero16, cond_mask=at_prev)
            prev_count = rr.vadd(
                prev_count, rr.vreduce_max(picked, mask=full16), mask=full16
            )
        prev_count = rr.vselect(zero16, prev_count, cond_mask=is_zero)
        next_k = rr.vsub(
            rr.vdups(bottom_k, dtypes.uint16, mask=full16), prev_count, mask=full16
        )
        rr.vstore(tmp_idx, 0, next_k, lane0)
        return (full16, high_target8)

    # 由高字节直方图选出目标桶，并写出后续低字节筛选所需阈值。
    @jit
    def _candidate_high_threshold(
        self, sort_key, valid_len, tmp_idx, histogram, high_target_carrier
    ):
        with vf(mode="simd"):
            full16 = rr.update_mask(128, elem_bits=16)[0]
            lane0 = rr.update_mask(1, elem_bits=16)[0]
            bottom_k = valid_len - self.topk + 1
            all_ones = rr.vdups(UINT16_ALL_BITS, dtypes.uint16, mask=full16)
            hist0 = rr.vload(histogram, 0)
            hist1 = rr.vload(histogram, 128)
            candidate0 = rr.vselect(
                rr.varange(0, dtypes.uint16),
                all_ones,
                cond_mask=rr.vges(hist0, bottom_k, mask=full16),
            )
            candidate1 = rr.vselect(
                rr.varange(128, dtypes.uint16),
                all_ones,
                cond_mask=rr.vges(hist1, bottom_k, mask=full16),
            )
            high_target = rr.vmin(
                rr.vreduce_min(candidate0, mask=full16),
                rr.vreduce_min(candidate1, mask=full16),
                mask=lane0,
            )
            rr.vstore(
                high_target_carrier,
                0,
                rr.vreinterpret(
                    rr.vunpack(high_target, dtypes.uint32, part="lower"), dtypes.int32
                ),
                rr.update_mask(1, elem_bits=32)[0],
            )
            rr.vmem_bar("vst_vld")

    # 由低字节直方图确定最终阈值，并写入阈值载体。
    @jit
    def _candidate_low_threshold(
        self, sort_key, valid_len, tmp_idx, histogram, high_target_carrier
    ):
        with vf(mode="simd"):
            full16 = rr.update_mask(128, elem_bits=16)[0]
            lane0 = rr.update_mask(1, elem_bits=16)[0]
            next_k = rr.vload_broadcast(tmp_idx, 0)
            all_ones = rr.vdups(UINT16_ALL_BITS, dtypes.uint16, mask=full16)
            hist0 = rr.vload(histogram, 0)
            hist1 = rr.vload(histogram, 128)
            candidate0 = rr.vselect(
                rr.varange(0, dtypes.uint16),
                all_ones,
                cond_mask=rr.vge(hist0, next_k, mask=full16),
            )
            candidate1 = rr.vselect(
                rr.varange(128, dtypes.uint16),
                all_ones,
                cond_mask=rr.vge(hist1, next_k, mask=full16),
            )
            low_target = rr.vmin(
                rr.vreduce_min(candidate0, mask=full16),
                rr.vreduce_min(candidate1, mask=full16),
                mask=lane0,
            )
            rr.vstore(tmp_idx, 0, low_target, lane0)
            rr.vmem_bar("vst_vld")

    # 候选块使用自己的精确选择器，复用父类单 trunk 搬运与输出流程。
    @jit
    def _find_indices_dispatch(
        self,
        sort_key,
        valid_len,
        tmp_idx,
        histogram,
        high_target_carrier,
        cached_hist=0,
        hist_row=0,
    ):
        self._find_indices(sort_key, valid_len, tmp_idx, histogram, high_target_carrier)

    # 加载当前输入块及历史候选到合并缓冲区。
    @jit
    def _load_merge(
        self, gm_keys, history, merge, source_offset, history_prefix, length
    ):
        copied = ((length + 127) // 128) * 128
        merge_slot = merge.next_write()
        mem_copy(
            reinterpret(
                _offset_view(merge_slot, (0, history_prefix)),
                layout=make_layout((1, copied)),
            ),
            gm_keys[0:1, source_offset : source_offset + copied],
        )
        with vf(mode="simd"):
            mask = rr.update_mask(128, elem_bits=16)[0]
            for chunk in dsl_range(history_prefix // 128, unroll=1):
                rr.vstore(merge_slot, chunk * 128, rr.vload(history, chunk * 128), mask)
            rr.vmem_bar("vst_vld")

    # 寻找当前候选集合的 TopK 索引。
    @jit
    def _find_indices(
        self, sort_key, valid_len, tmp_idx, histogram, high_target_carrier
    ):
        if self.workspace.candidate_optimized:
            self._find_indices_optimized(
                sort_key, valid_len, tmp_idx, histogram, high_target_carrier
            )
        else:
            self._find_indices_baseline(
                sort_key, valid_len, tmp_idx, histogram, high_target_carrier
            )

    # 使用通用直方图路径选择候选块 TopK。
    @jit
    def _find_indices_baseline(
        self, sort_key, valid_len, tmp_idx, histogram, high_target_carrier
    ):
        with vf(mode="simd"):
            full16 = rr.update_mask(128, elem_bits=16)[0]
            histogram_low_reg = rr.vdups(0, dtypes.uint16, mask=full16)
            histogram_high_reg = rr.vdups(0, dtypes.uint16, mask=full16)
            for chunk in dsl_range((valid_len + 255) // 256, unroll=1):
                valid_count = valid_len - chunk * 256
                mask8 = rr.update_mask(valid_count, elem_bits=8)[0]
                _, high = rr.vload_deinterleave(sort_key, chunk * 256, width="b8")
                histogram_low_reg = rr.vhistogram_accumulate(
                    histogram_low_reg, high, mask=mask8, bin=0
                )
                histogram_high_reg = rr.vhistogram_accumulate(
                    histogram_high_reg, high, mask=mask8, bin=1
                )
            rr.vstore(histogram, 0, histogram_low_reg, full16)
            rr.vstore(histogram, 128, histogram_high_reg, full16)
            rr.vmem_bar("vst_vld")

        self._candidate_high_threshold(
            sort_key, valid_len, tmp_idx, histogram, high_target_carrier
        )

        with vf(mode="simd"):
            full16, high_target8 = self._prepare_candidate_low_scan_vf(
                valid_len,
                histogram,
                tmp_idx,
                high_target_carrier,
            )

            l0 = rr.vdups(0, dtypes.uint16, mask=full16)
            l1 = rr.vdups(0, dtypes.uint16, mask=full16)
            for chunk in dsl_range((valid_len + 255) // 256, unroll=1):
                valid_count = valid_len - chunk * 256
                mask8 = rr.update_mask(valid_count, elem_bits=8)[0]
                low, high = rr.vload_deinterleave(sort_key, chunk * 256, width="b8")
                eq_high = rr.veq(high, high_target8, mask=mask8)
                l0 = rr.vhistogram_accumulate(l0, low, mask=eq_high, bin=0)
                l1 = rr.vhistogram_accumulate(l1, low, mask=eq_high, bin=1)
            rr.vstore(histogram, 0, l0, full16)
            rr.vstore(histogram, 128, l1, full16)
            rr.vmem_bar("vst_vld")

        self._candidate_low_threshold(
            sort_key, valid_len, tmp_idx, histogram, high_target_carrier
        )

        self._candidate_collect_indices(
            sort_key, valid_len, tmp_idx, histogram, high_target_carrier
        )

    # 使用候选块专用路径选择 TopK。
    @jit
    def _find_indices_optimized(
        self, sort_key, valid_len, tmp_idx, histogram, high_target_carrier
    ):
        with vf(mode="simd"):
            full16 = rr.update_mask(128, elem_bits=16)[0]
            h0a = rr.vdups(0, dtypes.uint16, mask=full16)
            h1a = rr.vdups(0, dtypes.uint16, mask=full16)
            h0b = rr.vdups(0, dtypes.uint16, mask=full16)
            h1b = rr.vdups(0, dtypes.uint16, mask=full16)
            for chunk in dsl_range(0, (valid_len + 255) // 256, 2, unroll=1):
                valid_count = valid_len - chunk * 256
                mask8 = rr.update_mask(valid_count, elem_bits=8)[0]
                _, high = rr.vload_deinterleave(sort_key, chunk * 256, width="b8")
                h0a = rr.vhistogram_accumulate(h0a, high, mask=mask8, bin=0)
                h1a = rr.vhistogram_accumulate(h1a, high, mask=mask8, bin=1)
            for chunk in dsl_range(1, (valid_len + 255) // 256, 2, unroll=1):
                valid_count = valid_len - chunk * 256
                mask8 = rr.update_mask(valid_count, elem_bits=8)[0]
                _, high = rr.vload_deinterleave(sort_key, chunk * 256, width="b8")
                h0b = rr.vhistogram_accumulate(h0b, high, mask=mask8, bin=0)
                h1b = rr.vhistogram_accumulate(h1b, high, mask=mask8, bin=1)
            histogram_low_reg = rr.vadd(h0a, h0b, mask=full16)
            histogram_high_reg = rr.vadd(h1a, h1b, mask=full16)
            rr.vstore(histogram, 0, histogram_low_reg, full16)
            rr.vstore(histogram, 128, histogram_high_reg, full16)
            rr.vmem_bar("vst_vld")

        self._candidate_high_threshold(
            sort_key, valid_len, tmp_idx, histogram, high_target_carrier
        )

        with vf(mode="simd"):
            full16, high_target8 = self._prepare_candidate_low_scan_vf(
                valid_len,
                histogram,
                tmp_idx,
                high_target_carrier,
            )

            l0a = rr.vdups(0, dtypes.uint16, mask=full16)
            l1a = rr.vdups(0, dtypes.uint16, mask=full16)
            l0b = rr.vdups(0, dtypes.uint16, mask=full16)
            l1b = rr.vdups(0, dtypes.uint16, mask=full16)
            for chunk in dsl_range(0, (valid_len + 255) // 256, 2, unroll=1):
                valid_count = valid_len - chunk * 256
                mask8 = rr.update_mask(valid_count, elem_bits=8)[0]
                low, high = rr.vload_deinterleave(sort_key, chunk * 256, width="b8")
                eq_high = rr.veq(high, high_target8, mask=mask8)
                l0a = rr.vhistogram_accumulate(l0a, low, mask=eq_high, bin=0)
                l1a = rr.vhistogram_accumulate(l1a, low, mask=eq_high, bin=1)
            for chunk in dsl_range(1, (valid_len + 255) // 256, 2, unroll=1):
                valid_count = valid_len - chunk * 256
                mask8 = rr.update_mask(valid_count, elem_bits=8)[0]
                low, high = rr.vload_deinterleave(sort_key, chunk * 256, width="b8")
                eq_high = rr.veq(high, high_target8, mask=mask8)
                l0b = rr.vhistogram_accumulate(l0b, low, mask=eq_high, bin=0)
                l1b = rr.vhistogram_accumulate(l1b, low, mask=eq_high, bin=1)
            l0 = rr.vadd(l0a, l0b, mask=full16)
            l1 = rr.vadd(l1a, l1b, mask=full16)
            rr.vstore(histogram, 0, l0, full16)
            rr.vstore(histogram, 128, l1, full16)
            rr.vmem_bar("vst_vld")

        self._candidate_low_threshold(
            sort_key, valid_len, tmp_idx, histogram, high_target_carrier
        )

        self._candidate_collect_indices(
            sort_key, valid_len, tmp_idx, histogram, high_target_carrier
        )

    # 选择单行 TopK 的适用路径并写出结果。
    @jit
    def select_row(
        self,
        gm_sort_key: Tensor,
        total_tokens,
        output_valid_count,
        gm_sparse_indices: Tensor,
        gm_selected_value_bits: Tensor,
        index_base,
        need_values=1,
    ):
        if (
            need_values == 0
            and total_tokens >= self.topk
            and total_tokens <= self.trunk_len
        ):
            merge_key = self.workspace.merge_key
            self._load_merge(
                gm_sort_key,
                self.workspace.history_key,
                merge_key,
                dtypes.int64(0),
                dtypes.int64(0),
                total_tokens,
            )
            self._find_indices(
                merge_key.next_read(),
                total_tokens,
                self.workspace.tmp_idx,
                self.workspace.histogram,
                self.workspace.high_target_carrier,
            )
            out = self.output_idx_stage.next_write()
            tmp = self.workspace.tmp_idx
            with vf(mode="simd"):
                self._emit_indices_only_vf(tmp, out, output_valid_count, index_base)
                rr.vmem_bar("vst_vld")
            mem_copy(gm_sparse_indices, self.output_idx_stage.next_read())
        elif (
            self.workspace.candidate_optimized
            and need_values != 0
            and total_tokens >= self.topk
            and total_tokens <= self.trunk_len
        ):
            self._select_row_single_trunk(
                gm_sort_key,
                total_tokens,
                output_valid_count,
                gm_sparse_indices,
                gm_selected_value_bits,
                index_base,
            )
        else:
            self._select_row_values(
                gm_sort_key,
                total_tokens,
                output_valid_count,
                gm_sparse_indices,
                gm_selected_value_bits,
                index_base,
            )


class CandidateLdTopKSelector(CandidateQliRawTopKSelector):
    def __init__(self, topk, workspace, splits):
        self.ub_pool = workspace.ub_pool
        super().__init__(topk, workspace)
        self._init_ld_buffers(splits)

    @jit
    def merge_ld(
        self,
        partial_indices,
        partial_bits,
        gm_sparse_indices,
        gm_selected_value_bits,
        valid_length,
        output_offset=0,
        preloaded=False,
        need_values=1,
        emit_gm=1,
    ):
        trunk_read, ld_idx_read, ld_bits_read = self._prepare_ld_merge(
            partial_indices,
            partial_bits,
            valid_length,
            preloaded,
        )
        self._emit_ld_values(
            trunk_read,
            ld_idx_read,
            ld_bits_read,
            output_offset,
            gm_sparse_indices,
            gm_selected_value_bits,
            emit_gm,
        )


class CandidateLdMergeStage(LdMergeStage):
    # 初始化候选块 LD 归并阶段的配置与缓冲区。
    def __init__(self, topk, splits, candidate_optimized=False):
        self.topk = topk
        self.length = topk * splits
        self.workspace = CandidateQliRawTopKWorkspace(
            CANDIDATE_LD_TRUNK_LEN,
            topk,
            candidate_optimized=candidate_optimized,
        )
        self.selector = CandidateLdTopKSelector(topk, self.workspace, splits)


class QliCube:
    """AIC-owned native MXFP4 QK resources."""

    def __init__(self, pa_block_size, n1, query_rows):
        self.key_tile_columns = QK_COLUMNS
        self.n1 = int(n1)
        self.tile_m = int(n1) * int(query_rows)
        self.pa_block_size = int(pa_block_size)
        self.copy_rows = gcd(QK_COLUMNS, self.pa_block_size)
        self.q_l1 = RingBuffer(
            MemLoc.L1,
            dtypes.fp4x2_e2m1,
            make_layout((self.tile_m, LOGICAL_D)),
            depth=2,
            data_format="nz",
        )
        self.k_l1 = RingBuffer(
            MemLoc.L1,
            dtypes.fp4x2_e2m1,
            make_layout((QK_COLUMNS, LOGICAL_D)),
            depth=4,
            data_format="nz",
        )
        self.q_scale_l1 = RingBuffer(
            MemLoc.L1,
            dtypes.float8_e8m0,
            make_layout((4, self.tile_m)),
            depth=2,
            data_format="nz",
        )
        self.k_scale_l1 = RingBuffer(
            MemLoc.L1,
            dtypes.float8_e8m0,
            make_layout((4, QK_COLUMNS)),
            depth=4,
            data_format="nz",
        )
        self.l0a0 = RingBuffer(
            MemLoc.L0A,
            dtypes.fp4x2_e2m1,
            make_layout((self.tile_m // 2, LOGICAL_D)),
            depth=1,
        )
        self.l0a1 = RingBuffer(
            MemLoc.L0A,
            dtypes.fp4x2_e2m1,
            make_layout((self.tile_m // 2, LOGICAL_D)),
            depth=1,
        )
        self.l0b = make_ring_buffer(
            [
                reinterpret(
                    L0B.view(65536),
                    dtype=dtypes.fp4x2_e2m1,
                    layout=make_layout((QK_COLUMNS, LOGICAL_D)),
                    offset=i * 16384,
                    data_format="nz",
                )
                for i in range(2)
            ]
        )
        self.l0c = make_ring_buffer(
            [
                reinterpret(
                    L0C.view(262144),
                    dtype=dtypes.float32,
                    layout=make_layout((self.tile_m // 2, QK_COLUMNS)),
                    offset=i * (self.tile_m // 2) * QK_COLUMNS * 4,
                    data_format="nz",
                )
                for i in range(2)
            ]
        )
        self.scale_a = None
        self.scale_b = None

    @jit
    def _load_query_l1(self, query, query_scale, query_token_start):
        source_row = dtypes.int64(query_token_start) * self.n1
        avail = dtypes.int64(query.shape[0]) - source_row
        avail = avail if avail > 0 else dtypes.int64(0)
        rows_to_copy = avail if avail < self.tile_m else dtypes.int64(self.tile_m)
        mem_copy(
            tile_slice(self.q_l1.next_write(), (self.tile_m, LOGICAL_D), (0, 0)),
            query[source_row : source_row + rows_to_copy, None],
        )
        mem_copy(
            self.q_scale_l1.next_write().permute(1, 0),
            tile_slice(
                _offset_view(query_scale, (source_row, 0, 0)),
                (self.tile_m, 2, 2),
                (0, 0, 0),
            ).flatten(1, 2),
            options=self.scale_a,
        )
        query_read = self.q_l1.next_read()
        scale_read = self.q_scale_l1.next_read().permute(1, 0)
        return (query_read, scale_read)

    @jit
    def load_query(self, query, query_scale, query_token_start):
        query_read, scale_read = self._load_query_l1(
            query, query_scale, query_token_start
        )
        mem_copy(
            self.l0a0.next_write(),
            tile_slice(
                reinterpret(query_read, layout=make_layout((self.tile_m, LOGICAL_D))),
                (self.tile_m // 2, LOGICAL_D),
                (0, 0),
            ),
            options=copy_options(
                mx_scale=tile_slice(
                    reinterpret(scale_read, layout=make_layout((self.tile_m, 4))),
                    (self.tile_m // 2, 4),
                    (0, 0),
                )
            ),
        )
        mem_copy(
            self.l0a1.next_write(),
            tile_slice(
                reinterpret(query_read, layout=make_layout((self.tile_m, LOGICAL_D))),
                (self.tile_m // 2, LOGICAL_D),
                (1, 0),
            ),
            options=copy_options(
                mx_scale=tile_slice(
                    reinterpret(scale_read, layout=make_layout((self.tile_m, 4))),
                    (self.tile_m // 2, 4),
                    (1, 0),
                )
            ),
        )

    @jit
    def prefetch_key(
        self,
        key,
        key_scale,
        block_table,
        batch_idx,
        actual_s2,
        key_stride0,
        key_dequant_scale_stride0,
        has_block_table,
        token_tile,
        qk_handoff,
    ):
        key_write = self.k_l1.next_write()
        scale_write = self.k_scale_l1.next_write()
        for piece in range(self.key_tile_columns // self.copy_rows):
            logical_row = token_tile * TILE_N + piece * self.copy_rows
            safe_row = logical_row
            if logical_row >= dtypes.int64(actual_s2):
                safe_row = 0
            logical_page = safe_row // self.pa_block_size
            page_offset = safe_row % self.pa_block_size
            physical_page = logical_page
            if has_block_table:
                physical_page = dtypes.int64(block_table[batch_idx, logical_page])
            source_row = (
                physical_page * (key_stride0 // KEY_STORAGE_ROW_ELEMENTS) + page_offset
            )
            scale_row = (
                physical_page
                * (key_dequant_scale_stride0 // KEY_SCALE_STORAGE_ROW_ELEMENTS)
                + page_offset
            )
            mem_copy(
                tile_slice(key_write, (self.copy_rows, LOGICAL_D), (piece, 0)),
                key[source_row : source_row + self.copy_rows, None],
                options=copy_options(l2_cache_ctl=0),
            )
            mem_copy(
                tile_slice(scale_write, (4, self.copy_rows), (0, piece)),
                key_scale[scale_row : scale_row + self.copy_rows, None, None]
                .flatten(1, 2)
                .permute(1, 0),
                options=copy_options(self.scale_b, l2_cache_ctl=0),
            )

    @jit
    def compute_qk(
        self,
        key,
        key_scale,
        block_table,
        batch_idx,
        actual_s2,
        key_stride0,
        key_dequant_scale_stride0,
        has_block_table,
        token_tile,
        qk_handoff,
    ):
        mem_copy(
            self.l0b.next_write(),
            self.k_l1.next_read(),
            options=copy_options(mx_scale=self.k_scale_l1.next_read()),
        )
        key_read = self.l0b.next_read()
        dsl_matmul(
            self.l0c.next_write(),
            self.l0a0.next_read(),
            key_read,
            init=True,
            mx_mode=True,
            unit_flag=3,
        )
        qk_l0c_read0 = self.l0c.next_read()
        mem_copy(
            reinterpret(
                qk_handoff, layout=make_layout((self.tile_m // 2, 128), stride=(256, 1))
            ),
            relu(
                reinterpret(qk_l0c_read0, layout=make_layout((self.tile_m // 2, 128)))
            ),
            options=copy_options(dst_subblock=0, unit_flag=3),
        )
        mem_copy(
            reinterpret(
                qk_handoff,
                layout=make_layout((self.tile_m // 2, 128), stride=(256, 1)),
                offset=(self.tile_m // 2) * 512,
            ),
            relu(
                reinterpret(
                    qk_l0c_read0,
                    layout=make_layout((self.tile_m // 2, 128)),
                    offset=(self.tile_m // 2) * 128 * 4,
                )
            ),
            options=copy_options(dst_subblock=0, unit_flag=3),
        )
        dsl_matmul(
            self.l0c.next_write(),
            self.l0a1.next_read(),
            key_read,
            init=True,
            mx_mode=True,
            unit_flag=3,
        )
        qk_l0c_read1 = self.l0c.next_read()
        mem_copy(
            reinterpret(
                qk_handoff, layout=make_layout((self.tile_m // 2, 128), stride=(256, 1))
            ),
            relu(
                reinterpret(qk_l0c_read1, layout=make_layout((self.tile_m // 2, 128)))
            ),
            options=copy_options(dst_subblock=1, unit_flag=3),
        )
        mem_copy(
            reinterpret(
                qk_handoff,
                layout=make_layout((self.tile_m // 2, 128), stride=(256, 1)),
                offset=(self.tile_m // 2) * 512,
            ),
            relu(
                reinterpret(
                    qk_l0c_read1,
                    layout=make_layout((self.tile_m // 2, 128)),
                    offset=(self.tile_m // 2) * 128 * 4,
                )
            ),
            options=copy_options(dst_subblock=1, unit_flag=3),
        )


class OwnerCube512(QliCube):
    """AIC-owned native MXFP4 QK resources."""

    # 初始化M64/N512 Cube 计算的配置与缓冲区。
    def __init__(self, pa_block_size, n1, query_rows, shared_cube):
        self.key_tile_columns = 512
        self.n1 = int(n1)
        self.tile_m = int(n1) * int(query_rows)
        self.pa_block_size = int(pa_block_size)
        self.copy_rows = gcd(512, self.pa_block_size)
        self.q_l1 = shared_cube.q_l1
        self.k_l1 = RingBuffer(
            MemLoc.L1,
            dtypes.fp4x2_e2m1,
            make_layout((512, LOGICAL_D)),
            depth=4,
            data_format="nz",
        )
        self.q_scale_l1 = shared_cube.q_scale_l1
        self.k_scale_l1 = RingBuffer(
            MemLoc.L1,
            dtypes.float8_e8m0,
            make_layout((4, 512)),
            depth=4,
            data_format="nz",
        )
        self.l0b = make_ring_buffer(
            [
                reinterpret(
                    L0B.view(65536),
                    dtype=dtypes.fp4x2_e2m1,
                    layout=make_layout((512, LOGICAL_D)),
                    offset=i * 32768,
                    data_format="nz",
                )
                for i in range(2)
            ]
        )
        self.l0c_root = L0C.view(262144)
        self.micro_l0c = make_ring_buffer(
            [
                reinterpret(
                    self.l0c_root,
                    dtype=dtypes.float32,
                    layout=make_layout((64, 512)),
                    offset=0,
                    data_format="nz",
                ),
                reinterpret(
                    self.l0c_root,
                    dtype=dtypes.float32,
                    layout=make_layout((64, 512)),
                    offset=131072,
                    data_format="nz",
                ),
            ]
        )
        self.scale_a = None
        self.scale_b = None
        self.micro_a0 = RingBuffer(
            MemLoc.L0A, dtypes.fp4x2_e2m1, make_layout((64, LOGICAL_D)), depth=1
        )
        self.micro_a1 = RingBuffer(
            MemLoc.L0A, dtypes.fp4x2_e2m1, make_layout((64, LOGICAL_D)), depth=1
        )
        self.micro_a2 = RingBuffer(
            MemLoc.L0A, dtypes.fp4x2_e2m1, make_layout((64, LOGICAL_D)), depth=1
        )

    # 将六个 query 分成三组，准备每组 M64 的 Cube 输入。
    @jit
    def load_micro_query(self, query, query_scale, query_token_start):
        query_read, scale_read = self._load_query_l1(
            query, query_scale, query_token_start
        )
        mem_copy(
            self.micro_a0.next_write(),
            tile_slice(query_read, (64, LOGICAL_D), (0, 0)),
            options=copy_options(mx_scale=tile_slice(scale_read, (64, 4), (0, 0))),
        )
        mem_copy(
            self.micro_a1.next_write(),
            tile_slice(query_read, (64, LOGICAL_D), (1, 0)),
            options=copy_options(mx_scale=tile_slice(scale_read, (64, 4), (1, 0))),
        )
        mem_copy(
            self.micro_a2.next_write(),
            tile_slice(query_read, (64, LOGICAL_D), (2, 0)),
            options=copy_options(mx_scale=tile_slice(scale_read, (64, 4), (2, 0))),
        )


class QliVector:
    """Each AIV consumes its 64-row split and produces two QLI rows."""

    def __init__(
        self,
        subblock_idx,
        mask_mode,
        cmp_ratio,
        candidate_enabled,
        candidate_topk_count,
        n1,
        query_rows,
        weight_addr,
        cache_enabled,
    ):
        self.ub_pool = _UbPool(weight_addr)
        self.cache_enabled = bool(cache_enabled)
        self.n1 = int(n1)
        self.query_rows = int(query_rows)
        self.vec_rows = int(query_rows) // 2
        self.subblock_idx = subblock_idx
        self.mask_mode = int(mask_mode)
        self.cmp_ratio = int(cmp_ratio)
        self.candidate_enabled = bool(candidate_enabled)
        self.candidate_topk_count = int(candidate_topk_count)
        self.weight_ub = _ub_ring_buffer(
            self.ub_pool,
            MemLoc.UB,
            (self.vec_rows, self.n1),
            dtypes.float32,
            depth=1,
            addr=weight_addr,
        )
        self.packed_weight = _ub_buffer(
            self.ub_pool,
            MemLoc.UB,
            (self.vec_rows, 128),
            dtypes.bfloat16,
        )
        self.score_ub = _ub_buffer(
            self.ub_pool,
            MemLoc.UB,
            (self.vec_rows, QK_COLUMNS),
            dtypes.bfloat16,
        )
        self.key_ub = _ub_ring_buffer(
            self.ub_pool,
            MemLoc.UB,
            (self.vec_rows, QK_COLUMNS),
            dtypes.uint16,
            depth=2,
        )
        if self.candidate_enabled:
            # Each row is written by VF before one batched DMA reads them;
            # the RingBuffer slots own the V->MTE3 ordering.
            self.candidate_key_ub = _ub_ring_buffer(
                self.ub_pool,
                MemLoc.UB,
                (self.vec_rows, QK_COLUMNS // CANDIDATE_BLOCK_SIZE),
                dtypes.uint16,
                depth=1,
            )
        self.candidate_length_ub = _ub_ring_buffer(
            self.ub_pool,
            MemLoc.UB,
            (1,),
            dtypes.int32,
            depth=1,
        )

        self.byte_enabled = False

    @jit
    def _accumulate_scores_vf(self, qk_handoff, group_count, mask16, apply_relu):
        bf16_weight0 = rr.vload(self.packed_weight, 0 * 128)
        query0_s2_half0_acc_reg = rr.vdups(0.0, dtypes.bfloat16, mask=mask16)
        query0_s2_half1_acc_reg = rr.vdups(0.0, dtypes.bfloat16, mask=mask16)
        bf16_weight1 = rr.vload(self.packed_weight, 1 * 128)
        query1_s2_half0_acc_reg = rr.vdups(0.0, dtypes.bfloat16, mask=mask16)
        query1_s2_half1_acc_reg = rr.vdups(0.0, dtypes.bfloat16, mask=mask16)
        if const_expr(self.vec_rows == 3):
            bf16_weight2 = rr.vload(self.packed_weight, 2 * 128)
        query2_s2_half0_acc_reg = rr.vdups(0.0, dtypes.bfloat16, mask=mask16)
        query2_s2_half1_acc_reg = rr.vdups(0.0, dtypes.bfloat16, mask=mask16)
        for group in dsl_range(group_count, unroll=4):
            lane = rr.vdups(group, dtypes.uint16, mask=mask16)
            w0 = rr.vgather_reg(bf16_weight0, lane)
            w1 = rr.vgather_reg(bf16_weight1, lane)
            if const_expr(self.vec_rows == 3):
                w2 = rr.vgather_reg(bf16_weight2, lane)
            qk0_0 = rr.vload(
                qk_handoff,
                (0 * self.n1 + group) * 256 + 0 * (self.n1 * self.vec_rows) * 256,
            )
            if const_expr(apply_relu):
                qk0_0 = rr.vmaxs(qk0_0, 0.0, mask=mask16)
            query0_s2_half0_acc_reg = rr.vmadd(
                qk0_0, w0, query0_s2_half0_acc_reg, mask=mask16
            )
            qk0_1 = rr.vload(
                qk_handoff,
                (0 * self.n1 + group) * 256 + 1 * (self.n1 * self.vec_rows) * 256,
            )
            if const_expr(apply_relu):
                qk0_1 = rr.vmaxs(qk0_1, 0.0, mask=mask16)
            query0_s2_half1_acc_reg = rr.vmadd(
                qk0_1, w0, query0_s2_half1_acc_reg, mask=mask16
            )
            qk1_0 = rr.vload(
                qk_handoff,
                (1 * self.n1 + group) * 256 + 0 * (self.n1 * self.vec_rows) * 256,
            )
            if const_expr(apply_relu):
                qk1_0 = rr.vmaxs(qk1_0, 0.0, mask=mask16)
            query1_s2_half0_acc_reg = rr.vmadd(
                qk1_0, w1, query1_s2_half0_acc_reg, mask=mask16
            )
            qk1_1 = rr.vload(
                qk_handoff,
                (1 * self.n1 + group) * 256 + 1 * (self.n1 * self.vec_rows) * 256,
            )
            if const_expr(apply_relu):
                qk1_1 = rr.vmaxs(qk1_1, 0.0, mask=mask16)
            query1_s2_half1_acc_reg = rr.vmadd(
                qk1_1, w1, query1_s2_half1_acc_reg, mask=mask16
            )
            if const_expr(self.vec_rows == 3):
                qk2_0 = rr.vload(
                    qk_handoff,
                    (2 * self.n1 + group) * 256 + 0 * (self.n1 * self.vec_rows) * 256,
                )
                if const_expr(apply_relu):
                    qk2_0 = rr.vmaxs(qk2_0, 0.0, mask=mask16)
                query2_s2_half0_acc_reg = rr.vmadd(
                    qk2_0, w2, query2_s2_half0_acc_reg, mask=mask16
                )
                qk2_1 = rr.vload(
                    qk_handoff,
                    (2 * self.n1 + group) * 256 + 1 * (self.n1 * self.vec_rows) * 256,
                )
                if const_expr(apply_relu):
                    qk2_1 = rr.vmaxs(qk2_1, 0.0, mask=mask16)
                query2_s2_half1_acc_reg = rr.vmadd(
                    qk2_1, w2, query2_s2_half1_acc_reg, mask=mask16
                )
        return (
            query0_s2_half0_acc_reg,
            query0_s2_half1_acc_reg,
            query1_s2_half0_acc_reg,
            query1_s2_half1_acc_reg,
            query2_s2_half0_acc_reg,
            query2_s2_half1_acc_reg,
        )

    @jit
    def _cache_byte_chunk_vf(
        self,
        bucket_key,
        byte_base,
        byte_used,
        row,
        valid_count,
        byte_cache,
        cache_offset,
        histogram_low_reg,
        histogram_high_reg,
        mask16,
        zero_hist,
    ):
        key_high = rr.vmins(
            rr.vsub(
                rr.vmax(bucket_key, byte_base, mask=mask16), byte_base, mask=mask16
            ),
            255,
            mask=mask16,
        )
        byte_mask = rr.update_mask(valid_count, elem_bits=16)[0]
        byte_over = rr.veqs(key_high, 255, mask=byte_mask)
        byte_prefix = rr.vsqueeze(byte_over, dtypes.uint16)
        byte_pos = rr.vadd(byte_used, byte_prefix, mask=mask16)
        byte_keep = rr.mask_and(
            byte_over, rr.vlts(byte_pos, 512, mask=mask16), exec_mask=mask16
        )
        rr.vscatter(
            self.byte_overflow,
            bucket_key,
            rr.vadds(byte_pos, row * 512, mask=mask16),
            mask=byte_keep,
        )
        byte_one = rr.vselect(
            rr.vdups(1, dtypes.uint16, mask=mask16), zero_hist, cond_mask=byte_over
        )
        byte_used = rr.vadd(
            byte_used,
            rr.vdup(
                rr.vreduce_max(
                    rr.vadd(byte_prefix, byte_one, mask=mask16), mask=mask16
                ),
                mask=mask16,
            ),
            mask=mask16,
        )
        packed_high = rr.vpack(key_high, dtypes.uint8, part="lower")
        rr.vstore(
            byte_cache, cache_offset, packed_high, rr.update_mask(128, elem_bits=8)[0]
        )
        hist_mask = rr.update_mask(valid_count, elem_bits=8)[0]
        histogram_low_reg = rr.vhistogram_accumulate(
            histogram_low_reg, packed_high, mask=hist_mask, bin=0
        )
        histogram_high_reg = rr.vhistogram_accumulate(
            histogram_high_reg, packed_high, mask=hist_mask, bin=1
        )
        return (byte_used, histogram_low_reg, histogram_high_reg)

    @jit
    def _store_byte_score_row(
        self,
        score0,
        score1,
        row,
        valid,
        byte_cache,
        key_write,
        hist,
        workspace_offset,
        mask16,
        zero_hist,
        reset16,
    ):
        score0 = rr.vadds(score0, 0.0, mask=mask16)
        sortable_bits0_0 = _encode_bf16_sort_key(score0, mask16)
        rr.vstore(key_write, row * QK_COLUMNS + 0 * 128, sortable_bits0_0, mask16)
        score1 = rr.vadds(score1, 0.0, mask=mask16)
        sortable_bits0_1 = _encode_bf16_sort_key(score1, mask16)
        rr.vstore(key_write, row * QK_COLUMNS + 1 * 128, sortable_bits0_1, mask16)
        histogram_low_reg = rr.vselect(
            zero_hist, rr.vload(hist, row * 512 + 0), cond_mask=reset16
        )
        histogram_high_reg = rr.vselect(
            zero_hist, rr.vload(hist, row * 512 + 256), cond_mask=reset16
        )
        if workspace_offset == 0:
            rr.vstore(self.byte_seed, row * 256, sortable_bits0_0, mask16)
            rr.vstore(self.byte_seed, row * 256 + 128, sortable_bits0_1, mask16)
        else:
            byte_meta = rr.vload(self.byte_meta, row * 128)
            byte_base = rr.vdup(byte_meta, mask=mask16)
            byte_used = rr.vgather_reg(
                byte_meta, rr.vdups(1, dtypes.uint16, mask=mask16)
            )
            if workspace_offset == 256:
                seed0 = rr.vload(self.byte_seed, row * 256)
                seed1 = rr.vload(self.byte_seed, row * 256 + 128)
                seed2 = sortable_bits0_0
                seed3 = sortable_bits0_1
                best = rr.vdup(
                    rr.vreduce_max(
                        rr.vmax(
                            rr.vmax(seed0, seed1, mask=mask16),
                            rr.vmax(seed2, seed3, mask=mask16),
                            mask=mask16,
                        ),
                        mask=mask16,
                    ),
                    mask=mask16,
                )
                for rank in range_constexpr(7):
                    seed0 = rr.vselect(
                        seed0, zero_hist, cond_mask=rr.vlt(seed0, best, mask=mask16)
                    )
                    seed1 = rr.vselect(
                        seed1, zero_hist, cond_mask=rr.vlt(seed1, best, mask=mask16)
                    )
                    seed2 = rr.vselect(
                        seed2, zero_hist, cond_mask=rr.vlt(seed2, best, mask=mask16)
                    )
                    seed3 = rr.vselect(
                        seed3, zero_hist, cond_mask=rr.vlt(seed3, best, mask=mask16)
                    )
                    best = rr.vdup(
                        rr.vreduce_max(
                            rr.vmax(
                                rr.vmax(seed0, seed1, mask=mask16),
                                rr.vmax(seed2, seed3, mask=mask16),
                                mask=mask16,
                            ),
                            mask=mask16,
                        ),
                        mask=mask16,
                    )
                byte_base = rr.vmins(
                    rr.vsub(
                        rr.vmaxs(best, 126, mask=mask16),
                        rr.vdups(126, dtypes.uint16, mask=mask16),
                        mask=mask16,
                    ),
                    65280,
                    mask=mask16,
                )
                byte_used = zero_hist
                rr.vstore_first(self.byte_meta, row * 128, byte_base)
                bucket_key = rr.vload(self.byte_seed, row * 256 + 0 * 128)
                byte_used, histogram_low_reg, histogram_high_reg = (
                    self._cache_byte_chunk_vf(
                        bucket_key,
                        byte_base,
                        byte_used,
                        row,
                        128,
                        byte_cache,
                        0 * 128,
                        histogram_low_reg,
                        histogram_high_reg,
                        mask16,
                        zero_hist,
                    )
                )
                bucket_key = rr.vload(self.byte_seed, row * 256 + 1 * 128)
                byte_used, histogram_low_reg, histogram_high_reg = (
                    self._cache_byte_chunk_vf(
                        bucket_key,
                        byte_base,
                        byte_used,
                        row,
                        128,
                        byte_cache,
                        1 * 128,
                        histogram_low_reg,
                        histogram_high_reg,
                        mask16,
                        zero_hist,
                    )
                )
            bucket_key = sortable_bits0_0
            byte_used, histogram_low_reg, histogram_high_reg = (
                self._cache_byte_chunk_vf(
                    bucket_key,
                    byte_base,
                    byte_used,
                    row,
                    max(0, min(128, valid - 0 * 128)),
                    byte_cache,
                    dtypes.int32(workspace_offset) + 0 * 128,
                    histogram_low_reg,
                    histogram_high_reg,
                    mask16,
                    zero_hist,
                )
            )
            bucket_key = sortable_bits0_1
            byte_used, histogram_low_reg, histogram_high_reg = (
                self._cache_byte_chunk_vf(
                    bucket_key,
                    byte_base,
                    byte_used,
                    row,
                    max(0, min(128, valid - 1 * 128)),
                    byte_cache,
                    dtypes.int32(workspace_offset) + 1 * 128,
                    histogram_low_reg,
                    histogram_high_reg,
                    mask16,
                    zero_hist,
                )
            )
            rr.vstore_first(self.byte_meta, row * 128 + 1, byte_used)
        rr.vstore(hist, row * 512 + 0, histogram_low_reg, mask16)
        rr.vstore(hist, row * 512 + 256, histogram_high_reg, mask16)

    @jit
    def load_weights(self, weights, public_row_base, task_span_count):
        weight_row = (
            dtypes.int64(public_row_base)
            + dtypes.int64(self.subblock_idx) * self.vec_rows
        )
        if dtypes.int32(self.subblock_idx) * self.vec_rows >= task_span_count:
            weight_row = dtypes.int64(public_row_base)
        mem_copy(
            self.weight_ub.next_write(),
            tile_slice(
                _offset_view(weights, (weight_row, 0)), (self.vec_rows, self.n1), (0, 0)
            ),
        )
        weight_read = self.weight_ub.next_read()
        with vf(mode="simd"):
            mask32 = rr.update_mask(self.n1, elem_bits=32)[0]
            mask16 = rr.update_mask(self.n1, elem_bits=16)[0]
            for row in range_constexpr(self.vec_rows):
                weight = rr.vcast(
                    rr.vload(weight_read, row * self.n1),
                    dtypes.bfloat16,
                    mask=mask32,
                    reg_layout=rr.RegLayout.ZERO,
                )
                packed = rr.vpack(
                    rr.vreinterpret_lanes(weight, dtypes.uint32),
                    dtypes.uint16,
                    part="lower",
                )
                rr.vstore(
                    self.packed_weight,
                    row * 128,
                    rr.vreinterpret(packed, dtypes.bfloat16),
                    mask16,
                )
            rr.vmem_bar("vst_vld")

    @jit
    def store_empty_candidate_lengths(
        self, candidate_length_out, public_row_base, task_span_count
    ):
        """Write zero Candidate length for both rows owned by this AIV."""
        for row_in_subblock in range_constexpr(self.vec_rows):
            local_row = dtypes.int32(
                self.subblock_idx * self.vec_rows + row_in_subblock
            )
            if local_row < task_span_count:
                length_write = self.candidate_length_ub.next_write()
                with vf(mode="simd"):
                    length_mask = rr.update_mask(1, elem_bits=32)[0]
                    rr.vstore(
                        length_write,
                        0,
                        rr.vdups(0, dtypes.int32, mask=length_mask),
                        length_mask,
                    )
                    rr.vmem_bar("vst_vld")

                mem_copy(
                    tile_slice(
                        candidate_length_out,
                        (1,),
                        (dtypes.int64(public_row_base) + dtypes.int64(local_row),),
                    ),
                    self.candidate_length_ub.next_read(),
                )

    @jit
    def compute_vector1(
        self,
        qk_handoff,
        weights,
        key_out,
        candidate_key_out,
        public_row_base,
        task_span_count,
        task_active_count,
        workspace_task_idx,
        token_offset,
        actual_s2,
        task_q_offset,
        task_used_q,
        cmp_residual,
        total_candidate_tokens,
        candidate_length_out,
        workspace_token_offset,
        tile_tokens,
        stream_hist,
        hist,
        cache_task,
        stream_task=0,
        stream_total=0,
    ):
        if stream_task:
            self._score_rows(qk_handoff, dtypes.int32(self.n1), "plain")
            self.streaming.capture(self.key_ub.next_read(), workspace_token_offset)
            if workspace_token_offset + 256 == stream_total:
                v0 = actual_s2 - token_offset
                v1 = v0
                v2 = v0
                if self.mask_mode == 3:
                    query0 = (
                        task_q_offset + dtypes.int32(self.subblock_idx) * self.vec_rows
                    )
                    v0 = (
                        max(
                            0,
                            min(
                                actual_s2,
                                (
                                    actual_s2 * self.cmp_ratio
                                    + cmp_residual
                                    - task_used_q
                                    + query0
                                    + 1
                                )
                                // self.cmp_ratio,
                            ),
                        )
                        - token_offset
                    )
                    v1 = (
                        max(
                            0,
                            min(
                                actual_s2,
                                (
                                    actual_s2 * self.cmp_ratio
                                    + cmp_residual
                                    - task_used_q
                                    + query0
                                    + 2
                                )
                                // self.cmp_ratio,
                            ),
                        )
                        - token_offset
                    )
                    v2 = (
                        max(
                            0,
                            min(
                                actual_s2,
                                (
                                    actual_s2 * self.cmp_ratio
                                    + cmp_residual
                                    - task_used_q
                                    + query0
                                    + 3
                                )
                                // self.cmp_ratio,
                            ),
                        )
                        - token_offset
                    )
                self.streaming.mask_tail(workspace_token_offset, v0, v1, v2)
        else:
            self._compute_vector1_fallback(
                qk_handoff,
                weights,
                key_out,
                candidate_key_out,
                public_row_base,
                task_span_count,
                task_active_count,
                workspace_task_idx,
                token_offset,
                actual_s2,
                task_q_offset,
                task_used_q,
                cmp_residual,
                total_candidate_tokens,
                candidate_length_out,
                workspace_token_offset,
                tile_tokens,
                stream_hist,
                hist,
                cache_task,
                0,
                stream_total,
            )

    @jit
    def _compute_vector1_fallback(
        self,
        qk_handoff,
        weights,
        key_out,
        candidate_key_out,
        public_row_base,
        task_span_count,
        task_active_count,
        workspace_task_idx,
        token_offset,
        actual_s2,
        task_q_offset,
        task_used_q,
        cmp_residual,
        total_candidate_tokens,
        candidate_length_out,
        workspace_token_offset,
        tile_tokens,
        stream_hist,
        hist,
        cache_task,
        stream_task=0,
        stream_total=0,
    ):
        group_count = dtypes.int32(0)
        if dtypes.int32(self.subblock_idx) * self.vec_rows < task_active_count:
            group_count = dtypes.int32(self.n1)
        s2_len_v0 = actual_s2
        if self.mask_mode == 3:
            s2_len_v0 = max(
                0,
                min(
                    actual_s2,
                    (
                        actual_s2 * self.cmp_ratio
                        + cmp_residual
                        - task_used_q
                        + task_q_offset
                        + dtypes.int32(self.subblock_idx) * self.vec_rows
                        + 0
                        + 1
                    )
                    // self.cmp_ratio,
                ),
            )
        if dtypes.int32(self.subblock_idx) * self.vec_rows + 0 >= task_active_count:
            s2_len_v0 = 0
        s2_len_v0 = max(
            0, min(dtypes.int32(tile_tokens), s2_len_v0 - dtypes.int32(token_offset))
        )
        s2_len_v1 = actual_s2
        if self.mask_mode == 3:
            s2_len_v1 = max(
                0,
                min(
                    actual_s2,
                    (
                        actual_s2 * self.cmp_ratio
                        + cmp_residual
                        - task_used_q
                        + task_q_offset
                        + dtypes.int32(self.subblock_idx) * self.vec_rows
                        + 1
                        + 1
                    )
                    // self.cmp_ratio,
                ),
            )
        if dtypes.int32(self.subblock_idx) * self.vec_rows + 1 >= task_active_count:
            s2_len_v1 = 0
        s2_len_v1 = max(
            0, min(dtypes.int32(tile_tokens), s2_len_v1 - dtypes.int32(token_offset))
        )
        s2_len_v2 = actual_s2
        if self.mask_mode == 3:
            s2_len_v2 = max(
                0,
                min(
                    actual_s2,
                    (
                        actual_s2 * self.cmp_ratio
                        + cmp_residual
                        - task_used_q
                        + task_q_offset
                        + dtypes.int32(self.subblock_idx) * self.vec_rows
                        + 2
                        + 1
                    )
                    // self.cmp_ratio,
                ),
            )
        if dtypes.int32(self.subblock_idx) * self.vec_rows + 2 >= task_active_count:
            s2_len_v2 = 0
        s2_len_v2 = max(
            0, min(dtypes.int32(tile_tokens), s2_len_v2 - dtypes.int32(token_offset))
        )
        if self.cache_enabled and cache_task == 1:
            self._score_rows(
                qk_handoff,
                group_count,
                "cache",
                hist,
                workspace_token_offset,
                s2_len_v0,
                s2_len_v1,
                s2_len_v2,
            )
        else:
            if self.candidate_enabled:
                self._score_rows(qk_handoff, group_count, "candidate")
            elif self.byte_enabled and cache_task == 2:
                self._score_three_byte(
                    qk_handoff,
                    group_count,
                    hist,
                    workspace_token_offset,
                    s2_len_v0,
                    s2_len_v1,
                    s2_len_v2,
                )
            elif stream_hist == 0:
                self._score_rows(qk_handoff, group_count, "plain")
            else:
                self._score_rows(
                    qk_handoff,
                    group_count,
                    "hist",
                    hist,
                    workspace_token_offset,
                    s2_len_v0,
                    s2_len_v1,
                    s2_len_v2,
                )
            if self.candidate_enabled:
                cand_key_write = self.candidate_key_ub.next_write()
                for row_in_subblock in range_constexpr(self.vec_rows):
                    local_row = dtypes.int32(
                        self.subblock_idx * self.vec_rows + row_in_subblock
                    )
                    public_row = dtypes.int64(public_row_base) + dtypes.int64(local_row)
                    (
                        workspace_task_idx * self.query_rows
                        + self.subblock_idx * self.vec_rows
                        + row_in_subblock
                    )
                    valid_s2 = actual_s2
                    if self.mask_mode == 3:
                        query_in_batch = (
                            dtypes.int32(task_q_offset)
                            + dtypes.int32(self.subblock_idx) * self.vec_rows
                            + dtypes.int32(row_in_subblock)
                        )
                        valid_s2 = (
                            actual_s2 * self.cmp_ratio
                            + cmp_residual
                            - task_used_q
                            + query_in_batch
                            + 1
                        ) // self.cmp_ratio
                        if valid_s2 < 0:
                            valid_s2 = 0
                        if valid_s2 > actual_s2:
                            valid_s2 = actual_s2
                    if local_row >= task_active_count:
                        valid_s2 = 0
                    valid_candidate_count = (
                        valid_s2 + CANDIDATE_BLOCK_SIZE - 1
                    ) // CANDIDATE_BLOCK_SIZE
                    local_candidate_count = (
                        valid_candidate_count
                        - dtypes.int32(token_offset) // CANDIDATE_BLOCK_SIZE
                    )
                    if local_candidate_count < 0:
                        local_candidate_count = 0
                    if local_candidate_count > QK_COLUMNS // CANDIDATE_BLOCK_SIZE:
                        local_candidate_count = QK_COLUMNS // CANDIDATE_BLOCK_SIZE
                    partial_target = dtypes.int32(QK_COLUMNS // CANDIDATE_BLOCK_SIZE)
                    if valid_s2 % CANDIDATE_BLOCK_SIZE != 0:
                        if valid_s2 > dtypes.int32(token_offset):
                            if valid_s2 < dtypes.int32(token_offset) + QK_COLUMNS:
                                partial_target = (
                                    valid_s2 - dtypes.int32(token_offset)
                                ) // CANDIDATE_BLOCK_SIZE
                    if self.candidate_enabled:
                        with vf(mode="simd"):
                            candidate_count = QK_COLUMNS // CANDIDATE_BLOCK_SIZE
                            candidate_mask16 = rr.update_mask(
                                candidate_count, elem_bits=16
                            )[0]
                            candidate_idx = rr.vadds(
                                rr.vmuls(
                                    rr.varange(0, dtypes.uint16),
                                    CANDIDATE_BLOCK_SIZE,
                                    mask=candidate_mask16,
                                ),
                                row_in_subblock * QK_COLUMNS,
                                mask=candidate_mask16,
                            )
                            candidate_max = rr.vgather(
                                self.score_ub, candidate_idx, mask=candidate_mask16
                            )
                            for lane in range_constexpr(1, CANDIDATE_BLOCK_SIZE):
                                candidate_max = rr.vmax(
                                    candidate_max,
                                    rr.vgather(
                                        self.score_ub,
                                        rr.vadds(
                                            candidate_idx, lane, mask=candidate_mask16
                                        ),
                                        mask=candidate_mask16,
                                    ),
                                    mask=candidate_mask16,
                                )
                            candidate_global_idx = rr.varange(0, dtypes.uint16)
                            candidate_max = rr.vselect(
                                candidate_max,
                                rr.vreinterpret(
                                    rr.vdups(
                                        65408, dtypes.uint16, mask=candidate_mask16
                                    ),
                                    dtypes.bfloat16,
                                ),
                                cond_mask=rr.vlts(
                                    candidate_global_idx,
                                    local_candidate_count,
                                    mask=candidate_mask16,
                                ),
                            )
                            candidate_max = rr.vselect(
                                rr.vreinterpret(
                                    rr.vdups(
                                        32640, dtypes.uint16, mask=candidate_mask16
                                    ),
                                    dtypes.bfloat16,
                                ),
                                candidate_max,
                                cond_mask=rr.veqs(
                                    candidate_global_idx,
                                    partial_target,
                                    mask=candidate_mask16,
                                ),
                            )
                            candidate_bits16 = rr.vreinterpret(
                                candidate_max, dtypes.uint16
                            )
                            candidate_key = rr.vbitwise_xor(
                                candidate_bits16,
                                rr.vdups(
                                    BF16_SIGN_BIT, dtypes.uint16, mask=candidate_mask16
                                ),
                                mask=candidate_mask16,
                            )
                            candidate_negative = rr.veq(
                                rr.vbitwise_and(
                                    candidate_bits16,
                                    rr.vdups(
                                        BF16_SIGN_BIT,
                                        dtypes.uint16,
                                        mask=candidate_mask16,
                                    ),
                                    mask=candidate_mask16,
                                ),
                                rr.vdups(
                                    BF16_SIGN_BIT, dtypes.uint16, mask=candidate_mask16
                                ),
                                mask=candidate_mask16,
                            )
                            candidate_key = rr.vselect(
                                rr.vbitwise_xor(
                                    candidate_bits16,
                                    rr.vdups(
                                        UINT16_ALL_BITS,
                                        dtypes.uint16,
                                        mask=candidate_mask16,
                                    ),
                                    mask=candidate_mask16,
                                ),
                                candidate_key,
                                cond_mask=candidate_negative,
                            )
                            candidate_key = rr.vselect(
                                candidate_key,
                                rr.vdups(0, dtypes.uint16, mask=candidate_mask16),
                                cond_mask=rr.vlts(
                                    rr.varange(0, dtypes.uint16),
                                    local_candidate_count,
                                    mask=candidate_mask16,
                                ),
                            )
                            rr.vstore(
                                cand_key_write,
                                row_in_subblock * (QK_COLUMNS // CANDIDATE_BLOCK_SIZE),
                                candidate_key,
                                candidate_mask16,
                            )
                            rr.vmem_bar("vst_vld")
                    if self.candidate_enabled:
                        if local_row < task_span_count:
                            if token_offset == 0:
                                length = (
                                    valid_s2 + CANDIDATE_BLOCK_SIZE - 1
                                ) // CANDIDATE_BLOCK_SIZE
                                if length > self.candidate_topk_count:
                                    length = dtypes.int32(self.candidate_topk_count)
                                length_write = self.candidate_length_ub.next_write()
                                with vf(mode="simd"):
                                    mask = rr.update_mask(1, elem_bits=32)[0]
                                    rr.vstore(
                                        length_write,
                                        0,
                                        rr.vdups(length, dtypes.int32, mask=mask),
                                        mask,
                                    )
                                mem_copy(
                                    tile_slice(
                                        candidate_length_out, (1,), (public_row,)
                                    ),
                                    self.candidate_length_ub.next_read(),
                                )
            rows_to_store = (
                task_span_count - dtypes.int32(self.subblock_idx) * self.vec_rows
            )
            if rows_to_store < 0:
                rows_to_store = 0
            if rows_to_store > self.vec_rows:
                rows_to_store = self.vec_rows
            first_workspace_row = (
                workspace_task_idx * self.query_rows + self.subblock_idx * self.vec_rows
            )
            key_read = self.key_ub.next_read()
            if stream_task:
                self.streaming.capture(key_read, workspace_token_offset)
            else:
                mem_copy(
                    tile_slice(
                        _offset_view(
                            key_out, (first_workspace_row, workspace_token_offset)
                        ),
                        (self.vec_rows, QK_COLUMNS),
                        (0, 0),
                    ),
                    reinterpret(
                        key_read,
                        layout=make_layout((dtypes.int64(rows_to_store), QK_COLUMNS)),
                    ),
                )
            if self.candidate_enabled:
                mem_copy(
                    tile_slice(
                        _offset_view(
                            candidate_key_out,
                            (
                                first_workspace_row,
                                workspace_token_offset // CANDIDATE_BLOCK_SIZE,
                            ),
                        ),
                        (self.vec_rows, QK_COLUMNS // CANDIDATE_BLOCK_SIZE),
                        (0, 0),
                    ),
                    reinterpret(
                        self.candidate_key_ub.next_read(),
                        layout=make_layout(
                            (
                                dtypes.int64(rows_to_store),
                                QK_COLUMNS // CANDIDATE_BLOCK_SIZE,
                            )
                        ),
                    ),
                )

    @jit
    def _store_scored_row(
        self,
        score0,
        score1,
        row,
        valid,
        mask16,
        mode,
        key_write,
        hist,
        workspace_offset,
        zero_hist,
        reset16,
    ):
        score0 = rr.vadds(score0, 0.0, mask=mask16)
        if const_expr(mode == "candidate"):
            rr.vstore(self.score_ub, row * QK_COLUMNS + 0 * 128, score0, mask16)
        sortable_bits0_0 = _encode_bf16_sort_key(score0, mask16)
        if const_expr(mode == "cache"):
            rr.vstore(
                self.full_key_cache,
                row * CACHE_TOKENS + dtypes.int32(workspace_offset) + 0 * 128,
                sortable_bits0_0,
                mask16,
            )
        else:
            rr.vstore(
                key_write,
                row * QK_COLUMNS + 0 * 128,
                sortable_bits0_0,
                mask16,
            )
        score1 = rr.vadds(score1, 0.0, mask=mask16)
        if const_expr(mode == "candidate"):
            rr.vstore(self.score_ub, row * QK_COLUMNS + 1 * 128, score1, mask16)
        sortable_bits0_1 = _encode_bf16_sort_key(score1, mask16)
        if const_expr(mode == "cache"):
            rr.vstore(
                self.full_key_cache,
                row * CACHE_TOKENS + dtypes.int32(workspace_offset) + 1 * 128,
                sortable_bits0_1,
                mask16,
            )
        else:
            rr.vstore(
                key_write,
                row * QK_COLUMNS + 1 * 128,
                sortable_bits0_1,
                mask16,
            )
        if const_expr(mode == "hist" or mode == "cache"):
            histogram_low_reg = rr.vselect(
                zero_hist, rr.vload(hist, row * 512 + 0), cond_mask=reset16
            )
            histogram_high_reg = rr.vselect(
                zero_hist, rr.vload(hist, row * 512 + 256), cond_mask=reset16
            )
            key_high = rr.vshr(
                sortable_bits0_0,
                8,
                mask=mask16,
            )
            packed_high = rr.vpack(key_high, dtypes.uint8, part="lower")
            hist_mask = rr.update_mask(max(0, min(128, valid - 0)), elem_bits=8)[0]
            histogram_low_reg = rr.vhistogram_accumulate(
                histogram_low_reg,
                packed_high,
                mask=hist_mask,
                bin=0,
            )
            histogram_high_reg = rr.vhistogram_accumulate(
                histogram_high_reg,
                packed_high,
                mask=hist_mask,
                bin=1,
            )
            key_high = rr.vshr(
                sortable_bits0_1,
                8,
                mask=mask16,
            )
            packed_high = rr.vpack(key_high, dtypes.uint8, part="lower")
            hist_mask = rr.update_mask(max(0, min(128, valid - 128)), elem_bits=8)[0]
            histogram_low_reg = rr.vhistogram_accumulate(
                histogram_low_reg,
                packed_high,
                mask=hist_mask,
                bin=0,
            )
            histogram_high_reg = rr.vhistogram_accumulate(
                histogram_high_reg,
                packed_high,
                mask=hist_mask,
                bin=1,
            )
            rr.vstore(hist, row * 512 + 0, histogram_low_reg, mask16)
            rr.vstore(hist, row * 512 + 256, histogram_high_reg, mask16)

    @jit
    def _score_rows(
        self,
        qk_handoff,
        group_count,
        mode,
        hist=None,
        workspace_offset=0,
        valid0=0,
        valid1=0,
        valid2=0,
    ):
        key_write = None
        if const_expr(mode != "cache"):
            key_write = self.key_ub.next_write()
        with vf(mode="simd"):
            zero_hist = None
            reset16 = None
            mask16 = rr.update_mask(128, elem_bits=16)[0]
            if const_expr(mode == "hist" or mode == "cache"):
                zero_hist = rr.vdups(0, dtypes.uint16, mask=mask16)
                reset16 = rr.veqs(
                    rr.vdups(
                        dtypes.int32(workspace_offset < 512), dtypes.uint16, mask=mask16
                    ),
                    1,
                    mask=mask16,
                )
            (
                query0_s2_half0_acc_reg,
                query0_s2_half1_acc_reg,
                query1_s2_half0_acc_reg,
                query1_s2_half1_acc_reg,
                query2_s2_half0_acc_reg,
                query2_s2_half1_acc_reg,
            ) = self._accumulate_scores_vf(
                qk_handoff,
                group_count,
                mask16,
                False,
            )
            self._store_scored_row(
                query0_s2_half0_acc_reg,
                query0_s2_half1_acc_reg,
                0,
                valid0,
                mask16,
                mode,
                key_write,
                hist,
                workspace_offset,
                zero_hist,
                reset16,
            )
            self._store_scored_row(
                query1_s2_half0_acc_reg,
                query1_s2_half1_acc_reg,
                1,
                valid1,
                mask16,
                mode,
                key_write,
                hist,
                workspace_offset,
                zero_hist,
                reset16,
            )
            if const_expr(self.vec_rows == 3):
                self._store_scored_row(
                    query2_s2_half0_acc_reg,
                    query2_s2_half1_acc_reg,
                    2,
                    valid2,
                    mask16,
                    mode,
                    key_write,
                    hist,
                    workspace_offset,
                    zero_hist,
                    reset16,
                )
            rr.vmem_bar("vst_vld")

    @jit
    def _score_three_byte(
        self, qk_handoff, group_count, hist, workspace_offset, valid0, valid1, valid2
    ):
        key_write = self.key_ub.next_write()
        with vf(mode="simd"):
            mask16 = rr.update_mask(128, elem_bits=16)[0]
            zero_hist = rr.vdups(0, dtypes.uint16, mask=mask16)
            reset16 = rr.veqs(
                rr.vdups(
                    dtypes.int32(workspace_offset < 512), dtypes.uint16, mask=mask16
                ),
                1,
                mask=mask16,
            )
            _weight_mask = rr.update_mask(self.n1, elem_bits=32)[0]
            (
                query0_s2_half0_acc_reg,
                query0_s2_half1_acc_reg,
                query1_s2_half0_acc_reg,
                query1_s2_half1_acc_reg,
                query2_s2_half0_acc_reg,
                query2_s2_half1_acc_reg,
            ) = self._accumulate_scores_vf(
                qk_handoff,
                group_count,
                mask16,
                True,
            )
            self._store_byte_score_row(
                query0_s2_half0_acc_reg,
                query0_s2_half1_acc_reg,
                0,
                valid0,
                self.byte_cache0,
                key_write,
                hist,
                workspace_offset,
                mask16,
                zero_hist,
                reset16,
            )

            self._store_byte_score_row(
                query1_s2_half0_acc_reg,
                query1_s2_half1_acc_reg,
                1,
                valid1,
                self.byte_cache1,
                key_write,
                hist,
                workspace_offset,
                mask16,
                zero_hist,
                reset16,
            )

            self._store_byte_score_row(
                query2_s2_half0_acc_reg,
                query2_s2_half1_acc_reg,
                2,
                valid2,
                self.byte_cache2,
                key_write,
                hist,
                workspace_offset,
                mask16,
                zero_hist,
                reset16,
            )

            rr.vmem_bar("vst_vld")


class ExactTopK:
    """在单 trunk UB 候选池上执行精确 radix TopK，索引和值通过输出张量返回。"""

    pool_capacity = TOPK_SELECT_POOL_CAPACITY

    # 初始化精确 TopK的配置与缓冲区。
    def __init__(self, original, pool):
        self.original = original
        self.positions = _ub_buffer(
            pool, MemLoc.UB, (self.pool_capacity,), dtypes.uint16
        )
        self.histogram = _ub_buffer(pool, MemLoc.UB, (256,), dtypes.uint16)
        self.carrier = _ub_buffer(pool, MemLoc.UB, (64,), dtypes.int32)

    # 执行精确 TopK 并提供后续剪枝所需的值或阈值。
    @jit
    def select(self, input_values, count, output_indices, output_values, threshold):
        self.original._find_indices(
            input_values,
            count,
            self.positions,
            self.histogram,
            self.carrier,
            threshold,
        )
        with vf(mode="simd"):
            mask = rr.update_mask(128, elem_bits=16)[0]
            for chunk in dsl_range(TOPK_SELECT_TOPK // 128, unroll=1):
                positions = rr.vload(self.positions, chunk * 128)
                rr.vstore(output_indices, chunk * 128, positions, mask)
                rr.vstore(
                    output_values,
                    chunk * 128,
                    rr.vgather(input_values, positions, mask=mask),
                    mask,
                )
            rr.vmem_bar("vst_vld")


class TopKSelectPreprocessor:
    """Filter to one global candidate pool; exact selection is delegated."""

    # 初始化顺序输入 TopKSelect 预处理的配置与缓冲区。
    def __init__(self, original, prefetch=False):
        self.prefetch = bool(prefetch)
        pool = _UbPool()
        self.values = _ub_buffer(
            pool, MemLoc.UB, (TOPK_SELECT_POOL_CAPACITY,), dtypes.uint16
        )
        self.global_indices = _ub_buffer(
            pool, MemLoc.UB, (TOPK_SELECT_POOL_CAPACITY,), dtypes.int32
        )
        self.hits_positions = _ub_buffer(
            pool, MemLoc.UB, (TOPK_SELECT_GROUP_SIZE,), dtypes.uint16
        )
        self.topk = ExactTopK(original, pool)
        self.selected_indices = _ub_buffer(
            pool, MemLoc.UB, (TOPK_SELECT_TOPK,), dtypes.uint16
        )
        self.selected_values = _ub_buffer(
            pool, MemLoc.UB, (TOPK_SELECT_TOPK,), dtypes.uint16
        )
        self.selected_global_indices = _ub_buffer(
            pool, MemLoc.UB, (TOPK_SELECT_TOPK,), dtypes.int32
        )
        self.threshold = _ub_buffer(pool, MemLoc.UB, (16,), dtypes.uint16)
        self.output_indices = _ub_ring_buffer(
            pool, MemLoc.UB, (1, TOPK_SELECT_TOPK), dtypes.int32
        )
        self.output_bits = _ub_ring_buffer(
            pool, MemLoc.UB, (1, TOPK_SELECT_TOPK), dtypes.uint16
        )
        assert max(end for _, end in pool.regions) <= CACHED_QK_UB_BYTES
        depth = 2 if self.prefetch else 1
        self.input_slots = _ub_ring_buffer(
            pool,
            MemLoc.UB,
            (1, TOPK_SELECT_LOAD_TRUNK),
            dtypes.uint16,
            depth=depth,
            addr=CACHED_QK_UB_BYTES,
        )
        self.ub_end = CACHED_QK_UB_BYTES + depth * TOPK_SELECT_LOAD_TRUNK * 2
        assert self.ub_end <= 229376

    # 在当前 VF 内过滤一组输入并压缩相对位置；调用方保留分组和同步边界。
    @jit
    def _filter_group_vf(self, source, start, count, threshold, position_base):
        cursor = rr.vstore_unalign_begin(self.hits_positions)
        for chunk in dsl_range((count + 127) // 128, unroll=1):
            mask = rr.update_mask(count - chunk * 128, elem_bits=16)[0]
            keep = rr.vgt(rr.vload(source, start + chunk * 128), threshold, mask=mask)
            positions = rr.varange(chunk * 128, dtypes.uint16)
            packed = rr.vsqueeze_and_storeunalign_init(positions, mask=keep)
            rr.vsqueeze_and_storeunalign(
                self.hits_positions, position_base, packed, cursor
            )
        rr.vstore_unalign_post(self.hits_positions, position_base, cursor)

    # 从 GM 加载当前 trunk 到预处理缓冲区。
    @jit
    def _load_trunk(self, gm_keys, row, offset, count):
        slot = self.input_slots.next_write()
        mem_copy(
            reinterpret(slot, layout=make_layout((1, count))),
            gm_keys[row : row + 1, offset : offset + count],
        )

    # 加载初始候选并记录对应的原始位置。
    @jit
    def _seed(self, source, start, count, global_base, target):
        with vf(mode="simd"):
            mask16 = rr.update_mask(128, elem_bits=16)[0]
            for chunk in dsl_range(count // 128, unroll=1):
                rr.vstore(
                    self.values,
                    target + chunk * 128,
                    rr.vload(source, start + chunk * 128),
                    mask16,
                )
            mask32 = rr.update_mask(64, elem_bits=32)[0]
            for chunk in dsl_range(count // 64, unroll=1):
                rr.vstore(
                    self.global_indices,
                    target + chunk * 64,
                    rr.varange(global_base + chunk * 64, dtypes.int32),
                    mask32,
                )
            rr.vmem_bar("vst_vld")

    # 比较阈值并压缩命中元素，记录命中数量。
    @jit
    def _filter(self, source, start, count):
        with vf(mode="simd"):
            full = rr.update_mask(128, elem_bits=16)[0]
            threshold = rr.vdup(rr.vload(self.threshold, 0), mask=full)
            self._filter_group_vf(source, start, count, threshold, 0)
            rr.vmem_bar("vst_vld")

    # 将筛选命中值及其原始位置追加到候选池。
    @jit
    def _append(
        self, source, start, global_base, selected_count, target, position_base=0
    ):
        with vf(mode="simd"):
            full16 = rr.update_mask(128, elem_bits=16)[0]
            zero = rr.vdups(0, dtypes.uint16, mask=full16)
            for chunk in dsl_range((selected_count + 127) // 128, unroll=1):
                mask = rr.update_mask(selected_count - chunk * 128, elem_bits=16)[0]
                positions = rr.vload(self.hits_positions, position_base + chunk * 128)
                values = rr.vgather(
                    source, rr.vadds(positions, start, mask=mask), mask=mask
                )
                rr.vstore(
                    self.values,
                    target + chunk * 128,
                    rr.vselect(values, zero, cond_mask=mask),
                    full16,
                )
            full32 = rr.update_mask(64, elem_bits=32)[0]
            for chunk in dsl_range((selected_count + 127) // 128 * 2, unroll=1):
                mask = rr.update_mask(selected_count - chunk * 64, elem_bits=32)[0]
                positions = rr.vunpack(
                    rr.vload(self.hits_positions, position_base + chunk * 64),
                    dtypes.uint32,
                    part="lower",
                )
                indices = rr.vadds(
                    rr.vreinterpret(positions, dtypes.int32), global_base, mask=mask
                )
                rr.vstore(
                    self.global_indices,
                    target + chunk * 64,
                    rr.vselect(
                        indices, rr.vdups(-1, dtypes.int32, mask=full32), cond_mask=mask
                    ),
                    full32,
                )
            rr.vmem_bar("vst_vld")

    # 对候选池执行精确 TopK 收缩并更新阈值。
    @jit
    def compact(self, count):
        self.topk.select(
            self.values,
            count,
            self.selected_indices,
            self.selected_values,
            self.threshold,
        )
        with vf(mode="simd"):
            mask32 = rr.update_mask(64, elem_bits=32)[0]
            for chunk in dsl_range(TOPK_SELECT_TOPK // 64, unroll=1):
                positions = rr.vunpack(
                    rr.vload(self.selected_indices, chunk * 64),
                    dtypes.uint32,
                    part="lower",
                )
                rr.vstore(
                    self.selected_global_indices,
                    chunk * 64,
                    rr.vgather(self.global_indices, positions, mask=mask32),
                    mask32,
                )
            rr.vmem_bar("vst_vld")
        with vf(mode="simd"):
            mask16 = rr.update_mask(128, elem_bits=16)[0]
            for chunk in dsl_range(TOPK_SELECT_TOPK // 128, unroll=1):
                rr.vstore(
                    self.values,
                    chunk * 128,
                    rr.vload(self.selected_values, chunk * 128),
                    mask16,
                )
            mask32 = rr.update_mask(64, elem_bits=32)[0]
            for chunk in dsl_range(TOPK_SELECT_TOPK // 64, unroll=1):
                rr.vstore(
                    self.global_indices,
                    chunk * 64,
                    rr.vload(self.selected_global_indices, chunk * 64),
                    mask32,
                )
            rr.vmem_bar("vst_vld")

    # 遍历输入并过滤候选，控制候选池容量。
    @jit
    def preprocess(self, gm_keys, length, row):
        pool_used_elements = dtypes.int64(0)
        group_number = dtypes.int64(0)
        if self.prefetch:
            self._load_trunk(gm_keys, row, 0, min(length, TOPK_SELECT_LOAD_TRUNK))
        for trunk in dsl_range(
            (length + TOPK_SELECT_LOAD_TRUNK - 1) // TOPK_SELECT_LOAD_TRUNK, unroll=1
        ):
            offset = trunk * TOPK_SELECT_LOAD_TRUNK
            count = length - offset
            if count > TOPK_SELECT_LOAD_TRUNK:
                count = dtypes.int64(TOPK_SELECT_LOAD_TRUNK)
            if not self.prefetch:
                self._load_trunk(gm_keys, row, offset, count)
            source = self.input_slots.next_read()
            if self.prefetch:
                next_offset = offset + TOPK_SELECT_LOAD_TRUNK
                if next_offset < length:
                    next_count = length - next_offset
                    if next_count > TOPK_SELECT_LOAD_TRUNK:
                        next_count = dtypes.int64(TOPK_SELECT_LOAD_TRUNK)
                    # Start next transfer before the current trunk's Vector work.
                    self._load_trunk(gm_keys, row, next_offset, next_count)
            for group in dsl_range(
                (count + TOPK_SELECT_GROUP_SIZE - 1) // TOPK_SELECT_GROUP_SIZE, unroll=1
            ):
                start = group * TOPK_SELECT_GROUP_SIZE
                group_count = count - start
                if group_count > TOPK_SELECT_GROUP_SIZE:
                    group_count = dtypes.int64(TOPK_SELECT_GROUP_SIZE)
                if group_number < 2:
                    self._seed(
                        source, start, group_count, offset + start, pool_used_elements
                    )
                    pool_used_elements += group_count
                    if group_number == 1:
                        self.compact(pool_used_elements)
                        pool_used_elements = dtypes.int64(TOPK_SELECT_TOPK)
                else:
                    self._filter(source, start, group_count)
                    vec_sync_notify(PIPE.V, PIPE.S, 6)
                    vec_sync_wait(PIPE.V, PIPE.S, 6)
                    selected_count = dtypes.int64(get_squeeze_status()) // 2
                    self._append(
                        source,
                        start,
                        offset + start,
                        selected_count,
                        pool_used_elements,
                    )
                    pool_used_elements += (
                        (selected_count + POOL_ALIGNMENT_ELEMENTS - 1)
                        // POOL_ALIGNMENT_ELEMENTS
                    ) * POOL_ALIGNMENT_ELEMENTS
                    if (
                        pool_used_elements > TOPK_SELECT_COMPACT_LIMIT
                        and offset + start + group_count < length
                    ):
                        self.compact(pool_used_elements)
                        pool_used_elements = dtypes.int64(TOPK_SELECT_TOPK)
                group_number += 1
        return pool_used_elements

    # 将候选池局部索引映射为原始输入索引并写出。
    @jit
    def write_indices(self, local_indices, gm_indices, index_base):
        out = self.output_indices.next_write()
        with vf(mode="simd"):
            mask = rr.update_mask(64, elem_bits=32)[0]
            for chunk in dsl_range(TOPK_SELECT_TOPK // 64, unroll=1):
                positions = rr.vunpack(
                    rr.vload(local_indices, chunk * 64), dtypes.uint32, part="lower"
                )
                indices = rr.vgather(self.global_indices, positions, mask=mask)
                rr.vstore(
                    out, chunk * 64, rr.vadds(indices, index_base, mask=mask), mask
                )
            rr.vmem_bar("vst_vld")
        mem_copy(gm_indices, self.output_indices.next_read())

    # 选择单行 TopK 的适用路径并写出结果。
    @jit
    def select_row(
        self,
        gm_keys,
        workspace_row,
        length,
        gm_indices,
        gm_bits,
        index_base,
        need_values,
    ):
        candidate_count = self.preprocess(gm_keys, length, workspace_row)
        self.topk.select(
            self.values,
            candidate_count,
            self.selected_indices,
            self.selected_values,
            self.threshold,
        )
        self.write_indices(self.selected_indices, gm_indices, index_base)
        if need_values != 0:
            out = self.output_bits.next_write()
            with vf(mode="simd"):
                mask = rr.update_mask(128, elem_bits=16)[0]
                for chunk in dsl_range(TOPK_SELECT_TOPK // 128, unroll=1):
                    keys = rr.vload(self.selected_values, chunk * 128)
                    positive = rr.vges(keys, 32768, mask=mask)
                    bits = rr.vselect(
                        rr.vbitwise_xor(
                            keys, rr.vdups(0x8000, dtypes.uint16, mask=mask), mask=mask
                        ),
                        rr.vbitwise_xor(
                            keys, rr.vdups(0xFFFF, dtypes.uint16, mask=mask), mask=mask
                        ),
                        cond_mask=positive,
                    )
                    rr.vstore(out, chunk * 128, bits, mask)
                rr.vmem_bar("vst_vld")
            mem_copy(gm_bits, self.output_bits.next_read())


class DmaTopKSelectExactTopK(ExactTopK):
    """DMA 候选池复用精确选择工作区，额外提供 VF 内调用接口。"""

    pool_capacity = DMA_TOPK_SELECT_POOL_CAPACITY

    # 在当前 VF 内执行精确 TopK，供筛选流水复用。
    @jit
    def select_vf(self, input_values, count, output_indices, output_values, threshold):
        self.original._find_indices_vf(
            input_values, count, self.positions, self.histogram, self.carrier, threshold
        )
        rr.vmem_bar("vst_vld")
        mask = rr.update_mask(128, elem_bits=16)[0]
        for chunk in dsl_range(DMA_TOPK_SELECT_TOPK // 128, unroll=1):
            positions = rr.vload(self.positions, chunk * 128)
            rr.vstore(output_indices, chunk * 128, positions, mask)
            rr.vstore(
                output_values,
                chunk * 128,
                rr.vgather(input_values, positions, mask=mask),
                mask,
            )
        rr.vmem_bar("vst_vld")

    # 执行精确 TopK 并提供后续剪枝所需的值或阈值。
    @jit
    def select(self, input_values, count, output_indices, output_values, threshold):
        with vf(mode="simd"):
            self.select_vf(
                input_values, count, output_indices, output_values, threshold
            )


class DmaTopKSelectPreprocessor(TopKSelectPreprocessor):
    """Filter to one global candidate pool; exact selection is delegated."""

    # 初始化伪随机页 DMA TopKSelect 预处理的配置与缓冲区。
    def __init__(self, original, prefetch=False):
        self.prefetch = bool(prefetch)
        pool = _UbPool()
        self.values = _ub_buffer(
            pool, MemLoc.UB, (DMA_TOPK_SELECT_POOL_CAPACITY,), dtypes.uint16
        )
        self.global_indices = _ub_buffer(
            pool, MemLoc.UB, (DMA_TOPK_SELECT_POOL_CAPACITY,), dtypes.int32
        )
        self.hits_positions = None
        self.page_map = _ub_buffer(pool, MemLoc.UB, (256,), dtypes.int32)
        self.filter_counts = _ub_buffer(pool, MemLoc.UB, (32,), dtypes.uint32)
        self.topk = DmaTopKSelectExactTopK(original, pool)
        self.hits_positions = self.topk.positions
        self.selected_indices = _ub_buffer(
            pool, MemLoc.UB, (DMA_TOPK_SELECT_TOPK,), dtypes.uint16
        )
        self.selected_values = _ub_buffer(
            pool, MemLoc.UB, (DMA_TOPK_SELECT_TOPK,), dtypes.uint16
        )
        self.selected_global_indices = _ub_buffer(
            pool, MemLoc.UB, (DMA_TOPK_SELECT_TOPK,), dtypes.int32
        )
        self.threshold = _ub_buffer(pool, MemLoc.UB, (16,), dtypes.uint16)
        self.output_indices = _ub_ring_buffer(
            pool, MemLoc.UB, (1, DMA_TOPK_SELECT_TOPK), dtypes.int32
        )
        depth = 2 if self.prefetch else 1
        input_base = (max((end for _, end in pool.regions)) + 127) // 128 * 128
        self.input_slots = _ub_ring_buffer(
            pool,
            MemLoc.UB,
            (1, DMA_TOPK_SELECT_GROUP_SIZE),
            dtypes.uint16,
            depth=depth,
            addr=input_base,
        )
        self.ub_end = input_base + depth * DMA_TOPK_SELECT_GROUP_SIZE * 2
        assert self.ub_end <= 221184, self.ub_end

    # 将逻辑页号映射到伪随机访问的物理页号。
    @jit
    def _physical_page(self, logical_page, length, seed0, seed1):
        """TopKSelect's page permutation; seed is derived from this query's input."""
        num_pages = (
            length + DMA_TOPK_SELECT_PAGE_SIZE - 1
        ) // DMA_TOPK_SELECT_PAGE_SIZE
        start = logical_page * DMA_TOPK_SELECT_PAGE_SIZE
        count = length - start
        if count > DMA_TOPK_SELECT_PAGE_SIZE:
            count = dtypes.int64(DMA_TOPK_SELECT_PAGE_SIZE)
        if True:
            if num_pages > 2:
                if logical_page == 1:
                    start = (length - DMA_TOPK_SELECT_PAGE_SIZE + 127) // 128 * 128
                    count = length - start
                elif logical_page == num_pages - 1:
                    start = (num_pages - 2) * DMA_TOPK_SELECT_PAGE_SIZE
                    tail_start = (length - DMA_TOPK_SELECT_PAGE_SIZE + 127) // 128 * 128
                    count = tail_start - start
                else:
                    size = dtypes.uint32(num_pages - 2)
                    value = dtypes.uint32(logical_page)
                    if logical_page >= 2:
                        value = dtypes.uint32(logical_page - 1)
                    value = (value * dtypes.uint32(65537) + seed0) % size
                    if value <= seed1:
                        value = seed1 - value
                    else:
                        value = size + seed1 - value
                    page = (value * dtypes.uint32(65521) + seed1) % size
                    start = dtypes.int64(page) * DMA_TOPK_SELECT_PAGE_SIZE
                    count = dtypes.int64(DMA_TOPK_SELECT_PAGE_SIZE)
        return (start, count)

    # 在 VF 内计算页号映射所需的取模。
    @jit
    def _mod_pages_vf(self, value, size, mask):
        denominator = rr.vdups(size, dtypes.uint32, mask=mask)
        value_f = rr.vcast(
            rr.vreinterpret(value, dtypes.int32), dtypes.float32, mask=mask
        )
        denominator_f = rr.vcast(
            rr.vreinterpret(denominator, dtypes.int32), dtypes.float32, mask=mask
        )
        quotient_i = rr.vcast(
            rr.vdiv(value_f, denominator_f, mask=mask, precision_mode=True),
            dtypes.int32,
            mask=mask,
            rounding="rz",
        )
        quotient = rr.vreinterpret(quotient_i, dtypes.uint32)
        product = rr.vmul(quotient, denominator, mask=mask)
        remainder = rr.vsub(value, product, mask=mask)
        remainder = rr.vselect(
            rr.vadd(remainder, denominator, mask=mask),
            remainder,
            cond_mask=rr.vgt(product, value, mask=mask),
        )
        remainder = rr.vselect(
            rr.vsub(remainder, denominator, mask=mask),
            remainder,
            cond_mask=rr.vge(remainder, denominator, mask=mask),
        )
        return remainder

    # 初始化伪随机页访问映射。
    @jit
    def _init_page_map(self, length, seed0, seed1):
        num_pages = (
            length + DMA_TOPK_SELECT_PAGE_SIZE - 1
        ) // DMA_TOPK_SELECT_PAGE_SIZE
        with vf(mode="simd"):
            for chunk in dsl_range((num_pages + 63) // 64, unroll=1):
                mask = rr.update_mask(num_pages - chunk * 64, elem_bits=32)[0]
                logical = rr.varange(chunk * 64, dtypes.uint32)
                offsets = rr.vmuls(logical, DMA_TOPK_SELECT_PAGE_SIZE, mask=mask)
                if num_pages > 2:
                    value = rr.vselect(
                        logical,
                        rr.vadds(logical, 4294967295, mask=mask),
                        cond_mask=rr.veqs(logical, 0, mask=mask),
                    )
                    value = rr.vadds(
                        rr.vmuls(value, 65537, mask=mask), seed0, mask=mask
                    )
                    value = self._mod_pages_vf(
                        value, dtypes.uint32(num_pages - 2), mask
                    )
                    pivot = rr.vdups(seed1, dtypes.uint32, mask=mask)
                    reflected = rr.vsub(pivot, value, mask=mask)
                    value = rr.vselect(
                        reflected,
                        rr.vadds(reflected, dtypes.uint32(num_pages - 2), mask=mask),
                        cond_mask=rr.vge(pivot, value, mask=mask),
                    )
                    value = rr.vadds(
                        rr.vmuls(value, 65521, mask=mask), seed1, mask=mask
                    )
                    value = self._mod_pages_vf(
                        value, dtypes.uint32(num_pages - 2), mask
                    )
                    offsets = rr.vmuls(value, DMA_TOPK_SELECT_PAGE_SIZE, mask=mask)
                    offsets = rr.vselect(
                        rr.vdups(
                            dtypes.uint32(
                                (length - DMA_TOPK_SELECT_PAGE_SIZE + 127) // 128 * 128
                            ),
                            dtypes.uint32,
                            mask=mask,
                        ),
                        offsets,
                        cond_mask=rr.veqs(logical, 1, mask=mask),
                    )
                    offsets = rr.vselect(
                        rr.vdups(
                            dtypes.uint32((num_pages - 2) * DMA_TOPK_SELECT_PAGE_SIZE),
                            dtypes.uint32,
                            mask=mask,
                        ),
                        offsets,
                        cond_mask=rr.veqs(
                            logical, dtypes.uint32(num_pages - 1), mask=mask
                        ),
                    )
                rr.vstore(
                    self.page_map,
                    chunk * 64,
                    rr.vreinterpret(offsets, dtypes.int32),
                    mask,
                )
            rr.vmem_bar("vst_vld")

    # 按页映射通过 DMA 加载本批输入。
    @jit
    def _load_pages(
        self, gm_keys, row, logical_begin, page_limit, length, seed0, seed1
    ):
        """Pack up to four physical pages into one double-buffered scan block."""
        slot = self.input_slots.next_write()
        num_pages = (
            length + DMA_TOPK_SELECT_PAGE_SIZE - 1
        ) // DMA_TOPK_SELECT_PAGE_SIZE
        for page in dsl_range(page_limit, unroll=1):
            logical_page = logical_begin + page
            if logical_page < num_pages:
                offset = dtypes.int64(self.page_map[logical_page])
                count = dtypes.int64(DMA_TOPK_SELECT_PAGE_SIZE)
                if logical_page == 1:
                    count = (
                        length - (length - DMA_TOPK_SELECT_PAGE_SIZE + 127) // 128 * 128
                    )
                elif logical_page == num_pages - 1:
                    count = (length - DMA_TOPK_SELECT_PAGE_SIZE + 127) // 128 * 128 - (
                        num_pages - 2
                    ) * DMA_TOPK_SELECT_PAGE_SIZE
                if count == DMA_TOPK_SELECT_PAGE_SIZE:
                    mem_copy(
                        tile_slice(slot, (1, DMA_TOPK_SELECT_PAGE_SIZE), (0, page)),
                        tile_slice(
                            gm_keys[row : row + 1, offset:],
                            (1, DMA_TOPK_SELECT_PAGE_SIZE),
                            (0, 0),
                        ),
                    )
                else:
                    mem_copy(
                        slot[
                            0:1,
                            page * DMA_TOPK_SELECT_PAGE_SIZE : page
                            * DMA_TOPK_SELECT_PAGE_SIZE
                            + count,
                        ],
                        gm_keys[row : row + 1, offset : offset + count],
                    )

    # 加载初始候选并记录对应的原始位置。
    @jit
    def _seed(self, source, start, count, global_base, target):
        with vf(mode="simd"):
            mask16 = rr.update_mask(128, elem_bits=16)[0]
            for chunk in dsl_range((count + 127) // 128, unroll=1):
                mask16 = rr.update_mask(count - chunk * 128, elem_bits=16)[0]
                rr.vstore(
                    self.values,
                    target + chunk * 128,
                    rr.vload(source, start + chunk * 128),
                    mask16,
                )
            mask32 = rr.update_mask(64, elem_bits=32)[0]
            for chunk in dsl_range((count + 63) // 64, unroll=1):
                mask32 = rr.update_mask(count - chunk * 64, elem_bits=32)[0]
                rr.vstore(
                    self.global_indices,
                    target + chunk * 64,
                    rr.varange(global_base + chunk * 64, dtypes.int32),
                    mask32,
                )
            rr.vmem_bar("vst_vld")

    # 对候选池执行精确 TopK 收缩并更新阈值。
    @jit
    def compact(self, count):
        with vf(mode="simd"):
            self.topk.original._find_indices_vf(
                self.values,
                count,
                self.topk.positions,
                self.topk.histogram,
                self.topk.carrier,
                self.threshold,
            )
            rr.vmem_bar("vst_vld")
            mask16 = rr.update_mask(128, elem_bits=16)[0]
            for chunk in dsl_range(DMA_TOPK_SELECT_TOPK // 128, unroll=1):
                positions = rr.vload(self.topk.positions, chunk * 128)
                rr.vstore(
                    self.selected_values,
                    chunk * 128,
                    rr.vgather(self.values, positions, mask=mask16),
                    mask16,
                )
            mask32 = rr.update_mask(64, elem_bits=32)[0]
            for chunk in dsl_range(DMA_TOPK_SELECT_TOPK // 64, unroll=1):
                positions = rr.vunpack(
                    rr.vload(self.topk.positions, chunk * 64),
                    dtypes.uint32,
                    part="lower",
                )
                rr.vstore(
                    self.selected_global_indices,
                    chunk * 64,
                    rr.vgather(self.global_indices, positions, mask=mask32),
                    mask32,
                )
            rr.vmem_bar("vst_vld")
            for chunk in dsl_range(DMA_TOPK_SELECT_TOPK // 128, unroll=1):
                rr.vstore(
                    self.values,
                    chunk * 128,
                    rr.vload(self.selected_values, chunk * 128),
                    mask16,
                )
            for chunk in dsl_range(DMA_TOPK_SELECT_TOPK // 64, unroll=1):
                rr.vstore(
                    self.global_indices,
                    chunk * 64,
                    rr.vload(self.selected_global_indices, chunk * 64),
                    mask32,
                )
            rr.vmem_bar("vst_vld")

    # 完成候选池最终选择，并恢复原始输入索引。
    @jit
    def finish_indices(self, count, gm_indices, index_base):
        out = self.output_indices.next_write()
        with vf(mode="simd"):
            self.topk.original._find_indices_vf(
                self.values,
                count,
                self.topk.positions,
                self.topk.histogram,
                self.topk.carrier,
            )
            rr.vmem_bar("vst_vld")
            mask32 = rr.update_mask(64, elem_bits=32)[0]
            for chunk in dsl_range(DMA_TOPK_SELECT_TOPK // 64, unroll=1):
                positions = rr.vunpack(
                    rr.vload(self.topk.positions, chunk * 64),
                    dtypes.uint32,
                    part="lower",
                )
                virtual = rr.vgather(self.global_indices, positions, mask=mask32)
                page = rr.vshr(
                    rr.vreinterpret(virtual, dtypes.uint32),
                    DMA_TOPK_SELECT_PAGE_BITS,
                    mask=mask32,
                )
                within = rr.vbitwise_and(
                    virtual,
                    rr.vdups(DMA_TOPK_SELECT_PAGE_SIZE - 1, dtypes.int32, mask=mask32),
                    mask=mask32,
                )
                physical = rr.vadd(
                    rr.vgather(self.page_map, page, mask=mask32), within, mask=mask32
                )
                rr.vstore(
                    out, chunk * 64, rr.vadds(physical, index_base, mask=mask32), mask32
                )
            rr.vmem_bar("vst_vld")
        mem_copy(gm_indices, self.output_indices.next_read())

    # 在同一个 VF 内完成一批四组阈值过滤。
    @jit
    def _filter_batch(self, source, start, count):
        with vf(mode="simd"):
            full = rr.update_mask(128, elem_bits=16)[0]
            threshold = rr.vdup(rr.vload(self.threshold, 0), mask=full)
            for group in dsl_range(
                (count + DMA_TOPK_SELECT_GROUP_SIZE - 1) // DMA_TOPK_SELECT_GROUP_SIZE,
                unroll=1,
            ):
                group_count = count - group * DMA_TOPK_SELECT_GROUP_SIZE
                if group_count > DMA_TOPK_SELECT_GROUP_SIZE:
                    group_count = dtypes.int64(DMA_TOPK_SELECT_GROUP_SIZE)
                self._filter_group_vf(
                    source,
                    start + group * DMA_TOPK_SELECT_GROUP_SIZE,
                    group_count,
                    threshold,
                    group * DMA_TOPK_SELECT_GROUP_SIZE,
                )
            rr.vmem_bar("vst_vld")

    # 遍历输入并过滤候选，控制候选池容量。
    @jit
    def preprocess(self, gm_keys, length, row):
        num_pages = (
            length + DMA_TOPK_SELECT_PAGE_SIZE - 1
        ) // DMA_TOPK_SELECT_PAGE_SIZE
        low = dtypes.uint32(gm_keys[row, 0])
        high = dtypes.uint32(gm_keys[row, 1])
        seed0 = (low ^ dtypes.uint32(length)) % dtypes.uint32(65536)
        seed1 = (
            (high ^ dtypes.uint32(length) // dtypes.uint32(65536))
            % dtypes.uint32(65536)
            % dtypes.uint32(max(num_pages - 2, 1))
        )
        self._init_page_map(length, seed0, seed1)
        self._load_pages(gm_keys, row, 0, 2, length, seed0, seed1)
        source = self.input_slots.next_read()
        count1 = length - (length - DMA_TOPK_SELECT_PAGE_SIZE + 127) // 128 * 128
        self._seed(source, 0, DMA_TOPK_SELECT_PAGE_SIZE, 0, 0)
        self._seed(
            source,
            DMA_TOPK_SELECT_PAGE_SIZE,
            count1,
            DMA_TOPK_SELECT_PAGE_SIZE,
            DMA_TOPK_SELECT_PAGE_SIZE,
        )
        self.compact(DMA_TOPK_SELECT_PAGE_SIZE + count1)
        pool_used_elements = dtypes.int64(DMA_TOPK_SELECT_TOPK)
        stable_pages = num_pages - 2
        batches = (stable_pages + 3) // 4
        self._load_pages(gm_keys, row, 2, 4, length, seed0, seed1)
        for block in dsl_range(batches, unroll=1):
            logical_begin = 2 + block * 4
            source = self.input_slots.next_read()
            if logical_begin + 4 < num_pages:
                self._load_pages(
                    gm_keys, row, logical_begin + 4, 4, length, seed0, seed1
                )
            pages = num_pages - logical_begin
            if pages > 4:
                pages = dtypes.int64(4)
            count = pages * DMA_TOPK_SELECT_PAGE_SIZE
            if logical_begin + pages == num_pages:
                tail = (length - DMA_TOPK_SELECT_PAGE_SIZE + 127) // 128 * 128 - (
                    num_pages - 2
                ) * DMA_TOPK_SELECT_PAGE_SIZE
                count = (pages - 1) * DMA_TOPK_SELECT_PAGE_SIZE + tail
            self._filter_batch(source, 0, count)
            selected_count = dtypes.int64(get_squeeze_status()) // 2
            self._append(
                source,
                0,
                logical_begin * DMA_TOPK_SELECT_PAGE_SIZE,
                selected_count,
                pool_used_elements,
                0,
            )
            pool_used_elements += (
                (selected_count + POOL_ALIGNMENT_ELEMENTS - 1)
                // POOL_ALIGNMENT_ELEMENTS
                * POOL_ALIGNMENT_ELEMENTS
            )
            if (
                pool_used_elements > DMA_TOPK_SELECT_COMPACT_LIMIT
                and block + 1 < batches
            ):
                self.compact(pool_used_elements)
                pool_used_elements = dtypes.int64(DMA_TOPK_SELECT_TOPK)
        return pool_used_elements

    # 将候选池局部索引映射为原始输入索引并写出。
    @jit
    def write_indices(self, local_indices, gm_indices):
        out = self.output_indices.next_write()
        with vf(mode="simd"):
            mask = rr.update_mask(64, elem_bits=32)[0]
            for chunk in dsl_range(DMA_TOPK_SELECT_TOPK // 64, unroll=1):
                positions = rr.vunpack(
                    rr.vload(local_indices, chunk * 64), dtypes.uint32, part="lower"
                )
                rr.vstore(
                    out,
                    chunk * 64,
                    rr.vgather(self.global_indices, positions, mask=mask),
                    mask,
                )
            rr.vmem_bar("vst_vld")
        mem_copy(gm_indices, self.output_indices.next_read())

    # 选择单行 TopK 的适用路径并写出结果。
    @jit
    def select_row(self, gm_keys, workspace_row, length, gm_indices, index_base):
        candidate_count = self.preprocess(gm_keys, length, workspace_row)
        self.finish_indices(candidate_count, gm_indices, index_base)


class StreamingLdTopK:
    """Private CV-live workspace: exact local TopK with online filtering."""

    # 初始化流式 LD TopK的配置与缓冲区。
    def __init__(self, original):
        self.original = original
        self.root_arena = _ub_buffer(
            _UbPool(),
            MemLoc.UB,
            (STREAM_ARENA_OFFSET + STREAM_ARENA_BYTES,),
            dtypes.uint8,
            addr=0,
        )
        self.arena = reinterpret(
            self.root_arena,
            dtype=dtypes.uint8,
            offset=STREAM_ARENA_OFFSET,
            layout=make_layout((STREAM_ARENA_BYTES,)),
        )
        self.values = reinterpret(
            self.arena,
            dtype=dtypes.uint16,
            offset=0,
            layout=make_layout((3, STREAM_POOL_TOKENS)),
        )
        self.indices = reinterpret(
            self.arena,
            dtype=dtypes.uint16,
            offset=STREAM_INDICES_OFFSET,
            layout=make_layout((3, STREAM_POOL_TOKENS)),
        )
        self.positions = reinterpret(
            self.arena,
            dtype=dtypes.uint16,
            offset=STREAM_POSITIONS_OFFSET,
            layout=make_layout((STREAM_POOL_TOKENS,)),
        )
        self.histogram = reinterpret(
            self.arena,
            dtype=dtypes.uint16,
            offset=STREAM_HISTOGRAM_OFFSET,
            layout=make_layout((256,)),
        )
        self.carrier = reinterpret(
            self.arena,
            dtype=dtypes.int32,
            offset=STREAM_CARRIER_OFFSET,
            layout=make_layout((64,)),
        )
        self.thresholds = reinterpret(
            self.arena,
            dtype=dtypes.uint16,
            offset=STREAM_THRESHOLDS_OFFSET,
            layout=make_layout((3, 16)),
        )
        self.counts = reinterpret(
            self.arena,
            dtype=dtypes.int32,
            offset=STREAM_COUNTS_OFFSET,
            layout=make_layout((3, 8)),
        )
        self.packed = reinterpret(
            self.arena,
            dtype=dtypes.uint16,
            offset=STREAM_PACKED_OFFSET,
            layout=make_layout((STREAM_PAGE_TOKENS,)),
        )
        self.tmp_values = reinterpret(
            self.arena,
            dtype=dtypes.uint16,
            offset=STREAM_TMP_VALUES_OFFSET,
            layout=make_layout((512,)),
        )
        self.tmp_indices = reinterpret(
            self.arena,
            dtype=dtypes.uint16,
            offset=STREAM_TMP_INDICES_OFFSET,
            layout=make_layout((512,)),
        )
        self.stage = reinterpret(
            self.arena,
            dtype=dtypes.uint16,
            offset=STREAM_STAGE_OFFSET,
            layout=make_layout((3, STREAM_PAGE_TOKENS)),
        )
        self.out_indices = original.output_idx_stage
        self.out_bits = original.output_value_bits_stage

    # 重置当前 LD 任务的候选池和阈值状态。
    @jit
    def reset(self):
        with vf(mode="simd"):
            m32 = rr.update_mask(24, elem_bits=32)[0]
            rr.vstore(self.counts, 0, rr.vdups(0, dtypes.int32, mask=m32), m32)
            m16 = rr.update_mask(48, elem_bits=16)[0]
            rr.vstore(self.thresholds, 0, rr.vdups(0, dtypes.uint16, mask=m16), m16)
            rr.vmem_bar("vst_vld")

    # 对候选池执行精确 TopK 收缩并更新阈值。
    @jit
    def compact(self, row, count):
        vals = tile_slice(self.values, (1, STREAM_POOL_TOKENS), (row, 0))
        ids = tile_slice(self.indices, (1, STREAM_POOL_TOKENS), (row, 0))
        threshold = tile_slice(self.thresholds, (1, 16), (row, 0))
        self.original._find_indices(
            vals, count, self.positions, self.histogram, self.carrier, threshold
        )
        with vf(mode="simd"):
            m16 = rr.update_mask(128, elem_bits=16)[0]
            for c in dsl_range(4, unroll=1):
                pos = rr.vload(self.positions, c * 128)
                rr.vstore(
                    self.tmp_values, c * 128, rr.vgather(vals, pos, mask=m16), m16
                )
            for c in dsl_range(4, unroll=1):
                pos = rr.vload(self.positions, c * 128)
                rr.vstore(
                    self.tmp_indices, c * 128, rr.vgather(ids, pos, mask=m16), m16
                )
            rr.vmem_bar("vst_vld")
        with vf(mode="simd"):
            m16 = rr.update_mask(128, elem_bits=16)[0]
            for c in dsl_range(4, unroll=1):
                rr.vstore(vals, c * 128, rr.vload(self.tmp_values, c * 128), m16)
            for c in dsl_range(4, unroll=1):
                rr.vstore(ids, c * 128, rr.vload(self.tmp_indices, c * 128), m16)
            rr.vmem_bar("vst_vld")

    # 缓存当前 CV 评分块，供流式 LD TopK 消费。
    @jit
    def capture(self, source, offset):
        with vf(mode="simd"):
            mask = rr.update_mask(128, elem_bits=16)[0]
            for row in dsl_range(3, unroll=1):
                for c in dsl_range(2, unroll=1):
                    rr.vstore(
                        self.stage,
                        row * STREAM_PAGE_TOKENS
                        + offset % STREAM_PAGE_TOKENS
                        + c * 128,
                        rr.vload(source, row * 256 + c * 128),
                        mask,
                    )
            rr.vmem_bar("vst_vld")

    # 屏蔽各 query 尾部不可见的 token。
    @jit
    def mask_tail(self, offset, v0, v1, v2):
        with vf(mode="simd"):
            mask = rr.update_mask(128, elem_bits=16)[0]
            zero = rr.vdups(0, dtypes.uint16, mask=mask)
            for row in range_constexpr(3):
                valid = v0 if row == 0 else v1 if row == 1 else v2
                for c in dsl_range(2, unroll=1):
                    pos = (
                        row * STREAM_PAGE_TOKENS + offset % STREAM_PAGE_TOKENS + c * 128
                    )
                    good = rr.vlts(rr.varange(c * 128, dtypes.uint16), valid, mask=mask)
                    rr.vstore(
                        self.stage,
                        pos,
                        rr.vselect(rr.vload(self.stage, pos), zero, cond_mask=good),
                        mask,
                    )
            rr.vmem_bar("vst_vld")

    # 消费已捕获的评分块并推进各 query 的独立候选池。
    @jit
    def process(self, offset, total, c0, c1, c2):
        if (offset + 256) % STREAM_PAGE_TOKENS == 0:
            c0, c1, c2 = self.consume(
                self.stage, offset - (STREAM_PAGE_TOKENS - 256), total, c0, c1, c2
            )
        return (c0, c1, c2)

    # 执行 LD 分片的种子选择、过滤、追加及必要收缩。
    @jit
    def consume(self, source, offset, total, c0, c1, c2):
        for row in dsl_range(3, unroll=1):
            count = c0
            if row == 1:
                count = c1
            elif row == 2:
                count = c2
            selected_count = dtypes.int64(STREAM_PAGE_TOKENS)
            if offset >= STREAM_POOL_TOKENS:
                with vf(mode="simd"):
                    mask = rr.update_mask(128, elem_bits=16)[0]
                    threshold = rr.vdup(rr.vload(self.thresholds, row * 16), mask=mask)
                    cursor = rr.vstore_unalign_begin(self.packed)
                    for c in dsl_range(32, unroll=1):
                        keep = rr.vgt(
                            rr.vload(source, row * STREAM_PAGE_TOKENS + c * 128),
                            threshold,
                            mask=mask,
                        )
                        packed = rr.vsqueeze_and_storeunalign_init(
                            rr.varange(c * 128, dtypes.uint16), mask=keep
                        )
                        rr.vsqueeze_and_storeunalign(self.packed, 0, packed, cursor)
                    rr.vstore_unalign_post(self.packed, 0, cursor)
                    rr.vmem_bar("vst_vld")
                vec_sync_notify(PIPE.V, PIPE.S, 6)
                vec_sync_wait(PIPE.V, PIPE.S, 6)
                selected_count = dtypes.int64(get_squeeze_status()) // 2
            else:
                with vf(mode="simd"):
                    mask = rr.update_mask(128, elem_bits=16)[0]
                    for c in dsl_range(32, unroll=1):
                        rr.vstore(
                            self.packed,
                            c * 128,
                            rr.varange(c * 128, dtypes.uint16),
                            mask,
                        )
                    rr.vmem_bar("vst_vld")
            with vf(mode="simd"):
                for c in dsl_range((selected_count + 127) // 128, unroll=1):
                    mask = rr.update_mask(selected_count - c * 128, elem_bits=16)[0]
                    pos = rr.vload(self.packed, c * 128)
                    value = rr.vgather(
                        source,
                        rr.vadds(pos, row * STREAM_PAGE_TOKENS, mask=mask),
                        mask=mask,
                    )
                    full = rr.update_mask(128, elem_bits=16)[0]
                    rr.vstore(
                        self.values,
                        row * STREAM_POOL_TOKENS + count + c * 128,
                        rr.vselect(
                            value, rr.vdups(0, dtypes.uint16, mask=full), cond_mask=mask
                        ),
                        full,
                    )
                for c in dsl_range((selected_count + 127) // 128, unroll=1):
                    mask = rr.update_mask(selected_count - c * 128, elem_bits=16)[0]
                    pos = rr.vload(self.packed, c * 128)
                    ids = rr.vadds(pos, offset, mask=mask)
                    full = rr.update_mask(128, elem_bits=16)[0]
                    rr.vstore(
                        self.indices,
                        row * STREAM_POOL_TOKENS + count + c * 128,
                        rr.vselect(
                            ids,
                            rr.vdups(UINT16_ALL_BITS, dtypes.uint16, mask=full),
                            cond_mask=mask,
                        ),
                        full,
                    )
                rr.vmem_bar("vst_vld")
            count += (
                (selected_count + POOL_ALIGNMENT_ELEMENTS - 1)
                // POOL_ALIGNMENT_ELEMENTS
                * POOL_ALIGNMENT_ELEMENTS
            )
            if (
                offset == STREAM_PAGE_TOKENS
                or (offset >= STREAM_POOL_TOKENS and count >= STREAM_PAGE_TOKENS)
                or offset + STREAM_PAGE_TOKENS >= total
            ):
                self.compact(row, count)
                count = dtypes.int64(512)
            if row == 0:
                c0 = count
            elif row == 1:
                c1 = count
            else:
                c2 = count
        return (c0, c1, c2)

    # 写出当前 query 分片的 TopK 索引和值，供 LD 归并。
    @jit
    def emit(self, row, gm_indices, gm_bits, base):
        oi = self.out_indices.next_write()
        ov = self.out_bits.next_write()
        with vf(mode="simd"):
            m32 = rr.update_mask(64, elem_bits=32)[0]
            for c in dsl_range(8, unroll=1):
                rr.vstore(
                    oi,
                    c * 64,
                    rr.vadds(
                        rr.vreinterpret(
                            rr.vunpack(
                                rr.vload(
                                    self.indices, row * STREAM_POOL_TOKENS + c * 64
                                ),
                                dtypes.uint32,
                                part="lower",
                            ),
                            dtypes.int32,
                        ),
                        base,
                        mask=m32,
                    ),
                    m32,
                )
            m16 = rr.update_mask(128, elem_bits=16)[0]
            for c in dsl_range(4, unroll=1):
                key = rr.vload(self.values, row * STREAM_POOL_TOKENS + c * 128)
                bit = rr.vselect(
                    rr.vbitwise_xor(
                        key, rr.vdups(BF16_SIGN_BIT, dtypes.uint16, mask=m16), mask=m16
                    ),
                    rr.vbitwise_xor(
                        key,
                        rr.vdups(UINT16_ALL_BITS, dtypes.uint16, mask=m16),
                        mask=m16,
                    ),
                    cond_mask=rr.vges(key, 32768, mask=m16),
                )
                rr.vstore(ov, c * 128, bit, m16)
            rr.vmem_bar("vst_vld")
        mem_copy(gm_indices, self.out_indices.next_read())
        mem_copy(gm_bits, self.out_bits.next_read())


class QliFusedKernel:
    def __init__(
        self,
        topk,
        candidate_enabled,
        candidate_topk,
        mask_mode,
        cmp_ratio,
        has_cu_seqlens_q,
        has_seqused_q,
        has_seqused_k,
        has_block_table,
        pa_block_size,
        n1,
        query_rows,
        metadata,
        output_offset_address,
        return_values,
        streaming_mode=False,
        tnd_layout=False,
    ):
        self.streaming_mode = bool(streaming_mode)
        self.tnd_layout = bool(tnd_layout)
        self.ub_pool = _UbPool()
        self.output_offset_address = output_offset_address
        self.metadata = metadata
        self.return_values = return_values
        self.need_topk_values = _need_topk_values(metadata, return_values)
        self.n1 = int(n1)
        self.query_rows = int(query_rows)
        self.vec_rows = int(query_rows) // 2
        self.tile_m = int(n1) * int(query_rows)
        self.topk_count = int(topk)
        self.cache_enabled = (
            not candidate_enabled
            and self.topk_count == CACHED_TOPK_COUNT
            and self.tile_m <= 192
        )
        self.candidate_enabled = bool(candidate_enabled)
        self.candidate_topk_count = int(candidate_topk)
        self.mask_mode = int(mask_mode)
        self.cmp_ratio = int(cmp_ratio)
        self.has_cu_seqlens_q = bool(has_cu_seqlens_q)
        self.has_seqused_q = bool(has_seqused_q)
        self.has_seqused_k = bool(has_seqused_k)
        self.has_cmp_residual_k = self.mask_mode == 3 and self.cmp_ratio != 1
        self.has_block_table = bool(has_block_table)
        self.qk_slot0 = _ub_buffer(
            self.ub_pool,
            MemLoc.UB,
            (self.tile_m, 128),
            dtypes.bfloat16,
            stride=(256, 1),
            addr=0,
        )
        self.qk_slot1 = _ub_buffer(
            self.ub_pool,
            MemLoc.UB,
            (self.tile_m, 128),
            dtypes.bfloat16,
            stride=(256, 1),
            addr=256,
        )
        self.cube = (QliTndCube if self.tnd_layout else QliCube)(
            pa_block_size, n1, query_rows
        )
        self.topk_workspace = QliRawTopKWorkspace(
            TOPK_TRUNK_LEN if self.cache_enabled else STREAM_TOPK_TRUNK_LEN,
            max(
                self.topk_count,
                self.candidate_topk_count if self.candidate_enabled else 1,
            ),
            pipelined=self.cache_enabled,
            live_histogram=(
                CACHED_HISTOGRAM_UB_OFFSET
                if self.cache_enabled
                else self.tile_m * QK_ROW_BYTES + HISTOGRAM_AFTER_QK_BYTES,
                (
                    CACHED_HISTOGRAM_UB_OFFSET
                    if self.cache_enabled
                    else self.tile_m * QK_ROW_BYTES + HISTOGRAM_AFTER_QK_BYTES
                )
                + self.query_rows * QK_ROW_BYTES,
            ),
        )
        self.topk = QliRawTopKSelector(self.topk_count, self.topk_workspace)
        if self.candidate_enabled:
            self.candidate_topk = QliRawTopKSelector(
                self.candidate_topk_count,
                self.topk_workspace,
            )

        self.vector = QliVector(
            get_subblock_id(),
            self.mask_mode,
            self.cmp_ratio,
            self.candidate_enabled,
            self.candidate_topk_count,
            n1,
            query_rows,
            CACHED_VECTOR_UB_OFFSET
            if self.cache_enabled
            else self.tile_m * QK_ROW_BYTES + VECTOR_AFTER_QK_BYTES,
            self.cache_enabled,
        )
        self.stream_supported = (
            not candidate_enabled
            and int(n1) == 32
            and int(query_rows) == 6
            and int(topk) == 512
        )
        self.streaming = StreamingLdTopK(self.topk)
        self.vector.streaming = self.streaming
        # Dedicated 512B-strided bank-separated histogram slots, main phase only.
        self.hist_slot0 = _ub_buffer(
            self.ub_pool,
            MemLoc.UB,
            (self.vec_rows * 2, 128),
            dtypes.uint16,
            stride=(256, 1),
            addr=CACHED_HISTOGRAM_UB_OFFSET
            if self.cache_enabled
            else self.tile_m * QK_ROW_BYTES + HISTOGRAM_AFTER_QK_BYTES,
        )
        self.hist_slot1 = _ub_buffer(
            self.ub_pool,
            MemLoc.UB,
            (self.vec_rows * 2, 128),
            dtypes.uint16,
            stride=(256, 1),
            addr=CACHED_HISTOGRAM_UB_OFFSET + HISTOGRAM_SLOT_OFFSET
            if self.cache_enabled
            else self.tile_m * QK_ROW_BYTES
            + HISTOGRAM_AFTER_QK_BYTES
            + HISTOGRAM_SLOT_OFFSET,
        )
        # Partition-local pruning adds threshold/packing work without reducing
        # the final LD merge. Keep exact local TopK; whole-row TopKSelect remains.
        self.topkselect_enabled = self.cache_enabled and self.n1 == 32
        if self.topkselect_enabled:
            self.topkselect = TopKSelectPreprocessor(self.topk, prefetch=True)
            self.dma_topkselect = DmaTopKSelectPreprocessor(self.topk, prefetch=True)
        self.topk.hist_slot0 = self.hist_slot0
        self.topk.hist_slot1 = self.hist_slot1
        # Main phase: QK [0,98304), cached keys [98304,251904),
        # histogram [251904,254976), weights and fallback key stages above255488.
        # Cached TopK phase: temporary/output buffers reuse the drained QK area.
        if self.cache_enabled:
            self.full_key_cache = _ub_buffer(
                self.ub_pool,
                MemLoc.UB,
                (self.vec_rows, CACHE_TOKENS),
                dtypes.uint16,
                addr=CACHED_QK_UB_BYTES,
            )
            self.vector.full_key_cache = self.full_key_cache
            self.topk.cache_tmp = _ub_buffer(
                self.ub_pool,
                MemLoc.UB,
                (CACHE_TOKENS,),
                dtypes.uint16,
                addr=0,
            )
            self.topk.cache_output_idx_stage = _ub_ring_buffer(
                self.ub_pool,
                MemLoc.UB,
                (1, self.topk_count),
                dtypes.int32,
                depth=1,
                addr=256000,
            )
            self.topk.cache_output_value_bits_stage = _ub_ring_buffer(
                self.ub_pool,
                MemLoc.UB,
                (1, self.topk_count),
                dtypes.uint16,
                depth=1,
                addr=258048,
            )
        self.owner_supported = (
            not self.tnd_layout
            and not candidate_enabled
            and int(n1) == 32
            and int(query_rows) == 6
        )
        if self.owner_supported:
            self.owner_cube = OwnerCube512(pa_block_size, n1, query_rows, self.cube)
            self.owner64_vector = Owner64Vector(n1)
            self.owner64_slot0 = _ub_buffer(
                self.ub_pool, MemLoc.UB, (64, 512), dtypes.bfloat16, addr=0
            )
            self.owner64_slot1 = _ub_buffer(
                self.ub_pool, MemLoc.UB, (64, 512), dtypes.bfloat16, addr=65536
            )
            self.owner64_slot2 = _ub_buffer(
                self.ub_pool, MemLoc.UB, (64, 512), dtypes.bfloat16, addr=131072
            )

        # Byte-cache and streaming pools alias UB and execute exclusively.
        # One arena keeps aliased byte buffers within the hardware buf_id limit.
        self.byte_arena = self.streaming.arena
        self.byte_cache0 = reinterpret(
            self.byte_arena,
            dtype=dtypes.uint8,
            offset=0,
            layout=make_layout((BYTE_CACHE_TOKENS,)),
        )
        self.byte_cache1 = reinterpret(
            self.byte_arena,
            dtype=dtypes.uint8,
            offset=BYTE_CACHE_TOKENS,
            layout=make_layout((BYTE_CACHE_TOKENS,)),
        )
        self.byte_cache2 = reinterpret(
            self.byte_arena,
            dtype=dtypes.uint8,
            offset=BYTE_ROW2_OFFSET,
            layout=make_layout((BYTE_CACHE_TOKENS,)),
        )
        self.byte_overflow = reinterpret(
            self.byte_arena,
            dtype=dtypes.uint16,
            offset=BYTE_OVERFLOW_OFFSET,
            layout=make_layout((STREAM_ROWS, CACHED_TOPK_COUNT)),
        )
        self.byte_meta = reinterpret(
            self.byte_arena,
            dtype=dtypes.uint16,
            offset=BYTE_META_OFFSET,
            layout=make_layout((STREAM_ROWS, 128)),
        )
        self.vector.byte_cache0 = self.byte_cache0
        self.vector.byte_cache1 = self.byte_cache1
        self.vector.byte_cache2 = self.byte_cache2
        self.vector.byte_overflow = self.byte_overflow
        self.vector.byte_meta = self.byte_meta
        self.topk.byte_fallback_keys = reinterpret(
            self.streaming.root_arena,
            dtype=dtypes.uint16,
            offset=0,
            layout=make_layout((1, BYTE_CACHE_TOKENS)),
        )
        # Fallback keys occupy [0, 99328); scratch must start after them.
        self.topk.byte_fallback_tmp = reinterpret(
            self.byte_arena,
            dtype=dtypes.uint16,
            offset=BYTE_FALLBACK_SCRATCH_OFFSET,
            layout=make_layout((BYTE_CACHE_TOKENS,)),
        )
        self.topk.byte_tmp = reinterpret(
            self.topk.byte_fallback_keys, layout=make_layout((49152,))
        )
        self.vector.byte_seed = reinterpret(
            self.byte_arena,
            dtype=dtypes.uint16,
            offset=BYTE_SEED_OFFSET,
            layout=make_layout((STREAM_ROWS, TILE_N)),
        )

        self.vector.byte_enabled = self.streaming_mode and self.stream_supported

    @jit
    def _combine_hist(self, row):
        with vf(mode="simd"):
            mask = rr.update_mask(128, elem_bits=16)[0]
            histogram_low_reg = rr.vadd(
                rr.vload(self.hist_slot0, row * 512),
                rr.vload(self.hist_slot1, row * 512),
                mask=mask,
            )
            histogram_high_reg = rr.vadd(
                rr.vload(self.hist_slot0, row * 512 + 256),
                rr.vload(self.hist_slot1, row * 512 + 256),
                mask=mask,
            )
            rr.vstore(self.topk_workspace.histogram, 0, histogram_low_reg, mask)
            rr.vstore(self.topk_workspace.histogram, 128, histogram_high_reg, mask)
            rr.vmem_bar("vst_vld")

    @jit
    def _compute_nonempty_task(
        self,
        query: Tensor,
        key: Tensor,
        weights: Tensor,
        query_scale: Tensor,
        key_scale: Tensor,
        key_stride0,
        key_dequant_scale_stride0,
        block_table: Tensor,
        key_out: Tensor,
        candidate_key_out: Tensor,
        sparse_indices: Tensor,
        sparse_value_bits: Tensor,
        candidate_indices: Tensor,
        candidate_value_bits: Tensor,
        candidate_length: Tensor,
        batch_idx,
        public_row_base,
        query_offset,
        task_span_count,
        task_active_count,
        used_q,
        actual_s2,
        cmp_residual,
        workspace_task_idx,
        tokens,
        tile_begin,
        tile_end,
        row_ld,
        previous_owner,
    ):
        # 完整任务只按公开返回值需求保留分数；LD 分片仍保留归并值。
        task_need_values = dtypes.int64(1 if row_ld != 0 else self.return_values)
        tokens = (tile_end - tile_begin) * TILE_N
        candidate_tokens = tokens // CANDIDATE_BLOCK_SIZE
        stream_hist = dtypes.int32(0)
        cache_task = dtypes.int32(0)
        c0 = dtypes.int64(0)
        c1 = dtypes.int64(0)
        c2 = dtypes.int64(0)
        stream_task = dtypes.int32(0)
        owner_cv = dtypes.int32(0)
        if self.owner_supported:
            owner_cv = dtypes.int32(
                row_ld == 0
                and task_span_count == 6
                and (actual_s2 >= OWNER_MIN_TOKENS)
                and (actual_s2 <= OWNER_MAX_TOKENS)
            )
        if previous_owner >= 0 and (previous_owner != 0 or owner_cv != 0):
            cube_sync_all()
            cube_sync_intra_arrive(PIPE.FIXPIPE, 14)
            cube_sync_intra_arrive(PIPE.FIXPIPE, 30)
            vec_sync_intra_wait(PIPE.MTE3, 14)
            vec_sync_all()
            vec_sync_intra_arrive(PIPE.MTE3, 14)
            cube_sync_intra_wait(PIPE.FIXPIPE, 14)
            cube_sync_intra_wait(PIPE.FIXPIPE, 30)
        if owner_cv:
            if self.owner_supported:
                self._compute_cv_owner64(
                    query,
                    key,
                    weights,
                    query_scale,
                    key_scale,
                    key_stride0,
                    key_dequant_scale_stride0,
                    block_table,
                    key_out,
                    candidate_key_out,
                    sparse_indices,
                    sparse_value_bits,
                    candidate_indices,
                    candidate_value_bits,
                    candidate_length,
                    batch_idx,
                    public_row_base,
                    query_offset,
                    task_span_count,
                    task_active_count,
                    used_q,
                    actual_s2,
                    cmp_residual,
                    workspace_task_idx,
                    tokens,
                    tile_begin,
                    tile_end,
                    row_ld,
                )
        else:
            vec_sync_all()
            self.cube.load_query(query, query_scale, public_row_base)
            self.vector.load_weights(weights, public_row_base, task_span_count)
            # PA TopK above one tile resets both histogram slots in its first two score tiles.
            if const_expr(
                self.candidate_enabled or self.tnd_layout or self.topk_count <= TILE_N
            ):
                with vf(mode="simd"):
                    zero_mask16 = rr.update_mask(128, elem_bits=16)[0]
                    zero_hist16 = rr.vdups(0, dtypes.uint16, mask=zero_mask16)
                    for hist_row in range_constexpr(self.vector.vec_rows * 2):
                        rr.vstore(
                            self.hist_slot0, hist_row * 256, zero_hist16, zero_mask16
                        )
                        rr.vstore(
                            self.hist_slot1, hist_row * 256, zero_hist16, zero_mask16
                        )
                    rr.vmem_bar("vst_vld")
            tokens = (tile_end - tile_begin) * TILE_N
            stream_hist = dtypes.int32(0)
            if not self.candidate_enabled:
                stream_hist = dtypes.int32(
                    tokens >= self.topk_count
                    and tokens <= self.topk_workspace.max_trunk_len
                )
            cache_min_valid = actual_s2
            if self.mask_mode == 3:
                cache_min_valid = max(
                    0,
                    min(
                        actual_s2,
                        (
                            actual_s2 * self.cmp_ratio
                            + cmp_residual
                            - used_q
                            + query_offset
                            + 1
                        )
                        // self.cmp_ratio,
                    ),
                )
            cache_task = dtypes.int32(0)
            if self.cache_enabled:
                cache_task = dtypes.int32(
                    self.topk_count == CACHED_TOPK_COUNT
                    and tokens >= self.topk_count
                    and (tokens <= CACHE_TOKENS)
                    and (task_active_count == self.query_rows)
                    and (task_span_count == self.query_rows)
                    and (task_need_values != 0)
                    and (
                        dtypes.int64(cache_min_valid) - tile_begin * TILE_N
                        >= self.topk_count
                    )
                )
            if self.streaming_mode:
                if self.stream_supported:
                    if (
                        row_ld != 0
                        and tokens > CACHE_TOKENS
                        and (tokens <= BYTE_CACHE_TOKENS)
                        and (task_active_count == 6)
                        and (task_span_count == 6)
                        and (task_need_values != 0)
                        and (dtypes.int64(cache_min_valid) - tile_begin * TILE_N >= 512)
                    ):
                        cache_task = dtypes.int32(2)
                        with vf(mode="simd"):
                            byte_mask = rr.update_mask(128, elem_bits=16)[0]
                            byte_zero = rr.vdups(0, dtypes.uint16, mask=byte_mask)
                            for row in range_constexpr(3):
                                rr.vstore(
                                    self.byte_meta, row * 128, byte_zero, byte_mask
                                )
                                rr.vstore(
                                    self.vector.byte_seed,
                                    row * 256,
                                    byte_zero,
                                    byte_mask,
                                )
                                rr.vstore(
                                    self.vector.byte_seed,
                                    row * 256 + 128,
                                    byte_zero,
                                    byte_mask,
                                )
                            rr.vmem_bar("vst_vld")
                if self.stream_supported:
                    stream_task = dtypes.int32(
                        cache_task != 2
                        and row_ld != 0
                        and (task_active_count == 6)
                        and (task_span_count == 6)
                        and (tokens >= 32768)
                        and (tokens <= 65536)
                        and (tokens % 4096 == 0)
                        and (dtypes.int64(cache_min_valid) >= (tile_end - 1) * TILE_N)
                    )
                    if stream_task:
                        stream_hist = dtypes.int32(0)
                        cache_task = dtypes.int32(0)
                        self.streaming.reset()
            tokens // TILE_N
            candidate_tokens = tokens // CANDIDATE_BLOCK_SIZE
            vec_sync_intra_arrive(PIPE.V, 8)
            vec_sync_intra_arrive(PIPE.V, 9)
            for preload in dsl_range(
                tile_begin, min(tile_end, tile_begin + 3), unroll=1
            ):
                self.cube.prefetch_key(
                    key,
                    key_scale,
                    block_table,
                    batch_idx,
                    actual_s2,
                    key_stride0,
                    key_dequant_scale_stride0,
                    self.has_block_table,
                    preload,
                    self.qk_slot0,
                )
            for token_tile in dsl_range(tile_begin, tile_end, unroll=1):
                if token_tile + 3 < tile_end:
                    self.cube.prefetch_key(
                        key,
                        key_scale,
                        block_table,
                        batch_idx,
                        actual_s2,
                        key_stride0,
                        key_dequant_scale_stride0,
                        self.has_block_table,
                        token_tile + 3,
                        self.qk_slot0,
                    )
                if (token_tile - tile_begin) % 2 == 0:
                    logical_start = token_tile * TILE_N
                    cube_sync_intra_wait(PIPE.FIXPIPE, 8)
                    cube_sync_intra_wait(PIPE.FIXPIPE, 24)
                    qk_write = self.qk_slot0
                    self.cube.compute_qk(
                        key,
                        key_scale,
                        block_table,
                        batch_idx,
                        actual_s2,
                        key_stride0,
                        key_dequant_scale_stride0,
                        self.has_block_table,
                        token_tile,
                        qk_write,
                    )
                    cube_sync_intra_arrive(PIPE.FIXPIPE, 8)
                    cube_sync_intra_arrive(PIPE.FIXPIPE, 24)
                    vec_sync_intra_wait(PIPE.V, 8)
                    qk_read = self.qk_slot0
                    if self.streaming_mode:
                        self.vector.compute_vector1(
                            qk_read,
                            weights,
                            key_out,
                            candidate_key_out,
                            public_row_base,
                            task_span_count,
                            task_active_count,
                            workspace_task_idx,
                            logical_start,
                            actual_s2,
                            query_offset,
                            used_q,
                            cmp_residual,
                            candidate_tokens,
                            candidate_length,
                            logical_start - tile_begin * TILE_N,
                            min(QK_COLUMNS, (tile_end - token_tile) * TILE_N),
                            stream_hist,
                            self.hist_slot0,
                            cache_task,
                            stream_task,
                            tokens,
                        )
                    else:
                        self.vector._compute_vector1_fallback(
                            qk_read,
                            weights,
                            key_out,
                            candidate_key_out,
                            public_row_base,
                            task_span_count,
                            task_active_count,
                            workspace_task_idx,
                            logical_start,
                            actual_s2,
                            query_offset,
                            used_q,
                            cmp_residual,
                            candidate_tokens,
                            candidate_length,
                            logical_start - tile_begin * TILE_N,
                            min(QK_COLUMNS, (tile_end - token_tile) * TILE_N),
                            stream_hist,
                            self.hist_slot0,
                            cache_task,
                        )
                    vec_sync_intra_arrive(PIPE.V, 8)
                    if self.streaming_mode:
                        if stream_task:
                            c0, c1, c2 = self.streaming.process(
                                logical_start - tile_begin * TILE_N, tokens, c0, c1, c2
                            )
                else:
                    logical_start = token_tile * TILE_N
                    cube_sync_intra_wait(PIPE.FIXPIPE, 9)
                    cube_sync_intra_wait(PIPE.FIXPIPE, 25)
                    qk_write = self.qk_slot1
                    self.cube.compute_qk(
                        key,
                        key_scale,
                        block_table,
                        batch_idx,
                        actual_s2,
                        key_stride0,
                        key_dequant_scale_stride0,
                        self.has_block_table,
                        token_tile,
                        qk_write,
                    )
                    cube_sync_intra_arrive(PIPE.FIXPIPE, 9)
                    cube_sync_intra_arrive(PIPE.FIXPIPE, 25)
                    vec_sync_intra_wait(PIPE.V, 9)
                    qk_read = self.qk_slot1
                    if self.streaming_mode:
                        self.vector.compute_vector1(
                            qk_read,
                            weights,
                            key_out,
                            candidate_key_out,
                            public_row_base,
                            task_span_count,
                            task_active_count,
                            workspace_task_idx,
                            logical_start,
                            actual_s2,
                            query_offset,
                            used_q,
                            cmp_residual,
                            candidate_tokens,
                            candidate_length,
                            logical_start - tile_begin * TILE_N,
                            min(QK_COLUMNS, (tile_end - token_tile) * TILE_N),
                            stream_hist,
                            self.hist_slot1,
                            cache_task,
                            stream_task,
                            tokens,
                        )
                    else:
                        self.vector._compute_vector1_fallback(
                            qk_read,
                            weights,
                            key_out,
                            candidate_key_out,
                            public_row_base,
                            task_span_count,
                            task_active_count,
                            workspace_task_idx,
                            logical_start,
                            actual_s2,
                            query_offset,
                            used_q,
                            cmp_residual,
                            candidate_tokens,
                            candidate_length,
                            logical_start - tile_begin * TILE_N,
                            min(QK_COLUMNS, (tile_end - token_tile) * TILE_N),
                            stream_hist,
                            self.hist_slot1,
                            cache_task,
                        )
                    vec_sync_intra_arrive(PIPE.V, 9)
                    if self.streaming_mode:
                        if stream_task:
                            c0, c1, c2 = self.streaming.process(
                                logical_start - tile_begin * TILE_N, tokens, c0, c1, c2
                            )
            cube_sync_intra_wait(PIPE.FIXPIPE, 8)
            cube_sync_intra_wait(PIPE.FIXPIPE, 24)
            cube_sync_intra_wait(PIPE.FIXPIPE, 9)
            cube_sync_intra_wait(PIPE.FIXPIPE, 25)
        if owner_cv:
            vec_sync_block_arrive(PIPE.MTE3, 10, mode=1)
            vec_sync_block_wait(PIPE.MTE2, 10, mode=1)
        vec_sync_notify(PIPE.MTE3, PIPE.MTE2, 0)
        vec_sync_wait(PIPE.MTE3, PIPE.MTE2, 0)
        if self.streaming_mode:
            if cache_task == 2:
                vec_sync_all()
                with vf(mode="simd"):
                    mask = rr.update_mask(1, elem_bits=32)[0]
                    rr.vstore_first(
                        self.topk_workspace.high_target_carrier,
                        30,
                        rr.vdups(0, dtypes.int32, mask=mask),
                    )
                    for row in range_constexpr(3):
                        base = rr.vload(self.byte_meta, row * 128)
                        rr.vstore_first(
                            self.topk_workspace.high_target_carrier,
                            32 + row * 8,
                            rr.vreinterpret(
                                rr.vunpack(base, dtypes.uint32, part="lower"),
                                dtypes.int32,
                            ),
                        )
                    rr.vmem_bar("vst_vld")
        topk_rows = dtypes.int32(self.query_rows // 2)
        for row_in_subblock in dsl_range(topk_rows, unroll=1):
            if owner_cv and self.metadata[LD_RECORD_START, 0] != 0:
                vec_sync_all()
            local_row = dtypes.int32(
                get_subblock_id() * (self.query_rows // 2) + row_in_subblock
            )
            public_row = dtypes.int64(public_row_base) + dtypes.int64(local_row)
            workspace_row = workspace_task_idx * self.query_rows + local_row
            valid_s2 = actual_s2
            if self.mask_mode == 3:
                query_in_batch = query_offset + local_row
                valid_s2 = (
                    actual_s2 * self.cmp_ratio
                    + cmp_residual
                    - used_q
                    + query_in_batch
                    + 1
                ) // self.cmp_ratio
                if valid_s2 < 0:
                    valid_s2 = 0
                if valid_s2 > actual_s2:
                    valid_s2 = actual_s2
            if local_row >= task_active_count:
                valid_s2 = 0
            valid_s2 = valid_s2 - dtypes.int32(tile_begin * TILE_N)
            if valid_s2 < 0:
                valid_s2 = dtypes.int32(0)
            if valid_s2 > dtypes.int32(tokens):
                valid_s2 = dtypes.int32(tokens)
            if local_row < task_span_count:
                output_offset = dtypes.int32(0)
                if self.output_offset_address != 0 and row_ld == 0:
                    offsets = make_tensor(
                        make_pointer(
                            dtypes.int32,
                            dtypes.int64(self.output_offset_address),
                            MemLoc.GM,
                        ),
                        make_layout((weights.shape[0], 1), stride=(1, 1)),
                    )
                    output_offset = offsets[public_row, 0]
                if self.streaming_mode:
                    if cache_task == 2:
                        self.topk._select_row_byte(
                            _offset_view(key_out, (workspace_row, 0)),
                            reinterpret(
                                self.byte_cache0,
                                offset=row_in_subblock * BYTE_CACHE_TOKENS,
                                layout=make_layout((BYTE_CACHE_TOKENS,)),
                            ),
                            tile_slice(
                                self.byte_overflow, (1, 512), (row_in_subblock, 0)
                            ).view(512),
                            dtypes.int64(valid_s2),
                            tile_slice(
                                sparse_indices, (1, self.topk_count), (public_row, 0)
                            ),
                            tile_slice(
                                sparse_value_bits, (1, self.topk_count), (public_row, 0)
                            ),
                            dtypes.int32(tile_begin * TILE_N) + output_offset,
                            row_in_subblock,
                        )
                    elif stream_task:
                        self.streaming.emit(
                            row_in_subblock,
                            tile_slice(
                                sparse_indices, (1, self.topk_count), (public_row, 0)
                            ),
                            tile_slice(
                                sparse_value_bits, (1, self.topk_count), (public_row, 0)
                            ),
                            dtypes.int32(tile_begin * TILE_N) + output_offset,
                        )
                    elif (
                        self.topkselect_enabled
                        and row_ld == 0
                        and (cache_task == 0)
                        and (task_need_values == 0)
                        and (valid_s2 >= 32768)
                        and (valid_s2 <= 131072)
                    ):
                        if not owner_cv or row_in_subblock == 0:
                            vec_sync_all()
                        if is_aiv():
                            self.dma_topkselect.select_row(
                                key_out,
                                workspace_row,
                                dtypes.int64(valid_s2),
                                tile_slice(
                                    sparse_indices,
                                    (1, self.topk_count),
                                    (public_row, 0),
                                ),
                                dtypes.int32(tile_begin * TILE_N) + output_offset,
                            )
                    elif (
                        self.topkselect_enabled
                        and row_ld == 0
                        and (cache_task == 0)
                        and (valid_s2 >= 32768)
                        and (valid_s2 <= 131072)
                    ):
                        if not owner_cv or row_in_subblock == 0:
                            vec_sync_all()
                        self.topkselect.select_row(
                            key_out,
                            workspace_row,
                            dtypes.int64(valid_s2),
                            tile_slice(
                                sparse_indices, (1, self.topk_count), (public_row, 0)
                            ),
                            tile_slice(
                                sparse_value_bits, (1, self.topk_count), (public_row, 0)
                            ),
                            dtypes.int32(tile_begin * TILE_N) + output_offset,
                            task_need_values,
                        )
                    elif self.cache_enabled and cache_task:
                        self.topk._select_row_cached_ub(
                            tile_slice(
                                self.full_key_cache,
                                (1, CACHE_TOKENS),
                                (row_in_subblock, 0),
                            ),
                            dtypes.int64(valid_s2),
                            valid_s2,
                            tile_slice(
                                sparse_indices, (1, self.topk_count), (public_row, 0)
                            ),
                            tile_slice(
                                sparse_value_bits, (1, self.topk_count), (public_row, 0)
                            ),
                            dtypes.int32(tile_begin * TILE_N) + output_offset,
                            1,
                            row_in_subblock,
                        )
                    else:
                        self.topk.select_row(
                            _offset_view(key_out, (workspace_row, 0)),
                            dtypes.int64(valid_s2),
                            valid_s2,
                            tile_slice(
                                sparse_indices, (1, self.topk_count), (public_row, 0)
                            ),
                            tile_slice(
                                sparse_value_bits, (1, self.topk_count), (public_row, 0)
                            ),
                            dtypes.int32(tile_begin * TILE_N) + output_offset,
                            task_need_values,
                            stream_hist,
                            row_in_subblock,
                        )
                elif (
                    self.topkselect_enabled
                    and row_ld == 0
                    and (cache_task == 0)
                    and (task_need_values == 0)
                    and (valid_s2 >= 32768)
                    and (valid_s2 <= 131072)
                ):
                    if not owner_cv or row_in_subblock == 0:
                        vec_sync_all()
                    if is_aiv():
                        self.dma_topkselect.select_row(
                            key_out,
                            workspace_row,
                            dtypes.int64(valid_s2),
                            tile_slice(
                                sparse_indices, (1, self.topk_count), (public_row, 0)
                            ),
                            dtypes.int32(tile_begin * TILE_N) + output_offset,
                        )
                elif (
                    self.topkselect_enabled
                    and row_ld == 0
                    and (cache_task == 0)
                    and (valid_s2 >= 32768)
                    and (valid_s2 <= 131072)
                ):
                    if not owner_cv or row_in_subblock == 0:
                        vec_sync_all()
                    self.topkselect.select_row(
                        key_out,
                        workspace_row,
                        dtypes.int64(valid_s2),
                        tile_slice(
                            sparse_indices, (1, self.topk_count), (public_row, 0)
                        ),
                        tile_slice(
                            sparse_value_bits, (1, self.topk_count), (public_row, 0)
                        ),
                        dtypes.int32(tile_begin * TILE_N) + output_offset,
                        task_need_values,
                    )
                elif self.cache_enabled and cache_task:
                    self.topk._select_row_cached_ub(
                        tile_slice(
                            self.full_key_cache, (1, CACHE_TOKENS), (row_in_subblock, 0)
                        ),
                        dtypes.int64(valid_s2),
                        valid_s2,
                        tile_slice(
                            sparse_indices, (1, self.topk_count), (public_row, 0)
                        ),
                        tile_slice(
                            sparse_value_bits, (1, self.topk_count), (public_row, 0)
                        ),
                        dtypes.int32(tile_begin * TILE_N) + output_offset,
                        1,
                        row_in_subblock,
                    )
                else:
                    self.topk.select_row(
                        _offset_view(key_out, (workspace_row, 0)),
                        dtypes.int64(valid_s2),
                        valid_s2,
                        tile_slice(
                            sparse_indices, (1, self.topk_count), (public_row, 0)
                        ),
                        tile_slice(
                            sparse_value_bits, (1, self.topk_count), (public_row, 0)
                        ),
                        dtypes.int32(tile_begin * TILE_N) + output_offset,
                        task_need_values,
                        stream_hist,
                        row_in_subblock,
                    )
                if self.candidate_enabled:
                    valid_candidate_count = (
                        valid_s2 + CANDIDATE_BLOCK_SIZE - 1
                    ) // CANDIDATE_BLOCK_SIZE
                    self.candidate_topk.select_candidate_row(
                        _offset_view(candidate_key_out, (workspace_row, 0)),
                        candidate_tokens,
                        valid_candidate_count,
                        tile_slice(
                            candidate_indices,
                            (1, self.candidate_topk_count),
                            (public_row, 0),
                        ),
                        tile_slice(
                            candidate_value_bits,
                            (1, self.candidate_topk_count),
                            (public_row, 0),
                        ),
                        dtypes.int32(tile_begin * TILE_N // CANDIDATE_BLOCK_SIZE),
                        dtypes.int64(row_ld != 0),
                    )
        if owner_cv:
            vec_sync_block_arrive(PIPE.MTE2, 11, mode=1)
            vec_sync_block_wait(PIPE.MTE3, 11, mode=1)

    @jit
    def _compute_cv_owner_group(
        self,
        query_ring,
        qk_slot,
        key_read,
        query_start,
        owner,
        key_out,
        task_span_count,
        workspace_task_idx,
        token_tile,
        tile_begin,
        tokens,
    ):
        dsl_matmul(
            reinterpret(
                self.owner_cube.micro_l0c.next_write(), layout=make_layout((64, 512))
            ),
            query_ring.next_read(),
            key_read,
            init=True,
            mx_mode=True,
            unit_flag=3,
        )
        mem_copy(
            qk_slot,
            relu(
                reinterpret(
                    self.owner_cube.micro_l0c.next_read(), layout=make_layout((64, 512))
                )
            ),
            options=copy_options(dst_subblock=owner, unit_flag=3),
        )
        if dtypes.int32(get_subblock_id()) == owner:
            self.owner64_vector.score(qk_slot, dtypes.int32(self.n1), query_start)
            rows_to_store = min(2, task_span_count - query_start)
            row_base = workspace_task_idx * self.query_rows + query_start
            col = (token_tile - tile_begin) * TILE_N
            mem_copy(
                key_out[
                    row_base : row_base + rows_to_store, col : min(col + 512, tokens)
                ],
                self.owner64_vector.key_ub.next_read()[None, : min(512, tokens - col)],
            )

    @jit
    def _compute_cv_owner64(
        self,
        query: Tensor,
        key: Tensor,
        weights: Tensor,
        query_scale: Tensor,
        key_scale: Tensor,
        key_stride0,
        key_dequant_scale_stride0,
        block_table: Tensor,
        key_out: Tensor,
        candidate_key_out: Tensor,
        sparse_indices: Tensor,
        sparse_value_bits: Tensor,
        candidate_indices: Tensor,
        candidate_value_bits: Tensor,
        candidate_length: Tensor,
        batch_idx,
        public_row_base,
        query_offset,
        task_span_count,
        task_active_count,
        used_q,
        actual_s2,
        cmp_residual,
        workspace_task_idx,
        tokens,
        tile_begin,
        tile_end,
        row_ld,
    ):
        vec_sync_all()
        self.owner_cube.load_micro_query(query, query_scale, public_row_base)
        self.owner64_vector.load_weights(weights, public_row_base, task_span_count)
        for preload in dsl_range(
            tile_begin, min(tile_end, tile_begin + 6), 2, unroll=1
        ):
            self.owner_cube.prefetch_key(
                key,
                key_scale,
                block_table,
                batch_idx,
                actual_s2,
                key_stride0,
                key_dequant_scale_stride0,
                self.has_block_table,
                preload,
                self.owner64_slot0,
            )
        for token_tile in dsl_range(tile_begin, tile_end, 2, unroll=1):
            if token_tile + 6 < tile_end:
                self.owner_cube.prefetch_key(
                    key,
                    key_scale,
                    block_table,
                    batch_idx,
                    actual_s2,
                    key_stride0,
                    key_dequant_scale_stride0,
                    self.has_block_table,
                    token_tile + 6,
                    self.owner64_slot0,
                )
            mem_copy(
                self.owner_cube.l0b.next_write(),
                self.owner_cube.k_l1.next_read(),
                options=copy_options(mx_scale=self.owner_cube.k_scale_l1.next_read()),
            )
            key_read = self.owner_cube.l0b.next_read()
            if task_span_count > 0:
                owner = dtypes.int32((0 + (token_tile - tile_begin) // 2) % 2)
                self._compute_cv_owner_group(
                    self.owner_cube.micro_a0,
                    self.owner64_slot0,
                    key_read,
                    0,
                    owner,
                    key_out,
                    task_span_count,
                    workspace_task_idx,
                    token_tile,
                    tile_begin,
                    tokens,
                )
            if task_span_count > 2:
                owner = dtypes.int32((1 + (token_tile - tile_begin) // 2) % 2)
                self._compute_cv_owner_group(
                    self.owner_cube.micro_a1,
                    self.owner64_slot1,
                    key_read,
                    2,
                    owner,
                    key_out,
                    task_span_count,
                    workspace_task_idx,
                    token_tile,
                    tile_begin,
                    tokens,
                )
            if task_span_count > 4:
                owner = dtypes.int32((0 + (token_tile - tile_begin) // 2) % 2)
                self._compute_cv_owner_group(
                    self.owner_cube.micro_a2,
                    self.owner64_slot2,
                    key_read,
                    4,
                    owner,
                    key_out,
                    task_span_count,
                    workspace_task_idx,
                    token_tile,
                    tile_begin,
                    tokens,
                )

    @jit
    def _compute_task(
        self,
        query: Tensor,
        key: Tensor,
        weights: Tensor,
        query_scale: Tensor,
        key_scale: Tensor,
        key_stride0,
        key_dequant_scale_stride0,
        block_table: Tensor,
        key_out: Tensor,
        candidate_key_out: Tensor,
        sparse_indices: Tensor,
        sparse_value_bits: Tensor,
        candidate_indices: Tensor,
        candidate_value_bits: Tensor,
        candidate_length: Tensor,
        batch_idx,
        public_row_base,
        query_offset,
        task_span_count,
        task_active_count,
        used_q,
        actual_s2,
        cmp_residual,
        workspace_task_idx,
        tokens,
        tile_begin,
        tile_end,
        task_valid_s2,
        row_ld,
        previous_owner,
    ):
        if task_valid_s2 > dtypes.int32(tile_begin * TILE_N) and tile_end > tile_begin:
            self._compute_nonempty_task(
                query,
                key,
                weights,
                query_scale,
                key_scale,
                key_stride0,
                key_dequant_scale_stride0,
                block_table,
                key_out,
                candidate_key_out,
                sparse_indices,
                sparse_value_bits,
                candidate_indices,
                candidate_value_bits,
                candidate_length,
                batch_idx,
                public_row_base,
                query_offset,
                task_span_count,
                task_active_count,
                used_q,
                actual_s2,
                cmp_residual,
                workspace_task_idx,
                tokens,
                tile_begin,
                tile_end,
                row_ld,
                previous_owner,
            )
        else:
            self._store_empty_rows(
                sparse_indices,
                sparse_value_bits,
                candidate_indices,
                candidate_value_bits,
                candidate_length,
                public_row_base,
                task_span_count,
            )

    @jit
    def _store_empty_rows(
        self,
        sparse_indices,
        sparse_value_bits,
        candidate_indices,
        candidate_value_bits,
        candidate_length,
        public_row_base,
        task_span_count,
    ):
        for row_in_subblock in range_constexpr(self.query_rows // 2):
            local_row = dtypes.int32(
                get_subblock_id() * (self.query_rows // 2) + row_in_subblock
            )
            public_row = dtypes.int64(public_row_base) + dtypes.int64(local_row)
            if local_row < task_span_count:
                self.topk.store_empty_row(
                    tile_slice(
                        sparse_indices,
                        (1, self.topk_count),
                        (public_row, 0),
                    ),
                    tile_slice(
                        sparse_value_bits,
                        (1, self.topk_count),
                        (public_row, 0),
                    ),
                )
                if self.candidate_enabled:
                    self.candidate_topk.store_empty_row(
                        tile_slice(
                            candidate_indices,
                            (1, self.candidate_topk_count),
                            (public_row, 0),
                        ),
                        tile_slice(
                            candidate_value_bits,
                            (1, self.candidate_topk_count),
                            (public_row, 0),
                        ),
                    )
        if self.candidate_enabled:
            self.vector.store_empty_candidate_lengths(
                candidate_length, public_row_base, task_span_count
            )

    @jit
    def _initialize_unscheduled_padding(
        self,
        cu_seqlens_q,
        seqused_q,
        candidate_length,
        worker,
        worker_count,
        total_q,
        batch_count,
        final_idx_address,
        final_bits_address,
        final_candidate_address,
        final_candidate_bits_address,
    ):
        # Only whole query tiles beyond effective Q are omitted from ASC-style
        # compute groups. The last active tile already initializes its own tail.
        indices = make_tensor(
            make_pointer(dtypes.int32, dtypes.int64(final_idx_address), MemLoc.GM),
            make_layout((total_q, self.topk_count), stride=(self.topk_count, 1)),
        )
        bits = make_tensor(
            make_pointer(dtypes.uint16, dtypes.int64(final_bits_address), MemLoc.GM),
            make_layout((total_q, self.topk_count), stride=(self.topk_count, 1)),
        )
        candidates = make_tensor(
            make_pointer(
                dtypes.int32, dtypes.int64(final_candidate_address), MemLoc.GM
            ),
            make_layout(
                (total_q, self.candidate_topk_count),
                stride=(self.candidate_topk_count, 1),
            ),
        )
        candidate_bits = make_tensor(
            make_pointer(
                dtypes.uint16, dtypes.int64(final_candidate_bits_address), MemLoc.GM
            ),
            make_layout(
                (total_q, self.candidate_topk_count),
                stride=(self.candidate_topk_count, 1),
            ),
        )
        for batch_idx in dsl_range(batch_count, unroll=1):
            begin = dtypes.int64(0)
            end = dtypes.int64(total_q)
            if self.has_cu_seqlens_q:
                begin = dtypes.int64(cu_seqlens_q[batch_idx])
                end = dtypes.int64(cu_seqlens_q[batch_idx + 1])
            used = dtypes.int64(seqused_q[batch_idx])
            first_padding = (
                (used + self.query_rows - 1) // self.query_rows
            ) * self.query_rows
            offset = first_padding + worker * self.query_rows
            while offset < end - begin:
                self._store_empty_rows(
                    indices,
                    bits,
                    candidates,
                    candidate_bits,
                    candidate_length,
                    begin + offset,
                    dtypes.int32(min(self.query_rows, end - begin - offset)),
                )
                offset += dtypes.int64(worker_count) * self.query_rows

    @jit
    def __call__(
        self,
        query: Tensor,
        key: Tensor,
        weights: Tensor,
        query_scale: Tensor,
        key_scale: Tensor,
        key_stride0,
        key_dequant_scale_stride0,
        cu_seqlens_q: Tensor,
        seqused_q: Tensor,
        seqused_k: Tensor,
        cmp_residual_k: Tensor,
        block_table: Tensor,
        key_out: Tensor,
        candidate_key_out: Tensor,
        sparse_indices: Tensor,
        sparse_value_bits: Tensor,
        candidate_indices: Tensor,
        candidate_value_bits: Tensor,
        candidate_length: Tensor,
        worker_count,
        total_q,
        batch_count,
        tokens,
        max_tasks_per_batch,
        metadata: Tensor,
        splits,
        task_count,
        partial_idx_address,
        partial_bits_address,
        partial_candidate_address,
        partial_candidate_bits_address,
        final_idx_address,
        final_bits_address,
        final_candidate_address,
        final_candidate_bits_address,
    ):
        previous_owner = dtypes.int32(-1)
        worker = dtypes.int64(get_block_idx())
        workspace_cursor = dtypes.int64(metadata[worker, 7])
        if metadata[worker, 0] != 0:
            first_b = dtypes.int64(metadata[worker, 1])
            last_b = dtypes.int64(metadata[worker, 4])
            for batch_idx in dsl_range(first_b, min(last_b + 1, batch_count), unroll=1):
                query_begin = dtypes.int64(0)
                query_end = dtypes.int64(total_q)
                if self.has_cu_seqlens_q:
                    query_begin = dtypes.int64(cu_seqlens_q[batch_idx])
                    query_end = dtypes.int64(cu_seqlens_q[batch_idx + 1])
                used_q = dtypes.int32(query_end - query_begin)
                if self.has_seqused_q:
                    used_q = seqused_q[batch_idx]
                actual_s2 = dtypes.int32(tokens)
                if self.has_seqused_k:
                    actual_s2 = seqused_k[batch_idx]
                elif self.tnd_layout and self.has_block_table:
                    actual_s2 = block_table[batch_idx + 1] - block_table[batch_idx]
                remainder = dtypes.int32(0)
                if self.has_cmp_residual_k:
                    remainder = cmp_residual_k[batch_idx]
                first_m = dtypes.int64(0)
                last_m = (
                    query_end - query_begin + self.query_rows - 1
                ) // self.query_rows
                if batch_idx == first_b:
                    first_m = dtypes.int64(metadata[worker, 2])
                if batch_idx == last_b:
                    last_m = dtypes.int64(metadata[worker, 5]) + dtypes.int64(
                        (1 if metadata[worker, 6] > 0 else 0)
                    )
                for m_idx in dsl_range(first_m, last_m, unroll=1):
                    query_offset = dtypes.int32(m_idx * self.query_rows)
                    public_row = dtypes.int32(query_begin + dtypes.int64(query_offset))
                    span = dtypes.int32(
                        min(self.query_rows, query_end - dtypes.int64(public_row))
                    )
                    active = min(span, max(dtypes.int32(0), used_q - query_offset))
                    visible = actual_s2
                    if self.mask_mode == 3:
                        visible = min(
                            actual_s2,
                            max(
                                dtypes.int32(0),
                                (
                                    actual_s2 * self.cmp_ratio
                                    + remainder
                                    - used_q
                                    + query_offset
                                    + active
                                )
                                // self.cmp_ratio,
                            ),
                        )
                    if active == 0:
                        visible = dtypes.int32(0)
                    tile_begin = dtypes.int64(0)
                    tile_end = (dtypes.int64(visible) + TILE_N - 1) // TILE_N
                    if batch_idx == first_b and m_idx == first_m:
                        tile_begin = dtypes.int64(metadata[worker, 3])
                    if batch_idx == last_b and m_idx == dtypes.int64(
                        metadata[worker, 5]
                    ):
                        tile_end = min(tile_end, dtypes.int64(metadata[worker, 6]))
                    row_ld = (
                        tile_begin > 0
                        or tile_end < (dtypes.int64(visible) + TILE_N - 1) // TILE_N
                    )
                    delta = workspace_cursor * self.query_rows - dtypes.int64(
                        public_row
                    )
                    idx_address = dtypes.int64(
                        (
                            dtypes.int64(partial_idx_address)
                            + delta * self.topk_count * 4
                            if row_ld
                            else final_idx_address
                        )
                    )
                    bits_address = dtypes.int64(
                        (
                            dtypes.int64(partial_bits_address)
                            + delta * self.topk_count * 2
                            if row_ld
                            else final_bits_address
                        )
                    )
                    candidate_address = dtypes.int64(
                        (
                            dtypes.int64(partial_candidate_address)
                            + delta * self.candidate_topk_count * 4
                            if row_ld
                            else final_candidate_address
                        )
                    )
                    candidate_bits_address = dtypes.int64(
                        (
                            dtypes.int64(partial_candidate_bits_address)
                            + delta * self.candidate_topk_count * 2
                            if row_ld
                            else final_candidate_bits_address
                        )
                    )
                    row_indices = make_tensor(
                        make_pointer(dtypes.int32, idx_address, MemLoc.GM),
                        make_layout(
                            (total_q, self.topk_count), stride=(self.topk_count, 1)
                        ),
                    )
                    row_bits = make_tensor(
                        make_pointer(dtypes.uint16, bits_address, MemLoc.GM),
                        make_layout(
                            (total_q, self.topk_count), stride=(self.topk_count, 1)
                        ),
                    )
                    row_candidates = make_tensor(
                        make_pointer(dtypes.int32, candidate_address, MemLoc.GM),
                        make_layout(
                            (total_q, self.candidate_topk_count),
                            stride=(self.candidate_topk_count, 1),
                        ),
                    )
                    row_candidate_bits = make_tensor(
                        make_pointer(dtypes.uint16, candidate_bits_address, MemLoc.GM),
                        make_layout(
                            (total_q, self.candidate_topk_count),
                            stride=(self.candidate_topk_count, 1),
                        ),
                    )
                    self._compute_task(
                        query,
                        key,
                        weights,
                        query_scale,
                        key_scale,
                        key_stride0,
                        key_dequant_scale_stride0,
                        block_table,
                        key_out,
                        candidate_key_out,
                        row_indices,
                        row_bits,
                        row_candidates,
                        row_candidate_bits,
                        candidate_length,
                        batch_idx,
                        public_row,
                        query_offset,
                        span,
                        active,
                        used_q,
                        actual_s2,
                        remainder,
                        worker,
                        tokens,
                        tile_begin,
                        tile_end,
                        visible,
                        row_ld,
                        previous_owner,
                    )
                    if visible > tile_begin * TILE_N and tile_end > tile_begin:
                        previous_owner = dtypes.int32(0)
                        if self.owner_supported:
                            previous_owner = dtypes.int32(
                                row_ld == 0
                                and span == 6
                                and actual_s2 >= 32768
                                and actual_s2 <= 131072
                            )
                    # FD tasks cover active rows only. The first partial owner
                    # initializes inactive public rows directly in the final output.
                    if row_ld and tile_begin == 0:
                        final_indices = make_tensor(
                            make_pointer(
                                dtypes.int32, dtypes.int64(final_idx_address), MemLoc.GM
                            ),
                            make_layout(
                                (total_q, self.topk_count), stride=(self.topk_count, 1)
                            ),
                        )
                        final_bits = make_tensor(
                            make_pointer(
                                dtypes.uint16,
                                dtypes.int64(final_bits_address),
                                MemLoc.GM,
                            ),
                            make_layout(
                                (total_q, self.topk_count), stride=(self.topk_count, 1)
                            ),
                        )
                        final_candidates = make_tensor(
                            make_pointer(
                                dtypes.int32,
                                dtypes.int64(final_candidate_address),
                                MemLoc.GM,
                            ),
                            make_layout(
                                (total_q, self.candidate_topk_count),
                                stride=(self.candidate_topk_count, 1),
                            ),
                        )
                        final_candidate_bits = make_tensor(
                            make_pointer(
                                dtypes.uint16,
                                dtypes.int64(final_candidate_bits_address),
                                MemLoc.GM,
                            ),
                            make_layout(
                                (total_q, self.candidate_topk_count),
                                stride=(self.candidate_topk_count, 1),
                            ),
                        )
                        for local in range_constexpr(self.query_rows // 2):
                            r = dtypes.int32(
                                get_subblock_id() * (self.query_rows // 2) + local
                            )
                            if r >= active and r < span:
                                self.topk.store_empty_row(
                                    tile_slice(
                                        final_indices,
                                        (1, self.topk_count),
                                        (dtypes.int64(public_row) + dtypes.int64(r), 0),
                                    ),
                                    tile_slice(
                                        final_bits,
                                        (1, self.topk_count),
                                        (dtypes.int64(public_row) + dtypes.int64(r), 0),
                                    ),
                                )
                                if self.candidate_enabled:
                                    self.candidate_topk.store_empty_row(
                                        tile_slice(
                                            final_candidates,
                                            (1, self.candidate_topk_count),
                                            (
                                                dtypes.int64(public_row)
                                                + dtypes.int64(r),
                                                0,
                                            ),
                                        ),
                                        tile_slice(
                                            final_candidate_bits,
                                            (1, self.candidate_topk_count),
                                            (
                                                dtypes.int64(public_row)
                                                + dtypes.int64(r),
                                                0,
                                            ),
                                        ),
                                    )
                    if row_ld:
                        workspace_cursor += 1

        if self.has_seqused_q:
            self._initialize_unscheduled_padding(
                cu_seqlens_q,
                seqused_q,
                candidate_length,
                worker,
                worker_count,
                total_q,
                batch_count,
                final_idx_address,
                final_bits_address,
                final_candidate_address,
                final_candidate_bits_address,
            )


_COMPILED_KERNEL: dict[tuple[object, ...], object] = {}
_COMPILED_KERNEL_LOCK = threading.Lock()


def _build_compiled_fused_runner(
    topk,
    candidate_enabled,
    candidate_topk,
    mask_mode,
    cmp_ratio,
    has_cu_seqlens_q,
    has_seqused_q,
    has_seqused_k,
    has_block_table,
    pa_block_size,
    split_count,
    n1=32,
    query_tile_rows=6,
    streaming_mode=False,
    tnd_layout=False,
):
    """Compile the dynamic TensorSpec contract using the standard DSL entry."""
    N1 = int(n1)
    TILE_M = N1 * int(query_tile_rows)
    has_cmp_residual_k = int(mask_mode) == 3 and int(cmp_ratio) != 1
    topk = int(topk)
    candidate_enabled = bool(candidate_enabled)
    candidate_topk = int(candidate_topk)
    mask_mode = int(mask_mode)
    cmp_ratio = int(cmp_ratio)

    # 执行融合设备内核，按 metadata 调度完整任务和 LD 分片。
    @kernel
    def Quant_lightning_indexer(
        query_address,
        key_address,
        weights: Tensor,
        query_scale,
        key_scale_address,
        key_stride0,
        key_dequant_scale_stride0,
        cu_seqlens_q: Tensor,
        seqused_q: Tensor,
        seqused_k: Tensor,
        cmp_residual_k: Tensor,
        block_table: Tensor,
        key_out: Tensor,
        candidate_key_out: Tensor,
        sparse_indices_address,
        sparse_value_bits_address,
        candidate_indices_address,
        candidate_value_bits_address,
        candidate_length: Tensor,
        worker_count,
        total_q,
        batch_count,
        tokens,
        max_tasks_per_batch,
        key_rows,
        scale_rows,
        metadata: Tensor,
        task_count,
        final_sparse_indices_address,
        final_sparse_bits_address,
        final_candidate_indices_address,
        final_candidate_bits_address,
        return_values,
        merge_workers,
        output_offset_address,
    ):
        ld_enabled = metadata[LD_RECORD_START, 0] != 0
        final_sparse_indices = make_tensor(
            make_pointer(
                dtypes.int32, dtypes.int64(final_sparse_indices_address), MemLoc.GM
            ),
            make_layout((total_q, topk), stride=(topk, 1)),
        )
        sparse_indices = make_tensor(
            make_pointer(dtypes.int32, dtypes.int64(sparse_indices_address), MemLoc.GM),
            make_layout((64 * query_tile_rows, topk), stride=(topk, 1)),
        )
        sparse_value_bits = make_tensor(
            make_pointer(
                dtypes.uint16, dtypes.int64(sparse_value_bits_address), MemLoc.GM
            ),
            make_layout((64 * query_tile_rows, topk), stride=(topk, 1)),
        )
        final_sparse_bits = make_tensor(
            make_pointer(
                dtypes.uint16, dtypes.int64(final_sparse_bits_address), MemLoc.GM
            ),
            make_layout((total_q, topk), stride=(topk, 1)),
        )
        final_candidate_indices = make_tensor(
            make_pointer(
                dtypes.int32, dtypes.int64(final_candidate_indices_address), MemLoc.GM
            ),
            make_layout((total_q, candidate_topk), stride=(candidate_topk, 1)),
        )
        final_candidate_bits = make_tensor(
            make_pointer(
                dtypes.uint16, dtypes.int64(final_candidate_bits_address), MemLoc.GM
            ),
            make_layout((total_q, candidate_topk), stride=(candidate_topk, 1)),
        )
        candidate_indices = make_tensor(
            make_pointer(
                dtypes.int32, dtypes.int64(candidate_indices_address), MemLoc.GM
            ),
            make_layout(
                (64 * query_tile_rows, candidate_topk), stride=(candidate_topk, 1)
            ),
        )
        candidate_value_bits = make_tensor(
            make_pointer(
                dtypes.uint16, dtypes.int64(candidate_value_bits_address), MemLoc.GM
            ),
            make_layout(
                (64 * query_tile_rows, candidate_topk), stride=(candidate_topk, 1)
            ),
        )
        query = make_tensor(
            make_pointer(dtypes.fp4x2_e2m1, dtypes.int64(query_address), MemLoc.GM),
            make_layout((total_q * N1, LOGICAL_D), stride=(LOGICAL_D, 1)),
        )
        key = make_tensor(
            make_pointer(dtypes.fp4x2_e2m1, dtypes.int64(key_address), MemLoc.GM),
            make_layout(
                (key_rows, LOGICAL_D),
                stride=(key_stride0 * 2 if tnd_layout else LOGICAL_D, 1),
            ),
        )
        key_scale = make_tensor(
            make_pointer(
                dtypes.float8_e8m0, dtypes.int64(key_scale_address), MemLoc.GM
            ),
            make_layout(
                (scale_rows, 2, 2),
                stride=(key_dequant_scale_stride0 if tnd_layout else 4, 2, 1),
            ),
        )
        QliFusedKernel(
            topk,
            candidate_enabled,
            candidate_topk,
            mask_mode,
            cmp_ratio,
            has_cu_seqlens_q,
            has_seqused_q,
            has_seqused_k,
            has_block_table,
            pa_block_size,
            N1,
            query_tile_rows,
            metadata,
            output_offset_address,
            return_values,
            streaming_mode,
            tnd_layout,
        )(
            query,
            key,
            weights,
            query_scale,
            key_scale,
            key_stride0,
            key_dequant_scale_stride0,
            cu_seqlens_q,
            seqused_q,
            seqused_k,
            cmp_residual_k,
            block_table,
            key_out,
            candidate_key_out,
            sparse_indices,
            sparse_value_bits,
            candidate_indices,
            candidate_value_bits,
            candidate_length,
            worker_count,
            total_q,
            batch_count,
            tokens,
            max_tasks_per_batch,
            metadata,
            split_count,
            task_count,
            sparse_indices_address,
            sparse_value_bits_address,
            candidate_indices_address,
            candidate_value_bits_address,
            final_sparse_indices_address,
            final_sparse_bits_address,
            final_candidate_indices_address,
            final_candidate_bits_address,
        )
        if ld_enabled:
            global_sync_all()
            # 显式 UB 视图复用已排空的阶段空间，不重置 RingBuffer 编译期资源。
            if candidate_enabled:
                CandidateLdMergeStage(topk, split_count, candidate_optimized=True)(
                    sparse_indices,
                    sparse_value_bits,
                    final_sparse_indices,
                    final_sparse_bits,
                    metadata,
                    total_q,
                    worker_count * 2,
                    cu_seqlens_q,
                    query_tile_rows,
                    has_cu_seqlens_q,
                    output_offset_address,
                )
            else:
                LdMergeStage(topk, split_count)(
                    sparse_indices,
                    sparse_value_bits,
                    final_sparse_indices,
                    final_sparse_bits,
                    metadata,
                    total_q,
                    worker_count * 2,
                    cu_seqlens_q,
                    query_tile_rows,
                    has_cu_seqlens_q,
                    output_offset_address,
                    return_values,
                )

            if candidate_enabled:
                vec_sync_all()
                # 显式 UB 视图复用已排空的阶段空间，不重置 RingBuffer 编译期资源。
                CandidateLdMergeStage(
                    candidate_topk, split_count, candidate_optimized=True
                )(
                    candidate_indices,
                    candidate_value_bits,
                    final_candidate_indices,
                    final_candidate_bits,
                    metadata,
                    total_q,
                    worker_count * 2,
                    cu_seqlens_q,
                    query_tile_rows,
                    has_cu_seqlens_q,
                )

    # 绑定张量形状与工作区，启动融合设备内核。
    @host
    def run_fused_cv(
        query_address,
        key_address,
        weights: Tensor,
        query_scale,
        key_scale_address,
        key_stride0,
        key_dequant_scale_stride0,
        cu_seqlens_q: Tensor,
        seqused_q: Tensor,
        seqused_k: Tensor,
        cmp_residual_k: Tensor,
        block_table: Tensor,
        key_out: Tensor,
        candidate_key_out: Tensor,
        sparse_indices_address,
        sparse_value_bits_address,
        candidate_indices_address,
        candidate_value_bits_address,
        candidate_length: Tensor,
        worker_count,
        total_q,
        batch_count,
        tokens,
        max_tasks_per_batch,
        key_rows,
        scale_rows,
        metadata: Tensor,
        task_count,
        final_sparse_indices_address,
        final_sparse_bits_address,
        final_candidate_indices_address,
        final_candidate_bits_address,
        return_values,
        merge_workers,
        output_offset_address,
    ):
        Quant_lightning_indexer[worker_count](
            query_address,
            key_address,
            weights,
            query_scale,
            key_scale_address,
            key_stride0,
            key_dequant_scale_stride0,
            cu_seqlens_q,
            seqused_q,
            seqused_k,
            cmp_residual_k,
            block_table,
            key_out,
            candidate_key_out,
            sparse_indices_address,
            sparse_value_bits_address,
            candidate_indices_address,
            candidate_value_bits_address,
            candidate_length,
            worker_count,
            total_q,
            batch_count,
            tokens,
            max_tasks_per_batch,
            key_rows,
            scale_rows,
            metadata,
            task_count,
            final_sparse_indices_address,
            final_sparse_bits_address,
            final_candidate_indices_address,
            final_candidate_bits_address,
            return_values,
            merge_workers,
            output_offset_address,
        )

    query_rows_dim = cannbotdsl.Dim("T1")
    batch_dim = cannbotdsl.Dim("B")
    block_table_width_dim = cannbotdsl.Dim("MAX_BLOCKS")
    token_capacity_dim = cannbotdsl.Dim("S2_PAD")
    candidate_capacity_dim = cannbotdsl.Dim("CANDIDATE_PAD")
    cannbotdsl.Dim("SPARSE_WIDTH")
    candidate_width = cannbotdsl.Dim("CANDIDATE_WIDTH")
    partial_rows = cannbotdsl.Dim("PARTIAL_ROWS")
    cannbotdsl.Dim("METADATA_TASKS")
    workspace_rows = cannbotdsl.Dim(
        "WORKSPACE_ROWS", min=1, max=MAX_CUBE_WORKERS * (TILE_M // N1)
    )
    fake = TensorSpec
    dummy_i32_spec = fake((1,), dtypes.int32)
    # TND 可同时省略 used_q/used_k；此时 CU 长度直接绑定动态轴，
    # 避免 B 只以 B+1 派生表达式出现而被编译器拒绝。
    cu_length_dim = batch_dim + 1
    if tnd_layout and not (has_seqused_q or has_seqused_k or has_cmp_residual_k):
        cu_length_dim = cannbotdsl.Dim("CU_LENGTH")
    cu_spec = (
        fake((cu_length_dim,), dtypes.int32) if has_cu_seqlens_q else dummy_i32_spec
    )
    seqused_q_spec = (
        fake((batch_dim,), dtypes.int32) if has_seqused_q else dummy_i32_spec
    )
    seqused_k_spec = (
        fake((batch_dim,), dtypes.int32) if has_seqused_k else dummy_i32_spec
    )
    residual_spec = (
        fake((batch_dim,), dtypes.int32) if has_cmp_residual_k else dummy_i32_spec
    )
    block_table_spec = (
        (
            fake((cu_length_dim,), dtypes.int32)
            if tnd_layout
            else fake((batch_dim, block_table_width_dim), dtypes.int32)
        )
        if has_block_table
        else fake((1, 1), dtypes.int32)
    )
    candidate_key_spec = (
        fake(
            (
                workspace_rows,
                candidate_capacity_dim,
            ),
            dtypes.uint16,
        )
        if candidate_enabled
        else fake((workspace_rows, 1), dtypes.uint16)
    )
    (
        fake((query_rows_dim, candidate_width), dtypes.int32)
        if candidate_enabled
        else fake((1, candidate_width), dtypes.int32)
    )

    compiled = cannbotdsl.compile(
        run_fused_cv,
        dtypes.int64,
        dtypes.int64,
        fake((query_rows_dim, N1), dtypes.float32),
        fake((query_rows_dim * N1, 2, 2), dtypes.float8_e8m0),
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        cu_spec,
        seqused_q_spec,
        seqused_k_spec,
        residual_spec,
        block_table_spec,
        fake((workspace_rows, token_capacity_dim), dtypes.uint16),
        candidate_key_spec,
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        fake((partial_rows,), dtypes.int32),
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        fake((METADATA_ROWS, METADATA_FIELDS), dtypes.int32),
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
    )
    return compiled


def clear_caches():
    """Close cached dynamic executables before and after native export."""
    with _COMPILED_KERNEL_LOCK:
        for compiled in _COMPILED_KERNEL.values():
            compiled.close()
        _COMPILED_KERNEL.clear()


def _get_compiled_fused_runner(*config):
    """Cache the dynamic QLI executable, independently of B/T/S2."""
    # DSL native policy is process-wide; never compile a missing runtime binary.
    cache_key = tuple(config)
    with _COMPILED_KERNEL_LOCK:
        compiled = _COMPILED_KERNEL.get(cache_key)
        if compiled is None:
            compiled = _build_compiled_fused_runner(*config)
            _COMPILED_KERNEL[cache_key] = compiled
        return compiled


def _validate_sequence_tensor(name, tensor, shape, device):
    if tensor is None:
        return
    if tuple(tensor.shape) != tuple(shape):
        raise ValueError(f"{name} must have shape {tuple(shape)}")
    if tensor.dtype != torch.int32:
        raise TypeError(f"{name} must be int32")
    if tensor.device != device:
        raise ValueError(f"{name} must be on the same NPU as q")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _validate_qli_pa_sequences(
    query,
    cu_seqlens_q,
    seqused_q,
    seqused_k,
    cmp_residual_k,
    block_table,
    output_idx_offset,
    mask_mode,
    cmp_ratio,
):
    """校验 PA 分页路径的序列边界、输出偏移与页表元信息。"""
    t = int(query.shape[0])
    if t <= 0:
        raise ValueError("Q must contain at least one T1 row")
    has_cu_seqlens_q = cu_seqlens_q is not None
    batch = int(cu_seqlens_q.numel()) - 1 if has_cu_seqlens_q else 1
    if batch <= 0:
        raise ValueError("Cu_seqlens_q must have shape (B+1,)")

    if has_cu_seqlens_q:
        _validate_sequence_tensor(
            "cu_seqlens_q", cu_seqlens_q, (batch + 1,), query.device
        )
    _validate_sequence_tensor(
        "output_idx_offset", output_idx_offset, (t, 1), query.device
    )
    _validate_sequence_tensor("seqused_q", seqused_q, (batch,), query.device)
    _validate_sequence_tensor("seqused_k", seqused_k, (batch,), query.device)
    if int(mask_mode) == 3 and int(cmp_ratio) != 1:
        if cmp_residual_k is None:
            raise ValueError(
                "Cmp_residual_k is required for mask_mode=3 and cmp_ratio!=1"
            )
        _validate_sequence_tensor(
            "cmp_residual_k", cmp_residual_k, (batch,), query.device
        )
    elif cmp_residual_k is not None:
        raise ValueError(
            "Cmp_residual_k must be None unless mask_mode=3 and cmp_ratio!=1"
        )
    if block_table is not None:
        if block_table.ndim != 2 or int(block_table.shape[0]) != batch:
            raise ValueError("Block_table must have shape (B,max_num_blocks_per_seq)")
        if block_table.dtype != torch.int32:
            raise TypeError("Block_table must be int32")
        if block_table.device != query.device:
            raise ValueError("Block_table must be on the same NPU as q")
        if not block_table.is_contiguous():
            raise ValueError("Block_table must be contiguous")

    return t, batch, has_cu_seqlens_q


def _validate_qli_candidate(candidate_topk_blocks, candidate_block_size):
    """Candidate 关闭与开启时分别校验配套属性。"""
    if candidate_topk_blocks == -1:
        if candidate_block_size != -1:
            raise ValueError(
                "Candidate_block_size must be -1 when candidate is disabled"
            )
    elif candidate_topk_blocks <= 0 or candidate_block_size != CANDIDATE_BLOCK_SIZE:
        raise ValueError("Candidate requires positive topk_blocks and block_size=8")


def _validate_qli_compute_attributes(topk, mask_mode, cmp_ratio):
    """校验选择容量、掩码模式与压缩比例的支持范围。"""
    if int(topk) < 1 or int(topk) > 8192:
        raise ValueError("Topk must be in ASC-supported range [1,8192]")
    if int(mask_mode) not in (0, 3):
        raise ValueError("Mask_mode must be 0 or 3")
    if int(cmp_ratio) < 1 or int(cmp_ratio) > 128:
        raise ValueError("Cmp_ratio must be in [1,128]")


def _prepare_key_storage(key, key_dequant_scale, block_table, tnd_layout):
    if tnd_layout:
        if block_table is not None:
            raise ValueError("Block_table must be None for TND K")
        key_stride0 = _tnd_axis0_stride(key, "k", (PACKED_D, 1), PACKED_D)
        key_dequant_scale_stride0 = _tnd_axis0_stride(
            key_dequant_scale, "descale_k", (4, 2, 1), 4
        )
        key_storage_rows = int(key.shape[0])
        key_storage = key
        key_dequant_scale_storage = key_dequant_scale
    else:
        key_stride0, key_storage_rows = _pa_axis0_storage_rows(
            key,
            "k",
            (PACKED_D, PACKED_D, 1),
            KEY_STORAGE_ROW_ELEMENTS,
        )
        (
            key_dequant_scale_stride0,
            key_dequant_scale_storage_rows,
        ) = _pa_axis0_storage_rows(
            key_dequant_scale,
            "descale_k",
            (KEY_SCALE_STORAGE_ROW_ELEMENTS, 2 * 2, 2, 1),
            KEY_SCALE_STORAGE_ROW_ELEMENTS,
        )
        # CANNBotDSL's packed-FP4 host carrier is compact-only. Expose the
        # allocation span, including axis-0 gaps, as a zero-copy compact carrier;
        # the original page strides are passed separately and consumed on NPU.
        key_storage = torch.as_strided(
            key,
            (key_storage_rows, PACKED_D),
            (KEY_STORAGE_ROW_ELEMENTS, 1),
        )
        key_dequant_scale_storage = torch.as_strided(
            key_dequant_scale,
            (key_dequant_scale_storage_rows, 4),
            (KEY_SCALE_STORAGE_ROW_ELEMENTS, 1),
        )
    return (
        key_storage,
        key_dequant_scale_storage,
        key_storage_rows,
        key_stride0,
        key_dequant_scale_stride0,
    )


def _validate_public_tensor_contract(
    query,
    key,
    weights,
    query_dequant_scale,
    key_dequant_scale,
    quant_mode,
    layout_q,
    layout_k,
    tnd_layout,
):
    if int(quant_mode) != 1:
        raise ValueError(
            "Quant_lightning_indexer public contract supports quant_mode=1 only"
        )
    if layout_q != "TND":
        raise ValueError("Layout_q must be TND under the current xlsx contract")
    if layout_k not in ("PA_BBND", "TND"):
        raise ValueError("Layout_k must be PA_BBND under the current xlsx contract")
    if query.device.type != "npu" or key.device.type != "npu":
        raise ValueError("Quant_lightning_indexer requires NPU q/k tensors")
    if query.dtype != torch.uint8 or key.dtype != torch.uint8:
        raise TypeError("Public q/k storage must be uint8 packed MXFP4")
    if key.ndim != (3 if tnd_layout else 4):
        raise ValueError("Q must be [T1,32,64], k must be [block_num,block_size,1,64]")
    if key.shape[-2] != N2 or key.shape[-1] != PACKED_D:
        raise ValueError("Public static axes require N1=32, N2=1 and logical D=128")
    if tuple(weights.shape) != tuple(query.shape[:2]) or weights.dtype != torch.float32:
        raise ValueError("W must be FP32 with shape (T1,N1)")
    if query_dequant_scale is None:
        raise ValueError("Descale_q is a required xlsx input")
    expected_q_scale = (*query.shape[:2], LOGICAL_D // 64, 2)
    if tuple(query_dequant_scale.shape) != expected_q_scale:
        raise ValueError("Descale_q must have corrected xlsx shape (T1,N1,D/64,2)")
    if key_dequant_scale is None:
        raise ValueError("Descale_k is a required xlsx input")
    expected_k_scale = tuple(key.shape[:-1]) + (2, 2)
    if tuple(key_dequant_scale.shape) != expected_k_scale:
        raise ValueError("Descale_k must have xlsx shape (block_num,block_size,N2,2,2)")
    if key.device != query.device or key_dequant_scale.device != query.device:
        raise ValueError("Q, k and descale_k must share an NPU device")


def quant_lightning_indexer(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    descale_q: torch.Tensor,
    descale_k: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    cmp_residual_k: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    output_idx_offset: Optional[torch.Tensor] = None,
    metadata: Optional[torch.Tensor] = None,
    *,
    topk: int,
    quant_mode: int,
    max_seqlen_q: int = -1,
    mask_mode: int = 0,
    cmp_ratio: int = 1,
    layout_q: str = "TND",
    layout_k: str = "TND",
    return_value: bool = False,
    candidate_topk_blocks: int = -1,
    candidate_block_size: int = -1,
) -> tuple[torch.Tensor, ...]:
    """在 NPU 上执行公开的 N1=32、MXFP4 QLI 接口。

    cu_seqlens_q 描述 TND 查询边界；seqused_k 给出每个 batch 的有效 K 长度。
    PA_BBND 通过 block_table 映射物理页，TND K 使用序列边界。
    metadata 决定完整任务及 LD 分片，具体 CV 切块由编译配置选择。
    """
    if q.ndim != 3 or tuple(q.shape[1:]) != (N1, PACKED_D):
        raise ValueError("QLI requires N1=32 and logical D=128")
    tnd_layout = layout_k == "TND"
    if int(max_seqlen_q) < -1:
        raise ValueError("Max_seqlen_q must be -1 or non-negative")
    if metadata is None:
        raise ValueError(
            "Metadata is required; call quant_lightning_indexer_metadata before this operator"
        )
    query_heads = int(q.shape[1])
    query_tile_rows = min(QUERY_TILE_ROWS, 256 // query_heads)
    tile_m = query_heads * query_tile_rows
    query, key, weights = q, k, w
    query_dequant_scale, key_dequant_scale = descale_q, descale_k
    _validate_public_tensor_contract(
        query,
        key,
        weights,
        query_dequant_scale,
        key_dequant_scale,
        quant_mode,
        layout_q,
        layout_k,
        tnd_layout,
    )
    (
        key_storage,
        key_dequant_scale_storage,
        key_storage_rows,
        key_stride0,
        key_dequant_scale_stride0,
    ) = _prepare_key_storage(key, key_dequant_scale, block_table, tnd_layout)
    _validate_qli_candidate(candidate_topk_blocks, candidate_block_size)
    # Keep this as a Python/JIT specialization constant. Disabled launches
    # compile out Candidate block-max, key generation, TopK and UB resources.
    candidate_enabled = int(candidate_topk_blocks) > 0
    pa_block_size = TILE_N if tnd_layout else int(key.shape[1])
    if (
        pa_block_size <= 0
        or pa_block_size > MAX_PA_BLOCK_SIZE
        or pa_block_size % PA_BLOCK_ALIGNMENT != 0
    ):
        raise ValueError("PA block_size must be a multiple of 16 in (0, 1024]")
    _validate_qli_compute_attributes(topk, mask_mode, cmp_ratio)
    if query_dequant_scale.dtype != torch.uint8:
        raise TypeError("Descale_q must use uint8 E8M0 storage")
    if key_dequant_scale.dtype != torch.uint8:
        raise TypeError("Descale_k must use uint8 E8M0 storage")
    for name, tensor in (
        ("w", weights),
        ("descale_q", query_dequant_scale),
    ):
        if tensor.device != query.device:
            raise ValueError(f"{name} must be on the same NPU as q")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    if not query.is_contiguous():
        raise ValueError("Q must be contiguous")

    t, batch, has_cu_seqlens_q = _validate_qli_pa_sequences(
        query,
        cu_seqlens_q,
        seqused_q,
        seqused_k,
        cmp_residual_k,
        block_table,
        output_idx_offset,
        mask_mode,
        cmp_ratio,
    )
    has_seqused_q = seqused_q is not None
    has_seqused_k = seqused_k is not None
    has_cmp_residual_k = cmp_residual_k is not None
    if tnd_layout:
        if cu_seqlens_k is None:
            if batch != 1:
                raise ValueError("Cu_seqlens_k is required for multiple TND batches")
        else:
            _validate_sequence_tensor(
                "cu_seqlens_k", cu_seqlens_k, (batch + 1,), query.device
            )
    has_block_table = (
        cu_seqlens_k is not None if tnd_layout else block_table is not None
    )
    logical_k_capacity = (
        key_storage_rows
        if tnd_layout
        else (int(block_table.shape[1]) if has_block_table else int(key.shape[0]))
        * pa_block_size
    )
    tokens_pad = ceil_div(logical_k_capacity, TILE_N) * TILE_N
    if tokens_pad <= 0:
        raise ValueError("K storage must expose at least one PA page")
    # Optional xlsx tensors remain optional at the public boundary.  One
    # uninitialized carrier is sufficient because the JIT specialization
    # compiles out every load when the corresponding tensor is absent.
    dummy_i32 = torch.empty((1,), dtype=torch.int32, device=query.device)
    dummy_table = torch.empty((1, 1), dtype=torch.int32, device=query.device)
    cu_kernel = cu_seqlens_q if has_cu_seqlens_q else dummy_i32
    seqused_q_kernel = seqused_q if has_seqused_q else dummy_i32
    seqused_k_kernel = seqused_k if has_seqused_k else dummy_i32
    cmp_residual_kernel = cmp_residual_k if has_cmp_residual_k else dummy_i32
    block_table_kernel = (
        (cu_seqlens_k if tnd_layout else block_table)
        if has_block_table
        else dummy_table
    )

    max_query_length = int(max_seqlen_q) if int(max_seqlen_q) > 0 else t
    max_tasks_per_batch = ceil_div(max_query_length, tile_m // query_heads)
    # Workspace capacity; AICPU metadata owns the actual shard boundaries.
    split_count = max(1, min(MAX_MERGE_PARTS, STREAM_TOPK_TRUNK_LEN // int(topk)))
    if (
        metadata.dtype != torch.int32
        or metadata.ndim != 1
        or metadata.numel() != METADATA_ELEMENTS
        or not metadata.is_contiguous()
    ):
        raise ValueError(
            "Metadata must be contiguous int32 [1024] in ASC LI/LD boundary format"
        )
    if metadata.device != q.device:
        raise ValueError("Metadata must be on the same device as q")
    metadata = metadata.view(METADATA_ROWS, METADATA_FIELDS)
    worker_count = int(torch.npu.get_device_properties(q.device).cube_core_num)
    if not 1 <= worker_count <= LD_RECORD_START:
        raise RuntimeError(
            f"Unsupported Cube core count {worker_count}; LI metadata supports 1..{LD_RECORD_START}"
        )
    workspace_row_count = worker_count * (tile_m // query_heads)
    shard_pad = tokens_pad
    # Public outputs and GM scratch are allocation-only Torch calls.  The
    # fused kernel writes every public row, including invalid/padded rows.
    sparse_indices_2d = torch.empty(
        (worker_count * 2 * query_tile_rows, int(topk)),
        dtype=torch.int32,
        device=query.device,
    )
    sparse_values_2d = torch.empty(
        (worker_count * 2 * query_tile_rows, int(topk)),
        dtype=torch.bfloat16,
        device=weights.device,
    )
    candidate_count = max(0, int(candidate_topk_blocks))
    candidate_indices_2d = torch.empty(
        (worker_count * 2 * query_tile_rows, candidate_count),
        dtype=torch.int32,
        device=query.device,
    )
    candidate_length_1d = torch.empty(
        (t * split_count,), dtype=torch.int32, device=query.device
    )

    key_workspace = torch.empty(
        (workspace_row_count, shard_pad),
        dtype=torch.uint16,
        device=query.device,
    )
    candidate_key_workspace = torch.empty(
        (
            workspace_row_count,
            shard_pad // CANDIDATE_BLOCK_SIZE if candidate_enabled else 1,
        ),
        dtype=torch.uint16,
        device=query.device,
    )
    kernel_candidate_count = candidate_count if candidate_enabled else 1
    candidate_indices_kernel = (
        candidate_indices_2d
        if candidate_enabled
        else torch.empty((1, split_count), dtype=torch.int32, device=query.device)
    )
    candidate_value_workspace = torch.empty(
        (worker_count * 2 * query_tile_rows, kernel_candidate_count),
        dtype=torch.uint16,
        device=query.device,
    )

    # GM aliases describe native packed formats without device copies.
    query_address = query.data_ptr()
    query_scale_storage = query_dequant_scale.reshape(t * query_heads, 2, 2).view(
        torch.float8_e8m0fnu
    )

    final_sparse_indices = torch.empty(
        (t, int(topk)), dtype=torch.int32, device=query.device
    )
    final_sparse_values = torch.empty(
        (t, int(topk)), dtype=torch.bfloat16, device=query.device
    )
    final_candidate_indices = torch.empty(
        (t, kernel_candidate_count), dtype=torch.int32, device=query.device
    )
    final_candidate_bits = torch.empty(
        (t, kernel_candidate_count), dtype=torch.uint16, device=query.device
    )
    _get_compiled_fused_runner(
        int(topk),
        candidate_enabled,
        kernel_candidate_count,
        int(mask_mode),
        int(cmp_ratio),
        has_cu_seqlens_q,
        has_seqused_q,
        has_seqused_k,
        has_block_table,
        pa_block_size,
        split_count,
        query_heads,
        query_tile_rows,
        bool(
            not candidate_enabled
            and int(topk) == 512
            and query_heads == 32
            and query_tile_rows == 6
            and logical_k_capacity > 65536
            and batch < worker_count
            and t <= worker_count * query_tile_rows
            and not tnd_layout
        ),
        tnd_layout,
    )(
        query_address,
        key_storage.data_ptr(),
        weights,
        query_scale_storage,
        key_dequant_scale_storage.data_ptr(),
        key_stride0,
        key_dequant_scale_stride0,
        cu_kernel,
        seqused_q_kernel,
        seqused_k_kernel,
        cmp_residual_kernel,
        block_table_kernel,
        key_workspace,
        candidate_key_workspace,
        sparse_indices_2d.data_ptr(),
        sparse_values_2d.view(torch.uint16).data_ptr(),
        candidate_indices_kernel.data_ptr(),
        candidate_value_workspace.data_ptr(),
        candidate_length_1d,
        worker_count,
        t,
        batch,
        logical_k_capacity,
        max_tasks_per_batch,
        key_storage_rows,
        key_dequant_scale_storage.shape[0],
        metadata,
        metadata.shape[0],
        final_sparse_indices.data_ptr(),
        final_sparse_values.view(torch.uint16).data_ptr(),
        final_candidate_indices.data_ptr(),
        final_candidate_bits.data_ptr(),
        int(return_value),
        min(64, t),
        0 if output_idx_offset is None else output_idx_offset.data_ptr(),
    )

    sparse_indices_2d, sparse_values_2d = final_sparse_indices, final_sparse_values
    if candidate_enabled:
        candidate_indices_2d = final_candidate_indices
    candidate_length_1d = candidate_length_1d[:t]
    return (
        sparse_indices_2d.view(t, N2, int(topk)),
        sparse_values_2d.view(t, N2, int(topk))
        if return_value
        else sparse_values_2d.reshape(-1)[:0],
        candidate_indices_2d.view(t, N2, candidate_count)
        if candidate_enabled
        else candidate_indices_2d.reshape(-1),
        candidate_length_1d.view(t, N2)
        if candidate_enabled
        else candidate_length_1d[:0],
    )


@export("qli")
def export_qli():
    """Export static profiles with dynamic tensor axes, without NPU execution."""
    import json

    defaults = {
        "topk": 512,
        "candidate_enabled": True,
        "candidate_topk": 2048,
        "mask_mode": 3,
        "cmp_ratio": 1,
        "has_cu_seqlens_q": True,
        "has_seqused_q": True,
        "has_seqused_k": True,
        "has_block_table": True,
        "pa_block_size": 128,
        "split_count": 8,
        "n1": 32,
        "query_tile_rows": 6,
        "tnd_layout": False,
    }
    profiles = [
        dict(
            candidate_enabled=enabled,
            candidate_topk=2048 if enabled else 1,
            mask_mode=mask,
            cmp_ratio=ratio,
            pa_block_size=page,
            has_seqused_q=has_q,
            has_seqused_k=has_k,
            has_cu_seqlens_q=has_cu,
            has_block_table=has_table,
            tnd_layout=tnd,
        )
        for enabled in (False, True)
        for mask in (0, 3)
        for ratio in ((1, 2) if mask == 3 else (1,))
        for tnd in (False, True)
        for page in ((TILE_N,) if tnd else (64, 128))
        for has_q in (False, True)
        for has_cu in (False, True)
        for has_k in (False, True)
        for has_table in (False, True)
    ]
    profile_path = os.environ.get("CANNBOTDSL_DS41_PROFILES")
    if profile_path:
        with open(profile_path, encoding="utf-8") as stream:
            profiles = json.load(stream).get("qli", profiles)
    clear_caches()
    try:
        exported = set()
        for profile in profiles:
            unknown = set(profile) - set(defaults)
            if unknown:
                raise ValueError(f"Unknown export profile fields: {sorted(unknown)}")
            config = dict(defaults, **profile)
            # cmp_ratio is unused without the causal mask; export one binary.
            identity = tuple(
                (name, value)
                for name, value in config.items()
                if name != "cmp_ratio" or config["mask_mode"] == 3
            )
            if identity in exported:
                continue
            exported.add(identity)
            _get_compiled_fused_runner(
                *(config[name] for name in defaults if name != "tnd_layout"),
                False,
                config["tnd_layout"],
            )
            if (
                not config["tnd_layout"]
                and not config["candidate_enabled"]
                and config["topk"] == 512
                and config["n1"] == 32
                and config["query_tile_rows"] == 6
            ):
                _get_compiled_fused_runner(
                    *(config[name] for name in defaults if name != "tnd_layout"),
                    True,
                    False,
                )
    finally:
        clear_caches()


class QliTndCube(QliCube):
    @jit
    def prefetch_key(
        self,
        key,
        key_scale,
        block_table,
        batch_idx,
        actual_s2,
        key_stride0,
        key_dequant_scale_stride0,
        has_block_table,
        token_tile,
        qk_handoff,
    ):
        key_write = self.k_l1.next_write()
        scale_write = self.k_scale_l1.next_write()
        base = dtypes.int64(0)
        if has_block_table:
            base = dtypes.int64(block_table[batch_idx])
        logical_row = token_tile * TILE_N
        count = min(dtypes.int64(QK_COLUMNS), dtypes.int64(actual_s2) - logical_row)
        source_row = base + logical_row
        # Tile windows pin the full NZ frame while the dynamic source slice
        # keeps the copy height clamped to the available key rows.
        mem_copy(
            tile_slice(key_write, (QK_COLUMNS, LOGICAL_D), (0, 0)),
            key[source_row : source_row + count, None],
        )
        mem_copy(
            tile_slice(scale_write, (4, QK_COLUMNS), (0, 0)),
            key_scale[source_row : source_row + count, None, None]
            .flatten(1, 2)
            .permute(1, 0),
            options=self.scale_b,
        )


def _tnd_axis0_stride(tensor, name, inner_strides, row_bytes):
    if tensor.shape[0] <= 0 or tuple(tensor.stride()[1:]) != inner_strides:
        raise ValueError(f"{name} requires positive T2 and contiguous inner axes")
    stride = int(tensor.stride(0))
    if stride < row_bytes:
        raise ValueError(f"{name} requires non-overlapping positive axis-0 stride")
    return stride
