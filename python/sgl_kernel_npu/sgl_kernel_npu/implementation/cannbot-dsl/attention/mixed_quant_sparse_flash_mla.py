# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import math
import threading
from functools import lru_cache
from itertools import product
from types import SimpleNamespace

import cannbotdsl
import torch
from cannbotdsl import (
    PIPE,
    CArray,
    Dim,
    MemLoc,
    ProfileSpec,
    Tensor,
    TensorSpec,
    datastruct,
    dtypes,
    host,
    is_aic,
    make_layout,
)
from cannbotdsl.aot import export
from cannbotdsl.buffer import Buffer
from cannbotdsl.lang.constexpr import const_expr, range_constexpr
from cannbotdsl.lang.jit import jit
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops import reg as rr
from cannbotdsl.ops.arch import get_block_idx, get_subblock_dim, get_subblock_id
from cannbotdsl.ops.cube import set_mmad_direction
from cannbotdsl.ops.matmul import matmul
from cannbotdsl.ops.memcpy import copy_options, mem_copy
from cannbotdsl.ops.scalar import vec_store_bypass
from cannbotdsl.ops.sync import (
    cube_sync_block_arrive,
    cube_sync_intra_wait,
    global_sync_all,
    onchip_rewind,
    vec_busy_wait,
    vec_sync_all,
    vec_sync_block_wait,
    vec_sync_intra_arrive,
)
from cannbotdsl.ring_buffer import RingBuffer
from cannbotdsl.tensor import reinterpret, tile_slice

# Metadata constants and public host entry are re-exported from the metadata module.
if __package__:
    from .mixed_quant_sparse_flash_mla_metadata import (
        AIC_CORE_MAX_NUM,
        FA_BN2_END_INDEX,
        FA_BN2_START_INDEX,
        FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX,
        FA_M_END_INDEX,
        FA_M_START_INDEX,
        FA_METADATA_SIZE,
        FA_S2_END_INDEX,
        FA_S2_START_INDEX,
        FD_BN2_IDX_INDEX,
        FD_CORE_ENABLE_INDEX,
        FD_M_IDX_INDEX,
        FD_M_START_INDEX,
        FD_METADATA_BASE,
        FD_METADATA_SIZE,
        FD_USED_VEC_NUM_WORD,
        FD_WORKSPACE_IDX_INDEX,
        FD_WORKSPACE_NUM_INDEX,
        MQSMLA_METADATA_TOTAL_SIZE,
        _get_cube_core_num,
    )
else:
    from mixed_quant_sparse_flash_mla_metadata import (
        AIC_CORE_MAX_NUM,
        FA_BN2_END_INDEX,
        FA_BN2_START_INDEX,
        FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX,
        FA_M_END_INDEX,
        FA_M_START_INDEX,
        FA_METADATA_SIZE,
        FA_S2_END_INDEX,
        FA_S2_START_INDEX,
        FD_BN2_IDX_INDEX,
        FD_CORE_ENABLE_INDEX,
        FD_M_IDX_INDEX,
        FD_M_START_INDEX,
        FD_METADATA_BASE,
        FD_METADATA_SIZE,
        FD_USED_VEC_NUM_WORD,
        FD_WORKSPACE_IDX_INDEX,
        FD_WORKSPACE_NUM_INDEX,
        MQSMLA_METADATA_TOTAL_SIZE,
        _get_cube_core_num,
    )

# Address precomputation preserves logical sparse-column order in both UB paths.
# Dense power-of-two pages use vector addressing; other layouts use scalar addressing.
_ADDR_VEC_OVERRIDE = None
_ADDR_VEC_MODES = ("off", "pipe", "local")


# NZ face stride in BF16 elements; UB padding is skipped by copy-out.
_NZ_ROW = 16
_NZ_PAD_ROWS = 17
_NZ_CHUNK = _NZ_PAD_ROWS * _NZ_ROW
_FD_READY_BASE = FD_USED_VEC_NUM_WORD + 1
_FD_ASYNC_MODE_WORD = FD_USED_VEC_NUM_WORD + 50

# Address-prepass buffers are reclaimed after the block handshake.
# Main-loop and FD reduction buffers reuse UB at separate phases.
_ADDR_COMPACT_BT = 8192
_ADDR_LOCAL_MAX_ROWS = 128
_ADDR_LOCAL_ROWS = 4
_ADDR_LOCAL_ORI_K = 256
_ADDR_LOCAL_CMP_K = 1024
_ADDR_LOCAL_ORI_INDEX_OFFSET = 2 * _ADDR_COMPACT_BT * 4
_ADDR_LOCAL_CMP_INDEX_OFFSET = (
    _ADDR_LOCAL_ORI_INDEX_OFFSET + _ADDR_LOCAL_ROWS * _ADDR_LOCAL_ORI_K * 4
)
_ADDR_LOCAL_ORI_LENGTH_OFFSET = (
    _ADDR_LOCAL_CMP_INDEX_OFFSET + _ADDR_LOCAL_ROWS * _ADDR_LOCAL_CMP_K * 4
)
_ADDR_LOCAL_CMP_LENGTH_OFFSET = _ADDR_LOCAL_ORI_LENGTH_OFFSET + 32
_ADDR_LOCAL_INPUT_ELEMS = (_ADDR_LOCAL_CMP_LENGTH_OFFSET + 32) // 4
_ADDR_LOCAL_OUTPUT_ELEMS = (
    _ADDR_LOCAL_ROWS * (_ADDR_LOCAL_ORI_K + _ADDR_LOCAL_CMP_K) // 2
)
_ADDR_BATCH_ROWS = 16
_ADDR_PIPE_BT = 8192
_ADDR_PIPE_ORI_K = 128
_ADDR_PIPE_CMP_K = 512
# 两个 86016 B 输入槽按实际页表宽度紧凑布局；两个 40960 B 输出槽保持完整行。
# 总量 253952 B；小页表保持 16 行，最大双侧页表/最大 K 自动缩至 7 行。
_ADDR_PIPE_INPUT_ELEMS = 21504
_ADDR_BATCH_DEPTH = 2

# Compile-time shape contract.
D = 512  # Logical head dimension: nope[448] followed by rope[64].
N1 = 64  # Q heads
N2 = 1  # KV heads; the query-to-KV head ratio is 64.
TILE_N = 128  # KV tokens per S2 task.

# Per-token physical layout: nope[448], rope[64], then inline BF16 scales.
# FP8 E4M3 uses one scale per 32 values: 512 data bytes + 32 scale bytes.
# FP4 E2M1 uses one scale per 16 values: 256 packed bytes + 64 scale bytes.
# The even FP4 value occupies the low nibble; inputs use uint8 byte views.
KV_ROW_BYTES_ORI = 544
KV_ROW_BYTES_CMP = 320
# Empirical threshold on total dense KV payload, not the hardware L2 capacity.
_KV_NOALLOC_THRESHOLD_BYTES = 32 * 1024 * 1024
# Empirical prefill reuse threshold per batch; cache hints do not affect math.
_KV_L2_PREFILL_MIN_QUERIES = 128
ROW_BF16_ORI = KV_ROW_BYTES_ORI // 2  # ORI row width in BF16 elements.
SCALE_BF16_OFF_ORI = 256  # ORI scale offset in BF16 elements.
SCALE_BF16_OFF_CMP = 128  # CMP scale offset in BF16 elements.

# FD workspace: normalized O slots followed by per-slot max/sum.
# Each core stages at most two partial rows: a leading continuation and a trailing split.
FD_SLOTS_PER_CORE = 2
FD_O_ELEMS = N1 * D  # FP32 output elements per slot.
FD_MS_ELEMS = 2 * N1  # FP32 max[64] and sum[64] per slot.
FD_SLOT_ELEMS = FD_O_ELEMS + FD_MS_ELEMS  # 32896 fp32 = 131584 B
FD_BYTES_PER_CORE = FD_SLOTS_PER_CORE * FD_SLOT_ELEMS * 4  # 263168
WS_BYTES_PER_CORE = FD_BYTES_PER_CORE  # 263168

# FD is selected solely by runtime metadata. Partial O is written directly
# from res_o; reduction storage reuses UB after the global barrier.
_FD_NEG_INF = -1.0e38
_FD_MP_SLOTS = AIC_CORE_MAX_NUM  # One row may span every launched Cube core.

# Dequantized KV is written directly from UB to the shared L1 ring.

# Default ORI offsets; _vec0_body selects the offsets for each pool.
_ROW_BF16 = ROW_BF16_ORI  # Both input pools use the ORI 544-byte UB row stride.
_SCALE_BF16_OFF = SCALE_BF16_OFF_ORI  # ORI scale offset in the shared input layout.

_VL = 128  # BF16 vector width for dequantization.
_TILE_ROWS = 16  # Rows per dequantization block and per NZ face.
# Sentinel beyond every valid query index; prevents reads past the final cu_q span.
_ADDR_M_INF = 1 << 40


TILE = 64
D_TILES = D // TILE  # Eight 64-column chunks: QK reduction / PV output.

# L1 per AIC: KV 384 KiB + Q 112 KiB + P 16 KiB = 512 KiB.
# KV slots remain live through QK and PV; Q slots remain live for a query.
_KV_RING = 3  # KV slots shared by QK and PV.

_Q_RING = 2  # Q[64,384] 双槽，加 Q[64,128] tail 单槽，共 112 KiB.

_P_L1_DEPTH = 1  # Cross-core P slots from AIV store_p to AIC PV.

# Cross-core RingBuffer slots occupy sync ids below 9. The address-table
# publication barrier uses id 9; AIV1's intra-core token is offset by 16.
_AIV1_ID_OFFSET = 16

_KV_NUM_PER_LOOP = 128
_TILE_SIZE = 64
_COMBINE_DIM = _ROW_BF16 * 2

# Rank contracts and hardware limits (not tensor data values).
_TENSOR_RANK_1D = 1
_TENSOR_RANK_2D = 2
_TENSOR_RANK_3D = 3
_TENSOR_RANK_4D = 4
_MAX_PA_BLOCK_SIZE = 1024
# UB/GM paired-copy source stride is encoded in an unsigned 39-bit field.
_DMA_SOURCE_STRIDE_MAX_BYTES = (1 << 39) - 1

_COMPILED_KERNELS = {}
_COMPILED_KERNEL_LOCK = threading.Lock()


def _addr_vec_mode():
    return _ADDR_VEC_OVERRIDE or "off"


def _addr_vec_eligible(ori_kv, ori_bt, ori_idx, cmp_kv, cmp_bt, cmp_idx):
    """Vector addresses require dense physical pages of 64, 128 or 256 rows.

    Mode-specific staging limits are checked by _select_addr_mode.
    """

    def pool_supports_vector_addressing(kv, bt, idx, row_bytes):
        block_size, table_width, index_width = kv.shape[1], bt.shape[1], idx.shape[2]
        if not all(isinstance(n, int) for n in (block_size, table_width, index_width)):
            return False
        return block_size in (64, 128, 256) and kv.stride(0) == block_size * row_bytes

    return pool_supports_vector_addressing(
        ori_kv, ori_bt, ori_idx, KV_ROW_BYTES_ORI
    ) and (
        cmp_kv is None
        or pool_supports_vector_addressing(cmp_kv, cmp_bt, cmp_idx, KV_ROW_BYTES_CMP)
    )


def _addr_tab_w(k):
    """Pad logical sparse columns to a full 128-token tile in either UB mode."""
    return ((k + 127) // 128) * 128


def _addr_pipe_view(slot, begin, count):
    root = reinterpret(slot, layout=make_layout((1, _ADDR_PIPE_INPUT_ELEMS)))
    return root[0:1, begin : begin + count]


def _copy_addr_elements_from_gm(ch, gm, elem_off, count):
    mem_copy(
        reinterpret(ch, layout=make_layout((1, count))),
        gm.view(gm.shape[0] * gm.shape[1])[elem_off : elem_off + count,].view(1, count),
    )


def _copy_addr_from_gm(ch, gm, elem_off, burst_bytes):
    count = burst_bytes // 4
    mem_copy(
        reinterpret(ch, layout=make_layout((1, count))),
        gm.view(gm.shape[0] * gm.shape[1])[elem_off : elem_off + count,].view(1, count),
    )


def _i64(gm, *idx):
    raw = gm[idx] if len(idx) > 1 else gm[idx[0]]
    # Extend int32 GM indices to int64 before physical-address multiplication.
    return dtypes.int64(raw)


class Kvcache:
    """Resolve PA_BBND block-table indices to physical pages and row offsets.

    Block-table values are global physical page indices; batch selects the table.
    Runtime page sizes need not be powers of two.
    """

    @jit
    def pa_blk_off(self, gm_block_table, s2_idx, bo_idx, bs):
        # Common page size avoids runtime division; all other sizes remain valid.
        blk = dtypes.int64(0)
        off = dtypes.int64(0)
        if bs == 128:
            blk = s2_idx // 128
            off = s2_idx % 128
        else:
            blk = s2_idx // bs
            off = s2_idx % bs
        return _i64(gm_block_table, bo_idx, blk), off


class Matmul:
    """Load Q/KV into L1, compute QK and PV, and send results through FIXPIPE.

    Q splits lower/upper heads into two 384-column slots and one 128-column tail.
    Lower/upper KV each use three [64,512] slots, live through QK and PV.
    C1/C2 share four L0A and four L0B slots. QK has one [64,128]
    accumulator; PV has eight independent [64,64] accumulators.
    """

    def __init__(self, tile_cube_m, tile_vec_m, tile_n):
        self.tile_cube_m = tile_cube_m
        self.tile_vec_m = tile_vec_m
        self.tile_n = tile_n
        self.tile_d = D

        self.q_lower_heads_l1 = [
            Buffer(
                MemLoc.L1,
                dtypes.bfloat16,
                make_layout((tile_vec_m, width)),
                data_format="nz",
            )
            for width in (384, 384, 128)
        ]
        self.kv_channels = [
            RingBuffer(
                MemLoc.L1,
                dtypes.bfloat16,
                make_layout((tile_n // 2 * D,)),
                data_format="nd",  # codespell:ignore nd
                depth=_KV_RING,
            )
        ]
        self.p_l1 = RingBuffer(
            MemLoc.L1, dtypes.bfloat16, make_layout((tile_cube_m, tile_n)), depth=1
        )
        self.kv_channels.append(
            RingBuffer(
                MemLoc.L1,
                dtypes.bfloat16,
                make_layout((tile_n // 2 * D,)),
                data_format="nd",  # codespell:ignore nd
                depth=_KV_RING,
            )
        )
        self.q_upper_heads_l1 = [
            Buffer(
                MemLoc.L1,
                dtypes.bfloat16,
                make_layout((tile_vec_m, width)),
                data_format="nz",
            )
            for width in (384, 384, 128)
        ]
        self.l0a = RingBuffer(
            MemLoc.L0A,
            dtypes.bfloat16,
            make_layout((tile_cube_m, 128)),
            depth=4,
            data_format="nz",
            physical_layout=make_layout(
                (8, tile_cube_m // 16, 16, 16), stride=(tile_cube_m * 16, 256, 16, 1)
            ),
        )
        self.l0b = [
            Buffer(
                MemLoc.L0B,
                dtype=dtypes.bfloat16,
                data_format="nz",
                layout=make_layout((tile_n, 64)),
                physical_layout=make_layout(
                    (4, tile_n // 16, 16, 16), stride=(tile_n * 16, 256, 16, 1)
                ),
            )
            for _ in range(4)
        ]
        self.l0c_pv = [
            Buffer(MemLoc.L0C, dtypes.float32, make_layout((tile_cube_m, TILE)))
            for _ in range(8)
        ]
        self.l0c_qk = Buffer(
            MemLoc.L0C, dtypes.float32, make_layout((tile_cube_m, tile_n))
        )

    @jit
    def load_q_wide(self, q_tile_gm, job, load_d_tail):
        q = q_tile_gm.view(self.tile_cube_m, self.tile_d)
        q_lo, q_hi = self.q_lower_heads_l1[0], self.q_upper_heads_l1[0]
        start, width = 0, 384
        if const_expr(load_d_tail):
            q_lo, q_hi = self.q_lower_heads_l1[2], self.q_upper_heads_l1[2]
            start, width = 384, 128
        elif job % 2 != 0:
            q_lo, q_hi = self.q_lower_heads_l1[1], self.q_upper_heads_l1[1]
        mem_copy(
            q_lo,
            q[0 : self.tile_vec_m, start : start + width],
            copy_options(l2_cache_ctl=4),
        )
        mem_copy(
            q_hi,
            q[self.tile_vec_m : self.tile_cube_m, start : start + width],
            copy_options(l2_cache_ctl=4),
        )

    @jit
    def bmm1_fanout_q(self, job, kv_lower_rows_l1, kv_upper_rows_l1):
        set_mmad_direction("m")
        q_lo, q_hi = self.q_lower_heads_l1[0], self.q_upper_heads_l1[0]
        for col in range_constexpr(8):
            a_raw = self.l0a.next_write()
            a_slot = reinterpret(
                a_raw, layout=make_layout((self.tile_cube_m, 64)), data_format="nz"
            )
            if const_expr(col < 6):
                q_col = col
                if job % 2 == 0:
                    q_lo, q_hi = self.q_lower_heads_l1[0], self.q_upper_heads_l1[0]
                else:
                    q_lo, q_hi = self.q_lower_heads_l1[1], self.q_upper_heads_l1[1]
            else:
                q_lo, q_hi = self.q_lower_heads_l1[2], self.q_upper_heads_l1[2]
                q_col = col - 6
            # Slice the full L0 tile so both halves retain its physical pitch.
            mem_copy(
                tile_slice(a_slot, (self.tile_vec_m, 64), (0, 0)),
                tile_slice(q_lo, (self.tile_vec_m, 64), (0, q_col)),
            )
            mem_copy(
                tile_slice(a_slot, (self.tile_vec_m, 64), (1, 0)),
                tile_slice(q_hi, (self.tile_vec_m, 64), (0, q_col)),
            )
            b_slot = reinterpret(
                self.l0b[col % 4],
                layout=make_layout((self.tile_n, 64)),
                data_format="nz",
            )
            mem_copy(
                tile_slice(b_slot, (self.tile_n // 2, 64), (0, 0)),
                tile_slice(
                    reinterpret(
                        kv_lower_rows_l1,
                        layout=make_layout((self.tile_n // 2, D)),
                        data_format="nz",
                    ),
                    (self.tile_n // 2, 64),
                    (0, col),
                ),
            )
            mem_copy(
                tile_slice(b_slot, (self.tile_n // 2, 64), (1, 0)),
                tile_slice(
                    reinterpret(
                        kv_upper_rows_l1,
                        layout=make_layout((self.tile_n // 2, D)),
                        data_format="nz",
                    ),
                    (self.tile_n // 2, 64),
                    (0, col),
                ),
            )
            matmul(
                self.l0c_qk,
                a_slot,
                b_slot,
                init=(col == 0),
                unit_flag=3 if col == 7 else 2,
            )

    def store_s(self, buf_score):
        mem_copy(
            buf_score, self.l0c_qk.partition(dim=0), options=copy_options(unit_flag=3)
        )

    @jit
    def compute_pv_fanout(
        self, p_l1, results, kv_lower_rows_l1, kv_upper_rows_l1, is_first
    ):
        set_mmad_direction("n")
        a_slot = self.l0a.next_write()
        for col in range_constexpr(8):
            b_slot = reinterpret(
                self.l0b[col % 4],
                layout=make_layout((64, self.tile_n)),
                data_format="nz",
            )
            mem_copy(
                reinterpret(
                    b_slot,
                    layout=make_layout((64, self.tile_n // 2)),
                    offset=0,
                    data_format="nz",
                ),
                tile_slice(
                    reinterpret(
                        kv_lower_rows_l1,
                        layout=make_layout((self.tile_n // 2, D)),
                        data_format="nz",
                    ),
                    (self.tile_n // 2, 64),
                    (0, col),
                ).permute(1, 0),
            )
            mem_copy(
                reinterpret(
                    b_slot,
                    layout=make_layout((64, self.tile_n // 2)),
                    offset=64 * (self.tile_n // 2) * 2,
                    data_format="nz",
                ),
                tile_slice(
                    reinterpret(
                        kv_upper_rows_l1,
                        layout=make_layout((self.tile_n // 2, D)),
                        data_format="nz",
                    ),
                    (self.tile_n // 2, 64),
                    (0, col),
                ).permute(1, 0),
            )
            if const_expr(col == 0):
                mem_copy(a_slot, p_l1)
            matmul(self.l0c_pv[col], a_slot, b_slot, init=True, unit_flag=0)
            acc_target, pv_target = results[(col // 4) * 2], results[(col // 4) * 2 + 1]
            if is_first == 1:
                mem_copy(
                    tile_slice(acc_target, (self.tile_vec_m, 64), (0, col % 4)),
                    self.l0c_pv[col].partition(dim=0),
                    options=copy_options(unit_flag=0),
                )
            else:
                mem_copy(
                    tile_slice(pv_target, (self.tile_vec_m, 64), (0, col % 4)),
                    self.l0c_pv[col].partition(dim=0),
                    options=copy_options(unit_flag=0),
                )


class Vector:
    """Compute online softmax and output accumulation for 32 heads per AIV.

    For old state (m,l,o) and scaled scores s:
      m_new = max(m, max(s)); alpha = exp(m-m_new); p = exp(s-m_new).
      l_new = alpha*l + sum(p); o_new = alpha*o + BF16(p) @ KV.
    Only valid columns contribute to max, exponent/P and denominator.
    Sinks seed (m=sink,l=1,o=0) without a value contribution. Final output is
    BF16(o/l), with LSE=log(l)+m. A zero sink still contributes exp(0)=1.
    Double-buffer max/sum by nonempty query and alpha by tick. The two output
    halves serve consecutive queries, with copy completion before reuse.
    """

    def __init__(self, tile_vec_m, tile_n, tile_d, subblock_idx, sinks_span):
        self.sinks_span = sinks_span
        self.tile_vec_m = tile_vec_m
        self.tile_n = tile_n
        self.tile_d = tile_d
        self.subblock_idx = subblock_idx

        # Double-buffer max/sum by query and exp by task tick.
        self.sm_max_tb = [
            Buffer(MemLoc.UB, dtypes.float32, make_layout((tile_vec_m, 2)))
            for _ in range(2)
        ]
        self.sm_sum_tb = [
            Buffer(MemLoc.UB, dtypes.float32, make_layout((tile_vec_m, 2)))
            for _ in range(2)
        ]
        self.sm_exp_tb = [
            Buffer(MemLoc.UB, dtypes.float32, make_layout((tile_vec_m, 2)))
            for _ in range(2)
        ]
        # Partial weights are needed only after the main-loop rewind.
        self.fd_mp_tb = []

        self.lse_ub = Buffer(MemLoc.UB, dtypes.float32, make_layout((tile_vec_m, 1)))
        # Main-loop output views are bound independently for each 256-column half.

        self.sinks_ub = Buffer(
            MemLoc.UB, dtypes.float32, make_layout((1, self.sinks_span))
        )

    def bind_result(self, results):
        self.result_views = results
        self.out_halves = [
            reinterpret(
                results[half * 2 + 1],
                dtypes.bfloat16,
                layout=make_layout((self.tile_vec_m, 256), stride=(512, 1)),
            )
            for half in range(2)
        ]

    @jit
    def softmax_rest_full(
        self,
        job,
        tile,
        scale,
        first,
        all_valid,
        buf_score,
        buf_p,
        valid_tile_kv_rows,
        has_sink,
    ):
        sm_max = self.sm_max_tb[job % 2]
        sm_sum = self.sm_sum_tb[job % 2]
        sm_exp = self.sm_exp_tb[tile % 2]
        # VF softmax: reduce QK scores, update online state and pack BF16 P in NZ order.
        with vf(mode="simd"):
            mask = rr.full_mask(elem_bits=32)
            mask16 = rr.full_mask(elem_bits=16)
            # Keep two FP32 copies per head for the interleaved online softmax state.
            two_lanes, _ = rr.update_mask(2, elem_bits=32)
            softmax_state_offset = 0
            if const_expr(first and has_sink):
                sink = rr.vload(self.sinks_ub)
                neg_inf = rr.veqs(
                    rr.vreinterpret(sink, dtypes.uint32), 0xFF800000, mask=mask
                )
                pos_inf = rr.veqs(
                    rr.vreinterpret(sink, dtypes.uint32), 0x7F800000, mask=mask
                )
                unscaled_sink_reg = rr.vmuls(sink, 1.0 / scale, mask=mask)
                unscaled_sink_reg = rr.vdups(
                    (-3.4028234663852886e38),
                    dtypes.float32,
                    mask=neg_inf,
                    mode="merging",
                    merge=unscaled_sink_reg,
                )
                unscaled_sink_reg = rr.vdups(
                    -(-3.4028234663852886e38),
                    dtypes.float32,
                    mask=pos_inf,
                    mode="merging",
                    merge=unscaled_sink_reg,
                )
                replicated, unused = rr.vinterleave(
                    unscaled_sink_reg, unscaled_sink_reg
                )
                rr.vstore(sm_max, softmax_state_offset, replicated, mask)
                initial_sum = rr.vdups(1.0, dtypes.float32)
                initial_sum = rr.vdups(
                    0.0, dtypes.float32, mask=neg_inf, mode="merging", merge=initial_sum
                )
                replicated, unused = rr.vinterleave(initial_sum, initial_sum)
                rr.vstore(sm_sum, softmax_state_offset, replicated, mask)
                # Order these state stores before the following state loads.
                rr.vmem_bar("vst_vld")
            if const_expr(not first or has_sink):
                old_max = rr.vload(sm_max, softmax_state_offset)
                rr.vmem_bar("vld_vst")
                old_sum = rr.vload(sm_sum, softmax_state_offset)
                old_max = rr.vmuls(old_max, scale, mask=mask)

            neg_inf_reg = rr.vreinterpret(
                rr.vdups(0xFF800000, dtypes.uint32), dtypes.float32
            )
            if const_expr(not all_valid):
                even_index_reg = rr.vmuls(rr.varange(0, dtypes.int32), 2, mask=mask)
                odd_index_reg = rr.vadds(even_index_reg, 1, mask=mask)
                invalid_even = rr.vges(
                    even_index_reg, dtypes.int32(valid_tile_kv_rows), mask=mask
                )
                invalid_odd = rr.vges(
                    odd_index_reg, dtypes.int32(valid_tile_kv_rows), mask=mask
                )
            max_cursor = rr.vstore_unalign_begin(sm_max)
            # Explicit int32 loop values select the VF scalar ABI.
            for query_head_row in cannbotdsl.range(
                dtypes.int32(0),
                dtypes.int32(self.tile_vec_m),
                dtypes.int32(1),
                unroll=1,
            ):
                score_even_reg, score_odd_reg = rr.vload_deinterleave(
                    buf_score, query_head_row * TILE_N
                )
                if const_expr(not all_valid):
                    score_even_reg = rr.vselect(
                        neg_inf_reg, score_even_reg, cond_mask=invalid_even
                    )
                    score_odd_reg = rr.vselect(
                        neg_inf_reg, score_odd_reg, cond_mask=invalid_odd
                    )
                row_max_reg = rr.vmax(
                    rr.vreduce_max(score_even_reg, mask=mask),
                    rr.vreduce_max(score_odd_reg, mask=mask),
                    mask=mask,
                )
                row_max_reg = rr.vdup(row_max_reg, mask=mask)
                if const_expr(first and not has_sink):
                    prior_max_reg = rr.vdups((-3.4028234663852886e38), dtypes.float32)
                else:
                    prior_max_reg = rr.vload_broadcast(
                        sm_max, softmax_state_offset + 2 * query_head_row
                    )
                row_max_reg = rr.vmax(row_max_reg, prior_max_reg, mask=mask)
                squeezed = rr.vsqueeze_and_storeunalign_init(
                    row_max_reg, mask=two_lanes
                )
                rr.vsqueeze_and_storeunalign(
                    sm_max, softmax_state_offset, squeezed, max_cursor
                )
            rr.vsqueeze_and_storeunalign_finalize(
                sm_max, softmax_state_offset, max_cursor
            )
            rr.vmem_bar("vst_vld")
            if const_expr(not first or has_sink):
                row_max_reg = rr.vmuls(
                    rr.vload(sm_max, softmax_state_offset),
                    scale,
                    mask=mask,
                )
                rescale_factor_reg = rr.vexp_sub(old_max, row_max_reg, mask=mask)
                rr.vstore(
                    sm_exp,
                    0,
                    rescale_factor_reg,
                    mask,
                )
            sum_cursor = rr.vstore_unalign_begin(sm_sum)
            for query_head_row in cannbotdsl.range(
                dtypes.int32(0),
                dtypes.int32(self.tile_vec_m),
                dtypes.int32(1),
                unroll=1,
            ):
                score_even_reg, score_odd_reg = rr.vload_deinterleave(
                    buf_score, query_head_row * TILE_N
                )
                if const_expr(not all_valid):
                    score_even_reg = rr.vselect(
                        neg_inf_reg, score_even_reg, cond_mask=invalid_even
                    )
                    score_odd_reg = rr.vselect(
                        neg_inf_reg, score_odd_reg, cond_mask=invalid_odd
                    )
                row_max_reg = rr.vmuls(
                    rr.vload_broadcast(
                        sm_max, softmax_state_offset + 2 * query_head_row
                    ),
                    scale,
                    mask=mask,
                )
                exp_even_reg = rr.vexp_sub(
                    rr.vmuls(score_even_reg, scale, mask=mask), row_max_reg, mask=mask
                )
                exp_odd_reg = rr.vexp_sub(
                    rr.vmuls(score_odd_reg, scale, mask=mask), row_max_reg, mask=mask
                )
                row_sum_reg = rr.vadd(
                    rr.vreduce_sum(exp_even_reg, mask=mask),
                    rr.vreduce_sum(exp_odd_reg, mask=mask),
                    mask=mask,
                )
                row_sum_reg = rr.vdup(row_sum_reg, mask=mask)
                squeezed = rr.vsqueeze_and_storeunalign_init(
                    row_sum_reg, mask=two_lanes
                )
                rr.vsqueeze_and_storeunalign(
                    sm_sum, softmax_state_offset, squeezed, sum_cursor
                )
                p_even_reg = rr.vcast(
                    exp_even_reg,
                    dtypes.bfloat16,
                    mask=mask,
                    rounding="rn",
                    reg_layout=rr.RegLayout.ZERO,
                )
                p_odd_reg = rr.vcast(
                    exp_odd_reg,
                    dtypes.bfloat16,
                    mask=mask,
                    rounding="rn",
                    reg_layout=rr.RegLayout.ONE,
                )
                packed_p_reg = rr.vadd(p_even_reg, p_odd_reg, mask=mask16)
                rr.vstore_strided(
                    buf_p,
                    query_head_row * _NZ_ROW,
                    packed_p_reg,
                    mask16,
                    block_stride=self.tile_vec_m + 1,
                    repeat_stride=0,
                )
            rr.vsqueeze_and_storeunalign_finalize(
                sm_sum, softmax_state_offset, sum_cursor
            )
            if const_expr(not first or has_sink):
                rr.vmem_bar("vst_vld")
                row_sum_reg = rr.vmadd(
                    old_sum,
                    rescale_factor_reg,
                    rr.vload(sm_sum, softmax_state_offset),
                    mask=mask,
                )
                rr.vstore(sm_sum, softmax_state_offset, row_sum_reg, mask)

    @jit
    def finalize_softmax_state(self, sum_idx: int, scale):
        # Compact duplicated head state; retain scaled max/sum for FD and LSE.
        sm_max = self.sm_max_tb[sum_idx]
        sm_sum = self.sm_sum_tb[sum_idx]
        with vf(mode="simd"):
            mask, _ = rr.update_mask(self.tile_vec_m, elem_bits=32)
            maximum = rr.vload(sm_max, 0)
            total = rr.vload(sm_sum, 0)
            maximum, unused = rr.vdeinterleave(maximum, maximum)
            total, unused = rr.vdeinterleave(total, total)
            rr.vstore(sm_max, 0, rr.vmuls(maximum, scale, mask=mask), mask)
            rr.vstore(sm_sum, 0, total, mask)
            rr.vmem_bar("vst_vld")

    @jit
    def update_o(
        self, half, first, last, exp_idx: int, sum_idx: int, pack_output=False
    ):
        sm_exp = self.sm_exp_tb[exp_idx]
        sm_sum = self.sm_sum_tb[sum_idx]
        accumulator = self.result_views[half * 2]
        partial = self.result_views[half * 2 + 1]
        out = self.out_halves[half]
        with vf(mode="simd"):
            mask = rr.full_mask(elem_bits=32)
            for row in range(self.tile_vec_m):
                alpha = rr.vdups(1.0, dtypes.float32)
                denorm = rr.vdups(1.0, dtypes.float32)
                if const_expr(not first):
                    alpha = rr.vload_broadcast(sm_exp, row * 2)
                if const_expr(last):
                    denorm = rr.vdiv(
                        rr.vdups(1.0, dtypes.float32),
                        rr.vload_broadcast(sm_sum, row),
                        mask=mask,
                    )
                for col in range_constexpr(4):
                    offset = row * 256 + col * 64
                    if const_expr(first):
                        value = rr.vload(accumulator, offset)
                    else:
                        value = rr.vload(partial, offset)
                        old = rr.vmul(rr.vload(accumulator, offset), alpha, mask=mask)
                        value = rr.vadd(old, value, mask=mask)
                    if const_expr(last):
                        value = rr.vmul(value, denorm, mask=mask)
                    if const_expr(pack_output):
                        packed = rr.vcast(
                            value, dtypes.bfloat16, mask=mask, rounding="rn"
                        )
                        rr.vstore_pack(
                            out,
                            row * 512 + col * 64,
                            packed,
                            mask,
                            pack_mode=rr.PackMode.B32_TO_B16,
                        )
                    else:
                        rr.vstore(accumulator, offset, value, mask)

    def finalize_o(self, o_tile_gm, half, sum_idx: int, lse_gm=None, lse_base=None):
        # Copy each completed half immediately. The output RingBuffer owns the
        # V-to-MTE3 dependency and protects this alias from the next PV write.
        if const_expr(half == 0 and lse_gm is not None):
            self._finalize_lse_vf(lse_gm, lse_base, sum_idx)
        heads = tile_slice(
            o_tile_gm, (self.tile_vec_m, self.tile_d), (self.subblock_idx, 0)
        )
        mem_copy(
            tile_slice(heads, (self.tile_vec_m, 256), (0, half)),
            self.out_halves[half],
            options=copy_options(l2_cache_ctl=4),
        )

    @jit
    def finalize_empty(self, o_tile_gm, sinks_gm, lse_gm=None, lse_base=None):
        # Empty rows produce O=0 and LSE=sink through a separate zero buffer.
        half = tile_slice(
            o_tile_gm, (self.tile_vec_m, self.tile_d), (self.subblock_idx, 0)
        )
        empty_out = Buffer(MemLoc.UB, dtypes.bfloat16, make_layout((1, self.tile_d)))
        with vf(mode="simd"):
            full, _ = rr.update_mask(128, elem_bits=16)
            zero = rr.vdups(0.0, dtypes.bfloat16, mask=full)
            for col in range(0, self.tile_d, 128):
                rr.vstore(empty_out, col, zero, full)
        for row in range(self.tile_vec_m):
            mem_copy(tile_slice(half, (1, self.tile_d), (row, 0)), empty_out)
        if const_expr(lse_gm is not None):
            # Keep lse_ub's producer on V, as in _finalize_lse_vf; using
            # MTE2 here would change the shared buffer's event protocol.
            mem_copy(
                self.sinks_ub,
                tile_slice(sinks_gm, (1, self.sinks_span), (0, self.subblock_idx)),
            )
            with vf(mode="simd"):
                mask, _ = rr.update_mask(self.tile_vec_m, elem_bits=32)
                rr.vstore(self.lse_ub, 0, rr.vload(self.sinks_ub, 0), mask)
            mem_copy(
                tile_slice(lse_gm, (self.tile_vec_m,), (lse_base,)),
                reinterpret(self.lse_ub, layout=make_layout((self.tile_vec_m,))),
            )

    # Keep each FD VF in a separate JIT method: register/mask locals must not
    # become loop-carried scalar values in a surrounding runtime loop.

    @jit
    def stage_partial(self, fd_o, fd_ms, slot, sum_idx: int):
        # Stage the normalized FP32 partial O and its max/sum for this AIV.
        # The caller normalizes O with update_o before staging.
        # O is staged as two 32 KiB halves; max/sum reuse the 32-FP32 lse buffer.
        # Workspace layout: O[slots*64,512], followed by max/sum[slots*128].
        # Each statistics slot contains max[0:64] followed by sum[0:64].
        sub = self.subblock_idx
        for half in range_constexpr(2):
            mem_copy(
                tile_slice(
                    fd_o,
                    (self.tile_vec_m, 256),
                    (slot * (N1 // self.tile_vec_m) + sub, half),
                ),
                self.result_views[half * 2],
            )
        self._fd_ms_to_ub(0, sum_idx)
        mem_copy(
            tile_slice(
                fd_ms,
                (self.tile_vec_m,),
                (slot * (FD_MS_ELEMS // self.tile_vec_m) + sub,),
            ),
            reinterpret(self.lse_ub, layout=make_layout((self.tile_vec_m,))),
        )
        self._fd_ms_to_ub(1, sum_idx)
        mem_copy(
            tile_slice(
                fd_ms,
                (self.tile_vec_m,),
                (
                    slot * (FD_MS_ELEMS // self.tile_vec_m)
                    + (N1 // self.tile_vec_m)
                    + sub,
                ),
            ),
            reinterpret(self.lse_ub, layout=make_layout((self.tile_vec_m,))),
        )

    @jit
    def fd_reduce(self, fd_o, fd_ms, fd_ub, out_gm, lse_gm, row, slot0, k, h0):
        # Reduce k normalized partials for the 32 heads beginning at h0.
        # M = max(m_p); t_p = exp(m_p-M)*s_p; G = sum(t_p).
        # O = sum((t_p/G)*O_p); LSE = M + log(G).
        # Accumulate in metadata slot order for deterministic reassociation.
        # After the barrier, reuse main-loop UB for max/sum and the output accumulator.
        # weights remain in registers; res_o accumulates, fd_ub receives O parts.
        sub = h0 // self.tile_vec_m
        self._fd_init_acc()
        ms_offset = slot0 * FD_MS_ELEMS
        ms_count = k * FD_MS_ELEMS
        ms_source = fd_ms[ms_offset : ms_offset + ms_count,].view(k, FD_MS_ELEMS)
        mem_copy(
            self.fd_ms_ub.next_write(),
            tile_slice(ms_source, (_FD_MP_SLOTS, FD_MS_ELEMS), (0, 0)),
        )
        ms_buf = self.fd_ms_ub.next_read()
        for p in range(k):
            self._fd_acc_max_save(p, ms_buf)
        for p in range(k):
            self._fd_acc_sum(p, ms_buf)
        # Optionally store LSE = M + log(G).
        if const_expr(lse_gm is not None):
            self._fd_lse_to_ub(self.lse_ub.next_write())
            mem_copy(
                tile_slice(lse_gm, (self.tile_vec_m,), (row * 2 + sub,)),
                reinterpret(
                    self.lse_ub.next_read(), layout=make_layout((self.tile_vec_m,))
                ),
            )
        # Accumulate normalized partials using cached weights t_p/G.
        mem_copy(
            fd_ub.next_write(),
            tile_slice(
                fd_o,
                (self.tile_vec_m, self.tile_d),
                (slot0 * (N1 // self.tile_vec_m) + sub, 0),
            ),
        )
        self._fd_acc_full(fd_ub.next_read(), 0, True)
        for p in range(1, k):
            mem_copy(
                fd_ub.next_write(),
                tile_slice(
                    fd_o,
                    (self.tile_vec_m, self.tile_d),
                    ((slot0 + p) * (N1 // self.tile_vec_m) + sub, 0),
                ),
            )
            self._fd_acc_full(fd_ub.next_read(), p, False)
        # Cast the normalized result in place and write its 32-head tile.
        # The pre-reduction barrier drained old MTE3 operations; the following
        # V-to-MTE3 dependency still protects this newly computed output.
        self._cast_output()
        mem_copy(
            tile_slice(
                out_gm,
                (self.tile_vec_m, self.tile_d),
                (row * (N1 // self.tile_vec_m) + sub, 0),
            ),
            self.out_view,
        )

    @jit
    def _finalize_lse_vf(self, lse_gm, lse_base, sum_idx: int):
        # LSE = log(sum) + max; the framework tracks Buffer V-to-MTE3 accesses.
        sm_max_buf = self.sm_max_tb[sum_idx]
        sm_sum_buf = self.sm_sum_tb[sum_idx]
        with vf(mode="simd"):
            mask, _ = rr.update_mask(self.tile_vec_m, elem_bits=32)
            total = rr.vload(sm_sum_buf, 0)
            maximum = rr.vload(sm_max_buf, 0)
            rr.vstore(
                self.lse_ub,
                0,
                rr.vadd(rr.vlog(total, mask=mask), maximum, mask=mask),
                mask,
            )
        mem_copy(
            tile_slice(lse_gm, (self.tile_vec_m,), (lse_base,)),
            reinterpret(self.lse_ub, layout=make_layout((self.tile_vec_m,))),
        )

    @jit
    def _cast_output(self):
        # Cast FP32 res_o in place to the first 32 KiB BF16 output view.
        # Each 64-element read precedes its packed write; that write ends before
        # the next unread FP32 chunk, preserving forward traversal safety.
        with vf(mode="simd"):
            full = rr.full_mask()
            for offset in range(0, self.tile_vec_m * self.tile_d, 64):
                value = rr.vload(self.res_o, offset)
                result = rr.vcast(value, dtypes.bfloat16, mask=full)
                rr.vstore_pack(
                    self.out_view,
                    offset,
                    result,
                    full,
                    pack_mode=rr.PackMode.B32_TO_B16,
                )

    @jit
    def _fd_ms_to_ub(self, which: int, sum_idx: int):
        # Copy max (which=0) or sum (which=1) to the LSE staging buffer.
        if const_expr(which == 0):
            buf = self.sm_max_tb[sum_idx]
        else:
            buf = self.sm_sum_tb[sum_idx]
        with vf(mode="simd"):
            rowmask, _ = rr.update_mask(self.tile_vec_m, elem_bits=32)
            rr.vstore(self.lse_ub, 0, rr.vload(buf, 0), rowmask)

    @jit
    def _fd_init_acc(self):
        with vf(mode="simd"):
            rowmask, _ = rr.update_mask(self.tile_vec_m, elem_bits=32)
            rr.vstore(
                self.sm_max_tb[0],
                0,
                rr.vdups(_FD_NEG_INF, dtypes.float32, mask=rowmask),
                rowmask,
            )
            rr.vstore(
                self.sm_sum_tb[0],
                0,
                rr.vdups(0.0, dtypes.float32, mask=rowmask),
                rowmask,
            )
            rr.vmem_bar("vst_vld")

    @jit
    def _fd_acc_max_save(self, p: int, lse_buf):
        # Cache m_p and accumulate the global maximum; the next pass reuses
        # the same weight slot for t_p = s_p * exp(m_p - M).
        mp_buf = tile_slice(self.fd_mp_tb, (self.tile_vec_m, 1), (p, 0))
        with vf(mode="simd"):
            rowmask, _ = rr.update_mask(self.tile_vec_m, elem_bits=32)
            mp = rr.vload(
                lse_buf, p * FD_MS_ELEMS + self.subblock_idx * self.tile_vec_m + 0
            )
            nm = rr.vmax(rr.vload(self.sm_max_tb[0], 0), mp, mask=rowmask)
            rr.vstore(self.sm_max_tb[0], 0, nm, rowmask)
            rr.vstore(mp_buf, 0, mp, rowmask)
            rr.vmem_bar("vst_vld")

    @jit
    def _fd_acc_sum(self, p: int, lse_buf):
        # Bulk max/sum landing: t_p = s_p·exp(m_p−M); G += t_p. Cache t_p.
        # Reuse cached m_p for t_p, then form t_p/G during output accumulation.
        mp_buf = tile_slice(self.fd_mp_tb, (self.tile_vec_m, 1), (p, 0))
        with vf(mode="simd"):
            rowmask, _ = rr.update_mask(self.tile_vec_m, elem_bits=32)
            e = rr.vexp_sub(
                rr.vload(mp_buf, 0), rr.vload(self.sm_max_tb[0], 0), mask=rowmask
            )
            t = rr.vmul(
                rr.vload(
                    lse_buf, p * FD_MS_ELEMS + self.subblock_idx * self.tile_vec_m + 64
                ),
                e,
                mask=rowmask,
            )
            acc = rr.vadd(rr.vload(self.sm_sum_tb[0], 0), t, mask=rowmask)
            rr.vstore(self.sm_sum_tb[0], 0, acc, rowmask)
            rr.vstore(mp_buf, 0, t, rowmask)
            rr.vmem_bar("vst_vld")

    @jit
    def _fd_lse_to_ub(self, lse_buf):
        # lse_ub ← M + log(G)
        with vf(mode="simd"):
            rowmask, _ = rr.update_mask(self.tile_vec_m, elem_bits=32)
            lse = rr.vadd(
                rr.vlog(rr.vload(self.sm_sum_tb[0], 0), mask=rowmask),
                rr.vload(self.sm_max_tb[0], 0),
                mask=rowmask,
            )
            rr.vstore(lse_buf, 0, lse, rowmask)

    @jit
    def _fd_acc_full(self, fd_ub, p: int, first: cannbotdsl.Constexpr[bool]):
        # Compute each head weight in registers while consuming partial O.
        # Part zero initializes the accumulator; later parts retain slot order.
        mp_buf = tile_slice(self.fd_mp_tb, (self.tile_vec_m, 1), (p, 0))
        with vf(mode="simd"):
            full = rr.full_mask()
            for r in range(self.tile_vec_m):
                wb = rr.vdiv(
                    rr.vload_broadcast(mp_buf, r),
                    rr.vload_broadcast(self.sm_sum_tb[0], r),
                    mask=full,
                )
                for col in range(0, self.tile_d, 2048 // 32):
                    off = r * self.tile_d + col
                    cur = rr.vload(fd_ub, off)
                    if const_expr(first):
                        value = rr.vmul(cur, wb, mask=full)
                    else:
                        pre = rr.vload(self.res_o, off)
                        value = rr.vmadd(cur, wb, pre, mask=full)
                    rr.vstore(self.res_o, off, value, full)
            rr.vmem_bar("vst_vld")

    def _setup_fd_reduce_buffers(self):
        # After the global barrier, reuse main-loop UB for one full partial O,
        # an FP32 accumulator, all max/sum rows and up to 36 partial weights.
        # Main-loop allocations and reduction allocations are not simultaneous.
        self.fd_ub = RingBuffer(
            MemLoc.UB,
            dtypes.float32,
            make_layout((self.tile_vec_m, self.tile_d)),
            depth=1,
        )
        self.fd_ms_ub = RingBuffer(
            MemLoc.UB, dtypes.float32, make_layout((_FD_MP_SLOTS, FD_MS_ELEMS)), depth=1
        )
        self.res_o = Buffer(
            MemLoc.UB, dtypes.float32, make_layout((self.tile_vec_m, self.tile_d))
        )
        self.out_view = reinterpret(
            self.res_o,
            dtypes.bfloat16,
            layout=make_layout((self.tile_vec_m, self.tile_d)),
        )
        self.lse_ub = RingBuffer(
            MemLoc.UB, dtypes.float32, make_layout((self.tile_vec_m, 1)), depth=1
        )
        self.sm_max_tb[0] = Buffer(
            MemLoc.UB, dtypes.float32, make_layout((self.tile_vec_m, 1))
        )
        self.sm_sum_tb[0] = Buffer(
            MemLoc.UB, dtypes.float32, make_layout((self.tile_vec_m, 1))
        )
        self.tmp_new_max = Buffer(
            MemLoc.UB, dtypes.float32, make_layout((self.tile_vec_m, 1))
        )
        self.fd_mp_tb = Buffer(
            MemLoc.UB, dtypes.float32, make_layout((_FD_MP_SLOTS * self.tile_vec_m, 1))
        )


@datastruct
class PipelineRingState:
    clock: CArray[dtypes.int64, 1]
    metadata: CArray[dtypes.int64, 64]


class SlotMajorDelayLineGroup:
    def __init__(self):
        self.state = PipelineRingState(clock=[0], metadata=[0] * 64)
        self.fields = ("m", "n", "t", "last", "first", "sum_slot", "valid", "ori_tiles")

    def push(self, **kwargs):
        metadata_slot_offset = self.state.clock[0] % 8 * 8
        for i, field in enumerate(self.fields):
            self.state.metadata[metadata_slot_offset + i] = dtypes.int64(kwargs[field])

    def tap(self, lag):
        metadata_slot_offset = (self.state.clock[0] + 8 - lag) % 8 * 8
        return SimpleNamespace(
            **{
                field: self.state.metadata[metadata_slot_offset + i]
                for i, field in enumerate(self.fields)
            }
        )

    def advance(self):
        self.state.clock[0] = (self.state.clock[0] + 1) % 8


class MqsmlaKernel:
    """Device attention pipeline; tensor views and launch live in _run_mqsmla."""

    def __init__(self, tile_cube_m, tile_vec_m, tile_n, has_cmp=False):
        self.tile_cube_m = tile_cube_m
        self.tile_vec_m = tile_vec_m
        self.tile_n = tile_n
        self.tile_d = D
        self.has_cmp = has_cmp
        # Cross-core chains: QK (AIC to AIV, depth 1), P (AIV to AIC, depth 1),
        # and PV (AIC to AIV, depth 1). UB shapes cover one AIV; P covers both halves.

        self.block_idx = get_block_idx()
        self.subblock_idx = get_subblock_id()

        self.sinks_span = tile_vec_m

        self.matmul = Matmul(tile_cube_m, tile_vec_m, tile_n)
        self.p_l1 = self.matmul.p_l1
        self.kvcache = Kvcache()

    @jit
    def __call__(
        self,
        out_gm: Tensor,
        q_gm: Tensor,
        sinks_gm: Tensor,
        pa_phys_gm: Tensor = None,
        pa_bt_gm: Tensor = None,
        ws_gm: Tensor = None,
        pa_phys_cmp_gm: Tensor = None,
        pa_bt_cmp_gm: Tensor = None,
        cmp_idx_gm: Tensor = None,
        ori_idx_gm: Tensor = None,
        cu_seqlens_q: Tensor = None,
        softmax_lse: Tensor = None,
        ori_topk_length: Tensor = None,
        cmp_topk_length: Tensor = None,
        metadata_gm: Tensor = None,
        softmax_scale: dtypes.float32 = 1.0,
        batch_consistency: dtypes.int32 = 0,
        addr_ws_gm: Tensor = None,
    ):
        # Workspace capacity determines the launch grid and FD slot count.
        self._b_rt = cu_seqlens_q.shape[1] - 1
        self._scale = softmax_scale
        self._batch_consistency = batch_consistency
        self._ori_page_stride = pa_phys_gm.stride[0]
        self._pa_bs_rt = pa_phys_gm.shape[1]
        if const_expr(self.has_cmp):
            self._cmp_page_stride = pa_phys_cmp_gm.stride[0]
            self._pa_cmp_bs_rt = pa_phys_cmp_gm.shape[1]
        self._ws = ws_gm
        self._q = q_gm
        self._sinks = sinks_gm
        self._out = out_gm
        self._lse = softmax_lse
        self._ori_topk_length = ori_topk_length
        self._cmp_topk_length = cmp_topk_length

        self._cu_q = cu_seqlens_q

        self._pa_phys, self._pa_bt = pa_phys_gm, pa_bt_gm
        self._ori_idx = ori_idx_gm
        if const_expr(self.has_cmp):
            self._pa_phys_cmp, self._pa_bt_cmp = pa_phys_cmp_gm, pa_bt_cmp_gm
            self._cmp_idx = cmp_idx_gm
        # Address-prepass storage exists only in the vector-addressing template.
        _avec = _addr_vec_mode()
        if const_expr(_avec != "off"):
            self._avec_bs_ori = pa_phys_gm.shape[1]
            self._avec_k1 = ori_idx_gm.shape[1]
            self._avec_k1_tab = _addr_tab_w(self._avec_k1)
            self._avec_k2_tab = 0
            if const_expr(self.has_cmp):
                self._avec_bs_cmp = pa_phys_cmp_gm.shape[1]
                self._avec_k2 = cmp_idx_gm.shape[1]
                self._avec_k2_tab = _addr_tab_w(self._avec_k2)
            if const_expr(_avec == "local"):
                # 两侧页表与四行输入共 86080 B；本 AIV 的地址输出独占 10240 B。
                self._addr_local_input = Buffer(
                    MemLoc.UB, dtypes.int32, make_layout((1, _ADDR_LOCAL_INPUT_ELEMS))
                )
                self._addr_local_output = Buffer(
                    MemLoc.UB, dtypes.int32, make_layout((1, _ADDR_LOCAL_OUTPUT_ELEMS))
                )
            if const_expr(_addr_vec_mode() == "pipe"):
                # DMA 分区按 32 B 对齐；48 个元素覆盖两侧索引 padding 与两组长度。
                self._addr_pipe_bt1 = max(64, ((self._pa_bt.shape[1] + 7) // 8) * 8)
                self._addr_pipe_bt2 = 0
                pipe_k2 = 0
                if const_expr(self.has_cmp):
                    self._addr_pipe_bt2 = max(
                        64, ((self._pa_bt_cmp.shape[1] + 7) // 8) * 8
                    )
                    pipe_k2 = self._avec_k2
                self._addr_pipe_ori_index = self._addr_pipe_bt1 + self._addr_pipe_bt2
                self._addr_pipe_rows = min(
                    _ADDR_BATCH_ROWS,
                    (_ADDR_PIPE_INPUT_ELEMS - self._addr_pipe_ori_index - 48)
                    // (_ADDR_PIPE_ORI_K + _ADDR_PIPE_CMP_K),
                )
                self._addr_pipe_cmp_index = (
                    self._addr_pipe_ori_index
                    + ((self._addr_pipe_rows * self._avec_k1 + 7) // 8) * 8
                )
                self._addr_pipe_ori_length = (
                    self._addr_pipe_cmp_index
                    + ((self._addr_pipe_rows * pipe_k2 + 7) // 8) * 8
                )
                self._addr_pipe_cmp_length = self._addr_pipe_ori_length + 16
                self._addr_input_ring = RingBuffer(
                    MemLoc.UB,
                    dtypes.int32,
                    make_layout((1, _ADDR_PIPE_INPUT_ELEMS)),
                    depth=_ADDR_BATCH_DEPTH,
                )
                self._addr_out_ring = RingBuffer(
                    MemLoc.UB,
                    dtypes.int32,
                    make_layout(
                        (1, _ADDR_BATCH_ROWS * (_ADDR_PIPE_ORI_K + _ADDR_PIPE_CMP_K))
                    ),
                    depth=2,
                )
            tab_cols = self._avec_k1_tab + (self._avec_k2_tab if self.has_cmp else 0)
            self._addr_tab = addr_ws_gm.view(dtypes.int32).view(
                addr_ws_gm.shape[0] // 4 // tab_cols, tab_cols
            )
            if const_expr(_avec == "local"):
                # block_idx 分开同 query 的分核副本；逻辑列布局与 pipe 一致。
                first_row = self.block_idx
                self._addr_tab = self._addr_tab[
                    first_row : first_row + self._ori_idx.shape[0], 0:tab_cols
                ]
            self._addr_bar_flag = 9
            # Both AIVs share an intra-sync ID; AIV1 uses the fixed ID offset on the AIC side.
            self._addr_bar_sid = 9

        # FA metadata follows the AscendC field layout.
        # A core owns rows [m_start,m_end], excluding m_end when s2_end is zero.
        # Boundary rows use the recorded S2 tile limits; zero denotes a full boundary.
        # Consecutive cores partition one contiguous (row,tile) interval.
        # Disabled cores have zero rows and only participate in required barriers.
        fa = self.block_idx * FA_METADATA_SIZE
        bn2_start = _i64(metadata_gm, fa + FA_BN2_START_INDEX)
        m_start_g = _i64(metadata_gm, fa + FA_M_START_INDEX)
        bn2_end = _i64(metadata_gm, fa + FA_BN2_END_INDEX)
        m_end_g = _i64(metadata_gm, fa + FA_M_END_INDEX)
        self._m_start = _i64(self._cu_q, 0, bn2_start) + m_start_g
        self._s2_start = _i64(metadata_gm, fa + FA_S2_START_INDEX)
        m_end = _i64(self._cu_q, 0, bn2_end) + m_end_g
        s2_end = _i64(metadata_gm, fa + FA_S2_END_INDEX)
        self._metadata_gm = metadata_gm
        self._first_fd = _i64(metadata_gm, fa + FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX)
        self._fd_async = _i64(metadata_gm, _FD_ASYNC_MODE_WORD)
        row_count = m_end - self._m_start + (1 if s2_end > 0 else 0)
        self._first_split_row = (
            dtypes.int64(0) if self._s2_start > 0 else dtypes.int64(-1)
        )
        self._last_split_row = row_count - 1 if s2_end > 0 else dtypes.int64(-1)
        # Zero s2 boundaries encode whole-row execution. Compute tile bounds
        # directly; the global FD gate is read only after the main loop.
        if const_expr(_addr_vec_mode() != "off"):
            if const_expr(_addr_vec_mode() == "local"):
                self._addr_precompute_local(row_count)
                onchip_rewind(reset_sync_id=False)
            else:
                # Publish the address table before the main loop consumes it.
                if const_expr(_addr_vec_mode() == "pipe"):
                    self._addr_start_batch = bn2_start
                    self._addr_end_batch = bn2_end
                    # 长 query 复用时，AIC 只读的 query 计数与 AIV 地址计算并行。
                    # Decode 保持原 MTE1 队列时序；-1 表示主循环仍需计数。
                    active_queries = dtypes.int64(-1)
                    if (
                        is_aic()
                        and self._ori_idx.shape[0]
                        >= _KV_L2_PREFILL_MIN_QUERIES * self._b_rt
                    ):
                        active_queries = dtypes.int64(0)
                        for row in range(row_count):
                            row_begin, row_end = self.query_tile_range(
                                row, row_count, s2_end
                            )
                            active_queries += 1 if row_end > row_begin else 0
                    else:
                        self._addr_precompute_batched(row_count)
                    self._pipe_active_queries = active_queries
                # AIC waits for both AIV writes before releasing the block.
                # Use MTE1 so the wait does not block Q prefetch on MTE2.
                cube_sync_intra_wait(PIPE.MTE1, self._addr_bar_sid)
                cube_sync_intra_wait(PIPE.MTE1, self._addr_bar_sid + _AIV1_ID_OFFSET)
                # Wait and release must use the same pipe to preserve ordering.
                cube_sync_block_arrive(PIPE.MTE1, self._addr_bar_flag, mode=2)

                # The prepass MTE3 arrive follows the last workspace store;
                # its S wait observes both AIV completions through the AIC relay.
                # All address Buffer reads/writes are therefore complete before any
                # main-loop UB use. Rewind reclaims storage, not synchronization.
                onchip_rewind(reset_sync_id=False)

        self._setup_main_ub()

        if is_aic():
            self.pipeline_cube(row_count, s2_end)
        else:
            self.pipeline_vector(row_count, s2_end)

        # Empty queries never enter the delay line. Write their outputs after
        # draining so the shared output buffer cannot disturb pending PVs.
        for m_seq in range(0, row_count):
            m_idx = self._m_idx(m_seq)
            if self._n_end_rt(m_idx) == 0:
                o_tile = tile_slice(
                    self._out, (self.tile_cube_m, self.tile_d), (m_idx, 0)
                )
                lse_base = (
                    m_idx * 2 + self.subblock_idx
                    if const_expr(self._lse is not None)
                    else None
                )
                self.vector.finalize_empty(o_tile, self._sinks, self._lse, lse_base)

        # The uniform fd_any field gates synchronization and reduction on every core.
        # Without split rows, skip both. Enabled AIVs reduce their assigned partials.
        fd_any = _i64(metadata_gm, FD_USED_VEC_NUM_WORD)
        if fd_any != 0:
            # Async FD drains local UB users here; per-slot ready polling below
            # provides remote publication ordering. Synchronous FD instead needs
            # the all-core barrier before reading any remote partials. Neither
            # RingBuffer rewind nor new local slots establish these cross-core dependencies.
            if self._fd_async == 1:
                vec_sync_all()
            else:
                global_sync_all()
            # After the barrier, reclaim main-loop UB for the full (32,512) FP32 reduction landing.
            onchip_rewind(reset_sync_id=False)
            self.vector._setup_fd_reduce_buffers()
            aiv_idx = self.block_idx * get_subblock_dim() + self.subblock_idx
            fd_metadata_base = FD_METADATA_BASE + aiv_idx * FD_METADATA_SIZE
            fd_enable = _i64(metadata_gm, fd_metadata_base + FD_CORE_ENABLE_INDEX)
            if fd_enable != 0:
                fd_bn2 = _i64(metadata_gm, fd_metadata_base + FD_BN2_IDX_INDEX)
                fd_m = _i64(metadata_gm, fd_metadata_base + FD_M_IDX_INDEX)
                fd_row = _i64(self._cu_q, 0, fd_bn2) + fd_m
                fd_ws = _i64(metadata_gm, fd_metadata_base + FD_WORKSPACE_IDX_INDEX)
                fd_k = _i64(metadata_gm, fd_metadata_base + FD_WORKSPACE_NUM_INDEX)
                fd_h0 = _i64(metadata_gm, fd_metadata_base + FD_M_START_INDEX)
                fd_o, fd_ms = self._fd_workspace_views()
                if self._fd_async == 1:
                    ready_word = _FD_READY_BASE + (fd_ws // 2) * 2 + self.subblock_idx
                    vec_busy_wait(metadata_gm[ready_word], dtypes.int32(1))
                    # Consume this launch's notification. The next launch may reuse
                    # metadata, so it must not observe the previous ready value.
                    vec_store_bypass(metadata_gm.ptr((ready_word,)), dtypes.int32(0))
                # Both parts are staged before readiness; consume the existing GM
                # workspace without reacquiring an already released C2V output slot.
                self.vector.fd_reduce(
                    fd_o,
                    fd_ms,
                    self.vector.fd_ub,
                    self._out,
                    self._lse,
                    fd_row,
                    fd_ws,
                    fd_k,
                    fd_h0,
                )

    @staticmethod
    def _scale_idx(grp_lane, base: int, m16):
        # Scale gather uses uint16 indices, so its arithmetic requires a 16-bit mask.
        return rr.vadd(grp_lane, rr.vdups(base, dtypes.uint16), mask=m16)

    @jit
    def _setup_main_ub(self):
        # Allocate only after address-prepass storage is dead. L1/L0 and their
        # cross-core protocols remain live across rewind; do not reset IDs.
        self._ori_l2_hint, self._cmp_l2_hint = self.kv_l2_hints()
        tile_vec_m, tile_n = self.tile_vec_m, self.tile_n
        # 两个 256 列半区分别交接，每个 channel 的前/后 32 行为累加区/部分结果。
        self.result_channels = [
            RingBuffer(
                MemLoc.UB,
                dtypes.float32,
                make_layout((2 * tile_vec_m, 256)),
                depth=1,
                data_format="nd",  # codespell:ignore nd
            )
            for half in range(2)
        ]
        self.qk_ub = RingBuffer(
            MemLoc.UB,
            dtypes.float32,
            make_layout((tile_vec_m, tile_n)),
            depth=1,
            data_format="nd",  # codespell:ignore nd
        )
        self.p_out = Buffer(
            MemLoc.UB, dtypes.bfloat16, make_layout(((tile_vec_m + 1) * tile_n,))
        )
        self.vector = Vector(
            tile_vec_m,
            tile_n,
            self.tile_d,
            self.subblock_idx,
            sinks_span=self.sinks_span,
        )

        mem_copy(
            self.vector.sinks_ub,
            tile_slice(self._sinks, (1, self.sinks_span), (0, self.subblock_idx)),
        )

        # The existing scalar PA path accepts runtime page sizes and does not
        # allocate an on-chip buffer proportional to the block-table capacity.
        ch_rows = _TILE_ROWS
        # gather 领先反量化一拍；先填当前 lower 两槽，再消费上一拍 lower，最多六槽在途。
        self.ch_batch_bytes = RingBuffer(
            MemLoc.UB, dtypes.uint8, make_layout((2 * ch_rows, _COMBINE_DIM)), depth=3
        )
        # DMA and VF share one byte RingBuffer; interpret scale bits in registers.
        # Do not create a second typed RingBuffer for the same slot.
        self._v0_out = RingBuffer(
            MemLoc.UB,
            dtypes.bfloat16,
            make_layout((D // 16, _NZ_PAD_ROWS, 16)),
            depth=2,
        )

    # m_seq is the core-local query index; m_idx is the global query index.
    # n spans ORI then CMP tiles; tick increases continuously across queries.
    # Each pool uses its valid prefix for both gather and softmax masking.
    # CMP tile coordinates subtract the ORI tile count.
    # S2 start values are sparse-index column offsets, not token values.

    def _m_idx(self, m_seq: int):
        # Map the local query index with the preloaded metadata start.
        return self._m_start + m_seq

    @jit
    def _batch_of(self, m_idx: int):
        # Dynamic upper-bound search over the B query spans.
        lo, hi = 0, self._b_rt
        while lo < hi:
            mid = (lo + hi) // 2
            take = _i64(self._cu_q, 0, mid + 1) <= m_idx
            lo = mid + 1 if take else lo
            hi = hi if take else mid
        return lo

    @jit
    def _bound_after(self, batch: int, b_last: int):
        # Read the next batch boundary once. At the final span, return a sentinel
        # instead of reading beyond cu_q; this matches _batch_of clamping.
        nxt = batch + 1 if batch < b_last else b_last
        v = _i64(self._cu_q, 0, nxt)
        return v if batch < b_last else dtypes.int64(_ADDR_M_INF)

    def _cmp_tiles_rt(self, m_idx: int):
        w = self._csa_valid_w(m_idx)
        return (w + self.tile_n - 1) // self.tile_n

    @jit
    def _ori_valid_w(self, m_idx: int):
        # ORI width is min(K1, topk_length), or K1 when lengths are absent.
        # Promote it to runtime Int64 even without lengths so FD tile-bound
        # predicates remain SSA values rather than Python booleans.
        if const_expr(self._ori_topk_length is not None):
            return min(self._ori_idx.shape[1], _i64(self._ori_topk_length, m_idx))
        return dtypes.int64(self._ori_idx.shape[1])

    @jit
    def _cmp_topk_len(self, m_idx: int):
        if const_expr(self._cmp_topk_length is not None):
            return min(self._cmp_idx.shape[1], _i64(self._cmp_topk_length, m_idx))
        return dtypes.int64(self._cmp_idx.shape[1])

    def _ori_tiles_rt(self, m_idx: int):
        w = self._ori_valid_w(m_idx)
        return (w + self.tile_n - 1) // self.tile_n

    def _n_end_rt(self, m_idx: int):
        # Runtime query tile count: ORI tiles plus CMP tiles.
        ori_tiles = self._ori_tiles_rt(m_idx)
        if const_expr(not self.has_cmp):
            return ori_tiles
        return ori_tiles + self._cmp_tiles_rt(m_idx)

    @jit
    def _actual_n(self, m_idx: int, n):
        # Return the selected pool tile width for exponent and P masks.
        # The unselected conditional width does not describe an actual task.
        ori_w = self._ori_width(m_idx, n)
        if const_expr(not self.has_cmp):
            return ori_w
        ori_tiles = self._ori_tiles_rt(m_idx)
        cmp_tile_index = n - ori_tiles
        cmp_w = self._cmp_valid(m_idx, cmp_tile_index)
        return cmp_w if n >= ori_tiles else ori_w

    @jit
    def _ori_width(self, m_idx: int, n):
        w = self._ori_valid_w(m_idx)
        return min(self.tile_n, w - n * self.tile_n)

    def _csa_valid_w(self, m_idx: int):
        return self._cmp_topk_len(m_idx)

    @jit
    def _cmp_valid(self, m_idx: int, c):
        return min(self.tile_n, self._csa_valid_w(m_idx) - c * self.tile_n)

    @jit
    def _antiquant_vf_fp8_g32(self, kv_fp8, out):
        # ORI byte order is nope[448], rope[64], then 16 BF16 scales.
        # Dequantize four 128-value chunks through FP32 to BF16 and compact even lanes.
        # Each chunk uses four group-32 scales and writes matching NZ columns.
        subtile_rows = _TILE_ROWS
        scale_view = reinterpret(
            kv_fp8,
            layout=make_layout((_TILE_ROWS, _ROW_BF16)),
            dtype=dtypes.bfloat16,
            data_format="nd",  # codespell:ignore nd
        )
        with vf(mode="simd"):
            m32, _ = rr.update_mask(_TILE_SIZE, elem_bits=32)
            m16, _ = rr.update_mask(_VL, elem_bits=16)
            m8, _ = rr.update_mask(256, elem_bits=8)
            row_bf16, row_fp8 = _ROW_BF16, _COMBINE_DIM

            # 每个 token 只读取一次 inline scale，四个 D chunk 复用寄存器。
            for i in range(subtile_rows):
                scale_a = rr.vreinterpret_lanes(
                    rr.vload_broadcast(
                        scale_view,
                        _SCALE_BF16_OFF + i * row_bf16,
                        mode="elem2datablock",
                    ),
                    dtypes.bfloat16,
                )
                scale_b = rr.vreinterpret_lanes(
                    rr.vload_broadcast(
                        scale_view,
                        _SCALE_BF16_OFF + i * row_bf16 + 8,
                        mode="elem2datablock",
                    ),
                    dtypes.bfloat16,
                )
                s0, s1 = rr.vinterleave(scale_a, scale_a)
                s2, s3 = rr.vinterleave(scale_b, scale_b)
                scale_vectors = (s0, s1, s2, s3)
                for j in tuple(range(D // _KV_NUM_PER_LOOP)):
                    src_j = j * _KV_NUM_PER_LOOP
                    dst_j = src_j
                    s = i * row_fp8 + src_j
                    unpacked = rr.vload_unpack(kv_fp8, s, unpack_mode="b8_to_b16")
                    fp8 = rr.vreinterpret_lanes(unpacked, dtypes.float8_e4m3fn)
                    even = rr.vcast(
                        fp8, dtypes.float32, mask=m8, reg_layout=rr.RegLayout.ZERO
                    )
                    odd = rr.vcast(
                        fp8, dtypes.float32, mask=m8, reg_layout=rr.RegLayout.TWO
                    )
                    # 有限 FP8 精确扩展到 FP32 后低16位为零；合并相邻 BF16。
                    even_bits = rr.vshr(
                        rr.vreinterpret_lanes(even, dtypes.uint32), 16, mask=m32
                    )
                    packed_bits = rr.vbitwise_or(
                        even_bits, rr.vreinterpret_lanes(odd, dtypes.uint32), mask=m32
                    )
                    raw = rr.vreinterpret_lanes(packed_bits, dtypes.bfloat16)
                    svec = scale_vectors[j]
                    packed = rr.vmul(raw, svec, mask=m16)
                    self._v0_store(out, i, dst_j, packed, m16)

    @jit
    def _antiquant_vf_fp4_g16(self, kv_fp8, out):
        # CMP layout: 224 nope bytes, 32 rope bytes, and 64 scale bytes.
        # UNPACK4 + packed FP4 reinterpret + quarter-0 cast.
        # Packed FP4 consumes an 8-bit mask; each byte expands to two BF16 values.
        # Scales follow groups of 16 features, and _v0_store writes NZ17 faces.
        subtile_rows = _TILE_ROWS
        scale_view = reinterpret(
            kv_fp8,
            layout=make_layout((_TILE_ROWS, _ROW_BF16)),
            dtype=dtypes.bfloat16,
            data_format="nd",  # codespell:ignore nd
        )
        with vf(mode="simd"):
            m16, _ = rr.update_mask(_VL, elem_bits=16)
            row_fp8 = _COMBINE_DIM
            scale_off = SCALE_BF16_OFF_CMP

            # Gather each FP4 group scale from the byte buffer and interpret it as BF16.
            fm = rr.full_mask()
            fm.elem_bits = 8
            for i in range(subtile_rows):
                for p in tuple(range(D // _KV_NUM_PER_LOOP)):
                    src_p = p * _TILE_SIZE
                    dst_p = p * _KV_NUM_PER_LOOP
                    u = rr.vload_unpack(
                        kv_fp8, src_p + i * row_fp8, unpack_mode=rr.UnpackMode.UNPACK4
                    )
                    f4 = rr.vreinterpret_lanes(u, dtypes.fp4x2_e2m1)
                    vals = rr.vcast(
                        f4, dtypes.bfloat16, mask=fm, reg_layout=rr.RegLayout.ZERO
                    )
                    scales = rr.vreinterpret_lanes(
                        rr.vload_broadcast(
                            scale_view,
                            scale_off + i * _ROW_BF16 + 8 * p,
                            mode="elem2datablock",
                        ),
                        dtypes.bfloat16,
                    )
                    packed = rr.vmul(vals, scales, mask=m16)
                    self._v0_store(out, i, dst_p, packed, m16)

    def _v0_store(self, out, i, col, val, mask):
        # i is the token row and col the D column; offsets use BF16 elements.
        # NZ17 address = (col//16)*(17*16) + i*16 + col%16.
        # Each 16-column store advances by 17 blocks of 32 bytes; row 17 is padding.
        rr.vstore_strided(
            out,
            (col // 16) * _NZ_CHUNK + i * _NZ_ROW,
            val,
            mask,
            block_stride=_NZ_PAD_ROWS,
            repeat_stride=0,
        )

    @jit
    def kv_l2_hints(self):
        # Keep ORI cached; prefill queries can reuse the CMP pool.
        ori_hint, cmp_hint = dtypes.int32(0), dtypes.int32(0)
        if const_expr(self.has_cmp and _addr_vec_mode() != "off"):
            query_count = self._q.shape[0] // self.tile_cube_m
            cmp_hint = (
                dtypes.int32(0)
                if query_count >= self._b_rt * _KV_L2_PREFILL_MIN_QUERIES
                else dtypes.int32(4)
            )
        return ori_hint, cmp_hint

    @jit
    def gather_batch(
        self,
        raw_half,
        half,
        kv_pool,
        blk_table,
        blk_size,
        s2_start: int,
        n_real,
        idx_tab=None,
        idx_row=None,
        batch=None,
        is_cmp=False,
    ):
        # Each AIV handles four 16-token subblocks of a 128-token tile.
        # Pair DMA sorts KV and inline scales together; otherwise copy rows individually.
        # Tail rows repeat the last valid index; masked P excludes them from PV.
        # Both pools use 544-byte UB rows; cmp reads only its 320-byte payload.
        # Reuse launch-level hints for both paired and fallback row loads.
        kv_l2_policy = self._ori_l2_hint
        if const_expr(is_cmp):
            kv_l2_policy = self._cmp_l2_hint
        subtile_rows = _TILE_ROWS
        if const_expr(is_cmp):
            row_bytes = KV_ROW_BYTES_CMP
            page_stride = self._cmp_page_stride
        else:
            row_bytes = KV_ROW_BYTES_ORI
            page_stride = self._ori_page_stride
        n_rt = n_real
        # Cover physical storage including page gaps; do not flatten logical tokens.
        # mem_copy preserves the RingBuffer producer identity.
        storage_bytes = (kv_pool.shape[0] - 1) * page_stride + blk_size * row_bytes
        kv_bytes = kv_pool.view(storage_bytes)

        _mode = _addr_vec_mode()
        _consumer_vec = _mode != "off"
        tab = None
        if const_expr(_mode != "off"):
            # Read the already-published PA row table directly in scalar gather.
            tab = self._addr_tab
            col_base = self._avec_k1_tab if const_expr(is_cmp) else 0
        # Every enqueued tile has n_real > 0. Write all 16-row tiles, including
        # padding, so both AIVs publish one complete KV slot. QK masks padding;
        # PV uses zero probabilities there. Never publish an empty alias of KV.
        for chunk in range_constexpr(2):
            # Both AIVs publish each KV channel: 32 rows per AIV per half.
            # This keeps the retained QK/PV handoff counters consistent in the DSL.
            sub_b = half * 4 + self.subblock_idx * 2 + chunk
            sub_row = sub_b * subtile_rows
            buf = tile_slice(raw_half, (_TILE_ROWS, _COMBINE_DIM), (chunk, 0))
            bo_sp = 0 if batch is None else batch
            # Clamp padded rows to the final token in the valid prefix before addressing.
            last_off = n_rt - 1 - sub_row
            addresses = []
            pair_first = []
            pair_gap = []
            all_pairs_ok = dtypes.int32(1)
            if const_expr(_consumer_vec):
                # 两种预处理均补齐完整 128-token tile，尾列已经重复最后一个有效地址。
                index_base = col_base + s2_start + sub_row
                for pair in range_constexpr(subtile_rows // 2):
                    pair_row_offset = pair * 2
                    first_index_col = index_base + pair_row_offset
                    second_index_col = first_index_col + 1
                    first_row_index = _i64(tab, idx_row, first_index_col)
                    second_row_index = _i64(tab, idx_row, second_index_col)
                    addr0 = first_row_index * row_bytes
                    addr1 = second_row_index * row_bytes
                    first = min(addr0, addr1)
                    gap = max(addr0, addr1) - first
                    addresses.append(addr0)
                    addresses.append(addr1)
                    pair_first.append(first)
                    pair_gap.append(gap)
                    all_pairs_ok = (
                        (
                            all_pairs_ok
                            if gap - row_bytes <= _DMA_SOURCE_STRIDE_MAX_BYTES
                            else dtypes.int32(0)
                        )
                        if gap >= row_bytes
                        else dtypes.int32(0)
                    )
            else:
                for pair in range_constexpr(subtile_rows // 2):
                    pair_row_offset = pair * 2
                    pos0 = s2_start + sub_row + min(pair_row_offset, last_off)
                    pos1 = s2_start + sub_row + min(pair_row_offset + 1, last_off)
                    tok0 = _i64(idx_tab, idx_row, pos0)
                    tok1 = _i64(idx_tab, idx_row, pos1)
                    blk0, off0 = self.kvcache.pa_blk_off(
                        blk_table, tok0, bo_sp, blk_size
                    )
                    blk1, off1 = self.kvcache.pa_blk_off(
                        blk_table, tok1, bo_sp, blk_size
                    )
                    addr0 = blk0 * page_stride + off0 * row_bytes
                    addr1 = blk1 * page_stride + off1 * row_bytes
                    first = min(addr0, addr1)
                    gap = max(addr0, addr1) - first
                    addresses.append(addr0)
                    addresses.append(addr1)
                    pair_first.append(first)
                    pair_gap.append(gap)
                    all_pairs_ok = (
                        (
                            all_pairs_ok
                            if gap - row_bytes <= _DMA_SOURCE_STRIDE_MAX_BYTES
                            else dtypes.int32(0)
                        )
                        if gap >= row_bytes
                        else dtypes.int32(0)
                    )
            # Deterministic level 3 requires sparse-index order, so disable pair sorting.
            all_pairs_ok = (
                dtypes.int32(0) if self._batch_consistency != 0 else all_pairs_ok
            )
            # Fill all 16 rows before publishing the slot. Reorder KV with inline scales;
            # QK/PV share that order. Duplicate tail addresses use per-row copies.
            if all_pairs_ok != 0:
                for pair in range_constexpr(subtile_rows // 2):
                    first = pair_first[pair]
                    second = first + pair_gap[pair]
                    dst = tile_slice(buf, (2, row_bytes), (pair, 0))
                    mem_copy(
                        dst,
                        (
                            kv_bytes[first : first + row_bytes,].view(1, row_bytes),
                            kv_bytes[second : second + row_bytes,].view(1, row_bytes),
                        ),
                        copy_options(l2_cache_ctl=kv_l2_policy),
                    )
            else:
                for pair_row_offset in range_constexpr(subtile_rows):
                    addr = addresses[pair_row_offset]
                    row = kv_bytes[addr : addr + row_bytes,].view(1, row_bytes)
                    mem_copy(
                        tile_slice(buf, (1, row_bytes), (pair_row_offset, 0)),
                        row,
                        copy_options(l2_cache_ctl=kv_l2_policy),
                    )

    @jit
    def antiquant_copy_batch(
        self, raw_half, half, kv_lower_rows_l1, kv_upper_rows_l1, is_cmp=False
    ):
        # 消费上一拍 gather 的两个 16-token 槽；VF 写 NZ17，再由 MTE3 发布到 L1。
        subtile_rows = _TILE_ROWS
        for chunk in range_constexpr(2):
            v0_out = self._v0_out.next_write()
            rslot_bytes = tile_slice(raw_half, (_TILE_ROWS, _COMBINE_DIM), (chunk, 0))

            if const_expr(is_cmp):
                self._antiquant_vf_fp4_g16(rslot_bytes, v0_out)
            else:
                self._antiquant_vf_fp8_g32(rslot_bytes, v0_out)
            # NZ17 faces become one regular strided UB-to-L1 copy.
            src = reinterpret(
                v0_out,
                layout=make_layout((D // 16, subtile_rows * 16), stride=(_NZ_CHUNK, 1)),
            )
            lt = self.subblock_idx * 2 + chunk
            if half == 0:
                l1_nd = reinterpret(
                    kv_lower_rows_l1,
                    layout=make_layout((D // 16, self.tile_n // 2 * 16)),
                    data_format="nd",  # codespell:ignore nd
                )
                dst = tile_slice(l1_nd, (D // 16, subtile_rows * 16), (0, lt))
                mem_copy(dst, src)
            else:
                l1_nd = reinterpret(
                    kv_upper_rows_l1,
                    layout=make_layout((D // 16, self.tile_n // 2 * 16)),
                    data_format="nd",  # codespell:ignore nd
                )
                dst = tile_slice(l1_nd, (D // 16, subtile_rows * 16), (0, lt))
                mem_copy(dst, src)

    @jit
    def _addr_load_local_group(self, batch, first, nr):
        # 每组 query 不跨 batch；两侧索引、页表和长度由 MTE2 连续填充。
        slot = self._addr_local_input
        bt = reinterpret(slot, layout=make_layout((1, _ADDR_COMPACT_BT)))
        _copy_addr_from_gm(
            bt, self._pa_bt, batch * self._pa_bt.shape[1], self._pa_bt.shape[1] * 4
        )
        sp = reinterpret(
            slot,
            layout=make_layout((1, _ADDR_LOCAL_ROWS * _ADDR_LOCAL_ORI_K)),
            offset=_ADDR_LOCAL_ORI_INDEX_OFFSET,
        )
        _copy_addr_from_gm(
            sp, self._ori_idx, first * self._avec_k1, nr * self._avec_k1 * 4
        )
        if const_expr(self._ori_topk_length is not None):
            lens = reinterpret(
                slot,
                layout=make_layout((1, _ADDR_LOCAL_ROWS)),
                offset=_ADDR_LOCAL_ORI_LENGTH_OFFSET,
            )
            mem_copy(
                reinterpret(lens, layout=make_layout((1, nr))),
                self._ori_topk_length[first : first + nr,].view(1, nr),
            )
        if const_expr(self.has_cmp):
            bt = reinterpret(
                slot,
                layout=make_layout((1, _ADDR_COMPACT_BT)),
                offset=_ADDR_COMPACT_BT * 4,
            )
            _copy_addr_from_gm(
                bt,
                self._pa_bt_cmp,
                batch * self._pa_bt_cmp.shape[1],
                self._pa_bt_cmp.shape[1] * 4,
            )
            sp = reinterpret(
                slot,
                layout=make_layout((1, _ADDR_LOCAL_ROWS * _ADDR_LOCAL_CMP_K)),
                offset=_ADDR_LOCAL_CMP_INDEX_OFFSET,
            )
            _copy_addr_from_gm(
                sp, self._cmp_idx, first * self._avec_k2, nr * self._avec_k2 * 4
            )
            if const_expr(self._cmp_topk_length is not None):
                lens = reinterpret(
                    slot,
                    layout=make_layout((1, _ADDR_LOCAL_ROWS)),
                    offset=_ADDR_LOCAL_CMP_LENGTH_OFFSET,
                )
                mem_copy(
                    reinterpret(lens, layout=make_layout((1, nr))),
                    self._cmp_topk_length[first : first + nr,].view(1, nr),
                )

    @jit
    def _addr_compute_local_side(
        self, sp, lens, bt, out, nr, kwidth, ktab, bs, tab_cols, col0
    ):
        # int32 索引 -> 页号 -> 物理行号，空行的 gather 掩码全关。
        mask = rr.full_mask(elem_bits=32)
        # 三种固定移位结果按页大小掩码合并，VF 内没有运行时分支。
        page_size = rr.vdups(bs, dtypes.int32)
        row_mask = rr.vadds(page_size, -1, mask=mask)
        use64 = rr.veqs(page_size, 64, mask=mask)
        use256 = rr.veqs(page_size, 256, mask=mask)
        lane = rr.varange(0, dtypes.int32)
        # 每 AIV 生成各半 tile 内连续 32 列，MTE3 将紧凑 UB 输出散布回逻辑列。
        own_lane = rr.vadd(
            rr.vmuls(rr.vshr(lane, 5, mask=mask), 64, mask=mask),
            rr.vbitwise_and(lane, rr.vdups(dtypes.int32(31), dtypes.int32), mask=mask),
            mask=mask,
        )
        own_lane = rr.vadds(own_lane, dtypes.int32(self.subblock_idx) * 32, mask=mask)
        for row in cannbotdsl.range(
            dtypes.int32(0), dtypes.int32(nr), dtypes.int32(1), unroll=1
        ):
            active = mask
            last = rr.vdups(dtypes.int32(kwidth - 1), dtypes.int32)
            if const_expr(lens is not None):
                width = rr.vload_broadcast(lens, row)
                width = rr.vmins(width, kwidth, mask=mask)
                active = rr.vgts(width, 0, mask=mask)
                last = rr.vadds(rr.vmaxs(width, 1, mask=mask), -1, mask=mask)
            for chunk in cannbotdsl.range(
                dtypes.int32(0), dtypes.int32(ktab // 128), dtypes.int32(1), unroll=1
            ):
                # 补齐完整 tile；每次生成本 AIV 的 32+32 列，尾列重复最后有效索引。
                mk = mask
                active_chunk = active
                col = rr.vmin(rr.vadds(own_lane, chunk * 128, mask=mk), last, mask=mk)
                source_col = rr.vadds(col, row * kwidth, mask=mk)
                idx = rr.vgather(
                    sp, rr.vreinterpret(source_col, dtypes.uint32), mask=active_chunk
                )
                page = rr.vshr(idx, 7, mask=active_chunk)
                page = rr.vselect(
                    rr.vshr(idx, 6, mask=active_chunk), page, cond_mask=use64
                )
                page = rr.vselect(
                    rr.vshr(idx, 8, mask=active_chunk), page, cond_mask=use256
                )
                physical_page = rr.vgather(
                    bt, rr.vreinterpret(page, dtypes.uint32), mask=active_chunk
                )
                # 物理行 = 物理页号 * 页大小 + 页内偏移；全程 int32 精确计算。
                in_page = rr.vbitwise_and(idx, row_mask, mask=active_chunk)
                physical_base = rr.vmuls(physical_page, bs, mask=active_chunk)
                physical_row = rr.vadd(physical_base, in_page, mask=active_chunk)
                rr.vstore(out, row * tab_cols + col0 + chunk * 64, physical_row, mk)

    @jit
    def _addr_compute_local_group(self, slot, out, nr):
        bt1 = reinterpret(slot, layout=make_layout((1, _ADDR_COMPACT_BT)))
        sp1 = reinterpret(
            slot,
            layout=make_layout((1, _ADDR_LOCAL_ROWS * _ADDR_LOCAL_ORI_K)),
            offset=_ADDR_LOCAL_ORI_INDEX_OFFSET,
        )
        len1 = None
        if const_expr(self._ori_topk_length is not None):
            len1 = reinterpret(
                slot,
                layout=make_layout((1, _ADDR_LOCAL_ROWS)),
                offset=_ADDR_LOCAL_ORI_LENGTH_OFFSET,
            )
        cols = (self._avec_k1_tab + (self._avec_k2_tab if self.has_cmp else 0)) // 2
        if const_expr(self.has_cmp):
            bt2 = reinterpret(
                slot,
                layout=make_layout((1, _ADDR_COMPACT_BT)),
                offset=_ADDR_COMPACT_BT * 4,
            )
            sp2 = reinterpret(
                slot,
                layout=make_layout((1, _ADDR_LOCAL_ROWS * _ADDR_LOCAL_CMP_K)),
                offset=_ADDR_LOCAL_CMP_INDEX_OFFSET,
            )
            len2 = None
            if const_expr(self._cmp_topk_length is not None):
                len2 = reinterpret(
                    slot,
                    layout=make_layout((1, _ADDR_LOCAL_ROWS)),
                    offset=_ADDR_LOCAL_CMP_LENGTH_OFFSET,
                )
        nr = dtypes.int32(nr)
        cols = dtypes.int32(cols)
        kw1, kt1 = dtypes.int32(self._avec_k1), dtypes.int32(self._avec_k1_tab)
        if const_expr(self.has_cmp):
            kw2, kt2 = dtypes.int32(self._avec_k2), dtypes.int32(self._avec_k2_tab)
        ori_bs = dtypes.int32(self._avec_bs_ori)
        if const_expr(self.has_cmp):
            cmp_bs = dtypes.int32(self._avec_bs_cmp)
        with vf(mode="simd"):
            self._addr_compute_local_side(
                sp1, len1, bt1, out, nr, kw1, kt1, ori_bs, cols, dtypes.int32(0)
            )
            if const_expr(self.has_cmp):
                self._addr_compute_local_side(
                    sp2, len2, bt2, out, nr, kw2, kt2, cmp_bs, cols, kt1 // 2
                )

    @jit
    def _addr_precompute_local(self, m_count):
        # 两个 AIV 写互不重叠的 32 列段；主体只读本 AIV 生成的列，无跨 AIV 消费。
        cols = self._avec_k1_tab + (self._avec_k2_tab if self.has_cmp else 0)
        batch = self._batch_of(self._m_start)
        bound = self._bound_after(batch, dtypes.int64(self._b_rt))
        current = dtypes.int64(0)
        while current < m_count:
            first = self._m_idx(current)
            while first >= bound:
                batch = batch + 1
                bound = self._bound_after(batch, dtypes.int64(self._b_rt))
            nr = min(_ADDR_LOCAL_ROWS, min(m_count - current, bound - first))
            self._addr_load_local_group(batch, first, nr)
            self._addr_compute_local_group(
                self._addr_local_input, self._addr_local_output, nr
            )
            # 一次 strided MTE3 将每 32 列落在完整逻辑行内，行列编号与 pipe 相同。
            bursts = nr * cols // 64
            flat = self._addr_tab.view(self._addr_tab.shape[0] * cols)
            first_col = first * cols
            dst = flat[first_col : first_col + nr * cols,].view(bursts, 64)
            dst = dst[0:bursts, self.subblock_idx * 32 : self.subblock_idx * 32 + 32]
            mem_copy(
                dst,
                reinterpret(self._addr_local_output, layout=make_layout((bursts, 32))),
            )
            current = current + nr
        # 每个 AIV 仅消费自己的地址区，S 等待本地 MTE3 完成后才回收预处理存储。
        vec_sync_all()

    @jit
    def _addr_load_group(self, batch, first, nr):
        # 同一槽中的页表、索引和长度按本次调用的容量布局，组不跨 batch。
        slot = self._addr_input_ring.next_write()
        bt = _addr_pipe_view(slot, 0, self._addr_pipe_bt1)
        _copy_addr_elements_from_gm(
            bt, self._pa_bt, batch * self._pa_bt.shape[1], self._pa_bt.shape[1]
        )
        sp = _addr_pipe_view(
            slot, self._addr_pipe_ori_index, self._addr_pipe_rows * self._avec_k1
        )
        _copy_addr_elements_from_gm(
            sp, self._ori_idx, first * self._avec_k1, nr * self._avec_k1
        )
        if const_expr(self._ori_topk_length is not None):
            lens = _addr_pipe_view(slot, self._addr_pipe_ori_length, 16)
            mem_copy(
                reinterpret(lens, layout=make_layout((1, nr))),
                self._ori_topk_length[first : first + nr,].view(1, nr),
            )
        if const_expr(self.has_cmp):
            bt = _addr_pipe_view(slot, self._addr_pipe_bt1, self._addr_pipe_bt2)
            _copy_addr_elements_from_gm(
                bt,
                self._pa_bt_cmp,
                batch * self._pa_bt_cmp.shape[1],
                self._pa_bt_cmp.shape[1],
            )
            sp = _addr_pipe_view(
                slot, self._addr_pipe_cmp_index, self._addr_pipe_rows * self._avec_k2
            )
            _copy_addr_elements_from_gm(
                sp, self._cmp_idx, first * self._avec_k2, nr * self._avec_k2
            )
            if const_expr(self._cmp_topk_length is not None):
                lens = _addr_pipe_view(slot, self._addr_pipe_cmp_length, 16)
                mem_copy(
                    reinterpret(lens, layout=make_layout((1, nr))),
                    self._cmp_topk_length[first : first + nr,].view(1, nr),
                )

    @jit
    def _addr_compute_group_side(
        self, sp, lens, bt, out, nr, kwidth, chunks, bs, page_scale, tab_cols, col0
    ):
        # int32 索引 -> 页号 -> 物理行号，空行的 gather 掩码全关。
        mask = rr.full_mask(elem_bits=32)
        # 页大小限定 64/128/256，页号为 (idx * (256 / bs)) >> 8。
        page_size = rr.vdups(bs, dtypes.int32)
        row_mask = rr.vadds(page_size, -1, mask=mask)
        lane = rr.varange(0, dtypes.int32)
        for row in cannbotdsl.range(
            dtypes.int32(0), dtypes.int32(nr), dtypes.int32(1), unroll=1
        ):
            active = mask
            last = rr.vdups(dtypes.int32(kwidth - 1), dtypes.int32)
            if const_expr(lens is not None):
                width = rr.vload_broadcast(lens, row)
                width = rr.vmins(width, kwidth, mask=mask)
                active = rr.vgts(width, 0, mask=mask)
                last = rr.vadds(rr.vmaxs(width, 1, mask=mask), -1, mask=mask)
            for chunk in cannbotdsl.range(
                dtypes.int32(0), chunks, dtypes.int32(1), unroll=1
            ):
                mk = mask
                active_chunk = active
                col = rr.vmin(rr.vadds(lane, chunk * 64, mask=mk), last, mask=mk)
                source_col = rr.vadds(col, row * kwidth, mask=mk)
                idx = rr.vgather(
                    sp, rr.vreinterpret(source_col, dtypes.uint32), mask=active_chunk
                )
                page = rr.vshr(
                    rr.vmuls(idx, page_scale, mask=active_chunk), 8, mask=active_chunk
                )
                page_index = rr.vreinterpret(page, dtypes.uint32)
                # 同一路径直接查 UB 页表，避免每组寄存器前缀和双路选择的额外指令。
                physical_page = rr.vgather(bt, page_index, mask=active_chunk)
                # 物理行 = 物理页号 * 页大小 + 页内偏移；全程 int32 精确计算。
                in_page = rr.vbitwise_and(idx, row_mask, mask=active_chunk)
                physical_base = rr.vmuls(physical_page, bs, mask=active_chunk)
                physical_row = rr.vadd(physical_base, in_page, mask=active_chunk)
                rr.vstore(out, row * tab_cols + col0 + chunk * 64, physical_row, mk)

    @jit
    def _addr_compute_group(self, slot, out, nr):
        bt1 = _addr_pipe_view(slot, 0, self._addr_pipe_bt1)
        sp1 = _addr_pipe_view(
            slot, self._addr_pipe_ori_index, self._addr_pipe_rows * self._avec_k1
        )
        len1 = None
        if const_expr(self._ori_topk_length is not None):
            len1 = _addr_pipe_view(slot, self._addr_pipe_ori_length, 16)
        cols = self._avec_k1_tab + (self._avec_k2_tab if self.has_cmp else 0)
        if const_expr(self.has_cmp):
            bt2 = _addr_pipe_view(slot, self._addr_pipe_bt1, self._addr_pipe_bt2)
            sp2 = _addr_pipe_view(
                slot, self._addr_pipe_cmp_index, self._addr_pipe_rows * self._avec_k2
            )
            len2 = None
            if const_expr(self._cmp_topk_length is not None):
                len2 = _addr_pipe_view(slot, self._addr_pipe_cmp_length, 16)
        nr = dtypes.int32(nr)
        cols = dtypes.int32(cols)
        kw1, kt1 = dtypes.int32(self._avec_k1), dtypes.int32(self._avec_k1_tab)
        if const_expr(self.has_cmp):
            kw2, _kt2 = dtypes.int32(self._avec_k2), dtypes.int32(self._avec_k2_tab)
        ori_bs = dtypes.int32(self._avec_bs_ori)
        ori_chunks = dtypes.int32(self._avec_k1_tab // 64)
        ori_page_scale = 4 if ori_bs == 64 else (2 if ori_bs == 128 else 1)
        if const_expr(self.has_cmp):
            cmp_bs = dtypes.int32(self._avec_bs_cmp)
            cmp_chunks = dtypes.int32(self._avec_k2_tab // 64)
            cmp_page_scale = 4 if cmp_bs == 64 else (2 if cmp_bs == 128 else 1)
        with vf(mode="simd"):
            self._addr_compute_group_side(
                sp1,
                len1,
                bt1,
                out,
                nr,
                kw1,
                ori_chunks,
                ori_bs,
                ori_page_scale,
                cols,
                dtypes.int32(0),
            )
            if const_expr(self.has_cmp):
                self._addr_compute_group_side(
                    sp2,
                    len2,
                    bt2,
                    out,
                    nr,
                    kw2,
                    cmp_chunks,
                    cmp_bs,
                    cmp_page_scale,
                    cols,
                    kt1,
                )

    @jit
    def _addr_precompute_batched(self, m_count):
        cols = self._avec_k1_tab + (self._avec_k2_tab if self.has_cmp else 0)
        core = dtypes.int64(self.subblock_idx)
        per, tail = m_count // 2, m_count % 2
        start = per * core + min(core, tail)
        stop = start + per + (dtypes.int64(1) if core < tail else dtypes.int64(0))
        current = start
        # metadata 将本核 query 限定在已知 batch 区间；同 batch 时无需读 cu_q 搜索。
        batch, upper = self._addr_start_batch, self._addr_end_batch
        target = self._m_idx(start)
        while batch < upper:
            mid = (batch + upper) // 2
            take = _i64(self._cu_q, 0, mid + 1) <= target
            batch = mid + 1 if take else batch
            upper = upper if take else mid
        bound = self._bound_after(batch, dtypes.int64(self._b_rt))
        # 输入预取推进 batch 前，保存消费游标的初始位置。
        read_batch, read_bound = batch, bound
        # 先填满输入环，再边计算最早的一组边补入后续组；组不跨 batch。
        for prefetch in range(_ADDR_BATCH_DEPTH - 1):
            if current < stop:
                first = self._m_idx(current)
                while first >= bound:
                    batch = batch + 1
                    bound = self._bound_after(batch, dtypes.int64(self._b_rt))
                nr = min(self._addr_pipe_rows, min(stop - current, bound - first))
                self._addr_load_group(batch, first, nr)
                current = current + nr
        consumed = start
        while consumed < stop:
            if current < stop:
                first = self._m_idx(current)
                while first >= bound:
                    batch = batch + 1
                    bound = self._bound_after(batch, dtypes.int64(self._b_rt))
                nr = min(self._addr_pipe_rows, min(stop - current, bound - first))
                self._addr_load_group(batch, first, nr)
                current = current + nr
            first = self._m_idx(consumed)
            while first >= read_bound:
                read_batch = read_batch + 1
                read_bound = self._bound_after(read_batch, dtypes.int64(self._b_rt))
            nr = min(self._addr_pipe_rows, min(stop - consumed, read_bound - first))
            slot = self._addr_input_ring.next_read()
            out = self._addr_out_ring.next_write()
            self._addr_compute_group(slot, out, nr)
            ready = self._addr_out_ring.next_read()
            mem_copy(
                self._addr_tab[first : first + nr, 0:cols],
                reinterpret(ready, layout=make_layout((nr, cols))),
            )
            consumed = consumed + nr
        vec_sync_intra_arrive(PIPE.MTE3, self._addr_bar_sid)
        vec_sync_block_wait(PIPE.S, self._addr_bar_flag, mode=2)

    @jit
    def gather_quantized(self, m_seq: int, n, half, actual_n, ori_tiles):
        # 分支外只选一次 32-token 槽，两种格式共享同一生产游标。
        raw_half = self.ch_batch_bytes.next_write()
        m_idx = self._m_idx(m_seq)
        # Only scalar addressing needs a per-tile batch lookup.
        batch = None
        if const_expr(_addr_vec_mode() == "off"):
            batch = self._batch_of(m_idx)

        if const_expr(not self.has_cmp):
            self.gather_batch(
                raw_half,
                half,
                self._pa_phys,
                self._pa_bt,
                self._pa_bs_rt,
                n * self.tile_n,
                actual_n,
                idx_tab=self._ori_idx,
                idx_row=m_idx,
                batch=batch,
                is_cmp=False,
            )
        else:
            ori_bound = ori_tiles
            if n >= ori_bound:
                cmp_tile_index = n - ori_bound
                self.gather_batch(
                    raw_half,
                    half,
                    self._pa_phys_cmp,
                    self._pa_bt_cmp,
                    self._pa_cmp_bs_rt,
                    cmp_tile_index * self.tile_n,
                    actual_n,
                    idx_tab=self._cmp_idx,
                    idx_row=m_idx,
                    batch=batch,
                    is_cmp=True,
                )
            else:
                self.gather_batch(
                    raw_half,
                    half,
                    self._pa_phys,
                    self._pa_bt,
                    self._pa_bs_rt,
                    n * self.tile_n,
                    actual_n,
                    idx_tab=self._ori_idx,
                    idx_row=m_idx,
                    batch=batch,
                    is_cmp=False,
                )

    @jit
    def stage_antiquant(self, n, ori_tiles, half, kv_lower_rows_l1, kv_upper_rows_l1):
        # 格式来自延迟 task；分支外选槽，保持每半块一次消费。
        raw_half = self.ch_batch_bytes.next_read()
        if const_expr(self.has_cmp):
            if n >= ori_tiles:
                self.antiquant_copy_batch(
                    raw_half, half, kv_lower_rows_l1, kv_upper_rows_l1, True
                )
            else:
                self.antiquant_copy_batch(
                    raw_half, half, kv_lower_rows_l1, kv_upper_rows_l1, False
                )
        else:
            self.antiquant_copy_batch(
                raw_half, half, kv_lower_rows_l1, kv_upper_rows_l1, False
            )

    @jit
    def stage_softmax(
        self,
        tick: int,
        m_seq: int,
        n,
        is_first: int,
        job: int,
        is_last: int,
        score,
        p_l1_write,
        actual_n,
    ):
        continuation = (self._s2_start > 0) if m_seq == 0 else False
        if actual_n == self.tile_n:
            if is_first == 1:
                if continuation:
                    self.vector.softmax_rest_full(
                        job,
                        tick,
                        self._scale,
                        True,
                        True,
                        score,
                        self.p_out,
                        actual_n,
                        False,
                    )
                else:
                    self.vector.softmax_rest_full(
                        job,
                        tick,
                        self._scale,
                        True,
                        True,
                        score,
                        self.p_out,
                        actual_n,
                        True,
                    )
            else:
                self.vector.softmax_rest_full(
                    job,
                    tick,
                    self._scale,
                    False,
                    True,
                    score,
                    self.p_out,
                    actual_n,
                    True,
                )
        else:
            if is_first == 1:
                if continuation:
                    self.vector.softmax_rest_full(
                        job,
                        tick,
                        self._scale,
                        True,
                        False,
                        score,
                        self.p_out,
                        actual_n,
                        False,
                    )
                else:
                    self.vector.softmax_rest_full(
                        job,
                        tick,
                        self._scale,
                        True,
                        False,
                        score,
                        self.p_out,
                        actual_n,
                        True,
                    )
            else:
                self.vector.softmax_rest_full(
                    job,
                    tick,
                    self._scale,
                    False,
                    False,
                    score,
                    self.p_out,
                    actual_n,
                    True,
                )
        source = self.p_out.view(self.tile_n // 16, (self.tile_vec_m + 1) * 16)[
            None, 0 : self.tile_vec_m * 16
        ]
        target = tile_slice(
            reinterpret(
                p_l1_write,
                layout=make_layout((self.tile_n // 16, self.tile_cube_m * 16)),
                data_format="nd",  # codespell:ignore nd
            ),
            (self.tile_n // 16, self.tile_vec_m * 16),
            (0, self.subblock_idx),
        )
        mem_copy(target, source)
        if is_last == 1:
            self.vector.finalize_softmax_state(job, self._scale)

    def next_kv(self, write):
        return (
            (
                self.matmul.kv_channels[0].next_write(),
                self.matmul.kv_channels[1].next_write(),
            )
            if write
            else (
                self.matmul.kv_channels[0].next_read(),
                self.matmul.kv_channels[1].next_read(),
            )
        )

    def next_result(self, write):
        lo = (
            self.result_channels[0].next_write()
            if write
            else self.result_channels[0].next_read()
        )
        hi = (
            self.result_channels[1].next_write()
            if write
            else self.result_channels[1].next_read()
        )
        return (
            tile_slice(lo, (self.tile_vec_m, 256), (0, 0)),
            tile_slice(lo, (self.tile_vec_m, 256), (1, 0)),
            tile_slice(hi, (self.tile_vec_m, 256), (0, 0)),
            tile_slice(hi, (self.tile_vec_m, 256), (1, 0)),
        )

    @jit
    def matmul_qk(self, job, kv_lower_rows_l1, kv_upper_rows_l1):
        self.matmul.bmm1_fanout_q(job, kv_lower_rows_l1, kv_upper_rows_l1)
        self.matmul.store_s(self.qk_ub.next_write())

    @jit
    def cube_pv(self, kv_lower_rows_l1, kv_upper_rows_l1, m_seq: int, n, is_first: int):
        self.matmul.compute_pv_fanout(
            self.p_l1.next_read(),
            self.next_result(True),
            kv_lower_rows_l1,
            kv_upper_rows_l1,
            is_first,
        )

    @jit
    def query_tile_range(self, m_seq, row_count, s2_end):
        tiles = self._n_end_rt(self._m_idx(m_seq))
        begin = self._s2_start if m_seq == 0 else dtypes.int64(0)
        end = (s2_end if s2_end > 0 else tiles) if m_seq == row_count - 1 else tiles
        return begin, end

    @jit
    def next_query(self, m_seq, row_count, s2_end):
        stop = row_count
        while m_seq < stop:
            begin, end = self.query_tile_range(m_seq, row_count, s2_end)
            stop = m_seq if end > begin else stop
            m_seq = m_seq if end > begin else m_seq + 1
        return m_seq

    def query_view(self, m_seq):
        return self._q.view(
            self._q.shape[0] // self.tile_cube_m, self.tile_cube_m, self.tile_d
        )[self._m_idx(m_seq), None, None]

    @jit
    def pipeline_cube(self, row_count, s2_end):
        m_seq = self.next_query(dtypes.int64(0), row_count, s2_end)
        job = dtypes.int64(0)
        if m_seq < row_count:
            begin, end = self.query_tile_range(m_seq, row_count, s2_end)
            self.matmul.load_q_wide(self.query_view(m_seq), job, False)
            self.matmul.load_q_wide(self.query_view(m_seq), job, True)
            kv_lower_rows_l1, kv_upper_rows_l1 = self.next_kv(False)
            self.matmul_qk(job, kv_lower_rows_l1, kv_upper_rows_l1)
            active_rows = dtypes.int64(0)
            if const_expr(_addr_vec_mode() == "pipe"):
                active_rows = self._pipe_active_queries
                if active_rows < 0:
                    active_rows = dtypes.int64(0)
                    for row in range(row_count):
                        row_begin, row_end = self.query_tile_range(
                            row, row_count, s2_end
                        )
                        active_rows += 1 if row_end > row_begin else 0
            else:
                for row in range(row_count):
                    row_begin, row_end = self.query_tile_range(row, row_count, s2_end)
                    active_rows += 1 if row_end > row_begin else 0
            for _ in range(active_rows):
                next_m = self.next_query(m_seq + 1, row_count, s2_end)
                if next_m < row_count:
                    self.matmul.load_q_wide(self.query_view(next_m), job + 1, False)
                for n in range(begin, end - 1):
                    previous_kv_lower_rows_l1, previous_kv_upper_rows_l1 = (
                        kv_lower_rows_l1,
                        kv_upper_rows_l1,
                    )
                    kv_lower_rows_l1, kv_upper_rows_l1 = self.next_kv(False)
                    self.matmul_qk(job, kv_lower_rows_l1, kv_upper_rows_l1)
                    self.cube_pv(
                        previous_kv_lower_rows_l1,
                        previous_kv_upper_rows_l1,
                        m_seq,
                        n,
                        1 if n == begin else 0,
                    )
                previous_kv_lower_rows_l1, previous_kv_upper_rows_l1 = (
                    kv_lower_rows_l1,
                    kv_upper_rows_l1,
                )
                if next_m < row_count:
                    self.matmul.load_q_wide(self.query_view(next_m), job + 1, True)
                    kv_lower_rows_l1, kv_upper_rows_l1 = self.next_kv(False)
                    self.matmul_qk(job + 1, kv_lower_rows_l1, kv_upper_rows_l1)
                self.cube_pv(
                    previous_kv_lower_rows_l1,
                    previous_kv_upper_rows_l1,
                    m_seq,
                    end - 1,
                    1 if end - begin == 1 else 0,
                )
                m_seq = next_m
                job = job + 1
                if m_seq < row_count:
                    begin, end = self.query_tile_range(m_seq, row_count, s2_end)

    @jit
    def run_vector_pipeline_step(
        self, delay, tick, draining, kv_lower_rows_l1, kv_upper_rows_l1
    ):
        # gather(t)、反量化(t-1)、softmax(t-3)、update(t-4) 分别保留独立生命周期。
        score = self.qk_ub.next_read()
        p_l1_write = self.p_l1.next_write()
        results = self.next_result(False)
        gather = delay.tap(0)
        antiquant = delay.tap(1)
        softmax = delay.tap(3)
        update = delay.tap(4)
        if not draining:
            self.gather_quantized(gather.m, gather.n, 0, gather.valid, gather.ori_tiles)
        if tick >= 1 and antiquant.n >= 0:
            self.stage_antiquant(
                antiquant.n, antiquant.ori_tiles, 0, kv_lower_rows_l1, kv_upper_rows_l1
            )
        if tick >= 3 and softmax.n >= 0:
            self.stage_softmax(
                softmax.t,
                softmax.m,
                softmax.n,
                softmax.first,
                softmax.sum_slot,
                softmax.last,
                score,
                p_l1_write,
                softmax.valid,
            )
        if not draining:
            self.gather_quantized(gather.m, gather.n, 1, gather.valid, gather.ori_tiles)
        if tick >= 1 and antiquant.n >= 0:
            self.stage_antiquant(
                antiquant.n, antiquant.ori_tiles, 1, kv_lower_rows_l1, kv_upper_rows_l1
            )
            kv_lower_rows_l1, kv_upper_rows_l1 = self.next_kv(True)
        if tick >= 4 and update.n >= 0:
            self.vector.bind_result(results)
            if update.first == 1:
                for half in range_constexpr(2):
                    with vf(mode="simd"):
                        mask, _ = rr.update_mask(1, elem_bits=32)
                        value = rr.vload(results[half * 2], 0)
                        rr.vstore(results[half * 2], 0, value, mask)
            for half in range_constexpr(2):
                self.stage_update(
                    update.t, update.m, update.last, update.first, update.sum_slot, half
                )
        delay.advance()
        return kv_lower_rows_l1, kv_upper_rows_l1

    @jit
    def pipeline_vector(self, row_count, s2_end):
        delay = SlotMajorDelayLineGroup()
        tick = dtypes.int64(0)
        job = dtypes.int64(0)
        kv_lower_rows_l1, kv_upper_rows_l1 = self.next_kv(True)
        for m_seq in range(row_count):
            begin, end = self.query_tile_range(m_seq, row_count, s2_end)
            m_idx = self._m_idx(m_seq)
            ori_length = self._ori_valid_w(m_idx)
            ori_tiles = (ori_length + self.tile_n - 1) // self.tile_n
            cmp_length = dtypes.int64(0)
            if const_expr(self.has_cmp):
                cmp_length = self._cmp_topk_len(m_idx)
            for n in range(begin, end):
                valid = min(self.tile_n, ori_length - n * self.tile_n)
                if const_expr(self.has_cmp):
                    if n >= ori_tiles:
                        valid = min(
                            self.tile_n, cmp_length - (n - ori_tiles) * self.tile_n
                        )
                delay.push(
                    m=m_seq,
                    n=n,
                    t=tick,
                    last=1 if n == end - 1 else 0,
                    first=1 if n == begin else 0,
                    sum_slot=job % 2,
                    valid=valid,
                    ori_tiles=ori_tiles,
                )
                kv_lower_rows_l1, kv_upper_rows_l1 = self.run_vector_pipeline_step(
                    delay, tick, False, kv_lower_rows_l1, kv_upper_rows_l1
                )
                tick = tick + 1
            job = job + (1 if end > begin else 0)
        for _ in range(4):
            delay.push(
                m=0, n=-1, t=tick, last=0, first=0, sum_slot=0, valid=0, ori_tiles=0
            )
            kv_lower_rows_l1, kv_upper_rows_l1 = self.run_vector_pipeline_step(
                delay, tick, True, kv_lower_rows_l1, kv_upper_rows_l1
            )
            tick = tick + 1

    @jit
    def stage_update(
        self, tick: int, m_seq: int, is_last: int, is_first: int, sum_slot: int, half
    ):
        # At a part boundary, normalize O and either stage the split part or write final output.
        # Empty rows bypass the pipeline and are finalized after draining.
        exp_idx = tick % 2
        sum_idx = sum_slot

        m_idx = self._m_idx(m_seq)
        o_row_tile = m_idx
        lse_tile_base = (
            (o_row_tile * (self.tile_cube_m // self.tile_vec_m) + self.subblock_idx)
            if const_expr(self._lse is not None)
            else None
        )
        # LSE coordinates count 32-element tiles: flat offset m_idx*64 + aiv_id*32.
        if is_last == 1:
            split_count = (
                dtypes.int64(1) if m_seq == self._first_split_row else dtypes.int64(0)
            )
            split_count += (
                dtypes.int64(1) if m_seq == self._last_split_row else dtypes.int64(0)
            )
            if split_count > 0:
                if is_first == 1:
                    self.vector.update_o(half, True, True, exp_idx, sum_idx, False)
                else:
                    self.vector.update_o(half, False, True, exp_idx, sum_idx, False)
                if const_expr(half == 1):
                    # A leading continuation uses slot 0. A trailing split uses slot 1 only
                    # when this core also starts with a continuation.
                    part_idx = (
                        dtypes.int64(0)
                        if m_seq == 0
                        else (
                            dtypes.int64(1) if self._s2_start > 0 else dtypes.int64(0)
                        )
                    )
                    slot = self._first_fd + part_idx
                    fd_o, fd_ms = self._fd_workspace_views()
                    self.vector.stage_partial(fd_o, fd_ms, slot, sum_idx)
                    if self._fd_async == 1 and slot % 2 == 1:
                        # The second part publishes readiness only after all
                        # three MTE3 staging writes have completed.
                        # Drain MTE3 before publishing ready to the remote reducer.
                        # This is a local completion fence, not an all-core barrier.
                        vec_sync_all()
                        ready_word = (
                            _FD_READY_BASE + (slot // 2) * 2 + self.subblock_idx
                        )
                        vec_store_bypass(
                            self._metadata_gm.ptr((ready_word,)), dtypes.int32(1)
                        )
            else:
                if is_first == 1:
                    self.vector.update_o(half, True, True, exp_idx, sum_idx, True)
                else:
                    self.vector.update_o(half, False, True, exp_idx, sum_idx, True)
                o_tile_gm = tile_slice(
                    self._out, (self.tile_cube_m, self.tile_d), (o_row_tile, 0)
                )
                self.vector.finalize_o(
                    o_tile_gm, half, sum_idx, lse_gm=self._lse, lse_base=lse_tile_base
                )
        else:
            if is_first == 1:
                pass
            else:
                self.vector.update_o(half, False, False, exp_idx, sum_idx, False)

    @jit
    def _fd_workspace_views(self):
        # FD views are needed only by split-row staging and final reduction.
        blocks = self._ws.shape[0] // WS_BYTES_PER_CORE
        slots = blocks * FD_SLOTS_PER_CORE
        fd_o0 = dtypes.int64(0)
        ws_f32 = self._ws.view(dtypes.float32)
        fd_o = ws_f32[fd_o0 : fd_o0 + slots * FD_O_ELEMS,].view(slots * N1, D)
        fd_ms0 = fd_o0 + slots * FD_O_ELEMS
        fd_ms = ws_f32[fd_ms0 : fd_ms0 + slots * FD_MS_ELEMS,].view(slots * FD_MS_ELEMS)
        return fd_o, fd_ms


def _batch_consistency_enabled():
    """Enable batch consistency at torch_npu deterministic level 3.

    This selects per-row sparse KV copies instead of paired-row DMA.
    """
    try:
        import torch_npu

        return int(torch_npu.npu._get_deterministic_level()) == 3
    except (AttributeError, ImportError):
        return False


def _validate_inputs(
    q,
    *,
    ori_kv=None,
    cmp_kv=None,
    ori_sparse_indices=None,
    cmp_sparse_indices=None,
    ori_block_table=None,
    cmp_block_table=None,
    cu_seqlens_q=None,
    seqused_q=None,
    seqused_ori_kv=None,
    seqused_cmp_kv=None,
    ori_topk_length=None,
    cmp_topk_length=None,
    sinks=None,
    metadata=None,
    quant_mode,
    layout_q="TND",
    layout_kv="PA_BBND",
):
    """Validate tensor metadata only; never read tensor values on the host.

    Callers supply valid prefix sums, sequence lengths and active indices.
    seqused_q, when supplied, must equal the cu_seqlens_q spans (no padding).
    """
    if ori_kv is None:
        raise ValueError("Invalid input: ori_kv is required")
    if str(layout_q).upper() != "TND" or str(layout_kv).upper() != "PA_BBND":
        raise ValueError(
            "Invalid input: mixed_quant_sparse_flash_mla supports only "
            "layout_q='TND' and layout_kv='PA_BBND'"
        )
    if int(quant_mode) != 1:
        raise ValueError(
            "Invalid input: mixed_quant_sparse_flash_mla supports only quant_mode=1"
        )

    if not isinstance(q, torch.Tensor):
        raise TypeError("Invalid input: q must be a torch.Tensor")
    if q.dtype != torch.bfloat16 or q.dim() != _TENSOR_RANK_3D or int(q.shape[-1]) != D:
        raise ValueError(
            "Invalid input: q must be a BF16 TND tensor with shape [T1, N1, 512]"
        )
    rows, n1 = int(q.shape[0]), int(q.shape[1])
    if n1 != N1:
        raise ValueError(
            f"N1 must be {N1} (the only supported query-head count), got {n1}"
        )
    if rows <= 0:
        raise ValueError("Invalid input: q must contain at least one token")

    if cu_seqlens_q is None:
        raise ValueError(
            "Invalid input: cu_seqlens_q is required for TND varlen and is never synthesized: "
            "pass an int32 [B+1] prefix-sum tensor ending at q.shape[0]"
        )
    if (
        not isinstance(cu_seqlens_q, torch.Tensor)
        or cu_seqlens_q.dtype != torch.int32
        or cu_seqlens_q.dim() != _TENSOR_RANK_1D
    ):
        raise ValueError(
            "Invalid input: cu_seqlens_q must be an int32 tensor of shape (B+1,)"
        )
    if cu_seqlens_q.numel() < 2:
        raise ValueError("Invalid input: cu_seqlens_q must describe at least one batch")
    b = cu_seqlens_q.numel() - 1

    if seqused_q is not None:
        if not isinstance(seqused_q, torch.Tensor):
            raise TypeError(
                "Invalid input: seqused_q must be a torch.Tensor when provided"
            )
        if seqused_q.dtype != torch.int32 or seqused_q.dim() != _TENSOR_RANK_1D:
            raise ValueError(
                "Invalid input: seqused_q must be a 1-D int32 tensor of shape (B,)"
            )
        if seqused_q.numel() != b:
            raise ValueError(
                "Invalid input: seqused_q must have one length per cu_seqlens_q batch"
            )

    for name, seqused in (
        ("seqused_ori_kv", seqused_ori_kv),
        ("seqused_cmp_kv", seqused_cmp_kv),
    ):
        if seqused is None:
            continue
        if not isinstance(seqused, torch.Tensor):
            raise TypeError(
                f"Invalid input: {name} must be a torch.Tensor when provided"
            )
        if seqused.dtype != torch.int32 or seqused.dim() != _TENSOR_RANK_1D:
            raise ValueError(
                f"Invalid input: {name} must be a 1-D int32 tensor of shape (B,)"
            )
        if int(seqused.numel()) != b:
            raise ValueError(
                f"Invalid input: {name} must have one entry per batch element (B={b})"
            )

    has_cmp = (
        cmp_kv is not None
        or cmp_sparse_indices is not None
        or cmp_block_table is not None
    )
    if has_cmp:
        if cmp_kv is None or cmp_sparse_indices is None or cmp_block_table is None:
            raise ValueError(
                "ORI_CMP_SPARSE requires cmp_kv, cmp_sparse_indices, "
                "cmp_block_table together"
            )
    elif cmp_topk_length is not None:
        raise ValueError("Invalid input: cmp inputs are not valid for ORI_SPARSE")

    _validate_pa_side(
        ori_kv, ori_block_table, name="ori_kv", row_bytes=KV_ROW_BYTES_ORI
    )
    _validate_indices(ori_sparse_indices, name="ori_sparse_indices", rows=rows)
    _validate_lengths(ori_topk_length, name="ori_topk_length", rows=rows)

    if has_cmp:
        _validate_pa_side(
            cmp_kv, cmp_block_table, name="cmp_kv", row_bytes=KV_ROW_BYTES_CMP
        )
        _validate_indices(cmp_sparse_indices, name="cmp_sparse_indices", rows=rows)
        _validate_lengths(cmp_topk_length, name="cmp_topk_length", rows=rows)

    if sinks is None:
        raise ValueError(
            "Invalid input: sinks is required: pass an FP32 [N1] tensor shared by all batches"
        )
    if (
        not isinstance(sinks, torch.Tensor)
        or sinks.dtype != torch.float32
        or tuple(sinks.shape) != (n1,)
    ):
        raise ValueError(
            "Invalid input: sinks must be an FP32 tensor with shape [N1], shared by all batches"
        )

    if metadata is None:
        raise ValueError(
            "Invalid input: metadata is required and is never synthesized: generate it with "
            "mixed_quant_sparse_flash_mla_metadata(ori_topk_length, cmp_topk_length, "
            "num_heads_q=64, num_heads_kv=1, head_dim=512, quant_mode=1, has_cmp_kv=...) and pass "
            "the result in (two-stage call)"
        )
    if not isinstance(metadata, torch.Tensor):
        raise TypeError("Invalid input: metadata must be a torch.Tensor")
    if (
        metadata.dtype != torch.int32
        or metadata.dim() != _TENSOR_RANK_1D
        or metadata.numel() != MQSMLA_METADATA_TOTAL_SIZE
    ):
        raise ValueError(
            f"Invalid input: metadata must be a 1-D int32 tensor with exactly "
            f"{MQSMLA_METADATA_TOTAL_SIZE} elements (fixed shape)"
        )


def _validate_pa_side(kv, block_table, *, name, row_bytes):
    if not isinstance(kv, torch.Tensor) or not isinstance(block_table, torch.Tensor):
        raise TypeError(
            f"Invalid input: {name} and {name}_block_table must be torch.Tensor"
        )
    if (
        kv.dtype != torch.uint8
        or kv.dim() != _TENSOR_RANK_4D
        or int(kv.shape[2]) != N2
        or int(kv.shape[-1]) != row_bytes
    ):
        side = "FP8 (ori)" if name != "cmp_kv" else "FP4 (cmp)"
        raise ValueError(
            f"Invalid input: {name} must be a uint8 [blocknum, blocksize, 1, {row_bytes}] "
            f"tensor ({side} byte view); got shape {tuple(kv.shape)}, "
            f"dtype {kv.dtype}"
        )
    if int(kv.shape[0]) <= 0 or int(kv.shape[1]) <= 0:
        raise ValueError(f"Invalid input: {name} must be non-empty")
    if (
        block_table.dtype != torch.int32
        or block_table.dim() != _TENSOR_RANK_2D
        or int(block_table.shape[0]) <= 0
        or int(block_table.shape[1]) <= 0
    ):
        raise ValueError(
            f"Invalid input: {name}_block_table must be a non-empty int32 "
            f"[batch, num_blocks] tensor"
        )
    bs = int(kv.shape[1])
    strides = tuple(kv.stride())
    page_stride = int(
        kv.stride(0)
    )  # The uint8 page stride is measured in bytes, independently of page length.
    if strides[1:] != (N2 * row_bytes, row_bytes, 1):
        raise ValueError(
            f"Invalid input: {name} only supports axis-0 non-contiguity; "
            f"page dimensions must be contiguous, got strides {strides}"
        )
    if page_stride < bs * N2 * row_bytes:
        raise ValueError(
            f"Invalid input: {name} page stride must be at least {bs * N2 * row_bytes} "
            "bytes (positive, non-overlapping pages)"
        )
    if bs > _MAX_PA_BLOCK_SIZE:
        raise ValueError("PA block size must be in 1..1024")


def _validate_indices(idx, *, name, rows):
    if not isinstance(idx, torch.Tensor) or idx.dtype != torch.int32:
        raise ValueError(f"Invalid input: {name} must be an int32 torch.Tensor")
    if (
        idx.dim() != _TENSOR_RANK_3D
        or int(idx.shape[0]) != rows
        or int(idx.shape[1]) != N2
    ):
        raise ValueError(
            f"Invalid input: {name} must be int32 with shape [T1, 1, K] "
            f"(one row per query token, {rows} rows)"
        )
    k = int(idx.shape[2])
    if k <= 0:
        raise ValueError(f"Invalid input: {name} must have at least one index column")


def _validate_lengths(x, *, name, rows):
    if x is None:
        return
    if not isinstance(x, torch.Tensor) or x.dtype != torch.int32:
        raise ValueError(f"Invalid input: {name} must be an int32 torch.Tensor")
    if x.dim() != _TENSOR_RANK_2D or int(x.shape[0]) != rows or int(x.shape[1]) != N2:
        raise ValueError(f"Invalid input: {name} must be int32 with shape [T1, 1]")


def construct_workspace(metadata, addr_bytes=0):
    # Query the same device core count as metadata; workspace capacity determines the launch grid.
    blocks = _get_cube_core_num(metadata.device)
    if not 0 < blocks <= AIC_CORE_MAX_NUM:
        raise ValueError(
            f"Invalid input: core count must be in 1..{AIC_CORE_MAX_NUM}, got {blocks}"
        )
    ws_bytes = blocks * WS_BYTES_PER_CORE
    # Allocation only: avoid a zeros device op during ACLGraph capture.
    # FD slots are fully written by their staging core before publication.
    # Append the prepass row-index table to the same device allocation.
    # The kernel writes and reads the table without host staging or H2D copies.
    return torch.empty(ws_bytes + addr_bytes, dtype=torch.uint8, device=metadata.device)


# Preserve the class launcher's device symbol for existing profiler filters.
def MqsmlaKernel___call__(
    out_gm: Tensor,
    q_gm: Tensor,
    sinks_gm: Tensor,
    pa_phys_gm: Tensor,
    pa_bt_gm: Tensor,
    ws_gm: Tensor,
    pa_phys_cmp_gm: Tensor,
    pa_bt_cmp_gm: Tensor,
    cmp_idx_gm: Tensor,
    ori_idx_gm: Tensor,
    cu_seqlens_q: Tensor,
    softmax_lse: Tensor,
    ori_topk_length: Tensor,
    cmp_topk_length: Tensor,
    metadata_gm: Tensor,
    softmax_scale: dtypes.float32,
    batch_consistency: dtypes.int32,
    addr_ws_gm: Tensor,
):
    # Construct the pipeline inside the device trace, as the class launcher does.
    op = MqsmlaKernel(N1, N1 // 2, TILE_N, pa_phys_cmp_gm is not None)
    op(
        out_gm,
        q_gm,
        sinks_gm,
        pa_phys_gm,
        pa_bt_gm,
        ws_gm,
        pa_phys_cmp_gm,
        pa_bt_cmp_gm,
        cmp_idx_gm,
        ori_idx_gm,
        cu_seqlens_q,
        softmax_lse,
        ori_topk_length,
        cmp_topk_length,
        metadata_gm,
        softmax_scale,
        batch_consistency,
        addr_ws_gm,
    )


@lru_cache(maxsize=None)
def _profiled_mqsmla_kernel(has_cmp, has_ori_length, has_cmp_length, has_lse):
    # Only present Tensor arguments can be reported. Scratch buffers are internal.
    inputs = (
        "q_gm",
        "sinks_gm",
        "pa_phys_gm",
        "pa_bt_gm",
        "ori_idx_gm",
        "cu_seqlens_q",
        "metadata_gm",
    )
    if has_cmp:
        inputs += ("pa_phys_cmp_gm", "pa_bt_cmp_gm", "cmp_idx_gm")
    if has_ori_length:
        inputs += ("ori_topk_length",)
    if has_cmp_length:
        inputs += ("cmp_topk_length",)
    outputs = ("out_gm", "softmax_lse") if has_lse else ("out_gm",)
    return kernel(
        MqsmlaKernel___call__,
        profile=ProfileSpec(
            name="mixed_quant_sparse_flash_mla", inputs=inputs, outputs=outputs
        ),
    )


@host
def _run_mqsmla(
    q_gm: Tensor,
    ori_kv_gm: Tensor,
    ori_idx_gm: Tensor,
    ori_bt_gm: Tensor,
    ori_len_gm: Tensor,
    out_gm: Tensor,
    sinks_gm: Tensor,
    cu_seqlens_q: Tensor,
    metadata_gm: Tensor,
    workspace_gm: Tensor,
    cmp_kv_gm: Tensor = None,
    cmp_idx_gm: Tensor = None,
    cmp_bt_gm: Tensor = None,
    cmp_len_gm: Tensor = None,
    lse_gm: Tensor = None,
    softmax_scale: dtypes.float32 = 1.0,
    batch_consistency: dtypes.int32 = 0,
    addr_ws_gm: Tensor = None,
):
    """Adapt dynamic tensor views and launch the device kernel from a JIT entry."""
    # Adapt DSL views during tracing without torch data conversions.
    # Flatten Q/O to [T1*64,512], indices to [T1,K], lengths to [T1],
    # cu_q to [1,B+1], sinks to [1,64], and LSE to [T1*64].
    # Keep KV byte-addressed so physical page padding cannot become logical tokens.
    m_rows = q_gm.shape[0] * q_gm.shape[1]
    q_flat = q_gm.view(m_rows, D)
    out_flat = out_gm.view(m_rows, D)
    ori_pool = ori_kv_gm
    ori_idx_flat = ori_idx_gm.view(
        ori_idx_gm.shape[0] * ori_idx_gm.shape[1], ori_idx_gm.shape[2]
    )
    ori_len_flat = None
    if const_expr(ori_len_gm is not None):
        ori_len_flat = ori_len_gm.view(ori_len_gm.shape[0] * ori_len_gm.shape[1])
    cu_q = cu_seqlens_q.view(1, cu_seqlens_q.shape[0])
    sinks_k = sinks_gm.view(1, q_gm.shape[1])
    lse_flat = None
    if const_expr(lse_gm is not None):
        lse_flat = lse_gm.view(lse_gm.shape[1] * lse_gm.shape[2])
    cmp_pool = None
    cmp_idx_flat = None
    cmp_len_flat = None
    if const_expr(cmp_kv_gm is not None):
        cmp_pool = cmp_kv_gm
        cmp_idx_flat = cmp_idx_gm.view(
            cmp_idx_gm.shape[0] * cmp_idx_gm.shape[1], cmp_idx_gm.shape[2]
        )
        if const_expr(cmp_len_gm is not None):
            cmp_len_flat = cmp_len_gm.view(cmp_len_gm.shape[0] * cmp_len_gm.shape[1])
    op = _profiled_mqsmla_kernel(
        cmp_kv_gm is not None,
        ori_len_gm is not None,
        cmp_len_gm is not None,
        lse_gm is not None,
    )
    # Each core has two FD staging slots; workspace capacity determines the dynamic launch grid.
    blocks = workspace_gm.shape[0] // WS_BYTES_PER_CORE
    op[blocks](
        out_flat,
        q_flat,
        sinks_k,
        ori_pool,
        ori_bt_gm,
        workspace_gm,
        cmp_pool,
        cmp_bt_gm,
        cmp_idx_flat,
        ori_idx_flat,
        cu_q,
        lse_flat,
        ori_len_flat,
        cmp_len_flat,
        metadata_gm,
        softmax_scale,
        batch_consistency,
        addr_ws_gm,
    )


def clear_caches():
    """Close cached executables."""
    with _COMPILED_KERNEL_LOCK:
        for compiled in _COMPILED_KERNELS.values():
            compiled.close()
        _COMPILED_KERNELS.clear()


def _select_addr_mode(ori_kv, ori_bt, ori_idx, cmp_kv, cmp_bt, cmp_idx):
    """Select staging and address units; K/page widths remain dynamic in AOT."""
    if not _addr_vec_eligible(ori_kv, ori_bt, ori_idx, cmp_kv, cmp_bt, cmp_idx):
        return "off"
    has_cmp = cmp_kv is not None
    local_fits = (
        ori_bt.shape[1] <= _ADDR_COMPACT_BT
        and ori_idx.shape[2] <= _ADDR_LOCAL_ORI_K
        and (not has_cmp or cmp_bt.shape[1] <= _ADDR_COMPACT_BT)
        and (not has_cmp or cmp_idx.shape[2] <= _ADDR_LOCAL_CMP_K)
    )
    pipe_fits = (
        ori_bt.shape[1] <= _ADDR_PIPE_BT
        and ori_idx.shape[2] <= _ADDR_PIPE_ORI_K
        and (not has_cmp or cmp_bt.shape[1] <= _ADDR_PIPE_BT)
        and (not has_cmp or cmp_idx.shape[2] <= _ADDR_PIPE_CMP_K)
    )
    if local_fits and ori_idx.shape[0] <= _ADDR_LOCAL_MAX_ROWS:
        return "local"
    if pipe_fits:
        return "pipe"
    return "local" if local_fits else "off"


def _get_compiled_kernel(
    ori_kv,
    ori_sparse_indices,
    ori_block_table,
    ori_topk_length,
    cmp_kv=None,
    cmp_sparse_indices=None,
    cmp_block_table=None,
    cmp_topk_length=None,
    lse=None,
):
    """Cache dynamic executables by optional inputs and address representation."""
    addr_mode = _select_addr_mode(
        ori_kv,
        ori_block_table,
        ori_sparse_indices,
        cmp_kv,
        cmp_block_table,
        cmp_sparse_indices,
    )
    return _get_compiled_variant(
        cmp_kv is not None,
        ori_topk_length is not None,
        cmp_topk_length is not None,
        lse is not None,
        addr_mode,
    )


def _get_compiled_variant(
    has_cmp,
    has_ori_length,
    has_cmp_length,
    has_lse,
    addr_mode,
):
    """Share compilation, caching and address-mode lifetime with AOT export."""
    if addr_mode not in _ADDR_VEC_MODES:
        raise ValueError(f"Unsupported address mode: {addr_mode}")
    key = (has_cmp, has_ori_length, has_cmp_length, has_lse, addr_mode)
    with _COMPILED_KERNEL_LOCK:
        compiled = _COMPILED_KERNELS.get(key)
        if compiled is not None:
            return compiled
        global _ADDR_VEC_OVERRIDE
        previous_mode = _ADDR_VEC_OVERRIDE
        _ADDR_VEC_OVERRIDE = addr_mode
        try:
            # CANNBotDSL resolves packaged resources through its compile hook.
            # The same signature is observed during collection and runtime.
            compiled = _compile_mqsmla(*key)
            # Publish only after compilation succeeds; failures can be retried.
            _COMPILED_KERNELS[key] = compiled
            return compiled
        finally:
            _ADDR_VEC_OVERRIDE = previous_mode


def _compile_mqsmla(
    has_cmp,
    has_ori_length,
    has_cmp_length,
    has_lse,
    addr_mode,
):
    """Declare the dynamic signature; the caller holds the compilation lock."""
    rows = Dim("T1", min=1)
    batch = Dim("B", min=1)
    ori_block_size = Dim("ORI_BS", min=1, max=_MAX_PA_BLOCK_SIZE)
    cmp_block_size = Dim("CMP_BS", min=1, max=_MAX_PA_BLOCK_SIZE)
    ori_table_width = Dim("ORI_BT", min=1)
    cmp_table_width = Dim("CMP_BT", min=1)
    ori_slots = Dim("K1", min=1)
    cmp_slots = Dim("K2", min=1)
    ori_stride = Dim("ORI_STRIDE", min=1)
    cmp_stride = Dim("CMP_STRIDE", min=1)
    # Divisibility lets the kernel recover its runtime launch grid.
    workspace_bytes = Dim(
        "WORKSPACE_BYTES", min=WS_BYTES_PER_CORE, multiple_of=WS_BYTES_PER_CORE
    )
    i32 = dtypes.int32
    # Argument order matches _run_mqsmla, including absent optional tensors.
    specs = (
        TensorSpec((rows, N1, D), dtypes.bfloat16),
        TensorSpec(
            (Dim("ORI_PAGES", min=1), ori_block_size, N2, KV_ROW_BYTES_ORI),
            dtypes.uint8,
            stride=(ori_stride, KV_ROW_BYTES_ORI, KV_ROW_BYTES_ORI, 1),
        ),
        TensorSpec((rows, N2, ori_slots), i32),
        TensorSpec((batch, ori_table_width), i32),
        TensorSpec((rows, N2), i32) if has_ori_length else None,
        TensorSpec((rows, N1, D), dtypes.bfloat16),
        TensorSpec((N1,), dtypes.float32),
        TensorSpec((batch + 1,), i32),
        TensorSpec((MQSMLA_METADATA_TOTAL_SIZE,), i32),
        TensorSpec((workspace_bytes,), dtypes.uint8),
        (
            TensorSpec(
                (Dim("CMP_PAGES", min=1), cmp_block_size, N2, KV_ROW_BYTES_CMP),
                dtypes.uint8,
                stride=(cmp_stride, KV_ROW_BYTES_CMP, KV_ROW_BYTES_CMP, 1),
            )
            if has_cmp
            else None
        ),
        TensorSpec((rows, N2, cmp_slots), i32) if has_cmp else None,
        TensorSpec((batch, cmp_table_width), i32) if has_cmp else None,
        TensorSpec((rows, N2), i32) if has_cmp_length else None,
        TensorSpec((N2, rows, N1), dtypes.float32) if has_lse else None,
        dtypes.float32,
        i32,
        (
            TensorSpec((Dim("ADDR_WS_BYTES", min=1),), dtypes.uint8)
            if addr_mode != "off"
            else None
        ),
    )
    return cannbotdsl.compile(_run_mqsmla, *specs)


def mixed_quant_sparse_flash_mla(
    q,
    *,
    ori_kv=None,
    cmp_kv=None,
    ori_sparse_indices=None,
    cmp_sparse_indices=None,
    ori_block_table=None,
    cmp_block_table=None,
    cu_seqlens_q=None,
    seqused_q=None,
    seqused_ori_kv=None,
    seqused_cmp_kv=None,
    ori_topk_length=None,
    cmp_topk_length=None,
    sinks=None,
    metadata=None,
    quant_mode,
    softmax_scale=None,
    layout_q="TND",
    layout_kv="PA_BBND",
    return_softmax_lse=False,
    out,
    lse,
):
    _validate_inputs(
        q,
        ori_kv=ori_kv,
        cmp_kv=cmp_kv,
        ori_sparse_indices=ori_sparse_indices,
        cmp_sparse_indices=cmp_sparse_indices,
        ori_block_table=ori_block_table,
        cmp_block_table=cmp_block_table,
        cu_seqlens_q=cu_seqlens_q,
        seqused_q=seqused_q,
        seqused_ori_kv=seqused_ori_kv,
        seqused_cmp_kv=seqused_cmp_kv,
        ori_topk_length=ori_topk_length,
        cmp_topk_length=cmp_topk_length,
        sinks=sinks,
        metadata=metadata,
        quant_mode=quant_mode,
        layout_q=layout_q,
        layout_kv=layout_kv,
    )
    for name, tensor, shape, dtype in (
        ("out", out, tuple(q.shape), torch.bfloat16),
        ("lse", lse, (N2, q.shape[0], q.shape[1]), torch.float32),
    ):
        if name == "lse" and not return_softmax_lse:
            shape = (0,)
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"Invalid input: {name} must be a caller-allocated tensor")
        if (
            tuple(tensor.shape) != shape
            or tensor.dtype != dtype
            or not tensor.is_contiguous()
        ):
            raise ValueError(
                f"Invalid input: {name} must be contiguous {dtype} {shape}"
            )
    # An absent length tensor activates all sparse-index columns; the kernel
    # reads K from the index shape without allocating a replacement tensor.
    addr_ws = None
    addr_bytes = 0
    addr_mode = _select_addr_mode(
        ori_kv,
        ori_block_table,
        ori_sparse_indices,
        cmp_kv,
        cmp_block_table,
        cmp_sparse_indices,
    )
    if addr_mode != "off":
        columns = _addr_tab_w(ori_sparse_indices.shape[2])
        if cmp_kv is not None:
            columns += _addr_tab_w(cmp_sparse_indices.shape[2])
        addr_bytes = q.shape[0] * columns * 4
        if addr_mode == "local":
            # 分核副本由表视图起点区分；两 AIV 各写完整逻辑行中的互斥列段。
            addr_bytes = (q.shape[0] + _get_cube_core_num(q.device)) * columns * 4
    # Allocate FD staging and the address table together.
    workspace_full = construct_workspace(metadata, addr_bytes)
    if addr_bytes:
        workspace = workspace_full[:-addr_bytes]
        addr_ws = workspace_full[-addr_bytes:]
    else:
        workspace = workspace_full
    tensors = (
        q,
        ori_kv,
        ori_sparse_indices,
        ori_block_table,
        ori_topk_length,
        out,
        sinks,
        cu_seqlens_q,
        metadata,
        workspace,
        cmp_kv,
        cmp_sparse_indices,
        cmp_block_table,
        cmp_topk_length,
        lse if return_softmax_lse else None,
    )
    compiled = _get_compiled_kernel(
        ori_kv,
        ori_sparse_indices,
        ori_block_table,
        ori_topk_length,
        cmp_kv,
        cmp_sparse_indices,
        cmp_block_table,
        cmp_topk_length,
        lse if return_softmax_lse else None,
    )
    # AOT fixes absent Tensor parameters at compile time and removes them from
    # the runtime signature. Remaining Tensor arguments keep their order.
    scale = (
        float(softmax_scale)
        if softmax_scale is not None
        else 1.0 / math.sqrt(q.shape[-1])
    )
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("softmax_scale must be finite and positive")
    batch_consistency = 1 if _batch_consistency_enabled() else 0
    compiled(
        *(tensor for tensor in tensors if tensor is not None),
        scale,
        batch_consistency,
        *([addr_ws] if addr_ws is not None else []),
    )


@export("mqsmla")
def export_mqsmla():
    """Collect all 36 optional-input/addressing variants with dynamic shapes."""
    clear_caches()
    try:
        for has_cmp in (True, False):
            cmp_lengths = (True, False) if has_cmp else (False,)
            for has_ori_length, has_cmp_length, has_lse, addr_mode in product(
                (True, False),
                cmp_lengths,
                (True, False),
                _ADDR_VEC_MODES,
            ):
                _get_compiled_variant(
                    has_cmp,
                    has_ori_length,
                    has_cmp_length,
                    has_lse,
                    addr_mode,
                )
    finally:
        clear_caches()
