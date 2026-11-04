# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import tempfile
import threading

from cannbotdsl.aicpu import gm_in, gm_out, I32, I64, aicpu_kernel, current_raw_stream

AIC_CORE_MAX_NUM = 36
AIV_CORE_MAX_NUM = 72
MQSMLA_METADATA_TOTAL_SIZE = 1024

FA_METADATA_SIZE = 9
FD_METADATA_SIZE = 8
FD_METADATA_BASE = AIC_CORE_MAX_NUM * FA_METADATA_SIZE  # 324
# The first reserved word after FA[36][9] and FD[72][8] stores fdUsedVecNum.
# The consumer uses this uniform count to gate the barrier and FD reduction.
FD_USED_VEC_NUM_WORD = (
    AIC_CORE_MAX_NUM * FA_METADATA_SIZE + AIV_CORE_MAX_NUM * FD_METADATA_SIZE
)  # 900

FA_CORE_ENABLE_INDEX = 0
FA_BN2_START_INDEX = 1
FA_M_START_INDEX = 2
FA_S2_START_INDEX = 3
FA_BN2_END_INDEX = 4
FA_M_END_INDEX = 5
FA_S2_END_INDEX = 6
FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX = 7

FD_CORE_ENABLE_INDEX = 0
FD_BN2_IDX_INDEX = 1
FD_M_IDX_INDEX = 2
FD_WORKSPACE_IDX_INDEX = 3
FD_WORKSPACE_NUM_INDEX = 4
FD_M_START_INDEX = 5
FD_M_NUM_INDEX = 6

FA_TOLERANCE_RATIO = 2


# Metadata owns the FD policy. SUPPORT_FD enables the general cost-based planner;
# SELECTIVE_FD_72 independently enables the exact 72-row plan. All other inputs
# use the general planner when SUPPORT_FD is enabled. The attention AICore
# binary supports all three plans.
SUPPORT_FD = True
# Require at least four 128-token tiles of modeled critical-path saving.
# Short-row splits do not amortize partial writes, the barrier and reduction.
FD_MIN_SAVED_TILES = 4
# Target four tiles per FA core: one-tile parts make the serial FD reducer
# more expensive than the work removed from a long, otherwise single-core row.
FD_TARGET_TILES_PER_CORE = 4
# 短行保留完整 online softmax，避免把少量有效列拆成独立 BF16 P 舍入基准。
# 长行仍可 FD；固定五 tile 的 selective plan 不受此限制影响。
FD_MIN_ROW_TILES = 5
SELECTIVE_FD_72 = True  # Eight groups of nine rows; three split rows per group.


_COST_FULL_TILE = 44
_COST_M_PART = 24

_METADATA_COMPILED = None
_METADATA_DIRECTORY = None
_METADATA_LOCK = threading.Lock()


def _get_cube_core_num(device=None):
    """Return the cube-core count of the active NPU (same source as the consumer)."""
    import torch

    return int(torch.npu.get_device_properties(device).cube_core_num)


# AICPU planner: preserve the fixed 1024 batch-relative FA/FD metadata contract.
# SUPPORT_FD is passed at launch; no input values are copied to the host.
# Whole-row mode uses uniform partition. FD mode uses the
# AscendC cost-greedy row/tile scheme and serializes cross-core reduction tasks.


def _md_uniform(a):
    # Initialize every metadata field consumed by attention, including unused FA/FD entries.
    for i in range(0, MQSMLA_METADATA_TOTAL_SIZE):
        a.metadata[i] = 0
    rows_per_core = a.rows // a.blocks
    extra = a.rows % a.blocks
    start = 0
    batch1 = a.batch + 1
    for core in range(0, a.blocks):
        count = rows_per_core
        if core < extra:
            count += 1
        end = start + count
        if count > 0:
            bn2_s = 0
            for b in range(0, batch1):
                if a.cu_q[b] <= start:
                    bn2_s = b
            m_s = start - a.cu_q[bn2_s]
            bn2_e = 0
            for b in range(0, batch1):
                if a.cu_q[b] <= end:
                    bn2_e = b
            m_e = end - a.cu_q[bn2_e]
            fa = FA_METADATA_SIZE * core
            a.metadata[fa + FA_CORE_ENABLE_INDEX] = 1
            a.metadata[fa + FA_BN2_START_INDEX] = bn2_s
            a.metadata[fa + FA_M_START_INDEX] = m_s
            a.metadata[fa + FA_BN2_END_INDEX] = bn2_e
            a.metadata[fa + FA_M_END_INDEX] = m_e
        start = end
    return 0


class _MqsmlaMetadataArgs:
    win_len: gm_in(I32)  # [T1] valid ORI lengths.
    cmp_len: gm_in(I32)  # [T1] valid CMP lengths; ignored when has_cmp=0.
    metadata: gm_out(I32)  # Fixed int32[1024] FA/FD layout.
    rows: I32  # T1
    blocks: I32  # AIC launch count, matching workspace capacity.
    has_cmp: I32
    cu_q: gm_in(I32)
    batch: I32
    support_fd: I32


# AICPU planner adapted from AscendC mixed_quant_sparse_flash_mla_metadata.
# Work coordinates are (global query row, row-local S2 tile).
# Tile cost = 6*ceil(M/16) + 10*ceil(S2/64), with M=64.
# Greedily assign rows with a half-tail-tile tolerance, then individual tiles
# when FD is allowed; force one tile if needed to guarantee progress.
# A core ending inside a row creates a deferred FD task.
# N2=1 and TND remove the BN2 loop; two AIVs split each 64-head FD task.


def _md_row_tiles(w, wc):
    n = 0
    if w > 0:
        n = (w + 127) // 128
    if wc > 0:
        n = n + (wc + 127) // 128
    return n


def _md_row_cost(w, wc):
    cost = 0
    if w > 0:
        cost = _COST_FULL_TILE * (w // 128)
        t = w % 128
        if t != 0:
            cost = cost + _COST_M_PART + 10 * ((t + 63) // 64)
    if wc > 0:
        costc = _COST_FULL_TILE * (wc // 128)
        tc = wc % 128
        if tc != 0:
            costc = costc + _COST_M_PART + 10 * ((tc + 63) // 64)
        cost = cost + costc
    return cost


def _md_row_last(w, wc):
    # Cost of the final tile; CMP follows ORI when present.
    if wc > 0:
        t = wc % 128
        if t == 0:
            return _COST_FULL_TILE
        return _COST_M_PART + 10 * ((t + 63) // 64)
    if w > 0:
        t = w % 128
        if t == 0:
            return _COST_FULL_TILE
        return _COST_M_PART + 10 * ((t + 63) // 64)
    return 0


def _md_tile_cost(w, wc, t):
    # Cost of row-local tile t. Empty rows can follow a completed split row;
    # return zero so the tile loop can advance across them.
    tw = 0
    if w > 0:
        tw = (w + 127) // 128
    if t < tw:
        wt = w % 128
        if wt != 0 and t == tw - 1:
            return _COST_M_PART + 10 * ((wt + 63) // 64)
        return _COST_FULL_TILE
    tc = 0
    if wc > 0:
        tc = (wc + 127) // 128
    if tw + tc == 0:
        return 0
    c = t - tw
    ct = wc % 128
    if ct != 0 and c == tc - 1:
        return _COST_M_PART + 10 * ((ct + 63) // 64)
    return _COST_FULL_TILE


def _md_selective_fd_72(a):
    # Exact 72 x 5-tile contiguous plan: eight 9-row groups on four cores
    # (11/11/11/12 tiles). Three local split rows per group.
    if a.has_cmp != 1:
        return _md_uniform(a)
    for row in range(0, 72):
        if a.win_len[row] != 128 or a.cmp_len[row] != 512:
            return _md_uniform(a)
    for i in range(0, FD_USED_VEC_NUM_WORD + 51):
        a.metadata[i] = 0
    for core in range(0, 32):
        group = core // 4
        local = core % 4
        base_tile = group * 45
        start_tile = base_tile
        end_tile = base_tile + 11
        first_fd = group * 6
        if local == 1:
            start_tile = base_tile + 11
            end_tile = base_tile + 22
            first_fd = first_fd + 1
        if local == 2:
            start_tile = base_tile + 22
            end_tile = base_tile + 33
            first_fd = first_fd + 3
        if local == 3:
            start_tile = base_tile + 33
            end_tile = base_tile + 45
            first_fd = first_fd + 5
        start = start_tile // 5
        end = end_tile // 5
        bs = 0
        be = 0
        for bi in range(0, a.batch + 1):
            if a.cu_q[bi] <= start:
                bs = bi
            if a.cu_q[bi] <= end:
                be = bi
        base = core * FA_METADATA_SIZE
        a.metadata[base + FA_CORE_ENABLE_INDEX] = 1
        a.metadata[base + FA_BN2_START_INDEX] = bs
        a.metadata[base + FA_M_START_INDEX] = start - a.cu_q[bs]
        a.metadata[base + FA_S2_START_INDEX] = start_tile % 5
        a.metadata[base + FA_BN2_END_INDEX] = be
        a.metadata[base + FA_M_END_INDEX] = end - a.cu_q[be]
        a.metadata[base + FA_S2_END_INDEX] = end_tile % 5
        a.metadata[base + FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX] = first_fd
    for task in range(0, 24):
        group = task // 3
        which = task % 3
        row = group * 9 + 2 + 2 * which
        reducer_core = group * 4 + which
        batch = 0
        for bi in range(0, a.batch + 1):
            if a.cu_q[bi] <= row:
                batch = bi
        for half in range(0, 2):
            base = FD_METADATA_BASE + (2 * reducer_core + half) * FD_METADATA_SIZE
            a.metadata[base + FD_CORE_ENABLE_INDEX] = 1
            a.metadata[base + FD_BN2_IDX_INDEX] = batch
            a.metadata[base + FD_M_IDX_INDEX] = row - a.cu_q[batch]
            a.metadata[base + FD_WORKSPACE_IDX_INDEX] = 2 * task
            a.metadata[base + FD_WORKSPACE_NUM_INDEX] = 2
            a.metadata[base + FD_M_START_INDEX] = 32 * half
            a.metadata[base + FD_M_NUM_INDEX] = 32
    a.metadata[FD_USED_VEC_NUM_WORD] = 48
    a.metadata[FD_USED_VEC_NUM_WORD + 50] = 1
    return 0


@aicpu_kernel
def _mqsmla_metadata_kernel(a: _MqsmlaMetadataArgs):
    if a.rows <= 0 or a.blocks <= 0 or a.blocks > AIC_CORE_MAX_NUM:
        return 1
    if a.support_fd >= 2:
        selective = 0
        if a.rows == 72 and a.blocks == 32 and a.has_cmp == 1:
            selective = 1
            for row in range(0, 72):
                if a.win_len[row] != 128 or a.cmp_len[row] != 512:
                    selective = 0
        if selective == 1:
            return _md_selective_fd_72(a)
        if a.support_fd == 2:
            return _md_uniform(a)
    if a.support_fd == 0:
        return _md_uniform(a)
    # zeros is an AICPU DSL intrinsic lowered by the compiler.
    fa_m_start = zeros(I64, AIC_CORE_MAX_NUM)  # noqa: F821
    fa_s2_start = zeros(I64, AIC_CORE_MAX_NUM)  # noqa: F821
    fa_m_end = zeros(I64, AIC_CORE_MAX_NUM)  # noqa: F821
    fa_s2_end = zeros(I64, AIC_CORE_MAX_NUM)  # noqa: F821
    fa_first_fd = zeros(I64, AIC_CORE_MAX_NUM)  # noqa: F821
    fd_row = zeros(I64, AIC_CORE_MAX_NUM)  # noqa: F821
    fd_ws = zeros(I64, AIC_CORE_MAX_NUM)  # noqa: F821
    fd_num = zeros(I64, AIC_CORE_MAX_NUM)  # noqa: F821

    for i in range(0, MQSMLA_METADATA_TOTAL_SIZE):
        a.metadata[i] = 0

    # Pass 1: total cost, tile count and maximum tiles per row.
    total_cost = 0
    total_tiles = 0
    max_row_tiles = 0
    r = 0
    while r < a.rows:
        w = a.win_len[r]
        wc = 0
        if a.has_cmp == 1:
            wc = a.cmp_len[r]
        rc = _md_row_cost(w, wc)
        rt = _md_row_tiles(w, wc)
        total_cost = total_cost + rc
        total_tiles = total_tiles + rt
        if rt > max_row_tiles:
            max_row_tiles = rt
        r = r + 1

    if max_row_tiles < FD_MIN_ROW_TILES:
        return _md_uniform(a)

    # Limit split fanout while retaining at least whole-row parallelism.
    plan_blocks = total_tiles // FD_TARGET_TILES_PER_CORE
    if plan_blocks < a.rows:
        plan_blocks = a.rows
    if plan_blocks > a.blocks:
        plan_blocks = a.blocks
    # Require enough modeled tile savings to amortize staging, synchronization and reduction.
    ceil_rows_blocks = (a.rows + a.blocks - 1) // a.blocks
    fd_floor = (total_tiles + plan_blocks - 1) // plan_blocks
    uniform_crit = ceil_rows_blocks * max_row_tiles
    # Even fewer tiles than cores benefit from splitting a long single row.
    # Gate on saved work, not total occupancy; short balanced rows retain the
    # whole-row path to avoid unnecessary staging and reduction.
    if uniform_crit - fd_floor < FD_MIN_SAVED_TILES:
        return _md_uniform(a)

    # Pass 2: greedy core assignment, following CalcSplitPlan/AssignBlocksToCore.
    cur_row = 0
    cur_tile = 0
    finished = 0
    wv = a.win_len[0]
    wcv = 0
    if a.has_cmp == 1:
        wcv = a.cmp_len[0]
    tiles_c = _md_row_tiles(wv, wcv)
    rem_cost = _md_row_cost(wv, wcv)
    last_c = _md_row_last(wv, wcv)
    unassigned = total_cost
    kv_split = 1  # Parts in the currently open split row.
    pre_fd = 0  # Workspace slots occupied by recorded FD tasks.
    num_fd = 0
    used = 0
    core = 0
    while core < plan_blocks:
        if finished == 1:
            break
        # Do not stop when unassigned cost reaches zero: trailing empty rows
        # still need an enabled core to write their outputs.
        fa_first_fd[core] = pre_fd + kv_split - 1
        fa_m_start[core] = cur_row
        fa_s2_start[core] = cur_tile
        limit = unassigned // (plan_blocks - core)
        cost = 0
        blk = 0
        # Assign whole rows, advancing directly past zero-cost empty rows.
        while finished == 0:
            if (
                tiles_c == 0
                or limit + last_c // FA_TOLERANCE_RATIO >= cost + rem_cost
                or (cur_tile == 0 and tiles_c < FD_MIN_ROW_TILES and blk == 0)
            ):
                cost = cost + rem_cost
                blk = blk + (tiles_c - cur_tile)
                cur_row = cur_row + 1
                cur_tile = 0
                if cur_row >= a.rows:
                    finished = 1
                else:
                    wv = a.win_len[cur_row]
                    wcv = 0
                    if a.has_cmp == 1:
                        wcv = a.cmp_len[cur_row]
                    tiles_c = _md_row_tiles(wv, wcv)
                    rem_cost = _md_row_cost(wv, wcv)
                    last_c = _md_row_last(wv, wcv)
            else:
                break
        # The FD gate has passed. Half-tile tolerance
        # keeps a split boundary inside the row, matching the AscendC invariant.
        while finished == 0:
            if cur_tile == 0 and tiles_c < FD_MIN_ROW_TILES:
                break
            tcost = _md_tile_cost(wv, wcv, cur_tile)
            if limit + tcost // FA_TOLERANCE_RATIO >= cost + tcost:
                cost = cost + tcost
                blk = blk + 1
                rem_cost = rem_cost - tcost
                cur_tile = cur_tile + 1
                if cur_tile >= tiles_c:
                    cur_row = cur_row + 1
                    cur_tile = 0
                    if cur_row >= a.rows:
                        finished = 1
                    else:
                        wv = a.win_len[cur_row]
                        wcv = 0
                        if a.has_cmp == 1:
                            wcv = a.cmp_len[cur_row]
                        tiles_c = _md_row_tiles(wv, wcv)
                        rem_cost = _md_row_cost(wv, wcv)
                        last_c = _md_row_last(wv, wcv)
            else:
                break
        # Force one tile to guarantee forward progress.
        if blk == 0 and finished == 0:
            tcost = _md_tile_cost(wv, wcv, cur_tile)
            cost = cost + tcost
            blk = 1
            rem_cost = rem_cost - tcost
            cur_tile = cur_tile + 1
            if cur_tile >= tiles_c:
                cur_row = cur_row + 1
                cur_tile = 0
                if cur_row >= a.rows:
                    finished = 1
                else:
                    wv = a.win_len[cur_row]
                    wcv = 0
                    if a.has_cmp == 1:
                        wcv = a.cmp_len[cur_row]
                    tiles_c = _md_row_tiles(wv, wcv)
                    rem_cost = _md_row_cost(wv, wcv)
                    last_c = _md_row_last(wv, wcv)
        fa_m_end[core] = cur_row
        fa_s2_end[core] = cur_tile
        unassigned = unassigned - cost
        # Record a deferred FD task when the cursor passes the previous split row.
        if core > 0:
            if kv_split > 1:
                if cur_row != fa_m_end[core - 1]:
                    fd_row[num_fd] = fa_m_end[core - 1]
                    fd_ws[num_fd] = pre_fd
                    fd_num[num_fd] = kv_split
                    num_fd = num_fd + 1
                    pre_fd = pre_fd + kv_split
                    kv_split = 1
        # An unfinished row at this core boundary adds one pending partial.
        if cur_tile > 0:
            kv_split = kv_split + 1
        used = core + 1
        core = core + 1

    # Serialize the plan into the fixed 1024-word metadata layout.
    i = 0
    while i < a.blocks:
        base = i * FA_METADATA_SIZE
        if i < used:
            a.metadata[base + FA_CORE_ENABLE_INDEX] = 1
            batch_start = 0
            batch_end = 0
            for batch_index in range(0, a.batch + 1):
                if a.cu_q[batch_index] <= fa_m_start[i]:
                    batch_start = batch_index
                if a.cu_q[batch_index] <= fa_m_end[i]:
                    batch_end = batch_index
            a.metadata[base + FA_BN2_START_INDEX] = batch_start
            a.metadata[base + FA_BN2_END_INDEX] = batch_end
            a.metadata[base + FA_M_START_INDEX] = fa_m_start[i] - a.cu_q[batch_start]
            a.metadata[base + FA_S2_START_INDEX] = fa_s2_start[i]
            a.metadata[base + FA_M_END_INDEX] = fa_m_end[i] - a.cu_q[batch_end]
            a.metadata[base + FA_S2_END_INDEX] = fa_s2_end[i]
            a.metadata[base + FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX] = fa_first_fd[i]
        i = i + 1
    # FD task t maps to AIVs 2*t and 2*t+1, each handling 32 heads.
    # At most used-1 tasks require fewer than 2*blocks AIVs.
    j = 0
    while j < 2 * num_fd:
        base = FD_METADATA_BASE + j * FD_METADATA_SIZE
        t = j // 2
        a.metadata[base + FD_CORE_ENABLE_INDEX] = 1
        fd_batch_index = 0
        for batch_index in range(0, a.batch + 1):
            if a.cu_q[batch_index] <= fd_row[t]:
                fd_batch_index = batch_index
        a.metadata[base + FD_BN2_IDX_INDEX] = fd_batch_index
        a.metadata[base + FD_M_IDX_INDEX] = fd_row[t] - a.cu_q[fd_batch_index]
        a.metadata[base + FD_WORKSPACE_IDX_INDEX] = fd_ws[t]
        a.metadata[base + FD_WORKSPACE_NUM_INDEX] = fd_num[t]
        a.metadata[base + FD_M_START_INDEX] = (j % 2) * 32
        a.metadata[base + FD_M_NUM_INDEX] = 32
        j = j + 1
    # Publish the uniform reducer count; zero skips the FD barrier and reduction.
    a.metadata[FD_USED_VEC_NUM_WORD] = 2 * num_fd
    return 0


def _get_compiled_metadata():
    """Load or compile the mqsmla metadata AICPU kernel once per process."""
    global _METADATA_COMPILED, _METADATA_DIRECTORY
    from pathlib import Path
    from cannbotdsl.aicpu.toolchain import CompiledAicpuKernel, compile_aicpu_kernel

    with _METADATA_LOCK:
        if _METADATA_COMPILED is None:
            binary = Path(__file__).parent / "_aicpu" / "mqsmla_metadata_kernel.so"
            if binary.is_file():
                _METADATA_COMPILED = CompiledAicpuKernel(
                    str(binary),
                    _mqsmla_metadata_kernel.spec,
                    name=_mqsmla_metadata_kernel.fn.__name__,
                    launch_mode="interface",
                )
                return _METADATA_COMPILED
            _METADATA_DIRECTORY = tempfile.TemporaryDirectory(
                prefix="mqsmla_metadata_aicpu_"
            )
            _METADATA_COMPILED = compile_aicpu_kernel(
                _mqsmla_metadata_kernel,
                workdir=_METADATA_DIRECTORY.name,
                launch_mode="interface",
            )
        return _METADATA_COMPILED


def mixed_quant_sparse_flash_mla_metadata(
    ori_topk_length,
    cmp_topk_length,
    *,
    cu_seqlens_q=None,
    seqused_q=None,
    seqused_ori_kv=None,
    seqused_cmp_kv=None,
    batch_size=None,
    max_seqlen_q=None,
    max_seqlen_ori_kv=None,
    max_seqlen_cmp_kv=None,
    num_heads_q,
    num_heads_kv,
    head_dim,
    quant_mode,
    layout_q="TND",
    layout_kv="PA_BBND",
    has_ori_kv=True,
    has_cmp_kv=True,
):
    """Generate batch-relative FA/FD metadata on AICPU.

    SELECTIVE_FD_72 selects the exact 72-row plan when eligible. Otherwise
    SUPPORT_FD enables cost-based splitting, or False selects whole rows.
    fd_used_vec_num controls the attention kernel's reduction barrier.
    """
    import torch

    for name, value in (
        ("ori_topk_length", ori_topk_length),
        ("cmp_topk_length", cmp_topk_length),
    ):
        if (
            not isinstance(value, torch.Tensor)
            or value.dtype != torch.int32
            or value.dim() != 2
            or value.shape[1] != num_heads_kv
            or value.shape[0] <= 0
        ):
            raise ValueError(
                f"Invalid input: {name} must be an int32 tensor of shape [T1, N2]"
            )
        if not value.is_contiguous():
            raise ValueError(
                f"Invalid input: {name} must be contiguous (flat AICPU indexing)"
            )
    if ori_topk_length.shape != cmp_topk_length.shape:
        raise ValueError("Invalid input: topk_length tensors must have the same shape")
    if (num_heads_q, num_heads_kv, head_dim, quant_mode) != (64, 1, 512, 1):
        raise ValueError(
            "Invalid input: metadata supports num_heads_q=64, num_heads_kv=1, head_dim=512, quant_mode=1"
        )
    if layout_q != "TND" or layout_kv != "PA_BBND" or not has_ori_kv:
        raise ValueError(
            "Invalid input: metadata supports TND / PA_BBND with has_ori_kv=True"
        )
    device = ori_topk_length.device
    if cmp_topk_length.device != device:
        raise ValueError(
            "Invalid input: topk_length tensors must live on the same device"
        )

    rows = ori_topk_length.shape[0]
    blocks = _get_cube_core_num(device)
    if not 0 < blocks <= AIC_CORE_MAX_NUM or rows > 2147483647:
        raise ValueError(
            "Invalid input: rows and cube core count must fit positive int32 metadata"
        )

    # No input tensor values are read on the host:
    # cu_seqlens_q only needs to be a well-formed [B+1] int32 span on the device.
    if cu_seqlens_q is None:
        cu_q = torch.tensor([0, rows], dtype=torch.int32, device=device)
    else:
        cu_q = cu_seqlens_q
        if (
            cu_q.dtype != torch.int32
            or cu_q.dim() != 1
            or cu_q.numel() < 2
            or cu_q.device != device
        ):
            raise ValueError(
                "Invalid input: cu_seqlens_q must be a 1-D int32 tensor on the same device"
            )
        if not cu_q.is_contiguous():
            raise ValueError("Invalid input: cu_seqlens_q must be contiguous")

    batch = cu_q.numel() - 1
    compiled = _get_compiled_metadata()
    with torch.npu.device(device):
        metadata = torch.empty(
            MQSMLA_METADATA_TOTAL_SIZE, dtype=torch.int32, device=device
        )
        compiled.launch(
            current_raw_stream(device.index),
            metadata=metadata.data_ptr(),
            cu_q=cu_q.data_ptr(),
            rows=rows,
            blocks=blocks,
            batch=batch,
            win_len=ori_topk_length.data_ptr(),
            cmp_len=cmp_topk_length.data_ptr(),
            has_cmp=int(has_cmp_kv),
            support_fd=(
                2 + int(SUPPORT_FD)
                if SELECTIVE_FD_72 and rows == 72 and blocks == 32 and has_cmp_kv
                else int(SUPPORT_FD)
            ),
        )
    return metadata
