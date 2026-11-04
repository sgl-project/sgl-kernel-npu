# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""QLI AICPU 分核：固定 int32[1024] 的 LI/LD 边界 ABI。

LI: 36 records x 8 fields: enable, start B/M/S2, end B/M/S2,
first LD workspace index. LD: 72 records x 8 fields: enable, B/M,
workspace index/count, M start/count, reserved. Remaining words are zero.
M is the operator's query tile index; S2 is a 256-token tile index.
All query groups schedule indivisible 256-token blocks and write those
indices directly. Odd lengths use a final tail.
"""

__all__ = ["quant_lightning_indexer_metadata"]

from functools import lru_cache
import tempfile
import torch
from cannbotdsl.aicpu import (
    gm_in as GmIn,
    gm_out as GmOut,
    I32,
    I64,
    U32,
    aicpu_kernel,
    current_raw_stream,
)


# LI/LD 固定输出布局；第 36 行为 LD/FD 归并记录起始位置。
METADATA_ELEMENTS = 1024
METADATA_FIELDS = 8
LD_RECORD_START = 36
LI_WORKSPACE_OFFSET_FIELD = 7
GROUP_QUERY_OFFSET_FIELD = 7
S2_TILE_TOKENS = 256


GROUP_CACHE_FIELDS = 6


GROUP_CACHE_ID = 0


GROUP_CACHE_BATCH = 1


GROUP_CACHE_OFFSET = 2


GROUP_CACHE_ACTIVE = 3


GROUP_CACHE_TILES = 4


GROUP_CACHE_COST = 5


class Args:
    cu: GmIn(I32)
    used_q: GmIn(I32)
    used_k: GmIn(I32)

    cu_k: GmIn(I32)

    residual: GmIn(I32)
    candidate_length: GmIn(I32)
    output_offset: GmIn(I32)
    output: GmOut(I32)
    scratch: GmOut(I64)
    has_cu: U32
    has_q: U32
    has_k: U32

    has_cu_k: U32

    has_residual: U32
    sparse: U32
    mask: U32
    ratio: U32
    total_q: I64
    batch: I64
    max_tasks: I64
    capacity: I64
    splits: I64
    workers: I64
    groups: I64
    query_rows: I64
    heads: I64
    ld: U32
    has_offset: U32


def _group_field(args, g, total_q, field):
    if g >= args.scratch[args.batch]:
        if field == 6:
            return args.batch
        return 0
    cache = args.batch + 1
    if args.scratch[cache + GROUP_CACHE_ID] != g:
        batch_idx = args.scratch[cache + GROUP_CACHE_BATCH]
        # 连续访问同一 batch 的分组时复用前缀区间，否则才做二分查找。
        if g < args.scratch[batch_idx] or g >= args.scratch[batch_idx + 1]:
            lo = 0
            hi = args.batch
            while lo + 1 < hi:
                mid = (lo + hi) // 2
                if args.scratch[mid] <= g:
                    lo = mid
                else:
                    hi = mid
            batch_idx = lo
        qoff = (g - args.scratch[batch_idx]) * args.query_rows
        begin = 0
        end = total_q
        if args.has_cu != 0:
            begin = args.cu[batch_idx]
            end = args.cu[batch_idx + 1]
        used = end - begin
        if args.has_q != 0:
            used = args.used_q[batch_idx]
        span = min(args.query_rows, max(0, end - begin - qoff))
        active = min(span, max(0, used - qoff))
        actual = args.capacity
        if args.has_k != 0:
            actual = args.used_k[batch_idx]
        elif args.has_cu_k != 0:
            actual = args.cu_k[batch_idx + 1] - args.cu_k[batch_idx]
        rem = 0
        if args.has_residual != 0:
            rem = args.residual[batch_idx]
        visible = actual
        if args.mask == 3:
            visible = min(
                actual,
                max(
                    0, (actual * args.ratio + rem - used + qoff + active) // args.ratio
                ),
            )
        if active == 0:
            visible = 0
        # 仅缓存查询结果，不改变调度成本模型和 LI/LD 输出。
        args.scratch[cache + GROUP_CACHE_ID] = g
        args.scratch[cache + GROUP_CACHE_BATCH] = batch_idx
        args.scratch[cache + GROUP_CACHE_OFFSET] = qoff
        args.scratch[cache + GROUP_CACHE_ACTIVE] = active
        args.scratch[cache + GROUP_CACHE_TILES] = (
            visible + S2_TILE_TOKENS - 1
        ) // S2_TILE_TOKENS
        args.scratch[cache + GROUP_CACHE_COST] = (
            6 * ((active * args.heads + 15) // 16) + 20
        )
    if field == 6:
        return args.scratch[cache + GROUP_CACHE_BATCH]
    if field == GROUP_QUERY_OFFSET_FIELD:
        return args.scratch[cache + GROUP_CACHE_OFFSET]
    if field == 2:
        return args.scratch[cache + GROUP_CACHE_ACTIVE]
    if field == 9:
        return args.scratch[cache + GROUP_CACHE_COST]
    return args.scratch[cache + GROUP_CACHE_TILES]


@aicpu_kernel
def metadata_kernel(args: Args):
    total_q = args.total_q
    if args.has_cu != 0:
        total_q = args.cu[args.batch]
    elif total_q < 0 and args.has_q != 0:
        total_q = args.used_q[0]
    # ASC GetS1SeqSize: seqused_q takes precedence over the TND span.
    # B+1 prefix entries plus six cached fields; workspace is independent of Q/K length.
    groups = 0
    args.scratch[0] = 0
    for batch_idx in range(args.batch):
        length = total_q
        if args.has_cu != 0:
            length = args.cu[batch_idx + 1] - args.cu[batch_idx]
        if args.has_q != 0:
            length = args.used_q[batch_idx]
        groups += (length + args.query_rows - 1) // args.query_rows
        args.scratch[batch_idx + 1] = groups
    args.scratch[args.batch + 1 + GROUP_CACHE_ID] = -1
    args.scratch[args.batch + 1 + GROUP_CACHE_BATCH] = 0
    workers = args.workers
    for i in range(METADATA_ELEMENTS):
        args.output[i] = 0
    total_cost = 0
    for g in range(groups):
        n = _group_field(args, g, total_q, 8)
        total_cost += n * _group_field(args, g, total_q, 9)
    # ASC AssignByBatch -> AssignByRow -> AssignByBlock -> ForceAssign.
    # The cursor is a lexicographic (B, M, S2) boundary. No per-query split cap.
    first_group = zeros(I64, workers)
    last_group = zeros(I64, workers)
    first_tile = zeros(I64, workers)
    last_tile = zeros(I64, workers)
    max_row_cost = 0
    for g in range(groups):
        n = _group_field(args, g, total_q, 8)
        if n > 0:
            max_row_cost = max(
                max_row_cost,
                (n - 1) * _group_field(args, g, total_q, 9)
                + _group_field(args, g, total_q, 9),
            )
    g = 0
    at = 0
    remaining = total_cost
    used_cores = 0
    for c in range(workers):
        if g >= groups or remaining <= 0:
            break
        first_group[c] = g
        first_tile[c] = at
        limit = remaining // (workers - c)
        if args.ld == 0:
            limit = max(limit, max_row_cost)
        used = 0
        blocks = 0
        # Assign remaining complete batches with ASC half-last-block tolerance.
        while g < groups:
            batch_idx = _group_field(args, g, total_q, 6)
            end = g
            batch_cost = 0
            batch_blocks = 0
            batch_last = 0
            while end < groups and _group_field(args, end, total_q, 6) == batch_idx:
                n = _group_field(args, end, total_q, 8)
                start = 0
                if end == g:
                    start = at
                if n > start:
                    batch_cost += (n - start - 1) * _group_field(
                        args, end, total_q, 9
                    ) + _group_field(args, end, total_q, 9)
                    batch_blocks += n - start
                    batch_last = _group_field(args, end, total_q, 9)
                end += 1
            if batch_cost != 0 and used + batch_cost > limit + batch_last // 2:
                break
            used += batch_cost
            blocks += batch_blocks
            g = end
            at = 0
        # Assign complete M rows.
        while g < groups:
            n = _group_field(args, g, total_q, 8)
            row_cost = 0
            if n > at:
                row_cost = (n - at - 1) * _group_field(
                    args, g, total_q, 9
                ) + _group_field(args, g, total_q, 9)
            if used + row_cost > limit + _group_field(args, g, total_q, 9) // 2:
                break
            used += row_cost
            blocks += n - at
            g += 1
            at = 0
        # Assign S2 blocks only when LD is enabled.
        if args.ld != 0 and g < groups:
            n = _group_field(args, g, total_q, 8)
            # Every tile in this group has the same CV cost. This is exactly
            # the former loop's stopping condition, computed without iteration.
            cost = _group_field(args, g, total_q, 9)
            take = min(n - at, max(0, (limit + cost // 2 - used) // cost))
            used += take * cost
            blocks += take
            at += take
            if blocks == 0 and at < n:
                cost = _group_field(args, g, total_q, 9)
                if at == n - 1:
                    cost = _group_field(args, g, total_q, 9)
                used += cost
                at += 1
            if at >= n:
                g += 1
                at = 0
        remaining -= used
        if remaining <= 0:
            g = groups
            at = 0
        last_group[c] = g
        last_tile[c] = at
        used_cores = c + 1
    # Cost-aware contiguous LD partitioning. A fragment pays a fixed local
    # selection cost in addition to CV tile work. Six tile-equivalents is an
    # empirical cost-model coefficient, not a shape/core-specific quota.
    # Preserve the original scheduler when it did not split any query group.
    split_groups = 0
    for c in range(used_cores):
        if first_tile[c] != 0 or last_tile[c] != 0:
            split_groups = 1
    if args.sparse == 0 and args.ld != 0 and split_groups != 0:
        # Bound metadata scratch by the number of workers, not input size.
        cached_tiles = zeros(I64, workers)
        cached_weight = zeros(I64, workers)
        for group_index in range(min(groups, workers)):
            cached_tiles[group_index] = _group_field(args, group_index, total_q, 8)
            cached_weight[group_index] = _group_field(args, group_index, total_q, 9)
        upper = total_cost
        lower = max(1, (total_cost + workers - 1) // workers)
        for group_index in range(groups):
            if (
                cached_tiles[group_index]
                if group_index < workers
                else _group_field(args, group_index, total_q, 8)
            ) > 0:
                weight = (
                    cached_weight[group_index]
                    if group_index < workers
                    else _group_field(args, group_index, total_q, 9)
                )
                upper += 6 * weight
                lower = max(lower, 7 * weight)
        # Greedy packing is a feasibility test for the proposed maximum cost.
        # Binary search the smallest feasible integer cost; no shape table.
        while lower < upper:
            limit = (lower + upper) // 2
            g = 0
            at = 0
            for c in range(workers):
                left = limit
                while g < groups:
                    n = (
                        cached_tiles[g]
                        if g < workers
                        else _group_field(args, g, total_q, 8)
                    )
                    if n <= at:
                        g += 1
                        at = 0
                    else:
                        weight = (
                            cached_weight[g]
                            if g < workers
                            else _group_field(args, g, total_q, 9)
                        )
                        take = min(n - at, max(0, left // weight - 6))
                        if take == 0:
                            break
                        left -= (take + 6) * weight
                        at += take
                        if at == n:
                            g += 1
                            at = 0
            if g >= groups:
                upper = limit
            else:
                lower = limit + 1
        g = 0
        at = 0
        used_cores = 0
        for c in range(workers):
            if g >= groups:
                break
            first_group[c] = g
            first_tile[c] = at
            left = upper
            while g < groups:
                n = (
                    cached_tiles[g]
                    if g < workers
                    else _group_field(args, g, total_q, 8)
                )
                if n <= at:
                    g += 1
                    at = 0
                else:
                    weight = (
                        cached_weight[g]
                        if g < workers
                        else _group_field(args, g, total_q, 9)
                    )
                    take = min(n - at, max(0, left // weight - 6))
                    if take == 0:
                        break
                    left -= (take + 6) * weight
                    at += take
                    if at == n:
                        g += 1
                        at = 0
            last_group[c] = g
            last_tile[c] = at
            used_cores = c + 1
    if used_cores == 0:
        used_cores = 1
        first_group[0] = 0
        last_group[0] = groups
    for c in range(used_cores):
        args.output[c * 8] = 1
        start = first_group[c]
        end = last_group[c]
        if start < groups:
            args.output[c * 8 + 1] = _group_field(args, start, total_q, 6)
            m = _group_field(args, start, total_q, GROUP_QUERY_OFFSET_FIELD)
            if args.sparse == 0:
                m = m // args.query_rows
            args.output[c * 8 + 2] = m
            args.output[c * 8 + 3] = first_tile[c]
        if end < groups:
            args.output[c * 8 + 4] = _group_field(args, end, total_q, 6)
            m = _group_field(args, end, total_q, GROUP_QUERY_OFFSET_FIELD)
            if args.sparse == 0:
                m = m // args.query_rows
            args.output[c * 8 + 5] = m
            args.output[c * 8 + 6] = last_tile[c]
        else:
            args.output[c * 8 + 4] = args.batch
    # Record only cross-core rows. Compact workspace slots follow ASC FD order.
    fd_group = zeros(I64, workers)
    fd_parts = zeros(I64, workers)
    fd_base = zeros(I64, workers)
    fd_rows = zeros(I64, workers)
    fd_num = 0
    workspace = 0
    total_fd_load = 0
    c = 0
    while c < used_cores - 1:
        if last_tile[c] > 0:
            group = last_group[c]
            first = c
            end = c + 1
            while (
                end < used_cores - 1 and last_group[end] == group and last_tile[end] > 0
            ):
                end += 1
            parts = end - first + 1
            fd_group[fd_num] = group
            fd_parts[fd_num] = parts
            fd_base[fd_num] = workspace
            fd_rows[fd_num] = _group_field(args, group, total_q, 2)
            total_fd_load += parts * fd_rows[fd_num]
            workspace += parts
            fd_num += 1
            c = end
        else:
            c += 1
    # ASC firstFdDataWorkspaceIdx is the running partial-slot cursor, including
    # cores whose work is entirely unsplit.
    for owner in range(used_cores):
        cursor = workspace
        for f in range(fd_num):
            if first_group[owner] <= fd_group[f]:
                cursor = fd_base[f]
                if first_group[owner] == fd_group[f]:
                    for previous in range(owner):
                        if (
                            last_group[previous] == fd_group[f]
                            and last_tile[previous] > 0
                        ):
                            cursor += 1
                break
        args.output[owner * METADATA_FIELDS + LI_WORKSPACE_OFFSET_FIELD] = cursor
    # ASC SplitFD balances parts * M rows over the available AIVs.
    vectors = workers * 2
    average = (total_fd_load + vectors - 1) // vectors
    spare = vectors - fd_num
    vector = 0
    for f in range(fd_num):
        count = 1
        row_chunk = fd_rows[f]
        if spare > 0:
            count = max(1, fd_parts[f] * fd_rows[f] // average)
            row_chunk = (fd_rows[f] + count - 1) // count
            count = (fd_rows[f] + row_chunk - 1) // row_chunk
            count = min(count, spare + 1)
        group = fd_group[f]
        m = _group_field(args, group, total_q, GROUP_QUERY_OFFSET_FIELD)
        if args.sparse == 0:
            m = m // args.query_rows
        for v in range(count):
            dst = LD_RECORD_START * METADATA_FIELDS + vector * METADATA_FIELDS
            args.output[dst] = 1
            args.output[dst + 1] = _group_field(args, group, total_q, 6)
            args.output[dst + 2] = m
            args.output[dst + 3] = fd_base[f]
            args.output[dst + 4] = fd_parts[f]
            args.output[dst + 5] = v * row_chunk
            rows = row_chunk
            if v == count - 1:
                rows = fd_rows[f] - v * row_chunk
            args.output[dst + 6] = rows
            vector += 1
        spare -= count - 1

    return 0


@lru_cache(None)
def compiled_metadata():
    from pathlib import Path
    from cannbotdsl.aicpu.toolchain import CompiledAicpuKernel, compile_aicpu_kernel

    binary = Path(__file__).parent / "_aicpu" / "metadata_kernel.so"
    if binary.is_file():
        return None, CompiledAicpuKernel(
            str(binary),
            metadata_kernel.spec,
            name="metadata_kernel",
            launch_mode="interface",
        )
    directory = tempfile.TemporaryDirectory(prefix="qli_metadata_aicpu_")
    return directory, compile_aicpu_kernel(
        metadata_kernel, workdir=directory.name, launch_mode="interface"
    )


def choose_splits(topk):
    # UB capacity for one streaming merge pass, not a limit on LD partitions.
    # AICPU computes the actual partition count and per-core ownership.
    return max(1, min(8, 16384 // topk))


def build_metadata(
    q,
    cu,
    used_q,
    used_k,
    residual,
    candidate_length,
    *,
    batch,
    max_tasks,
    capacity,
    splits,
    mask,
    ratio,
    sparse=False,
    query_rows=6,
    ld=True,
    output_idx_offset=None,
    cu_k=None,
):
    groups = q.shape[0] if sparse else batch * max_tasks
    workers = int(torch.npu.get_device_properties(q.device).cube_core_num)
    if not 1 <= workers <= LD_RECORD_START:
        raise RuntimeError(
            f"Unsupported Cube core count {workers}; LI metadata supports 1..{LD_RECORD_START}"
        )
    output = torch.empty((METADATA_ELEMENTS,), dtype=torch.int32, device=q.device)
    scratch = torch.empty(
        (batch + 1 + GROUP_CACHE_FIELDS,), dtype=torch.int64, device=q.device
    )
    _, compiled = compiled_metadata()
    tensors = dict(
        cu=cu,
        used_q=used_q,
        used_k=used_k,
        cu_k=cu_k,
        residual=residual,
        candidate_length=candidate_length,
        output_offset=output_idx_offset,
    )
    compiled.launch(
        current_raw_stream(q.device.index),
        **{k: 0 if v is None else v.data_ptr() for k, v in tensors.items()},
        output=output.data_ptr(),
        scratch=scratch.data_ptr(),
        has_cu=int(cu is not None),
        has_q=int(used_q is not None),
        has_k=int(used_k is not None),
        has_cu_k=int(cu_k is not None),
        has_residual=int(mask == 3 and ratio != 1 and residual is not None),
        sparse=int(sparse),
        mask=mask,
        ratio=ratio,
        total_q=q.shape[0],
        batch=batch,
        max_tasks=max_tasks,
        capacity=capacity,
        splits=splits,
        workers=workers,
        groups=groups,
        query_rows=query_rows,
        heads=q.shape[1],
        ld=int(ld),
        has_offset=int(output_idx_offset is not None),
    )
    stream = torch.npu.current_stream(q.device)
    scratch.record_stream(stream)
    for v in tensors.values():
        if v is not None:
            v.record_stream(stream)
    return output


def metadata_geometry(
    cu_seqlens_q,
    seqused_q,
    seqused_k,
    cmp_residual_k,
    *,
    batch_size,
    max_seqlen_q,
    max_seqlen_k,
    num_heads_q,
    num_heads_k,
    head_dim,
    topk,
    mask_mode,
    cmp_ratio,
    layout_q,
    layout_k,
):
    from types import SimpleNamespace

    if layout_q != "TND" or layout_k != "PA_BBND":
        raise ValueError("Metadata supports TND queries and PA_BBND keys")
    if num_heads_q not in (32, 64) or num_heads_k != 1 or head_dim != 128:
        raise ValueError(
            "MX4 metadata requires Nq=32/64, Nk=1 and logical head_dim=128"
        )
    if mask_mode not in (0, 3) or not 1 <= cmp_ratio <= 128 or topk <= 0:
        raise ValueError("Invalid mask_mode, cmp_ratio or topk")
    tensors = [
        x for x in (cu_seqlens_q, seqused_q, seqused_k, cmp_residual_k) if x is not None
    ]
    device = (
        tensors[0].device
        if tensors
        else torch.device("npu", torch.npu.current_device())
    )
    if device.type != "npu":
        raise ValueError("Metadata sequence inputs must be on NPU")
    if any(
        x.dtype != torch.int32
        or x.ndim != 1
        or x.device != device
        or not x.is_contiguous()
        for x in tensors
    ):
        raise ValueError(
            "Sequence inputs must be one-dimensional int32 tensors on one device"
        )
    batch = (
        cu_seqlens_q.numel() - 1
        if cu_seqlens_q is not None
        else (seqused_q.numel() if seqused_q is not None else (batch_size or 1))
    )
    if batch <= 0 or (batch_size is not None and batch_size != batch):
        raise ValueError("Batch_size disagrees with sequence input shapes")
    if any(
        x is not None and x.numel() != batch
        for x in (seqused_q, seqused_k, cmp_residual_k)
    ):
        raise ValueError("Sequence length tensors must have B elements")
    if cu_seqlens_q is None and batch != 1:
        raise ValueError("TND metadata requires cu_seqlens_q for multiple batches")
    # Sequence values are consumed by AICPU, including during graph replay.
    max_q = int(max_seqlen_q)
    max_k = int(max_seqlen_k)
    if max_q == 0 or max_q < -1 or max_k < -1:
        raise ValueError("Invalid maximum sequence length")
    if max_q == -1 and cu_seqlens_q is None and seqused_q is None:
        raise ValueError("Max_seqlen_q or query sequence lengths are required")
    if max_k == -1 and seqused_k is None:
        raise ValueError("PA metadata requires max_seqlen_k or seqused_k")

    geometry = SimpleNamespace(
        shape=(batch * max_q if max_q >= 0 else -1, num_heads_q), device=device
    )
    return geometry, batch, max_q, max_k


def quant_lightning_indexer_metadata(
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_q=None,
    seqused_k=None,
    cmp_residual_k=None,
    *,
    batch_size=None,
    max_seqlen_q=-1,
    max_seqlen_k=-1,
    num_heads_q,
    num_heads_k,
    head_dim,
    topk,
    mask_mode=0,
    cmp_ratio=1,
    layout_q="TND",
    layout_k="TND",
    candidate_topk_blocks=-1,
    candidate_block_size=-1,
):
    """Build flat AICPU metadata without requiring Q/K data tensors."""
    if (num_heads_q, num_heads_k, head_dim) != (32, 1, 128):
        raise ValueError("QLI metadata requires N1=32, N2=1 and logical D=128")
    if candidate_topk_blocks == -1:
        if candidate_block_size != -1:
            raise ValueError("Disabled candidate requires block size -1")
    elif candidate_topk_blocks <= 0 or candidate_block_size != 8:
        raise ValueError("Candidate requires positive capacity and block size 8")
    if layout_k == "TND":
        return _tnd_metadata(
            cu_seqlens_q,
            cu_seqlens_k,
            seqused_q,
            seqused_k,
            cmp_residual_k,
            batch_size=batch_size,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            num_heads_q=num_heads_q,
            num_heads_k=num_heads_k,
            head_dim=head_dim,
            topk=topk,
            mask_mode=mask_mode,
            cmp_ratio=cmp_ratio,
            layout_q=layout_q,
            candidate_topk_blocks=candidate_topk_blocks,
            candidate_block_size=candidate_block_size,
        )
    q, batch, max_q, capacity = metadata_geometry(
        cu_seqlens_q,
        seqused_q,
        seqused_k,
        cmp_residual_k,
        batch_size=batch_size,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        num_heads_q=num_heads_q,
        num_heads_k=num_heads_k,
        head_dim=head_dim,
        topk=topk,
        mask_mode=mask_mode,
        cmp_ratio=cmp_ratio,
        layout_q=layout_q,
        layout_k=layout_k,
    )
    rows = min(6, 256 // num_heads_q)
    tasks = (max_q + rows - 1) // rows if max_q >= 0 else -1
    return build_metadata(
        q,
        cu_seqlens_q,
        seqused_q,
        seqused_k,
        cmp_residual_k,
        None,
        batch=batch,
        max_tasks=tasks,
        capacity=capacity,
        splits=choose_splits(topk),
        mask=mask_mode,
        ratio=cmp_ratio,
        query_rows=rows,
    )


def _tnd_metadata(
    cu_q,
    cu_k,
    used_q,
    used_k,
    residual,
    *,
    batch_size,
    max_seqlen_q,
    max_seqlen_k,
    num_heads_q,
    num_heads_k,
    head_dim,
    topk,
    mask_mode,
    cmp_ratio,
    layout_q,
    candidate_topk_blocks,
    candidate_block_size,
):
    if (num_heads_q, num_heads_k, head_dim) != (32, 1, 128):
        raise ValueError("TND QLI requires Nq=32, Nk=1 and logical D=128")
    if batch_size is not None:
        raise ValueError("TND metadata requires batch_size=None")
    if layout_q != "TND" or max_seqlen_q < -1 or max_seqlen_k < -1:
        raise ValueError("Invalid layout or sequence maximum")
    if topk <= 0 or mask_mode not in (0, 3) or not 1 <= cmp_ratio <= 128:
        raise ValueError("Invalid topk, mask_mode or cmp_ratio")
    if candidate_topk_blocks == -1:
        if candidate_block_size != -1:
            raise ValueError("Disabled candidate requires block size -1")
    elif candidate_topk_blocks <= 0 or candidate_block_size != 8:
        raise ValueError("Candidate requires positive capacity and block size 8")
    tensors = [
        tensor
        for tensor in (cu_q, cu_k, used_q, used_k, residual)
        if tensor is not None
    ]
    device = (
        tensors[0].device
        if tensors
        else torch.device("npu", torch.npu.current_device())
    )
    if device.type != "npu":
        raise ValueError("Sequence inputs must be on NPU")
    for tensor in tensors:
        if (
            tensor.device != device
            or tensor.dtype != torch.int32
            or tensor.ndim != 1
            or not tensor.is_contiguous()
        ):
            raise ValueError(
                "Sequence inputs must be contiguous int32 vectors on one NPU"
            )
    batch = cu_q.numel() - 1 if cu_q is not None else 1
    if batch <= 0 or (cu_k is not None and cu_k.numel() != batch + 1):
        raise ValueError("Cu_q and cu_k must describe the same positive batch count")
    if any(
        tensor is not None and tensor.numel() != batch
        for tensor in (used_q, used_k, residual)
    ):
        raise ValueError("Sequence lengths must have B elements")
    if (mask_mode == 3 and cmp_ratio != 1) != (residual is not None):
        raise ValueError("Cmp_residual_k is required only for causal compressed K")
    if cu_k is None and batch != 1:
        raise ValueError("Multiple TND batches require cu_seqlens_k")
    if used_k is None and cu_k is None and max_seqlen_k < 0:
        raise ValueError("K boundaries, used lengths or max_seqlen_k are required")
    query, batch, max_query, capacity = metadata_geometry(
        cu_q,
        used_q,
        used_k,
        residual,
        batch_size=None,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max(0, max_seqlen_k),
        num_heads_q=num_heads_q,
        num_heads_k=num_heads_k,
        head_dim=head_dim,
        topk=topk,
        mask_mode=mask_mode,
        cmp_ratio=cmp_ratio,
        layout_q=layout_q,
        layout_k="PA_BBND",
    )
    tasks = (max_query + 5) // 6 if max_query >= 0 else -1
    return build_metadata(
        query,
        cu_q,
        used_q,
        used_k,
        residual,
        None,
        batch=batch,
        max_tasks=tasks,
        capacity=capacity,
        splits=choose_splits(topk),
        mask=mask_mode,
        ratio=cmp_ratio,
        query_rows=6,
        cu_k=cu_k,
    )
