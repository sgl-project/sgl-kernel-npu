# SPDX-License-Identifier: Apache-2.0
"""Paged grouped-query attention with learned per-head sink logits.

Migrated from the internal ``fa_sink.py`` prototype. The public operators
validate metadata before dispatch; this module schedules work, allocates
workspaces and launches the Triton kernels.

A sink is one scalar logit ``s_h`` per query head that joins the softmax
denominator with a zero value contribution:

    O = sum_j exp(qk_j) v_j / (sum_j exp(qk_j) + exp(s_h))

``paged_attention_sink_prefill`` walks the causal prefix of packed queries;
``paged_attention_sink_decode`` scans one query row per request and can split
that scan across cube cores (flash decode), folding the sink in exactly once
during the vector-core reduction so it is never counted per split.
"""

import heapq

import torch
import triton
import triton.language as tl

from ....utils.triton_utils import get_device_properties

# Query rows handled by one prefill task, and the causal mask table width used
# by the block kernels. A masked block starts at most one KV tile before the
# causal boundary, so a table covering one tile of diagonal offset is enough.
_QUERY_TILE = 128
_AUX_MASK_SIZE = 1024
_AUX_MASK_COLUMNS = 3 * _AUX_MASK_SIZE
_aux_mask_cache = {}

# Flash-decode heuristics: below this fraction of cube cores a batched decode
# grid cannot fill the machine, and each split needs this many KV tokens.
_FD_CORE_RATIO = 0.4
_FD_MIN_KV_LEN = 256
_FD_MIN_KV_LEN_GQA = 2048
_KV_SPLIT_LIMIT = 256


def _num_cores(kind):
    """Return the cube or vector core count reported by the driver."""
    if kind not in ("cube", "vector"):
        raise ValueError("kind must be 'cube' or 'vector'")
    aicore, vectorcore = get_device_properties()
    return aicore if kind == "cube" else vectorcore


def _causal_mask_table(device):
    """Return the cached causal mask consumed by the block kernels.

    Entry ``[i, _AUX_MASK_SIZE + d]`` is ``key_position + d <= query_position``
    for ``|d| <= _AUX_MASK_SIZE``. Kernels clamp the per-block diagonal offset,
    which is exact because masked blocks begin within one tile of the boundary.
    """
    table = _aux_mask_cache.get(device)
    if table is None:
        table = torch.ones(
            (_AUX_MASK_SIZE, _AUX_MASK_COLUMNS), device=device, dtype=torch.bool
        ).tril(_AUX_MASK_SIZE)
        _aux_mask_cache[device] = table
    return table


def _tensor_to_cpu_list(tensor):
    return [int(value) for value in tensor.detach().cpu().tolist()]


def _build_lpt_schedule(
    cu_q_lens,
    seqlens_kv,
    num_q_heads,
    num_kv_heads,
    q_rows,
    block_size_m,
    block_size_n,
    page_size,
    max_pages,
    cube_num,
    gqa_interleave,
):
    """Longest-processing-time schedule for the one-dimensional prefill grid.

    Each task computes one (request, query block, query head) triple and costs
    the number of KV blocks it scans. Assigning the most expensive task to the
    currently cheapest core keeps the kernel free of global atomics while
    avoiding the worst tail imbalance of round-robin assignment.

    Returns ``(task_b, task_q_block, task_q_head, core_task_offsets)`` on the
    length tensor's device; ``core_task_offsets`` has ``cube_num + 1`` entries.
    """
    cu_q_lens_host = _tensor_to_cpu_list(cu_q_lens)
    seqlens_kv_host = None if seqlens_kv is None else _tensor_to_cpu_list(seqlens_kv)
    batch_size = len(cu_q_lens_host) - 1
    if not cu_q_lens_host or cu_q_lens_host[0] != 0:
        raise ValueError("cu_q_lens must start at 0")
    if cu_q_lens_host[-1] != q_rows:
        raise ValueError(
            "cu_q_lens[-1] ({}) must equal the number of query rows ({})".format(
                cu_q_lens_host[-1], q_rows
            )
        )
    if any(end < start for start, end in zip(cu_q_lens_host, cu_q_lens_host[1:])):
        raise ValueError("cu_q_lens must be non-decreasing")

    weighted_tasks = []
    for b_id in range(batch_size):
        q_seq_len = cu_q_lens_host[b_id + 1] - cu_q_lens_host[b_id]
        if q_seq_len <= 0:
            continue

        kv_seq_len = q_seq_len if seqlens_kv_host is None else seqlens_kv_host[b_id]
        kv_cache_len = kv_seq_len - q_seq_len
        if kv_cache_len < 0:
            raise ValueError(
                "seqlens_kv[{}] ({}) must be >= its query length ({})".format(
                    b_id, kv_seq_len, q_seq_len
                )
            )
        if triton.cdiv(kv_seq_len, page_size) > max_pages:
            raise ValueError(
                "block_tables row {} has {} pages but request {} needs {}".format(
                    b_id, max_pages, b_id, triton.cdiv(kv_seq_len, page_size)
                )
            )

        for q_block_id in range(triton.cdiv(q_seq_len, block_size_m)):
            q_block_end = min((q_block_id + 1) * block_size_m, q_seq_len)
            cost = max(1, triton.cdiv(kv_cache_len + q_block_end, block_size_n))
            for q_head_id in range(num_q_heads):
                if gqa_interleave:
                    kv_head_id = q_head_id % num_kv_heads
                else:
                    kv_head_id = q_head_id // (num_q_heads // num_kv_heads)
                weighted_tasks.append((b_id, kv_head_id, q_block_id, cost, q_head_id))

    heap = [(0, core_id) for core_id in range(cube_num)]
    per_core_tasks = [[] for _ in range(cube_num)]

    # Sorting the full tuple, as the prototype did, orders by request id first
    # and only breaks ties by cost. Sort on the cost estimate to balance cores.
    for task in sorted(weighted_tasks, key=lambda item: item[3], reverse=True):
        b_id, _, q_block_id, cost, q_head_id = task
        core_cost, core_id = heapq.heappop(heap)
        per_core_tasks[core_id].append((b_id, q_head_id, q_block_id))
        heapq.heappush(heap, (core_cost + cost, core_id))

    task_b = []
    task_q_block = []
    task_q_head = []
    core_task_offsets = [0]
    for tasks in per_core_tasks:
        for b_id, q_head_id, q_block_id in tasks:
            task_b.append(b_id)
            task_q_block.append(q_block_id)
            task_q_head.append(q_head_id)
        core_task_offsets.append(len(task_b))

    device = cu_q_lens.device
    return (
        torch.tensor(task_b, device=device, dtype=torch.int32),
        torch.tensor(task_q_block, device=device, dtype=torch.int32),
        torch.tensor(task_q_head, device=device, dtype=torch.int32),
        torch.tensor(core_task_offsets, device=device, dtype=torch.int32),
    )


@triton.jit
def _causal_mask(
    mask_ptr,
    mask_size,
    mask_stride_m,
    mask_stride_n,
    q_start,
    kv_start,
    Q_BLOCK,
    KV_BLOCK,
):
    offset_causal = tl.minimum(tl.maximum(kv_start - q_start, -mask_size), mask_size)
    offsets_mask_causal = (tl.arange(0, Q_BLOCK)[:, None]) * mask_stride_m + (
        mask_size + offset_causal + tl.arange(0, KV_BLOCK)[None, :]
    ) * mask_stride_n
    return tl.load(mask_ptr + offsets_mask_causal).to(tl.int1)


@triton.jit
def _sdpa_infer_single_block(
    acc,
    l_i,
    m_i,
    q,
    k_T_block_ptr,
    v_block_ptr,
    qk_scale,
    mask,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    tl.static_assert(
        HEAD_DIM <= BLOCK_D, "BLOCK_SIZE_D should not be less than HEAD_DIM"
    )
    k_T = tl.load(k_T_block_ptr, boundary_check=(0, 1), padding_option="zero")
    qk = tl.dot(q, k_T)

    qk = qk * qk_scale
    if mask is not None:
        qk = tl.where(mask, qk, float("-inf"))

    m_ij = tl.maximum(
        m_i,
        tl.max(qk, 1, propagate_nan=True),
        propagate_nan=tl.PropagateNan.ALL,
    )
    qk = qk - m_ij[:, None]

    p = tl.math.exp(qk)
    p_cast = p.to(k_T.dtype)

    v = tl.load(v_block_ptr, boundary_check=(0, 1), padding_option="zero")

    l_ij = tl.sum(p, 1)
    alpha = tl.math.exp(m_i - m_ij)
    l_i = l_i * alpha + l_ij
    acc = acc * alpha[:, None]
    acc = tl.dot(p_cast, v, acc)

    return acc, l_i, m_ij


@triton.jit
def _paged_prefill_kernel(
    q_ptr,
    key_cache_ptr,
    value_cache_ptr,
    o_ptr,
    aux_mask_ptr,
    task_b_ptr,
    task_q_block_ptr,
    task_q_head_ptr,
    core_task_offsets_ptr,
    cu_q_lens_ptr,
    seqlens_kv_ptr,
    block_tables_ptr,
    sinks_ptr,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_k_block,
    stride_k_head,
    stride_k_blksz,
    stride_k_dim,
    stride_v_block,
    stride_v_head,
    stride_v_blksz,
    stride_v_dim,
    stride_ot,
    stride_oh,
    stride_od,
    stride_bt_batch,
    stride_bt_block,
    stride_mask_m,
    stride_mask_n,
    stride_sink_head,
    softmax_scale,
    AUX_MASK_SIZE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GQA_INTERLEAVE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    SINK_ENABLED: tl.constexpr,
):
    pid = tl.program_id(0)

    tl.static_assert(
        PAGE_SIZE % BLOCK_SIZE_N == 0, "BLOCK_SIZE_N must be a divisor of PAGE_SIZE"
    )

    task_begin = tl.load(core_task_offsets_ptr + pid)
    task_end = tl.load(core_task_offsets_ptr + pid + 1)

    for task_idx in range(task_begin, task_end):
        b_id = tl.load(task_b_ptr + task_idx)
        q_block_id = tl.load(task_q_block_ptr + task_idx)
        q_head_id = tl.load(task_q_head_ptr + task_idx)

        q_start_loc = tl.load(cu_q_lens_ptr + b_id).to(tl.int32)
        q_end_loc = tl.load(cu_q_lens_ptr + b_id + 1).to(tl.int32)
        q_seq_len = q_end_loc - q_start_loc

        if seqlens_kv_ptr is None:
            kv_seq_len = q_seq_len
        else:
            kv_seq_len = tl.load(seqlens_kv_ptr + b_id)
        kv_cache_len = kv_seq_len - q_seq_len

        if GQA_INTERLEAVE:
            kv_head_id = q_head_id % NUM_KV_HEADS
        else:
            kv_head_id = q_head_id // (NUM_Q_HEADS // NUM_KV_HEADS)

        q_block_start_in_seq = q_block_id * BLOCK_SIZE_M
        q_block_end_in_seq = tl.minimum(q_block_start_in_seq + BLOCK_SIZE_M, q_seq_len)
        q_block_len = q_block_end_in_seq - q_block_start_in_seq

        Q_block_ptr = tl.make_block_ptr(
            base=q_ptr
            + (q_start_loc + q_block_start_in_seq) * stride_qt
            + q_head_id * stride_qh,
            shape=(q_block_len, HEAD_DIM),
            strides=(stride_qt, stride_qd),
            offsets=(0, 0),
            block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_D),
            order=(1, 0),
        )
        O_block_ptr = tl.make_block_ptr(
            base=o_ptr
            + (q_start_loc + q_block_start_in_seq) * stride_ot
            + q_head_id * stride_oh,
            shape=(q_block_len, HEAD_DIM),
            strides=(stride_ot, stride_od),
            offsets=(0, 0),
            block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_D),
            order=(1, 0),
        )

        q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")

        # Online-softmax initial state. The sink logit seeds the running max and
        # contributes one unit to the denominator with a zero value vector.
        m_i = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
        l_i = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
        if SINK_ENABLED:
            s_h = tl.load(sinks_ptr + q_head_id * stride_sink_head).to(tl.float32)
            m_i = m_i + s_h
            l_i = l_i + 1.0
        else:
            m_i = m_i - float("inf")
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_D), dtype=tl.float32)

        num_kv_blocks = tl.cdiv(kv_cache_len + q_block_end_in_seq, BLOCK_SIZE_N)
        # KV blocks fully below the cached prefix need no causal mask.
        num_no_mask_blocks = (kv_cache_len + q_block_start_in_seq) // BLOCK_SIZE_N

        for kv_block_id in range(0, num_no_mask_blocks):
            kv_block_start_in_seq = kv_block_id * BLOCK_SIZE_N
            kv_block_end_in_seq = tl.minimum(
                kv_block_start_in_seq + BLOCK_SIZE_N, kv_seq_len
            )
            kv_block_len = kv_block_end_in_seq - kv_block_start_in_seq

            logical_page_id = kv_block_start_in_seq // PAGE_SIZE
            kv_block_start_in_page = kv_block_start_in_seq % PAGE_SIZE
            physical_page_id = tl.load(
                block_tables_ptr
                + b_id * stride_bt_batch
                + logical_page_id * stride_bt_block
            )

            K_T_block_ptr = tl.make_block_ptr(
                base=key_cache_ptr
                + physical_page_id * stride_k_block
                + kv_head_id * stride_k_head
                + kv_block_start_in_page * stride_k_blksz,
                shape=(HEAD_DIM, kv_block_len),
                strides=(stride_k_dim, stride_k_blksz),
                offsets=(0, 0),
                block_shape=(BLOCK_SIZE_D, BLOCK_SIZE_N),
                order=(0, 1),
            )
            V_block_ptr = tl.make_block_ptr(
                base=value_cache_ptr
                + physical_page_id * stride_v_block
                + kv_head_id * stride_v_head
                + kv_block_start_in_page * stride_v_blksz,
                shape=(kv_block_len, HEAD_DIM),
                strides=(stride_v_blksz, stride_v_dim),
                offsets=(0, 0),
                block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_D),
                order=(1, 0),
            )

            acc, l_i, m_i = _sdpa_infer_single_block(
                acc,
                l_i,
                m_i,
                q,
                K_T_block_ptr,
                V_block_ptr,
                softmax_scale,
                None,
                HEAD_DIM,
                BLOCK_SIZE_M,
                BLOCK_SIZE_N,
                BLOCK_SIZE_D,
            )

        for kv_block_id in range(num_no_mask_blocks, num_kv_blocks):
            kv_block_start_in_seq = kv_block_id * BLOCK_SIZE_N
            kv_block_end_in_seq = tl.minimum(
                kv_block_start_in_seq + BLOCK_SIZE_N, kv_seq_len
            )
            kv_block_len = kv_block_end_in_seq - kv_block_start_in_seq

            logical_page_id = kv_block_start_in_seq // PAGE_SIZE
            kv_block_start_in_page = kv_block_start_in_seq % PAGE_SIZE
            physical_page_id = tl.load(
                block_tables_ptr
                + b_id * stride_bt_batch
                + logical_page_id * stride_bt_block
            )

            mask = _causal_mask(
                aux_mask_ptr,
                AUX_MASK_SIZE,
                stride_mask_m,
                stride_mask_n,
                kv_cache_len + q_block_start_in_seq,
                kv_block_start_in_seq,
                BLOCK_SIZE_M,
                BLOCK_SIZE_N,
            )

            K_T_block_ptr = tl.make_block_ptr(
                base=key_cache_ptr
                + physical_page_id * stride_k_block
                + kv_head_id * stride_k_head
                + kv_block_start_in_page * stride_k_blksz,
                shape=(HEAD_DIM, kv_block_len),
                strides=(stride_k_dim, stride_k_blksz),
                offsets=(0, 0),
                block_shape=(BLOCK_SIZE_D, BLOCK_SIZE_N),
                order=(0, 1),
            )
            V_block_ptr = tl.make_block_ptr(
                base=value_cache_ptr
                + physical_page_id * stride_v_block
                + kv_head_id * stride_v_head
                + kv_block_start_in_page * stride_v_blksz,
                shape=(kv_block_len, HEAD_DIM),
                strides=(stride_v_blksz, stride_v_dim),
                offsets=(0, 0),
                block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_D),
                order=(1, 0),
            )

            acc, l_i, m_i = _sdpa_infer_single_block(
                acc,
                l_i,
                m_i,
                q,
                K_T_block_ptr,
                V_block_ptr,
                softmax_scale,
                mask,
                HEAD_DIM,
                BLOCK_SIZE_M,
                BLOCK_SIZE_N,
                BLOCK_SIZE_D,
            )

        accumulator = acc / l_i[:, None]
        tl.store(
            O_block_ptr, accumulator.to(o_ptr.type.element_ty), boundary_check=(0, 1)
        )


@triton.jit
def _paged_prefill_page_aggregation_kernel(
    q_ptr,
    key_cache_ptr,
    value_cache_ptr,
    o_ptr,
    aux_mask_ptr,
    task_b_ptr,
    task_q_block_ptr,
    task_q_head_ptr,
    core_task_offsets_ptr,
    batched_max_pages,
    cu_q_lens_ptr,
    seqlens_kv_ptr,
    block_tables_ptr,
    sinks_ptr,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_k_block,
    stride_k_head,
    stride_k_blksz,
    stride_k_dim,
    stride_v_block,
    stride_v_head,
    stride_v_blksz,
    stride_v_dim,
    stride_ot,
    stride_oh,
    stride_od,
    stride_bt_batch,
    stride_bt_block,
    stride_mask_m,
    stride_mask_n,
    stride_sink_head,
    softmax_scale,
    AUX_MASK_SIZE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GQA_INTERLEAVE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    PAGE_AGGREGATION_NUM: tl.constexpr,
    SINK_ENABLED: tl.constexpr,
):
    pid = tl.program_id(0)

    task_begin = tl.load(core_task_offsets_ptr + pid)
    task_end = tl.load(core_task_offsets_ptr + pid + 1)

    for task_idx in range(task_begin, task_end):
        b_id = tl.load(task_b_ptr + task_idx)
        q_block_id = tl.load(task_q_block_ptr + task_idx)
        q_head_id = tl.load(task_q_head_ptr + task_idx)

        q_start_loc = tl.load(cu_q_lens_ptr + b_id).to(tl.int32)
        q_end_loc = tl.load(cu_q_lens_ptr + b_id + 1).to(tl.int32)
        q_seq_len = q_end_loc - q_start_loc

        if seqlens_kv_ptr is None:
            kv_seq_len = q_seq_len
        else:
            kv_seq_len = tl.load(seqlens_kv_ptr + b_id)
        kv_cache_len = kv_seq_len - q_seq_len

        if GQA_INTERLEAVE:
            kv_head_id = q_head_id % NUM_KV_HEADS
        else:
            kv_head_id = q_head_id // (NUM_Q_HEADS // NUM_KV_HEADS)

        q_block_start_in_seq = q_block_id * BLOCK_SIZE_M
        q_block_end_in_seq = tl.minimum(q_block_start_in_seq + BLOCK_SIZE_M, q_seq_len)
        q_block_len = q_block_end_in_seq - q_block_start_in_seq

        Q_block_ptr = tl.make_block_ptr(
            base=q_ptr
            + (q_start_loc + q_block_start_in_seq) * stride_qt
            + q_head_id * stride_qh,
            shape=(q_block_len, HEAD_DIM),
            strides=(stride_qt, stride_qd),
            offsets=(0, 0),
            block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_D),
            order=(1, 0),
        )
        O_block_ptr = tl.make_block_ptr(
            base=o_ptr
            + (q_start_loc + q_block_start_in_seq) * stride_ot
            + q_head_id * stride_oh,
            shape=(q_block_len, HEAD_DIM),
            strides=(stride_ot, stride_od),
            offsets=(0, 0),
            block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_D),
            order=(1, 0),
        )

        q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")

        # Online-softmax initial state with the optional sink.
        m_i = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
        l_i = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
        if SINK_ENABLED:
            s_h = tl.load(sinks_ptr + q_head_id * stride_sink_head).to(tl.float32)
            m_i = m_i + s_h
            l_i = l_i + 1.0
        else:
            m_i = m_i - float("inf")
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_D), dtype=tl.float32)

        num_kv_blocks = tl.cdiv(kv_cache_len + q_block_end_in_seq, BLOCK_SIZE_N)
        for kv_block_id in range(0, num_kv_blocks, PAGE_AGGREGATION_NUM):
            mask = _causal_mask(
                aux_mask_ptr,
                AUX_MASK_SIZE,
                stride_mask_m,
                stride_mask_n,
                kv_cache_len + q_block_start_in_seq,
                kv_block_id * BLOCK_SIZE_N,
                BLOCK_SIZE_M,
                BLOCK_SIZE_N * PAGE_AGGREGATION_NUM,
            )

            # Gather PAGE_AGGREGATION_NUM consecutive KV blocks into one tile so
            # a single cube instruction covers the whole 128-row window.
            k = tl.zeros(
                (PAGE_AGGREGATION_NUM * BLOCK_SIZE_N, BLOCK_SIZE_D),
                dtype=key_cache_ptr.dtype.element_ty,
            )
            for page_iter in tl.extra.cann.extension.parallel(0, PAGE_AGGREGATION_NUM):
                kv_block_start = (kv_block_id + page_iter) * BLOCK_SIZE_N
                kv_block_end = tl.minimum(kv_block_start + BLOCK_SIZE_N, kv_seq_len)
                kv_block_len = tl.maximum(kv_block_end - kv_block_start, 0)
                logical_page_id = tl.minimum(
                    kv_block_start // PAGE_SIZE, batched_max_pages - 1
                )
                kv_block_start_in_page = kv_block_start % PAGE_SIZE
                physical_page_id = tl.load(
                    block_tables_ptr
                    + b_id * stride_bt_batch
                    + logical_page_id * stride_bt_block
                )
                cur_k_block_ptr = tl.make_block_ptr(
                    base=key_cache_ptr
                    + physical_page_id * stride_k_block
                    + kv_head_id * stride_k_head
                    + kv_block_start_in_page * stride_k_blksz,
                    shape=(kv_block_len, HEAD_DIM),
                    strides=(stride_k_blksz, stride_k_dim),
                    offsets=(0, 0),
                    block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_D),
                    order=(1, 0),
                )
                k_slice = tl.load(
                    cur_k_block_ptr, boundary_check=(0, 1), padding_option="zero"
                )
                k = tl.extra.cann.extension.insert_slice(
                    k,
                    k_slice,
                    offsets=(page_iter * BLOCK_SIZE_N, 0),
                    sizes=(BLOCK_SIZE_N, BLOCK_SIZE_D),
                    strides=(1, 1),
                )
            k_T = tl.trans(k)
            qk = tl.dot(q, k_T)

            qk = qk * softmax_scale
            if mask is not None:
                qk = tl.where(mask, qk, float("-inf"))

            m_ij = tl.maximum(
                m_i,
                tl.max(qk, 1, propagate_nan=True),
                propagate_nan=tl.PropagateNan.ALL,
            )
            qk = qk - m_ij[:, None]

            p = tl.math.exp(qk)
            p_cast = p.to(k_T.dtype)

            l_ij = tl.sum(p, 1)
            alpha = tl.math.exp(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, None]

            v = tl.zeros(
                (PAGE_AGGREGATION_NUM * BLOCK_SIZE_N, BLOCK_SIZE_D),
                dtype=value_cache_ptr.dtype.element_ty,
            )
            for page_iter in tl.extra.cann.extension.parallel(0, PAGE_AGGREGATION_NUM):
                kv_block_start = (kv_block_id + page_iter) * BLOCK_SIZE_N
                kv_block_end = tl.minimum(kv_block_start + BLOCK_SIZE_N, kv_seq_len)
                kv_block_len = tl.maximum(kv_block_end - kv_block_start, 0)
                logical_page_id = tl.minimum(
                    kv_block_start // PAGE_SIZE, batched_max_pages - 1
                )
                kv_block_start_in_page = kv_block_start % PAGE_SIZE
                physical_page_id = tl.load(
                    block_tables_ptr
                    + b_id * stride_bt_batch
                    + logical_page_id * stride_bt_block
                )
                cur_v_block_ptr = tl.make_block_ptr(
                    base=value_cache_ptr
                    + physical_page_id * stride_v_block
                    + kv_head_id * stride_v_head
                    + kv_block_start_in_page * stride_v_blksz,
                    shape=(kv_block_len, HEAD_DIM),
                    strides=(stride_v_blksz, stride_v_dim),
                    offsets=(0, 0),
                    block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_D),
                    order=(1, 0),
                )
                v_slice = tl.load(
                    cur_v_block_ptr, boundary_check=(0, 1), padding_option="zero"
                )
                v = tl.extra.cann.extension.insert_slice(
                    v,
                    v_slice,
                    offsets=(page_iter * BLOCK_SIZE_N, 0),
                    sizes=(BLOCK_SIZE_N, BLOCK_SIZE_D),
                    strides=(1, 1),
                )
            acc = tl.dot(p_cast, v, acc)

            m_i = m_ij

        accumulator = acc / l_i[:, None]
        tl.store(
            O_block_ptr, accumulator.to(o_ptr.type.element_ty), boundary_check=(0, 1)
        )


def _should_use_flash_decode(
    batch_size, num_kv_heads, group_size, max_kv_len, cube_num
):
    """Split the KV scan when the batched grid cannot fill all cube cores.

    The split also requires a KV sequence long enough to amortise the
    workspace and the reduction pass; GQA needs a longer sequence than
    MHA/MQA because each split re-reads a much wider K/V tile.
    """
    if max_kv_len < _FD_MIN_KV_LEN:
        return False
    loop_times = batch_size * num_kv_heads
    if loop_times >= _FD_CORE_RATIO * cube_num:
        return False
    if group_size == 1:
        return True
    return max_kv_len >= _FD_MIN_KV_LEN_GQA


def _compute_kv_split_parts(batch_size, num_kv_heads, max_kv_len, cube_num):
    """Bound the split count by idle cores and by KV tokens per split."""
    loop_times = batch_size * num_kv_heads
    max_by_cores = cube_num // loop_times
    max_by_len = max_kv_len // _KV_SPLIT_LIMIT
    return max(1, min(max_by_cores, max_by_len))


@triton.jit
def _paged_decode_fd_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    seqlens_ptr,
    block_tables_ptr,
    acc_ws_ptr,
    lse_ws_ptr,
    stride_qb,
    stride_qh,
    stride_qd,
    stride_k_block,
    stride_k_head,
    stride_k_blksz,
    stride_k_dim,
    stride_v_block,
    stride_v_head,
    stride_v_blksz,
    stride_v_dim,
    stride_bt_batch,
    stride_bt_block,
    stride_aws_task,
    stride_aws_g,
    stride_aws_d,
    stride_lse_task,
    stride_lse_g,
    softmax_scale,
    BATCH_SIZE,
    KV_SPLIT_PARTS: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GQA_INTERLEAVE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    GROUP_SIZE: tl.constexpr = NUM_Q_HEADS // NUM_KV_HEADS

    pid = tl.program_id(0)
    n_progs = tl.num_programs(0)
    total_fd_tasks = BATCH_SIZE * NUM_KV_HEADS * KV_SPLIT_PARTS

    for fd_task_id in range(pid, total_fd_tasks, n_progs):
        split_idx = fd_task_id % KV_SPLIT_PARTS
        kv_task = fd_task_id // KV_SPLIT_PARTS
        b_id = kv_task // NUM_KV_HEADS
        kv_head_id = kv_task % NUM_KV_HEADS

        kv_seq_len = tl.load(seqlens_ptr + b_id)

        # Align each split to page boundaries so no KV block crosses two pages;
        # a cross-page block would read a single physical page for two logical
        # pages and produce wrong values.
        raw_chunk = tl.cdiv(kv_seq_len, KV_SPLIT_PARTS)
        chunk_size = tl.cdiv(raw_chunk, PAGE_SIZE) * PAGE_SIZE
        kv_start = split_idx * chunk_size
        kv_end = tl.minimum(kv_start + chunk_size, kv_seq_len)

        ws_task_idx = (b_id * NUM_KV_HEADS + kv_head_id) * KV_SPLIT_PARTS + split_idx

        g_offsets = tl.arange(0, GROUP_SIZE)
        if GQA_INTERLEAVE:
            q_head_ids = kv_head_id + g_offsets * NUM_KV_HEADS
        else:
            q_head_ids = kv_head_id * GROUP_SIZE + g_offsets

        offs_d = tl.arange(0, BLOCK_SIZE_D)

        q_ptrs = (
            q_ptr
            + b_id * stride_qb
            + q_head_ids[:, None] * stride_qh
            + offs_d[None, :] * stride_qd
        )
        q = tl.load(q_ptrs, mask=offs_d[None, :] < HEAD_DIM, other=0.0)

        # The sink is intentionally absent here: each split would add it again.
        m_i = tl.zeros((GROUP_SIZE,), dtype=tl.float32) - float("inf")
        l_i = tl.zeros((GROUP_SIZE,), dtype=tl.float32)
        acc = tl.zeros((GROUP_SIZE, BLOCK_SIZE_D), dtype=tl.float32)

        num_kv_blocks = tl.cdiv(kv_end - kv_start, BLOCK_SIZE_N)

        for kv_block_id in range(num_kv_blocks):
            kv_block_start = kv_start + kv_block_id * BLOCK_SIZE_N
            kv_block_end = tl.minimum(kv_block_start + BLOCK_SIZE_N, kv_end)
            kv_block_len = kv_block_end - kv_block_start

            logical_page_id = kv_block_start // PAGE_SIZE
            kv_block_start_in_page = kv_block_start % PAGE_SIZE
            physical_page_id = tl.load(
                block_tables_ptr
                + b_id * stride_bt_batch
                + logical_page_id * stride_bt_block
            )

            K_T_block_ptr = tl.make_block_ptr(
                base=k_cache_ptr
                + physical_page_id * stride_k_block
                + kv_head_id * stride_k_head
                + kv_block_start_in_page * stride_k_blksz,
                shape=(HEAD_DIM, kv_block_len),
                strides=(stride_k_dim, stride_k_blksz),
                offsets=(0, 0),
                block_shape=(BLOCK_SIZE_D, BLOCK_SIZE_N),
                order=(0, 1),
            )
            V_block_ptr = tl.make_block_ptr(
                base=v_cache_ptr
                + physical_page_id * stride_v_block
                + kv_head_id * stride_v_head
                + kv_block_start_in_page * stride_v_blksz,
                shape=(kv_block_len, HEAD_DIM),
                strides=(stride_v_blksz, stride_v_dim),
                offsets=(0, 0),
                block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_D),
                order=(1, 0),
            )

            mask = tl.arange(0, BLOCK_SIZE_N) < kv_block_len

            k_T = tl.load(K_T_block_ptr, boundary_check=(0, 1), padding_option="zero")
            v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

            qk = tl.dot(q, k_T)
            qk = qk * softmax_scale
            qk = tl.where(mask[None, :], qk, float("-inf"))

            m_ij = tl.maximum(
                m_i,
                tl.max(qk, 1, propagate_nan=True),
                propagate_nan=tl.PropagateNan.ALL,
            )
            qk = qk - m_ij[:, None]
            p = tl.math.exp(qk)
            l_ij = tl.sum(p, 1)
            alpha = tl.math.exp(m_i - m_ij)

            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, None] + tl.dot(p.to(k_T.dtype), v)
            m_i = m_ij

        # Store a locally normalised partial output; the reduction re-weights
        # each split by exp(lse_i - lse_max).
        l_i_safe = tl.where(l_i > 0, l_i, 1.0)
        acc = acc / l_i_safe[:, None]
        lse_i = tl.where(l_i > 0, m_i + tl.math.log(l_i), float("-inf"))

        lse_ptrs = lse_ws_ptr + ws_task_idx * stride_lse_task + g_offsets * stride_lse_g
        tl.store(lse_ptrs, lse_i)

        acc_ptrs = (
            acc_ws_ptr
            + ws_task_idx * stride_aws_task
            + g_offsets[:, None] * stride_aws_g
            + offs_d[None, :] * stride_aws_d
        )
        tl.store(acc_ptrs, acc, mask=offs_d[None, :] < HEAD_DIM)


@triton.jit
def _paged_decode_fd_reduce_kernel(
    acc_ws_ptr,
    lse_ws_ptr,
    o_ptr,
    sinks_ptr,
    stride_aws_task,
    stride_aws_g,
    stride_aws_d,
    stride_lse_task,
    stride_lse_g,
    stride_ob,
    stride_oh,
    stride_od,
    stride_sink_head,
    BATCH_SIZE,
    KV_SPLIT_PARTS: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GQA_INTERLEAVE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    SINK_ENABLED: tl.constexpr,
):
    GROUP_SIZE: tl.constexpr = NUM_Q_HEADS // NUM_KV_HEADS

    pid = tl.program_id(0)
    n_progs = tl.num_programs(0)
    total_reduce_tasks = BATCH_SIZE * NUM_KV_HEADS

    for reduce_task_id in range(pid, total_reduce_tasks, n_progs):
        b_id = reduce_task_id // NUM_KV_HEADS
        kv_head_id = reduce_task_id % NUM_KV_HEADS

        g_offsets = tl.arange(0, GROUP_SIZE)
        if GQA_INTERLEAVE:
            q_head_ids = kv_head_id + g_offsets * NUM_KV_HEADS
        else:
            q_head_ids = kv_head_id * GROUP_SIZE + g_offsets

        offs_d = tl.arange(0, BLOCK_SIZE_D)

        if SINK_ENABLED:
            s_h = tl.load(
                sinks_ptr + q_head_ids * stride_sink_head,
                mask=q_head_ids < NUM_Q_HEADS,
                other=-float("inf"),
            ).to(tl.float32)

        # Pass 1: per-head maximum log-sum-exp across the splits.
        lse_max = tl.zeros((GROUP_SIZE,), dtype=tl.float32) - float("inf")
        for split_idx in tl.static_range(KV_SPLIT_PARTS):
            ws_task_idx = (
                b_id * NUM_KV_HEADS + kv_head_id
            ) * KV_SPLIT_PARTS + split_idx
            lse_ptrs = (
                lse_ws_ptr + ws_task_idx * stride_lse_task + g_offsets * stride_lse_g
            )
            lse_max = tl.maximum(lse_max, tl.load(lse_ptrs))

        if SINK_ENABLED:
            lse_max = tl.maximum(lse_max, s_h)

        # Pass 2: re-weight and accumulate the partial outputs.
        out = tl.zeros((GROUP_SIZE, BLOCK_SIZE_D), dtype=tl.float32)
        exp_sum = tl.zeros((GROUP_SIZE,), dtype=tl.float32)

        for split_idx in tl.static_range(KV_SPLIT_PARTS):
            ws_task_idx = (
                b_id * NUM_KV_HEADS + kv_head_id
            ) * KV_SPLIT_PARTS + split_idx

            lse_ptrs = (
                lse_ws_ptr + ws_task_idx * stride_lse_task + g_offsets * stride_lse_g
            )
            lse = tl.load(lse_ptrs)
            # Empty splits store -inf and therefore contribute a zero weight.
            w = tl.math.exp(lse - lse_max)
            exp_sum += w

            acc_ptrs = (
                acc_ws_ptr
                + ws_task_idx * stride_aws_task
                + g_offsets[:, None] * stride_aws_g
                + offs_d[None, :] * stride_aws_d
            )
            acc_split = tl.load(acc_ptrs, mask=offs_d[None, :] < HEAD_DIM, other=0.0)
            out += w[:, None] * acc_split

        # The sink adds a normalisation-only term with a zero value vector.
        if SINK_ENABLED:
            exp_sum += tl.math.exp(s_h - lse_max)

        exp_sum_safe = tl.where(exp_sum > 0, exp_sum, 1.0)
        out = out / exp_sum_safe[:, None]

        o_ptrs = (
            o_ptr
            + b_id * stride_ob
            + q_head_ids[:, None] * stride_oh
            + offs_d[None, :] * stride_od
        )
        tl.store(
            o_ptrs, out.to(o_ptr.dtype.element_ty), mask=offs_d[None, :] < HEAD_DIM
        )


@triton.jit
def _paged_decode_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    o_ptr,
    seqlens_ptr,
    block_tables_ptr,
    sinks_ptr,
    BATCH_SIZE,
    MAX_PAGES,
    stride_qb,
    stride_qh,
    stride_qd,
    stride_k_block,
    stride_k_head,
    stride_k_blksz,
    stride_k_dim,
    stride_v_block,
    stride_v_head,
    stride_v_blksz,
    stride_v_dim,
    stride_ob,
    stride_oh,
    stride_od,
    stride_bt_batch,
    stride_bt_block,
    stride_sink_head,
    softmax_scale,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GQA_INTERLEAVE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    SINK_ENABLED: tl.constexpr,
):
    GROUP_SIZE: tl.constexpr = NUM_Q_HEADS // NUM_KV_HEADS
    tl.static_assert(
        HEAD_DIM <= BLOCK_SIZE_D, "HEAD_DIM should be less than BLOCK_SIZE_D"
    )
    tl.static_assert(
        PAGE_SIZE % BLOCK_SIZE_N == 0, "BLOCK_SIZE_N must be a divisor of PAGE_SIZE"
    )
    pid = tl.program_id(0)
    n_progs = tl.num_programs(0)

    num_tasks = BATCH_SIZE * NUM_KV_HEADS

    for kv_task_id in range(pid, num_tasks, n_progs):
        kv_head_id = kv_task_id % NUM_KV_HEADS
        b_id = kv_task_id // NUM_KV_HEADS

        kv_seq_len = tl.load(seqlens_ptr + b_id)

        g_offsets = tl.arange(0, GROUP_SIZE)
        if GQA_INTERLEAVE:
            q_head_ids = kv_head_id + g_offsets * NUM_KV_HEADS
        else:
            q_head_ids = kv_head_id * GROUP_SIZE + g_offsets

        offs_d = tl.arange(0, BLOCK_SIZE_D)
        q_ptrs = (
            q_ptr
            + b_id * stride_qb
            + q_head_ids[:, None] * stride_qh
            + offs_d[None, :] * stride_qd
        )
        q = tl.load(q_ptrs, mask=offs_d[None, :] < HEAD_DIM, other=0.0)

        m_i = tl.zeros((GROUP_SIZE,), dtype=tl.float32)
        l_i = tl.zeros((GROUP_SIZE,), dtype=tl.float32)
        if SINK_ENABLED:
            s_h = tl.load(
                sinks_ptr + q_head_ids * stride_sink_head,
                mask=q_head_ids < NUM_Q_HEADS,
                other=-float("inf"),
            ).to(tl.float32)
            m_i = m_i + s_h
            l_i = l_i + 1.0
        else:
            m_i = m_i - float("inf")
        acc = tl.zeros((GROUP_SIZE, BLOCK_SIZE_D), dtype=tl.float32)

        num_kv_blocks = tl.cdiv(kv_seq_len, BLOCK_SIZE_N)

        for kv_block_id in range(0, num_kv_blocks):
            kv_block_start_in_seq = kv_block_id * BLOCK_SIZE_N
            kv_block_end_in_seq = tl.minimum(
                kv_block_start_in_seq + BLOCK_SIZE_N, kv_seq_len
            )
            kv_block_len = kv_block_end_in_seq - kv_block_start_in_seq

            logical_page_id = tl.minimum(
                kv_block_start_in_seq // PAGE_SIZE, MAX_PAGES - 1
            )
            kv_block_start_in_page = kv_block_start_in_seq % PAGE_SIZE
            physical_page_id = tl.load(
                block_tables_ptr
                + b_id * stride_bt_batch
                + logical_page_id * stride_bt_block
            )

            K_T_block_ptr = tl.make_block_ptr(
                base=k_cache_ptr
                + physical_page_id * stride_k_block
                + kv_head_id * stride_k_head
                + kv_block_start_in_page * stride_k_blksz,
                shape=(HEAD_DIM, kv_block_len),
                strides=(stride_k_dim, stride_k_blksz),
                offsets=(0, 0),
                block_shape=(BLOCK_SIZE_D, BLOCK_SIZE_N),
                order=(0, 1),
            )
            V_block_ptr = tl.make_block_ptr(
                base=v_cache_ptr
                + physical_page_id * stride_v_block
                + kv_head_id * stride_v_head
                + kv_block_start_in_page * stride_v_blksz,
                shape=(kv_block_len, HEAD_DIM),
                strides=(stride_v_blksz, stride_v_dim),
                offsets=(0, 0),
                block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_D),
                order=(1, 0),
            )

            mask = tl.arange(0, BLOCK_SIZE_N) < kv_block_len

            k_T = tl.load(K_T_block_ptr, boundary_check=(0, 1), padding_option="zero")
            v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

            qk = tl.dot(q, k_T)
            qk = qk * softmax_scale
            qk = tl.where(mask[None, :], qk, float("-inf"))

            m_ij = tl.maximum(
                m_i,
                tl.max(qk, 1, propagate_nan=True),
                propagate_nan=tl.PropagateNan.ALL,
            )
            qk = qk - m_ij[:, None]

            p = tl.math.exp(qk)
            p_cast = p.to(k_T.dtype)
            pv = tl.dot(p_cast, v)

            l_ij = tl.sum(p, 1)
            alpha = tl.math.exp(m_i - m_ij)

            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, None] + pv

            m_i = m_ij

        if kv_seq_len > 0:
            acc = acc / l_i[:, None]

        o_ptrs = (
            o_ptr
            + b_id * stride_ob
            + q_head_ids[:, None] * stride_oh
            + offs_d[None, :] * stride_od
        )
        tl.store(o_ptrs, acc.to(o_ptr.type.element_ty), mask=offs_d[None, :] < HEAD_DIM)


def _kv_strides(cache):
    """Map page-major ``[pages, page_size, heads, dim]`` strides to the kernel.

    The kernels receive the cache strides in ``(page, head, page_position,
    dim)`` order, so the head and page-position strides are swapped here.
    """
    return (
        cache.stride(0),
        cache.stride(2),
        cache.stride(1),
        cache.stride(3),
    )


def _prepare_pages(block_tables):
    """Return int32 page indices for the kernels, copying only if needed."""
    if block_tables.dtype == torch.int32:
        return block_tables
    return block_tables.to(torch.int32)


def paged_attention_sink_prefill(
    q,
    k_cache,
    v_cache,
    cu_q_lens,
    seqlens_kv,
    block_tables,
    sinks,
    gqa_interleave,
    softmax_scale,
):
    """Launch the scheduled causal prefill and return ``[T, Hq, D]`` output."""
    num_q_heads = q.shape[1]
    num_kv_heads = k_cache.shape[2]
    page_size = k_cache.shape[1]
    head_dim = q.shape[2]

    block_size_n = min(128, triton.next_power_of_2(page_size))
    cube_num = _num_cores("cube")

    task_b, task_q_block, task_q_head, core_task_offsets = _build_lpt_schedule(
        cu_q_lens,
        seqlens_kv,
        num_q_heads,
        num_kv_heads,
        q.shape[0],
        _QUERY_TILE,
        block_size_n,
        page_size,
        block_tables.shape[1],
        cube_num,
        gqa_interleave,
    )
    pages = _prepare_pages(block_tables)

    sink_enabled = sinks is not None
    sinks_pass = (
        sinks if sink_enabled else torch.empty(1, dtype=q.dtype, device=q.device)
    )
    aux_mask = _causal_mask_table(q.device)
    output = torch.empty_like(q)
    grid = (cube_num,)

    if not (page_size < 128 and 128 % page_size == 0):
        _paged_prefill_kernel[grid](
            q,
            k_cache,
            v_cache,
            output,
            aux_mask,
            task_b,
            task_q_block,
            task_q_head,
            core_task_offsets,
            cu_q_lens,
            seqlens_kv,
            pages,
            sinks_pass,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            *_kv_strides(k_cache),
            *_kv_strides(v_cache),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            pages.stride(0),
            pages.stride(1),
            aux_mask.stride(0),
            aux_mask.stride(1),
            sinks_pass.stride(0),
            softmax_scale,
            aux_mask.shape[0],
            page_size,
            num_q_heads,
            num_kv_heads,
            gqa_interleave,
            head_dim,
            BLOCK_SIZE_M=_QUERY_TILE,
            BLOCK_SIZE_N=block_size_n,
            BLOCK_SIZE_D=head_dim,
            SINK_ENABLED=sink_enabled,
            limit_auto_multi_buffer_buffer="no-limit",
            hfusion_enable_multiple_consumer_fusion=True,
            intra_cache_num=3,
            inter_cache_num=2,
            enable_buffer_insert_optimization=True,
            enable_ub_refine_opt=True,
        )
        return output

    page_aggregation_num = 128 // page_size
    _paged_prefill_page_aggregation_kernel[grid](
        q,
        k_cache,
        v_cache,
        output,
        aux_mask,
        task_b,
        task_q_block,
        task_q_head,
        core_task_offsets,
        pages.shape[1],
        cu_q_lens,
        seqlens_kv,
        pages,
        sinks_pass,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        *_kv_strides(k_cache),
        *_kv_strides(v_cache),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        pages.stride(0),
        pages.stride(1),
        aux_mask.stride(0),
        aux_mask.stride(1),
        sinks_pass.stride(0),
        softmax_scale,
        aux_mask.shape[0],
        page_size,
        num_q_heads,
        num_kv_heads,
        gqa_interleave,
        head_dim,
        BLOCK_SIZE_M=_QUERY_TILE,
        BLOCK_SIZE_N=block_size_n,
        BLOCK_SIZE_D=head_dim,
        PAGE_AGGREGATION_NUM=page_aggregation_num,
        SINK_ENABLED=sink_enabled,
        enable_dynamic_cv_pipeline=True,
        enable_cube_block_merge=True,
        enable_buffer_insert_optimization=True,
        enable_ub_refine_opt=True,
    )
    return output


def paged_attention_sink_decode(
    q,
    k_cache,
    v_cache,
    seqlens,
    block_tables,
    sinks,
    gqa_interleave,
    softmax_scale,
    max_kv_len=None,
):
    """Launch paged decode, optionally splitting long KV scans across cores.

    ``max_kv_len`` lets the caller supply a known upper bound so no
    device-to-host read is needed to choose the split path.
    """
    batch_size, num_q_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[2]
    page_size = k_cache.shape[1]

    output = torch.empty_like(q)
    if batch_size == 0:
        return output

    pages = _prepare_pages(block_tables)
    block_size_d = triton.next_power_of_2(head_dim)
    block_size_n = min(128, triton.next_power_of_2(page_size))
    group_size = num_q_heads // num_kv_heads
    cube_num = _num_cores("cube")

    # A split needs the longest KV row on the host. Callers can pass that bound
    # to avoid the read; otherwise only split candidates read it, and batches
    # large enough to fill the machine clamp page indices instead.
    if max_kv_len is None:
        max_kv_len = 0
        if batch_size * num_kv_heads < _FD_CORE_RATIO * cube_num:
            max_kv_len = int(seqlens.max().item())
    if max_kv_len and triton.cdiv(max_kv_len, page_size) > pages.shape[1]:
        raise ValueError(
            "block_tables has {} pages per request but a KV length of {} "
            "needs {}".format(
                pages.shape[1], max_kv_len, triton.cdiv(max_kv_len, page_size)
            )
        )

    sink_enabled = sinks is not None
    sinks_pass = (
        sinks if sink_enabled else torch.empty(1, dtype=q.dtype, device=q.device)
    )

    if _should_use_flash_decode(
        batch_size, num_kv_heads, group_size, max_kv_len, cube_num
    ):
        kv_split_parts = _compute_kv_split_parts(
            batch_size, num_kv_heads, max_kv_len, cube_num
        )
        vector_num = _num_cores("vector")

        num_ws_tasks = batch_size * num_kv_heads * kv_split_parts
        acc_ws = torch.empty(
            (num_ws_tasks, group_size, head_dim),
            dtype=torch.float32,
            device=q.device,
        )
        lse_ws = torch.full(
            (num_ws_tasks, group_size),
            float("-inf"),
            dtype=torch.float32,
            device=q.device,
        )

        # Phase 1: cube cores, one partial softmax per (request, head, split).
        # The sink is excluded here so it cannot be counted once per split.
        _paged_decode_fd_kernel[(cube_num,)](
            q,
            k_cache,
            v_cache,
            seqlens,
            pages,
            acc_ws,
            lse_ws,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            *_kv_strides(k_cache),
            *_kv_strides(v_cache),
            pages.stride(0),
            pages.stride(1),
            acc_ws.stride(0),
            acc_ws.stride(1),
            acc_ws.stride(2),
            lse_ws.stride(0),
            lse_ws.stride(1),
            softmax_scale,
            batch_size,
            KV_SPLIT_PARTS=kv_split_parts,
            NUM_Q_HEADS=num_q_heads,
            NUM_KV_HEADS=num_kv_heads,
            GQA_INTERLEAVE=gqa_interleave,
            HEAD_DIM=head_dim,
            PAGE_SIZE=page_size,
            BLOCK_SIZE_D=block_size_d,
            BLOCK_SIZE_N=block_size_n,
        )

        # Phase 2: vector cores, merge the splits and fold in the sink once.
        _paged_decode_fd_reduce_kernel[(vector_num,)](
            acc_ws,
            lse_ws,
            output,
            sinks_pass,
            acc_ws.stride(0),
            acc_ws.stride(1),
            acc_ws.stride(2),
            lse_ws.stride(0),
            lse_ws.stride(1),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            sinks_pass.stride(0),
            batch_size,
            KV_SPLIT_PARTS=kv_split_parts,
            NUM_Q_HEADS=num_q_heads,
            NUM_KV_HEADS=num_kv_heads,
            GQA_INTERLEAVE=gqa_interleave,
            HEAD_DIM=head_dim,
            BLOCK_SIZE_D=block_size_d,
            SINK_ENABLED=sink_enabled,
        )
        return output

    _paged_decode_kernel[(cube_num,)](
        q,
        k_cache,
        v_cache,
        output,
        seqlens,
        pages,
        sinks_pass,
        batch_size,
        pages.shape[1],
        q.stride(0),
        q.stride(1),
        q.stride(2),
        *_kv_strides(k_cache),
        *_kv_strides(v_cache),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        pages.stride(0),
        pages.stride(1),
        sinks_pass.stride(0),
        softmax_scale,
        num_q_heads,
        num_kv_heads,
        gqa_interleave,
        head_dim,
        page_size,
        BLOCK_SIZE_D=block_size_d,
        BLOCK_SIZE_N=block_size_n,
        SINK_ENABLED=sink_enabled,
    )
    return output
