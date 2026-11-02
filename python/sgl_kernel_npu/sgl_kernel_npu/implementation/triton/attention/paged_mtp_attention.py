# SPDX-License-Identifier: Apache-2.0
"""Device-length paged GQA for static DSpark draft blocks.

Adapted from shengzhaotian/sglang@d1bf05bd30. The public operator validates
metadata before dispatching here; this module allocates and launches kernels.

``seq_lens`` contains the valid KV end, including the current query block.
``max_kv_len`` is only a scheduling bound; it never controls valid KV reads.
The query block and GQA heads share a matrix tile. Long contexts can split
the KV scan across cores and merge the partial softmax results.
"""

from functools import lru_cache

import torch
import triton
import triton.language as tl


@triton.jit
def _paged_mtp_kernel(
    Q,
    K,
    V,
    O,
    SeqLens,
    CuQ,
    Pages,
    Acc,
    Lse,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_kp,
    stride_kt,
    stride_vp,
    stride_vt,
    stride_ot,
    stride_oh,
    stride_od,
    stride_pb,
    stride_pp,
    scale,
    BATCH: tl.constexpr,
    HQ: tl.constexpr,
    HKV: tl.constexpr,
    DIM: tl.constexpr,
    PAGE: tl.constexpr,
    GAMMA: tl.constexpr,
    SPLITS: tl.constexpr,
    Q_TILE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    GROUP: tl.constexpr = HQ // HKV
    for task in range(tl.program_id(0), BATCH * HKV * SPLITS, tl.num_programs(0)):
        split = task % SPLITS
        kv_head = (task // SPLITS) % HKV
        batch = task // (SPLITS * HKV)
        kv_len = tl.load(SeqLens + batch)
        q_start = tl.load(CuQ + batch)
        chunk = tl.cdiv(tl.cdiv(kv_len, SPLITS), PAGE) * PAGE
        kv_start = split * chunk
        kv_end = tl.minimum(kv_start + chunk, kv_len)

        rows = tl.arange(0, Q_TILE)
        q_index = rows // GROUP
        q_head = kv_head * GROUP + rows % GROUP
        dims = tl.arange(0, BLOCK_D)
        # Keep row bounds affine. Quotient-based masks have produced A5 tail
        # writes for non-power-of-two GQA groups, including the next request.
        q = tl.load(
            Q
            + (q_start + q_index[:, None]) * stride_qt
            + q_head[:, None] * stride_qh
            + dims[None, :] * stride_qd,
            mask=(rows[:, None] < GAMMA * GROUP) & (dims[None, :] < DIM),
            other=0.0,
        )
        maximum = tl.full((Q_TILE,), float("-inf"), tl.float32)
        denominator = tl.zeros((Q_TILE,), tl.float32)
        acc = tl.zeros((Q_TILE, BLOCK_D), tl.float32)

        for block in range(tl.cdiv(tl.maximum(kv_end - kv_start, 0), BLOCK_N)):
            start = kv_start + block * BLOCK_N
            count = tl.minimum(BLOCK_N, kv_end - start)
            physical_page = tl.load(
                Pages + batch * stride_pb + (start // PAGE) * stride_pp
            )
            in_page = start % PAGE
            k_ptr = tl.make_block_ptr(
                base=K
                + physical_page * stride_kp
                + kv_head * DIM
                + in_page * stride_kt,
                shape=(DIM, count),
                strides=(1, stride_kt),
                offsets=(0, 0),
                block_shape=(BLOCK_D, BLOCK_N),
                order=(0, 1),
            )
            v_ptr = tl.make_block_ptr(
                base=V
                + physical_page * stride_vp
                + kv_head * DIM
                + in_page * stride_vt,
                shape=(count, DIM),
                strides=(stride_vt, 1),
                offsets=(0, 0),
                block_shape=(BLOCK_N, BLOCK_D),
                order=(1, 0),
            )
            keys = tl.load(k_ptr, boundary_check=(0, 1), padding_option="zero")
            values = tl.load(v_ptr, boundary_check=(0, 1), padding_option="zero")
            scores = tl.dot(q, keys) * scale
            kv_positions = start + tl.arange(0, BLOCK_N)
            allowed = (kv_positions[None, :] < kv_end) & (rows[:, None] < GAMMA * GROUP)
            if CAUSAL:
                allowed = allowed & (
                    kv_positions[None, :] <= kv_len - GAMMA + q_index[:, None]
                )
            scores = tl.where(allowed, scores, float("-inf"))
            next_max = tl.maximum(maximum, tl.max(scores, 1))
            # A split can be empty for a causal query or a padded request.
            safe_max = tl.where(next_max == float("-inf"), 0.0, next_max)
            probability = tl.exp(scores - safe_max[:, None])
            correction = tl.exp(maximum - safe_max)
            denominator = denominator * correction + tl.sum(probability, 1)
            acc = acc * correction[:, None] + tl.dot(probability.to(keys.dtype), values)
            maximum = next_max

        safe_denominator = tl.where(denominator > 0, denominator, 1.0)
        acc = acc / safe_denominator[:, None]
        if SPLITS == 1:
            tl.store(
                O
                + (q_start + q_index[:, None]) * stride_ot
                + q_head[:, None] * stride_oh
                + dims[None, :] * stride_od,
                acc,
                mask=(rows[:, None] < GAMMA * GROUP) & (dims[None, :] < DIM),
            )
        else:
            tl.store(
                Acc + task * Q_TILE * DIM + rows[:, None] * DIM + dims[None, :],
                acc,
                mask=dims[None, :] < DIM,
            )
            lse = tl.where(
                denominator > 0, maximum + tl.log(safe_denominator), float("-inf")
            )
            tl.store(Lse + task * Q_TILE + rows, lse)


@triton.jit
def _reduce_mtp_kernel(
    Acc,
    Lse,
    O,
    CuQ,
    stride_ot,
    stride_oh,
    stride_od,
    BATCH: tl.constexpr,
    HQ: tl.constexpr,
    HKV: tl.constexpr,
    DIM: tl.constexpr,
    GAMMA: tl.constexpr,
    SPLITS: tl.constexpr,
    Q_TILE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    GROUP: tl.constexpr = HQ // HKV
    for task in range(tl.program_id(0), BATCH * HKV, tl.num_programs(0)):
        batch = task // HKV
        kv_head = task % HKV
        q_start = tl.load(CuQ + batch)
        rows = tl.arange(0, Q_TILE)
        dims = tl.arange(0, BLOCK_D)
        maximum = tl.full((Q_TILE,), float("-inf"), tl.float32)
        for split in tl.static_range(SPLITS):
            lse = tl.load(Lse + (task * SPLITS + split) * Q_TILE + rows)
            maximum = tl.maximum(maximum, lse)
        safe_max = tl.where(maximum == float("-inf"), 0.0, maximum)
        denominator = tl.zeros((Q_TILE,), tl.float32)
        result = tl.zeros((Q_TILE, BLOCK_D), tl.float32)
        for split in tl.static_range(SPLITS):
            workspace_task = task * SPLITS + split
            lse = tl.load(Lse + workspace_task * Q_TILE + rows)
            weight = tl.exp(lse - safe_max)
            partial = tl.load(
                Acc
                + workspace_task * Q_TILE * DIM
                + rows[:, None] * DIM
                + dims[None, :],
                mask=dims[None, :] < DIM,
                other=0.0,
            )
            result += weight[:, None] * partial
            denominator += weight
        result /= tl.where(denominator > 0, denominator, 1.0)[:, None]
        q_index = rows // GROUP
        q_head = kv_head * GROUP + rows % GROUP
        tl.store(
            O
            + (q_start + q_index[:, None]) * stride_ot
            + q_head[:, None] * stride_oh
            + dims[None, :] * stride_od,
            result,
            mask=(rows[:, None] < GAMMA * GROUP) & (dims[None, :] < DIM),
        )


@triton.jit
def _a3_grouped_mtp_kernel(
    Q,
    K,
    V,
    O,
    SeqLens,
    CuQ,
    Pages,
    Acc,
    Lse,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_kp,
    stride_kt,
    stride_vp,
    stride_vt,
    stride_ot,
    stride_oh,
    stride_od,
    stride_pb,
    stride_pp,
    scale,
    BATCH: tl.constexpr,
    HQ: tl.constexpr,
    HKV: tl.constexpr,
    DIM: tl.constexpr,
    PAGE: tl.constexpr,
    GAMMA: tl.constexpr,
    SPLITS: tl.constexpr,
    Q_TILE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    GROUP: tl.constexpr = HQ // HKV
    for task in range(tl.program_id(0), BATCH * HKV * SPLITS, tl.num_programs(0)):
        split = task % SPLITS
        kv_head = (task // SPLITS) % HKV
        batch = task // (SPLITS * HKV)
        kv_len = tl.load(SeqLens + batch)
        q_start = tl.load(CuQ + batch)
        chunk = tl.cdiv(tl.cdiv(kv_len, SPLITS), PAGE) * PAGE
        kv_start = split * chunk
        kv_end = tl.minimum(kv_start + chunk, kv_len)

        rows = tl.arange(0, Q_TILE)
        q_index = rows // GROUP
        q_head = kv_head * GROUP + rows % GROUP
        dims = tl.arange(0, BLOCK_D)
        # Affine rows from [batch, kv_head, query, group, dim] avoid A3's
        # incorrect dot results with folded query/head gather indexing.
        q = tl.load(
            Q
            + (batch * HKV + kv_head) * GAMMA * GROUP * DIM
            + rows[:, None] * DIM
            + dims[None, :],
            mask=(q_index[:, None] < GAMMA) & (dims[None, :] < DIM),
            other=0.0,
        )
        maximum = tl.full((Q_TILE,), float("-inf"), tl.float32)
        denominator = tl.zeros((Q_TILE,), tl.float32)
        acc = tl.zeros((Q_TILE, BLOCK_D), tl.float32)

        for block in range(tl.cdiv(tl.maximum(kv_end - kv_start, 0), BLOCK_N)):
            start = kv_start + block * BLOCK_N
            count = tl.minimum(BLOCK_N, kv_end - start)
            physical_page = tl.load(
                Pages + batch * stride_pb + (start // PAGE) * stride_pp
            )
            in_page = start % PAGE
            k_ptr = tl.make_block_ptr(
                base=K
                + physical_page * stride_kp
                + kv_head * DIM
                + in_page * stride_kt,
                shape=(DIM, count),
                strides=(1, stride_kt),
                offsets=(0, 0),
                block_shape=(BLOCK_D, BLOCK_N),
                order=(0, 1),
            )
            v_ptr = tl.make_block_ptr(
                base=V
                + physical_page * stride_vp
                + kv_head * DIM
                + in_page * stride_vt,
                shape=(count, DIM),
                strides=(stride_vt, 1),
                offsets=(0, 0),
                block_shape=(BLOCK_N, BLOCK_D),
                order=(1, 0),
            )
            keys = tl.load(k_ptr, boundary_check=(0, 1), padding_option="zero")
            values = tl.load(v_ptr, boundary_check=(0, 1), padding_option="zero")
            scores = tl.dot(q, keys) * scale
            kv_positions = start + tl.arange(0, BLOCK_N)
            allowed = (kv_positions[None, :] < kv_end) & (q_index[:, None] < GAMMA)
            if CAUSAL:
                allowed = allowed & (
                    kv_positions[None, :] <= kv_len - GAMMA + q_index[:, None]
                )
            scores = tl.where(allowed, scores, float("-inf"))
            next_max = tl.maximum(maximum, tl.max(scores, 1))
            # A split can be empty for a causal query or a padded request.
            safe_max = tl.where(next_max == float("-inf"), 0.0, next_max)
            probability = tl.exp(scores - safe_max[:, None])
            correction = tl.exp(maximum - safe_max)
            denominator = denominator * correction + tl.sum(probability, 1)
            acc = acc * correction[:, None] + tl.dot(probability.to(keys.dtype), values)
            maximum = next_max

        safe_denominator = tl.where(denominator > 0, denominator, 1.0)
        acc = acc / safe_denominator[:, None]
        if SPLITS == 1:
            tl.store(
                O
                + (q_start + q_index[:, None]) * stride_ot
                + q_head[:, None] * stride_oh
                + dims[None, :] * stride_od,
                acc,
                mask=(q_index[:, None] < GAMMA) & (dims[None, :] < DIM),
            )
        else:
            tl.store(
                Acc + task * Q_TILE * DIM + rows[:, None] * DIM + dims[None, :],
                acc,
                mask=dims[None, :] < DIM,
            )
            lse = tl.where(
                denominator > 0, maximum + tl.log(safe_denominator), float("-inf")
            )
            tl.store(Lse + task * Q_TILE + rows, lse)


@triton.jit
def _a3_grouped_reduce_kernel(
    Acc,
    Lse,
    O,
    CuQ,
    stride_ot,
    stride_oh,
    stride_od,
    BATCH: tl.constexpr,
    HQ: tl.constexpr,
    HKV: tl.constexpr,
    DIM: tl.constexpr,
    GAMMA: tl.constexpr,
    SPLITS: tl.constexpr,
    Q_TILE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    GROUP: tl.constexpr = HQ // HKV
    for task in range(tl.program_id(0), BATCH * HKV, tl.num_programs(0)):
        batch = task // HKV
        kv_head = task % HKV
        q_start = tl.load(CuQ + batch)
        rows = tl.arange(0, Q_TILE)
        dims = tl.arange(0, BLOCK_D)
        maximum = tl.full((Q_TILE,), float("-inf"), tl.float32)
        for split in tl.static_range(SPLITS):
            lse = tl.load(Lse + (task * SPLITS + split) * Q_TILE + rows)
            maximum = tl.maximum(maximum, lse)
        safe_max = tl.where(maximum == float("-inf"), 0.0, maximum)
        denominator = tl.zeros((Q_TILE,), tl.float32)
        result = tl.zeros((Q_TILE, BLOCK_D), tl.float32)
        for split in tl.static_range(SPLITS):
            workspace_task = task * SPLITS + split
            lse = tl.load(Lse + workspace_task * Q_TILE + rows)
            weight = tl.exp(lse - safe_max)
            partial = tl.load(
                Acc
                + workspace_task * Q_TILE * DIM
                + rows[:, None] * DIM
                + dims[None, :],
                mask=dims[None, :] < DIM,
                other=0.0,
            )
            result += weight[:, None] * partial
            denominator += weight
        result /= tl.where(denominator > 0, denominator, 1.0)[:, None]
        q_index = rows // GROUP
        q_head = kv_head * GROUP + rows % GROUP
        tl.store(
            O
            + (q_start + q_index[:, None]) * stride_ot
            + q_head[:, None] * stride_oh
            + dims[None, :] * stride_od,
            result,
            mask=(q_index[:, None] < GAMMA) & (dims[None, :] < DIM),
        )


@triton.jit
def _a3_single_mtp_kernel(
    Q,
    K,
    V,
    O,
    SeqLens,
    CuQ,
    Pages,
    Acc,
    Lse,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_kp,
    stride_kt,
    stride_vp,
    stride_vt,
    stride_ot,
    stride_oh,
    stride_od,
    stride_pb,
    stride_pp,
    scale,
    BATCH: tl.constexpr,
    HQ: tl.constexpr,
    HKV: tl.constexpr,
    DIM: tl.constexpr,
    PAGE: tl.constexpr,
    GAMMA: tl.constexpr,
    SPLITS: tl.constexpr,
    Q_TILE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
    SINGLE_QUERY_TASK: tl.constexpr,
    KV_BLOCK: tl.constexpr,
):
    GROUP: tl.constexpr = HQ // HKV
    TASK_QUERIES: tl.constexpr = GAMMA if SINGLE_QUERY_TASK else 1
    for task in range(
        tl.program_id(0), BATCH * HKV * TASK_QUERIES * SPLITS, tl.num_programs(0)
    ):
        split = task % SPLITS
        query_id = (task // SPLITS) % TASK_QUERIES
        kv_head = (task // (SPLITS * TASK_QUERIES)) % HKV
        batch = task // (SPLITS * TASK_QUERIES * HKV)
        kv_len = tl.load(SeqLens + batch)
        q_start = tl.load(CuQ + batch)
        chunk = tl.cdiv(tl.cdiv(kv_len, SPLITS), PAGE) * PAGE
        kv_start = split * chunk
        kv_end = tl.minimum(kv_start + chunk, kv_len)

        rows = tl.arange(0, Q_TILE)
        if SINGLE_QUERY_TASK:
            q_index = tl.full((Q_TILE,), 0, tl.int32) + query_id
            q_head = kv_head * GROUP + rows
            q_valid = rows < GROUP
        else:
            q_index = rows // GROUP
            q_head = kv_head * GROUP + rows % GROUP
            q_valid = q_index < GAMMA
        dims = tl.arange(0, BLOCK_D)
        q = tl.load(
            Q
            + (q_start + q_index[:, None]) * stride_qt
            + q_head[:, None] * stride_qh
            + dims[None, :] * stride_qd,
            mask=q_valid[:, None] & (dims[None, :] < DIM),
            other=0.0,
        )
        maximum = tl.full((Q_TILE,), float("-inf"), tl.float32)
        denominator = tl.zeros((Q_TILE,), tl.float32)
        acc = tl.zeros((Q_TILE, BLOCK_D), tl.float32)

        for block in range(tl.cdiv(tl.maximum(kv_end - kv_start, 0), KV_BLOCK)):
            start = kv_start + block * KV_BLOCK
            count = tl.minimum(KV_BLOCK, kv_end - start)
            physical_page = tl.load(
                Pages + batch * stride_pb + (start // PAGE) * stride_pp
            )
            in_page = start % PAGE
            k_ptr = tl.make_block_ptr(
                base=K
                + physical_page * stride_kp
                + kv_head * DIM
                + in_page * stride_kt,
                shape=(DIM, count),
                strides=(1, stride_kt),
                offsets=(0, 0),
                block_shape=(BLOCK_D, BLOCK_N),
                order=(0, 1),
            )
            v_ptr = tl.make_block_ptr(
                base=V
                + physical_page * stride_vp
                + kv_head * DIM
                + in_page * stride_vt,
                shape=(count, DIM),
                strides=(stride_vt, 1),
                offsets=(0, 0),
                block_shape=(BLOCK_N, BLOCK_D),
                order=(1, 0),
            )
            keys = tl.load(k_ptr, boundary_check=(0, 1), padding_option="zero")
            values = tl.load(v_ptr, boundary_check=(0, 1), padding_option="zero")
            scores = tl.dot(q, keys) * scale
            kv_positions = start + tl.arange(0, BLOCK_N)
            allowed = (tl.arange(0, BLOCK_N)[None, :] < count) & q_valid[:, None]
            if CAUSAL:
                allowed = allowed & (
                    kv_positions[None, :] <= kv_len - GAMMA + q_index[:, None]
                )
            scores = tl.where(allowed, scores, float("-inf"))
            next_max = tl.maximum(maximum, tl.max(scores, 1))
            # A split can be empty for a causal query or a padded request.
            safe_max = tl.where(next_max == float("-inf"), 0.0, next_max)
            probability = tl.exp(scores - safe_max[:, None])
            correction = tl.exp(maximum - safe_max)
            denominator = denominator * correction + tl.sum(probability, 1)
            acc = acc * correction[:, None] + tl.dot(probability.to(keys.dtype), values)
            maximum = next_max

        safe_denominator = tl.where(denominator > 0, denominator, 1.0)
        acc = acc / safe_denominator[:, None]
        if SPLITS == 1:
            tl.store(
                O
                + (q_start + q_index[:, None]) * stride_ot
                + q_head[:, None] * stride_oh
                + dims[None, :] * stride_od,
                acc,
                mask=q_valid[:, None] & (dims[None, :] < DIM),
            )
        else:
            tl.store(
                Acc + task * Q_TILE * DIM + rows[:, None] * DIM + dims[None, :],
                acc,
                mask=dims[None, :] < DIM,
            )
            lse = tl.where(
                denominator > 0, maximum + tl.log(safe_denominator), float("-inf")
            )
            tl.store(Lse + task * Q_TILE + rows, lse)


@triton.jit
def _a3_single_reduce_kernel(
    Acc,
    Lse,
    O,
    CuQ,
    stride_ot,
    stride_oh,
    stride_od,
    BATCH: tl.constexpr,
    HQ: tl.constexpr,
    HKV: tl.constexpr,
    DIM: tl.constexpr,
    GAMMA: tl.constexpr,
    SPLITS: tl.constexpr,
    Q_TILE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SINGLE_QUERY_TASK: tl.constexpr,
):
    GROUP: tl.constexpr = HQ // HKV
    TASK_QUERIES: tl.constexpr = GAMMA if SINGLE_QUERY_TASK else 1
    for task in range(tl.program_id(0), BATCH * HKV * TASK_QUERIES, tl.num_programs(0)):
        query_id = task % TASK_QUERIES
        batch = task // (HKV * TASK_QUERIES)
        kv_head = (task // TASK_QUERIES) % HKV
        q_start = tl.load(CuQ + batch)
        rows = tl.arange(0, Q_TILE)
        dims = tl.arange(0, BLOCK_D)
        maximum = tl.full((Q_TILE,), float("-inf"), tl.float32)
        for split in tl.static_range(SPLITS):
            lse = tl.load(Lse + (task * SPLITS + split) * Q_TILE + rows)
            maximum = tl.maximum(maximum, lse)
        safe_max = tl.where(maximum == float("-inf"), 0.0, maximum)
        denominator = tl.zeros((Q_TILE,), tl.float32)
        result = tl.zeros((Q_TILE, BLOCK_D), tl.float32)
        for split in tl.static_range(SPLITS):
            workspace_task = task * SPLITS + split
            lse = tl.load(Lse + workspace_task * Q_TILE + rows)
            weight = tl.exp(lse - safe_max)
            partial = tl.load(
                Acc
                + workspace_task * Q_TILE * DIM
                + rows[:, None] * DIM
                + dims[None, :],
                mask=dims[None, :] < DIM,
                other=0.0,
            )
            result += weight[:, None] * partial
            denominator += weight
        result /= tl.where(denominator > 0, denominator, 1.0)[:, None]
        if SINGLE_QUERY_TASK:
            q_index = tl.full((Q_TILE,), 0, tl.int32) + query_id
            q_head = kv_head * GROUP + rows
            q_valid = rows < GROUP
        else:
            q_index = rows // GROUP
            q_head = kv_head * GROUP + rows % GROUP
            q_valid = q_index < GAMMA
        tl.store(
            O
            + (q_start + q_index[:, None]) * stride_ot
            + q_head[:, None] * stride_oh
            + dims[None, :] * stride_od,
            result,
            mask=q_valid[:, None] & (dims[None, :] < DIM),
        )


@lru_cache(maxsize=1)
def _get_num_cores() -> tuple[int, int]:
    properties = triton.runtime.driver.active.utils.get_device_properties("npu")
    return properties["num_aicore"], properties["num_vectorcore"]


@lru_cache(maxsize=32)
def _is_a3_device(device: torch.device) -> bool:
    return device.type == "npu" and torch.npu.get_device_name(device.index).startswith(
        "Ascend910_93"
    )


def paged_mtp_attention_decode(
    *,
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    cu_q_lens: torch.Tensor,
    block_tables: torch.Tensor,
    page_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    gamma: int,
    softmax_scale: float,
    max_kv_len: int,
    causal: bool = False,
) -> torch.Tensor:
    """Attend to actual device KV lengths; ``max_kv_len`` only sizes the work.

    Inputs use FIA's ND paged layout ``[pages, page_size, kv_heads * dim]``.
    Zero lengths denote padding, whose output is zero. All request rows have
    ``gamma`` queries; ragged verification and sliding windows are unsupported.
    Causal queries are right aligned: query i sees KV positions through
    ``seq_lens[b] - gamma + i``. A3 packs Q before the grouped matrix scan;
    small pages and irregular groups use a per-query matrix tile instead.
    """
    batch_size = seq_lens.numel()
    if batch_size and _is_a3_device(query.device) and head_dim & (head_dim - 1):
        # A3 matrix tiles use a power-of-two head stride. Zero padding preserves
        # both dot products; keep the caller's scale and crop only the V result.
        padded_dim = triton.next_power_of_2(head_dim)
        padding = padded_dim - head_dim
        padded_query = torch.nn.functional.pad(query, (0, padding))
        padded_k = torch.nn.functional.pad(
            k_cache.view(-1, page_size, num_kv_heads, head_dim), (0, padding)
        ).view(-1, page_size, num_kv_heads * padded_dim)
        padded_v = torch.nn.functional.pad(
            v_cache.view(-1, page_size, num_kv_heads, head_dim), (0, padding)
        ).view(-1, page_size, num_kv_heads * padded_dim)
        padded_output = paged_mtp_attention_decode(
            query=padded_query,
            k_cache=padded_k,
            v_cache=padded_v,
            seq_lens=seq_lens,
            cu_q_lens=cu_q_lens,
            block_tables=block_tables,
            page_size=page_size,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=padded_dim,
            gamma=gamma,
            softmax_scale=softmax_scale,
            max_kv_len=max_kv_len,
            causal=causal,
        )
        return padded_output[..., :head_dim].contiguous()
    output = torch.empty_like(query)
    if not batch_size:
        return output
    cube_cores, vector_cores = _get_num_cores()
    group_size = num_q_heads // num_kv_heads
    is_a3 = _is_a3_device(query.device)
    # Keep A5's existing layout. A3 needs an affine matrix source for the
    # multi-query GQA tile; the packing operation is captured in NPU graphs.
    packed_query = (
        is_a3
        and gamma > 1
        and group_size > 1
        and not (group_size & (group_size - 1))
        and page_size >= 32
    )
    single_query = is_a3 and not packed_query
    query_strides = query.stride()
    if packed_query:
        query = (
            query.reshape(batch_size, gamma, num_kv_heads, group_size, head_dim)
            .permute(0, 2, 1, 3, 4)
            .contiguous()
        )
    tasks = batch_size * num_kv_heads * (gamma if single_query else 1)
    splits = 1
    if tasks < cube_cores and max_kv_len >= (256 if group_size == 1 else 2048):
        splits = max(1, min(cube_cores // tasks, max_kv_len // 256))
    q_tile = max(
        32 if single_query else 16,
        triton.next_power_of_2(group_size if single_query else gamma * group_size),
    )
    block_d = (
        max(32, triton.next_power_of_2(head_dim))
        if single_query
        else triton.next_power_of_2(head_dim)
    )
    kv_block = min(128, page_size)
    if is_a3 and q_tile * block_d >= 8192:
        # Bound the wide-head tile's A3 local-memory footprint.
        kv_block = min(kv_block, 64)
    # Pad A3 per-query tiles to 32 for aligned small-head/page transfers.
    # KV_BLOCK still advances by the physical page width for pages below 32.
    block_n = max(32, kv_block) if single_query else kv_block
    # Separate JIT definitions preserve the verified A3 layouts. Sharing a
    # conditional Q-layout kernel also produced incorrect causal dot results.
    main_kernel = (
        _a3_grouped_mtp_kernel
        if packed_query
        else (_a3_single_mtp_kernel if single_query else _paged_mtp_kernel)
    )
    reduce_kernel = (
        _a3_grouped_reduce_kernel
        if packed_query
        else (_a3_single_reduce_kernel if single_query else _reduce_mtp_kernel)
    )
    main_extra = (
        {"SINGLE_QUERY_TASK": True, "KV_BLOCK": kv_block} if single_query else {}
    )
    reduce_extra = {"SINGLE_QUERY_TASK": True} if single_query else {}
    if splits > 1:
        acc = torch.empty(
            (tasks * splits, q_tile, head_dim), dtype=torch.float32, device=query.device
        )
        lse = torch.empty(
            (tasks * splits, q_tile), dtype=torch.float32, device=query.device
        )
    else:
        acc = lse = output  # Unused pointer arguments in the unsplit specialization.
    main_kernel[(cube_cores,)](
        query,
        k_cache,
        v_cache,
        output,
        seq_lens,
        cu_q_lens,
        block_tables,
        acc,
        lse,
        *query_strides,
        k_cache.stride(0),
        k_cache.stride(1),
        v_cache.stride(0),
        v_cache.stride(1),
        *output.stride(),
        *block_tables.stride(),
        softmax_scale,
        BATCH=batch_size,
        HQ=num_q_heads,
        HKV=num_kv_heads,
        DIM=head_dim,
        PAGE=page_size,
        GAMMA=gamma,
        SPLITS=splits,
        Q_TILE=q_tile,
        BLOCK_D=block_d,
        BLOCK_N=block_n,
        CAUSAL=causal,
        **main_extra,
    )
    if splits > 1:
        reduce_kernel[(vector_cores,)](
            acc,
            lse,
            output,
            cu_q_lens,
            *output.stride(),
            BATCH=batch_size,
            HQ=num_q_heads,
            HKV=num_kv_heads,
            DIM=head_dim,
            GAMMA=gamma,
            SPLITS=splits,
            Q_TILE=q_tile,
            BLOCK_D=block_d,
            **reduce_extra,
        )
    return output
