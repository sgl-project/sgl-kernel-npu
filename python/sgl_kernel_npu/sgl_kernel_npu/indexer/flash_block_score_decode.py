from __future__ import annotations

from functools import lru_cache
from typing import Optional

import torch
import triton
import triton.language as tl


def _next_power_of_2(x: int) -> int:
    return 1 << (int(x) - 1).bit_length()


def _floor_power_of_2(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << (int(x).bit_length() - 1)


# ------------------------- Tunable launch configs -------------------------
_SCORE_CHUNK_NW = 4
_SCORE_CHUNK_NS = 2


@lru_cache(maxsize=1)
def _get_vectorcore_num_safe() -> int:
    """Ascend NPU vector-core count (cached; 32 off-NPU fallback)."""
    try:
        props = triton.runtime.driver.active.utils.get_device_properties(
            torch.npu.current_device()
        )
        vc_num = int(props.get("num_vectorcore", -1))
    except Exception:
        return 32
    return max(1, vc_num) if vc_num > 0 else 32


@lru_cache(maxsize=1)
def _get_native_minimax_indexer():
    # torch.ops.npu.minimax_indexer when available, else None (fall back to triton).
    try:
        import sgl_kernel_npu  # noqa: F401  (registers torch.ops.npu.minimax_indexer)

        return getattr(torch.ops.npu, "minimax_indexer", None)
    except (ImportError, RuntimeError):
        return None


def _choose_num_kv_chunks(
    batch_size: int,
    num_kv_heads: int,
    max_num_kv_chunks: int = 8,
) -> int:
    """Choose NUM_KV_CHUNKS (power-of-two) from a vector-core target-grid rule."""
    max_num_kv_chunks = max(1, int(max_num_kv_chunks))
    # Make the cap a power of two, because kernels/merge paths specialize on
    # NUM_KV_CHUNKS and power-of-two values are easier to reason about.
    max_num_kv_chunks = 1 << (max_num_kv_chunks.bit_length() - 1)

    vectorcore_num = _get_vectorcore_num_safe()
    # Oversubscribe vector cores modestly to hide latency, cap stays conservative.
    target_grid = max(1, vectorcore_num * 8)
    denom = max(1, int(batch_size) * int(num_kv_heads))

    target = max(1, min(max_num_kv_chunks, target_grid // denom))
    return 1 << (target.bit_length() - 1)


def _choose_num_score_chunks(
    max_seqblock: int,
    blocks_per_chunk: int = 16,
    max_chunks: int = 32,
    all_seqblock_q: int = 1,
    num_kv_heads: int = 1,
    program_cap: int = 32768,
) -> int:
    """Pick power-of-two block-tile count for the chunked score-only kernel.
    Grid is capped by ``program_cap`` (Ascend launch limit)."""
    if max_seqblock <= 0:
        return 1
    balance = (max_seqblock + max(1, blocks_per_chunk) - 1) // max(1, blocks_per_chunk)
    n = max(1, min(balance, max(1, max_chunks)))
    per_chunk = max(1, all_seqblock_q * max(1, num_kv_heads))
    n = min(n, max(1, program_cap // per_chunk))
    return 1 << (n.bit_length() - 1)


@triton.heuristics(
    {
        "BLOCK_SIZE_CANDIDATES": lambda args: triton.next_power_of_2(
            args["NUM_SCORE_CHUNKS"] * args["topk"]
        ),
    }
)
@triton.jit
def _merge_bnsd_score_topk_candidates_kernel(
    candidate_scores_ptr,  # [C, QH, B, topk]
    candidate_indices_ptr,  # [C, QH, B, topk]
    topk_indices_ptr,  # [QH, B, topk]
    # strides
    stride_cs_c,
    stride_cs_h,
    stride_cs_b,
    stride_cs_t,
    stride_ci_c,
    stride_ci_h,
    stride_ci_b,
    stride_ci_t,
    stride_ti_h,
    stride_ti_b,
    stride_ti_t,
    # meta
    NUM_SCORE_CHUNKS: tl.constexpr,
    topk: tl.constexpr,
    BLOCK_SIZE_CANDIDATES: tl.constexpr,
):
    """Merge fixed-size chunk-local candidates into global TopK indices."""
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    off_candidates = tl.arange(0, BLOCK_SIZE_CANDIDATES)
    candidate_chunk = off_candidates // topk
    candidate_rank = off_candidates - candidate_chunk * topk
    valid_candidate = off_candidates < NUM_SCORE_CHUNKS * topk

    score_offsets = (
        candidate_chunk * stride_cs_c
        + pid_h * stride_cs_h
        + pid_b * stride_cs_b
        + candidate_rank * stride_cs_t
    )
    index_offsets = (
        candidate_chunk * stride_ci_c
        + pid_h * stride_ci_h
        + pid_b * stride_ci_b
        + candidate_rank * stride_ci_t
    )
    candidate_scores = tl.load(
        candidate_scores_ptr + score_offsets,
        mask=valid_candidate,
        other=float("-inf"),
    ).to(tl.float32)
    candidate_indices = tl.load(
        candidate_indices_ptr + index_offsets,
        mask=valid_candidate,
        other=-1,
    ).to(tl.int32)
    candidate_scores = tl.where(
        candidate_indices >= 0,
        candidate_scores,
        float("-inf"),
    )

    for rank in tl.static_range(0, topk):
        best_score = tl.max(candidate_scores, axis=0)
        best_positions = tl.where(
            candidate_scores == best_score,
            off_candidates,
            tl.full((BLOCK_SIZE_CANDIDATES,), BLOCK_SIZE_CANDIDATES, tl.int32),
        )
        best_position = tl.min(best_positions, axis=0)
        selected_index = tl.max(
            tl.where(
                off_candidates == best_position,
                candidate_indices,
                tl.full((BLOCK_SIZE_CANDIDATES,), -1, tl.int32),
            ),
            axis=0,
        )
        tl.store(
            topk_indices_ptr
            + pid_h * stride_ti_h
            + pid_b * stride_ti_b
            + rank * stride_ti_t,
            selected_index,
        )
        candidate_scores = tl.where(
            off_candidates == best_position,
            float("-inf"),
            candidate_scores,
        )


def _merge_bnsd_score_topk_candidates(
    candidate_scores: torch.Tensor,
    candidate_indices: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    """Return global TopK indices from chunk-local score candidates."""
    num_score_chunks, num_q_heads, batch_size, _ = candidate_scores.shape
    topk_indices = torch.empty(
        (num_q_heads, batch_size, topk),
        dtype=torch.int32,
        device=candidate_scores.device,
    )
    _merge_bnsd_score_topk_candidates_kernel[(batch_size, num_q_heads)](
        candidate_scores,
        candidate_indices,
        topk_indices,
        candidate_scores.stride(0),
        candidate_scores.stride(1),
        candidate_scores.stride(2),
        candidate_scores.stride(3),
        candidate_indices.stride(0),
        candidate_indices.stride(1),
        candidate_indices.stride(2),
        candidate_indices.stride(3),
        topk_indices.stride(0),
        topk_indices.stride(1),
        topk_indices.stride(2),
        NUM_SCORE_CHUNKS=num_score_chunks,
        topk=topk,
        num_warps=1,
        num_stages=1,
    )
    return topk_indices


@triton.jit
def _merge_bnsd_score_topk_candidates_impl(
    candidate_scores_ptr,
    candidate_indices_ptr,
    topk_indices_ptr,
    pid_b,
    pid_h,
    stride_cs_c,
    stride_cs_h,
    stride_cs_b,
    stride_cs_t,
    stride_ci_c,
    stride_ci_h,
    stride_ci_b,
    stride_ci_t,
    stride_ti_h,
    stride_ti_b,
    stride_ti_t,
    NUM_SCORE_CHUNKS: tl.constexpr,
    topk: tl.constexpr,
    BLOCK_SIZE_CANDIDATES: tl.constexpr,
):
    off_candidates = tl.arange(0, BLOCK_SIZE_CANDIDATES)
    candidate_chunk = off_candidates // topk
    candidate_rank = off_candidates - candidate_chunk * topk
    valid_candidate = off_candidates < NUM_SCORE_CHUNKS * topk
    candidate_scores = tl.load(
        candidate_scores_ptr
        + candidate_chunk * stride_cs_c
        + pid_h * stride_cs_h
        + pid_b * stride_cs_b
        + candidate_rank * stride_cs_t,
        mask=valid_candidate,
        other=float("-inf"),
    ).to(tl.float32)
    candidate_indices = tl.load(
        candidate_indices_ptr
        + candidate_chunk * stride_ci_c
        + pid_h * stride_ci_h
        + pid_b * stride_ci_b
        + candidate_rank * stride_ci_t,
        mask=valid_candidate,
        other=-1,
    ).to(tl.int32)
    candidate_scores = tl.where(
        candidate_indices >= 0,
        candidate_scores,
        float("-inf"),
    )

    for rank in tl.static_range(0, topk):
        best_score = tl.max(candidate_scores, axis=0)
        best_positions = tl.where(
            candidate_scores == best_score,
            off_candidates,
            tl.full((BLOCK_SIZE_CANDIDATES,), BLOCK_SIZE_CANDIDATES, tl.int32),
        )
        best_position = tl.min(best_positions, axis=0)
        selected_index = tl.max(
            tl.where(
                off_candidates == best_position,
                candidate_indices,
                tl.full((BLOCK_SIZE_CANDIDATES,), -1, tl.int32),
            ),
            axis=0,
        )
        tl.store(
            topk_indices_ptr
            + pid_h * stride_ti_h
            + pid_b * stride_ti_b
            + rank * stride_ti_t,
            selected_index,
        )
        candidate_scores = tl.where(
            off_candidates == best_position,
            float("-inf"),
            candidate_scores,
        )


@triton.jit
def _merge_bnsd_score_topk_candidates_guarded_kernel(
    candidate_scores_ptr,
    candidate_indices_ptr,
    topk_indices_ptr,
    seq_lens,
    stride_sl_b,
    stride_sl_h,
    block_size: tl.constexpr,
    short_max_blocks: tl.constexpr,
    stride_cs_c,
    stride_cs_h,
    stride_cs_b,
    stride_cs_t,
    stride_ci_c,
    stride_ci_h,
    stride_ci_b,
    stride_ci_t,
    stride_ti_h,
    stride_ti_b,
    stride_ti_t,
    NUM_SCORE_CHUNKS: tl.constexpr,
    topk: tl.constexpr,
    BLOCK_SIZE_CANDIDATES: tl.constexpr,
    RUN_SHORT: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    seq_len = tl.load(seq_lens + pid_b * stride_sl_b + pid_h * stride_sl_h)
    num_blocks = tl.cdiv(seq_len, block_size)
    if RUN_SHORT:
        if num_blocks > short_max_blocks:
            return
    else:
        if num_blocks <= short_max_blocks:
            return
    _merge_bnsd_score_topk_candidates_impl(
        candidate_scores_ptr,
        candidate_indices_ptr,
        topk_indices_ptr,
        pid_b,
        pid_h,
        stride_cs_c,
        stride_cs_h,
        stride_cs_b,
        stride_cs_t,
        stride_ci_c,
        stride_ci_h,
        stride_ci_b,
        stride_ci_t,
        stride_ti_h,
        stride_ti_b,
        stride_ti_t,
        NUM_SCORE_CHUNKS=NUM_SCORE_CHUNKS,
        topk=topk,
        BLOCK_SIZE_CANDIDATES=BLOCK_SIZE_CANDIDATES,
    )


def _merge_bnsd_score_topk_candidates_adaptive(
    candidate_scores: torch.Tensor,
    candidate_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    topk: int,
    stride_sl_b: int,
    stride_sl_h: int,
    short_max_blocks: int,
    short_chunks: int,
) -> torch.Tensor:
    num_score_chunks, num_q_heads, batch_size, _ = candidate_scores.shape
    assert 0 < short_chunks <= num_score_chunks
    topk_indices = torch.empty(
        (num_q_heads, batch_size, topk),
        dtype=torch.int32,
        device=candidate_scores.device,
    )
    common_args = (
        candidate_scores,
        candidate_indices,
        topk_indices,
        seq_lens,
        stride_sl_b,
        stride_sl_h,
        block_size,
        short_max_blocks,
        candidate_scores.stride(0),
        candidate_scores.stride(1),
        candidate_scores.stride(2),
        candidate_scores.stride(3),
        candidate_indices.stride(0),
        candidate_indices.stride(1),
        candidate_indices.stride(2),
        candidate_indices.stride(3),
        topk_indices.stride(0),
        topk_indices.stride(1),
        topk_indices.stride(2),
    )
    grid = (batch_size, num_q_heads)
    _merge_bnsd_score_topk_candidates_guarded_kernel[grid](
        *common_args,
        NUM_SCORE_CHUNKS=short_chunks,
        topk=topk,
        BLOCK_SIZE_CANDIDATES=_next_power_of_2(short_chunks * topk),
        RUN_SHORT=True,
        num_warps=1,
        num_stages=1,
    )
    _merge_bnsd_score_topk_candidates_guarded_kernel[grid](
        *common_args,
        NUM_SCORE_CHUNKS=num_score_chunks,
        topk=topk,
        BLOCK_SIZE_CANDIDATES=_next_power_of_2(num_score_chunks * topk),
        RUN_SHORT=False,
        num_warps=1,
        num_stages=1,
    )
    return topk_indices


# =============================================================================
# Fused merge + causal-local-block append (decode/verify topk postprocess)
# =============================================================================


@triton.jit
def _merge_topk_append_local_impl(
    candidate_scores_ptr,  # [C, QH, B, topk]
    candidate_indices_ptr,  # [C, QH, B, topk]
    out_ptr,  # [QH, B, topk + 1]
    seq_lens_ptr,  # per-query causal KV len (stride_sl_* indexing)
    pid_b,
    pid_h,
    num_blocks,
    block_size: tl.constexpr,
    # strides
    stride_sl_b,
    stride_sl_h,
    stride_cs_c,
    stride_cs_h,
    stride_cs_b,
    stride_cs_t,
    stride_ci_c,
    stride_ci_h,
    stride_ci_b,
    stride_ci_t,
    stride_out_h,
    stride_out_b,
    stride_out_t,
    # meta
    NUM_SCORE_CHUNKS: tl.constexpr,
    topk: tl.constexpr,
    BLOCK_SIZE_CANDIDATES: tl.constexpr,
):
    """Global-TopK merge + causal-local-block append in one program (bit-exact).
    Per-rank argmax over [C*topk] candidates; local block at slot ``topk``
    (-1 when already selected -- dedup)."""
    seq_len = tl.load(seq_lens_ptr + pid_b * stride_sl_b + pid_h * stride_sl_h).to(
        tl.int32
    )
    query_pos = tl.maximum(seq_len - 1, 0)
    local_blk = tl.minimum(query_pos // block_size, num_blocks - 1)

    off_candidates = tl.arange(0, BLOCK_SIZE_CANDIDATES)
    candidate_chunk = off_candidates // topk
    candidate_rank = off_candidates - candidate_chunk * topk
    valid_candidate = off_candidates < NUM_SCORE_CHUNKS * topk
    candidate_scores = tl.load(
        candidate_scores_ptr
        + candidate_chunk * stride_cs_c
        + pid_h * stride_cs_h
        + pid_b * stride_cs_b
        + candidate_rank * stride_cs_t,
        mask=valid_candidate,
        other=float("-inf"),
    ).to(tl.float32)
    candidate_indices = tl.load(
        candidate_indices_ptr
        + candidate_chunk * stride_ci_c
        + pid_h * stride_ci_h
        + pid_b * stride_ci_b
        + candidate_rank * stride_ci_t,
        mask=valid_candidate,
        other=-1,
    ).to(tl.int32)
    candidate_scores = tl.where(
        candidate_indices >= 0,
        candidate_scores,
        float("-inf"),
    )

    out_base = out_ptr + pid_h * stride_out_h + pid_b * stride_out_b
    local_hit = 0
    for rank in tl.static_range(0, topk):
        best_score = tl.max(candidate_scores, axis=0)
        best_positions = tl.where(
            candidate_scores == best_score,
            off_candidates,
            tl.full((BLOCK_SIZE_CANDIDATES,), BLOCK_SIZE_CANDIDATES, tl.int32),
        )
        best_position = tl.min(best_positions, axis=0)
        selected_index = tl.max(
            tl.where(
                off_candidates == best_position,
                candidate_indices,
                tl.full((BLOCK_SIZE_CANDIDATES,), -1, tl.int32),
            ),
            axis=0,
        )
        valid = (
            (selected_index >= 0)
            & (selected_index < num_blocks)
            & (selected_index * block_size <= query_pos)
        )
        tl.store(
            out_base + rank * stride_out_t,
            tl.where(valid, selected_index, -1),
        )
        local_hit += (valid & (selected_index == local_blk)).to(tl.int32)
        candidate_scores = tl.where(
            off_candidates == best_position,
            float("-inf"),
            candidate_scores,
        )
    tl.store(
        out_base + topk * stride_out_t,
        tl.where(local_hit > 0, -1, local_blk),
    )


@triton.jit
def _merge_topk_append_local_kernel(
    candidate_scores_ptr,  # [C, QH, B, topk]
    candidate_indices_ptr,  # [C, QH, B, topk]
    out_ptr,  # [QH, B, topk + 1]
    seq_lens_ptr,
    num_blocks,
    # strides / meta
    block_size: tl.constexpr,
    short_max_blocks: tl.constexpr,
    stride_sl_b,
    stride_sl_h,
    stride_cs_c,
    stride_cs_h,
    stride_cs_b,
    stride_cs_t,
    stride_ci_c,
    stride_ci_h,
    stride_ci_b,
    stride_ci_t,
    stride_out_h,
    stride_out_b,
    stride_out_t,
    NUM_SCORE_CHUNKS: tl.constexpr,
    topk: tl.constexpr,
    BLOCK_SIZE_CANDIDATES: tl.constexpr,
    GUARDED: tl.constexpr,
    RUN_SHORT: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    if GUARDED:
        seq_len = tl.load(seq_lens_ptr + pid_b * stride_sl_b + pid_h * stride_sl_h)
        nb = tl.cdiv(seq_len, block_size)
        if RUN_SHORT:
            if nb > short_max_blocks:
                return
        else:
            if nb <= short_max_blocks:
                return
    _merge_topk_append_local_impl(
        candidate_scores_ptr,
        candidate_indices_ptr,
        out_ptr,
        seq_lens_ptr,
        pid_b,
        pid_h,
        num_blocks,
        block_size,
        stride_sl_b,
        stride_sl_h,
        stride_cs_c,
        stride_cs_h,
        stride_cs_b,
        stride_cs_t,
        stride_ci_c,
        stride_ci_h,
        stride_ci_b,
        stride_ci_t,
        stride_out_h,
        stride_out_b,
        stride_out_t,
        NUM_SCORE_CHUNKS=NUM_SCORE_CHUNKS,
        topk=topk,
        BLOCK_SIZE_CANDIDATES=BLOCK_SIZE_CANDIDATES,
    )


def _merge_topk_append_local(
    candidate_scores: torch.Tensor,
    candidate_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    topk: int,
    stride_sl_b: int,
    stride_sl_h: int,
    num_blocks: int,
) -> torch.Tensor:
    """Fused merge+append (single launch). Returns [QH, B, topk+1] int32."""
    num_score_chunks, num_q_heads, batch_size, _ = candidate_scores.shape
    out = torch.empty(
        (num_q_heads, batch_size, topk + 1),
        dtype=torch.int32,
        device=candidate_scores.device,
    )
    _merge_topk_append_local_kernel[(batch_size, num_q_heads)](
        candidate_scores,
        candidate_indices,
        out,
        seq_lens,
        num_blocks,
        block_size=block_size,
        short_max_blocks=0,
        stride_sl_b=stride_sl_b,
        stride_sl_h=stride_sl_h,
        stride_cs_c=candidate_scores.stride(0),
        stride_cs_h=candidate_scores.stride(1),
        stride_cs_b=candidate_scores.stride(2),
        stride_cs_t=candidate_scores.stride(3),
        stride_ci_c=candidate_indices.stride(0),
        stride_ci_h=candidate_indices.stride(1),
        stride_ci_b=candidate_indices.stride(2),
        stride_ci_t=candidate_indices.stride(3),
        stride_out_h=out.stride(0),
        stride_out_b=out.stride(1),
        stride_out_t=out.stride(2),
        NUM_SCORE_CHUNKS=num_score_chunks,
        topk=topk,
        BLOCK_SIZE_CANDIDATES=_next_power_of_2(num_score_chunks * topk),
        GUARDED=False,
        RUN_SHORT=False,
        num_warps=1,
        num_stages=1,
    )
    return out


def _merge_topk_append_local_adaptive(
    candidate_scores: torch.Tensor,
    candidate_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    topk: int,
    stride_sl_b: int,
    stride_sl_h: int,
    num_blocks: int,
    short_max_blocks: int,
    short_chunks: int,
) -> torch.Tensor:
    """Adaptive (short/long guarded) fused merge+append. [QH, B, topk+1]."""
    num_score_chunks, num_q_heads, batch_size, _ = candidate_scores.shape
    assert 0 < short_chunks <= num_score_chunks
    out = torch.empty(
        (num_q_heads, batch_size, topk + 1),
        dtype=torch.int32,
        device=candidate_scores.device,
    )
    common_args = (
        candidate_scores,
        candidate_indices,
        out,
        seq_lens,
        num_blocks,
    )
    grid = (batch_size, num_q_heads)
    _merge_topk_append_local_kernel[grid](
        *common_args,
        block_size=block_size,
        short_max_blocks=short_max_blocks,
        stride_sl_b=stride_sl_b,
        stride_sl_h=stride_sl_h,
        stride_cs_c=candidate_scores.stride(0),
        stride_cs_h=candidate_scores.stride(1),
        stride_cs_b=candidate_scores.stride(2),
        stride_cs_t=candidate_scores.stride(3),
        stride_ci_c=candidate_indices.stride(0),
        stride_ci_h=candidate_indices.stride(1),
        stride_ci_b=candidate_indices.stride(2),
        stride_ci_t=candidate_indices.stride(3),
        stride_out_h=out.stride(0),
        stride_out_b=out.stride(1),
        stride_out_t=out.stride(2),
        NUM_SCORE_CHUNKS=short_chunks,
        topk=topk,
        BLOCK_SIZE_CANDIDATES=_next_power_of_2(short_chunks * topk),
        GUARDED=True,
        RUN_SHORT=True,
        num_warps=1,
        num_stages=1,
    )
    _merge_topk_append_local_kernel[grid](
        *common_args,
        block_size=block_size,
        short_max_blocks=short_max_blocks,
        stride_sl_b=stride_sl_b,
        stride_sl_h=stride_sl_h,
        stride_cs_c=candidate_scores.stride(0),
        stride_cs_h=candidate_scores.stride(1),
        stride_cs_b=candidate_scores.stride(2),
        stride_cs_t=candidate_scores.stride(3),
        stride_ci_c=candidate_indices.stride(0),
        stride_ci_h=candidate_indices.stride(1),
        stride_ci_b=candidate_indices.stride(2),
        stride_ci_t=candidate_indices.stride(3),
        stride_out_h=out.stride(0),
        stride_out_b=out.stride(1),
        stride_out_t=out.stride(2),
        NUM_SCORE_CHUNKS=num_score_chunks,
        topk=topk,
        BLOCK_SIZE_CANDIDATES=_next_power_of_2(num_score_chunks * topk),
        GUARDED=True,
        RUN_SHORT=False,
        num_warps=1,
        num_stages=1,
    )
    return out


# =============================================================================


@triton.heuristics(
    {
        "BLOCK_SIZE_H": lambda args: max(
            16, triton.next_power_of_2(args["gqa_group_size"])
        ),
        "BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["head_dim"]),
    }
)
@triton.heuristics(
    {
        "BLOCK_SIZE_H": lambda args: max(
            16, triton.next_power_of_2(args["gqa_group_size"])
        ),
        "BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["head_dim"]),
        "BLOCK_SIZE_T": lambda args: triton.next_power_of_2(args["topk"]),
    }
)
@triton.jit
def _decode_bnsd_score_topk_chunk_kernel(
    q_ptr,  # [B, QH, D]
    k_cache_ptr,  # [NBLOCKS, BLOCK, KVH, D]
    block_table_ptr,  # [B, max_num_blocks] or typed direct-map placeholder
    req_to_token_ptr,  # [num_requests, max_context] in direct-map mode
    req_pool_indices_ptr,  # [B] in direct-map mode
    candidate_scores_ptr,  # [C, QH, B, topk]
    candidate_indices_ptr,  # [C, QH, B, topk]
    seq_lens,  # [B] per-request, or [B*gqa] per-row (packed draft queries)
    stride_sl_b,  # row-0 seq_lens stride per batch (1, or gqa when packed)
    stride_sl_h,  # per-row seq_lens stride (0 = shared per-request, 1 = packed)
    # shape
    batch_size,
    gqa_group_size,
    head_dim,
    # block/scaling
    block_size: tl.constexpr,
    sm_scale,
    init_blocks: tl.constexpr,
    local_blocks: tl.constexpr,
    num_score_chunks,
    # strides
    stride_q_b,
    stride_q_h,
    stride_q_d,
    stride_k_block,
    stride_k_offset,
    stride_k_h,
    stride_k_d,
    stride_bt_b,
    stride_bt_n,
    stride_rtt_r,
    stride_rtt_t,
    max_req_to_token_cols,
    num_pages,
    stride_cs_c,
    stride_cs_h,
    stride_cs_b,
    stride_cs_t,
    stride_ci_c,
    stride_ci_h,
    stride_ci_b,
    stride_ci_t,
    # meta
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    SCORE_TYPE: tl.constexpr,
    topk: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    USE_DIRECT_PAGE_LOOKUP: tl.constexpr,
    SANITIZE_PAGE_IDS: tl.constexpr,
    FILL_ONLY: tl.constexpr,
    RUNTIME_FILL_ONLY: tl.constexpr,
    RUNTIME_ADAPTIVE_SCORE_CHUNKS: tl.constexpr,
    RUNTIME_SCORE_SHORT_MAX_BLOCKS: tl.constexpr,
    RUNTIME_SCORE_SHORT_CHUNKS: tl.constexpr,
):
    """Fuse block score computation with one register-resident TopK per chunk."""
    tl.static_assert(SCORE_TYPE == "max" or SCORE_TYPE == "lse")
    tl.static_assert(BLOCK_SIZE_N >= block_size)

    pid_bc = tl.program_id(0)
    pid_kh = tl.program_id(1)
    pid_b = pid_bc % batch_size
    pid_c = pid_bc // batch_size
    pid_h = pid_kh * gqa_group_size

    off_h = tl.arange(0, BLOCK_SIZE_H)
    # Per-row seq_lens: stride_sl_h==0 broadcasts one shared length, ==1 packs
    # one causal length per gqa row; loop bounds use the row max.
    seq_len_rows = tl.load(
        seq_lens + pid_b * stride_sl_b + off_h * stride_sl_h,
        mask=off_h < gqa_group_size,
        other=0,
    ).to(tl.int32)
    seq_len_max = tl.max(seq_len_rows, axis=0)
    num_blocks_rows = tl.cdiv(seq_len_rows, block_size)
    num_blocks = tl.max(num_blocks_rows, axis=0)

    off_t = tl.arange(0, BLOCK_SIZE_T)
    candidate_mask = (off_h[:, None] < gqa_group_size) & (off_t[None, :] < topk)
    candidate_offsets = (
        pid_c * stride_cs_c
        + (pid_h + off_h[:, None]) * stride_cs_h
        + pid_b * stride_cs_b
        + off_t[None, :] * stride_cs_t
    )
    candidate_index_offsets = (
        pid_c * stride_ci_c
        + (pid_h + off_h[:, None]) * stride_ci_h
        + pid_b * stride_ci_b
        + off_t[None, :] * stride_ci_t
    )

    top_scores = tl.full((BLOCK_SIZE_H, BLOCK_SIZE_T), float("-inf"), tl.float32)
    top_indices = tl.full((BLOCK_SIZE_H, BLOCK_SIZE_T), -1, tl.int32)
    if num_blocks <= topk:
        top_indices = tl.where(
            (pid_c == 0) & (off_t[None, :] < num_blocks_rows[:, None]),
            off_t[None, :] + tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_T), tl.int32),
            top_indices,
        )
        top_scores = tl.where(
            top_indices >= 0,
            tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_T), tl.float32),
            top_scores,
        )
        tl.store(
            candidate_scores_ptr + candidate_offsets, top_scores, mask=candidate_mask
        )
        tl.store(
            candidate_indices_ptr + candidate_index_offsets,
            top_indices,
            mask=candidate_mask,
        )
        return

    active_num_score_chunks = num_score_chunks
    if RUNTIME_ADAPTIVE_SCORE_CHUNKS:
        active_num_score_chunks = tl.where(
            num_blocks <= RUNTIME_SCORE_SHORT_MAX_BLOCKS,
            RUNTIME_SCORE_SHORT_CHUNKS,
            num_score_chunks,
        )
    chunk_size_blocks = tl.maximum(1, tl.cdiv(num_blocks, active_num_score_chunks))
    chunk_start_block = pid_c * chunk_size_blocks
    chunk_end_block = tl.minimum(chunk_start_block + chunk_size_blocks, num_blocks)
    if chunk_start_block >= chunk_end_block:
        tl.store(
            candidate_scores_ptr + candidate_offsets, top_scores, mask=candidate_mask
        )
        tl.store(
            candidate_indices_ptr + candidate_index_offsets,
            top_indices,
            mask=candidate_mask,
        )
        return

    off_d = tl.arange(0, BLOCK_SIZE_D)
    off_n = tl.arange(0, BLOCK_SIZE_N)
    q_offsets = (
        pid_b * stride_q_b
        + (pid_h + off_h[:, None]) * stride_q_h
        + off_d[None, :] * stride_q_d
    )
    q = tl.load(
        q_ptr + q_offsets,
        mask=(off_h[:, None] < gqa_group_size) & (off_d[None, :] < head_dim),
        other=0.0,
    )

    sm_scale_log2e = sm_scale * 1.4426950409
    local_start_rows = tl.maximum(0, num_blocks_rows - local_blocks)
    num_steps = chunk_end_block - chunk_start_block
    # req_idx is loop-invariant (pid_b constant) -- hoist the scalar load out of
    # the per-step loop instead of reloading it each iter.
    if USE_DIRECT_PAGE_LOOKUP:
        req_idx = tl.load(req_pool_indices_ptr + pid_b).to(tl.int64)
    for step in tl.range(num_steps):
        logical_block = chunk_start_block + step
        if USE_DIRECT_PAGE_LOOKUP:
            token_col = tl.minimum(
                logical_block * block_size, max_req_to_token_cols - 1
            )
            token_slot = tl.load(
                req_to_token_ptr + req_idx * stride_rtt_r + token_col * stride_rtt_t
            ).to(tl.int64)
            physical_block = token_slot // block_size
            if SANITIZE_PAGE_IDS:
                physical_block = tl.minimum(
                    tl.maximum(physical_block, 0), num_pages - 1
                )
        else:
            physical_block = tl.load(
                block_table_ptr + pid_b * stride_bt_b + logical_block * stride_bt_n
            ).to(tl.int64)
        pos = logical_block * block_size + off_n
        # K is shared by all rows: load it with the row-max length, then apply
        # each row's own causal mask to qk so shorter packed rows keep -inf.
        pos_mask_k = pos < seq_len_max
        pos_mask = pos[None, :] < seq_len_rows[:, None]
        k_offsets = (
            physical_block * stride_k_block
            + off_n[None, :] * stride_k_offset
            + pid_kh * stride_k_h
            + off_d[:, None] * stride_k_d
        )
        k = tl.load(
            k_cache_ptr + k_offsets,
            mask=(off_d[:, None] < head_dim) & pos_mask_k[None, :],
            other=0.0,
        )
        qk = tl.dot(q, k) * sm_scale_log2e
        qk = tl.where(pos_mask, qk, float("-inf"))
        sub_max = tl.max(qk, axis=1)
        if SCORE_TYPE == "max":
            score = sub_max
        else:
            score = sub_max + tl.log2(tl.sum(tl.exp2(qk - sub_max[:, None]), axis=1))
            score = tl.where(score != score, float("-inf"), score)
        # init/local guards are constexpr-folded away (0/0 in production); for
        # nonzero values keep the row-vector guards (exact for shorter packed rows).
        if init_blocks > 0:
            is_init = (logical_block < init_blocks) & (logical_block < num_blocks_rows)
            score = tl.where(is_init, 1e30, score)
        if local_blocks > 0:
            is_local = (logical_block >= local_start_rows) & (
                logical_block < num_blocks_rows
            )
            score = tl.where(is_local, 1e29, score)

        if FILL_ONLY:
            # Store score/index directly to candidate output at slot=step (bypasses
            # loop-carried registers); unused slots keep -inf/-1 pre-init.
            head_mask = off_h < gqa_group_size
            cs_off = (
                pid_c * stride_cs_c
                + (pid_h + off_h) * stride_cs_h
                + pid_b * stride_cs_b
                + step * stride_cs_t
            )
            ci_off = (
                pid_c * stride_ci_c
                + (pid_h + off_h) * stride_ci_h
                + pid_b * stride_ci_b
                + step * stride_ci_t
            )
            tl.store(candidate_scores_ptr + cs_off, score, mask=head_mask)
            tl.store(candidate_indices_ptr + ci_off, logical_block, mask=head_mask)
        else:
            if RUNTIME_FILL_ONLY and chunk_size_blocks <= topk:
                # Graph capture uses max_context_len but replay can be shorter:
                # when chunk_size_blocks <= topk, bypass min/replacement reductions.
                head_mask = off_h < gqa_group_size
                cs_off = (
                    pid_c * stride_cs_c
                    + (pid_h + off_h) * stride_cs_h
                    + pid_b * stride_cs_b
                    + step * stride_cs_t
                )
                ci_off = (
                    pid_c * stride_ci_c
                    + (pid_h + off_h) * stride_ci_h
                    + pid_b * stride_ci_b
                    + step * stride_ci_t
                )
                tl.store(candidate_scores_ptr + cs_off, score, mask=head_mask)
                tl.store(candidate_indices_ptr + ci_off, logical_block, mask=head_mask)
            else:
                valid_topk_lane = off_t[None, :] < topk
                current_min = tl.min(top_scores, axis=1)
                min_positions = tl.where(
                    (top_scores == current_min[:, None]) & valid_topk_lane,
                    off_t[None, :],
                    tl.full((BLOCK_SIZE_H, BLOCK_SIZE_T), BLOCK_SIZE_T, tl.int32),
                )
                min_position = tl.min(min_positions, axis=1)
                replace = (
                    (off_t[None, :] == min_position[:, None])
                    & valid_topk_lane
                    & (score[:, None] > current_min[:, None])
                )
                top_scores = tl.where(replace, score[:, None], top_scores)
                top_indices = tl.where(replace, logical_block, top_indices)

    # FILL_ONLY wrote each block straight to the candidate output inside the loop
    # (store-per-block); the register tiles are stale, so skip this end-store for it.
    if not FILL_ONLY:
        if not RUNTIME_FILL_ONLY or chunk_size_blocks > topk:
            tl.store(
                candidate_scores_ptr + candidate_offsets,
                top_scores,
                mask=candidate_mask,
            )
            tl.store(
                candidate_indices_ptr + candidate_index_offsets,
                top_indices,
                mask=candidate_mask,
            )


# =============================================================================
# BNSD Decode Fused Score + Attention Chunk Kernel
# =============================================================================


@triton.heuristics(
    {
        "BLOCK_SIZE_H": lambda args: max(
            16, triton.next_power_of_2(args["gqa_group_size"])
        ),
        "BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["head_dim"]),
        "BLOCK_SIZE_T": lambda args: triton.next_power_of_2(args["topk"]),
        "HAS_SINK": lambda args: args["sink_ptr"] is not None,
    }
)
@triton.jit
def _decode_bnsd_score_attn_chunk_kernel(
    q_ptr,  # [B, QH, D]
    sink_ptr,  # optional [QH, D]
    k_cache_ptr,  # [NBLOCKS, BLOCK, KVH, D]
    v_cache_ptr,  # [NBLOCKS, BLOCK, KVH, D]
    block_table_ptr,  # [B, max_num_blocks]
    o_ptr,  # [C, B, QH, D]
    lse_ptr,  # [C, B, QH]
    candidate_scores_ptr,  # [C, QH, B, topk]
    candidate_indices_ptr,  # [C, QH, B, topk]
    seq_lens,  # [B]
    # shape
    batch_size,
    gqa_group_size,
    head_dim,
    # block/scaling
    block_size: tl.constexpr,
    sm_scale,
    init_blocks,
    local_blocks,
    # strides
    stride_q_b,
    stride_q_h,
    stride_q_d,
    stride_sink_h,
    stride_sink_d,
    stride_k_block,
    stride_k_offset,
    stride_k_h,
    stride_k_d,
    stride_v_block,
    stride_v_offset,
    stride_v_h,
    stride_v_d,
    stride_bt_b,
    stride_bt_n,
    stride_o_c,
    stride_o_b,
    stride_o_h,
    stride_o_d,
    stride_l_c,
    stride_l_b,
    stride_l_h,
    stride_cs_c,
    stride_cs_h,
    stride_cs_b,
    stride_cs_t,
    stride_ci_c,
    stride_ci_h,
    stride_ci_b,
    stride_ci_t,
    # meta
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    NUM_KV_CHUNKS: tl.constexpr,
    HAS_SINK: tl.constexpr,
    SCORE_TYPE: tl.constexpr,
    topk: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
):
    """Fused full-decode: block scores + attention output in one KV pass.
    Ascend-friendly: no make_block_ptr, per-chunk register TopK."""
    tl.static_assert(SCORE_TYPE == "max" or SCORE_TYPE == "lse")
    tl.static_assert(BLOCK_SIZE_N >= block_size)

    pid_bc = tl.program_id(0)
    pid_kh = tl.program_id(1)

    pid_b = pid_bc % batch_size
    pid_c = pid_bc // batch_size
    pid_h = pid_kh * gqa_group_size

    seq_len = tl.load(seq_lens + pid_b).to(tl.int32)
    num_blocks = tl.cdiv(seq_len, block_size)

    off_h = tl.arange(0, BLOCK_SIZE_H)
    off_t = tl.arange(0, BLOCK_SIZE_T)
    candidate_mask = (off_h[:, None] < gqa_group_size) & (off_t[None, :] < topk)
    candidate_offsets = (
        pid_c * stride_cs_c
        + (pid_h + off_h[:, None]) * stride_cs_h
        + pid_b * stride_cs_b
        + off_t[None, :] * stride_cs_t
    )
    candidate_index_offsets = (
        pid_c * stride_ci_c
        + (pid_h + off_h[:, None]) * stride_ci_h
        + pid_b * stride_ci_b
        + off_t[None, :] * stride_ci_t
    )
    top_scores = tl.full((BLOCK_SIZE_H, BLOCK_SIZE_T), float("-inf"), tl.float32)
    top_indices = tl.full((BLOCK_SIZE_H, BLOCK_SIZE_T), -1, tl.int32)

    chunk_size_blocks = tl.maximum(1, tl.cdiv(num_blocks, NUM_KV_CHUNKS))
    chunk_start_block = pid_c * chunk_size_blocks
    chunk_end_block = tl.minimum(chunk_start_block + chunk_size_blocks, num_blocks)

    if chunk_start_block >= chunk_end_block:
        tl.store(
            candidate_scores_ptr + candidate_offsets, top_scores, mask=candidate_mask
        )
        tl.store(
            candidate_indices_ptr + candidate_index_offsets,
            top_indices,
            mask=candidate_mask,
        )
        return

    off_d = tl.arange(0, BLOCK_SIZE_D)
    off_n = tl.arange(0, BLOCK_SIZE_N)

    q_offsets = (
        pid_b * stride_q_b
        + (pid_h + off_h[:, None]) * stride_q_h
        + off_d[None, :] * stride_q_d
    )
    q = tl.load(
        q_ptr + q_offsets,
        mask=(off_h[:, None] < gqa_group_size) & (off_d[None, :] < head_dim),
        other=0.0,
    )

    sm_scale_log2e = sm_scale * 1.4426950409

    if HAS_SINK:
        if pid_c == 0:
            sink_offsets = (pid_h + off_h[:, None]) * stride_sink_h + off_d[
                None, :
            ] * stride_sink_d
            sink = tl.load(
                sink_ptr + sink_offsets,
                mask=(off_h[:, None] < gqa_group_size) & (off_d[None, :] < head_dim),
                other=0.0,
            ).to(tl.float32)
            qsink = tl.sum(q.to(tl.float32) * sink, axis=1) * sm_scale_log2e
            m_i = qsink
            l_i = tl.full((BLOCK_SIZE_H,), 1.0, dtype=tl.float32)
        else:
            m_i = tl.full((BLOCK_SIZE_H,), float("-inf"), dtype=tl.float32)
            l_i = tl.full((BLOCK_SIZE_H,), 0.0, dtype=tl.float32)
    else:
        m_i = tl.full((BLOCK_SIZE_H,), float("-inf"), dtype=tl.float32)
        l_i = tl.full((BLOCK_SIZE_H,), 0.0, dtype=tl.float32)

    acc_o = tl.full((BLOCK_SIZE_H, BLOCK_SIZE_D), 0.0, dtype=tl.float32)
    local_start = tl.maximum(0, num_blocks - local_blocks)

    num_steps = chunk_end_block - chunk_start_block
    for step in tl.range(num_steps):
        logical_block = chunk_start_block + step
        physical_block = tl.load(
            block_table_ptr + pid_b * stride_bt_b + logical_block * stride_bt_n
        ).to(tl.int64)

        pos = logical_block * block_size + off_n
        pos_mask = pos < seq_len

        k_offsets = (
            physical_block * stride_k_block
            + off_n[None, :] * stride_k_offset
            + pid_kh * stride_k_h
            + off_d[:, None] * stride_k_d
        )
        k = tl.load(
            k_cache_ptr + k_offsets,
            mask=(off_d[:, None] < head_dim) & pos_mask[None, :],
            other=0.0,
        )

        v_offsets = (
            physical_block * stride_v_block
            + off_n[:, None] * stride_v_offset
            + pid_kh * stride_v_h
            + off_d[None, :] * stride_v_d
        )
        v = tl.load(
            v_cache_ptr + v_offsets,
            mask=pos_mask[:, None] & (off_d[None, :] < head_dim),
            other=0.0,
        )

        qk = tl.dot(q, k) * sm_scale_log2e
        qk = tl.where(pos_mask[None, :], qk, float("-inf"))

        # ---- block score path ----
        sub_max = tl.max(qk, axis=1)
        if SCORE_TYPE == "max":
            score = sub_max
        else:
            score = sub_max + tl.log2(tl.sum(tl.exp2(qk - sub_max[:, None]), axis=1))
            score = tl.where(score != score, float("-inf"), score)

        is_init = logical_block < init_blocks
        is_local = (logical_block >= local_start) & (logical_block < num_blocks)
        score = tl.where(is_init, 1e30, score)
        score = tl.where(is_local, 1e29, score)

        valid_topk_lane = off_t[None, :] < topk
        current_min = tl.min(top_scores, axis=1)
        min_positions = tl.where(
            (top_scores == current_min[:, None]) & valid_topk_lane,
            off_t[None, :],
            tl.full((BLOCK_SIZE_H, BLOCK_SIZE_T), BLOCK_SIZE_T, tl.int32),
        )
        min_position = tl.min(min_positions, axis=1)
        replace = (
            (off_t[None, :] == min_position[:, None])
            & valid_topk_lane
            & (score[:, None] > current_min[:, None])
        )
        top_scores = tl.where(replace, score[:, None], top_scores)
        top_indices = tl.where(replace, logical_block, top_indices)

        # ---- attention path ----
        m_new = tl.maximum(m_i, sub_max)
        p = tl.exp2(qk - m_new[:, None])
        l_new = tl.sum(p, axis=1)

        acc_scale = tl.exp2(m_i - m_new)
        acc_o = acc_o * acc_scale[:, None]
        acc_o += tl.dot(p.to(v.dtype), v)

        l_i = l_i * acc_scale + l_new
        m_i = m_new

    acc_o = acc_o / l_i[:, None]
    lse_i = m_i + tl.log2(l_i)

    o_offsets = (
        pid_c * stride_o_c
        + pid_b * stride_o_b
        + (pid_h + off_h[:, None]) * stride_o_h
        + off_d[None, :] * stride_o_d
    )
    tl.store(
        o_ptr + o_offsets,
        acc_o.to(o_ptr.dtype.element_ty),
        mask=(off_h[:, None] < gqa_group_size) & (off_d[None, :] < head_dim),
    )

    l_offsets = pid_c * stride_l_c + pid_b * stride_l_b + (pid_h + off_h) * stride_l_h
    tl.store(
        lse_ptr + l_offsets,
        lse_i.to(lse_ptr.dtype.element_ty),
        mask=off_h < gqa_group_size,
    )
    tl.store(candidate_scores_ptr + candidate_offsets, top_scores, mask=candidate_mask)
    tl.store(
        candidate_indices_ptr + candidate_index_offsets,
        top_indices,
        mask=candidate_mask,
    )


# =============================================================================
# BNSD Decode Attention Merge Kernel
# =============================================================================


@triton.heuristics(
    {
        "BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["head_dim"]),
    }
)
@triton.jit
def _merge_bnsd_attn_out_kernel(
    o_ptr,  # [C, B, QH, D]
    lse_ptr,  # [C, B, QH]
    seq_lens,  # [B]
    out_ptr,  # [B, QH, D]
    # shape
    head_dim,
    block_size: tl.constexpr,
    # strides
    stride_o_c,
    stride_o_b,
    stride_o_h,
    stride_o_d,
    stride_l_c,
    stride_l_b,
    stride_l_h,
    stride_out_b,
    stride_out_h,
    stride_out_d,
    # meta
    NUM_KV_CHUNKS: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    off_d = tl.arange(0, BLOCK_SIZE_D)

    seq_len = tl.load(seq_lens + pid_b).to(tl.int32)
    num_blocks = tl.cdiv(seq_len, block_size)

    chunk_size_blocks = tl.maximum(1, tl.cdiv(num_blocks, NUM_KV_CHUNKS))
    valid_chunks = tl.cdiv(num_blocks, chunk_size_blocks)

    m = tl.full((), float("-inf"), dtype=tl.float32)
    l = tl.full((), 0.0, dtype=tl.float32)
    acc = tl.full((BLOCK_SIZE_D,), 0.0, dtype=tl.float32)

    for c in tl.static_range(0, NUM_KV_CHUNKS):
        valid = c < valid_chunks

        lse_c = tl.load(
            lse_ptr + c * stride_l_c + pid_b * stride_l_b + pid_h * stride_l_h,
            mask=valid,
            other=float("-inf"),
        )

        o_c = tl.load(
            o_ptr
            + c * stride_o_c
            + pid_b * stride_o_b
            + pid_h * stride_o_h
            + off_d * stride_o_d,
            mask=valid & (off_d < head_dim),
            other=0.0,
        ).to(tl.float32)

        m_new = tl.maximum(m, lse_c)
        scale_old = tl.exp2(m - m_new)
        scale_new = tl.exp2(lse_c - m_new)

        acc = acc * scale_old + o_c * scale_new
        l = l * scale_old + scale_new
        m = m_new

    out = acc / l

    tl.store(
        out_ptr + pid_b * stride_out_b + pid_h * stride_out_h + off_d * stride_out_d,
        out.to(out_ptr.dtype.element_ty),
        mask=off_d < head_dim,
    )


# =============================================================================
# Decode topk postprocess
# =============================================================================


def _normalize_topk_idx_for_gqa(
    topk_idx: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    gqa_group_size: int,
) -> torch.Tensor:
    """Ensure topk_idx has shape [num_kv_heads, batch_size, topk].
    Per-query-head input takes the first q-head of each GQA group."""
    if topk_idx.shape[0] == num_kv_heads:
        return topk_idx.contiguous()

    if topk_idx.shape[0] == num_q_heads:
        batch_size = topk_idx.shape[1]
        max_topk = topk_idx.shape[2]
        return topk_idx.view(num_kv_heads, gqa_group_size, batch_size, max_topk)[
            :, 0, :, :
        ].contiguous()

    raise AssertionError(
        "topk_idx first dimension must be either num_kv_heads or num_q_heads, "
        f"got {topk_idx.shape[0]}, num_kv_heads={num_kv_heads}, "
        f"num_q_heads={num_q_heads}"
    )


@triton.jit
def _native_sanitize_topk_kernel(
    sel_ptr,  # [num_kv_heads, batch, SLOTS] int32 (in-place OUT)
    select_num_idx_ptr,  # [num_kv_heads, batch] int32 (OUT)
    seq_lens_ptr,  # [batch] int32
    stride_sel_h,
    stride_sel_b,
    stride_sel_s,
    stride_sn_h,
    stride_sn_b,
    block_size: tl.constexpr,
    SLOTS: tl.constexpr,
):
    """Sanitize the native op's select_idx/select_num_idx in one launch.
    sel >= cdiv(seq_len, block_size) -> -1; select_num_idx = count of valid."""
    pid_h = tl.program_id(0)
    pid_b = tl.program_id(1)
    seq_len = tl.load(seq_lens_ptr + pid_b)
    nblocks = tl.cdiv(seq_len, block_size)
    off = tl.arange(0, SLOTS)
    base = pid_h * stride_sel_h + pid_b * stride_sel_b
    sel = tl.load(sel_ptr + base + off * stride_sel_s)  # [SLOTS]
    sel = tl.where(sel >= nblocks, -1, sel)  # sanitize OOB
    count = tl.sum((sel >= 0).to(tl.int32), axis=0)  # scalar
    tl.store(sel_ptr + base + off * stride_sel_s, sel)
    tl.store(select_num_idx_ptr + pid_h * stride_sn_h + pid_b * stride_sn_b, count)


@triton.heuristics(
    {
        "BLOCK_SIZE_T": lambda args: triton.next_power_of_2(args["topk"]),
    }
)
@triton.jit
def _append_local_block_to_topk_idx_kernel(
    topk_idx_ptr,  # [num_kv_heads, batch_size, topk]
    seq_lens_ptr,  # [batch_size]
    out_ptr,  # [num_kv_heads, batch_size, topk + 1]
    batch_size,
    topk,
    num_blocks,
    block_size: tl.constexpr,
    stride_topk_h,
    stride_topk_b,
    stride_topk_t,
    stride_out_h,
    stride_out_b,
    stride_out_t,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
):
    """Append the causal local block (query-block-tiled, BSQ queries/program).
    Validates candidates and appends the local block (deduped to -1)."""
    pid_qb = tl.program_id(0)
    pid_h = tl.program_id(1)
    off_q = tl.arange(0, BLOCK_SIZE_Q)  # [BSQ]
    off_t = tl.arange(0, BLOCK_SIZE_T)  # [topk_pow2]
    q_tok = pid_qb * BLOCK_SIZE_Q + off_q  # [BSQ]
    q_valid = q_tok < batch_size

    # Per-query causal position + local block.
    seq_len = tl.load(seq_lens_ptr + q_tok, mask=q_valid, other=1).to(tl.int32)
    query_pos = tl.maximum(seq_len - 1, 0)
    local_blk = tl.minimum(query_pos // block_size, num_blocks - 1)

    # Load candidates [BSQ, topk] and validate.
    in_off = (
        pid_h * stride_topk_h
        + q_tok[:, None] * stride_topk_b
        + off_t[None, :] * stride_topk_t
    )
    cand = tl.load(
        topk_idx_ptr + in_off,
        mask=q_valid[:, None] & (off_t[None, :] < topk),
        other=-1,
    ).to(tl.int32)
    valid = (
        (cand >= 0) & (cand < num_blocks) & (cand * block_size <= query_pos[:, None])
    )
    cand_out = tl.where(valid, cand, -1)

    # Store validated candidates [BSQ, topk].
    out_off = (
        pid_h * stride_out_h
        + q_tok[:, None] * stride_out_b
        + off_t[None, :] * stride_out_t
    )
    tl.store(
        out_ptr + out_off, cand_out, mask=q_valid[:, None] & (off_t[None, :] < topk)
    )

    # Append local block at slot topk: -1 if already present (dedup).
    local_present = tl.sum((cand_out == local_blk[:, None]).to(tl.int32), axis=1) > 0
    out_local = tl.where(local_present, -1, local_blk)
    tl.store(
        out_ptr + pid_h * stride_out_h + q_tok * stride_out_b + topk * stride_out_t,
        tl.where(q_valid, out_local, -1),
        mask=q_valid,
    )


@torch.no_grad()
def append_local_block_to_topk_idx(
    topk_idx: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    num_blocks: int,
) -> torch.Tensor:
    """Fuse MiniMax's ``init=0, local=1`` decode top-k postprocess (in-place).
    Preserves candidate order; drops a local block only when already a candidate."""
    assert topk_idx.ndim == 3
    assert topk_idx.dtype == torch.int32
    assert topk_idx.is_contiguous()
    assert seq_lens.ndim == 1
    assert seq_lens.shape[0] == topk_idx.shape[1]
    assert seq_lens.is_contiguous()
    assert block_size > 0
    assert num_blocks > 0

    num_kv_heads, batch_size, topk = topk_idx.shape
    out = torch.empty(
        (num_kv_heads, batch_size, topk + 1),
        dtype=topk_idx.dtype,
        device=topk_idx.device,
    )
    BSQ = 16
    grid = (triton.cdiv(batch_size, BSQ), num_kv_heads)
    _append_local_block_to_topk_idx_kernel[grid](
        topk_idx,
        seq_lens,
        out,
        batch_size,
        topk,
        num_blocks,
        block_size=block_size,
        stride_topk_h=topk_idx.stride(0),
        stride_topk_b=topk_idx.stride(1),
        stride_topk_t=topk_idx.stride(2),
        stride_out_h=out.stride(0),
        stride_out_b=out.stride(1),
        stride_out_t=out.stride(2),
        BLOCK_SIZE_Q=BSQ,
        num_warps=4,
        num_stages=1,
    )
    return out


# =============================================================================
# Python Wrapper
# =============================================================================


@torch.no_grad()
def flash_decode_bnsd_with_topk_idx(
    q: torch.Tensor,  # [batch_size, num_q_heads, head_dim]
    sink: Optional[torch.Tensor],  # optional [num_q_heads, head_dim]
    k_cache_bnsd: torch.Tensor,  # [num_blocks, block_size, num_kv_heads, head_dim]
    v_cache_bnsd: Optional[torch.Tensor],
    block_table: Optional[torch.Tensor],  # [batch_size, max_num_blocks]
    seq_lens: torch.Tensor,  # [batch_size]
    max_seqlen: int,
    block_size: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    sm_scale: Optional[float] = None,
    score_type: str = "max",
    disable_index_value: bool = False,
    num_kv_chunks: Optional[int] = None,
    max_num_kv_chunks: int = 8,
    # Retained for call-site compatibility. Both short and long contexts use
    # the same chunk-candidate + merge Triton route.
    use_triton_topk: Optional[bool] = None,
    # Kept for API compatibility.
    num_topk_chunks: Optional[int] = None,
    # Retained for call-site compatibility. Score-only always uses the fused
    # chunk-candidate kernel.
    use_chunked_score: bool = True,
    # Target block tiles per program for the chunked score kernel. Larger ->
    # fewer programs but longer serial loop; smaller -> more parallelism.
    score_blocks_per_chunk: int = 16,
    # Cap on the chunk count (power-of-two rounded). Raise for small packed
    # batches (few programs) to keep the vector cores busy at long context.
    score_max_chunks: int = 32,
    # Direct request-map page source. This is intentionally an alternative to a
    # materialized block table so graph replay cannot reuse a stale layer buffer.
    req_to_token: Optional[torch.Tensor] = None,
    req_pool_indices: Optional[torch.Tensor] = None,
    max_num_blocks: Optional[int] = None,
    num_pages: Optional[int] = None,
    sanitize_page_ids: bool = False,
    # Pack the gqa row dim with PER-ROW seq_lens (draft-token verify): one K pass
    # scores all packed rows (row-max length; rows differ by <=1 block).
    packed_seq_lens: bool = False,
    # Decode graphs compile against max_context_len but replay is often shorter:
    # pick the direct-fill candidate path from the runtime chunk length.
    runtime_fill_only: bool = False,
    # Keep a long-context graph's static grid while activating fewer, wider
    # chunks for short sequences; both values must be specified together.
    runtime_score_short_max_blocks: int = 0,
    runtime_score_short_chunks: int = 0,
    # Fuse candidate-merge + causal-local-block append into the merge launch
    # (bit-exact): [topk+1] = validated candidates + local block.
    fused_append_local: bool = False,
    # Native AscendC indexer gate (framework-controlled; op repo is unaware).
    use_native: bool = True,
) -> tuple[Optional[torch.Tensor], torch.Tensor]:
    """Decode attention with BNSD KV cache and block-level topk indices.
    Returns ``o`` and ``topk_idx`` ([QH, B, topk], int32)."""
    assert score_type in ("max", "lse")
    assert q.dtype in (torch.float16, torch.bfloat16)
    assert k_cache_bnsd.dtype == q.dtype
    assert seq_lens.dtype in (torch.int32, torch.int64)

    use_direct_page_lookup = req_to_token is not None
    assert (req_pool_indices is not None) == use_direct_page_lookup
    if use_direct_page_lookup:
        assert block_table is None
        assert disable_index_value
        assert req_to_token.ndim == 2
        assert req_to_token.dtype in (torch.int32, torch.int64)
        assert req_pool_indices.ndim == 1
        assert req_pool_indices.dtype in (torch.int32, torch.int64)
        assert max_num_blocks is not None and max_num_blocks > 0
        assert num_pages is not None and num_pages > 0
    else:
        assert block_table is not None
        assert block_table.dtype in (torch.int32, torch.int64)

    if not disable_index_value:
        assert v_cache_bnsd is not None
        assert v_cache_bnsd.dtype == q.dtype
        assert v_cache_bnsd.shape == k_cache_bnsd.shape

    batch_size, num_q_heads, head_dim = q.shape
    _, block_size_from_cache, num_kv_heads, cache_head_dim = k_cache_bnsd.shape

    assert block_size_from_cache == block_size
    assert cache_head_dim == head_dim
    assert num_q_heads % num_kv_heads == 0
    gqa_group_size = num_q_heads // num_kv_heads
    if packed_seq_lens:
        assert disable_index_value, "packed_seq_lens is score-only"
        assert seq_lens.shape[0] == batch_size * gqa_group_size
        stride_sl_b, stride_sl_h = gqa_group_size, 1
    else:
        assert seq_lens.shape[0] == batch_size
        stride_sl_b, stride_sl_h = 1, 0
    if use_direct_page_lookup:
        assert req_pool_indices.shape[0] == batch_size
        assert max_num_blocks * block_size <= req_to_token.shape[1]
        page_source = req_to_token
        page_source_rows = req_pool_indices
        direct_num_pages = int(num_pages)
    else:
        assert block_table.shape[0] == batch_size
        page_source = block_table
        # Triton requires a typed pointer even for constexpr-dead direct-mode
        # arguments; seq_lens is never read as an index here.
        page_source_rows = seq_lens
        direct_num_pages = 1

    if sm_scale is None:
        sm_scale = head_dim**-0.5

    # AscendC minimax_indexer fast path (Cube Q@K + multi-core + per-head LD merge),
    # replacing the triton score+topk for the score-only (disable_index_value) path.
    _native_minimax_indexer = _get_native_minimax_indexer()
    if (
        disable_index_value
        and topk > 0
        and _native_minimax_indexer is not None
        and use_native
        and num_q_heads % 2 == 0
        and head_dim == 128
        and seq_lens.shape[0]
        in (batch_size, batch_size * (num_q_heads // num_kv_heads))
    ):
        gqa = num_q_heads // num_kv_heads
        # Direct mode passes req_to_token + req_pool_indices; the kernel gathers
        # logical->physical block ids in-kernel. block_table mode unchanged.
        if use_direct_page_lookup:
            bt_in = None
            req_rt = req_to_token
            req_pi = req_pool_indices
        else:
            bt_in = (
                block_table
                if block_table.dtype == torch.int32
                else block_table.to(torch.int32)
            )
            req_rt = None
            req_pi = None
        q_in = q.reshape(batch_size, 1, num_q_heads, head_dim)
        w_dummy = torch.empty(
            (batch_size, 1, num_q_heads), dtype=q.dtype, device=q.device
        )
        aq_dummy = torch.ones(batch_size, dtype=torch.int32, device=q.device)
        # Normalized decode carries [B] lengths; packed verify carries the full
        # [B*gqa] per-row lengths (packed_mode=1, no slicing).
        if seq_lens.shape[0] == batch_size:
            sl_in = seq_lens.to(torch.int32)
            packed_mode = 0
        else:
            sl_in = seq_lens.to(torch.int32)
            packed_mode = 1
        # Fused causal-local append: emits [QH, B, topk+1] with the local block at
        # slot topk (deduped to -1); no permute/contiguous copy.
        append_local = 1 if fused_append_local else 0
        out = _native_minimax_indexer(
            q_in,
            k_cache_bnsd,
            w_dummy,
            aq_dummy,
            sl_in,
            bt_in,
            "BSND",
            "PA_BSND",
            topk,
            0,
            init_blocks,
            local_blocks,
            float(sm_scale),
            req_rt,
            req_pi,
            append_local,
            packed_mode,
        )
        topk_idx = out.view(num_q_heads, batch_size, topk + append_local)
        return None, topk_idx

    max_seqblock = (max_seqlen + block_size - 1) // block_size
    block_size_n = _next_power_of_2(block_size)

    if disable_index_value:
        if topk <= 0:
            return None, torch.empty(
                (num_q_heads, batch_size, 0), dtype=torch.int32, device=q.device
            )
        num_score_chunks = _choose_num_score_chunks(
            max_seqblock,
            blocks_per_chunk=score_blocks_per_chunk,
            max_chunks=score_max_chunks,
            all_seqblock_q=batch_size,
            num_kv_heads=num_kv_heads,
        )
        # Clamp runtime_score_short_chunks so it never exceeds num_score_chunks
        # (eager mode / short KV can have num_score_chunks < the hardcoded 16).
        if runtime_score_short_chunks:
            runtime_score_short_chunks = min(
                runtime_score_short_chunks, num_score_chunks
            )
        use_runtime_adaptive_score_chunks = bool(
            runtime_score_short_max_blocks or runtime_score_short_chunks
        )
        if use_runtime_adaptive_score_chunks:
            assert runtime_score_short_max_blocks > 0
            assert runtime_score_short_chunks > 0
            assert runtime_score_short_chunks <= num_score_chunks
            assert (runtime_score_short_chunks & (runtime_score_short_chunks - 1)) == 0
        candidate_scores = torch.full(
            (num_score_chunks, num_q_heads, batch_size, topk),
            float("-inf"),
            dtype=torch.float32,
            device=q.device,
        )
        candidate_indices = torch.full(
            (num_score_chunks, num_q_heads, batch_size, topk),
            -1,
            dtype=torch.int32,
            device=q.device,
        )
        _decode_bnsd_score_topk_chunk_kernel[
            (batch_size * num_score_chunks, num_kv_heads)
        ](
            q,
            k_cache_bnsd,
            page_source,
            page_source,
            page_source_rows,
            candidate_scores,
            candidate_indices,
            seq_lens,
            stride_sl_b,
            stride_sl_h,
            batch_size,
            gqa_group_size,
            head_dim,
            block_size,
            sm_scale,
            init_blocks,
            local_blocks,
            num_score_chunks,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k_cache_bnsd.stride(0),
            k_cache_bnsd.stride(1),
            k_cache_bnsd.stride(2),
            k_cache_bnsd.stride(3),
            page_source.stride(0),
            page_source.stride(1),
            req_to_token.stride(0) if use_direct_page_lookup else 0,
            req_to_token.stride(1) if use_direct_page_lookup else 0,
            req_to_token.shape[1] if use_direct_page_lookup else 1,
            direct_num_pages,
            candidate_scores.stride(0),
            candidate_scores.stride(1),
            candidate_scores.stride(2),
            candidate_scores.stride(3),
            candidate_indices.stride(0),
            candidate_indices.stride(1),
            candidate_indices.stride(2),
            candidate_indices.stride(3),
            BLOCK_SIZE_N=block_size_n,
            SCORE_TYPE=score_type,
            topk=topk,
            USE_DIRECT_PAGE_LOOKUP=use_direct_page_lookup,
            SANITIZE_PAGE_IDS=sanitize_page_ids,
            FILL_ONLY=(
                ((max_seqblock + num_score_chunks - 1) // num_score_chunks) <= topk
            ),
            RUNTIME_FILL_ONLY=runtime_fill_only,
            RUNTIME_ADAPTIVE_SCORE_CHUNKS=use_runtime_adaptive_score_chunks,
            RUNTIME_SCORE_SHORT_MAX_BLOCKS=runtime_score_short_max_blocks,
            RUNTIME_SCORE_SHORT_CHUNKS=runtime_score_short_chunks,
            num_warps=_SCORE_CHUNK_NW,
            num_stages=_SCORE_CHUNK_NS,
        )
        if fused_append_local:
            assert max_num_blocks is not None and max_num_blocks > 0
            if use_runtime_adaptive_score_chunks:
                topk_indices = _merge_topk_append_local_adaptive(
                    candidate_scores,
                    candidate_indices,
                    seq_lens,
                    block_size,
                    topk,
                    stride_sl_b,
                    stride_sl_h,
                    max_num_blocks,
                    runtime_score_short_max_blocks,
                    runtime_score_short_chunks,
                )
            else:
                topk_indices = _merge_topk_append_local(
                    candidate_scores,
                    candidate_indices,
                    seq_lens,
                    block_size,
                    topk,
                    stride_sl_b,
                    stride_sl_h,
                    max_num_blocks,
                )
        elif use_runtime_adaptive_score_chunks:
            topk_indices = _merge_bnsd_score_topk_candidates_adaptive(
                candidate_scores,
                candidate_indices,
                seq_lens,
                block_size,
                topk,
                stride_sl_b,
                stride_sl_h,
                runtime_score_short_max_blocks,
                runtime_score_short_chunks,
            )
        else:
            topk_indices = _merge_bnsd_score_topk_candidates(
                candidate_scores, candidate_indices, topk
            )
        return None, topk_indices

    if num_kv_chunks is None:
        num_kv_chunks = _choose_num_kv_chunks(
            batch_size,
            num_kv_heads,
            max_num_kv_chunks=max_num_kv_chunks,
        )
    else:
        num_kv_chunks = int(num_kv_chunks)

    assert num_kv_chunks >= 1
    assert (num_kv_chunks & (num_kv_chunks - 1)) == 0

    o_chunks = torch.empty(
        (num_kv_chunks, batch_size, num_q_heads, head_dim),
        dtype=q.dtype,
        device=q.device,
    )
    lse_chunks = torch.empty(
        (num_kv_chunks, batch_size, num_q_heads),
        dtype=torch.float32,
        device=q.device,
    )
    candidate_scores = torch.empty(
        (num_kv_chunks, num_q_heads, batch_size, topk),
        dtype=torch.float32,
        device=q.device,
    )
    candidate_indices = torch.empty(
        (num_kv_chunks, num_q_heads, batch_size, topk),
        dtype=torch.int32,
        device=q.device,
    )

    grid_attn = (batch_size * num_kv_chunks, num_kv_heads)
    _decode_bnsd_score_attn_chunk_kernel[grid_attn](
        q,
        sink,
        k_cache_bnsd,
        v_cache_bnsd,
        block_table,
        o_chunks,
        lse_chunks,
        candidate_scores,
        candidate_indices,
        seq_lens,
        batch_size,
        gqa_group_size,
        head_dim,
        block_size,
        sm_scale,
        init_blocks,
        local_blocks,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        sink.stride(0) if sink is not None else 0,
        sink.stride(1) if sink is not None else 0,
        k_cache_bnsd.stride(0),
        k_cache_bnsd.stride(1),
        k_cache_bnsd.stride(2),
        k_cache_bnsd.stride(3),
        v_cache_bnsd.stride(0),
        v_cache_bnsd.stride(1),
        v_cache_bnsd.stride(2),
        v_cache_bnsd.stride(3),
        block_table.stride(0),
        block_table.stride(1),
        o_chunks.stride(0),
        o_chunks.stride(1),
        o_chunks.stride(2),
        o_chunks.stride(3),
        lse_chunks.stride(0),
        lse_chunks.stride(1),
        lse_chunks.stride(2),
        candidate_scores.stride(0),
        candidate_scores.stride(1),
        candidate_scores.stride(2),
        candidate_scores.stride(3),
        candidate_indices.stride(0),
        candidate_indices.stride(1),
        candidate_indices.stride(2),
        candidate_indices.stride(3),
        BLOCK_SIZE_N=block_size_n,
        NUM_KV_CHUNKS=num_kv_chunks,
        SCORE_TYPE=score_type,
        topk=topk,
        num_warps=4,
        num_stages=2,
    )

    topk_idx = _merge_bnsd_score_topk_candidates(
        candidate_scores, candidate_indices, topk
    )

    o = torch.empty_like(q)

    grid_merge = (batch_size, num_q_heads)
    _merge_bnsd_attn_out_kernel[grid_merge](
        o_chunks,
        lse_chunks,
        seq_lens,
        o,
        head_dim,
        block_size,
        o_chunks.stride(0),
        o_chunks.stride(1),
        o_chunks.stride(2),
        o_chunks.stride(3),
        lse_chunks.stride(0),
        lse_chunks.stride(1),
        lse_chunks.stride(2),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        NUM_KV_CHUNKS=num_kv_chunks,
        num_warps=4,
        num_stages=2,
    )

    return o, topk_idx
