"""Triton port of the DSV4 DSpark tree sparse attention operator.

Public entry point: :func:`sparse_attn_sharedkv_tree`, a Python-only port of
``csrc/attention/sparse_attn_sharedkv_tree`` from vllm-ascend commit
``8947f1f9f``.  The authoritative semantics live in
``docs/tree_ops_semantics.md`` §5 and the operator quick contract in
``tree/README.md``.

Design notes
------------
* One kernel, grid ``(T, cdiv(N1, 16))``.  A program owns the full 512-wide head
  dimension: QK is formed once from the four 128-wide K slices, then the four
  V slices are accumulated sequentially from the same score block.  No
  cross-program reduction is needed, QK is not recomputed per D slice, and the
  per-program UB footprint stays bounded for ``head_dim = 512`` (risk R9 in the
  brief).  The 512-row vendor merge buffer is a later step; this program still
  writes all four D slices of its own output rows directly.
* Original-KV slots are physical indices and are consumed in order up to the
  first ``-1`` (capped by ``ori_sparse_indices.shape[-1]`` and ``seqused_kv``).
  QK and PV gather the *same* offset vector, so slot identity is exact.  A
  separate index-only pass computes that valid count, and the scoring loop is
  bounded by it, so no padded chunk is ever scored.
* Compressed slots are resolved on device per §1.3: ``k < prefix // ratio`` maps
  through ``cmp_block_table`` into the paged pool, everything else maps through
  ``(k + 1) * ratio - 1 - prefix`` and ``tree_ancestors`` into the node pool.
  A negative ``tree_prefix`` resolves to no key at all.  C4 (``cmp_sparse_indices``
  supplied) filters logical positions and visits at most ``min(K, history)``
  columns; C128 omits it and walks the full compressed history.
* Softmax is online with an optional sink seeded as ``max = sink[h]`` and
  ``sum = 1`` (the sink's value is zero).  Probabilities are cast back to the KV
  dtype before PV, accumulation stays FP32, and an empty row (no key at all)
  produces exact zeros for both output and LSE.  ``softmax_scale`` is applied
  verbatim; a zero value is not replaced by an implicit ``1/sqrt(d)`` so the
  kernel stays aligned with the independent reference.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ._common import (
    TreeContractError,
    check_layout,
    check_rank,
    check_ratio,
    check_same_device,
    check_tensor,
    check_tree_metadata,
)

__all__ = ["sparse_attn_sharedkv_tree"]

_HEAD_DIM = 512
# A3 UB budget: the 16x128 attention tile with four Q/K slices exceeds
# 1,572,864 bits after multi-buffering/code motion.  Keep the cube M tile at
# 16 and halve the slot tile; the score/PV math is unchanged and the D slice
# remains 128.
#
# Full-D program now that the D-slice grid axis is gone: QK is formed once and
# the four V slices are accumulated sequentially, so the live tile set is
#   Q0..Q3 4 x 16x128 bf16 = 16,384 B
#   one K  64x128 bf16      = 16,384 B   (x2 if double-buffered = 32,768)
#   qk     16x64  f32       =  4,096 B
#   probs  16x64  f32       =  4,096 B   (+ 2,048 B bf16 cast)
#   acc0..3 4 x 16x128 f32  = 32,768 B
#   one V  64x128 bf16      = 16,384 B   (x2 if double-buffered = 32,768)
# =~ 90 KB single-buffered, =~ 122 KB with K and V double-buffered, i.e. under
# the 192 KB UB budget with room for code motion.
_BLOCK_H = 16
_BLOCK_N = 64
_BLOCK_D = 128


@triton.jit
def _attend_full(
    KV,
    OFFS,
    VALID,
    Q0,
    Q1,
    Q2,
    Q3,
    M,
    L,
    ACC0,
    ACC1,
    ACC2,
    ACC3,
    SCALE,
    LIMIT: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Score one chunk once and merge every V slice into the online state.

    ``OFFS`` is the element offset of each key row (int64) and ``VALID`` marks
    the slots that survived the source-specific resolution.  K and V share the
    row, so the same ``OFFS`` drives QK and PV.

    ``LIMIT`` is a final address guard: a slot that is marked valid by the
    caller but lands outside its source pool (or at a negative offset) is
    forced to the safe zero row.  Ascend still forms masked-load addresses.

    The 512-wide QK is produced exactly once (four 128-wide K dots), so
    ``m_new``/``alpha``/``probs``/``l_new`` are also computed once.  The four
    V slices are then folded in sequentially from the same row offsets, which
    removes the former per-D-slice grid axis and its 4x QK recompute.
    """
    d = tl.arange(0, BLOCK_D)
    in_range = VALID & (OFFS >= 0) & (OFFS < LIMIT)
    offs = tl.where(in_range, OFFS, 0)
    mask = in_range[:, None]

    # Sequential 128-wide K loads: each K chunk dies before the next is issued,
    # keeping the UB footprint bounded at head_dim 512 (risk R9).
    k = tl.load(KV + offs[:, None] + (0 * BLOCK_D + d)[None, :], mask=mask, other=0.0)
    qk = tl.dot(Q0, tl.trans(k))
    k = tl.load(KV + offs[:, None] + (1 * BLOCK_D + d)[None, :], mask=mask, other=0.0)
    qk += tl.dot(Q1, tl.trans(k))
    k = tl.load(KV + offs[:, None] + (2 * BLOCK_D + d)[None, :], mask=mask, other=0.0)
    qk += tl.dot(Q2, tl.trans(k))
    k = tl.load(KV + offs[:, None] + (3 * BLOCK_D + d)[None, :], mask=mask, other=0.0)
    qk += tl.dot(Q3, tl.trans(k))
    qk = tl.where(in_range[None, :], qk * SCALE, float("-inf"))

    m_new = tl.maximum(M, tl.max(qk, 1))
    alpha = tl.exp(M - m_new)
    probs = tl.exp(qk - m_new[:, None])
    l_new = L * alpha + tl.sum(probs, 1)

    # All four V dots are unconditional: a runtime branch around ``tl.dot``
    # splits the MIX sub-block and the Ascend binder rejects it.
    probs_v = probs.to(KV.dtype.element_ty)
    v = tl.load(KV + offs[:, None] + (0 * BLOCK_D + d)[None, :], mask=mask, other=0.0)
    acc0 = ACC0 * alpha[:, None] + tl.dot(probs_v, v)
    v = tl.load(KV + offs[:, None] + (1 * BLOCK_D + d)[None, :], mask=mask, other=0.0)
    acc1 = ACC1 * alpha[:, None] + tl.dot(probs_v, v)
    v = tl.load(KV + offs[:, None] + (2 * BLOCK_D + d)[None, :], mask=mask, other=0.0)
    acc2 = ACC2 * alpha[:, None] + tl.dot(probs_v, v)
    v = tl.load(KV + offs[:, None] + (3 * BLOCK_D + d)[None, :], mask=mask, other=0.0)
    acc3 = ACC3 * alpha[:, None] + tl.dot(probs_v, v)
    return m_new, l_new, acc0, acc1, acc2, acc3


@triton.jit
def _tree_attn_kernel(
    Q,
    OUT,
    LSE,
    ORI_KV,
    ORI_IDX,
    SEQUSED,
    CMP_KV,
    CMP_IDX,
    CMP_TABLE,
    TREE_KV,
    ANCESTORS,
    PREFIX,
    SINKS,
    SCALE,
    N1: tl.constexpr,
    ORI_W: tl.constexpr,
    ORI_ROWS: tl.constexpr,
    ORI_PAGES: tl.constexpr,
    ORI_S0: tl.constexpr,
    ORI_S1: tl.constexpr,
    CMP_ROWS: tl.constexpr,
    CMP_PAGES: tl.constexpr,
    CMP_S0: tl.constexpr,
    CMP_S1: tl.constexpr,
    CMP_TABLE_W: tl.constexpr,
    CMP_TABLE_S0: tl.constexpr,
    CMP_K: tl.constexpr,
    MAX_DEPTH: tl.constexpr,
    NODE_COUNT: tl.constexpr,
    NODE_D: tl.constexpr,
    ORI_LIMIT: tl.constexpr,
    CMP_LIMIT: tl.constexpr,
    TREE_LIMIT: tl.constexpr,
    SINK_STRIDE_T: tl.constexpr,
    SINK_STRIDE_H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    RATIO: tl.constexpr,
    HAS_ORI: tl.constexpr,
    HAS_CMP: tl.constexpr,
    USE_C4: tl.constexpr,
    HAS_SINK: tl.constexpr,
    SINK_PER_ROW: tl.constexpr,
    RETURN_LSE: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)

    heads = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    head_mask = heads < N1
    # Masked loads still form their addresses; use a safe head for the tail
    # lanes rather than forming an address past the Q/O/SINKS allocation.
    head_safe = tl.where(head_mask, heads, 0)
    d = tl.arange(0, BLOCK_D)

    q_base = Q + pid_t * (N1 * NODE_D) + head_safe[:, None] * NODE_D
    q0 = tl.load(q_base + (0 * BLOCK_D + d)[None, :], mask=head_mask[:, None], other=0.0)
    q1 = tl.load(q_base + (1 * BLOCK_D + d)[None, :], mask=head_mask[:, None], other=0.0)
    q2 = tl.load(q_base + (2 * BLOCK_D + d)[None, :], mask=head_mask[:, None], other=0.0)
    q3 = tl.load(q_base + (3 * BLOCK_D + d)[None, :], mask=head_mask[:, None], other=0.0)

    if HAS_SINK:
        sink_row = pid_t if SINK_PER_ROW else 0
        m = tl.load(
            SINKS + sink_row * SINK_STRIDE_T + head_safe * SINK_STRIDE_H,
            mask=head_mask,
            other=0.0,
        )
        l = tl.zeros([BLOCK_H], dtype=tl.float32) + 1.0
    else:
        # Finite floor instead of -inf: chunk scans may visit slots that are all
        # invalid, and ``exp(-inf - -inf)`` would produce NaN.
        m = tl.zeros([BLOCK_H], dtype=tl.float32) - 1.0e30
        l = tl.zeros([BLOCK_H], dtype=tl.float32)

    # Four parallel accumulators, one per 128-wide output slice.  Keeping them
    # separate (instead of one [BLOCK_H, 512] tile) lets each PV dot stay at a
    # legal 128-wide N and keeps the live tile set explicit.
    acc0 = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)
    acc1 = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)
    acc2 = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)
    acc3 = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)
    seen = tl.zeros([1], dtype=tl.int32)

    seqused = tl.load(SEQUSED + pid_t)

    if HAS_ORI:
        # Original-KV slots: ``GetOriSparseActualSeqLen`` (vendor
        # ``sparse_attn_sharedkv_tree_scfa_kernel.h:361-382``) counts the
        # leading entries before the first -1, capped by
        # ``min(ORI_W, seqused)``.  The count is taken in a dedicated pass over
        # the int32 slot table only, so walking the full padded width never
        # forms a KV address.  Taking the min across chunks makes the result
        # identical to the vendor's leading-run scan: a -1 can never resurrect
        # a later physical slot.
        valid_count = tl.minimum(seqused, ORI_W)
        for start in range(0, ORI_W, BLOCK_N):
            cols = start + tl.arange(0, BLOCK_N)
            cols_safe = tl.where(cols < ORI_W, cols, 0)
            slot = tl.load(
                ORI_IDX + pid_t * ORI_W + cols_safe,
                mask=cols < ORI_W,
                other=-1,
            )
            first_neg = tl.min(tl.where(slot < 0, cols, ORI_W))
            valid_count = tl.minimum(valid_count, first_neg)

        # Runtime-bounded loop, mirroring the vendor's
        # ``CeilDiv(oriS2Size, s2BaseSize)`` trip count: a chunk starting at or
        # past ``valid_count`` holds no key, so it is never scored.
        for start in tl.range(0, valid_count, BLOCK_N):
            cols = start + tl.arange(0, BLOCK_N)
            range_ok = cols < valid_count
            cols_safe = tl.where(range_ok, cols, 0)
            slot = tl.load(
                ORI_IDX + pid_t * ORI_W + cols_safe,
                mask=range_ok,
                other=-1,
            )

            page = slot // ORI_ROWS
            page_ok = (slot >= 0) & (page >= 0) & (page < ORI_PAGES)
            # ``range_ok`` already encodes ``cols < valid_count``, which is
            # ``<= min(ORI_W, seqused, first -1)``.
            valid = range_ok & page_ok
            # A padded -1 slot makes ``page`` negative.  Clamp every pointer
            # input for masked lanes: Ascend still forms the address even when
            # the load mask is false.
            page_safe = tl.where(page_ok, page, 0)
            slot_safe = tl.where(page_ok, slot, 0)
            off = page_safe.to(tl.int64) * ORI_S0 + (
                slot_safe - page_safe * ORI_ROWS
            ).to(tl.int64) * ORI_S1

            seen += tl.sum(valid.to(tl.int32))
            m, l, acc0, acc1, acc2, acc3 = _attend_full(
                ORI_KV, off, valid, q0, q1, q2, q3, m, l,
                acc0, acc1, acc2, acc3, SCALE, ORI_LIMIT, BLOCK_D, BLOCK_N,
            )

    if HAS_CMP:
        prefix = tl.load(PREFIX + pid_t)
        # ``GetKeyGmOffset`` (vendor
        # ``sparse_attn_sharedkv_tree_scfa_block_vector.h:545-557``) returns -1
        # for a negative prefix and never resolves a pool, so an invalid tree
        # context contributes no key at all.
        prefix_ok = prefix >= 0
        # ``cmpMaskRight = actOriS2Size - actS1Size`` and ``s1EndIdx = 0`` for
        # the per-node wrapper, hence
        # ``thresHold = (cmpMaskRight + s1EndIdx + 1) / cmpRatio = seqused // RATIO``
        # (vendor ``sparse_attn_sharedkv_tree_scfa_kernel.h:395-396``).
        history_limit = seqused // RATIO
        if USE_C4:
            # ``actCmpS2Size = min(sparseBlockCount * sparseBlockSize, thresHold)``:
            # C4 visits at most ``min(CMP_K, history_limit)`` top-k columns, and
            # every visited column is still filtered by its logical value.
            cmp_limit = tl.minimum(CMP_K, history_limit)
        else:
            # C128 walks the full compressed history; the top-k array is absent.
            cmp_limit = history_limit

        for start in tl.range(0, cmp_limit, BLOCK_N):
            cols = start + tl.arange(0, BLOCK_N)
            range_ok = cols < cmp_limit
            cols_safe = tl.where(range_ok, cols, 0)
            if USE_C4:
                logical = tl.load(
                    CMP_IDX + pid_t * CMP_K + cols_safe,
                    mask=range_ok,
                    other=-1,
                )
                idx_ok = range_ok & (logical >= 0) & (logical < history_limit)
            else:
                logical = cols
                idx_ok = range_ok

            paged_end = prefix // RATIO

            # Paged compressed history: logical -> (page, row) via the table.
            is_paged = prefix_ok & idx_ok & (logical >= 0) & (logical < paged_end)
            page_col = logical // CMP_ROWS
            page_col_ok = is_paged & (page_col >= 0) & (page_col < CMP_TABLE_W)
            page_col_safe = tl.where(page_col_ok, page_col, 0)
            page = tl.load(
                CMP_TABLE + pid_t * CMP_TABLE_S0 + page_col_safe,
                mask=page_col_ok,
                other=-1,
            )
            paged_valid = page_col_ok & (page >= 0) & (page < CMP_PAGES)
            page_safe = tl.where(paged_valid, page, 0)
            logical_safe = tl.where(paged_valid, logical, 0)
            paged_off = page_safe.to(tl.int64) * CMP_S0 + (
                logical_safe - page_col_safe * CMP_ROWS
            ).to(tl.int64) * CMP_S1
            seen += tl.sum(paged_valid.to(tl.int32))
            m, l, acc0, acc1, acc2, acc3 = _attend_full(
                CMP_KV, paged_off, paged_valid, q0, q1, q2, q3, m, l,
                acc0, acc1, acc2, acc3, SCALE, CMP_LIMIT, BLOCK_D, BLOCK_N,
            )

            # Tree node pool: step = (k + 1) * ratio - 1 - prefix, root-first.
            is_tree = prefix_ok & idx_ok & (logical >= 0) & (logical >= paged_end)
            step = (logical + 1) * RATIO - 1 - prefix
            step_ok = is_tree & (logical >= 0) & (step >= 0) & (step < MAX_DEPTH)
            step_safe = tl.where(step_ok, step, 0)
            node = tl.load(
                ANCESTORS + pid_t * MAX_DEPTH + step_safe,
                mask=step_ok,
                other=-1,
            )
            tree_valid = step_ok & (node >= 0) & (node < NODE_COUNT)
            node_safe = tl.where(tree_valid, node, 0)
            tree_off = node_safe * NODE_D
            seen += tl.sum(tree_valid.to(tl.int32))
            m, l, acc0, acc1, acc2, acc3 = _attend_full(
                TREE_KV, tree_off, tree_valid, q0, q1, q2, q3, m, l,
                acc0, acc1, acc2, acc3, SCALE, TREE_LIMIT, BLOCK_D, BLOCK_N,
            )

    # Branch-free empty-row handling: an `if` here also splits the MIX
    # sub-block after the dot chain, which the Ascend binder rejects.
    has_seen = tl.sum(seen) > 0
    inv_l = tl.where(has_seen, 1.0 / l, 0.0)
    out_base = OUT + pid_t * (N1 * NODE_D) + head_safe[:, None] * NODE_D
    tl.store(
        out_base + (0 * BLOCK_D + d)[None, :],
        (acc0 * inv_l[:, None]).to(OUT.dtype.element_ty),
        mask=head_mask[:, None],
    )
    tl.store(
        out_base + (1 * BLOCK_D + d)[None, :],
        (acc1 * inv_l[:, None]).to(OUT.dtype.element_ty),
        mask=head_mask[:, None],
    )
    tl.store(
        out_base + (2 * BLOCK_D + d)[None, :],
        (acc2 * inv_l[:, None]).to(OUT.dtype.element_ty),
        mask=head_mask[:, None],
    )
    tl.store(
        out_base + (3 * BLOCK_D + d)[None, :],
        (acc3 * inv_l[:, None]).to(OUT.dtype.element_ty),
        mask=head_mask[:, None],
    )

    # LSE is head-only, not per D-slice: this program owns the whole head block
    # now, so a single masked store writes it.  A store mask is still used
    # instead of a runtime branch for the same MIX-binder reason.
    if RETURN_LSE:
        lse = tl.where(has_seen, tl.log(l) + m, 0.0)
        tl.store(
            LSE + pid_t * N1 + head_safe,
            lse,
            mask=head_mask,
        )


def _validate_sinks(sinks, num_heads: int, num_rows: int):
    """Return ``(stride_t, stride_h, per_row)`` for the three sink layouts."""
    check_tensor("sinks", sinks, dtype=torch.float32)
    check_rank("sinks", sinks, (1, 2, 3))
    if sinks.dim() == 1:
        if sinks.numel() != num_heads:
            raise TreeContractError("sinks: head count mismatch")
        return 0, sinks.stride(0), False
    if sinks.dim() == 2:
        rows, width = int(sinks.shape[0]), int(sinks.shape[1])
        if width != num_heads or rows not in (1, num_rows):
            raise TreeContractError("sinks: expected [heads] or [T, heads]")
        stride_t = 0 if rows == 1 else sinks.stride(0)
        return stride_t, sinks.stride(1), rows == num_rows
    if int(sinks.shape[1]) != 1 or int(sinks.shape[2]) != num_heads:
        raise TreeContractError("sinks: expected [T, 1, heads]")
    rows = int(sinks.shape[0])
    if rows not in (1, num_rows):
        raise TreeContractError("sinks: row count mismatch")
    stride_t = 0 if rows == 1 else sinks.stride(0)
    return stride_t, sinks.stride(2), rows == num_rows


def _validate_cu(name: str, cu, num_rows: int) -> None:
    if cu is None:
        return
    check_tensor(name, cu, dtype=torch.int32, rank=1, contiguous=True)
    if cu.numel() != num_rows + 1:
        raise TreeContractError(f"{name}: expected {num_rows + 1} entries")


def _validate_paged_kv(name: str, tensor, dtype: torch.dtype) -> None:
    check_tensor(name, tensor, dtype=dtype, rank=4)
    if int(tensor.shape[2]) != 1 or int(tensor.shape[3]) != _HEAD_DIM:
        raise TreeContractError(
            f"{name}: expected [pages, block, 1, {_HEAD_DIM}], got "
            f"{tuple(tensor.shape)}"
        )
    if int(tensor.stride(3)) != 1:
        raise TreeContractError(f"{name}: last dimension must be innermost-contiguous")
    if int(tensor.shape[0]) <= 0 or int(tensor.shape[1]) <= 0:
        raise TreeContractError(
            f"{name}: expected non-empty [pages, block], got {tuple(tensor.shape)}"
        )


def sparse_attn_sharedkv_tree(
    q,
    *,
    ori_kv=None,
    cmp_kv=None,
    ori_sparse_indices=None,
    cmp_sparse_indices=None,
    ori_block_table=None,
    cmp_block_table=None,
    cu_seqlens_q=None,
    cu_seqlens_ori_kv=None,
    cu_seqlens_cmp_kv=None,
    seqused_q=None,
    seqused_kv=None,
    sinks=None,
    metadata=None,
    cmp_tree_kv=None,
    tree_ancestors=None,
    tree_prefix=None,
    softmax_scale=0,
    cmp_ratio=0,
    ori_mask_mode=4,
    cmp_mask_mode=3,
    ori_win_left=128,
    ori_win_right=0,
    layout_q="TND",
    layout_kv="PA_ND",
    return_softmax_lse=False,
):
    """Tree sparse attention over a shared KV cache.

    Returns ``(out, softmax_lse)``; ``softmax_lse`` is ``[T, N1, 1]`` FP32 when
    requested and an empty FP32 tensor otherwise (matching the reference).
    """
    check_tensor(
        "q",
        q,
        dtype=(torch.float16, torch.bfloat16),
        rank=3,
        contiguous=True,
    )
    num_rows, num_heads, head_dim = (int(v) for v in q.shape)
    if num_rows <= 0 or num_heads <= 0:
        raise TreeContractError("q: all dimensions must be > 0")
    if head_dim != _HEAD_DIM:
        raise TreeContractError(
            f"q: head_dim must be {_HEAD_DIM}, got {head_dim}"
        )

    check_layout(layout_q, name="layout_q")
    check_layout(layout_kv, name="layout_kv")
    if int(ori_mask_mode) != 4:
        raise TreeContractError(
            f"ori_mask_mode: tree attention requires 4, got {ori_mask_mode}"
        )
    if int(cmp_mask_mode) != 3:
        raise TreeContractError(
            f"cmp_mask_mode: tree attention requires 3, got {cmp_mask_mode}"
        )
    check_tree_metadata("metadata", metadata)

    if seqused_kv is None:
        raise TreeContractError("seqused_kv: required (logical prefix + ancestors)")
    check_tensor(
        "seqused_kv", seqused_kv, dtype=torch.int32, rank=1, contiguous=True
    )
    if seqused_kv.numel() != num_rows:
        raise TreeContractError("seqused_kv: expected one entry per query row")

    _validate_cu("cu_seqlens_q", cu_seqlens_q, num_rows)
    _validate_cu("cu_seqlens_ori_kv", cu_seqlens_ori_kv, num_rows)
    _validate_cu("cu_seqlens_cmp_kv", cu_seqlens_cmp_kv, num_rows)
    if seqused_q is not None:
        check_tensor(
            "seqused_q", seqused_q, dtype=torch.int32, rank=1, contiguous=True
        )
        if seqused_q.numel() not in (0, num_rows):
            raise TreeContractError("seqused_q: expected empty or [T]")
    if ori_block_table is not None:
        check_tensor(
            "ori_block_table",
            ori_block_table,
            dtype=torch.int32,
            rank=2,
            contiguous=True,
        )

    has_ori = ori_kv is not None
    if has_ori:
        _validate_paged_kv("ori_kv", ori_kv, q.dtype)
        if ori_sparse_indices is None:
            raise TreeContractError("ori_sparse_indices: required with ori_kv")
    elif ori_sparse_indices is not None:
        raise TreeContractError("ori_sparse_indices: requires ori_kv")

    if ori_sparse_indices is not None:
        check_tensor(
            "ori_sparse_indices",
            ori_sparse_indices,
            dtype=torch.int32,
            rank=3,
            contiguous=True,
        )
        if (
            int(ori_sparse_indices.shape[0]) != num_rows
            or int(ori_sparse_indices.shape[1]) != 1
        ):
            raise TreeContractError(
                "ori_sparse_indices: expected [T, 1, W]"
            )
        ori_width = int(ori_sparse_indices.shape[2])
        if ori_width <= 0 or ori_width % 128 != 0:
            raise TreeContractError(
                "ori_sparse_indices: width must be a positive multiple of 128"
            )
    else:
        ori_width = 0

    has_cmp = cmp_kv is not None
    use_c4 = False
    cmp_k = 0
    if has_cmp:
        _validate_paged_kv("cmp_kv", cmp_kv, q.dtype)
        ratio = check_ratio(int(cmp_ratio))
        if cmp_tree_kv is None or tree_ancestors is None or tree_prefix is None:
            raise TreeContractError(
                "cmp_tree_kv/tree_ancestors/tree_prefix: all-or-none with cmp_kv"
            )
        check_tensor(
            "cmp_tree_kv", cmp_tree_kv, dtype=q.dtype, rank=2, contiguous=True
        )
        if int(cmp_tree_kv.shape[1]) != _HEAD_DIM or int(cmp_tree_kv.shape[0]) <= 0:
            raise TreeContractError("cmp_tree_kv: expected [nodes, 512]")
        check_tensor(
            "tree_ancestors",
            tree_ancestors,
            dtype=torch.int64,
            rank=2,
            contiguous=True,
        )
        if (
            int(tree_ancestors.shape[0]) != num_rows
            or int(tree_ancestors.shape[1]) <= 0
        ):
            raise TreeContractError("tree_ancestors: expected [T, max_depth]")
        check_tensor(
            "tree_prefix", tree_prefix, dtype=torch.int32, rank=1, contiguous=True
        )
        if tree_prefix.numel() != num_rows:
            raise TreeContractError("tree_prefix: expected [T]")

        if cmp_sparse_indices is not None:
            if ratio != 4:
                raise TreeContractError(
                    "cmp_sparse_indices: only the C4 (ratio 4) path supplies "
                    "logical top-k positions"
                )
            check_tensor(
                "cmp_sparse_indices",
                cmp_sparse_indices,
                dtype=torch.int32,
                rank=3,
                contiguous=True,
            )
            if (
                int(cmp_sparse_indices.shape[0]) != num_rows
                or int(cmp_sparse_indices.shape[1]) != 1
                or int(cmp_sparse_indices.shape[2]) <= 0
            ):
                raise TreeContractError("cmp_sparse_indices: expected [T, 1, K]")
            use_c4 = True
            cmp_k = int(cmp_sparse_indices.shape[2])
        elif ratio != 128:
            raise TreeContractError(
                "cmp_sparse_indices: C128 (ratio 128) walks the full history and "
                "must omit logical top-k positions"
            )
        if cmp_block_table is not None:
            check_tensor(
                "cmp_block_table",
                cmp_block_table,
                dtype=torch.int32,
                rank=2,
                contiguous=True,
            )
            if int(cmp_block_table.shape[0]) != num_rows:
                raise TreeContractError("cmp_block_table: expected one row per token")
    else:
        if cmp_sparse_indices is not None:
            raise TreeContractError("cmp_sparse_indices: requires cmp_kv")
        if cmp_tree_kv is not None or tree_ancestors is not None or tree_prefix is not None:
            raise TreeContractError(
                "cmp_tree_kv/tree_ancestors/tree_prefix: require cmp_kv"
            )
        ratio = 4  # unused; keeps the constexpr specialization valid for SWA.

    tensors = [
        t
        for t in (
            q,
            ori_kv,
            cmp_kv,
            ori_sparse_indices,
            cmp_sparse_indices,
            ori_block_table,
            cmp_block_table,
            cu_seqlens_q,
            cu_seqlens_ori_kv,
            cu_seqlens_cmp_kv,
            seqused_q,
            seqused_kv,
            sinks,
            metadata,
            cmp_tree_kv,
            tree_ancestors,
            tree_prefix,
        )
        if isinstance(t, torch.Tensor)
    ]
    check_same_device("sparse_attn_sharedkv_tree", tensors)

    sink_stride_t, sink_stride_h, sink_per_row = (
        _validate_sinks(sinks, num_heads, num_rows)
        if sinks is not None
        else (0, 0, False)
    )

    out = torch.empty_like(q)
    if return_softmax_lse:
        softmax_lse = torch.zeros(
            (num_rows, num_heads, 1), dtype=torch.float32, device=q.device
        )
    else:
        softmax_lse = torch.empty(0, dtype=torch.float32, device=q.device)

    # Unused pointer arguments still need a tensor of the expected element type;
    # every load/store on them is removed by a constexpr guard, so their contents
    # and shape are irrelevant.  Length-1 dummies avoid zero-extent pointers.
    src_dummy = q
    i32_dummy = seqused_kv
    i64_dummy = torch.empty(1, dtype=torch.int64, device=q.device)
    f32_dummy = torch.empty(1, dtype=torch.float32, device=q.device)
    lse_ptr = softmax_lse if return_softmax_lse else f32_dummy

    ori_rows = int(ori_kv.shape[1]) if has_ori else 0
    ori_pages = int(ori_kv.shape[0]) if has_ori else 0
    ori_s0 = int(ori_kv.stride(0)) if has_ori else 0
    ori_s1 = int(ori_kv.stride(1)) if has_ori else 0

    cmp_rows = int(cmp_kv.shape[1]) if has_cmp else 0
    cmp_pages = int(cmp_kv.shape[0]) if has_cmp else 0
    cmp_s0 = int(cmp_kv.stride(0)) if has_cmp else 0
    cmp_s1 = int(cmp_kv.stride(1)) if has_cmp else 0
    if has_cmp and cmp_block_table is not None:
        cmp_table_w = int(cmp_block_table.shape[1])
        cmp_table_s0 = int(cmp_block_table.stride(0))
    else:
        cmp_table_w = 0
        cmp_table_s0 = 0

    max_depth = int(tree_ancestors.shape[1]) if has_cmp else 0
    node_count = int(cmp_tree_kv.shape[0]) if has_cmp else 0

    # One program per (row, head block): the program owns the full 512-wide D,
    # so there is no D-slice grid axis and QK is not recomputed per slice.
    grid = (
        num_rows,
        triton.cdiv(num_heads, _BLOCK_H),
    )
    _tree_attn_kernel[grid](
        q,
        out,
        lse_ptr,
        ori_kv if has_ori else src_dummy,
        ori_sparse_indices if ori_sparse_indices is not None else i32_dummy,
        seqused_kv,
        cmp_kv if has_cmp else src_dummy,
        cmp_sparse_indices if use_c4 else i32_dummy,
        cmp_block_table if cmp_block_table is not None else i32_dummy,
        cmp_tree_kv if has_cmp else src_dummy,
        tree_ancestors if has_cmp else i64_dummy,
        tree_prefix if has_cmp else i32_dummy,
        sinks if sinks is not None else f32_dummy,
        float(softmax_scale),
        N1=num_heads,
        ORI_W=ori_width,
        ORI_ROWS=ori_rows,
        ORI_PAGES=ori_pages,
        ORI_S0=ori_s0,
        ORI_S1=ori_s1,
        CMP_ROWS=cmp_rows,
        CMP_PAGES=cmp_pages,
        CMP_S0=cmp_s0,
        CMP_S1=cmp_s1,
        CMP_TABLE_W=cmp_table_w,
        CMP_TABLE_S0=cmp_table_s0,
        CMP_K=cmp_k,
        MAX_DEPTH=max_depth,
        NODE_COUNT=node_count,
        NODE_D=_HEAD_DIM,
        ORI_LIMIT=(ori_pages * ori_s0 if has_ori else 0),
        CMP_LIMIT=(cmp_pages * cmp_s0 if has_cmp else 0),
        TREE_LIMIT=(node_count * _HEAD_DIM),
        SINK_STRIDE_T=sink_stride_t,
        SINK_STRIDE_H=sink_stride_h,
        BLOCK_H=_BLOCK_H,
        BLOCK_N=_BLOCK_N,
        BLOCK_D=_BLOCK_D,
        RATIO=ratio,
        HAS_ORI=has_ori,
        HAS_CMP=has_cmp,
        USE_C4=use_c4,
        HAS_SINK=sinks is not None,
        SINK_PER_ROW=sink_per_row,
        RETURN_LSE=bool(return_softmax_lse),
        num_warps=1,
        num_stages=1,
    )
    return out, softmax_lse
