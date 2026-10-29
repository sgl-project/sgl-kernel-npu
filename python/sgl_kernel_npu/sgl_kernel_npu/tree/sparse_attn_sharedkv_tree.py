"""Triton port of the DSV4 DSpark tree sparse attention operator.

Public entry point: :func:`sparse_attn_sharedkv_tree`, a Python-only port of
``csrc/attention/sparse_attn_sharedkv_tree`` from vllm-ascend commit
``8947f1f9f``.  The authoritative semantics live in
``docs/tree_ops_semantics.md`` §5 and the operator quick contract in
``tree/README.md``.

Design notes
------------
* One kernel, grid ``(T, cdiv(N1, 16), 4)``.  The third axis splits the 512-wide
  head dimension into four 128-wide V slices.  Every program accumulates the
  full 512-wide QK in four 128-wide dot calls and owns one V slice, so no
  cross-program reduction is needed and the per-program UB footprint stays
  bounded for ``head_dim = 512`` (risk R9 in the brief).  The trade-off is that
  QK is recomputed once per V slice.
* Original-KV slots are physical indices and are consumed in order up to the
  first ``-1`` (capped by ``ori_sparse_indices.shape[-1]`` and ``seqused_kv``).
  QK and PV gather the *same* offset vector, so slot identity is exact.
* Compressed slots are resolved on device per §1.3: ``k < prefix // ratio`` maps
  through ``cmp_block_table`` into the paged pool, everything else maps through
  ``(k + 1) * ratio - 1 - prefix`` and ``tree_ancestors`` into the node pool.
  C4 (``cmp_sparse_indices`` supplied) filters logical positions; C128 omits it
  and walks the full compressed history.
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
_BLOCK_H = 16
_BLOCK_N = 128
_BLOCK_D = 128
_NUM_D_CHUNKS = _HEAD_DIM // _BLOCK_D


@triton.jit
def _attend_chunk(
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
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Score one 128-slot chunk and merge it into the online softmax state.

    ``OFFS`` is the element offset of each key row (int64) and ``VALID`` marks
    the slots that survived the source-specific resolution.  K and V share the
    row, so the same ``OFFS`` drives QK and PV.
    """
    d = tl.arange(0, BLOCK_D)
    mask = VALID[:, None]

    # Sequential 128-wide loads: each K chunk dies before the next is issued,
    # keeping the UB footprint bounded at head_dim 512 (risk R9).
    k = tl.load(KV + OFFS[:, None] + (0 * BLOCK_D + d)[None, :], mask=mask, other=0.0)
    qk = tl.dot(Q0, tl.trans(k))
    k = tl.load(KV + OFFS[:, None] + (1 * BLOCK_D + d)[None, :], mask=mask, other=0.0)
    qk += tl.dot(Q1, tl.trans(k))
    k = tl.load(KV + OFFS[:, None] + (2 * BLOCK_D + d)[None, :], mask=mask, other=0.0)
    qk += tl.dot(Q2, tl.trans(k))
    k = tl.load(KV + OFFS[:, None] + (3 * BLOCK_D + d)[None, :], mask=mask, other=0.0)
    qk += tl.dot(Q3, tl.trans(k))
    qk = tl.where(VALID[None, :], qk * SCALE, float("-inf"))

    m_new = tl.maximum(M, tl.max(qk, 1))
    alpha = tl.exp(M - m_new)
    probs = tl.exp(qk - m_new[:, None])
    l_new = L * alpha + tl.sum(probs, 1)

    v = tl.load(KV + OFFS[:, None] + (0 * BLOCK_D + d)[None, :], mask=mask, other=0.0)
    acc0_new = ACC0 * alpha[:, None] + tl.dot(probs.to(v.dtype), v)
    v = tl.load(KV + OFFS[:, None] + (1 * BLOCK_D + d)[None, :], mask=mask, other=0.0)
    acc1_new = ACC1 * alpha[:, None] + tl.dot(probs.to(v.dtype), v)
    v = tl.load(KV + OFFS[:, None] + (2 * BLOCK_D + d)[None, :], mask=mask, other=0.0)
    acc2_new = ACC2 * alpha[:, None] + tl.dot(probs.to(v.dtype), v)
    v = tl.load(KV + OFFS[:, None] + (3 * BLOCK_D + d)[None, :], mask=mask, other=0.0)
    acc3_new = ACC3 * alpha[:, None] + tl.dot(probs.to(v.dtype), v)
    return m_new, l_new, acc0_new, acc1_new, acc2_new, acc3_new


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
    d = tl.arange(0, BLOCK_D)

    q_base = Q + pid_t * (N1 * NODE_D) + heads[:, None] * NODE_D
    q0 = tl.load(q_base + (0 * BLOCK_D + d)[None, :], mask=head_mask[:, None], other=0.0)
    q1 = tl.load(q_base + (1 * BLOCK_D + d)[None, :], mask=head_mask[:, None], other=0.0)
    q2 = tl.load(q_base + (2 * BLOCK_D + d)[None, :], mask=head_mask[:, None], other=0.0)
    q3 = tl.load(q_base + (3 * BLOCK_D + d)[None, :], mask=head_mask[:, None], other=0.0)

    if HAS_SINK:
        sink_row = pid_t if SINK_PER_ROW else 0
        m = tl.load(
            SINKS + sink_row * SINK_STRIDE_T + heads * SINK_STRIDE_H,
            mask=head_mask,
            other=0.0,
        )
        l = tl.zeros([BLOCK_H], dtype=tl.float32) + 1.0
    else:
        # Finite floor instead of -inf: chunk scans may visit slots that are all
        # invalid, and ``exp(-inf - -inf)`` would produce NaN.
        m = tl.zeros([BLOCK_H], dtype=tl.float32) - 1.0e30
        l = tl.zeros([BLOCK_H], dtype=tl.float32)

    acc0 = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)
    acc1 = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)
    acc2 = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)
    acc3 = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)
    seen = tl.zeros([1], dtype=tl.int32)

    seqused = tl.load(SEQUSED + pid_t)

    if HAS_ORI:
        # Original-KV slots: leading entries before the first -1, capped by the
        # declared width and seqused_kv.  The running ``stop`` is recomputed per
        # chunk so a -1 can never resurrect a later physical slot.
        stop = tl.minimum(seqused, ORI_W)
        for start in range(0, ORI_W, BLOCK_N):
            cols = start + tl.arange(0, BLOCK_N)
            slot = tl.load(
                ORI_IDX + pid_t * ORI_W + cols, mask=cols < ORI_W, other=-1
            )
            first_neg = tl.min(tl.where(slot < 0, cols, ORI_W))
            stop = tl.minimum(stop, first_neg)

            valid = cols < stop
            page = slot // ORI_ROWS
            valid = valid & (slot >= 0) & (page < ORI_PAGES)
            off = page.to(tl.int64) * ORI_S0 + (
                slot - page * ORI_ROWS
            ).to(tl.int64) * ORI_S1

            seen += tl.sum(valid.to(tl.int32))
            m, l, acc0, acc1, acc2, acc3 = _attend_chunk(
                ORI_KV, off, valid, q0, q1, q2, q3, m, l,
                acc0, acc1, acc2, acc3, SCALE, BLOCK_D, BLOCK_N,
            )

    if HAS_CMP:
        prefix = tl.load(PREFIX + pid_t)
        history_limit = seqused // RATIO
        if USE_C4:
            # The C4 top-k array is filtered by its *logical value* (a padded
            # -1 or a value at/after the history limit is skipped), so every
            # column is scanned.
            loop_end = CMP_K
        else:
            # C128 walks the full compressed history; the top-k array is absent.
            loop_end = history_limit

        for start in range(0, loop_end, BLOCK_N):
            cols = start + tl.arange(0, BLOCK_N)
            range_ok = cols < loop_end
            if USE_C4:
                logical = tl.load(
                    CMP_IDX + pid_t * CMP_K + cols, mask=range_ok, other=-1
                )
                idx_ok = range_ok & (logical >= 0) & (logical < history_limit)
            else:
                logical = cols
                idx_ok = range_ok

            paged_end = prefix // RATIO

            # Paged compressed history: logical -> (page, row) via the table.
            is_paged = idx_ok & (logical >= 0) & (logical < paged_end)
            page_col = logical // CMP_ROWS
            page = tl.load(
                CMP_TABLE + pid_t * CMP_TABLE_S0 + page_col,
                mask=is_paged & (page_col < CMP_TABLE_W),
                other=-1,
            )
            paged_valid = is_paged & (page >= 0) & (page < CMP_PAGES)
            paged_off = page.to(tl.int64) * CMP_S0 + (
                logical - page_col * CMP_ROWS
            ).to(tl.int64) * CMP_S1
            seen += tl.sum(paged_valid.to(tl.int32))
            m, l, acc0, acc1, acc2, acc3 = _attend_chunk(
                CMP_KV, paged_off, paged_valid, q0, q1, q2, q3, m, l,
                acc0, acc1, acc2, acc3, SCALE, BLOCK_D, BLOCK_N,
            )

            # Tree node pool: step = (k + 1) * ratio - 1 - prefix, root-first.
            is_tree = idx_ok & (logical >= 0) & (logical >= paged_end)
            step = (logical + 1) * RATIO - 1 - prefix
            step_ok = is_tree & (step >= 0) & (step < MAX_DEPTH)
            node = tl.load(
                ANCESTORS + pid_t * MAX_DEPTH + step, mask=step_ok, other=-1
            )
            tree_valid = step_ok & (node >= 0) & (node < NODE_COUNT)
            tree_off = node * NODE_D
            seen += tl.sum(tree_valid.to(tl.int32))
            m, l, acc0, acc1, acc2, acc3 = _attend_chunk(
                TREE_KV, tree_off, tree_valid, q0, q1, q2, q3, m, l,
                acc0, acc1, acc2, acc3, SCALE, BLOCK_D, BLOCK_N,
            )

    # Branch-free empty-row handling: an `if` here also splits the MIX
    # sub-block after the dot chain, which the Ascend binder rejects.
    has_seen = tl.sum(seen) > 0
    inv_l = tl.where(has_seen, 1.0 / l, 0.0)
    out_base = OUT + pid_t * (N1 * NODE_D) + heads[:, None] * NODE_D
    result = acc0 * inv_l[:, None]
    tl.store(
        out_base + (0 * BLOCK_D + d)[None, :],
        result.to(OUT.dtype.element_ty),
        mask=head_mask[:, None],
    )
    result = acc1 * inv_l[:, None]
    tl.store(
        out_base + (1 * BLOCK_D + d)[None, :],
        result.to(OUT.dtype.element_ty),
        mask=head_mask[:, None],
    )
    result = acc2 * inv_l[:, None]
    tl.store(
        out_base + (2 * BLOCK_D + d)[None, :],
        result.to(OUT.dtype.element_ty),
        mask=head_mask[:, None],
    )
    result = acc3 * inv_l[:, None]
    tl.store(
        out_base + (3 * BLOCK_D + d)[None, :],
        result.to(OUT.dtype.element_ty),
        mask=head_mask[:, None],
    )

    if RETURN_LSE:
        lse = tl.where(has_seen, tl.log(l) + m, 0.0)
        tl.store(LSE + pid_t * N1 + heads, lse, mask=head_mask)


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
