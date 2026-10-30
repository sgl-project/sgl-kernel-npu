"""DSV4 DSpark tree lightning indexer (Triton port).

Reproduces the AscendC ``quant_lightning_indexer_tree`` operator
(``vllm-ascend@8947f1f9f``), whose semantics are pinned in
``docs/tree_ops_semantics.md`` §4 and summarised in ``tree/README.md``::

    quant_lightning_indexer_tree(
        query, history, weights, query_scale, history_scale,
        nodes, node_scale, ancestors, prefixes, requests,
        table, cu, lengths, residual, metadata, topk
    ) -> int32[n, 1, topk]

Score of logical compressed slot ``k`` of query row ``row`` (§4.3)::

    QK[g] = int8_dot(query[row, g, :], key)            # exact int32
    S[g]  = fp16_rne(relu(QK[g]) * 2**-10)             # reluPre=1, 0x3a800000
    Wq[g] = fp16_rne(weights[row, g] * query_scale[row, g])
    score = sum_g float(S[g]) * float(Wq[g]) * float(k_scale)   # FP32 reduce

Slot resolution (§1.3, ``tree_index.h``)::

    k <  prefixes[row] // 4  -> paged history: page = table[requests[row], k // block]
    else -> step = (k + 1) * 4 - 1 - prefixes[row]
            node = ancestors[row, step]
    out-of-range step / node (< 0 or >= num_nodes) / page -> score -inf, index -1

Top-k is the ``(score desc, index asc)`` prefix of the valid slots with ``-1``
padding.  Every logical slot below ``lengths[row]`` is a candidate: the
AscendC kernel scans **all** S2 tiles and merges each tile's top-2048 into a
2048-wide (``BASE_TOPK == 2048``) accumulator, so 2048 is the output
accumulator width, *not* a candidate cap.  D3 makes the emission order
non-contractual, so this port returns the same **set** the vendor returns but
emits it in ascending index order (the vendor emits it score-descending);
consumers treat the row as a set.  The reference oracle sorts every valid slot.

Three kernels (T5b, vendor-shaped):

* :func:`_gather_kernel` -- one program per ``(row, 128-slot block)``; resolves
  paged-history vs. tree-ancestor indirection once and materialises a
  contiguous ``[SLOTS, HEAD_DIM]`` int8 key matrix plus per-slot scale and
  validity.  This removes the ``64 heads * 16 D-chunks`` repeated gather that
  dominated the old fused fallback.
* :func:`_score_kernel` -- reads the contiguous key matrix and reproduces the
  vendor's two-cube chain: QK as an int8 x int8 ``tl.dot`` -> int32, fixpipe
  equivalent ``fp16(relu(QK) * 2**-10)``, ``Wq = fp16(weights * query_scale)``,
  then a second FP16 ``tl.dot`` that reduces the ``g`` (head) axis with 16
  replicated ``M`` rows, and finally ``* key_scale`` in FP32.
  ``service_cube_arch22.h:ComputeQk/FixpSToL1/ComputeWs/FixpResToGm`` is the
  reference.  An exact int32-widening path without any cube is available as an
  opt-in fallback (see R8).
* :func:`_select_topk_kernel` -- one program per row, streaming the score
  buffer in ``BASE_TOPK``-sized S2 tiles (the vendor's ``S2_BASE_SIZE`` /
  accumulator width).  It finds the k-th largest score by integer bisection on a
  monotone int32 key (exact, ties included, with only one 2048-slot tile of UB
  live at a time), then compacts every selected slot in ascending index order
  and fills the tail with ``-1``.  This replaces both the unbounded insertion
  TopK and the vendor's ``Sort32``/``MrgSort`` + 2-way merge: the result set is
  identical, the O(topk * slots) insertion cost and the unavailable hardware
  sort are both avoided.


``metadata`` is validated (int32, 1024 entries, contiguous) and then ignored
(D2); ``residual`` is ABI-only and unused; ``cu`` must equal ``arange(n + 1)``.

Risk R8 (int8 ``tl.dot``): the Ascend cube executes int8 x int8 -> int32
natively, which is exactly what the vendor kernel does, so the cube path is
the **default**.  Some triton-ascend builds cannot bind the resulting MIX/CV
sub-block for this shape (or reject ``M == 16`` FP16 dots); on such a build set
``SGL_TREE_INDEXER_INT8_DOT=0`` to switch to the exact int32-widening path
(``tl.sum`` over the head dim, no cube).  That fallback is correctness-only:
it is the pre-T5b scalar path and carries no performance claim.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from ._common import (
    TreeContractError,
    check_cu_arange,
    check_page_table,
    check_same_device,
    check_tensor,
    check_tree_metadata,
    next_power_of_2,
)

__all__ = ["quant_lightning_indexer_tree", "TreeContractError"]

# Indexer contract (docs/tree_ops_semantics.md §4.2): INT8 Q/K, 64 query heads,
# head_dim 128, C4 only, TND queries, paged history.
_QUERY_HEADS = 64
_HEAD_DIM = 128
_RATIO = 4

# NOTE: the QK dequant weight 2**-10 (SetFixpipePreQuantFlag(0x3a800000)) is
# written as the literal 0.0009765625 inside the kernels: Triton folds Python
# literals at compile time, so no module global is resolved in device code.

# Vendor accumulator width (BASE_TOPK = 2048) from
# ``service_vector_arch22.h:35``.  It is also the vendor's S2 tile
# (``S2_BASE_SIZE`` in the op_host tiler): every S2 tile is scored and reduced
# into this 2048-wide accumulator.  It is an *output width*, never a candidate
# cap -- the selection kernel below streams the full slot range in tiles of
# this width.
_BASE_TOPK = 2048
_TOP_K_TILE = _BASE_TOPK
_MAX_TOPK = 2048
_BLOCK_KV = 128

# Head tiles: wide on the cube path (one int8 QK dot per 32 heads, one FP16
# Ws dot per 32 heads), narrow on the manual path so the
# [HEAD_BLOCK, DOT_D_CHUNK, BLOCK_KV] int32 widening tile stays inside UB.
_HEAD_BLOCK_DOT = 32
_HEAD_BLOCK_MANUAL = 16
_DOT_D_CHUNK = 8

# The vendor reduces the ``g`` (query-head) axis with a cube whose ``M`` is a
# single BLOCK_CUBE (16) of replicated rows and takes row 0
# (``service_cube_arch22.h:ComputeWs`` + ``FixpResToGm`` mSize=1).  ``tl.dot``
# has the same M >= 16 floor, so the Ws dot uses 16 replicated rows.
_WS_DOT_M = 16

# R8: native int8 tl.dot -> int32 is what the vendor does, so the cube path is
# the default.  Set SGL_TREE_INDEXER_INT8_DOT=0 to opt into the exact
# int32-widening fallback on a triton-ascend build where the int8 dot (or the
# M=16 FP16 Ws dot) does not compile.  The fallback is correctness-only.
_USE_INT8_DOT = os.environ.get("SGL_TREE_INDEXER_INT8_DOT", "1") == "1"

_NUM_WARPS = 1
_NUM_STAGES = 1


@triton.jit(
    do_not_specialize=[
        "history_page_stride",
        "history_scale_page_stride",
        "num_pages",
        "table_rows",
        "table_width",
        "ancestor_width",
        "num_nodes",
        "block_size",
    ]
)
def _gather_kernel(
    history_ptr,  # [pages, block, 1, HEAD_DIM] int8
    history_scale_ptr,  # [pages, block, 1, 1] fp16
    nodes_ptr,  # [num_nodes, HEAD_DIM] int8
    node_scale_ptr,  # [num_nodes] fp16
    ancestors_ptr,  # [n, ancestor_width] int64
    prefixes_ptr,  # [n] int32
    requests_ptr,  # [n] int64
    table_ptr,  # [table_rows, table_width] int32
    lengths_ptr,  # [n] int32
    keys_ptr,  # [n, SLOTS, HEAD_DIM] int8 (out)
    key_scales_ptr,  # [n, SLOTS] fp32 (out)
    valid_ptr,  # [n, SLOTS] int8 (out)
    # runtime geometry
    history_page_stride,
    history_scale_page_stride,
    num_pages,
    table_rows,
    table_width,
    ancestor_width,
    num_nodes,
    block_size,
    # constexpr meta
    HEAD_DIM: tl.constexpr,
    RATIO: tl.constexpr,
    SLOTS: tl.constexpr,
    BLOCK_KV: tl.constexpr,
):
    """Resolve every logical slot once and materialise contiguous key rows.

    The previous fused score kernel re-resolved and re-gathered the same key
    row once per head and D chunk (``64 * 16`` times per slot block).  This
    kernel pays the paged/tree indirection exactly once; the score kernel then
    reads a plain ``[n, SLOTS, HEAD_DIM]`` int8 matrix.
    """
    pid_n = tl.program_id(0)
    pid_kv = tl.program_id(1)

    off_k = pid_kv * BLOCK_KV + tl.arange(0, BLOCK_KV)
    out_off = pid_n * SLOTS + off_k
    limit = tl.load(lengths_ptr + pid_n).to(tl.int32)

    prefix = tl.load(prefixes_ptr + pid_n).to(tl.int32)
    request = tl.load(requests_ptr + pid_n).to(tl.int64)

    # ---- slot -> key source (tree_index.h: paged history vs. tree ancestor) --
    is_history = off_k < prefix // RATIO
    col = off_k // block_size
    row_in_page = off_k - col * block_size
    request_ok = (request >= 0) & (request < table_rows)
    col_ok = is_history & (col >= 0) & (col < table_width)
    # Ascend still forms masked-load addresses.  Clamp the request/column even
    # when their masks are false, otherwise padding slots form negative or
    # out-of-allocation table addresses.
    request_safe = tl.where(request_ok, request, 0)
    col_safe = tl.where(col_ok, col, 0)
    page = tl.load(
        table_ptr + request_safe * table_width + col_safe,
        mask=col_ok & request_ok,
        other=-1,
    ).to(tl.int32)
    history_valid = col_ok & request_ok & (page >= 0) & (page < num_pages)

    step = (off_k + 1) * RATIO - 1 - prefix
    step_ok = (~is_history) & (step >= 0) & (step < ancestor_width)
    step_safe = tl.where(step_ok, step, 0)
    node = tl.load(
        ancestors_ptr + pid_n * ancestor_width + step_safe,
        mask=step_ok,
        other=-1,
    ).to(tl.int32)
    node_valid = step_ok & (node >= 0) & (node < num_nodes)

    valid = (off_k < limit) & (history_valid | node_valid)

    page_safe = tl.where(history_valid, page, 0)
    node_safe = tl.where(node_valid, node, 0)
    history_base = page_safe.to(tl.int64) * history_page_stride + row_in_page.to(
        tl.int64
    ) * HEAD_DIM
    node_base = node_safe.to(tl.int64) * HEAD_DIM

    history_scale = tl.load(
        history_scale_ptr
        + page_safe.to(tl.int64) * history_scale_page_stride
        + row_in_page,
        mask=history_valid,
        other=0.0,
    ).to(tl.float32)
    node_scale = tl.load(
        node_scale_ptr + node_safe, mask=node_valid, other=0.0
    ).to(tl.float32)
    key_scale = tl.where(history_valid, history_scale, node_scale)

    tl.store(key_scales_ptr + out_off, key_scale)
    tl.store(valid_ptr + out_off, valid.to(tl.int8))

    # One contiguous [BLOCK_KV, HEAD_DIM] key tile: 16 KiB for 128 x 128 int8.
    # Keeping the tile int8 avoids the int32 widening temporary that previously
    # made hivm-plan-memory fail for a [head, dim, slot] layout.
    off_d = tl.arange(0, HEAD_DIM)
    history_off = history_base[:, None] + off_d[None, :]
    node_off = node_base[:, None] + off_d[None, :]
    from_history = tl.load(
        history_ptr + history_off, mask=history_valid[:, None], other=0
    )
    from_nodes = tl.load(nodes_ptr + node_off, mask=node_valid[:, None], other=0)
    key_tile = tl.where(history_valid[:, None], from_history, from_nodes)
    tl.store(
        keys_ptr + out_off[:, None] * HEAD_DIM + off_d[None, :],
        key_tile,
    )


@triton.jit
def _score_key(scores):
    """Map an fp32 score to an int32 whose *signed* order is the score order.

    ``bits >> 31`` is the arithmetic shift, so the mask is ``0x7fffffff`` for a
    negative score and ``0`` otherwise: negatives get their low 31 bits
    inverted (which reverses sign-magnitude into two's complement order) while
    positives keep their raw IEEE bits, which already ascend with the value.
    The result is a strict order isomorphism on finite fp32, so equal scores
    always produce equal keys.

    ``-inf`` (``0xff800000``) maps to ``0x807fffff`` -- smaller than every
    finite score's key -- which is what makes an unresolved slot sort after
    every real candidate.  ``-0.0`` would map to ``-1`` (below ``+0.0``) even
    though it compares equal, so it is folded to ``+0.0`` first; equal scores
    therefore always share a key.
    """
    bits = scores.to(tl.int32, bitcast=True)
    bits = tl.where(bits == -2147483648, 0, bits)  # -0.0 -> +0.0
    return bits ^ ((bits >> 31) & 2147483647)


@triton.jit
def _score_kernel(
    query_ptr,  # [n, NUM_HEADS, HEAD_DIM] int8
    weights_ptr,  # [n, NUM_HEADS] fp16
    query_scale_ptr,  # [n, NUM_HEADS] fp16
    keys_ptr,  # [n, SLOTS, HEAD_DIM] int8
    key_scales_ptr,  # [n, SLOTS] fp32
    valid_ptr,  # [n, SLOTS] int8
    scores_ptr,  # [n, SLOTS] fp32 (out)
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SLOTS: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
    DOT_D_CHUNK: tl.constexpr,
    WS_DOT_M: tl.constexpr,
    USE_INT8_DOT: tl.constexpr,
):
    """Score ``BLOCK_KV`` contiguous key rows of one query row."""
    pid_n = tl.program_id(0)
    pid_kv = tl.program_id(1)

    off_k = pid_kv * BLOCK_KV + tl.arange(0, BLOCK_KV)
    out_off = pid_n * SLOTS + off_k
    key_base = out_off * HEAD_DIM
    off_d = tl.arange(0, HEAD_DIM)
    off_g = tl.arange(0, HEAD_BLOCK)
    query_base = query_ptr + pid_n * (NUM_HEADS * HEAD_DIM)
    weight_base = weights_ptr + pid_n * NUM_HEADS
    scale_base = query_scale_ptr + pid_n * NUM_HEADS

    acc = tl.zeros((BLOCK_KV,), dtype=tl.float32)
    if USE_INT8_DOT:
        # Vendor cube chain (service_cube_arch22.h):
        #   ComputeQk  int8 x int8 -> int32   (K = HEAD_DIM)
        #   FixpSToL1  reluPre=1 + pre-quant 0x3a800000 -> fp16(relu(QK)*2**-10)
        #   ComputeWs  FP16 cube, A = Wq replicated over WS_DOT_M rows
        #   FixpResToGm NoQuant -> fp32, mSize = 1 (row 0 only)
        # B operand is the contiguous [HEAD_DIM, BLOCK_KV] int8 key tile.
        key_tile = tl.load(keys_ptr + key_base[None, :] + off_d[:, None])
        for head_block in tl.range(0, NUM_HEADS // HEAD_BLOCK):
            head = head_block * HEAD_BLOCK + off_g
            query = tl.load(
                query_base + head[:, None] * HEAD_DIM + off_d[None, :]
            )
            qk = tl.dot(query, key_tile, out_dtype=tl.int32)
            # fp16 on purpose: it is the L0B operand of the Ws cube, exactly
            # like the fixpipe output the vendor reloads through sL1_.
            score0 = (tl.maximum(qk, 0).to(tl.float32) * 0.0009765625).to(
                tl.float16
            )
            # ProcessVec0 in the vendor: Wq = fp16(weights * query_scale).
            weight = tl.load(weight_base + head)
            query_scale = tl.load(scale_base + head)
            wq = (weight * query_scale).to(tl.float16)
            # Every replicated L0A row is the same vector, so the WS_DOT_M
            # output rows are bit-identical and take row 0 = the FP32 g-reduce.
            wq_rows = tl.broadcast_to(wq[None, :], (WS_DOT_M, HEAD_BLOCK))
            acc += tl.max(tl.dot(wq_rows, score0), axis=0)
    else:
        # R8 fallback (opt-in, correctness only): exact int32 widening.  The
        # key tile is already contiguous in [SLOTS, HEAD_DIM], so each chunk is
        # a single coalesced load instead of a paged/tree gather.  Keep qk 1-D:
        # a 3-D [head, dim, slot] temporary still makes hivm-plan-memory fail.
        for head in tl.range(0, NUM_HEADS):
            qk = tl.zeros((BLOCK_KV,), dtype=tl.int32)
            for chunk in tl.range(0, HEAD_DIM // DOT_D_CHUNK):
                cols = chunk * DOT_D_CHUNK + tl.arange(0, DOT_D_CHUNK)
                query = tl.load(query_base + head * HEAD_DIM + cols)
                key_chunk = tl.load(keys_ptr + key_base[:, None] + cols[None, :])
                qk += tl.sum(
                    query.to(tl.int32)[None, :] * key_chunk.to(tl.int32),
                    axis=1,
                )
            score0 = tl.maximum(qk, 0).to(tl.float32) * 0.0009765625
            score0 = score0.to(tl.float16).to(tl.float32)
            weight = tl.load(weight_base + head)
            query_scale = tl.load(scale_base + head)
            wq = (weight * query_scale).to(tl.float16).to(tl.float32)
            acc += score0 * wq

    valid = tl.load(valid_ptr + out_off) != 0
    key_scale = tl.load(key_scales_ptr + out_off)
    # + 0.0 turns a -0.0 score (relu -> 0 with a negative key scale) into
    # +0.0, so scores that compare equal also share a `_score_key`.  -inf and
    # every other value are unchanged.
    score = tl.where(valid, acc * key_scale, float("-inf")) + 0.0
    tl.store(scores_ptr + out_off, score)


@triton.jit(do_not_specialize=["topk"])
def _select_topk_kernel(
    scores_ptr,  # [n, SLOTS] fp32, -inf marks an absent slot
    out_ptr,  # [n, 1, topk] int32 (out)
    topk,
    SLOTS: tl.constexpr,
    TILE: tl.constexpr,  # per-S2-tile window: BASE_TOPK, or SLOTS if shorter
    FILL: tl.constexpr,  # _MAX_TOPK: -1 prefill width (>= topk)
    BISECT_STEPS: tl.constexpr,  # ceil(log2(int32 domain)) == 32
):
    """Exact top-k of one row's scores, emitted in ascending index order.

    Replaces both the unbounded insertion TopK (``topk * slots`` argmax passes)
    and the vendor's per-S2-tile ``Sort32``/``MrgSort`` + 2-way merge, which has
    no Triton equivalent.  The result *set* is identical to a stable
    ``argsort(scores, descending=True)`` prefix; only the emission order
    differs, and D3 makes that order non-contractual.

    Structure, three streaming phases over ``TILE``-wide S2 tiles (so UB never
    holds more than one 2048-slot tile):

    1. key range + resolvable-slot count (1 scan);
    2. bisection for the k-th largest key ``lo`` (``BISECT_STEPS`` scans) --
       exact for ties because it searches integer keys, not rounded floats;
    3. count of slots strictly above ``lo`` (1 scan), then compaction of
       ``{key > lo}`` plus the first ``rank - n_gt`` exact ties in ascending
       index order (1 scan), with ``-1`` padding for the tail.
    """
    pid_n = tl.program_id(0)
    off = tl.arange(0, TILE)
    row = scores_ptr + pid_n * SLOTS
    out_row = out_ptr + pid_n * topk
    n_tiles: tl.constexpr = SLOTS // TILE
    # Rank-0 int32 zero used to seed the carried scalars (a plain
    # ``tl.zeros(())`` is not accepted on every triton-ascend build).
    zero = tl.sum(tl.full((1,), 0, tl.int32), axis=0)

    # ---- phase 1: key range and the number of resolvable slots -----------
    tile_scores = tl.load(row + off)
    tile_keys = _score_key(tile_scores)
    key_lo = tl.min(tile_keys, axis=0)
    key_hi = tl.max(tile_keys, axis=0)
    n_valid = tl.sum((tile_scores > float("-inf")).to(tl.int32))
    for t in tl.range(1, n_tiles):
        tile_scores = tl.load(row + t * TILE + off)
        tile_keys = _score_key(tile_scores)
        key_lo = tl.minimum(key_lo, tl.min(tile_keys, axis=0))
        key_hi = tl.maximum(key_hi, tl.max(tile_keys, axis=0))
        n_valid += tl.sum((tile_scores > float("-inf")).to(tl.int32))

    # rank = the k we can actually fill; a row with fewer valid slots than

    # `topk` converges on its own smallest key and pads the rest with -1.
    rank = tl.minimum(n_valid, topk.to(tl.int32))

    # ---- phase 2: k-th largest key by integer bisection ------------------
    # `lo` is always a feasible threshold (count_ge(lo) >= rank), `hi` an upper
    # bound, so the loop converges on the exact k-th largest key.  ceil-mid
    # guarantees lo advances; (lo & hi) + ((lo ^ hi) >> 1) is overflow-free.
    lo = key_lo
    hi = key_hi
    for _ in tl.range(0, BISECT_STEPS):
        mid = (lo & hi) + ((lo ^ hi) >> 1)
        mid = mid + (lo < hi).to(tl.int32)
        count_ge = zero
        for t in tl.range(0, n_tiles):
            tile_keys = _score_key(tl.load(row + t * TILE + off))
            count_ge += tl.sum((tile_keys >= mid).to(tl.int32))
        feasible = count_ge >= rank
        lo = tl.where(feasible, mid, lo)
        hi = tl.where(feasible, hi, mid - 1)

    # ---- phase 3a: how many slots strictly beat the k-th largest ---------
    n_gt = zero
    for t in tl.range(0, n_tiles):
        tile_keys = _score_key(tl.load(row + t * TILE + off))
        n_gt += tl.sum((tile_keys > lo).to(tl.int32))
    # Exact ties still fit: the lowest-index ones fill the remaining ranks,
    # which is the reference's stable (score desc, index asc) tie-break.
    need = rank - n_gt

    # ---- phase 3b: compact, ascending index order, -1 padded ------------
    fill = tl.arange(0, FILL)
    tl.store(out_row + fill, tl.full((FILL,), -1, tl.int32), mask=fill < topk)
    pos = zero
    eq_seen = zero
    for t in tl.range(0, n_tiles):
        tile_keys = _score_key(tl.load(row + t * TILE + off))
        gt = tile_keys > lo
        eq = tile_keys == lo
        eq_i = eq.to(tl.int32)
        eq_before = tl.cumsum(eq_i, axis=0) - eq_i
        selected = gt | (eq & (eq_before + eq_seen < need))
        selected_i = selected.to(tl.int32)
        dst = tl.cumsum(selected_i, axis=0) - selected_i
        tl.store(out_row + pos + dst, off + t * TILE, mask=selected)
        pos += tl.sum(selected_i)
        eq_seen += tl.sum(eq_i)


def _select_all(lengths: torch.Tensor, topk: int) -> torch.Tensor:
    """Chronological ``arange(topk)`` masked by ``lengths`` (README shortcut)."""
    n = int(lengths.numel())
    positions = torch.arange(topk, dtype=torch.int32, device=lengths.device).expand(
        n, topk
    )
    keep = positions.to(torch.int64) < lengths.to(torch.int64).unsqueeze(1)
    return torch.where(keep, positions, torch.full_like(positions, -1)).unsqueeze(1)


@torch.no_grad()
def quant_lightning_indexer_tree(
    query: torch.Tensor,
    history: torch.Tensor,
    weights: torch.Tensor,
    query_scale: torch.Tensor,
    history_scale: torch.Tensor,
    nodes: torch.Tensor,
    node_scale: torch.Tensor,
    ancestors: torch.Tensor,
    prefixes: torch.Tensor,
    requests: torch.Tensor,
    table: torch.Tensor,
    cu: torch.Tensor,
    lengths: torch.Tensor,
    residual: torch.Tensor,
    metadata: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    """Return the TopK compressed-KV indices for every tree node, ``[n, 1, topk]``.

    Contract, dtypes, strides and the score formula are specified in
    ``docs/tree_ops_semantics.md`` §4.  ``metadata`` is validated and ignored
    (D2) and ``residual`` is ABI-only.  ``sparse_values`` is not produced
    (``return_value == 0`` in the vendor binding).
    """
    check_tensor("query", query, dtype=torch.int8, rank=3, contiguous=True)
    n = int(query.shape[0])
    if n <= 0:
        raise TreeContractError(f"query: expected at least one row, got {n}")
    if int(query.shape[1]) != _QUERY_HEADS or int(query.shape[2]) != _HEAD_DIM:
        raise TreeContractError(
            f"query: expected [{n}, {_QUERY_HEADS}, {_HEAD_DIM}], "
            f"got {tuple(query.shape)}"
        )

    check_tensor("history", history, dtype=torch.int8, rank=4)
    if int(history.shape[2]) != 1 or int(history.shape[3]) != _HEAD_DIM:
        raise TreeContractError(
            f"history: expected [pages, block, 1, {_HEAD_DIM}], "
            f"got {tuple(history.shape)}"
        )
    if history.stride(1) != _HEAD_DIM or history.stride(3) != 1:
        raise TreeContractError(
            "history: expected stride(1) == 128 and stride(3) == 1 "
            f"(padded page stride allowed), got {history.stride()}"
        )
    if int(history.shape[0]) <= 0 or int(history.shape[1]) <= 0:
        raise TreeContractError(
            f"history: expected non-empty [pages, block], got {tuple(history.shape)}"
        )

    check_tensor("history_scale", history_scale, dtype=torch.float16, rank=4)
    scale_shape = tuple(history.shape[:3]) + (1,)
    if tuple(history_scale.shape) != scale_shape:
        raise TreeContractError(
            f"history_scale: expected {scale_shape}, "
            f"got {tuple(history_scale.shape)}"
        )
    if history_scale.stride(1) != 1:
        raise TreeContractError(
            f"history_scale: expected stride(1) == 1, got {history_scale.stride()}"
        )

    for name, tensor in (("weights", weights), ("query_scale", query_scale)):
        check_tensor(name, tensor, dtype=torch.float16, rank=2, contiguous=True)
        if tuple(tensor.shape) != (n, _QUERY_HEADS):
            raise TreeContractError(
                f"{name}: expected [{n}, {_QUERY_HEADS}], got {tuple(tensor.shape)}"
            )

    check_tensor("nodes", nodes, dtype=torch.int8, rank=2, contiguous=True)
    if int(nodes.shape[1]) != _HEAD_DIM:
        raise TreeContractError(
            f"nodes: expected [num_nodes, {_HEAD_DIM}], got {tuple(nodes.shape)}"
        )
    num_nodes = int(nodes.shape[0])
    if num_nodes < n:
        raise TreeContractError(
            f"nodes: the key pool must cover every query row, got {num_nodes} < {n}"
        )
    check_tensor("node_scale", node_scale, dtype=torch.float16, rank=1, contiguous=True)
    if int(node_scale.numel()) != num_nodes:
        raise TreeContractError(
            f"node_scale: expected {num_nodes} entries, got {node_scale.numel()}"
        )

    check_tensor("ancestors", ancestors, dtype=torch.int64, rank=2, contiguous=True)
    if int(ancestors.shape[0]) != n:
        raise TreeContractError(
            f"ancestors: expected {n} rows, got {ancestors.shape[0]}"
        )
    if int(ancestors.shape[1]) <= 0:
        raise TreeContractError("ancestors: ancestor width must be positive")

    check_tensor("prefixes", prefixes, dtype=torch.int32, rank=1, contiguous=True)
    check_tensor("requests", requests, dtype=torch.int64, rank=1, contiguous=True)
    for name, tensor in (("prefixes", prefixes), ("requests", requests)):
        if int(tensor.numel()) != n:
            raise TreeContractError(
                f"{name}: expected {n} entries, got {tensor.numel()}"
            )

    check_page_table("table", table)
    check_cu_arange("cu", cu, n)

    for name, tensor in (("lengths", lengths), ("residual", residual)):
        check_tensor(name, tensor, dtype=torch.int32, rank=1, contiguous=True)
        if int(tensor.numel()) != n:
            raise TreeContractError(
                f"{name}: expected {n} entries, got {tensor.numel()}"
            )
    check_tree_metadata("metadata", metadata, required=True)

    topk = int(topk)
    if not 1 <= topk <= _MAX_TOPK:
        raise TreeContractError(f"topk: expected 1..{_MAX_TOPK}, got {topk}")

    check_same_device(
        "quant_lightning_indexer_tree",
        [
            query,
            history,
            history_scale,
            weights,
            query_scale,
            nodes,
            node_scale,
            ancestors,
            prefixes,
            requests,
            table,
            cu,
            lengths,
            residual,
            metadata,
        ],
    )

    # Tiling (host side, mirrors the vendor tiler's max_kv): the compressed
    # history fits in the output -> chronological indices, no kernel at all.
    max_len = int(lengths.max().item())
    max_kv = max_len * _RATIO
    if max(1, max_kv // _RATIO) <= topk:
        return _select_all(lengths, topk)

    # T5a/T5b: every logical slot is a candidate.  The vendor's 2048-wide
    # accumulator (``_BASE_TOPK``) is an output width, not a candidate cap --
    # the vendor scores all S2 tiles and reduces per tile into that width -- so
    # size the full-slot score buffer to the observed maximum length, rounded up
    # to a power of two and kept a multiple of the 128-slot per-program tile.
    slots = next_power_of_2(max(1, max_len))
    slots = max(slots, _BLOCK_KV)

    # Selection window: the vendor's S2 tile / BASE_TOPK width, shrunk only for
    # a row shorter than one tile (both operands are powers of two, so
    # ``slots % select_tile == 0`` always holds).
    select_tile = min(_TOP_K_TILE, slots)

    use_int8_dot = _USE_INT8_DOT
    head_block = _HEAD_BLOCK_DOT if use_int8_dot else _HEAD_BLOCK_MANUAL

    # Per-row contiguous key matrix.  n is the draft-tree row count (typically
    # <= 64), so this is a few MiB; the previous fused kernel traded that
    # memory for 1024 strided gathers per slot block and was ~20x slower.
    keys = torch.empty(
        (n, slots, _HEAD_DIM), dtype=torch.int8, device=query.device
    )
    key_scales = torch.empty(
        (n, slots), dtype=torch.float32, device=query.device
    )
    valid = torch.empty((n, slots), dtype=torch.int8, device=query.device)
    scores = torch.empty((n, slots), dtype=torch.float32, device=query.device)

    _gather_kernel[(n, slots // _BLOCK_KV)](
        history,
        history_scale,
        nodes,
        node_scale,
        ancestors,
        prefixes,
        requests,
        table,
        lengths,
        keys,
        key_scales,
        valid,
        history.stride(0),
        history_scale.stride(0),
        int(history.shape[0]),
        int(table.shape[0]),
        int(table.shape[1]),
        int(ancestors.shape[1]),
        num_nodes,
        int(history.shape[1]),
        HEAD_DIM=_HEAD_DIM,
        RATIO=_RATIO,
        SLOTS=slots,
        BLOCK_KV=_BLOCK_KV,
        num_warps=_NUM_WARPS,
        num_stages=_NUM_STAGES,
    )

    _score_kernel[(n, slots // _BLOCK_KV)](
        query,
        weights,
        query_scale,
        keys,
        key_scales,
        valid,
        scores,
        NUM_HEADS=_QUERY_HEADS,
        HEAD_DIM=_HEAD_DIM,
        SLOTS=slots,
        BLOCK_KV=_BLOCK_KV,
        HEAD_BLOCK=head_block,
        DOT_D_CHUNK=_DOT_D_CHUNK,
        WS_DOT_M=_WS_DOT_M,
        USE_INT8_DOT=use_int8_dot,
        num_warps=_NUM_WARPS,
        num_stages=_NUM_STAGES,
    )

    out = torch.empty((n, 1, topk), dtype=torch.int32, device=query.device)
    _select_topk_kernel[(n,)](
        scores,
        out,
        topk,
        SLOTS=slots,
        TILE=select_tile,
        FILL=_MAX_TOPK,
        BISECT_STEPS=32,
        num_warps=_NUM_WARPS,
        num_stages=_NUM_STAGES,
    )
    return out
