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

Top-k is ``(score desc, index asc)`` (D3: tie order is not contractual) with
``-1`` padding, over the first ``min(2048, n_slots)`` logical slots — the
AscendC kernel's ``BASE_TOPK == 2048`` accumulator width.  Callers whose
``lengths`` exceed 2048 therefore see the vendor behaviour, not a full sort of
the tail; the reference oracle sorts every valid slot, so tests keep
``lengths <= 2048``.

Two kernels:

* :func:`_score_kernel` — one program per ``(row, 128-slot block)``; gathers the
  key row for every slot (paged history *or* tree node), runs the int8 QK as
  ``tl.dot`` on the cube, then does relu / 2**-10 / FP16 rounding / FP16 weight
  product / FP32 head reduce / key-scale multiply and stores ``float32``.
* :func:`_topk_merge_kernel` — one program per row: a 2048-wide score
  accumulator reduced by stable insertion (``argmax`` with lowest-index
  tie-break), matching the merge pattern in
  ``indexer/flash_block_score_decode.py``.

``metadata`` is validated (int32, 1024 entries, contiguous) and then ignored
(D2); ``residual`` is ABI-only and unused; ``cu`` must equal ``arange(n + 1)``.

Risk R8 (int8 ``tl.dot``): the Ascend cube executes int8 x int8 -> int32
natively, which is what the vendor kernel does.  Some triton-ascend builds
cannot bind the resulting MIX/CV sub-block for this shape, so the default
path is the exact int32-widening fallback (``tl.sum`` over the head dim,
no cube).  Set ``SGL_TREE_INDEXER_INT8_DOT=1`` on a build where the int8
dot compiles to enable the faster cube path.
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

# Vendor accumulator width (BASE_TOPK = 2048) and the per-program slot tile.
_TOPK_SLOTS = 2048
_MAX_TOPK = 2048
_BLOCK_KV = 128

# Head tiles: wide on the cube path, narrow on the manual path so the
# [HEAD_BLOCK, DOT_D_CHUNK, BLOCK_KV] int32 widening tile stays inside UB.
_HEAD_BLOCK_DOT = 32
_HEAD_BLOCK_MANUAL = 16
_DOT_D_CHUNK = 8

# R8: native int8 tl.dot -> int32 is faster but some triton-ascend builds
# cannot bind the resulting MIX/CV sub-block for this shape.  Default to the
# exact int32-widening fallback; opt in on a build where int8 dot compiles.
_USE_INT8_DOT = os.environ.get("SGL_TREE_INDEXER_INT8_DOT", "0") == "1"

_NUM_WARPS = 1
_NUM_STAGES = 1


@triton.jit
def _gather_key_tile(
    history_ptr,
    nodes_ptr,
    history_base,
    node_base,
    history_valid,
    node_valid,
    off_d,
):
    """Gather one ``[len(off_d), slots]`` int8 key tile from either key source.

    ``history_base`` / ``node_base`` are per-slot element offsets; masked slots
    are zero-filled (their score is overwritten with ``-inf`` by the caller).
    """
    history_off = history_base[None, :] + off_d[:, None]
    node_off = node_base[None, :] + off_d[:, None]
    from_history = tl.load(
        history_ptr + history_off, mask=history_valid[None, :], other=0
    )
    from_nodes = tl.load(nodes_ptr + node_off, mask=node_valid[None, :], other=0)
    return tl.where(history_valid[None, :], from_history, from_nodes).to(tl.int8)


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
def _score_kernel(
    query_ptr,  # [n, NUM_HEADS, HEAD_DIM] int8
    history_ptr,  # [pages, block, 1, HEAD_DIM] int8
    history_scale_ptr,  # [pages, block, 1, 1] fp16
    weights_ptr,  # [n, NUM_HEADS] fp16
    query_scale_ptr,  # [n, NUM_HEADS] fp16
    nodes_ptr,  # [num_nodes, HEAD_DIM] int8
    node_scale_ptr,  # [num_nodes] fp16
    ancestors_ptr,  # [n, ancestor_width] int64
    prefixes_ptr,  # [n] int32
    requests_ptr,  # [n] int64
    table_ptr,  # [table_rows, table_width] int32
    lengths_ptr,  # [n] int32
    scores_ptr,  # [n, SLOTS] fp32 (out)
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
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    RATIO: tl.constexpr,
    SLOTS: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
    DOT_D_CHUNK: tl.constexpr,
    USE_INT8_DOT: tl.constexpr,
):
    """Score ``BLOCK_KV`` logical compressed slots of one query row."""
    pid_n = tl.program_id(0)
    pid_kv = tl.program_id(1)

    off_k = pid_kv * BLOCK_KV + tl.arange(0, BLOCK_KV)  # logical slot ids
    out_off = pid_n * SLOTS + off_k
    limit = tl.load(lengths_ptr + pid_n).to(tl.int32)

    # Do not early-return for a block past `limit`: on the MIX backend an
    # early return inside a program splits the AIC/AIV sub-block and
    # hivm-bind-sub-block rejects the kernel.  `valid = off_k < limit` below
    # makes every slot in that block a masked, -inf score instead.
    prefix = tl.load(prefixes_ptr + pid_n).to(tl.int32)
    request = tl.load(requests_ptr + pid_n).to(tl.int64)

    # ---- slot -> key source (tree_index.h: paged history vs. tree ancestor) --
    is_history = off_k < prefix // RATIO
    col = off_k // block_size
    row_in_page = off_k - col * block_size
    request_ok = (request >= 0) & (request < table_rows)
    col_ok = is_history & (col >= 0) & (col < table_width)
    page = tl.load(
        table_ptr + request * table_width + col,
        mask=col_ok & request_ok,
        other=-1,
    ).to(tl.int32)
    history_valid = col_ok & request_ok & (page >= 0) & (page < num_pages)

    step = (off_k + 1) * RATIO - 1 - prefix
    step_ok = (~is_history) & (step >= 0) & (step < ancestor_width)
    node = tl.load(
        ancestors_ptr + pid_n * ancestor_width + step, mask=step_ok, other=-1
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

    # ---- QK over all query heads, then the §4.3 dequant/reduce chain --------
    off_d = tl.arange(0, HEAD_DIM)
    off_g = tl.arange(0, HEAD_BLOCK)
    query_base = query_ptr + pid_n * (NUM_HEADS * HEAD_DIM)
    weight_base = weights_ptr + pid_n * NUM_HEADS
    scale_base = query_scale_ptr + pid_n * NUM_HEADS

    acc = tl.zeros((BLOCK_KV,), dtype=tl.float32)
    if USE_INT8_DOT:
        key_tile = _gather_key_tile(
            history_ptr,
            nodes_ptr,
            history_base,
            node_base,
            history_valid,
            node_valid,
            off_d,
        )
        for head_block in tl.static_range(0, NUM_HEADS // HEAD_BLOCK):
            head = head_block * HEAD_BLOCK + off_g
            query = tl.load(
                query_base + head[:, None] * HEAD_DIM + off_d[None, :]
            )
            qk = tl.dot(query, key_tile, out_dtype=tl.int32)
            score0 = tl.maximum(qk, 0).to(tl.float32) * 0.0009765625
            score0 = score0.to(tl.float16).to(tl.float32)
            weight = tl.load(weight_base + head)
            query_scale = tl.load(scale_base + head)
            wq = (weight * query_scale).to(tl.float16).to(tl.float32)
            acc += tl.sum(score0 * wq[:, None], axis=0)
    else:
        # R8 fallback: exact int32 widening, DOT_D_CHUNK head-dim slice per step.
        for head_block in tl.static_range(0, NUM_HEADS // HEAD_BLOCK):
            head = head_block * HEAD_BLOCK + off_g
            qk = tl.zeros((HEAD_BLOCK, BLOCK_KV), dtype=tl.int32)
            for chunk in tl.static_range(0, HEAD_DIM // DOT_D_CHUNK):
                cols = chunk * DOT_D_CHUNK + tl.arange(0, DOT_D_CHUNK)
                query = tl.load(query_base + head[:, None] * HEAD_DIM + cols[None, :])
                key_tile = _gather_key_tile(
                    history_ptr,
                    nodes_ptr,
                    history_base,
                    node_base,
                    history_valid,
                    node_valid,
                    cols,
                )
                qk += tl.sum(
                    query.to(tl.int32)[:, :, None] * key_tile.to(tl.int32)[None, :, :],
                    axis=1,
                )
            score0 = tl.maximum(qk, 0).to(tl.float32) * 0.0009765625
            score0 = score0.to(tl.float16).to(tl.float32)
            weight = tl.load(weight_base + head)
            query_scale = tl.load(scale_base + head)
            wq = (weight * query_scale).to(tl.float16).to(tl.float32)
            acc += tl.sum(score0 * wq[:, None], axis=0)

    score = tl.where(valid, acc * key_scale, float("-inf"))
    tl.store(scores_ptr + out_off, score)


@triton.jit(do_not_specialize=["topk"])
def _topk_merge_kernel(
    scores_ptr,  # [n, SLOTS] fp32, -inf marks an absent slot
    out_ptr,  # [n, 1, topk] int32 (out)
    topk,
    SLOTS: tl.constexpr,
):
    """Stable ``(score desc, index asc)`` TopK of one row; ``-1`` padding."""
    pid_n = tl.program_id(0)
    off = tl.arange(0, SLOTS)
    scores = tl.load(scores_ptr + pid_n * SLOTS + off).to(tl.float32)
    out_base = out_ptr + pid_n * topk

    for rank in tl.range(0, topk):
        best = tl.max(scores, axis=0)
        # Lowest position among ties -> index-ascending tie-break (D3).
        best_pos = tl.min(tl.where(scores == best, off, SLOTS), axis=0)
        tl.store(out_base + rank, tl.where(best > float("-inf"), best_pos, -1))
        # Consuming an -inf tail slot is a no-op, so padding repeats as -1.
        scores = tl.where(off == best_pos, float("-inf"), scores)


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

    slots = min(_TOPK_SLOTS, next_power_of_2(max(1, max_len)))
    slots = max(slots, _BLOCK_KV)

    use_int8_dot = _USE_INT8_DOT
    head_block = _HEAD_BLOCK_DOT if use_int8_dot else _HEAD_BLOCK_MANUAL

    scores = torch.empty((n, slots), dtype=torch.float32, device=query.device)
    _score_kernel[(n, slots // _BLOCK_KV)](
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
        lengths,
        scores,
        history.stride(0),
        history_scale.stride(0),
        int(history.shape[0]),
        int(table.shape[0]),
        int(table.shape[1]),
        int(ancestors.shape[1]),
        num_nodes,
        int(history.shape[1]),
        NUM_HEADS=_QUERY_HEADS,
        HEAD_DIM=_HEAD_DIM,
        RATIO=_RATIO,
        SLOTS=slots,
        BLOCK_KV=_BLOCK_KV,
        HEAD_BLOCK=head_block,
        DOT_D_CHUNK=_DOT_D_CHUNK,
        USE_INT8_DOT=use_int8_dot,
        num_warps=_NUM_WARPS,
        num_stages=_NUM_STAGES,
    )

    out = torch.empty((n, 1, topk), dtype=torch.int32, device=query.device)
    _topk_merge_kernel[(n,)](
        scores,
        out,
        topk,
        SLOTS=slots,
        num_warps=_NUM_WARPS,
        num_stages=_NUM_STAGES,
    )
    return out
