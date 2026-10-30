"""Triton port of the DSV4 DSpark ``compressor_tree`` epilogue.

Shape of the launch mirrors the native AIV kernel
(``op_kernel/compressor_tree.cpp:41-103``): ``min(n, AIV)`` programs, each one
walking its own strided subset of the nodes (``for node = blockIdx; node < n;
node += cores``) with the RMSNorm weight resident in UB for the whole walk.

Within one node the reduction is the source's two address passes:

1. pass 1 accumulates the per-dimension maximum over the valid slots;
2. pass 2 issues ``exp(score - max)`` and accumulates the denominator and the
   weight * kv numerator against that single global max.

Both passes walk the same ``RATIO * c`` slots, and the window is aggregated
**once** over the whole row: the interleaved RoPE pair ``(i, i ^ 1)`` is a
``tl.gather`` on the already-normalised row instead of a second and third window
aggregation over the even/odd tail lanes, so the node leaves through exactly one
contiguous BF16 store.

Nodes whose ``(prefix + depth + 1) % ratio`` is non-zero emit an exact zero row
(the source leaves ``value_`` zeroed and stores once at the epilogue).  State
inputs are read-only.

Semantics: ``docs/tree_ops_semantics.md`` §3; source pin vllm-ascend
``8947f1f9f`` ``csrc/attention/compressor_tree``.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ._common import (
    TreeContractError,
    check_head_dim,
    check_page_table,
    check_same_device,
    check_tensor,
    get_num_vectorcores,
    pick_specialization,
)

__all__ = ["compressor_tree"]

# Finite stand-in for -inf: ``exp(m - new_m)`` must never see ``-inf - -inf``,
# and the source initialises its running max with ``-FLT_MAX`` (line 48).
# Keep it as a literal inside the JIT body: some triton-ascend builds do not
# honor module-level constexpr annotations for globals.


@triton.jit
def _fp32_to_bf16_rne(x):
    """FP32 -> BF16 with round-to-nearest-even.

    Same bit-trick as ``norm/split_qkv_rmsnorm_rope_pos_cache_half_npu.py``,
    where the device default cast is not RNE.  The result is exactly the value
    the FP32 host reference rounds to.
    """
    u = tl.cast(x, tl.uint32, bitcast=True)
    lsb = (u >> 16) & 1
    u = (u + (0x7FFF + lsb)) & 0xFFFF0000
    return tl.cast(u, tl.float32, bitcast=True).to(tl.bfloat16)


@triton.jit
def _slot_load(
    state_ptr,
    nodes_ptr,
    table_ptr,
    ancestors_ptr,
    node,
    start,
    pos,
    req,
    req_ok,
    n,
    rows,
    pages,
    table_width,
    max_depth,
    state_stride0,
    state_stride1,
    w,
    lanes,
    D: tl.constexpr,
    C: tl.constexpr,
    RATIO: tl.constexpr,
    WITH_KV: tl.constexpr,
):
    """Resolve slot ``w`` and load its ``[1, D]`` score (and optionally kv) row.

    This is the source ``Address()`` (lines 105-122) followed by its ``Load()``:
    ``w // c`` walks the window position while ``w % c`` selects the
    previous/current projected half (C4 interleaves, C128 is flat).

    Slots are resolved branch-free: every candidate offset is clamped onto a
    live row, so a rejected slot reads in-bounds garbage that the caller masks
    out with ``valid`` (the source instead skips the transfer).  ``valid`` is the
    exact source predicate - position window, ancestor range, page range - so an
    invalid slot can never contribute to the reduction.
    """
    ROW_WIDTH: tl.constexpr = 2 * C * D
    half = w % C
    p = pos + 1 - RATIO * C + w // C + half * RATIO
    use_tree = p >= start

    # Pooled source: ancestors[node, step] selects nodes[ancestor + 1].
    step = p - start
    step_ok = (step >= 0) & (step < max_depth)
    step_safe = tl.minimum(tl.maximum(step, 0), max_depth - 1)
    ancestor = tl.load(ancestors_ptr + node * max_depth + step_safe).to(tl.int64)
    tree_ok = step_ok & (ancestor >= 0) & (ancestor < n)
    tree_base = (tl.where(tree_ok, ancestor, 0) + 1) * ROW_WIDTH

    # Paged source: table[requests[node], p // rows] selects state[page, p % rows].
    # ``p >= 0`` belongs in col_ok so an early slot can never produce a negative
    # row offset (the load is unconditional, the result is masked).
    col = p // rows
    col_ok = (p >= 0) & (col < table_width)
    page = tl.load(
        table_ptr + req * table_width + tl.where(col_ok, col, 0)
    ).to(tl.int64)
    page_ok = req_ok & col_ok & (page >= 0) & (page < pages)
    paged_base = tl.where(page_ok, page, 0) * state_stride0 + tl.where(
        page_ok, p % rows, 0
    ) * state_stride1

    valid = (p >= 0) & (p <= pos) & tl.where(use_tree, tree_ok, page_ok)
    lane = half * D + lanes

    if use_tree:
        score = tl.load(nodes_ptr + tree_base + C * D + lane)
        if WITH_KV:
            kv = tl.load(nodes_ptr + tree_base + lane)
        else:
            kv = tl.zeros((1, D), tl.float32)
    else:
        score = tl.load(state_ptr + paged_base + C * D + lane)
        if WITH_KV:
            kv = tl.load(state_ptr + paged_base + lane)
        else:
            kv = tl.zeros((1, D), tl.float32)
    return valid, score, kv


@triton.jit
def _compressor_tree_kernel(
    state_ptr,
    nodes_ptr,
    norm_ptr,
    cos_ptr,
    sin_ptr,
    table_ptr,
    prefix_ptr,
    ancestors_ptr,
    depths_ptr,
    requests_ptr,
    out_ptr,
    epsilon,
    n,
    rows,
    pages,
    table_width,
    table_rows,
    max_depth,
    cos_rows,
    state_stride0,
    state_stride1,
    cos_stride0,
    sin_stride0,
    cores,
    D: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    RATIO: tl.constexpr,
    C: tl.constexpr,
):
    """``min(n, cores)`` programs; program ``p`` owns nodes ``p, p+cores, ...``.

    ``D`` is the whole head width (128 or 512) and stays one tile: RMSNorm's
    reduction spans every ``d`` lane and the epilogue must leave through a single
    contiguous store, so splitting ``d`` across programs would need a
    cross-program reduction plus a ragged second pass.
    """
    pid = tl.program_id(0)
    lanes = tl.arange(0, D)[None, :]
    begin: tl.constexpr = D - ROPE_DIM

    # Node-invariant tiles, kept in UB for the whole walk (``Init()`` copying the
    # norm weight once, lines 30-33).  ``pair_index`` is the source ``indices_``
    # table (``i ^ 1``); ``sign`` is the -1/+1 pair sign; ``rope_lane`` maps an
    # absolute lane to its rope-local cos/sin column (clamped, ``tail`` masks it).
    norm_t = tl.load(norm_ptr + lanes)
    pair_index = lanes ^ 1
    sign = tl.where((lanes % 2) == 0, -1.0, 1.0)
    tail = lanes >= begin
    rope_lane = tl.where(tail, lanes - begin, 0)

    # Runtime trip count keeps ``cores`` off the specialization key: a constexpr
    # stride would recompile for every ``min(n, AIV)`` value.
    for it in range(0, (n - pid + cores - 1) // cores):
        node = pid + it * cores
        start = tl.load(prefix_ptr + node).to(tl.int64)
        pos = start + tl.load(depths_ptr + node).to(tl.int64)
        out_row = out_ptr + node.to(tl.int64) * D

        if (pos + 1) % RATIO != 0:
            # Non-emitting node: exact zero row (source: the emit guard + the
            # single unconditional epilogue store).
            tl.store(out_row + lanes, tl.zeros((1, D), tl.bfloat16))
        else:
            req_raw = tl.load(requests_ptr + node).to(tl.int64)
            req_ok = (req_raw >= 0) & (req_raw < table_rows)
            req = tl.where(req_ok, req_raw, 0)

            # Pass 1: global per-dimension max over the valid slots (lines 55-60).
            max_score = tl.full((1, D), -1e38, tl.float32)
            for w in range(0, RATIO * C):
                valid, score, _ = _slot_load(
                    state_ptr,
                    nodes_ptr,
                    table_ptr,
                    ancestors_ptr,
                    node.to(tl.int64),
                    start,
                    pos,
                    req,
                    req_ok,
                    n,
                    rows,
                    pages,
                    table_width,
                    max_depth,
                    state_stride0,
                    state_stride1,
                    w,
                    lanes,
                    D=D,
                    C=C,
                    RATIO=RATIO,
                    WITH_KV=False,
                )
                max_score = tl.maximum(max_score, tl.where(valid, score, -1e38))

            # Pass 2: exp/sum/weighted sum against that single max (lines 61-72).
            value = tl.zeros((1, D), tl.float32)
            denom = tl.zeros((1, D), tl.float32)
            for w in range(0, RATIO * C):
                valid, score, kv = _slot_load(
                    state_ptr,
                    nodes_ptr,
                    table_ptr,
                    ancestors_ptr,
                    node.to(tl.int64),
                    start,
                    pos,
                    req,
                    req_ok,
                    n,
                    rows,
                    pages,
                    table_width,
                    max_depth,
                    state_stride0,
                    state_stride1,
                    w,
                    lanes,
                    D=D,
                    C=C,
                    RATIO=RATIO,
                    WITH_KV=True,
                )
                weight = tl.where(valid, tl.exp(score - max_score), 0.0)
                denom += weight
                value += weight * tl.where(valid, kv, 0.0)

            # No reachable slot: 0/0 stays 0 so the row remains an exact zero.
            inv_denom = tl.where(denom > 0.0, 1.0 / denom, 0.0)
            value = value * inv_denom

            # RMSNorm over the whole row, eps inside the sqrt (lines 74-81).
            inv_rms = 1.0 / tl.sqrt(tl.sum(value * value) / D + epsilon)
            value = value * inv_rms * norm_t

            # RoPE on the trailing rope_dim lanes only.  ropePos lags the node
            # position by ratio - 1; clamping (instead of a host side range
            # check) keeps the launch free of a device sync, matching
            # norm/split_qkv_rmsnorm_rope_pos_cache_half_npu.py.  The pair
            # (i, i ^ 1) is read from the normalised row, i.e. the source
            # ``Gather(paired_, value_[begin], indices_, 0U, rd)``.
            rope_pos = pos + 1 - RATIO
            rope_pos = tl.minimum(tl.maximum(rope_pos, 0), cos_rows - 1)
            cos_row = cos_ptr + rope_pos * cos_stride0
            sin_row = sin_ptr + rope_pos * sin_stride0
            cos_t = tl.load(cos_row + rope_lane, mask=tail, other=0.0)
            sin_t = tl.load(sin_row + rope_lane, mask=tail, other=0.0)
            paired = tl.gather(value, pair_index, 1)
            rotated = value * cos_t + paired * sign * sin_t
            row = tl.where(tail, rotated, value)

            # Exactly one contiguous BF16 store per emitting node (lines 98-99).
            tl.store(out_row + lanes, _fp32_to_bf16_rne(row))


def compressor_tree(
    state,
    nodes,
    norm,
    cos,
    sin,
    table,
    prefix,
    ancestors,
    depths,
    requests,
    ratio,
    rope_dim,
    epsilon,
):
    """Branch-aware window compression of one draft-tree round.

    Args:
        state: FP32 paged history ``[pages, rows, 2*c*d]``; ``stride(1)`` may be
            padded, ``stride(2)`` must be 1.  Never written.
        nodes: FP32 pooled rows ``[n + 1, 1, 2*c*d]``; row 0 is reserved.
        norm: FP32 RMSNorm weight ``[d]``.
        cos: FP32 row table ``[..., >= rope_dim]`` with ``stride(-1) == 1``.
        sin: FP32 row table with the same layout as ``cos``.
        table: int32 page table ``[requests, pages]``.
        prefix: int32 ``[n]`` base history position per node.
        ancestors: int64 ``[n, max_depth]`` root-first node ids, ``-1`` padded.
        depths: int64 ``[n]`` tree depth per node.
        requests: int64 ``[n]`` request index per node.
        ratio: compression ratio, 4 (two interleaved halves) or 128 (one).
        rope_dim: 64.
        epsilon: RMSNorm epsilon, ``> 0``.

    Returns:
        BF16 ``[n, d]`` tensor; rows whose ``(prefix + depth + 1) % ratio`` is
        non-zero are exact zeros.
    """
    epsilon = float(epsilon)
    if not epsilon > 0.0:
        raise TreeContractError(f"epsilon: expected > 0, got {epsilon!r}")

    check_tensor("norm", norm, dtype=torch.float32, rank=1, contiguous=True)
    head_dim = check_head_dim(int(norm.numel()))
    ratio, c, d, rope_dim = pick_specialization(ratio, head_dim, rope_dim)
    row_width = 2 * c * d

    check_tensor("state", state, dtype=torch.float32, rank=3)
    if state.size(0) <= 0 or state.size(1) <= 0:
        raise TreeContractError(
            f"state: expected non-empty [pages, rows, {row_width}], "
            f"got shape {tuple(state.shape)}"
        )
    if state.size(2) != row_width:
        raise TreeContractError(
            f"state: last dim must be {row_width}, got {state.size(2)}"
        )
    if state.stride(1) < row_width:
        raise TreeContractError(
            f"state: stride(1) must be >= {row_width}, got {state.stride(1)}"
        )
    if state.stride(2) != 1:
        raise TreeContractError(f"state: stride(2) must be 1, got {state.stride(2)}")

    # Real callers allocate ``nodes`` as [n+1,1,1,2*c*d] (node_pool.py); the
    # native binding only cares about the flat row count/width.  Accept the
    # 3-D test/fixture form, the 4-D caller form, and a flat 1-D view.
    check_tensor("nodes", nodes, dtype=torch.float32, contiguous=True)
    if nodes.dim() == 1:
        if int(nodes.numel()) % row_width != 0:
            raise TreeContractError(
                f"nodes: numel {nodes.numel()} is not a multiple of {row_width}"
            )
        n = int(nodes.numel()) // row_width - 1
    else:
        n = int(nodes.size(0)) - 1
    if n < 0 or nodes.numel() != (n + 1) * row_width:
        raise TreeContractError(
            f"nodes: expected {n + 1} rows of {row_width}, "
            f"got shape {tuple(nodes.shape)}"
        )
    if nodes.dim() == 3 and (nodes.size(1) != 1 or nodes.size(2) != row_width):
        raise TreeContractError(
            f"nodes: expected [n + 1, 1, {row_width}], got shape {tuple(nodes.shape)}"
        )

    check_tensor("prefix", prefix, dtype=torch.int32, rank=1, contiguous=True)
    check_tensor("ancestors", ancestors, dtype=torch.int64, rank=2, contiguous=True)
    check_tensor("depths", depths, dtype=torch.int64, rank=1, contiguous=True)
    check_tensor("requests", requests, dtype=torch.int64, rank=1, contiguous=True)
    check_page_table("table", table)

    for name, tensor in (
        ("prefix", prefix),
        ("depths", depths),
        ("requests", requests),
    ):
        if tensor.numel() != n:
            raise TreeContractError(
                f"{name}: expected {n} entries, got {tensor.numel()}"
            )
    if ancestors.size(0) != n:
        raise TreeContractError(
            f"ancestors: expected {n} rows, got {ancestors.size(0)}"
        )
    max_depth = int(ancestors.size(1))
    if max_depth < 1:
        raise TreeContractError("ancestors: max_depth must be >= 1")

    tables = []
    for name, tensor in (("cos", cos), ("sin", sin)):
        check_tensor(name, tensor, dtype=torch.float32)
        if tensor.dim() < 2:
            raise TreeContractError(f"{name}: expected a row table, got rank {tensor.dim()}")
        if tensor.stride(-1) != 1:
            raise TreeContractError(f"{name}: stride(-1) must be 1, got {tensor.stride(-1)}")
        if tensor.shape[-1] < rope_dim:
            raise TreeContractError(
                f"{name}: row width must be >= rope_dim ({rope_dim}), got {tensor.shape[-1]}"
            )
        # Leading dims are flattened exactly like the CPU reference does.
        tables.append(
            tensor
            if tensor.dim() == 2
            else tensor.reshape(-1, tensor.shape[-1])
        )
    cos, sin = tables
    if int(cos.size(0)) <= 0 or int(sin.size(0)) <= 0:
        raise TreeContractError(
            f"cos/sin: expected non-empty row tables, got {tuple(cos.shape)} / {tuple(sin.shape)}"
        )

    check_same_device(
        "compressor_tree",
        [state, nodes, norm, cos, sin, table, prefix, ancestors, depths, requests],
    )

    out = torch.empty((n, d), dtype=torch.bfloat16, device=state.device)
    if n == 0:
        return out

    # ``cores = min(n, GetCoreNumAiv())`` with one program per core, exactly like
    # the host tiling (``compressor_tree_tiling.cpp:12-27``): ``SetBlockDim`` is
    # the core count and the kernel walks ``node += cores``.
    cores = min(n, get_num_vectorcores())

    _compressor_tree_kernel[(cores,)](
        state,
        nodes,
        norm,
        cos,
        sin,
        table,
        prefix,
        ancestors,
        depths,
        requests,
        out,
        epsilon,
        n,
        int(state.size(1)),
        int(state.size(0)),
        int(table.size(1)),
        int(table.size(0)),
        max_depth,
        int(cos.size(0)),
        int(state.stride(0)),
        int(state.stride(1)),
        int(cos.stride(0)),
        int(sin.stride(0)),
        cores,
        D=d,
        ROPE_DIM=rope_dim,
        RATIO=ratio,
        C=c,
        num_warps=1,
        num_stages=1,
    )
    return out
