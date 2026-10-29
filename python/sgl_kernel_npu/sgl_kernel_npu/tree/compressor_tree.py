"""Triton port of the DSV4 DSpark ``compressor_tree`` epilogue.

One program per tree node aggregates a ``ratio * c`` wide branch window of
``[kv | score]`` rows -- either the pooled tree-node rows or the paged FP32
history -- with a per-dimension softmax, then applies RMSNorm (``eps`` inside the
sqrt), interleaved RoPE on the trailing ``rope_dim`` lanes, and a BF16
round-to-nearest-even cast.  Nodes whose ``(prefix + depth + 1) % ratio`` is
non-zero emit an exact zero row.

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
    pick_specialization,
)

__all__ = ["compressor_tree"]

# Finite stand-in for -inf: ``exp(m - new_m)`` must never see ``-inf - -inf``.
# Triton JIT may only read globals annotated as constexpr.
_NEG: tl.constexpr = -1e38


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
def _window_average(
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
    lanes,
    N_LANES: tl.constexpr,
    D: tl.constexpr,
    C: tl.constexpr,
    RATIO: tl.constexpr,
):
    """Softmax-average the branch window into ``lanes`` of the ``d``-wide row.

    ``lanes`` are element offsets inside one projected half block, so the same
    routine serves the full-row pass and the two interleaved RoPE passes.  All
    slots are resolved branch-free: trees come from ``ancestors``/``nodes``, the
    history from ``table``/``state``, and every offset is clamped so an invalid
    slot never leaves its buffer.
    """
    row_width: tl.constexpr = 2 * C * D
    value = tl.zeros((1, N_LANES), tl.float32)
    denom = tl.zeros((1, N_LANES), tl.float32)
    max_score = tl.full((1, N_LANES), _NEG, tl.float32)

    # Slot order matches the source: w // c walks the window position and
    # w % c selects the previous/current half (C4 interleaves, C128 is flat).
    for w in range(0, RATIO * C):
        half = w % C
        p = pos + 1 - RATIO * C + w // C + half * RATIO
        use_tree = p >= start

        # Pooled source: ancestors[node, step] selects nodes[ancestor + 1].
        step = p - start
        step_ok = (step >= 0) & (step < max_depth)
        step_safe = tl.minimum(tl.maximum(step, 0), max_depth - 1)
        ancestor = tl.load(ancestors_ptr + node * max_depth + step_safe).to(tl.int64)
        tree_ok = step_ok & (ancestor >= 0) & (ancestor < n)
        tree_base = (tl.where(tree_ok, ancestor, 0) + 1) * row_width

        # Paged source: table[requests[node], p // rows] selects state[page, p % rows].
        # ``p >= 0`` belongs in page_ok so an early slot can never produce a
        # negative row offset (the load is unconditional, the result is masked).
        col = p // rows
        col_ok = (col >= 0) & (col < table_width)
        page = tl.load(
            table_ptr + req * table_width + tl.where(col_ok, col, 0)
        ).to(tl.int64)
        page_ok = req_ok & (p >= 0) & col_ok & (page >= 0) & (page < pages)
        paged_base = tl.where(page_ok, page, 0) * state_stride0 + tl.where(
            page_ok, p % rows, 0
        ) * state_stride1

        valid = (p >= 0) & (p <= pos) & tl.where(use_tree, tree_ok, page_ok)

        if use_tree:
            kv = tl.load(nodes_ptr + tree_base + half * D + lanes)
            score = tl.load(nodes_ptr + tree_base + C * D + half * D + lanes)
        else:
            kv = tl.load(state_ptr + paged_base + half * D + lanes)
            score = tl.load(state_ptr + paged_base + C * D + half * D + lanes)

        # Online, per-dimension softmax.  Skipped slots keep the running state:
        # keep == 1 and weight == 0 leave value/denom untouched.
        new_max = tl.where(valid, tl.maximum(max_score, score), max_score)
        keep = tl.where(valid, tl.exp(max_score - new_max), 1.0)
        weight = tl.where(valid, tl.exp(score - new_max), 0.0)
        denom = denom * keep + weight
        value = value * keep + weight * tl.where(valid, kv, 0.0)
        max_score = new_max

    # No reachable slot: 0/0 must stay 0 so the row stays an exact zero.
    inv_denom = tl.where(denom > 0.0, 1.0 / denom, 0.0)
    return value * inv_denom


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
    D: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    RATIO: tl.constexpr,
    C: tl.constexpr,
):
    """One program per node; ``D`` is the whole head width (128 or 512).

    RMSNorm's reduction spans every ``d`` lane, so splitting ``d`` across
    programs would need a cross-program reduction (workspace + second pass).
    One program keeps the row in UB: an FP32 ``[1, D]`` tile is at most 2 KiB.
    """
    node = tl.program_id(0)
    lanes = tl.arange(0, D)[None, :]
    out_row = out_ptr + node.to(tl.int64) * D

    start = tl.load(prefix_ptr + node).to(tl.int64)
    pos = start + tl.load(depths_ptr + node).to(tl.int64)

    if (pos + 1) % RATIO != 0:
        # Non-emitting node: exact zero row (source: the emit guard).
        tl.store(out_row + lanes, tl.zeros((1, D), tl.bfloat16))
        return

    req_raw = tl.load(requests_ptr + node).to(tl.int64)
    req_ok = (req_raw >= 0) & (req_raw < table_rows)

    value = _window_average(
        state_ptr,
        nodes_ptr,
        table_ptr,
        ancestors_ptr,
        node.to(tl.int64),
        start,
        pos,
        tl.where(req_ok, req_raw, 0),
        req_ok,
        n,
        rows,
        pages,
        table_width,
        max_depth,
        state_stride0,
        state_stride1,
        lanes,
        N_LANES=D,
        D=D,
        C=C,
        RATIO=RATIO,
    )

    # RMSNorm over the whole row: eps lives inside the sqrt.
    mean_square = tl.sum(value * value) / D + epsilon
    inv_rms = 1.0 / tl.sqrt(mean_square)
    value = value * inv_rms * tl.load(norm_ptr + lanes)

    begin = D - ROPE_DIM
    # The non-RoPE prefix is final now; only the trailing tail is rewritten below.
    tl.store(out_row + lanes, _fp32_to_bf16_rne(value), mask=lanes < begin)

    # ropePos lags the node position by ratio - 1.  Clamping (instead of a host
    # side range check) keeps the launch free of a device sync, matching
    # norm/split_qkv_rmsnorm_rope_pos_cache_half_npu.py.
    rope_pos = pos + 1 - RATIO
    rope_pos = tl.minimum(tl.maximum(rope_pos, 0), cos_rows - 1)

    pairs = tl.arange(0, ROPE_DIM // 2)[None, :]
    even_lanes = begin + 2 * pairs
    odd_lanes = even_lanes + 1

    # The interleaved pair (i, i ^ 1) needs both lanes of the normalised value,
    # so the tail is re-aggregated in even/odd order rather than gathered.
    even_value = _window_average(
        state_ptr,
        nodes_ptr,
        table_ptr,
        ancestors_ptr,
        node.to(tl.int64),
        start,
        pos,
        tl.where(req_ok, req_raw, 0),
        req_ok,
        n,
        rows,
        pages,
        table_width,
        max_depth,
        state_stride0,
        state_stride1,
        even_lanes,
        N_LANES=ROPE_DIM // 2,
        D=D,
        C=C,
        RATIO=RATIO,
    )
    odd_value = _window_average(
        state_ptr,
        nodes_ptr,
        table_ptr,
        ancestors_ptr,
        node.to(tl.int64),
        start,
        pos,
        tl.where(req_ok, req_raw, 0),
        req_ok,
        n,
        rows,
        pages,
        table_width,
        max_depth,
        state_stride0,
        state_stride1,
        odd_lanes,
        N_LANES=ROPE_DIM // 2,
        D=D,
        C=C,
        RATIO=RATIO,
    )
    even_value = even_value * inv_rms * tl.load(norm_ptr + even_lanes)
    odd_value = odd_value * inv_rms * tl.load(norm_ptr + odd_lanes)

    cos_row = cos_ptr + rope_pos * cos_stride0
    sin_row = sin_ptr + rope_pos * sin_stride0
    cos_even = tl.load(cos_row + 2 * pairs)
    cos_odd = tl.load(cos_row + 2 * pairs + 1)
    sin_even = tl.load(sin_row + 2 * pairs)
    sin_odd = tl.load(sin_row + 2 * pairs + 1)

    # sign(i) is -1 on even i and +1 on odd i; cos/sin are indexed by the
    # rope-local lane, not by the absolute head offset.
    tl.store(
        out_row + even_lanes,
        _fp32_to_bf16_rne(even_value * cos_even - odd_value * sin_even),
    )
    tl.store(
        out_row + odd_lanes,
        _fp32_to_bf16_rne(odd_value * cos_odd + even_value * sin_odd),
    )


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

    check_same_device(
        "compressor_tree",
        [state, nodes, norm, cos, sin, table, prefix, ancestors, depths, requests],
    )

    out = torch.empty((n, d), dtype=torch.bfloat16, device=state.device)
    if n == 0:
        return out

    _compressor_tree_kernel[(n,)](
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
        D=d,
        ROPE_DIM=rope_dim,
        RATIO=ratio,
        C=c,
        num_warps=1,
        num_stages=1,
    )
    return out
