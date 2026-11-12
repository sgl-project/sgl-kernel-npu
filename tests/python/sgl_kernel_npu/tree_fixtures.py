"""Deterministic, CPU-only fixtures for the DSV4 DSpark tree reference layer.

Only ``torch`` and the standard library are imported: no triton, no torch_npu,
no ``sgl_kernel_npu``.  Every fixture is seeded so failures reproduce exactly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

__all__ = [
    "TreeSpec",
    "make_tree",
    "make_project_case",
    "make_compressor_case",
    "make_indexer_case",
    "make_attention_case",
]


@dataclass(frozen=True)
class TreeSpec:
    """Per-query-node verify geometry for one branching draft tree round.

    ``ancestors`` is root-first (self included) and ``-1`` padded, with node
    ids globalized over the flattened ``batch * (budget + 1)`` query rows.
    """

    prefixes: torch.Tensor  # int32 [n]
    depths: torch.Tensor  # int64 [n]
    ancestors: torch.Tensor  # int64 [n, max_depth + 1]
    requests: torch.Tensor  # int64 [n]
    cu: torch.Tensor  # int32 [n + 1]
    positions: torch.Tensor  # int32 [n]
    seqused: torch.Tensor  # int32 [n]
    roots: torch.Tensor  # int64 [batch]
    query_width: int


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def make_tree(
    n=None,
    batch: int = 1,
    budget: int = 4,
    depth: int = 3,
    seed: int = 0,
    ratio: int = 4,
    prefixes=None,
) -> TreeSpec:
    """Build a deterministic branching tree (never a chain when ``budget >= 2``).

    ``budget`` counts non-root nodes per request; the verify width is
    ``budget + 1`` (anchor/root included).  Children are distributed
    breadth-first, at most two per parent, so node 0 of every request has two
    children whenever ``budget >= 2`` and ``depth >= 1``.
    """
    batch = int(batch)
    budget = int(budget)
    depth = int(depth)
    _check(batch >= 1, "batch must be >= 1")
    _check(budget >= 0, "budget must be >= 0")
    _check(depth >= 0, "depth must be >= 0")
    if budget > 0:
        _check(depth >= 1, "a non-empty tree needs depth >= 1")

    width = budget + 1
    total = batch * width
    if n is not None:
        _check(int(n) == total, f"n={n} != batch*(budget+1)={total}")

    # Local (per-request) parent/depth tables.
    parent: dict[int, int] = {}
    node_depth: dict[int, int] = {0: 0}
    next_id = 1
    level = [0]
    cur_depth = 1
    while next_id <= budget and cur_depth <= depth:
        new_level: list[int] = []
        for p in level:
            for _ in range(2):
                if next_id > budget:
                    break
                parent[next_id] = p
                node_depth[next_id] = cur_depth
                new_level.append(next_id)
                next_id += 1
            if next_id > budget:
                break
        if not new_level:
            break
        level = new_level
        cur_depth += 1
    _check(
        next_id > budget,
        f"budget {budget} exceeds the branching capacity for depth {depth}",
    )

    max_depth = max(node_depth.values()) if node_depth else 0

    if prefixes is None:
        prefix_list = [
            1 + ((int(seed) * 7 + 11 * req + 3) % max(1, 2 * int(ratio)))
            for req in range(batch)
        ]
    else:
        prefix_list = [int(v) for v in prefixes]
        _check(len(prefix_list) == batch, "prefixes must have one entry per request")

    prefixes_t = torch.tensor(prefix_list, dtype=torch.int32).repeat_interleave(width)
    depths_t = torch.zeros(total, dtype=torch.int64)
    ancestors_t = torch.full((total, max_depth + 1), -1, dtype=torch.int64)
    requests_t = torch.arange(batch, dtype=torch.int64).repeat_interleave(width)
    roots = torch.tensor([req * width for req in range(batch)], dtype=torch.int64)

    for req in range(batch):
        offset = req * width
        for local in range(width):
            row = offset + local
            depths_t[row] = node_depth[local]
            path = []
            cur = local
            while True:
                path.append(cur)
                if cur == 0:
                    break
                cur = parent[cur]
            path.reverse()
            _check(
                len(path) <= max_depth + 1,
                "ancestor path does not fit the padded width",
            )
            ancestors_t[row, : len(path)] = torch.tensor(
                [offset + node for node in path], dtype=torch.int64
            )

    positions_t = (prefixes_t.to(torch.int64) + depths_t).to(torch.int32)
    seqused_t = (positions_t.to(torch.int64) + 1).to(torch.int32)
    return TreeSpec(
        prefixes=prefixes_t,
        depths=depths_t,
        ancestors=ancestors_t,
        requests=requests_t,
        cu=torch.arange(total + 1, dtype=torch.int32),
        positions=positions_t,
        seqused=seqused_t,
        roots=roots,
        query_width=width,
    )


# ---------------------------------------------------------------------------
# A. compressor_tree_project
# ---------------------------------------------------------------------------


def make_project_case(
    *,
    seed: int = 0,
    ratio: int = 4,
    head_dim: int = 128,
    hidden: int = 8,
    n: int = 3,
) -> dict:
    """BF16 TH projection case: ``nodes`` row 0 is a sentinel (must survive)."""
    c = 2 if int(ratio) == 4 else 1
    d = int(head_dim)
    g = torch.Generator().manual_seed(int(seed))

    x = torch.randint(-4, 5, (int(n), int(hidden)), generator=g, dtype=torch.int16)
    wkv = torch.randint(
        -3, 4, (c * d, int(hidden)), generator=g, dtype=torch.int16
    )
    wgate = torch.randint(
        -3, 4, (c * d, int(hidden)), generator=g, dtype=torch.int16
    )
    nodes = torch.full((int(n) + 1, 1, 2 * c * d), -1.0, dtype=torch.float32)
    nodes[0].fill_(7.0)
    ape = torch.randn(int(ratio), c * d, generator=g, dtype=torch.float32) * 0.1
    positions = torch.randint(
        0, int(ratio), (int(n),), generator=g, dtype=torch.int32
    )
    return dict(
        x=x.to(torch.bfloat16),
        wkv=wkv.to(torch.bfloat16),
        wgate=wgate.to(torch.bfloat16),
        nodes=nodes,
        ape=ape,
        positions=positions,
        cu=torch.arange(int(n) + 1, dtype=torch.int32),
        ratio=int(ratio),
        head_dim=d,
        hidden=int(hidden),
    )


# ---------------------------------------------------------------------------
# B. compressor_tree
# ---------------------------------------------------------------------------


def make_compressor_case(
    *,
    seed: int = 0,
    ratio: int = 4,
    head_dim: int = 128,
    rows: int = 4,
    page_pad: int = 0,
    batch: int = 1,
    budget: int = 2,
    depth: int = 2,
    rope_dim: int = 64,
    eps: float = 1e-6,
    norm_value: float = 1.0,
    poison_halves: bool = False,
    invalid_ancestor: bool = False,
    bad_page: bool = False,
    cos_pad: int = 0,
    prefixes=None,
) -> dict:
    """Compressor aggregation case with padded state / cos / sin row strides.

    ``poison_halves`` fills the history KV with ``+1`` (half 0) and ``-1``
    (half 1) and the pooled node rows with ``+3`` / ``-3`` so that selecting
    the wrong C4 half flips the sign of the aggregated value.
    """
    c = 2 if int(ratio) == 4 else 1
    d = int(head_dim)
    rows_i = int(rows)
    spec = make_tree(
        batch=int(batch),
        budget=int(budget),
        depth=int(depth),
        seed=int(seed),
        ratio=int(ratio),
        prefixes=prefixes,
    )
    n = int(spec.prefixes.numel())
    g = torch.Generator().manual_seed(int(seed) + 97)

    max_pos = int(spec.positions.max().item())
    max_col = max_pos // rows_i + 1
    pages = max_col + 1

    width = 2 * c * d
    state_buf = torch.zeros(pages, rows_i, width + int(page_pad), dtype=torch.float32)
    state = state_buf[..., :width]
    if poison_halves:
        state[..., :d] = 1.0
        if c > 1:
            state[..., d : c * d] = -1.0
        state[..., c * d :] = 0.0
    else:
        state.copy_(torch.randn(pages, rows_i, width, generator=g, dtype=torch.float32))

    table = torch.full((int(batch), max_col), -1, dtype=torch.int32)
    for req in range(int(batch)):
        table[req, :max_col] = torch.arange(max_col, dtype=torch.int32)
    if bad_page:
        table[:, 0] = -1

    nodes = torch.zeros(n + 1, 1, width, dtype=torch.float32)
    if poison_halves:
        nodes[1:, 0, :d] = 3.0
        if c > 1:
            nodes[1:, 0, d : c * d] = -3.0
        nodes[1:, 0, c * d :] = 0.0
    else:
        nodes[1:].copy_(
            torch.randn(n, 1, width, generator=g, dtype=torch.float32)
        )

    ancestors = spec.ancestors.clone()
    if invalid_ancestor:
        for row in range(n):
            depth_row = int(spec.depths[row])
            if depth_row > 0:
                ancestors[row, depth_row] = -1

    norm = torch.full((d,), float(norm_value), dtype=torch.float32)
    if norm_value is None:
        norm = (torch.rand(d, generator=g, dtype=torch.float32) + 0.5).to(torch.float32)

    cos_rows = max_pos + 3
    idx = torch.arange(int(rope_dim), dtype=torch.float32)
    phase = torch.arange(cos_rows, dtype=torch.float32).unsqueeze(1) * 0.01
    cos_buf = torch.zeros(cos_rows, int(rope_dim) + int(cos_pad), dtype=torch.float32)
    sin_buf = torch.zeros(cos_rows, int(rope_dim) + int(cos_pad), dtype=torch.float32)
    cos_buf[:, : int(rope_dim)] = torch.cos(phase + idx * 0.01)
    sin_buf[:, : int(rope_dim)] = torch.sin(phase + idx * 0.01)
    cos = cos_buf[:, : int(rope_dim)]
    sin = sin_buf[:, : int(rope_dim)]

    args = dict(
        state=state,
        nodes=nodes,
        norm=norm,
        cos=cos,
        sin=sin,
        table=table,
        prefix=spec.prefixes,
        ancestors=ancestors,
        depths=spec.depths,
        requests=spec.requests,
        ratio=int(ratio),
        rope_dim=int(rope_dim),
        eps=float(eps),
    )
    return dict(
        args=args,
        n=n,
        spec=spec,
        # The state/cos/sin trailing columns are alignment padding; poisoning
        # them must not change the result.
        padding_views=[state_buf, cos_buf, sin_buf],
    )


# ---------------------------------------------------------------------------
# C. quant_lightning_indexer_tree
# ---------------------------------------------------------------------------


def make_indexer_case(
    *,
    seed: int = 0,
    rows: int = 4,
    topk: int = 4,
    batch: int = 1,
    budget: int = 2,
    depth: int = 2,
    invalid_ancestor: bool = False,
    prefixes=None,
    head_dim: int = 128,
) -> dict:
    """INT8 indexer case: tree node keys score far above the paged history."""
    d = int(head_dim)
    rows_i = int(rows)
    spec = make_tree(
        batch=int(batch),
        budget=int(budget),
        depth=int(depth),
        seed=int(seed),
        ratio=4,
        prefixes=prefixes,
    )
    n = int(spec.prefixes.numel())
    g = torch.Generator().manual_seed(int(seed) + 131)

    lengths = ((spec.prefixes.to(torch.int64) + spec.depths + 1) // 4).to(torch.int32)
    residual = ((spec.prefixes.to(torch.int64) + spec.depths + 1) % 4).to(torch.int32)

    query = torch.zeros(n, 1, d, dtype=torch.int8)
    query[:, 0, 0] = 1  # dot product reduces to element 0 of the key
    weights = torch.ones(n, 1, dtype=torch.float16)
    qs = torch.ones(n, 1, dtype=torch.float16)

    history_end = (spec.prefixes // 4).to(torch.int64)
    cols = max(1, (int(history_end.max().item()) + rows_i - 1) // rows_i)
    pages = cols
    history = torch.zeros(pages, rows_i, 1, d, dtype=torch.int8)
    history[..., 0, 0] = 1  # score 1/1024
    history_scale = torch.ones(pages, rows_i, 1, 1, dtype=torch.float16)
    block_table = torch.arange(cols, dtype=torch.int32).reshape(1, cols).repeat(
        int(batch), 1
    )

    tree_key = torch.zeros(n, d, dtype=torch.int8)
    tree_key[:, 0] = 100  # score 100/1024, wins everywhere
    tree_scale = torch.ones(n, dtype=torch.float16)

    ancestors = spec.ancestors.clone()
    if invalid_ancestor:
        for row in range(n):
            depth_row = int(spec.depths[row])
            if depth_row > 0:
                ancestors[row, depth_row] = -1

    args = dict(
        query=query,
        history=history,
        weights=weights,
        qs=qs,
        tree_key=tree_key,
        tree_scale=tree_scale,
        ancestors=ancestors,
        prefixes=spec.prefixes,
        reqs=spec.requests,
        block_table=block_table,
        cu=spec.cu,
        lengths=lengths,
        residual=residual,
        metadata=torch.zeros(1024, dtype=torch.int32),
        topk=int(topk),
        head_dim=d,
        history_scale=history_scale,
        max_kv=None,
    )
    return dict(
        args=dict(args),
        spec=spec,
        lengths=lengths,
        residual=residual,
        history_end=history_end,
    )


# ---------------------------------------------------------------------------
# D. sparse_attn_sharedkv_tree
# ---------------------------------------------------------------------------


def _align(value: int, alignment: int) -> int:
    return ((int(value) + alignment - 1) // alignment) * alignment


def make_attention_case(
    *,
    seed: int = 0,
    ratio: int = 4,
    batch: int = 1,
    budget: int = 1,
    depth: int = 1,
    rows: int = 4,
    window: int = 4,
    sink: bool = False,
    cmp_mode: str = "c4",
    dtype: torch.dtype = torch.bfloat16,
    head_dim: int = 512,
    heads: int = 1,
    prefixes=None,
) -> dict:
    """Tree attention case.  ``cmp_mode`` is ``c4``, ``c128`` or ``swa``."""
    _check(cmp_mode in ("c4", "c128", "swa"), "cmp_mode must be c4/c128/swa")
    d = int(head_dim)
    rows_i = int(rows)
    ratio_i = int(ratio)
    spec = make_tree(
        batch=int(batch),
        budget=int(budget),
        depth=int(depth),
        seed=int(seed),
        ratio=ratio_i,
        prefixes=prefixes,
    )
    n = int(spec.prefixes.numel())
    g = torch.Generator().manual_seed(int(seed) + 197)

    q = torch.randn(n, int(heads), d, generator=g, dtype=torch.float32).to(dtype)

    max_pos = int(spec.positions.max().item())
    ori_pages = max_pos // rows_i + 2
    ori_kv = torch.zeros(ori_pages, rows_i, 1, d, dtype=dtype)
    ori_kv.copy_(torch.randn(ori_pages, rows_i, 1, d, generator=g).to(dtype))

    width = _align(max(4, int(window) + 1), 4)
    ori_slots = torch.full((n, 1, width), -1, dtype=torch.int32)
    for row in range(n):
        pos = int(spec.positions[row])
        start = max(0, pos - int(window) + 1)
        slots = list(range(start, pos + 1))[:width]
        ori_slots[row, 0, : len(slots)] = torch.tensor(slots, dtype=torch.int32)

    limit = (spec.seqused // ratio_i).to(torch.int64)
    max_limit = int(limit.max().item())
    cmp_pages = max(1, (max(1, max_limit) + rows_i - 1) // rows_i)
    cmp_kv = None
    cmp_slots = None
    cmp_tree_kv = None
    ancestors = None
    prefix = None
    seqused = spec.seqused
    cmp_block_table = None

    if cmp_mode != "swa":
        cmp_kv = torch.zeros(cmp_pages, rows_i, 1, d, dtype=dtype)
        cmp_kv.copy_(torch.randn(cmp_pages, rows_i, 1, d, generator=g).to(dtype))
        # The native wrapper gathers block tables by request id, so the op sees
        # one row per query token.
        cmp_block_table = (
            torch.arange(cmp_pages, dtype=torch.int32)
            .reshape(1, cmp_pages)
            .repeat(n, 1)
        )
        cmp_tree_kv = torch.randn(n, d, generator=g).to(dtype)
        ancestors = spec.ancestors
        prefix = spec.prefixes
        if cmp_mode == "c4":
            k_width = max(1, _align(max_limit, 4))
            cmp_slots = torch.full((n, 1, k_width), -1, dtype=torch.int32)
            for row in range(n):
                length = int(limit[row])
                if length > 0:
                    cmp_slots[row, 0, :length] = torch.arange(
                        length, dtype=torch.int32
                    )

    sinks = None
    if sink:
        sinks = torch.randn(n, int(heads), generator=g, dtype=torch.float32)

    return dict(
        q=q,
        ori_kv=ori_kv,
        cmp_kv=cmp_kv,
        ori_sparse_indices=ori_slots,
        cmp_sparse_indices=cmp_slots,
        ori_block_table=(
            torch.arange(ori_pages, dtype=torch.int32)
            .reshape(1, ori_pages)
            .repeat(n, 1)
        ),
        cmp_block_table=(
            cmp_block_table
            if cmp_block_table is not None
            else torch.zeros((n, 1), dtype=torch.int32)
        ),
        cu_seqlens_q=spec.cu,
        cu_seqlens_ori_kv=torch.cat(
            (
                torch.zeros(1, dtype=torch.int32),
                torch.cumsum(seqused, 0, dtype=torch.int32),
            )
        ),
        cu_seqlens_cmp_kv=torch.cat(
            (
                torch.zeros(1, dtype=torch.int32),
                torch.cumsum(seqused // ratio_i, 0, dtype=torch.int32),
            )
        ),
        seqused_q=torch.empty(0, dtype=torch.int32),
        seqused_kv=seqused,
        sinks=sinks,
        metadata=torch.zeros(1024, dtype=torch.int32),
        cmp_tree_kv=cmp_tree_kv,
        tree_ancestors=ancestors,
        tree_prefix=prefix,
        softmax_scale=1.0 / math.sqrt(float(d)),
        cmp_ratio=ratio_i,
        ori_mask_mode=4,
        cmp_mask_mode=3,
        ori_win_left=int(window) - 1,
        ori_win_right=0,
        layout_q="TND",
        layout_kv="PA_ND",
        spec=spec,
        cmp_mode=cmp_mode,
    )
