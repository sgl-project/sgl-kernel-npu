"""Independent, CPU-runnable reference layer for the DSV4 DSpark tree ops.

Pure ``torch``: no triton, no ``torch_npu`` and no ``sgl_kernel_npu`` imports.
The four functions below re-implement, independently of the device kernels, the
integer/exact semantics of the AscendC tree operators that shipped in
vllm-ascend at commit ``8947f1f9f``:

* ``csrc/attention/compressor_tree_project`` (FP32 node projection)
* ``csrc/attention/compressor_tree``         (branch-aware window epilogue)
* ``csrc/attention/quant_lightning_indexer_tree`` (INT8 tree indexer)
* ``csrc/attention/sparse_attn_sharedkv_tree``    (tree sparse attention)

They are intentionally written as slow, explicit loops so that a Triton port
can be validated against them on a CPU-only host.
"""

from __future__ import annotations

import torch

__all__ = [
    "compressor_tree_project_ref",
    "compressor_tree_ref",
    "quant_lightning_indexer_tree_ref",
    "sparse_attn_sharedkv_tree_ref",
]


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def _c_of(ratio: int) -> int:
    return 2 if int(ratio) == 4 else 1


# ---------------------------------------------------------------------------
# A. compressor_tree_project
# ---------------------------------------------------------------------------


def compressor_tree_project_ref(
    x, wkv, wgate, nodes, ape, positions, cu, ratio, head_dim
) -> None:
    """Project each tree node once into its FP32 ``[kv | score]`` row.

    ``kv = x @ wkv.T`` and ``score = x @ wgate.T + ape[positions[node]]`` in
    FP32.  With ``c = 2`` for ``ratio == 4`` (previous/current window halves)
    and ``c = 1`` otherwise, the node row is laid out as

        ``[kv(H0) d | .. kv(Hc-1) d | score(H0) d | .. score(Hc-1) d]``

    Row 0 of ``nodes`` is a reserved sentinel and is never written.
    """
    ratio = int(ratio)
    head_dim = int(head_dim)
    c = _c_of(ratio)
    d = head_dim

    _check(x.dim() == 2, "x must be [T, H]")
    _check(x.dtype == torch.bfloat16, "x must be BF16")
    _check(nodes.dim() == 3 and nodes.shape[1] == 1, "nodes must be [n+1, 1, 2*c*d]")
    n = int(x.shape[0])
    _check(nodes.shape[0] == n + 1, "nodes must hold exactly n + 1 rows")
    _check(nodes.shape[2] == 2 * c * d, "nodes row width must be 2*c*d")
    _check(nodes.dtype == torch.float32, "nodes must be FP32")
    _check(
        wkv.shape == (c * d, x.shape[1]) and wgate.shape == (c * d, x.shape[1]),
        "wkv/wgate must be [c*d, H]",
    )
    _check(wkv.dtype == torch.bfloat16 and wgate.dtype == torch.bfloat16, "weights must be BF16")
    _check(ape.shape == (ratio, c * d), "ape must be [ratio, c*d]")
    _check(ape.dtype == torch.float32, "ape must be FP32")
    _check(positions.dtype == torch.int32 and positions.numel() == n, "positions must be int32 [n]")
    _check(cu.dtype == torch.int32 and cu.numel() == n + 1, "cu must be int32 [n+1]")
    _check(d in (128, 512), "head_dim must be 128 or 512")
    _check(ratio in (4, 128), "ratio must be 4 or 128")
    _check(bool((positions >= 0).all() and (positions < ratio).all()), "positions are APE residues")
    _check(
        torch.equal(cu, torch.arange(n + 1, dtype=torch.int32, device=cu.device)),
        "tree projection contract requires cu == arange(n + 1)",
    )

    xf = x.to(torch.float32)
    kv = xf @ wkv.to(torch.float32).t()
    score = xf @ wgate.to(torch.float32).t()
    score = score + ape.to(torch.float32)[positions.to(torch.int64)]

    with torch.no_grad():
        nodes[1:, 0, : c * d] = kv
        nodes[1:, 0, c * d :] = score
    return None


# ---------------------------------------------------------------------------
# B. compressor_tree
# ---------------------------------------------------------------------------


def compressor_tree_ref(
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
    eps,
):
    """Branch-aware window aggregation + RMSNorm + RoPE, BF16 output ``[n, d]``.

    Rows whose ``(pos + 1) % ratio`` is non-zero are emitted as exact zeros.
    Slots resolve either to a pooled tree node row (``p >= start``) or to the
    paged FP32 history (``p < start``); invalid slots are skipped, never read.
    """
    ratio = int(ratio)
    rope_dim = int(rope_dim)
    c = _c_of(ratio)
    _check(ratio in (4, 128), "ratio must be 4 or 128")

    _check(state.dim() == 3, "state must be [pages, rows, 2*c*d]")
    width = int(state.shape[2])
    _check(width % (2 * c) == 0, "state last dim must be divisible by 2*c")
    d = width // (2 * c)
    _check(d in (128, 512), "head dimension must be 128 or 512")
    _check(rope_dim > 0 and rope_dim <= d, "rope_dim must be within the head dim")
    _check(state.dtype == torch.float32, "state must be FP32")

    _check(nodes.dim() == 3 and nodes.shape[1] == 1, "nodes must be [n+1, 1, 2*c*d]")
    n = int(nodes.shape[0]) - 1
    _check(n >= 0, "nodes must have at least the sentinel row")
    _check(nodes.shape[2] == 2 * c * d, "nodes row width must be 2*c*d")
    _check(nodes.dtype == torch.float32, "nodes must be FP32")

    _check(norm.numel() == d, "norm must be [d]")
    norm_f = norm.to(torch.float32)

    _check(cos.dim() >= 1 and sin.dim() >= 1, "cos/sin must be row tables")
    _check(cos.stride(-1) == 1 and sin.stride(-1) == 1, "cos/sin must be innermost-contiguous")
    cos2 = cos if cos.dim() == 2 else cos.reshape(cos.numel() // cos.shape[-1], cos.shape[-1])
    sin2 = sin if sin.dim() == 2 else sin.reshape(sin.numel() // sin.shape[-1], sin.shape[-1])
    _check(cos2.shape[-1] >= rope_dim, "cos row is narrower than rope_dim")
    _check(sin2.shape[-1] >= rope_dim, "sin row is narrower than rope_dim")

    _check(table.dim() == 2, "table must be [requests, width]")
    _check(ancestors.dim() == 2 and ancestors.shape[0] == n, "ancestors must be [n, maxDepth]")
    max_depth = int(ancestors.shape[1])

    for name, value in (("prefix", prefix), ("depths", depths), ("requests", requests)):
        _check(value.numel() == n, f"{name} must have n entries")

    pages = int(state.shape[0]) if state.shape[0] > 0 else 0
    rows = int(state.shape[1])
    table_width = int(table.shape[1])

    prefix_i = prefix.to(torch.int64).tolist()
    depths_i = depths.to(torch.int64).tolist()
    requests_i = requests.to(torch.int64).tolist()
    ancestors_i = ancestors.to(torch.int64).tolist()

    out = torch.zeros((n, d), dtype=torch.bfloat16, device=state.device)

    for node in range(n):
        start = int(prefix_i[node])
        depth = int(depths_i[node])
        pos = start + depth
        if (pos + 1) % ratio != 0:
            continue  # exact zero row

        scores: list[torch.Tensor] = []
        kvs: list[torch.Tensor] = []
        for w in range(ratio * c):
            p = pos + 1 - ratio * c + w // c + (w % c) * ratio
            if p < 0 or p > pos:
                continue
            half = w % c
            if p >= start:
                step = p - start
                if step >= max_depth:
                    continue
                ancestor = int(ancestors_i[node][step])
                if ancestor < 0 or ancestor >= n:
                    continue
                source = nodes[ancestor + 1, 0]
                kv = source[half * d : (half + 1) * d]
                score = source[c * d + half * d : c * d + (half + 1) * d]
            else:
                col = p // rows
                if col >= table_width:
                    continue
                page = int(table[int(requests_i[node]), col].item())
                if page < 0 or page >= pages:
                    continue
                row = p % rows
                kv = state[page, row, half * d : (half + 1) * d]
                score = state[page, row, c * d + half * d : c * d + (half + 1) * d]
            scores.append(score.to(torch.float32))
            kvs.append(kv.to(torch.float32))

        if not scores:
            continue  # no reachable slot -> zero row

        scores_t = torch.stack(scores)  # [slots, d]
        kv_t = torch.stack(kvs)  # [slots, d]
        # The kernel tracks a per-dimension max/sum (elementwise vector softmax).
        max_score = scores_t.max(dim=0).values
        weight = torch.exp(scores_t - max_score)
        denom = weight.sum(dim=0)
        if not bool((denom > 0).all()):
            continue
        value = (weight * kv_t).sum(dim=0) / denom

        # RMSNorm: divisor is sqrt(mean(x^2) + eps), eps lives inside the root.
        mean_square = (value * value).sum() / d + float(eps)
        value = value / torch.sqrt(mean_square)
        value = value * norm_f

        # RoPE on the trailing rope_dim elements, interleaved pairs (i, i ^ 1).
        rope_pos = pos + 1 - ratio
        if rope_pos < 0 or rope_pos >= int(cos2.shape[0]):
            raise ValueError(f"rope position {rope_pos} is outside the cos/sin table")
        begin = d - rope_dim
        pre = value.clone()
        for i in range(rope_dim):
            j = i ^ 1
            sign = -1.0 if i % 2 == 0 else 1.0
            value[begin + i] = (
                pre[begin + i] * cos2[rope_pos, i]
                + sign * pre[begin + j] * sin2[rope_pos, i]
            )

        out[node] = value.to(torch.bfloat16)

    return out


# ---------------------------------------------------------------------------
# C. quant_lightning_indexer_tree
# ---------------------------------------------------------------------------


def quant_lightning_indexer_tree_ref(
    query,
    history,
    weights,
    qs,
    tree_key,
    tree_scale,
    ancestors,
    prefixes,
    reqs,
    block_table,
    cu,
    lengths,
    residual,
    metadata,
    topk,
    head_dim=128,
    *,
    history_scale=None,
    max_kv=None,
):
    """INT8 tree lightning indexer returning ``[n, 1, topk]`` int32 indices.

    Score for group ``g`` and compressed slot ``k`` is

        ``fp16(relu(int(q_g . k)) * 2**-10) * fp16(weights_g * qs_g)``

    summed in FP32 over the query groups and multiplied by the slot's FP16
    dequant scale.  Slots ``k < prefix // 4`` come from the paged INT8
    history; slots at/after the boundary resolve through ``ancestors`` to the
    per-node key pool.  Ties break on ascending ``k``.
    """
    del residual  # only used by the tiler, not by the score
    d = int(head_dim)
    _check(query.dim() in (2, 3), "query must be [n, G, d]")
    _check(query.dtype == torch.int8, "query must be INT8")
    _check(history.dim() == 4 and history.shape[2] == 1, "history must be [pages, rows, 1, d]")
    _check(history.shape[3] == d, "history head dim mismatch")
    _check(history.dtype == torch.int8, "history must be INT8")
    _check(history.stride(-1) == 1, "history must be innermost-contiguous")
    if history_scale is not None:
        _check(
            history_scale.shape == history.shape[:3] + (1,),
            "history_scale must be [pages, rows, 1, 1]",
        )
        _check(history_scale.dtype == torch.float16, "history_scale must be FP16")

    n = int(query.shape[0])
    groups = int(query.shape[1]) if query.dim() == 3 else 1
    _check(weights.numel() == n * groups and qs.numel() == n * groups, "weights/qs must be [n, G]")
    _check(weights.dtype == torch.float16 and qs.dtype == torch.float16, "weights/qs must be FP16")
    _check(tree_key.dim() == 2 and tree_key.shape[1] == d, "tree_key must be [nodes, d]")
    _check(tree_key.dtype == torch.int8, "tree_key must be INT8")
    _check(tree_scale.numel() == tree_key.shape[0], "tree_scale must be [nodes]")
    _check(tree_scale.dtype == torch.float16, "tree_scale must be FP16")
    nodes_count = int(tree_key.shape[0])
    _check(
        tree_key.shape[0] >= n,
        "the node key pool must cover every query row (binding contract)",
    )
    _check(ancestors.dim() == 2 and ancestors.shape[0] == n, "ancestors must be [n, width]")
    _check(
        prefixes.numel() == n and reqs.numel() == n and lengths.numel() == n,
        "per-row metadata size mismatch",
    )
    _check(cu.numel() == n + 1, "cu must be [n+1]")
    _check(block_table.dim() == 2, "block_table must be [requests, width]")
    _check(int(topk) > 0, "topk must be positive")
    _check(d == 128, "tree indexer requires head_dim == 128")

    if max_kv is None:
        max_kv = int(lengths.max().item()) * 4 if lengths.numel() else 0
    if max(1, int(max_kv) // 4) <= int(topk):
        lengths_i = lengths.to(torch.int64)
        positions = torch.arange(
            int(topk), dtype=torch.int32, device=lengths.device
        ).expand(n, -1)
        mask = positions.to(torch.int64) < lengths_i.unsqueeze(1)
        return torch.where(mask, positions, torch.full_like(positions, -1)).unsqueeze(1)

    query_i = query.to(torch.int64).reshape(n, groups, d)
    weights_f = weights.to(torch.float16).reshape(n, groups)
    query_weight = (weights_f * qs.to(torch.float16)).to(torch.float32)
    block_width = int(block_table.shape[1])
    hist_rows = int(history.shape[1])
    hist_pages = int(history.shape[0])

    ancestors_i = ancestors.to(torch.int64).tolist()
    ancestor_width = int(ancestors.shape[1])
    prefixes_i = prefixes.to(torch.int64).tolist()
    reqs_i = reqs.to(torch.int64).tolist()
    lengths_i = lengths.to(torch.int64).tolist()

    out = torch.full((n, 1, int(topk)), -1, dtype=torch.int32)

    for row in range(n):
        limit = int(lengths_i[row])
        if limit <= 0:
            continue
        start = int(prefixes_i[row])
        history_end = start // 4
        req = int(reqs_i[row])

        key_list: list[torch.Tensor] = []
        scale_list: list[float] = []
        index_list: list[int] = []
        for k in range(limit):
            if k < history_end:
                col = k // hist_rows
                if col >= block_width:
                    continue
                page = int(block_table[req, col].item())
                if page < 0 or page >= hist_pages:
                    continue
                row_in_page = k % hist_rows
                key = history[page, row_in_page, 0]
                scale = (
                    float(history_scale[page, row_in_page, 0, 0].item())
                    if history_scale is not None
                    else 1.0
                )
            else:
                step = (k + 1) * 4 - 1 - start
                if step < 0 or step >= ancestor_width:
                    continue
                node = int(ancestors_i[row][step])
                if node < 0 or node >= nodes_count:
                    continue
                key = tree_key[node]
                scale = float(tree_scale[node].item())
            key_list.append(key.to(torch.int64))
            scale_list.append(scale)
            index_list.append(k)

        if not key_list:
            continue

        keys_i = torch.stack(key_list)  # [L, d]
        # NPU BatchMatMul has no int64 dtype.  int8 products and a 128-term sum
        # are exact in fp32 (< 2^24), so use fp32 there; CPU keeps int64.
        if query_i.device.type == "npu":
            qk = torch.einsum(
                "gd,ld->gl", query_i[row].float(), keys_i.float()
            ).to(torch.int64)
        else:
            qk = torch.einsum("gd,ld->gl", query_i[row], keys_i)
        relu = torch.clamp(qk, min=0).to(torch.float32)
        score0 = (relu * (2.0**-10)).to(torch.float16).to(torch.float32)
        scale_t = torch.tensor(
            scale_list, dtype=torch.float32, device=query_i.device
        )
        scores = (score0 * query_weight[row].unsqueeze(1)).sum(dim=0) * scale_t

        order = torch.argsort(scores, descending=True, stable=True)
        take = min(int(topk), int(order.numel()))
        chosen = torch.tensor(
            index_list, dtype=torch.int32, device=query_i.device
        )[order[:take]]
        out[row, 0, :take] = chosen

    return out


# ---------------------------------------------------------------------------
# D. sparse_attn_sharedkv_tree
# ---------------------------------------------------------------------------


def _sink_row(sinks, t: int, heads: int) -> torch.Tensor:
    table = sinks.to(torch.float32)
    if table.dim() == 3:
        _check(table.shape[1] == 1, "sinks must be [heads], [n, heads] or [n, 1, heads]")
        table = table.squeeze(1)
    if table.dim() == 1:
        _check(table.numel() == heads, "sinks head count mismatch")
        return table
    _check(table.dim() == 2 and table.shape[1] == heads, "sinks shape mismatch")
    # Per-head (1 row) or per-query-row (T rows).  Row ``t`` must exist; the
    # oracle deliberately does not assume the total row count here.
    _check(
        table.shape[0] == 1 or table.shape[0] > t,
        "sinks row count mismatch",
    )
    return table[0] if table.shape[0] == 1 else table[t]


def sparse_attn_sharedkv_tree_ref(
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
    softmax_scale,
    cmp_ratio,
    ori_mask_mode,
    cmp_mask_mode,
    ori_win_left,
    ori_win_right,
    layout_q="TND",
    layout_kv="PA_ND",
    return_softmax_lse=False,
):
    """Tree sparse attention over shared KV, returning ``(out, lse)``.

    Original-KV slots are physical indices (``-1`` terminated).  Compressed
    keys are either explicit C4 logical positions (``cmp_sparse_indices``) or,
    for C128, every compressed slot below ``seqused_kv // cmp_ratio``.  A sink
    participates as an extra key with value 0, initial max ``sink`` and initial
    softmax sum ``1``.  An empty row produces zero output and zero LSE.
    """
    del metadata, ori_win_right, ori_win_left, cu_seqlens_q, cu_seqlens_ori_kv
    del cu_seqlens_cmp_kv, seqused_q, ori_block_table  # window/slots are prebuilt

    _check(layout_q == "TND", "tree attention requires layout_q == 'TND'")
    _check(layout_kv == "PA_ND", "tree attention requires layout_kv == 'PA_ND'")
    _check(int(ori_mask_mode) == 4, "tree attention requires ori_mask_mode == 4")
    _check(int(cmp_mask_mode) == 3, "tree attention requires cmp_mask_mode == 3")
    _check(q.dim() == 3, "q must be [T, N1, D]")
    _check(q.dtype in (torch.float16, torch.bfloat16), "q must be FP16 or BF16")
    d = int(q.shape[2])
    _check(d == 512, "tree attention requires head_dim == 512")
    t, n1, _ = (int(v) for v in q.shape)
    ratio = int(cmp_ratio)
    _check(ratio > 0, "cmp_ratio must be positive")
    _check(seqused_kv is not None and seqused_kv.numel() == t, "seqused_kv must be [T]")
    for name, value in (("tree_prefix", tree_prefix), ("tree_ancestors", tree_ancestors)):
        if value is not None:
            _check(int(value.shape[0]) == t, f"{name} must have T rows")
    if cmp_kv is not None:
        _check(
            cmp_kv.dim() == 4 and cmp_kv.shape[2] == 1 and cmp_kv.shape[3] == d,
            "cmp_kv must be [pages, rows, 1, d]",
        )
        _check(cmp_kv.dtype == q.dtype, "cmp_kv dtype must match q")
    if ori_kv is not None:
        _check(
            ori_kv.dim() == 4 and ori_kv.shape[2] == 1 and ori_kv.shape[3] == d,
            "ori_kv must be [pages, rows, 1, d]",
        )
        _check(ori_kv.dtype == q.dtype, "ori_kv dtype must match q")
        _check(ori_sparse_indices is not None, "ori_sparse_indices is required with ori_kv")
    if cmp_tree_kv is not None:
        _check(cmp_tree_kv.numel() % d == 0, "cmp_tree_kv row width mismatch")
        _check(
            tree_ancestors is not None and tree_prefix is not None,
            "tree inputs must be all-or-none",
        )
    if cmp_kv is not None:
        _check(
            cmp_tree_kv is not None
            and tree_ancestors is not None
            and tree_prefix is not None,
            "the tree op requires cmp_tree_kv/ancestors/prefix together with cmp_kv",
        )
        if cmp_block_table is not None:
            _check(
                int(cmp_block_table.shape[0]) == t,
                "cmp_block_table must hold one row per query token",
            )
    if cmp_sparse_indices is not None:
        _check(cmp_kv is not None, "cmp_sparse_indices requires cmp_kv")

    scale = float(softmax_scale)
    qf = q.to(torch.float32)
    ori_rows = int(ori_kv.shape[1]) if ori_kv is not None else 0
    cmp_rows = int(cmp_kv.shape[1]) if cmp_kv is not None else 0
    cmp_pages = int(cmp_kv.shape[0]) if cmp_kv is not None else 0
    tree_rows = (
        cmp_tree_kv.reshape(-1, d).to(torch.float32) if cmp_tree_kv is not None else None
    )
    node_count = int(tree_rows.shape[0]) if tree_rows is not None else 0
    ancestor_width = int(tree_ancestors.shape[1]) if tree_ancestors is not None else 0
    seqused_i = seqused_kv.to(torch.int64).tolist()

    out = torch.zeros_like(q)
    lse = torch.zeros((t, n1, 1), dtype=torch.float32, device=q.device)

    def compressed_row(k: int, prefix: int, row: int):
        """Resolve one logical compressed slot to its key row, or None."""
        if k < 0:
            return None
        if k < prefix // ratio:
            page_col = k // cmp_rows
            if cmp_block_table is None or page_col >= int(cmp_block_table.shape[1]):
                return None
            page = int(cmp_block_table[row, page_col].item())
            if page < 0 or page >= cmp_pages:
                return None
            return cmp_kv[page, k % cmp_rows, 0].to(torch.float32)
        step = (k + 1) * ratio - 1 - prefix
        if step < 0 or step >= ancestor_width:
            return None
        node = int(tree_ancestors[row, step].item())
        if node < 0 or node >= node_count:
            return None
        return tree_rows[node]

    for row in range(t):
        keys: list[torch.Tensor] = []
        if ori_kv is not None and ori_sparse_indices is not None:
            width = int(ori_sparse_indices.shape[-1])
            cap = min(width, int(seqused_i[row]) if seqused_i else width)
            for col in range(cap):
                slot = int(ori_sparse_indices[row, 0, col].item())
                if slot < 0:
                    break
                page = slot // ori_rows
                if page >= int(ori_kv.shape[0]):
                    break
                keys.append(ori_kv[page, slot % ori_rows, 0].to(torch.float32))

        prefix = int(tree_prefix[row].item()) if tree_prefix is not None else 0
        limit = int(seqused_i[row]) // ratio if ratio > 0 else 0
        if cmp_kv is not None:
            if cmp_sparse_indices is not None:
                for col in range(int(cmp_sparse_indices.shape[-1])):
                    k = int(cmp_sparse_indices[row, 0, col].item())
                    if k < 0 or k >= limit:
                        continue
                    resolved = compressed_row(k, prefix, row)
                    if resolved is not None:
                        keys.append(resolved)
            else:
                for k in range(limit):
                    resolved = compressed_row(k, prefix, row)
                    if resolved is not None:
                        keys.append(resolved)

        if not keys:
            # Empty row: zero output and zero LSE, sink is not applied.
            continue

        key_t = torch.stack(keys)  # [L, d] FP32
        logits = (qf[row] @ key_t.t()) * scale  # [N1, L]
        row_max = logits.max(dim=1).values  # [N1]
        if sinks is not None:
            sink = _sink_row(sinks, row, n1)
            max_all = torch.maximum(row_max, sink)
        else:
            sink = None
            max_all = row_max
        weight = torch.exp(logits - max_all.unsqueeze(1))
        total = weight.sum(dim=1)
        if sink is not None:
            total = total + torch.exp(sink - max_all)
            # Sink carries value 0, so it only inflates the denominator.

        probs = weight.to(q.dtype).to(torch.float32)
        accum = probs @ key_t
        out[row] = (accum / total.unsqueeze(1)).to(q.dtype)
        lse[row, :, 0] = torch.log(total) + max_all

    if not return_softmax_lse:
        return out, torch.empty(0, dtype=torch.float32, device=q.device)
    return out, lse
