"""Golden tests for the independent CPU reference layer of the DSV4 DSpark tree ops.

The expected values below are hand-computed from the reference semantics (they are
never produced by calling the reference twice).  Every test is CPU-only: no
triton, no torch_npu, no sgl_kernel_npu imports.

Semantics source: vllm-ascend @ 8947f1f9f
  csrc/attention/{compressor_tree_project,compressor_tree,
                   quant_lightning_indexer_tree,sparse_attn_sharedkv_tree}/
"""

import math

import torch

from tree_fixtures import (
    make_attention_case,
    make_compressor_case,
    make_indexer_case,
    make_project_case,
    make_tree,
)
from tree_reference import (
    compressor_tree_project_ref,
    compressor_tree_ref,
    quant_lightning_indexer_tree_ref,
    sparse_attn_sharedkv_tree_ref,
)

# Tolerances required by the wave-1 contract.
FP32_TOL = 2e-3
BF16_TOL = 1e-2
ATTN_TOL = 2e-2
LSE_TOL = 5e-2


def assert_close(actual, expected, tol):
    torch.testing.assert_close(
        actual.to(torch.float32), expected.to(torch.float32), rtol=tol, atol=tol
    )


# ---------------------------------------------------------------------------
# helpers shared by the hand-computed compressor / indexer / attention cases
# ---------------------------------------------------------------------------


def _state(pages, rows, d, c, kv0, kv1=None, score=0.0):
    state = torch.zeros(pages, rows, 2 * c * d, dtype=torch.float32)
    state[..., :d] = kv0
    if c > 1 and kv1 is not None:
        state[..., d : c * d] = kv1
    state[..., c * d :] = score
    return state


def _node_rows(n, d, c, kv0, kv1=None, score=0.0, row0=0.0):
    nodes = torch.zeros(n + 1, 1, 2 * c * d, dtype=torch.float32)
    nodes[0].fill_(row0)
    nodes[1:, 0, :d] = kv0
    if c > 1 and kv1 is not None:
        nodes[1:, 0, d : c * d] = kv1
    nodes[1:, 0, c * d :] = score
    return nodes


def _run_compressor(
    *,
    state,
    nodes,
    table,
    prefix,
    depths,
    ancestors,
    ratio,
    d,
    eps=0.0,
    norm=1.0,
    rope_dim=64,
    cos=None,
    sin=None,
    requests=None,
):
    n = int(depths.numel())
    if requests is None:
        requests = torch.zeros(n, dtype=torch.int64)
    if cos is None or sin is None:
        max_pos = int((prefix.max() + depths.max()).item()) + 2
        cos = torch.ones(max_pos, rope_dim, dtype=torch.float32)
        sin = torch.zeros(max_pos, rope_dim, dtype=torch.float32)
    norm_w = torch.full((d,), float(norm), dtype=torch.float32)
    return compressor_tree_ref(
        state,
        nodes,
        norm_w,
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
    )


def _indexer_bundle(
    *,
    query,
    weights,
    qs,
    history,
    history_scale,
    tree_key,
    tree_scale,
    ancestors,
    prefixes,
    reqs,
    block_table,
    lengths,
    residual,
    topk,
    max_kv=None,
):
    n = query.shape[0]
    return dict(
        query=query,
        history=history,
        history_scale=history_scale,
        weights=weights,
        qs=qs,
        tree_key=tree_key,
        tree_scale=tree_scale,
        ancestors=ancestors,
        prefixes=prefixes,
        reqs=reqs,
        block_table=block_table,
        cu=torch.arange(n + 1, dtype=torch.int32),
        lengths=lengths,
        residual=residual,
        metadata=torch.zeros(1024, dtype=torch.int32),
        topk=topk,
        max_kv=max_kv,
    )


def _call_indexer(bundle):
    return quant_lightning_indexer_tree_ref(
        bundle["query"],
        bundle["history"],
        bundle["weights"],
        bundle["qs"],
        bundle["tree_key"],
        bundle["tree_scale"],
        bundle["ancestors"],
        bundle["prefixes"],
        bundle["reqs"],
        bundle["block_table"],
        bundle["cu"],
        bundle["lengths"],
        bundle["residual"],
        bundle["metadata"],
        bundle["topk"],
        head_dim=128,
        history_scale=bundle["history_scale"],
        max_kv=bundle["max_kv"],
    )


def _two_row_indexer(
    *,
    hist_value,
    tree_value,
    hist_scale=1.0,
    tree_scale=1.0,
    ancestors_self=1,
    topk=1,
    hist_row1=0,
    max_kv=None,
):
    """Row 0: prefix 6, depth 0 (one valid key: history k=0).
    Row 1: prefix 6, depth 1 (history k=0, tree k=1 from ancestors[1, 1]).
    """
    d = 128
    query = torch.zeros(2, 1, d, dtype=torch.int8)
    query[:, 0, 0] = 1
    weights = torch.ones(2, 1, dtype=torch.float16)
    qs = torch.ones(2, 1, dtype=torch.float16)

    history = torch.zeros(2, 4, 1, d, dtype=torch.int8)
    history[0, 0, 0, 0] = int(hist_value)
    history[0, 1, 0, 0] = int(hist_row1)
    hscale = torch.full((2, 4, 1, 1), float(hist_scale), dtype=torch.float16)

    tree_key = torch.zeros(2, d, dtype=torch.int8)
    tree_key[1, 0] = int(tree_value)
    tscale = torch.tensor([1.0, float(tree_scale)], dtype=torch.float16)

    ancestors = torch.tensor([[0, -1], [0, ancestors_self]], dtype=torch.int64)
    return _indexer_bundle(
        query=query,
        weights=weights,
        qs=qs,
        history=history,
        history_scale=hscale,
        tree_key=tree_key,
        tree_scale=tscale,
        ancestors=ancestors,
        prefixes=torch.tensor([6, 6], dtype=torch.int32),
        reqs=torch.tensor([0, 0], dtype=torch.int64),
        block_table=torch.tensor([[0, 1]], dtype=torch.int32),
        lengths=torch.tensor([1, 2], dtype=torch.int32),
        residual=torch.tensor([3, 0], dtype=torch.int32),
        topk=topk,
        max_kv=max_kv,
    )


def _attn_call(case, *, return_softmax_lse=True):
    return sparse_attn_sharedkv_tree_ref(
        case["q"],
        case["ori_kv"],
        case["cmp_kv"],
        case["ori_sparse_indices"],
        case["cmp_sparse_indices"],
        case["ori_block_table"],
        case["cmp_block_table"],
        case["cu_seqlens_q"],
        case["cu_seqlens_ori_kv"],
        case["cu_seqlens_cmp_kv"],
        case["seqused_q"],
        case["seqused_kv"],
        case["sinks"],
        case["metadata"],
        case["cmp_tree_kv"],
        case["tree_ancestors"],
        case["tree_prefix"],
        case["softmax_scale"],
        case["cmp_ratio"],
        case["ori_mask_mode"],
        case["cmp_mask_mode"],
        case["ori_win_left"],
        case["ori_win_right"],
        layout_q=case["layout_q"],
        layout_kv=case["layout_kv"],
        return_softmax_lse=return_softmax_lse,
    )


def _attn_case(
    *,
    q,
    ori_slots,
    ori_rows,
    ori_pages=1,
    cmp_kv=None,
    cmp_slots=None,
    cmp_tree_kv=None,
    ancestors=None,
    prefix=None,
    seqused=None,
    sinks=None,
    ratio=4,
    scale=1.0 / 1024.0,
    block_rows=4,
    cmp_block_table=None,
):
    t, n1, d = q.shape
    if ori_slots is None:
        ori_slots = torch.full((t, 1, 4), -1, dtype=torch.int32)
    if prefix is None:
        prefix = torch.zeros(t, dtype=torch.int32)
    seqused_q = torch.empty(0, dtype=torch.int32)
    return dict(
        q=q,
        ori_kv=ori_kv_pool(ori_slots, ori_rows, d, ori_pages, q.dtype),
        cmp_kv=cmp_kv,
        ori_sparse_indices=ori_slots,
        cmp_sparse_indices=cmp_slots,
        ori_block_table=torch.zeros((t, 1), dtype=torch.int32),
        cmp_block_table=(
            cmp_block_table
            if cmp_block_table is not None
            else torch.zeros((t, 1), dtype=torch.int32)
        ),
        cu_seqlens_q=torch.arange(t + 1, dtype=torch.int32),
        cu_seqlens_ori_kv=torch.cat(
            (torch.zeros(1, dtype=torch.int32), torch.cumsum(seqused, 0, dtype=torch.int32))
        ),
        cu_seqlens_cmp_kv=torch.cat(
            (
                torch.zeros(1, dtype=torch.int32),
                torch.cumsum(seqused // ratio, 0, dtype=torch.int32),
            )
        ),
        seqused_q=seqused_q,
        seqused_kv=seqused.to(torch.int32),
        sinks=sinks,
        metadata=torch.zeros(1024, dtype=torch.int32),
        cmp_tree_kv=cmp_tree_kv,
        tree_ancestors=ancestors,
        tree_prefix=prefix,
        softmax_scale=scale,
        cmp_ratio=ratio,
        ori_mask_mode=4,
        cmp_mask_mode=3,
        ori_win_left=block_rows - 1,
        ori_win_right=0,
        layout_q="TND",
        layout_kv="PA_ND",
    )


def ori_kv_pool(ori_slots, slots, d, pages, dtype):
    pool = torch.zeros(pages, slots, 1, d, dtype=dtype)
    return pool


# ---------------------------------------------------------------------------
# A. compressor_tree_project
# ---------------------------------------------------------------------------


def test_project_one_hot_weights_exact_and_row0_preserved():
    case = make_project_case(seed=7, ratio=4, head_dim=128, hidden=2, n=2)
    # One-hot projection: column j reads x[:, j % hidden].
    d = case["head_dim"]
    c = 2
    w = torch.zeros(c * d, case["hidden"], dtype=torch.bfloat16)
    for j in range(c * d):
        w[j, j % case["hidden"]] = 1.0
    case["wkv"] = w
    case["wgate"] = w.clone()
    case["ape"].zero_()

    compressor_tree_project_ref(
        case["x"],
        case["wkv"],
        case["wgate"],
        case["nodes"],
        case["ape"],
        case["positions"],
        case["cu"],
        case["ratio"],
        case["head_dim"],
    )

    xf = case["x"].to(torch.float32)
    for i in range(case["x"].shape[0]):
        kv = case["nodes"][i + 1, 0, : c * d]
        expected = xf[i][torch.arange(c * d) % case["hidden"]]
        assert_close(kv, expected, FP32_TOL)
        # wgate is the same one-hot matrix, APE is zero.
        assert_close(case["nodes"][i + 1, 0, c * d :], expected, FP32_TOL)
    # Reserved sentinel row must be untouched.
    assert torch.equal(case["nodes"][0], torch.full_like(case["nodes"][0], 7.0))


def test_project_ape_applied_only_to_score():
    case = make_project_case(seed=3, ratio=4, head_dim=128, hidden=2, n=3)
    c = 2
    d = case["head_dim"]
    case["wkv"].zero_()
    case["wgate"].zero_()
    positions = torch.tensor([0, 1, 3], dtype=torch.int32)
    case["positions"] = positions
    ape = torch.arange(case["ratio"] * c * d, dtype=torch.float32).reshape(
        case["ratio"], c * d
    )
    case["ape"].copy_(ape)

    compressor_tree_project_ref(
        case["x"],
        case["wkv"],
        case["wgate"],
        case["nodes"],
        case["ape"],
        case["positions"],
        case["cu"],
        case["ratio"],
        case["head_dim"],
    )

    for i, residue in enumerate(positions.tolist()):
        assert torch.equal(
            case["nodes"][i + 1, 0, : c * d], torch.zeros(c * d, dtype=torch.float32)
        )
        assert_close(case["nodes"][i + 1, 0, c * d :], ape[residue], FP32_TOL)


# ---------------------------------------------------------------------------
# B. compressor_tree
# ---------------------------------------------------------------------------


def test_compressor_single_valid_slot_known_probability():
    d, ratio, c = 128, 4, 2
    # Root at position 7.  The eight C4 slots touch p = 0..7 with rows=1.
    state = _state(7, 1, d, c, kv0=2.0, kv1=-100.0)
    nodes = _node_rows(1, d, c, kv0=-100.0, kv1=50.0)
    table = torch.full((1, 7), -1, dtype=torch.int32)
    table[0, 0] = 0  # only p == 0 is a valid page slot
    ancestors = torch.tensor([[-1]], dtype=torch.int64)  # node slot masked

    out = _run_compressor(
        state=state,
        nodes=nodes,
        table=table,
        prefix=torch.tensor([7], dtype=torch.int32),
        depths=torch.tensor([0], dtype=torch.int64),
        ancestors=ancestors,
        ratio=ratio,
        d=d,
    )
    # Only the +2 half-0 page slot is visible -> RMSNorm(2) == 1.0.
    assert out.dtype == torch.bfloat16
    assert_close(out, torch.ones(1, d), BF16_TOL)


def test_compressor_c4_half_selection_poisoned():
    d, ratio, c = 128, 4, 2
    state = _state(7, 1, d, c, kv0=1.0, kv1=-1.0)
    nodes = _node_rows(1, d, c, kv0=3.0, kv1=-3.0)
    table = torch.arange(7, dtype=torch.int32).reshape(1, 7)
    ancestors = torch.tensor([[0]], dtype=torch.int64)

    out = _run_compressor(
        state=state,
        nodes=nodes,
        table=table,
        prefix=torch.tensor([7], dtype=torch.int32),
        depths=torch.tensor([0], dtype=torch.int64),
        ancestors=ancestors,
        ratio=ratio,
        d=d,
    )
    # 4 slots read half 0 (+1), 3 slots + the node read half 1 (-1), node is -3:
    #   (4 * 1 + 3 * -1 + -3) / 8 = -0.25 -> RMSNorm -> -1.0
    # Swapping the halves would flip the sign to +1.0.
    assert_close(out, torch.full((1, d), -1.0), BF16_TOL)

    state = _state(7, 1, d, c, kv0=-1.0, kv1=1.0)
    nodes = _node_rows(1, d, c, kv0=-3.0, kv1=3.0)
    out = _run_compressor(
        state=state,
        nodes=nodes,
        table=torch.arange(7, dtype=torch.int32).reshape(1, 7),
        prefix=torch.tensor([7], dtype=torch.int32),
        depths=torch.tensor([0], dtype=torch.int64),
        ancestors=torch.tensor([[0]], dtype=torch.int64),
        ratio=ratio,
        d=d,
    )
    assert_close(out, torch.full((1, d), 1.0), BF16_TOL)


def test_compressor_weighted_slots_use_node_plus_one_row():
    d, ratio, c = 128, 4, 2
    # Only p == 0 (history, half 0) and p == 7 (node, half 1) are reachable.
    state = _state(1, 1, d, c, kv0=-2.0, kv1=5.0, score=0.0)
    nodes = _node_rows(1, d, c, kv0=7.0, kv1=2.0, score=math.log(3.0))
    table = torch.full((1, 7), -1, dtype=torch.int32)
    table[0, 0] = 0
    out = _run_compressor(
        state=state,
        nodes=nodes,
        table=table,
        prefix=torch.tensor([7], dtype=torch.int32),
        depths=torch.tensor([0], dtype=torch.int64),
        ancestors=torch.tensor([[0]], dtype=torch.int64),
        ratio=ratio,
        d=d,
    )
    # weights 3 (node) : 1 (history) -> (3 * 2 + 1 * -2) / 4 == 1.0.
    # Reading nodes[a] instead of nodes[a + 1] would hit the zero sentinel.
    assert_close(out, torch.full((1, d), 1.0), BF16_TOL)


def test_compressor_non_emit_row_is_exact_zero():
    d, ratio, c = 128, 4, 2
    state = _state(7, 1, d, c, kv0=1.0, kv1=1.0)
    nodes = _node_rows(2, d, c, kv0=1.0, kv1=1.0)
    table = torch.arange(7, dtype=torch.int32).reshape(1, 7)
    out = _run_compressor(
        state=state,
        nodes=nodes,
        table=table,
        prefix=torch.tensor([7, 7], dtype=torch.int32),
        depths=torch.tensor([0, 1], dtype=torch.int64),
        ancestors=torch.tensor([[0, -1], [0, 1]], dtype=torch.int64),
        ratio=ratio,
        d=d,
    )
    assert out.shape == (2, d)
    assert torch.equal(out[1], torch.zeros(d, dtype=torch.bfloat16))
    assert out[0].abs().sum() > 0


def test_compressor_invalid_ancestor_masked():
    d, ratio, c = 128, 4, 2
    state = _state(1, 1, d, c, kv0=1.0, kv1=1.0)
    nodes = _node_rows(1, d, c, kv0=3.0, kv1=-3.0)
    table = torch.full((1, 7), -1, dtype=torch.int32)  # no history slot at all
    zero = torch.zeros(1, d, dtype=torch.bfloat16)

    for bad in (-1, 5):
        out = _run_compressor(
            state=state,
            nodes=nodes,
            table=table,
            prefix=torch.tensor([7], dtype=torch.int32),
            depths=torch.tensor([0], dtype=torch.int64),
            ancestors=torch.tensor([[bad]], dtype=torch.int64),
            ratio=ratio,
            d=d,
        )
        assert torch.equal(out, zero)

    out = _run_compressor(
        state=state,
        nodes=nodes,
        table=table,
        prefix=torch.tensor([7], dtype=torch.int32),
        depths=torch.tensor([0], dtype=torch.int64),
        ancestors=torch.tensor([[0]], dtype=torch.int64),
        ratio=ratio,
        d=d,
    )
    # p == 7 is the node slot, half = 1 -> -3 -> RMSNorm -> -1.
    assert_close(out, torch.full((1, d), -1.0), BF16_TOL)


def test_compressor_negative_positions_and_bad_page_masked():
    d, ratio, c = 128, 4, 2
    # prefix 3 -> root at p == 3; slots p = -4, 0, -3, 3, -2, 2, -1, 1.
    state = _state(1, 1, d, c, kv0=50.0, kv1=50.0)
    nodes = _node_rows(1, d, c, kv0=100.0, kv1=-2.0)
    table = torch.full((1, 8), -1, dtype=torch.int32)
    table[0, -1] = 0  # trap: an implementation without p < 0 masking wraps here
    out = _run_compressor(
        state=state,
        nodes=nodes,
        table=table,
        prefix=torch.tensor([3], dtype=torch.int32),
        depths=torch.tensor([0], dtype=torch.int64),
        ancestors=torch.tensor([[0]], dtype=torch.int64),
        ratio=ratio,
        d=d,
    )
    # p < 0 slots and the paged slots (cols 0,1,2 map to -1) are masked;
    # only the node slot p == 3 (half 1, kv = -2) survives.
    assert_close(out, torch.full((1, d), -1.0), BF16_TOL)


def test_compressor_rope_hand_computed():
    d, ratio, c = 128, 4, 2
    rope_dim = 64
    state = _state(1, 1, d, c, kv0=1.0, kv1=1.0)
    nodes = _node_rows(1, d, c, kv0=1.0, kv1=1.0)
    table = torch.full((1, 7), -1, dtype=torch.int32)
    rows = 16
    idx = torch.arange(rope_dim, dtype=torch.float32)
    cos = 1.0 + 0.01 * idx
    sin = 0.5 + 0.02 * idx
    cos = cos.unsqueeze(0).repeat(rows, 1)
    sin = sin.unsqueeze(0).repeat(rows, 1)
    out = _run_compressor(
        state=state,
        nodes=nodes,
        table=table,
        prefix=torch.tensor([7], dtype=torch.int32),
        depths=torch.tensor([0], dtype=torch.int64),
        ancestors=torch.tensor([[0]], dtype=torch.int64),
        ratio=ratio,
        d=d,
        rope_dim=rope_dim,
        cos=cos,
        sin=sin,
    )
    # value == 1 everywhere; ropePos = 7 + 1 - 4 = 4.
    # The paired element supplies value[i ^ 1], the sin/cos index stays i.
    expected = torch.ones(d, dtype=torch.float32)
    begin = d - rope_dim
    for i in range(rope_dim):
        sign = -1.0 if i % 2 == 0 else 1.0
        expected[begin + i] = cos[4, i] + sign * sin[4, i]
    assert_close(out[0], expected, BF16_TOL)
    # Head that is outside the rope window is passed through unchanged.
    assert_close(out[0, :begin], torch.ones(begin), BF16_TOL)


def test_compressor_rmsnorm_eps_placement():
    d, ratio, c = 128, 4, 2
    state = _state(1, 1, d, c, kv0=1.0, kv1=1.0)
    nodes = _node_rows(1, d, c, kv0=1.0, kv1=2.0)
    table = torch.full((1, 7), -1, dtype=torch.int32)

    def run(eps):
        return _run_compressor(
            state=state,
            nodes=nodes,
            table=table,
            prefix=torch.tensor([7], dtype=torch.int32),
            depths=torch.tensor([0], dtype=torch.int64),
            ancestors=torch.tensor([[0]], dtype=torch.int64),
            ratio=ratio,
            d=d,
            eps=eps,
            norm=3.0,
        )

    # value = 2; mean = 4; sqrt(mean + eps) is the divisor (eps inside the root).
    out = run(1.0)
    assert_close(out, torch.full((1, d), 6.0 / math.sqrt(5.0)), BF16_TOL)
    out = run(0.0)
    assert_close(out, torch.full((1, d), 3.0), BF16_TOL)


def test_compressor_fixture_poison_halves_is_loud():
    case = make_compressor_case(
        seed=1,
        ratio=4,
        head_dim=128,
        rows=4,
        budget=0,
        depth=0,
        prefixes=[7],
        poison_halves=True,
        eps=0.0,
    )
    args = dict(case["args"])
    args["cos"] = torch.ones_like(args["cos"])
    args["sin"] = torch.zeros_like(args["sin"])
    out = compressor_tree_ref(**args)
    # 4 half-0 slots at +1, 3 half-1 history slots at -1 and the half-1 node
    # row at -3: (4 - 3 - 3) / 8 = -0.25 -> RMSNorm -> -1.0.  A wrong half
    # selection flips the sign.
    assert_close(out, torch.full((1, 128), -1.0), BF16_TOL)


def test_compressor_fixture_padded_strides_and_dtype():
    case = make_compressor_case(
        seed=11, ratio=4, head_dim=128, budget=2, depth=2, cos_pad=8, page_pad=4
    )
    out = compressor_tree_ref(**case["args"])
    assert out.shape == (case["n"], 128)
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out.to(torch.float32)).all()

    # Garbage in the alignment padding must never be read.
    for tensor in case["padding_views"]:
        tensor[..., -1].fill_(float("nan"))
    out2 = compressor_tree_ref(**case["args"])
    assert torch.equal(out, out2)


# ---------------------------------------------------------------------------
# C. quant_lightning_indexer_tree
# ---------------------------------------------------------------------------


def test_indexer_int8_scale_and_history_vs_tree_address():
    bundle = _two_row_indexer(hist_value=1, tree_value=3, topk=1, hist_row1=100)
    out = _call_indexer(bundle)
    assert out.shape == (2, 1, 1)
    assert out.dtype == torch.int32
    assert out[0, 0, 0].item() == 0
    assert out[1, 0, 0].item() == 1  # tree key beats the 1/1024 history score

    # k == 1 must address the tree node (3/1024), not paged history row 1 (100).
    bundle = _two_row_indexer(
        hist_value=1,
        tree_value=3,
        hist_scale=8.0,
        hist_row1=100,
        topk=2,
        max_kv=4096,
    )
    out = _call_indexer(bundle)
    assert out[1].tolist() == [[0, 1]]  # 8/1024 (history) > 3/1024 (tree)

    bundle = _two_row_indexer(
        hist_value=1, tree_value=3, topk=2, max_kv=4096
    )
    out = _call_indexer(bundle)
    assert out[1].tolist() == [[1, 0]]


def test_indexer_fp16_score_rounding_and_tie_break():
    d = 128
    query = torch.zeros(2, 1, d, dtype=torch.int8)
    query[:, 0, :17] = 1
    weights = torch.ones(2, 1, dtype=torch.float16)
    qs = torch.ones(2, 1, dtype=torch.float16)
    history = torch.zeros(2, 4, 1, d, dtype=torch.int8)
    history[0, 0, 0, :16] = 127
    history[0, 0, 0, 16] = 16  # dot 2048 -> fp16 2.0
    hscale = torch.ones(2, 4, 1, 1, dtype=torch.float16)
    tree_key = torch.zeros(2, d, dtype=torch.int8)
    tree_key[1, :16] = 127
    tree_key[1, 16] = 17  # dot 2049 -> 2049/1024 ties to 2.0 in fp16
    tscale = torch.ones(2, dtype=torch.float16)
    bundle = _indexer_bundle(
        query=query,
        weights=weights,
        qs=qs,
        history=history,
        history_scale=hscale,
        tree_key=tree_key,
        tree_scale=tscale,
        ancestors=torch.tensor([[0, -1], [0, 1]], dtype=torch.int64),
        prefixes=torch.tensor([6, 6], dtype=torch.int32),
        reqs=torch.tensor([0, 0], dtype=torch.int64),
        block_table=torch.tensor([[0, 1]], dtype=torch.int32),
        lengths=torch.tensor([1, 2], dtype=torch.int32),
        residual=torch.tensor([3, 0], dtype=torch.int32),
        topk=1,
    )
    out = _call_indexer(bundle)
    # Both keys round to 2.0 -> deterministic tie-break keeps the lower index.
    assert out[1, 0, 0].item() == 0


def test_indexer_relu_clamps_negative_dots():
    d = 128
    query = torch.zeros(2, 1, d, dtype=torch.int8)
    query[:, 0, :17] = 1
    weights = torch.ones(2, 1, dtype=torch.float16)
    qs = torch.ones(2, 1, dtype=torch.float16)
    history = torch.zeros(2, 4, 1, d, dtype=torch.int8)
    history[0, 0, 0, :16] = -127
    history[0, 0, 0, 16] = -16  # dot -2048
    history_scale = torch.ones(2, 4, 1, 1, dtype=torch.float16)
    tree_key = torch.zeros(2, d, dtype=torch.int8)
    tree_key[1, :8] = -127
    tree_key[1, 8] = -8  # dot -1024
    tree_scale = torch.ones(2, dtype=torch.float16)
    bundle = _indexer_bundle(
        query=query,
        weights=weights,
        qs=qs,
        history=history,
        history_scale=history_scale,
        tree_key=tree_key,
        tree_scale=tree_scale,
        ancestors=torch.tensor([[0, -1], [0, 1]], dtype=torch.int64),
        prefixes=torch.tensor([6, 6], dtype=torch.int32),
        reqs=torch.tensor([0, 0], dtype=torch.int64),
        block_table=torch.tensor([[0, 1]], dtype=torch.int32),
        lengths=torch.tensor([1, 2], dtype=torch.int32),
        residual=torch.tensor([3, 0], dtype=torch.int32),
        topk=2,
        max_kv=4096,
    )
    out = _call_indexer(bundle)
    # relu clamps both negative dots to 0 -> exact tie -> lower index first.
    # Without relu the -1/1024 tree score would beat the -2/1024 history score.
    assert out[1].tolist() == [[0, 1]]


def test_indexer_invalid_ancestor_masked():
    for bad in (-1, 7):
        bundle = _two_row_indexer(
            hist_value=1, tree_value=100, ancestors_self=bad, topk=4, max_kv=4096
        )
        out = _call_indexer(bundle)
        assert out[0].tolist() == [[0, -1, -1, -1]]
        # k == 1 is invalid and must not leak its large score.
        assert out[1].tolist() == [[0, -1, -1, -1]]


def test_indexer_select_all_and_explicit_capacity():
    case = make_indexer_case(seed=5, topk=4, budget=3, depth=2, prefixes=[5])
    out = quant_lightning_indexer_tree_ref(**case["args"])
    lengths = case["lengths"].tolist()
    for row, length in enumerate(lengths):
        expected = [i if i < length else -1 for i in range(4)]
        assert out[row, 0].tolist() == expected

    # Explicitly large capacity forces the ranked path.
    args = dict(case["args"])
    args["max_kv"] = 4096
    ranked = quant_lightning_indexer_tree_ref(**args)
    assert ranked.shape == (4, 1, 4)
    assert ranked[3, 0].tolist()[0] == 1  # depth-2 row picks its tree key first


# ---------------------------------------------------------------------------
# D. sparse_attn_sharedkv_tree
# ---------------------------------------------------------------------------


def test_attention_two_key_softmax_no_sink():
    d = 512
    q = torch.ones(1, 1, d, dtype=torch.float16)
    case = _attn_case(
        q=q,
        ori_slots=torch.tensor([[[0, 1, -1, -1]]], dtype=torch.int32),
        ori_rows=4,
        ori_pages=1,
        seqused=torch.tensor([2], dtype=torch.int32),
        prefix=torch.tensor([0], dtype=torch.int32),
    )
    case["ori_kv"][0, 0, 0].fill_(2.0)
    case["ori_kv"][0, 1, 0].fill_(-2.0)

    out, lse = _attn_call(case)
    # q . k = +/-1024, scale 1/1024 -> logits +/-1.
    p0 = 1.0 / (1.0 + math.exp(-2.0))
    expected = 2.0 * p0 - 2.0 * (1.0 - p0)
    assert_close(out, torch.full((1, 1, d), expected), ATTN_TOL)
    assert out.dtype == torch.float16
    assert lse.shape == (1, 1, 1)
    # lse == log(sum(exp(logits))) in natural log.
    assert_close(lse, torch.tensor([[[math.log(math.exp(1.0) + math.exp(-1.0))]]]), LSE_TOL)


def test_attention_sink_weight():
    d = 512
    q = torch.ones(1, 1, d, dtype=torch.float16)
    base = dict(
        q=q,
        ori_slots=torch.tensor([[[0, -1, -1, -1]]], dtype=torch.int32),
        ori_rows=4,
        ori_pages=1,
        seqused=torch.tensor([1], dtype=torch.int32),
        prefix=torch.tensor([0], dtype=torch.int32),
    )
    case = _attn_case(**base)
    case["ori_kv"][0, 0, 0].fill_(2.0)
    out, lse = _attn_call(case)
    assert_close(out, torch.full((1, 1, d), 2.0), ATTN_TOL)  # single key -> value
    assert_close(lse, torch.tensor([[[1.0]]]), LSE_TOL)

    case = _attn_case(**base, sinks=torch.tensor([1.0], dtype=torch.float32))
    case["ori_kv"][0, 0, 0].fill_(2.0)
    out, lse = _attn_call(case)
    # logits: key 1.0, sink 1.0 -> weights 1:1 -> value weight 0 for the sink.
    assert_close(out, torch.full((1, 1, d), 1.0), ATTN_TOL)
    assert_close(lse, torch.tensor([[[math.log(2.0) + 1.0]]]), LSE_TOL)

    case = _attn_case(**base, sinks=torch.tensor([-1.0], dtype=torch.float32))
    case["ori_kv"][0, 0, 0].fill_(2.0)
    out, _ = _attn_call(case)
    assert_close(
        out, torch.full((1, 1, d), 2.0 / (1.0 + math.exp(-2.0))), ATTN_TOL
    )


def test_attention_c4_slot_construction():
    d = 512
    q = torch.ones(2, 1, d, dtype=torch.float16)
    cmp_kv = torch.zeros(1, 4, 1, d, dtype=torch.float16)
    cmp_kv[0, 0, 0].fill_(1.0)  # compressed history k == 0
    cmp_kv[0, 1, 0].fill_(5.0)  # poison: never the tree key
    cmp_tree = torch.zeros(2, d, dtype=torch.float16)
    cmp_tree[0].fill_(3.0)
    cmp_tree[1].fill_(-1.0)  # node 1 -> tree key for k == 1
    case = _attn_case(
        q=q,
        ori_slots=torch.full((2, 1, 4), -1, dtype=torch.int32),
        ori_rows=4,
        cmp_kv=cmp_kv,
        cmp_slots=torch.tensor([[[0, -1]], [[1, 0]]], dtype=torch.int32),
        cmp_tree_kv=cmp_tree,
        ancestors=torch.tensor([[0, -1], [0, 1]], dtype=torch.int64),
        prefix=torch.tensor([6, 6], dtype=torch.int32),
        seqused=torch.tensor([7, 8], dtype=torch.int32),
    )
    out, lse = _attn_call(case)
    # Row 0: single key k=0 -> +1.
    assert_close(out[0], torch.full((1, d), 1.0), ATTN_TOL)
    # Row 1: keys [-1 (k=1 tree), +1 (k=0 history)] with logits [-0.5, 0.5].
    w_hi = 1.0
    w_lo = math.exp(-1.0)
    expected = (w_hi * 1.0 + w_lo * -1.0) / (w_hi + w_lo)
    assert_close(out[1], torch.full((1, d), expected), ATTN_TOL)
    assert_close(
        lse[1, 0, 0],
        torch.tensor(math.log(w_hi + w_lo) + 0.5, dtype=torch.float32),
        LSE_TOL,
    )


def test_attention_c128_dense_slots():
    d = 512
    q = torch.ones(2, 1, d, dtype=torch.float16)
    cmp_kv = torch.zeros(2, 4, 1, d, dtype=torch.float16)
    cmp_kv[0, 0, 0].fill_(1.0)  # k == 0 history
    cmp_kv[0, 1, 0].fill_(7.0)  # poison
    cmp_kv[1, 0, 0].fill_(7.0)  # poison
    cmp_tree = torch.zeros(2, d, dtype=torch.float16)
    cmp_tree[0].fill_(-3.0)  # k == 1 resolves to root node 0
    case = _attn_case(
        q=q,
        ori_slots=torch.full((2, 1, 4), -1, dtype=torch.int32),
        ori_rows=4,
        cmp_kv=cmp_kv,
        cmp_slots=None,
        cmp_tree_kv=cmp_tree,
        ancestors=torch.tensor([[0, -1], [0, 1]], dtype=torch.int64),
        prefix=torch.tensor([255, 255], dtype=torch.int32),
        seqused=torch.tensor([256, 257], dtype=torch.int32),
        ratio=128,
        cmp_block_table=torch.zeros((2, 2), dtype=torch.int32),
    )
    out, lse = _attn_call(case)
    w_hi = 1.0
    w_lo = math.exp(-2.0)
    expected = (w_hi * 1.0 + w_lo * -3.0) / (w_hi + w_lo)
    assert_close(out[0], torch.full((1, d), expected), ATTN_TOL)
    assert_close(out[1], torch.full((1, d), expected), ATTN_TOL)
    assert_close(
        lse[0, 0, 0], torch.tensor(math.log(w_hi + w_lo) + 0.5), LSE_TOL
    )


def test_attention_empty_row_is_zero():
    d = 512
    q = torch.ones(1, 1, d, dtype=torch.bfloat16)
    case = _attn_case(
        q=q,
        ori_slots=torch.full((1, 1, 4), -1, dtype=torch.int32),
        ori_rows=4,
        seqused=torch.tensor([4], dtype=torch.int32),
        prefix=torch.tensor([4], dtype=torch.int32),
        sinks=torch.tensor([3.0], dtype=torch.float32),
    )
    out, lse = _attn_call(case)
    assert out.dtype == torch.bfloat16
    assert torch.equal(out, torch.zeros_like(out))
    assert torch.equal(lse, torch.zeros_like(lse))


def test_attention_c4_c128_slot_consistency():
    d = 512
    q = torch.ones(2, 1, d, dtype=torch.float16)
    cmp_kv = torch.zeros(1, 4, 1, d, dtype=torch.float16)
    cmp_kv[0, 0, 0].fill_(1.0)
    cmp_tree = torch.zeros(2, d, dtype=torch.float16)
    cmp_tree[1].fill_(-1.0)
    kwargs = dict(
        q=q,
        ori_slots=torch.full((2, 1, 4), -1, dtype=torch.int32),
        ori_rows=4,
        cmp_kv=cmp_kv,
        cmp_tree_kv=cmp_tree,
        ancestors=torch.tensor([[0, -1], [0, 1]], dtype=torch.int64),
        prefix=torch.tensor([6, 6], dtype=torch.int32),
        seqused=torch.tensor([7, 8], dtype=torch.int32),
    )
    c4 = _attn_case(
        **kwargs,
        cmp_slots=torch.tensor([[[0, -1]], [[0, 1]]], dtype=torch.int32),
    )
    c128 = _attn_case(**kwargs, cmp_slots=None, ratio=4)
    out4, lse4 = _attn_call(c4)
    out128, lse128 = _attn_call(c128)
    assert_close(out4, out128, ATTN_TOL)
    assert_close(lse4, lse128, LSE_TOL)


def test_attention_fixture_modes_smoke():
    for mode in ("c4", "c128", "swa"):
        case = make_attention_case(
            seed=2, cmp_mode=mode, budget=1, depth=1, prefixes=[6]
        )
        out, lse = _attn_call(case)
        assert out.shape == case["q"].shape
        assert out.dtype == case["q"].dtype
        assert lse.shape == (case["q"].shape[0], case["q"].shape[1], 1)
        assert torch.isfinite(out.to(torch.float32)).all()
        assert torch.isfinite(lse).all()


def test_tree_fixture_is_branching_and_padded():
    spec = make_tree(budget=4, depth=3, seed=1)
    assert spec.ancestors.shape == (spec.prefixes.numel(), spec.ancestors.shape[1])
    assert (spec.ancestors >= -1).all()
    assert torch.equal(spec.cu, torch.arange(spec.prefixes.numel() + 1, dtype=torch.int32))
    # Root-first order: the first column is the request root.
    roots = {int(v) for v in spec.ancestors[:, 0].tolist()}
    assert roots == set(spec.roots.tolist())
    # Budget 4 with depth 3 must contain a branch (node 1 has two children).
    assert spec.ancestors.shape[1] >= 2
    depth1 = (spec.depths == 1).sum().item()
    assert depth1 >= 2
