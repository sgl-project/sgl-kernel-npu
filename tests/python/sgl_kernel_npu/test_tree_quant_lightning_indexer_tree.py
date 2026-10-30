"""NPU parity and contract tests for the DSV4 DSpark tree lightning indexer.

Wave 3 kernel: ``quant_lightning_indexer_tree`` (``docs/tree_ops_semantics.md``
§4, ``tree/README.md``).  The independent CPU oracle is ``tree_reference.py``
(D5); production code never imports it.

Comparisons follow D3: top-k is compared as an index **multiset** plus a
**score set** (atol 1e-3), never as an ordered list, because tie order is not
contractual.  ``-1`` padding counts must match exactly.

This module skips cleanly on a CPU-only host: ``torch_npu``, ``triton`` and the
built ``libsgl_kernel_npu.so`` are all required by the import chain.
"""

from __future__ import annotations

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")
pytest.importorskip("triton")

# ``sgl_kernel_npu`` loads ``lib/libsgl_kernel_npu.so`` at import time.
tree_indexer = pytest.importorskip("sgl_kernel_npu.tree.quant_lightning_indexer_tree")

from tree_fixtures import make_indexer_case
from tree_reference import quant_lightning_indexer_tree_ref

quant_lightning_indexer_tree = tree_indexer.quant_lightning_indexer_tree
TreeContractError = tree_indexer.TreeContractError

DEVICE = "npu"
QUERY_HEADS = 64
HEAD_DIM = 128

# Score tolerance for the D1 "tolerance-based, not bit-parity" acceptance gate.
ATOL = 1e-3
RTOL = 1e-3

# n -> (batch, budget, depth) for the one-query-per-node fixture tree.
_GEOMETRY = {1: (1, 0, 0), 4: (1, 3, 2), 8: (1, 7, 3), 17: (1, 16, 4)}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def make_case(n=8, *, seed=0, rows=4, topk=4, prefixes=None):
    """Build an indexer fixture case for ``n`` query rows (one per tree node)."""
    batch, budget, depth = _GEOMETRY[int(n)]
    return make_indexer_case(
        seed=int(seed),
        rows=int(rows),
        topk=int(topk),
        batch=batch,
        budget=budget,
        depth=depth,
        prefixes=prefixes,
    )


def lift_heads(args):
    """Expand the fixture's ``[n, 1, *]`` group tensors to the 64-head contract.

    The fixture drives a single active query group; heads ``1..63`` are zero so
    their score contribution is exactly ``0 * wq``.
    """
    query, weights, qs = args["query"], args["weights"], args["qs"]
    n, groups = int(query.shape[0]), int(query.shape[1])
    lifted_query = torch.zeros(n, QUERY_HEADS, HEAD_DIM, dtype=torch.int8)
    lifted_query[:, :groups] = query
    lifted_weights = torch.zeros(n, QUERY_HEADS, dtype=torch.float16)
    lifted_weights[:, :groups] = weights
    lifted_scale = torch.ones(n, QUERY_HEADS, dtype=torch.float16)
    lifted_scale[:, :groups] = qs
    lifted = dict(args)
    lifted.update(query=lifted_query, weights=lifted_weights, qs=lifted_scale)
    return lifted


def to_device(args):
    return {
        key: (value.to(DEVICE) if isinstance(value, torch.Tensor) else value)
        for key, value in args.items()
    }


def call_kernel(args):
    """Call the production op with the README signature and fixture arguments."""
    device_args = to_device(lift_heads(args))
    return quant_lightning_indexer_tree(
        device_args["query"],
        device_args["history"],
        device_args["weights"],
        device_args["qs"],
        device_args["history_scale"],
        device_args["tree_key"],
        device_args["tree_scale"],
        device_args["ancestors"],
        device_args["prefixes"],
        device_args["reqs"],
        device_args["block_table"],
        device_args["cu"],
        device_args["lengths"],
        device_args["residual"],
        device_args["metadata"],
        int(device_args["topk"]),
    )


def slot_scores(args, row, indices):
    """CPU score of each slot index, re-derived from §4.3 (not from the kernel)."""
    query = args["query"].to(torch.int64)
    weight = (args["weights"].to(torch.float16) * args["qs"].to(torch.float16)).to(
        torch.float32
    )[row]
    prefix = int(args["prefixes"][row])
    history_end = prefix // 4
    block = int(args["history"].shape[1])
    pages = int(args["history"].shape[0])
    table = args["block_table"]
    req = int(args["reqs"][row])
    ancestors = args["ancestors"]
    nodes = args["tree_key"]

    scores = []
    for k in indices:
        if k < 0:
            continue
        if k < history_end:
            col = k // block
            page = int(table[req, col]) if col < int(table.shape[1]) else -1
            if page < 0 or page >= pages:
                scores.append(float("-inf"))
                continue
            key = args["history"][page, k % block, 0].to(torch.int64)
            scale = float(args["history_scale"][page, k % block, 0, 0])
        else:
            step = (k + 1) * 4 - 1 - prefix
            if step < 0 or step >= int(ancestors.shape[1]):
                scores.append(float("-inf"))
                continue
            node = int(ancestors[row, step])
            if node < 0 or node >= int(nodes.shape[0]):
                scores.append(float("-inf"))
                continue
            key = nodes[node].to(torch.int64)
            scale = float(args["tree_scale"][node])
        exact = (query[row] * key.unsqueeze(0)).sum(dim=1)  # int64, exact
        score0 = (exact.clamp(min=0).to(torch.float32) * (2.0**-10)).to(
            torch.float16
        ).to(torch.float32)
        scores.append(float((score0 * weight).sum() * scale))
    return scores


def assert_matches_reference(actual, args):
    """Compare against the independent oracle: pads, index multiset, score set."""
    lifted = lift_heads(args)
    expected = quant_lightning_indexer_tree_ref(**lifted)
    topk = int(args["topk"])
    n = int(args["query"].shape[0])

    assert actual.dtype == torch.int32
    assert expected.dtype == torch.int32
    assert tuple(actual.shape) == (n, 1, topk)
    assert tuple(expected.shape) == (n, 1, topk)

    got = actual.cpu()
    for row in range(n):
        ours = got[row, 0].tolist()
        oracle = expected[row, 0].tolist()
        assert ours.count(-1) == oracle.count(-1), f"row {row}: -1 padding count"
        assert sorted(i for i in ours if i >= 0) == sorted(
            i for i in oracle if i >= 0
        ), f"row {row}: index multiset"

        ours_scores = slot_scores(lifted, row, ours)
        oracle_scores = slot_scores(lifted, row, oracle)
        assert len(ours_scores) == len(oracle_scores)
        torch.testing.assert_close(
            torch.tensor(sorted(ours_scores)),
            torch.tensor(sorted(oracle_scores)),
            atol=ATOL,
            rtol=RTOL,
            msg=f"row {row}: selected score set",
        )
    return expected


def padded_pages(history, history_scale, pad):
    """Re-wrap history/history_scale with a padded page stride.

    ``stride(1) == 128`` / ``stride(3) == 1`` stay mandatory; only the page
    stride (``stride(0)``) is padded (docs/tree_ops_semantics.md R4).
    """
    if pad <= 0:
        return history, history_scale
    pages, block = int(history.shape[0]), int(history.shape[1])
    row_elems = block * HEAD_DIM
    hist_buf = torch.zeros(pages * (row_elems + pad), dtype=history.dtype)
    hist = torch.as_strided(
        hist_buf, (pages, block, 1, HEAD_DIM), (row_elems + pad, HEAD_DIM, HEAD_DIM, 1)
    )
    scale_buf = torch.zeros(pages * (block + pad), dtype=history_scale.dtype)
    scale = torch.as_strided(
        scale_buf, (pages, block, 1, 1), (block + pad, 1, 1, 1)
    )
    for page in range(pages):
        hist[page].copy_(history[page])
        scale[page].copy_(history_scale[page])
    return hist, scale


# ---------------------------------------------------------------------------
# top-k / n / select-all matrix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n,topk",
    [
        (1, 1),
        (1, 64),
        (8, 1),
        (8, 64),
        (8, 512),
        (17, 1),
        (17, 64),
        (17, 512),
        (17, 1024),
        (17, 2048),
    ],
)
def test_matches_reference_topk_matrix(n, topk):
    # prefixes=[512] -> lengths <= 129, so topk <= 64 takes the ranked path and
    # topk >= 512 takes the select-all shortcut.  Both branches are covered.
    case = make_case(n, rows=16, topk=topk, prefixes=[512])
    assert_matches_reference(call_kernel(case["args"]), case["args"])


def test_matches_reference_ranked_long_history():
    # lengths ~ 2000 (below the 2048-slot accumulator), topk 1024 -> ranked path.
    case = make_case(4, rows=128, topk=1024, prefixes=[8000])
    lengths = case["args"]["lengths"]
    assert int(lengths.max()) > 1024
    assert int(lengths.max()) <= 2048
    assert_matches_reference(call_kernel(case["args"]), case["args"])


def test_ranked_history_beyond_2048_tail_wins():
    # The vendor keeps a 2048-wide output accumulator but scans every S2 tile;
    # it must not truncate the candidate set at logical slot 2048.
    case = make_case(17, rows=128, topk=512, prefixes=[32768])
    args = dict(case["args"])
    history = args["history"].clone()
    history_page_size = int(history.shape[1])
    tail_start = 56 * history_page_size  # logical slots >= 7168
    history[56:, :, 0, 0] = 127  # beat the fixture's tree keys (100)
    args["history"] = history
    lengths = args["lengths"]
    assert int(lengths.max()) > 2048
    expected = quant_lightning_indexer_tree_ref(**args)
    assert any(int(i) >= 2048 for i in expected.reshape(-1).tolist() if i >= 0)
    assert_matches_reference(call_kernel(args), args)


# ---------------------------------------------------------------------------
# prefix / boundary matrix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("prefix", [0, 3, 4, 5, 128, 512])
def test_matches_reference_prefixes(prefix):
    # prefix 0/3 are history-only, 4/5 cross the first history|tree boundary.
    case = make_case(8, rows=16, topk=4, prefixes=[prefix])
    assert_matches_reference(call_kernel(case["args"]), case["args"])


# ---------------------------------------------------------------------------
# paged history geometry (block size, page count, padded page stride)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("block,pages,pad", [(16, 1, 0), (16, 4, 7), (128, 1, 0), (128, 4, 13)])
def test_matches_reference_pages_and_page_stride(block, pages, pad):
    prefix = (pages * block - 1) * 4  # history_end == pages * block - 1 -> cols == pages
    case = make_case(8, rows=block, topk=2, prefixes=[prefix])
    args = dict(case["args"])
    assert int(args["history"].shape[0]) == pages

    hist, scale = padded_pages(args["history"], args["history_scale"], pad)
    assert hist.stride(1) == HEAD_DIM
    assert hist.stride(3) == 1
    if pad:
        assert hist.stride(0) != block * HEAD_DIM
    args["history"], args["history_scale"] = hist, scale

    assert_matches_reference(call_kernel(args), args)


# ---------------------------------------------------------------------------
# ancestor masking (docs/tree_ops_semantics.md §1.2)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [-1, 8])  # absent sentinel, and node id >= n
def test_invalid_ancestors_masked(bad):
    case = make_case(8, rows=16, topk=3, prefixes=[64])
    baseline = call_kernel(case["args"]).cpu()

    args = dict(case["args"])
    ancestors = args["ancestors"].clone()
    depths = case["spec"].depths.tolist()
    for row, depth in enumerate(depths):
        if depth > 0:
            ancestors[row, depth] = bad
    args["ancestors"] = ancestors

    actual = call_kernel(args).cpu()
    assert_matches_reference(actual, args)

    # §1.2: an absent / out-of-range ancestor is never returned as a candidate.
    lifted = lift_heads(args)
    for row in range(8):
        for index in actual[row, 0].tolist():
            if index >= 0:
                assert slot_scores(lifted, row, [index])[0] > float("-inf")

    # The tree keys dominate the paged history, so masking them must reorder.
    assert not torch.equal(actual, baseline)


# ---------------------------------------------------------------------------
# fp16 score degenerate case: zero key scale -> exact ties -> index ascending
# ---------------------------------------------------------------------------


def test_zero_key_scale_ties_ascending():
    case = make_case(8, rows=16, topk=4, prefixes=[512])
    args = dict(case["args"])
    args["tree_scale"] = torch.zeros_like(args["tree_scale"])
    args["history_scale"] = torch.zeros_like(args["history_scale"])

    actual = call_kernel(args).cpu()
    assert_matches_reference(actual, args)
    for row in range(8):
        valid = int(args["lengths"][row])
        want = [i for i in range(min(4, valid))] + [-1] * max(0, 4 - valid)
        assert actual[row, 0].tolist() == want


# ---------------------------------------------------------------------------
# select-all shortcut (README: max(1, max_kv // 4) <= topk -> no kernel)
# ---------------------------------------------------------------------------


def test_select_all_shortcut_returns_chronological_indices():
    case = make_case(17, rows=16, topk=512, prefixes=[5])
    args = dict(case["args"])
    lengths = args["lengths"].tolist()
    assert max(lengths) <= 512

    actual = call_kernel(args).cpu()
    assert_matches_reference(actual, args)
    for row, length in enumerate(lengths):
        assert actual[row, 0].tolist() == [
            i if i < length else -1 for i in range(512)
        ]


# ---------------------------------------------------------------------------
# topk larger than the number of valid slots -> -1 padding
# ---------------------------------------------------------------------------


def test_topk_exceeds_valid_slots_pads_minus_one():
    case = make_case(8, rows=16, topk=8, prefixes=[64])
    args = dict(case["args"])
    # Short rows force padding while max(lengths) > topk keeps the ranked path.
    args["lengths"] = torch.tensor([1, 2, 3, 17, 17, 17, 17, 17], dtype=torch.int32)
    assert int(args["lengths"].max()) > 8

    actual = call_kernel(args).cpu()
    assert_matches_reference(actual, args)
    assert actual[0, 0].tolist() == [0, -1, -1, -1, -1, -1, -1, -1]
    assert actual[1, 0].tolist()[2:].count(-1) == 6


# ---------------------------------------------------------------------------
# determinism + D2 (metadata validated then ignored)
# ---------------------------------------------------------------------------


def test_deterministic_and_metadata_contents_ignored():
    case = make_case(17, rows=16, topk=8, prefixes=[512])
    args = dict(case["args"])

    first = call_kernel(args).cpu()
    second = call_kernel(args).cpu()
    assert torch.equal(first, second)

    metadata = torch.full((1024,), 12345, dtype=torch.int32)
    metadata[0] = 1
    metadata[36 * 8 : 36 * 8 + 8] = 7
    with_metadata = call_kernel(dict(args, metadata=metadata)).cpu()
    assert torch.equal(first, with_metadata)


# ---------------------------------------------------------------------------
# full 64-head reduction (the fixture only drives group 0)
# ---------------------------------------------------------------------------


def test_multi_head_reduction_matches_reference():
    case = make_case(8, rows=16, topk=8, prefixes=[512], seed=3)
    args = dict(case["args"])
    generator = torch.Generator().manual_seed(7)
    n = 8
    args["query"] = torch.randint(
        -127, 128, (n, QUERY_HEADS, HEAD_DIM), generator=generator, dtype=torch.int16
    ).to(torch.int8)
    args["weights"] = (torch.randn(n, QUERY_HEADS, generator=generator) * 0.01).to(
        torch.float16
    )
    args["qs"] = (torch.rand(n, QUERY_HEADS, generator=generator) + 0.5).to(
        torch.float16
    )
    args["tree_key"] = torch.randint(
        -127, 128, (n, HEAD_DIM), generator=generator, dtype=torch.int16
    ).to(torch.int8)

    assert_matches_reference(call_kernel(args), args)


# ---------------------------------------------------------------------------
# int8 tl.dot fallback (docs/tree_ops_semantics.md R8)
# ---------------------------------------------------------------------------


def test_int32_widening_fallback_matches_reference(monkeypatch):
    case = make_case(8, rows=16, topk=4, prefixes=[512])
    args = dict(case["args"])
    monkeypatch.setattr(tree_indexer, "_USE_INT8_DOT", False)
    assert_matches_reference(call_kernel(args), args)


# ---------------------------------------------------------------------------
# host contract
# ---------------------------------------------------------------------------


def test_contract_violations_raise():
    case = make_case(8, rows=16, topk=4, prefixes=[4])
    good = to_device(lift_heads(case["args"]))

    def call(**override):
        merged = dict(good)
        merged.update(override)
        return quant_lightning_indexer_tree(
            merged["query"],
            merged["history"],
            merged["weights"],
            merged["qs"],
            merged["history_scale"],
            merged["tree_key"],
            merged["tree_scale"],
            merged["ancestors"],
            merged["prefixes"],
            merged["reqs"],
            merged["block_table"],
            merged["cu"],
            merged["lengths"],
            merged["residual"],
            merged["metadata"],
            int(merged["topk"]),
        )

    # The valid contract must pass (short prefixes -> select-all path).
    assert call().shape == (8, 1, 4)

    bad_contracts = [
        {"metadata": torch.zeros(1024, dtype=torch.int64, device=DEVICE)},
        {"metadata": torch.zeros(512, dtype=torch.int32, device=DEVICE)},
        {"cu": torch.arange(9, dtype=torch.int32, device=DEVICE) + 1},
        {"lengths": torch.zeros(8, dtype=torch.int64, device=DEVICE)},
        {"block_table": torch.zeros((1, 1), dtype=torch.int64, device=DEVICE)},
        {"ancestors": torch.zeros((8, 5), dtype=torch.int32, device=DEVICE)},
        {"topk": 0},
        {"topk": 2049},
        {"history": torch.zeros((1, 4, 1, 256), dtype=torch.int8, device=DEVICE)[..., :HEAD_DIM]},
    ]
    for override in bad_contracts:
        with pytest.raises(TreeContractError):
            call(**override)
