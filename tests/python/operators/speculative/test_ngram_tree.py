"""NGRAM irregular trees: independent parent-list oracles and branch regressions."""

import random

import pytest
import torch
import torch_npu  # noqa: F401
from sgl_kernel_npu.operators.speculative import (
    reconstruct_indices_from_tree_mask,
    verify_ngram_tree_greedy,
)

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance


def _make_tree(parents):
    d = len(parents)
    mask = torch.eye(d, dtype=torch.bool, device="cpu")
    children = [[] for _ in parents]
    depth = [0] * d
    for node in range(1, d):
        parent = parents[node]
        mask[node] |= mask[parent]
        depth[node] = depth[parent] + 1
        children[parent].append(node)
    first = [nodes[0] if nodes else -1 for nodes in children]
    sibling = [-1] * d
    for nodes in children:
        for left, right in zip(nodes, nodes[1:]):
            sibling[left] = right
    return mask, depth, first, sibling


def _generate_reconstruct_data(bs, d, kind):
    rng = random.Random(20261013)
    trees = []
    for _ in range(bs):
        parents = [-1] + [
            n - 1 if kind == "chain" else 0 if kind == "star" else rng.randrange(n)
            for n in range(1, d)
        ]
        trees.append(_make_tree(parents))
    mask = (
        torch.stack([t[0] for t in trees])
        if bs
        else torch.empty((0, d, d), dtype=torch.bool, device="cpu")
    )
    prefixes = torch.arange(bs, dtype=torch.int64, device="cpu") * 17 + 3
    return trees, mask, prefixes


@pytest.mark.parametrize(
    "bs,d,kind",
    [
        pytest.param(6, 12, "random", id="ngram-serving-b6-d12-branches"),
        pytest.param(0, 4, "chain", id="empty-batch"),
        pytest.param(1, 1, "chain", id="root-only"),
        pytest.param(3, 4, "chain", id="linear-chain", marks=pytest.mark.extra_checks),
        pytest.param(3, 12, "star", id="wide-fanout", marks=pytest.mark.extra_checks),
        pytest.param(
            65, 12, "random", id="multi-wave-batch", marks=pytest.mark.extra_checks
        ),
        pytest.param(2, 32, "random", id="wide-tree", marks=pytest.mark.extra_checks),
    ],
)
def test_reconstruct_legal_ancestry(bs, d, kind):
    trees, mask, prefixes = _generate_reconstruct_data(bs, d, kind)
    positions = torch.full((bs * d,), -1, dtype=torch.int64, device="npu")
    outputs = [
        torch.full((bs, d), -1, dtype=torch.int64, device="npu") for _ in range(3)
    ]
    executor = check_accuracy(
        reconstruct_indices_from_tree_mask,
        kwargs=dict(
            tree_mask=mask.npu(),
            verified_seq_len=prefixes.npu(),
            positions=positions,
            retrive_index=outputs[0],
            retrive_next_token=outputs[1],
            retrive_next_sibling=outputs[2],
            batch_size=bs,
            draft_token_num=d,
        ),
        mutable_kwargs=(
            "positions",
            "retrive_index",
            "retrive_next_token",
            "retrive_next_sibling",
        ),
    )
    if executor.accuracy_checked:
        assert positions.cpu().tolist() == [
            int(prefixes[b]) + depth
            for b, tree in enumerate(trees)
            for depth in tree[1]
        ]
        assert outputs[0].cpu().tolist() == [
            [b * d + i for i in range(d)] for b in range(bs)
        ]
        assert outputs[1].cpu().tolist() == [t[2] for t in trees]
        assert outputs[2].cpu().tolist() == [t[3] for t in trees]
    check_performance(executor, enable_perf=(bs, d, kind) == (6, 12, "random"))


@pytest.mark.parametrize("accept_width", [3, 4], ids=["short-accept", "full-accept"])
@pytest.mark.parametrize(
    "permuted", [False, True], ids=["ordered-slots", "permuted-slots"]
)
def test_greedy_follows_branch_and_stops_at_leaf(accept_width, permuted):
    # Root 5 -> {10, 20}; 20 -> 30. NGRAM must follow a matching sibling
    # branch and stop when the accepted branch reaches a leaf.
    candidates = torch.tensor([[5, 10, 20, 30]] * 3, device="npu")
    targets = torch.tensor(
        [[20, 77, 30, 99], [10, 20, 30, 99], [88, 77, 66, 55]], device="cpu"
    )
    mapping = torch.tensor([0, 2, 1, 3] if permuted else [0, 1, 2, 3], device="cpu")
    retrieve = torch.stack([mapping + b * 4 for b in range(3)])
    stored_targets = torch.empty_like(targets)
    stored_targets[:, mapping] = targets
    first = torch.tensor([[1, -1, 3, -1]] * 3, device="npu")
    sibling = torch.tensor([[-1, 2, -1, -1]] * 3, device="npu")
    predicts = torch.full((12,), -1, dtype=torch.int32, device="npu")
    accept = torch.full((3, accept_width), -1, dtype=torch.int32, device="npu")
    lengths = torch.zeros(3, dtype=torch.int32, device="npu")
    executor = _check_verify(
        predicts,
        accept,
        lengths,
        candidates,
        retrieve.npu(),
        first,
        sibling,
        stored_targets.npu(),
    )
    if not executor.accuracy_checked:
        check_performance(executor, enable_perf=False)
        return
    expected_accept = torch.full((3, accept_width), -1, dtype=torch.int32, device="cpu")
    expected_predicts = torch.full((12,), -1, dtype=torch.int32, device="cpu")
    for b, path in enumerate([[0, 2, 3], [0, 1], [0]]):
        for offset, node in enumerate(path):
            index = int(retrieve[b, node])
            expected_accept[b, offset] = index
            expected_predicts[index] = targets[b, node]
    assert lengths.cpu().tolist() == [2, 1, 0]
    torch.testing.assert_close(accept.cpu(), expected_accept)
    torch.testing.assert_close(predicts.cpu(), expected_predicts)
    check_performance(executor, enable_perf=False)


def test_greedy_single_root():
    predicts = torch.full((2,), -1, dtype=torch.int32, device="npu")
    accept = torch.full((2, 1), -1, dtype=torch.int32, device="npu")
    lengths = torch.zeros(2, dtype=torch.int32, device="npu")
    executor = _check_verify(
        predicts,
        accept,
        lengths,
        torch.tensor([[5], [6]], device="npu"),
        torch.tensor([[0], [1]], device="npu"),
        torch.full((2, 1), -1, device="npu"),
        torch.full((2, 1), -1, device="npu"),
        torch.tensor([[42], [43]], device="npu"),
    )
    if executor.accuracy_checked:
        assert predicts.cpu().tolist() == [42, 43]
        assert accept.cpu().tolist() == [[0], [1]]
        assert lengths.cpu().tolist() == [0, 0]
    check_performance(executor, enable_perf=False)


def _check_verify(
    predicts, accept, lengths, candidates, retrieve, first, sibling, targets
):
    return check_accuracy(
        verify_ngram_tree_greedy,
        kwargs=dict(
            predicts=predicts,
            accept_index=accept,
            accept_token_num=lengths,
            candidates=candidates,
            retrive_index=retrieve,
            retrive_next_token=first,
            retrive_next_sibling=sibling,
            target_predict=targets,
        ),
        mutable_kwargs=("predicts", "accept_index", "accept_token_num"),
    )


def test_verify_ngram_serving_b6_d12():
    # Match the benchmark's batch/draft width, with a deterministic branch.
    bs, width = 6, 12
    _, _, first, sibling = _make_tree([-1] + [(n - 1) // 2 for n in range(1, width)])
    retrieve = torch.arange(bs * width, device="npu").reshape(bs, width)
    candidates = retrieve + 20
    first = torch.tensor(first, device="npu").repeat(bs, 1)
    sibling = torch.tensor(sibling, device="npu").repeat(bs, 1)
    targets = torch.full_like(candidates, 7)
    targets[:, 0] = candidates[:, 2]
    targets[:, 2] = candidates[:, 6]
    predicts = torch.full((bs * width,), -1, device="npu", dtype=torch.int32)
    accept = torch.full((bs, width), -1, device="npu", dtype=torch.int32)
    counts = torch.zeros(bs, device="npu", dtype=torch.int32)
    executor = _check_verify(
        predicts, accept, counts, candidates, retrieve, first, sibling, targets
    )
    if executor.accuracy_checked:
        assert counts.cpu().tolist() == [2] * bs
        assert torch.equal(accept[:, :3].cpu().long(), retrieve[:, [0, 2, 6]].cpu())
    check_performance(executor, enable_perf=True)
