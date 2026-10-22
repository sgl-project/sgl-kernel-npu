"""NGRAM irregular trees: independent parent-list oracles and branch regressions."""

import random

import pytest
import torch
import torch_npu  # noqa: F401
from sgl_kernel_npu.sample.verify_ngram_tree_greedy import verify_ngram_tree_greedy
from sgl_kernel_npu.speculative import reconstruct_indices_from_tree_mask_triton


def make_tree(parents):
    d = len(parents)
    mask = torch.eye(d, dtype=torch.bool)
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


@pytest.mark.parametrize("bs,d", [(0, 4), (1, 1), (3, 4), (3, 12), (65, 12), (2, 32)])
@pytest.mark.parametrize("kind", ["chain", "star", "random"])
def test_reconstruct_legal_ancestry(bs, d, kind):
    rng = random.Random(20261013)
    trees = []
    for _ in range(bs):
        parents = [-1] + [
            n - 1 if kind == "chain" else 0 if kind == "star" else rng.randrange(n)
            for n in range(1, d)
        ]
        trees.append(make_tree(parents))
    mask = (
        torch.stack([t[0] for t in trees])
        if bs
        else torch.empty((0, d, d), dtype=torch.bool)
    )
    prefixes = torch.arange(bs, dtype=torch.int64) * 17 + 3
    positions = torch.empty(bs * d, dtype=torch.int64, device="npu")
    outputs = [torch.empty((bs, d), dtype=torch.int64, device="npu") for _ in range(3)]
    reconstruct_indices_from_tree_mask_triton(
        mask.npu(), prefixes.npu(), positions, *outputs, bs, d
    )
    assert positions.cpu().tolist() == [
        int(prefixes[b]) + depth for b, tree in enumerate(trees) for depth in tree[1]
    ]
    assert outputs[0].cpu().tolist() == [
        [b * d + i for i in range(d)] for b in range(bs)
    ]
    assert outputs[1].cpu().tolist() == [t[2] for t in trees]
    assert outputs[2].cpu().tolist() == [t[3] for t in trees]


@pytest.mark.parametrize("accept_width", [3, 4])
@pytest.mark.parametrize("permuted", [False, True])
def test_greedy_follows_branch_and_stops_at_leaf(accept_width, permuted):
    # Root 5 -> {10, 20}; 20 -> 30. NGRAM must follow a matching sibling
    # branch and stop when the accepted branch reaches a leaf.
    candidates = torch.tensor([[5, 10, 20, 30]] * 3, device="npu")
    targets = torch.tensor([[20, 77, 30, 99], [10, 20, 30, 99], [88, 77, 66, 55]])
    mapping = torch.tensor([0, 2, 1, 3] if permuted else [0, 1, 2, 3])
    retrieve = torch.stack([mapping + b * 4 for b in range(3)])
    stored_targets = torch.empty_like(targets)
    stored_targets[:, mapping] = targets
    first = torch.tensor([[1, -1, 3, -1]] * 3, device="npu")
    sibling = torch.tensor([[-1, 2, -1, -1]] * 3, device="npu")
    predicts = torch.full((12,), -1, dtype=torch.int32, device="npu")
    accept = torch.full((3, accept_width), -1, dtype=torch.int32, device="npu")
    lengths = torch.zeros(3, dtype=torch.int32, device="npu")
    verify_ngram_tree_greedy(
        predicts,
        accept,
        lengths,
        candidates,
        retrieve.npu(),
        first,
        sibling,
        stored_targets.npu(),
    )
    expected_accept = torch.full((3, accept_width), -1, dtype=torch.int32)
    expected_predicts = torch.full((12,), -1, dtype=torch.int32)
    for b, path in enumerate([[0, 2, 3], [0, 1], [0]]):
        for offset, node in enumerate(path):
            index = int(retrieve[b, node])
            expected_accept[b, offset] = index
            expected_predicts[index] = targets[b, node]
    assert lengths.cpu().tolist() == [2, 1, 0]
    torch.testing.assert_close(accept.cpu(), expected_accept)
    torch.testing.assert_close(predicts.cpu(), expected_predicts)


def test_greedy_single_root():
    predicts = torch.full((2,), -1, dtype=torch.int32, device="npu")
    accept = torch.full((2, 1), -1, dtype=torch.int32, device="npu")
    lengths = torch.zeros(2, dtype=torch.int32, device="npu")
    verify_ngram_tree_greedy(
        predicts,
        accept,
        lengths,
        torch.tensor([[5], [6]], device="npu"),
        torch.tensor([[0], [1]], device="npu"),
        torch.full((2, 1), -1, device="npu"),
        torch.full((2, 1), -1, device="npu"),
        torch.tensor([[42], [43]], device="npu"),
    )
    assert predicts.cpu().tolist() == [42, 43]
    assert accept.cpu().tolist() == [[0], [1]]
    assert lengths.cpu().tolist() == [0, 0]
