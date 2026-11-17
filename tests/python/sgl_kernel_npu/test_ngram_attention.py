import pytest
import torch
import torch_npu  # noqa: F401
from sgl_kernel_npu.attention.ngram_attention import build_ngram_attention_mask


@pytest.mark.parametrize(
    "prefixes,d,kv_len", [([], 4, 128), ([3], 1, 128), ([3, 127], 4, 256)]
)
def test_paged_ancestor_mask(prefixes, d, kv_len):
    parents = [-1, 0, 0, 2][:d]
    allowed = torch.eye(d, dtype=torch.bool)
    for node, parent in enumerate(parents[1:], 1):
        allowed[node] |= allowed[parent]
    tree = allowed.repeat(len(prefixes), 1, 1)
    actual = build_ngram_attention_mask(
        tree.npu(), torch.tensor(prefixes, dtype=torch.int64).npu(), d, kv_len
    )
    expected = torch.ones(len(prefixes), 1, d, kv_len, dtype=torch.bool)
    for b, prefix in enumerate(prefixes):
        for query in range(d):
            ancestors = set()
            node = query
            while node != -1:
                ancestors.add(node)
                node = parents[node]
            for key in range(kv_len):
                expected[b, 0, query, key] = not (
                    key < prefix or key - prefix in ancestors
                )
    torch.testing.assert_close(actual.cpu(), expected)


@pytest.mark.parametrize("prefix_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize(
    "prefixes,kv_len", [([3, 127, 1023, 1024], 1152), ([3, 21000, 21887, 21888], 22016)]
)
def test_paged_ancestor_mask_long_prefix(prefix_dtype, prefixes, kv_len):
    d = 12
    parents = [-1, 0, 0, 1, 1, 2, 2, 3, 4, 5, 6, 7]
    tree = torch.eye(d, dtype=torch.bool)
    for node, parent in enumerate(parents[1:], 1):
        tree[node] |= tree[parent]
    tree = tree.repeat(len(prefixes), 1, 1)

    # Put the tree at the end of its backing allocation to exercise its tail.
    backing = torch.empty(tree.numel() + 17, dtype=torch.bool, device="npu")
    tree_view = backing[-tree.numel() :].view_as(tree)
    tree_view.copy_(tree)
    actual = build_ngram_attention_mask(
        tree_view, torch.tensor(prefixes, dtype=prefix_dtype, device="npu"), d, kv_len
    )

    expected = torch.ones(len(prefixes), 1, d, kv_len, dtype=torch.bool)
    for batch, prefix in enumerate(prefixes):
        expected[batch, 0, :, :prefix] = False
        for query in range(d):
            for ancestor in torch.where(tree[batch, query])[0].tolist():
                expected[batch, 0, query, prefix + ancestor] = False
    torch.testing.assert_close(actual.cpu(), expected)
