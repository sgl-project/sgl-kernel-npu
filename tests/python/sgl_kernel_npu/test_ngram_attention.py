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
