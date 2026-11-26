import pytest
import torch
import torch_npu  # noqa: F401
from sgl_kernel_npu.operators.attention import build_ngram_attention_mask

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance


@pytest.mark.parametrize(
    "prefixes,d,kv_len",
    [
        pytest.param([], 4, 128, id="empty-batch"),
        pytest.param([3], 1, 128, id="root-only"),
        pytest.param([3, 127], 4, 256, id="mixed-prefix-page-boundary"),
    ],
)
def test_paged_ancestor_mask(prefixes, d, kv_len):
    parents = [-1, 0, 0, 2][:d]
    allowed = torch.eye(d, dtype=torch.bool, device="cpu")
    for node, parent in enumerate(parents[1:], 1):
        allowed[node] |= allowed[parent]
    tree = allowed.repeat(len(prefixes), 1, 1)
    executor = check_accuracy(
        build_ngram_attention_mask,
        args=(
            tree.npu(),
            torch.tensor(prefixes, dtype=torch.int64, device="cpu").npu(),
            d,
            kv_len,
        ),
    )
    if not executor.accuracy_checked:
        check_performance(executor, enable_perf=False)
        return
    expected = torch.ones(len(prefixes), 1, d, kv_len, dtype=torch.bool, device="cpu")
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
    torch.testing.assert_close(executor().cpu(), expected)
    check_performance(executor, enable_perf=False)


@pytest.mark.parametrize("prefix_dtype", [torch.int32, torch.int64], ids=["i32", "i64"])
@pytest.mark.parametrize(
    "prefixes,kv_len",
    [
        pytest.param(
            [3, 127, 1023, 1024],
            1152,
            id="mixed-short-contexts",
            marks=pytest.mark.extra_checks,
        ),
        pytest.param(
            [3, 21000, 21887, 21888], 22016, id="ngram-serving-d12-21k-context"
        ),
    ],
)
def test_paged_ancestor_mask_long_prefix(prefix_dtype, prefixes, kv_len):
    d = 12
    parents = [-1, 0, 0, 1, 1, 2, 2, 3, 4, 5, 6, 7]
    tree = torch.eye(d, dtype=torch.bool, device="cpu")
    for node, parent in enumerate(parents[1:], 1):
        tree[node] |= tree[parent]
    tree = tree.repeat(len(prefixes), 1, 1)

    # Put the tree at the end of its backing allocation to exercise its tail.
    backing = torch.empty(tree.numel() + 17, dtype=torch.bool, device="npu")
    tree_view = backing[-tree.numel() :].view_as(tree)
    tree_view.copy_(tree)
    perf = kv_len == 22016 and prefix_dtype == torch.int64
    executor = check_accuracy(
        build_ngram_attention_mask,
        args=(
            tree_view,
            torch.tensor(prefixes, dtype=prefix_dtype, device="npu"),
            d,
            kv_len,
        ),
    )
    if not executor.accuracy_checked:
        check_performance(executor, enable_perf=perf)
        return

    expected = torch.ones(len(prefixes), 1, d, kv_len, dtype=torch.bool, device="cpu")
    for batch, prefix in enumerate(prefixes):
        expected[batch, 0, :, :prefix] = False
        for query in range(d):
            for ancestor in torch.where(tree[batch, query])[0].tolist():
                expected[batch, 0, query, prefix + ancestor] = False
    torch.testing.assert_close(executor().cpu(), expected)
    check_performance(executor, enable_perf=perf)
