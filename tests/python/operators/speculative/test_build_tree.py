"""Exact legacy fixture through the new public interface.

Fixture source: tests/python/sgl_kernel_npu/test_build_tree.py at the PR #976
baseline. parent_list is cat(parents_list[:-1]); selected_index is the sorted
top-7 score indices from that fixture, with no changed scores or dimensions.
The low-level tree operator does not consume token IDs or scores themselves.
Scenario: SGLang speculative/eagle_utils.py::build_tree_kernel_efficient passes
these parent/selection/prefix tensors and five output buffers to the NPU op.
This synthetic fixture checks that protocol with branching and unequal prefixes;
its dimensions are not presented as a captured production request.
"""

import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import assert_accuracy
from tests.python.utils.common import (
    get_skip_acc_check,
    get_test_impls,
    make_executor,
    require_distinct_impls,
    require_test_impl,
)
from tests.python.utils.performance import check_performance


def generate_test_data(device, prefix_lengths=(5, 10)):
    """Keep the legacy B=2 tree and select its verified-prefix lengths."""
    inputs = dict(
        parent_list=torch.tensor(
            [
                [-1, 0, 1, 2, 3, 4, 8, 9, 10, 20, 24, 21, 28],
                [-1, 0, 1, 2, 3, 4, 5, 6, 7, 24, 28, 20, 21],
            ],
            dtype=torch.int64,
            device=device,
        ),
        selected_index=torch.tensor(
            [[0, 1, 4, 8, 20, 24, 36], [0, 4, 5, 6, 7, 24, 36]],
            dtype=torch.int64,
            device=device,
        ),
        verified_seq_len=torch.tensor(prefix_lengths, dtype=torch.int64, device=device),
        topk=4,
        depth=4,
        draft_token_num=8,
    )

    batch, width = 2, 8
    size = sum(prefix_lengths) * width + batch * width * width
    outputs = dict(
        tree_mask=torch.ones(size, dtype=torch.bool, device=device),
        positions=torch.full((batch * width,), -1, dtype=torch.int64, device=device),
        retrive_index=torch.full((batch, width), -1, dtype=torch.int64, device=device),
        retrive_next_token=torch.full(
            (batch, width), -1, dtype=torch.int64, device=device
        ),
        retrive_next_sibling=torch.full(
            (batch, width), -1, dtype=torch.int64, device=device
        ),
    )

    return inputs, outputs


def assert_fixture_expectations(outputs, prefix_lengths=(5, 10)):
    device = outputs["positions"].device
    expected = dict(
        positions=[
            length + node_depth
            for length, depths in zip(
                prefix_lengths,
                ([0, 1, 1, 2, 2, 3, 3, 4], [0, 1, 2, 2, 2, 2, 3, 4]),
            )
            for node_depth in depths
        ],
        retrive_index=[[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]],
        retrive_next_token=[[1, 3, 4, 5, 6, 7, -1, -1], [1, 2, -1, 6, -1, -1, 7, -1]],
        retrive_next_sibling=[
            [-1, 2, -1, -1, -1, -1, -1, -1],
            [-1, -1, 3, 4, 5, -1, -1, -1],
        ],
    )
    for name, values in expected.items():
        assert_accuracy(
            outputs[name],
            torch.tensor(values, dtype=torch.int64, device=device),
            path=name,
        )
    # Explicit ancestor sets from the fixture's expected child/sibling lists.
    ancestors = [
        [
            [0],
            [0, 1],
            [0, 2],
            [0, 1, 3],
            [0, 2, 4],
            [0, 1, 3, 5],
            [0, 2, 4, 6],
            [0, 1, 3, 5, 7],
        ],
        [
            [0],
            [0, 1],
            [0, 1, 2],
            [0, 1, 3],
            [0, 1, 4],
            [0, 1, 5],
            [0, 1, 3, 6],
            [0, 1, 3, 6, 7],
        ],
    ]
    offset = 0
    for length, rows in zip(prefix_lengths, ancestors):
        columns = length + 8
        actual = outputs["tree_mask"][offset : offset + 8 * columns].view(8, columns)
        golden = torch.zeros((8, 8), dtype=torch.bool, device=device)
        for row, nodes in enumerate(rows):
            golden[row, nodes] = True
        assert_accuracy(actual[:, -8:], golden, path="tree_mask.ancestors")
        assert_accuracy(
            actual[:, :length],
            torch.ones_like(actual[:, :length]),
            path="tree_mask.prefix",
        )
        offset += 8 * columns


@pytest.mark.parametrize(
    "prefix_lengths",
    [
        pytest.param((5, 10), id="full-mask-b2-topk4-depth4-draft8-prefix5-10"),
        pytest.param(
            (0, 0),
            marks=pytest.mark.extra_checks,
            id="full-mask-b2-topk4-depth4-draft8-prefix0-0",
        ),
    ],
)
def test_build_tree(prefix_lengths):
    from sgl_kernel_npu.operators.speculative import build_tree

    impl, impl_ref = get_test_impls()
    skip_accuracy = bool(get_skip_acc_check())
    if skip_accuracy:
        require_test_impl(build_tree, impl)
    else:
        require_distinct_impls(build_tree, impl, impl_ref)
        results = []
        for backend in (impl_ref, impl):
            inputs, outputs = generate_test_data("npu", prefix_lengths)
            identities = {
                name: (id(value), value.data_ptr()) for name, value in outputs.items()
            }
            snapshots = {
                name: value.clone()
                for name, value in inputs.items()
                if isinstance(value, torch.Tensor)
            }
            result = build_tree(**inputs, **outputs, tree_mask_mode=0, impl=backend)
            assert result is None
            for name, value in outputs.items():
                assert (id(value), value.data_ptr()) == identities[name], name
            for name, value in snapshots.items():
                assert_accuracy(inputs[name], value, path=f"unchanged_input.{name}")
            assert_fixture_expectations(outputs, prefix_lengths)
            results.append(outputs)
        assert_accuracy(results[1], results[0])

    # FULL_MASK overwrites the same tree area and preserves prefix columns.
    inputs, outputs = generate_test_data("npu", prefix_lengths)
    executor = make_executor(
        build_tree,
        kwargs={**inputs, **outputs, "tree_mask_mode": 0},
        impl=impl,
        impl_ref=impl_ref,
        accuracy_checked=not skip_accuracy,
    )
    check_performance(executor, enable_perf=prefix_lengths == (5, 10))
