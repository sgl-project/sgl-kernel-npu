"""Exact legacy fixture through the new public interface.

Fixture source: tests/python/sgl_kernel_npu/test_build_tree.py at the PR #976
baseline. parent_list is cat(parents_list[:-1]); selected_index is the sorted
top-7 score indices from that fixture, with no changed scores or dimensions.
The low-level tree operator does not consume token IDs or scores themselves.

This file also supports separate-installation replay. Capture old outputs with
--impl legacy --save PATH, then pass --compare-legacy PATH in the new installation.
QLEN_ONLY is an explicit diagnostic, not a passing default regression: old/new
are compared first, then mathematical comparison is allowed to fail visibly.
"""

import argparse
from pathlib import Path

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

# B, topk, depth, draft_token_num and verified prefix lengths of the old fixture.
perf_shape = [(2, 4, 4, 8, (5, 10))]


def legacy_fixture(device):
    """Inputs reaching the old kernel: B=2, k=4, depth=4, D=8, L=[5,10]."""
    return dict(
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
        verified_seq_len=torch.tensor([5, 10], dtype=torch.int64, device=device),
        topk=4,
        depth=4,
        draft_token_num=8,
    )


def output_buffers(inputs, tree_mask_mode):
    device = inputs["parent_list"].device
    batch = inputs["parent_list"].shape[0]
    width = inputs["draft_token_num"]
    size = batch * width * width
    if tree_mask_mode == 0:
        size += int(inputs["verified_seq_len"].sum()) * width
    return dict(
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


def _assert_fixture_expectations(outputs, mode):
    device = outputs["positions"].device
    expected = dict(
        positions=[5, 6, 6, 7, 7, 8, 8, 9, 10, 11, 12, 12, 12, 12, 13, 14],
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
    for length, rows in zip((5, 10), ancestors):
        columns = length + 8 if mode == 0 else 8
        actual = outputs["tree_mask"][offset : offset + 8 * columns].view(8, columns)
        golden = torch.zeros((8, 8), dtype=torch.bool, device=device)
        for row, nodes in enumerate(rows):
            golden[row, nodes] = True
        assert_accuracy(actual[:, -8:], golden, path="tree_mask.ancestors")
        if mode == 0:
            assert_accuracy(
                actual[:, :length],
                torch.ones_like(actual[:, :length]),
                path="tree_mask.prefix",
            )
        offset += 8 * columns


def _run(operator, inputs, mode, impl):
    outputs = output_buffers(inputs, mode)
    identities = {
        name: (id(value), value.data_ptr()) for name, value in outputs.items()
    }
    snapshots = {
        name: value.clone()
        for name, value in inputs.items()
        if isinstance(value, torch.Tensor)
    }
    result = operator(**inputs, **outputs, tree_mask_mode=mode, impl=impl)
    assert result is None
    for name, value in outputs.items():
        assert (id(value), value.data_ptr()) == identities[name], name
    for name, value in snapshots.items():
        assert_accuracy(inputs[name], value, path=f"unchanged_input.{name}")
    return outputs


def test_build_tree_legacy_full_mask_fixture(impl_pair):
    from sgl_kernel_npu.operators.speculative import build_tree

    impl, impl_ref = get_test_impls()
    inputs = legacy_fixture("npu")
    skip_accuracy = bool(get_skip_acc_check())
    if skip_accuracy:
        require_test_impl(build_tree, impl)
        print("build_tree accuracy comparison skipped; no accuracy pass recorded")
    else:
        require_distinct_impls(build_tree, impl, impl_ref)
        reference = _run(build_tree, inputs, 0, impl_ref)
        _assert_fixture_expectations(reference, 0)
        actual = _run(build_tree, inputs, 0, impl)
        assert_accuracy(actual, reference)
        _assert_fixture_expectations(actual, 0)
    # Repeated FULL_MASK calls overwrite the same tree area and preserve prefix
    # columns, so the original buffers can be reused without reset/copy kernels.
    outputs = output_buffers(inputs, 0)
    executor = make_executor(
        build_tree,
        kwargs={**inputs, **outputs, "tree_mask_mode": 0},
        impl=impl,
        impl_ref=impl_ref,
        accuracy_checked=not skip_accuracy,
    )
    shape = (
        inputs["parent_list"].shape[0],
        inputs["topk"],
        inputs["depth"],
        inputs["draft_token_num"],
        tuple(inputs["verified_seq_len"].cpu().tolist()),
    )
    check_performance(executor, enable_perf=shape in perf_shape)


def _legacy_operator(*, impl, **kwargs):
    import sgl_kernel_npu  # noqa: F401: register the legacy extension in this process

    order = (
        "parent_list",
        "selected_index",
        "verified_seq_len",
        "tree_mask",
        "positions",
        "retrive_index",
        "retrive_next_token",
        "retrive_next_sibling",
        "topk",
        "depth",
        "draft_token_num",
        "tree_mask_mode",
    )
    return torch.ops.npu.build_tree_kernel_efficient(*(kwargs[name] for name in order))


def _replay_main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--impl", choices=("legacy", "ascendc", "torch_native"), required=True
    )
    parser.add_argument("--mode", choices=("full", "qlen"), default="full")
    parser.add_argument("--save", type=Path)
    parser.add_argument("--compare-legacy", type=Path)
    args = parser.parse_args()
    if args.impl == "legacy" and args.save is None:
        parser.error("legacy capture requires --save PATH")
    if args.impl == "ascendc" and args.compare_legacy is None:
        parser.error(
            "Ascend C replay requires --compare-legacy PATH for old/new comparison"
        )
    import torch_npu  # noqa: F401: make the NPU device available

    mode = 0 if args.mode == "full" else 1
    inputs = legacy_fixture("npu")
    if args.impl == "legacy":
        operator = _legacy_operator
    else:
        from sgl_kernel_npu.operators.speculative import build_tree

        operator = build_tree
    outputs = _run(operator, inputs, mode, args.impl)
    cpu_inputs = {
        k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()
    }
    cpu_outputs = {k: v.cpu() for k, v in outputs.items()}
    # Preserve a diagnostic artifact before the mathematical assertion can fail.
    if args.save is not None:
        torch.save(
            dict(inputs=cpu_inputs, mode=mode, outputs=cpu_outputs, impl=args.impl),
            args.save,
        )
    if args.compare_legacy is not None:
        captured = torch.load(
            args.compare_legacy, map_location="cpu", weights_only=True
        )
        assert (
            captured["impl"] == "legacy"
        ), "comparison artifact must come from the old implementation"
        assert captured["mode"] == mode
        assert_accuracy(cpu_inputs, captured["inputs"], path="replay.inputs")
        assert_accuracy(cpu_outputs, captured["outputs"], path="new_vs_old")
        print("Old/new direct comparison passed (all five outputs)")
    if args.impl != "legacy":
        from sgl_kernel_npu.operators.speculative import build_tree

        reference = _run(build_tree, inputs, mode, "torch_native")
        _assert_fixture_expectations(reference, mode)
        if args.impl == "torch_native":
            print(
                "Independent reference matched the fixture's explicit tree expectations"
            )
            return
        # Run both mathematical comparisons even when the retained QLEN_ONLY bug
        # makes them fail. Report those failures, never xfail, skip or relax them.
        failures = []
        comparisons = [(outputs, reference, "new_vs_mathematics")]
        if args.compare_legacy is not None:
            cpu_reference = {k: v.cpu() for k, v in reference.items()}
            comparisons.append(
                (captured["outputs"], cpu_reference, "old_vs_mathematics")
            )
        for actual, expected, label in comparisons:
            try:
                assert_accuracy(actual, expected, path=label)
            except AssertionError as error:
                failures.append(str(error))
        if failures:
            raise AssertionError("\n\n".join(failures))
        print("Old/reference and new/reference comparisons passed (all five outputs)")
    elif mode == 0:
        _assert_fixture_expectations(outputs, mode)


if __name__ == "__main__":
    _replay_main()
