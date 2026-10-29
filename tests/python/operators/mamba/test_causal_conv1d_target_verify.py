"""In-place target-verify tests: five linear cases and three tree cases.

Inputs are named operator kwargs. DUT/reference use separate copies; the
original inputs remain available for read-only-state and untouched-slot checks.
"""

import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import assert_accuracy
from tests.python.utils.common import (
    get_skip_acc_check,
    get_test_impls,
    require_distinct_impls,
    require_test_impl,
)
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard

# Borrowed model/sharding dimensions and a synthetic batch, not captured inputs.
perf_shape = [(64, 3072, 4)]
BF16_STANDARD = {"torch.bfloat16": AccuracyStandard(atol=5e-2, rtol=1e-2)}
EXACT_BF16_STANDARD = {"torch.bfloat16": AccuracyStandard(atol=0, rtol=0)}


def clone_test_tensor(tensor):
    """Copy storage before restoring this test's channel-last views."""
    if tensor.is_contiguous():
        return tensor.clone()
    # CANN 9.0 clone can make a transposed view contiguous.
    base = tensor.transpose(-1, -2)
    assert base.is_contiguous()
    return base.clone().transpose(-1, -2)


def generate_test_data(
    width, extra_window=0, *, tree=False, channel_last=False, batch=2, dim=5, steps=4
):
    """Return the same named operator inputs for every mode and layout."""
    if channel_last:
        torch.npu.manual_seed(303)
        # Logical [B,C,S] views over channel-last physical storage.
        x = torch.randn(batch, steps, dim, dtype=torch.bfloat16).transpose(1, 2)
        scratch = torch.full(
            (batch, steps, width - 1, dim), -23.0, dtype=torch.bfloat16
        ).transpose(-1, -2)
        conv_state = torch.randn(
            batch + 1, width - 1, dim, dtype=torch.bfloat16
        ).transpose(-1, -2)
        weight = torch.randn(width, dim, dtype=torch.bfloat16) * 0.25
        bias = None
        state_indices = torch.arange(batch, dtype=torch.int32).flip(0)
        scratch_indices = torch.arange(batch, dtype=torch.int32)
    else:
        torch.npu.manual_seed(202 if tree else 101)
        x = torch.randn(batch, dim, steps, dtype=torch.bfloat16)
        weight = torch.randn(dim, width, dtype=torch.bfloat16) * 0.25
        bias = None if tree else torch.randn(dim, dtype=torch.bfloat16) * 0.1
        conv_state = torch.randn(4 if tree else 5, dim, width - 1, dtype=torch.bfloat16)
        state_indices = torch.tensor([2] if tree else [3, 1], dtype=torch.int32)
        scratch_indices = torch.tensor([1] if tree else [2, 0], dtype=torch.int32)
        scratch = torch.full(
            (3 if tree else 4, steps + 1, dim, width - 1 + extra_window),
            -19.0 if tree else -23.0,
            dtype=torch.bfloat16,
        )
        if tree:
            weight = weight.transpose(0, 1).contiguous()

    # Tree: 0 -> 1; 1 has children 2 and 3; 3 -> 4.
    return dict(
        x=x,
        conv_state=conv_state,
        weight=weight,
        bias=bias,
        activation=None if tree else "silu",
        conv_state_indices=state_indices,
        intermediate_conv_window=scratch,
        intermediate_state_indices=scratch_indices,
        retrieve_next_token=(
            torch.tensor([[1, 2, -1, 4, -1]], dtype=torch.int32) if tree else None
        ),
        retrieve_next_sibling=(
            torch.tensor([[-1, -1, 3, -1, -1]], dtype=torch.int32) if tree else None
        ),
        retrieve_parent_token=(
            torch.full((batch, steps + 1), -77, dtype=torch.int32) if tree else None
        ),
    )


@pytest.mark.parametrize(
    "case",
    [
        # Case 01: Linear width2 with an exact scratch window.
        pytest.param(dict(width=2), id="linear-width2"),
        # Case 02: Extra scratch slot must remain untouched, width2.
        pytest.param(dict(width=2, extra_window=1), id="linear-width2-extra-window"),
        # Case 03: Extra scratch slot must remain untouched, width3.
        pytest.param(dict(width=3, extra_window=1), id="linear-width3-extra-window"),
        # Case 04: Common width4 with an exact scratch window.
        pytest.param(dict(width=4), id="linear-width4"),
        # Case 05: Typical channel-last B64/C3072/S4; also profile this case.
        pytest.param(
            dict(width=4, channel_last=True, batch=64, dim=3072, steps=4),
            id="typical-channel-last-b64-c3072-s4",
        ),
    ],
)
def test_causal_conv1d_target_verify_linear_npu(case):
    """In-place: overwrite x/scratch while preserving committed conv_state."""
    from sgl_kernel_npu.operators.mamba import causal_conv1d_target_verify

    impl, impl_ref = get_test_impls()
    accuracy_checked = get_skip_acc_check() is not True
    if accuracy_checked:
        require_distinct_impls(causal_conv1d_target_verify, impl, impl_ref)
    else:
        require_test_impl(causal_conv1d_target_verify, impl)
    shape = (case.get("batch", 2), case.get("dim", 5), case.get("steps", 4))
    profile_case = shape in perf_shape
    if not accuracy_checked and not profile_case:
        return

    # Inputs: one layout definition is shared by accuracy and performance.
    inputs = generate_test_data(**case)
    if case.get("channel_last", False):
        assert inputs["x"].stride(1) == 1 and not inputs["x"].is_contiguous()
        assert inputs["weight"].is_contiguous()

    # Accuracy: the original inputs stay untouched for state comparisons.
    if accuracy_checked:
        actual = {
            name: clone_test_tensor(value) if isinstance(value, torch.Tensor) else value
            for name, value in inputs.items()
        }
        reference = {
            name: clone_test_tensor(value) if isinstance(value, torch.Tensor) else value
            for name, value in inputs.items()
        }
        if case.get("channel_last", False):
            for name in ("x", "conv_state", "intermediate_conv_window"):
                assert actual[name].stride() == inputs[name].stride()
                assert reference[name].stride() == inputs[name].stride()
            reference["weight"] = inputs["weight"].transpose(0, 1).contiguous()
        expected_out = causal_conv1d_target_verify(**reference, impl=impl_ref)
        out = causal_conv1d_target_verify(**actual, impl=impl)

        assert out.data_ptr() == actual["x"].data_ptr()
        assert not torch.equal(actual["x"], inputs["x"])
        assert_accuracy(out, expected_out, BF16_STANDARD, path="linear.output")
        assert_accuracy(
            actual["intermediate_conv_window"],
            reference["intermediate_conv_window"],
            EXACT_BF16_STANDARD,
            path="linear.scratch",
        )
        assert_accuracy(
            actual["conv_state"],
            inputs["conv_state"],
            EXACT_BF16_STANDARD,
            path="linear.conv_state",
        )

    if not profile_case:
        return

    # Performance: prepare all fresh x/scratch states during the unprofiled call.
    warmup, active = 5, 5
    states = None

    def executor():
        nonlocal states
        if states is None:
            states = iter(
                [
                    dict(
                        inputs,
                        x=clone_test_tensor(inputs["x"]),
                        intermediate_conv_window=clone_test_tensor(
                            inputs["intermediate_conv_window"]
                        ),
                    )
                    for _ in range(1 + warmup + active)
                ]
            )
        return causal_conv1d_target_verify(**next(states), impl=impl)

    executor.func = causal_conv1d_target_verify
    executor.impl = impl
    executor.impl_ref = impl_ref
    executor.accuracy_checked = accuracy_checked
    check_performance(executor, enable_perf=profile_case, warmup=warmup, active=active)


@pytest.mark.parametrize(
    "width",
    [
        # Case 06: Tree width2: child and sibling traversal.
        2,
        # Case 07: Tree width3: child and sibling traversal.
        3,
        # Case 08: Tree width4: child and sibling traversal.
        4,
    ],
)
def test_causal_conv1d_target_verify_tree_npu(width):
    """In-place state: write scratch/parents; preserve x and conv_state."""
    from sgl_kernel_npu.operators.mamba import causal_conv1d_target_verify

    impl, impl_ref = get_test_impls()
    if get_skip_acc_check() is True:
        require_test_impl(causal_conv1d_target_verify, impl)
        return
    require_distinct_impls(causal_conv1d_target_verify, impl, impl_ref)

    inputs = generate_test_data(width, tree=True, batch=1, steps=5)
    actual = {
        name: value.clone() if isinstance(value, torch.Tensor) else value
        for name, value in inputs.items()
    }
    reference = {
        name: value.clone() if isinstance(value, torch.Tensor) else value
        for name, value in inputs.items()
    }
    reference["weight"] = inputs["weight"].transpose(0, 1).contiguous()
    expected_out = causal_conv1d_target_verify(**reference, impl=impl_ref)
    out = causal_conv1d_target_verify(**actual, impl=impl)

    assert out.data_ptr() != actual["x"].data_ptr()
    assert_accuracy(out, expected_out, BF16_STANDARD, path="tree.output")
    assert_accuracy(
        actual["retrieve_parent_token"],
        reference["retrieve_parent_token"],
        path="tree.parent",
    )
    assert_accuracy(
        actual["intermediate_conv_window"],
        reference["intermediate_conv_window"],
        EXACT_BF16_STANDARD,
        path="tree.scratch",
    )
    assert_accuracy(
        actual["conv_state"],
        inputs["conv_state"],
        EXACT_BF16_STANDARD,
        path="tree.conv_state",
    )
    assert_accuracy(
        actual["x"], inputs["x"], EXACT_BF16_STANDARD, path="tree.unchanged_x"
    )
