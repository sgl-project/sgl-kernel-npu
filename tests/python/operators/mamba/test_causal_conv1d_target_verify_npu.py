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

# C: TP16/attn-DP2 example gives C=3072; W: B=64; P: linear D=4.
perf_shape = [(64, 3072, 4)]

BF16_STANDARD = {
    "torch.bfloat16": AccuracyStandard(atol=5e-2, rtol=1e-2),
}
EXACT_BF16_STANDARD = {
    "torch.bfloat16": AccuracyStandard(atol=0, rtol=0),
}


def _native_target_verify_reference(
    x,
    conv_state,
    weight,
    bias,
    activation,
    conv_state_indices,
    intermediate_conv_window,
    intermediate_state_indices,
    retrieve_next_token=None,
    retrieve_next_sibling=None,
    retrieve_parent_token=None,
):
    from sgl_kernel_npu.operators.mamba import causal_conv1d_target_verify_npu

    _, impl_ref = get_test_impls()
    x_ref = x.clone()
    scratch_ref = intermediate_conv_window.clone()
    parent_ref = (
        None if retrieve_parent_token is None else retrieve_parent_token.clone()
    )
    out = causal_conv1d_target_verify_npu(
        x_ref,
        conv_state.clone(),
        weight,
        bias,
        activation,
        conv_state_indices,
        scratch_ref,
        intermediate_state_indices,
        retrieve_next_token,
        retrieve_next_sibling,
        parent_ref,
        impl=impl_ref,
    )
    return out, scratch_ref, parent_ref


def generate_linear_data(width, extra_window=0):
    torch.npu.manual_seed(101)
    batch, dim, steps = 2, 5, 4
    x = torch.randn(batch, dim, steps, dtype=torch.bfloat16)
    weight = torch.randn(dim, width, dtype=torch.bfloat16) * 0.25
    bias = torch.randn(dim, dtype=torch.bfloat16) * 0.1
    conv_state = torch.randn(5, dim, width - 1, dtype=torch.bfloat16)
    conv_state_indices = torch.tensor([3, 1], dtype=torch.int32)
    intermediate_state_indices = torch.tensor([2, 0], dtype=torch.int32)
    intermediate_conv_window = torch.full(
        (4, steps + 1, dim, width - 1 + extra_window), -23.0, dtype=torch.bfloat16
    )

    return (
        x,
        weight,
        bias,
        conv_state,
        conv_state_indices,
        intermediate_state_indices,
        intermediate_conv_window,
    )


# Minimum width, both extra-window regressions, and the common width=4 path.
@pytest.mark.parametrize("width,extra_window", [(2, 0), (2, 1), (3, 1), (4, 0)])
def test_causal_conv1d_target_verify_linear_npu(width, extra_window):
    from sgl_kernel_npu.operators.mamba import causal_conv1d_target_verify_npu

    impl, impl_ref = get_test_impls()
    if get_skip_acc_check() is not True:
        require_distinct_impls(causal_conv1d_target_verify_npu, impl, impl_ref)
    else:
        require_test_impl(causal_conv1d_target_verify_npu, impl)
    if get_skip_acc_check() is True:
        return
    (
        x,
        weight,
        bias,
        conv_state,
        conv_state_indices,
        intermediate_state_indices,
        intermediate_conv_window,
    ) = generate_linear_data(width, extra_window)
    x_before = x.clone()
    conv_state_before = conv_state.clone()
    expected_out, expected_scratch, _ = _native_target_verify_reference(
        x_before,
        conv_state_before,
        weight,
        bias,
        "silu",
        conv_state_indices,
        intermediate_conv_window,
        intermediate_state_indices,
    )

    out = causal_conv1d_target_verify_npu(
        x=x,
        conv_state=conv_state,
        weight=weight,
        bias=bias,
        activation="silu",
        conv_state_indices=conv_state_indices,
        intermediate_conv_window=intermediate_conv_window,
        intermediate_state_indices=intermediate_state_indices,
        impl=impl,
    )

    assert out.data_ptr() == x.data_ptr()
    assert not torch.equal(x, x_before)
    assert_accuracy(out, expected_out, BF16_STANDARD, path="linear.output")
    assert_accuracy(x, expected_out, BF16_STANDARD, path="linear.overwritten_x")
    assert_accuracy(
        intermediate_conv_window,
        expected_scratch,
        EXACT_BF16_STANDARD,
        path="linear.scratch",
    )
    assert_accuracy(
        conv_state,
        conv_state_before,
        EXACT_BF16_STANDARD,
        path="linear.conv_state",
    )


def generate_test_data(batch, dim, steps):
    torch.npu.manual_seed(303)
    width = 4
    # P: KDA passes transposed views over channel-last physical storage.
    initial_x_base = torch.randn(batch, steps, dim, dtype=torch.bfloat16)
    initial_scratch_base = torch.full(
        (batch, steps, width - 1, dim), -23.0, dtype=torch.bfloat16
    )
    conv_state_base = torch.randn(batch + 1, width - 1, dim, dtype=torch.bfloat16)
    weight = torch.randn(width, dim, dtype=torch.bfloat16) * 0.25
    conv_state_indices = torch.arange(batch, dtype=torch.int32).flip(0)
    intermediate_state_indices = torch.arange(batch, dtype=torch.int32)
    return (
        initial_x_base,
        initial_scratch_base,
        conv_state_base,
        weight,
        conv_state_indices,
        intermediate_state_indices,
    )


@pytest.mark.parametrize(
    "batch,dim,steps",
    [pytest.param(64, 3072, 4, id="W-glm53-flash-linear-b64-c3072-d4")],
)
def test_causal_conv1d_target_verify_production_linear_npu(batch, dim, steps):
    from sgl_kernel_npu.operators.mamba import causal_conv1d_target_verify_npu

    (
        initial_x_base,
        initial_scratch_base,
        conv_state_base,
        weight,
        conv_state_indices,
        intermediate_state_indices,
    ) = generate_test_data(batch, dim, steps)
    assert weight.is_contiguous()

    def make_views(x_base, scratch_base):
        return (
            x_base.transpose(1, 2),
            conv_state_base.transpose(-1, -2),
            scratch_base.transpose(-1, -2),
        )

    conv_state = conv_state_base.transpose(-1, -2)
    impl, impl_ref = get_test_impls()
    if get_skip_acc_check() is not True:
        require_distinct_impls(causal_conv1d_target_verify_npu, impl, impl_ref)
    else:
        require_test_impl(causal_conv1d_target_verify_npu, impl)
    accuracy_checked = get_skip_acc_check() is not True
    if accuracy_checked:
        x, conv_state, intermediate_conv_window = make_views(
            initial_x_base.clone(), initial_scratch_base.clone()
        )
        assert x.stride(1) == 1 and not x.is_contiguous()
        x_before = initial_x_base.transpose(1, 2).clone()
        conv_state_before = conv_state.clone()
        expected_out, expected_scratch, _ = _native_target_verify_reference(
            x_before,
            conv_state_before,
            weight.transpose(0, 1).contiguous(),
            None,
            "silu",
            conv_state_indices,
            initial_scratch_base.transpose(-1, -2),
            intermediate_state_indices,
        )

        out = causal_conv1d_target_verify_npu(
            x=x,
            conv_state=conv_state,
            weight=weight,
            bias=None,
            activation="silu",
            conv_state_indices=conv_state_indices,
            intermediate_conv_window=intermediate_conv_window,
            intermediate_state_indices=intermediate_state_indices,
            impl=impl,
        )
        assert out.data_ptr() == x.data_ptr()
        assert_accuracy(
            out, expected_out, BF16_STANDARD, path="production_linear.output"
        )
        assert_accuracy(
            intermediate_conv_window,
            expected_scratch,
            EXACT_BF16_STANDARD,
            path="production_linear.scratch",
        )
        assert_accuracy(
            conv_state,
            conv_state_before,
            EXACT_BF16_STANDARD,
            path="production_linear.conv_state",
        )

    snapshots = None
    call_index = 0

    def executor():
        nonlocal snapshots, call_index
        if snapshots is None:
            snapshots = [
                (initial_x_base.clone(), initial_scratch_base.clone())
                for _ in range(11)
            ]
        x_base, scratch_base = snapshots[call_index]
        call_index += 1
        perf_x, _, perf_scratch = make_views(x_base, scratch_base)
        causal_conv1d_target_verify_npu(
            x=perf_x,
            conv_state=conv_state,
            weight=weight,
            bias=None,
            activation="silu",
            conv_state_indices=conv_state_indices,
            intermediate_conv_window=perf_scratch,
            intermediate_state_indices=intermediate_state_indices,
            impl=impl,
        )

    executor.func = causal_conv1d_target_verify_npu
    executor.impl = impl
    executor.impl_ref = impl_ref
    executor.accuracy_checked = accuracy_checked
    check_performance(executor, enable_perf=(batch, dim, steps) in perf_shape)


def generate_tree_data(width):
    torch.npu.manual_seed(202)
    batch, dim, steps = 1, 5, 5
    x = torch.randn(batch, dim, steps, dtype=torch.bfloat16)
    weight = torch.randn(dim, width, dtype=torch.bfloat16) * 0.25
    conv_state = torch.randn(4, dim, width - 1, dtype=torch.bfloat16)
    conv_state_indices = torch.tensor([2], dtype=torch.int32)
    intermediate_state_indices = torch.tensor([1], dtype=torch.int32)
    intermediate_conv_window = torch.full(
        (3, steps + 1, dim, width - 1), -19.0, dtype=torch.bfloat16
    )

    # Tree edges: 0 -> 1 -> 2, 2's sibling 3, and 3 -> 4.
    retrieve_next_token = torch.tensor([[1, 2, -1, 4, -1]], dtype=torch.int32)
    retrieve_next_sibling = torch.tensor([[-1, -1, 3, -1, -1]], dtype=torch.int32)
    retrieve_parent_token = torch.full((batch, steps + 1), -77, dtype=torch.int32)

    return (
        x,
        weight,
        conv_state,
        conv_state_indices,
        intermediate_state_indices,
        intermediate_conv_window,
        retrieve_next_token,
        retrieve_next_sibling,
        retrieve_parent_token,
    )


@pytest.mark.parametrize("width", [2, 3, 4])
def test_causal_conv1d_target_verify_tree_npu(width):
    from sgl_kernel_npu.operators.mamba import causal_conv1d_target_verify_npu

    impl, impl_ref = get_test_impls()
    if get_skip_acc_check() is not True:
        require_distinct_impls(causal_conv1d_target_verify_npu, impl, impl_ref)
    else:
        require_test_impl(causal_conv1d_target_verify_npu, impl)
    if get_skip_acc_check() is True:
        return
    (
        x,
        weight,
        conv_state,
        conv_state_indices,
        intermediate_state_indices,
        intermediate_conv_window,
        retrieve_next_token,
        retrieve_next_sibling,
        retrieve_parent_token,
    ) = generate_tree_data(width)
    x_before = x.clone()
    conv_state_before = conv_state.clone()
    expected_out, expected_scratch, expected_parent = _native_target_verify_reference(
        x_before,
        conv_state_before,
        weight,
        None,
        None,
        conv_state_indices,
        intermediate_conv_window,
        intermediate_state_indices,
        retrieve_next_token=retrieve_next_token,
        retrieve_next_sibling=retrieve_next_sibling,
        retrieve_parent_token=retrieve_parent_token,
    )

    out = causal_conv1d_target_verify_npu(
        x=x,
        conv_state=conv_state,
        weight=weight.transpose(0, 1).contiguous(),
        bias=None,
        activation=None,
        conv_state_indices=conv_state_indices,
        intermediate_conv_window=intermediate_conv_window,
        intermediate_state_indices=intermediate_state_indices,
        retrieve_next_token=retrieve_next_token,
        retrieve_next_sibling=retrieve_next_sibling,
        retrieve_parent_token=retrieve_parent_token,
        impl=impl,
    )

    assert out.data_ptr() != x.data_ptr()
    assert_accuracy(out, expected_out, BF16_STANDARD, path="tree.output")
    assert_accuracy(
        retrieve_parent_token,
        expected_parent,
        path="tree.parent",
    )
    assert_accuracy(
        intermediate_conv_window,
        expected_scratch,
        EXACT_BF16_STANDARD,
        path="tree.scratch",
    )
    assert_accuracy(
        conv_state,
        conv_state_before,
        EXACT_BF16_STANDARD,
        path="tree.conv_state",
    )
    assert_accuracy(x, x_before, EXACT_BF16_STANDARD, path="tree.unchanged_x")
