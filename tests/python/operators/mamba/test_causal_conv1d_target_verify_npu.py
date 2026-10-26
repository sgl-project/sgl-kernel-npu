import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import assert_accuracy
from tests.python.utils.common import (
    get_skip_acc_check,
    get_test_impls,
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


def _torch_target_verify_reference(
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
    """Compute outputs and per-token causal windows from original inputs only."""
    device = x.device
    dtype = x.dtype
    x_ref = x.detach().cpu()
    state_ref = conv_state.detach().cpu()
    weight_ref = weight.detach().cpu().float()
    bias_ref = None if bias is None else bias.detach().cpu().float()
    state_indices = conv_state_indices.detach().cpu().tolist()
    scratch_indices = intermediate_state_indices.detach().cpu().tolist()
    expected_scratch = intermediate_conv_window.detach().cpu().clone()
    expected_parent = (
        None
        if retrieve_parent_token is None
        else retrieve_parent_token.detach().cpu().clone()
    )
    has_tree = retrieve_next_token is not None
    next_tokens = retrieve_next_token.detach().cpu().tolist() if has_tree else None
    next_siblings = retrieve_next_sibling.detach().cpu().tolist() if has_tree else None

    batch, dim, steps = x_ref.shape
    width = weight_ref.shape[1]
    expected_out = x_ref.new_empty(x_ref.shape, dtype=torch.float32)

    for batch_idx in range(batch):
        history = state_ref[state_indices[batch_idx], :, : width - 1].float()
        scratch_idx = scratch_indices[batch_idx]
        parents = [0] * steps if has_tree else None

        for token_idx in range(steps):
            acc = (
                bias_ref.clone()
                if bias_ref is not None
                else torch.zeros_like(weight_ref[:, 0])
            )

            if not has_tree:
                current = x_ref[batch_idx, :, token_idx].float()
                for tap in range(width - 1):
                    acc += history[:, tap] * weight_ref[:, tap]
                acc += current * weight_ref[:, width - 1]

                history = torch.cat((history[:, 1:], current.unsqueeze(1)), dim=1)
                expected_scratch[scratch_idx, token_idx] = history.to(dtype)
            else:
                child_idx = next_tokens[batch_idx][token_idx]
                if child_idx != -1:
                    parents[child_idx] = token_idx
                sibling_idx = next_siblings[batch_idx][token_idx]
                if sibling_idx != -1:
                    parents[sibling_idx] = parents[token_idx]

                ancestor_idx = token_idx
                value = x_ref[batch_idx, :, token_idx].float()
                for tap in range(width):
                    acc += value * weight_ref[:, width - tap - 1]
                    if tap == width - 1:
                        continue

                    expected_scratch[scratch_idx, token_idx, :, width - tap - 2] = (
                        value.to(dtype)
                    )
                    if ancestor_idx > 0:
                        ancestor_idx = parents[ancestor_idx]
                        value = x_ref[batch_idx, :, ancestor_idx].float()
                    else:
                        history_idx = width - 2 + ancestor_idx
                        value = history[:, history_idx]
                        ancestor_idx -= 1

            if activation in ("silu", "swish"):
                acc = torch.nn.functional.silu(acc)
            expected_out[batch_idx, :, token_idx] = acc

        if has_tree:
            expected_parent[batch_idx, :steps] = expected_parent.new_tensor(parents)

    expected_out = expected_out.to(dtype).to(device)
    expected_scratch = expected_scratch.to(device)
    if expected_parent is not None:
        expected_parent = expected_parent.to(device)
    return expected_out, expected_scratch, expected_parent


def test_causal_conv1d_target_verify_linear_npu():
    from sgl_kernel_npu.operators.mamba import causal_conv1d_target_verify_npu

    impl, _ = get_test_impls()
    require_test_impl(causal_conv1d_target_verify_npu, impl)
    if get_skip_acc_check() is True:
        return
    torch.npu.manual_seed(101)
    batch, dim, steps, width = 2, 3, 4, 4
    x = torch.randn(batch, dim, steps, dtype=torch.bfloat16)
    weight = torch.randn(dim, width, dtype=torch.bfloat16) * 0.25
    bias = torch.randn(dim, dtype=torch.bfloat16) * 0.1
    conv_state = torch.randn(5, dim, width - 1, dtype=torch.bfloat16)
    conv_state_indices = torch.tensor([3, 1], dtype=torch.int32)
    intermediate_state_indices = torch.tensor([2, 0], dtype=torch.int32)
    intermediate_conv_window = torch.full(
        (4, steps + 1, dim, width - 1), -23.0, dtype=torch.bfloat16
    )

    x_before = x.clone()
    conv_state_before = conv_state.clone()
    expected_out, expected_scratch, _ = _torch_target_verify_reference(
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


@pytest.mark.parametrize(
    "batch,dim,steps",
    [pytest.param(64, 3072, 4, id="W-glm53-flash-linear-b64-c3072-d4")],
)
def test_causal_conv1d_target_verify_production_linear_npu(batch, dim, steps):
    from sgl_kernel_npu.operators.mamba import causal_conv1d_target_verify_npu

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
    assert weight.is_contiguous()

    def make_views(x_base, scratch_base):
        return (
            x_base.transpose(1, 2),
            conv_state_base.transpose(-1, -2),
            scratch_base.transpose(-1, -2),
        )

    conv_state = conv_state_base.transpose(-1, -2)
    impl, _ = get_test_impls()
    require_test_impl(causal_conv1d_target_verify_npu, impl)
    accuracy_checked = get_skip_acc_check() is not True
    if accuracy_checked:
        x, conv_state, intermediate_conv_window = make_views(
            initial_x_base.clone(), initial_scratch_base.clone()
        )
        assert x.stride(1) == 1 and not x.is_contiguous()
        x_before = initial_x_base.transpose(1, 2).clone()
        conv_state_before = conv_state.clone()
        expected_out, expected_scratch, _ = _torch_target_verify_reference(
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
    executor.impl_ref = "torch_reference"
    executor.accuracy_checked = accuracy_checked
    check_performance(executor, enable_perf=(batch, dim, steps) in perf_shape)


def test_causal_conv1d_target_verify_tree_npu():
    from sgl_kernel_npu.operators.mamba import causal_conv1d_target_verify_npu

    impl, _ = get_test_impls()
    require_test_impl(causal_conv1d_target_verify_npu, impl)
    if get_skip_acc_check() is True:
        return
    torch.npu.manual_seed(202)
    batch, dim, steps, width = 1, 2, 5, 3
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

    x_before = x.clone()
    conv_state_before = conv_state.clone()
    expected_out, expected_scratch, expected_parent = _torch_target_verify_reference(
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
