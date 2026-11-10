"""CGDR accuracy and state contracts through the public attention interface."""

import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import assert_accuracy, check_accuracy
from tests.python.utils.common import (
    get_skip_acc_check,
    get_test_impls,
    make_executor,
    require_distinct_impls,
    require_test_impl,
)
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard

# The kernel rounds some FP32 state update operands to BF16 before matmul.
_FP32_STATE_STANDARD = {"torch.float32": AccuracyStandard(atol=1e-3, rtol=1e-5)}
_FP32_CHUNK_STATE_STANDARD = {"torch.float32": AccuracyStandard(atol=1.2e-3, rtol=1e-5)}


def generate_test_data(device, state_dtype=torch.bfloat16):
    """Original ARC-013 dimensions and input generation distributions.

    Source: ops-transformer 6bff82496a8720343fc6144218f059a06a7b9b48,
    attention/chunk_gated_delta_rule/tests/pytest/chunk_gated_delta_rule_main.py
    lines 286-289, 434-453, 471-527.
    The source normalizes q/k during generation. No already supplied input is
    normalized. Torch's seed was unspecified upstream and remains unspecified.
    """
    t, nk, nv, d = 2048, 8, 16, 128
    q = torch.nn.functional.normalize(
        torch.rand((t, nk, d), dtype=torch.bfloat16, device=device), dim=-1
    )
    k = torch.nn.functional.normalize(
        torch.rand((t, nk, d), dtype=torch.bfloat16, device=device), dim=-1
    )
    return dict(
        query=q,
        key=k,
        value=torch.rand((t, nv, d), dtype=torch.bfloat16, device=device),
        beta=torch.rand((t, nv), dtype=torch.bfloat16, device=device),
        initial_state=torch.rand((1, nv, d, d), dtype=state_dtype, device=device),
        actual_seq_lengths=torch.tensor([t], dtype=torch.int32, device=device),
        g=-1 + torch.rand((t, nv), dtype=torch.float32, device=device),
        scale=d**-0.5,
        chunk_state=None,  # Original upstream case does not request this local extension.
    )


def clone_test_data(inputs):
    """Retain non-dense strides as well as values and independent mutable state."""
    result = {}
    for name, value in inputs.items():
        if isinstance(value, torch.Tensor):
            copied = torch.empty_strided(
                value.shape, value.stride(), dtype=value.dtype, device=value.device
            )
            copied.copy_(value)
            result[name] = copied
        else:
            result[name] = value
    return result


@pytest.mark.parametrize(
    "state_dtype", [torch.bfloat16, torch.float32], ids=["state-bf16", "state-fp32"]
)
@pytest.mark.parametrize(
    "scale",
    [128**-0.5, "omitted", None, 1.0, 0.0],
    ids=[
        "upstream-explicit",
        "contract-omitted",
        "contract-none",
        "contract-one",
        "contract-zero",
    ],
)
@pytest.mark.parametrize("chunk_layout", ["absent", "contiguous", "noncontiguous"])
def test_chunk_gated_delta_rule(scale, chunk_layout, state_dtype):
    """Cover scale settings and chunk-state layouts for both state dtypes."""
    from sgl_kernel_npu.operators.attention.gdn import chunk_gated_delta_rule
    from sgl_kernel_npu.operators.utils.generation import Generation, get_hardware

    if state_dtype == torch.float32 and get_hardware().generation != Generation.ARCH35:
        pytest.skip("CGDR FP32 state cases require arch 3510")

    standard = _FP32_STATE_STANDARD if state_dtype == torch.float32 else None
    data = generate_test_data(torch.device("npu"), state_dtype)
    if scale == "omitted":
        del data["scale"]
    else:
        data["scale"] = scale
    if chunk_layout != "absent":
        data["chunk_state"] = torch.zeros(
            (32, 16, 128, 128), dtype=state_dtype, device=data["query"].device
        )
    if chunk_layout == "noncontiguous":
        padded = torch.full(
            (32, 16, 128, 129), -7, dtype=state_dtype, device=data["query"].device
        )
        data["chunk_state"] = padded[..., :128]
    if chunk_layout == "absent":
        initial_before = data["initial_state"].clone()
        executor = check_accuracy(
            chunk_gated_delta_rule, kwargs=data, standard=standard
        )
        if executor.accuracy_checked:
            assert torch.equal(data["initial_state"], initial_before)
    else:
        # Like bitmask, mutable state needs independent DUT/reference inputs.
        impl, impl_ref = get_test_impls()
        skip_accuracy = bool(get_skip_acc_check())
        if skip_accuracy:
            require_test_impl(chunk_gated_delta_rule, impl)
        else:
            require_distinct_impls(chunk_gated_delta_rule, impl, impl_ref)
            actual_inputs = clone_test_data(data)
            reference_inputs = clone_test_data(data)
            initial_before = actual_inputs["initial_state"].clone()
            chunk_before = actual_inputs["chunk_state"].clone()
            result_ref = chunk_gated_delta_rule(**reference_inputs, impl=impl_ref)
            result = chunk_gated_delta_rule(**actual_inputs, impl=impl)
            assert_accuracy(result, result_ref, standard=standard)
            chunk_standard = (
                _FP32_CHUNK_STATE_STANDARD
                if state_dtype == torch.float32 and chunk_layout == "contiguous"
                else None
            )
            assert_accuracy(
                actual_inputs["chunk_state"],
                reference_inputs["chunk_state"],
                standard=chunk_standard,
                path="chunk_state",
            )
            assert torch.equal(actual_inputs["initial_state"], initial_before)
            if chunk_layout == "noncontiguous":
                assert torch.equal(actual_inputs["chunk_state"], chunk_before)

        executor = make_executor(
            chunk_gated_delta_rule,
            kwargs=clone_test_data(data),
            impl=impl,
            impl_ref=impl_ref,
            accuracy_checked=not skip_accuracy,
        )
    check_performance(
        executor,
        enable_perf=scale == 128**-0.5 and chunk_layout == "absent",
    )
