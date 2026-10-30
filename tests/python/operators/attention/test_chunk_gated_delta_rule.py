"""CGDR accuracy and state contracts through the public attention interface."""

import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import assert_accuracy
from tests.python.utils.common import get_test_impls, require_distinct_impls


def generate_test_data(device):
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
        initial_state=torch.rand((1, nv, d, d), dtype=torch.bfloat16, device=device),
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
def test_chunk_gated_delta_rule(scale, chunk_layout):
    """Original explicit scale and separately labelled default/zero variants."""
    from sgl_kernel_npu.operators.attention import chunk_gated_delta_rule

    impl, impl_ref = get_test_impls()
    require_distinct_impls(chunk_gated_delta_rule, impl, impl_ref)
    data = generate_test_data(torch.device("npu"))
    if scale == "omitted":
        del data["scale"]
    else:
        data["scale"] = scale
    if chunk_layout != "absent":
        data["chunk_state"] = torch.zeros(
            (32, 16, 128, 128), dtype=torch.bfloat16, device=data["query"].device
        )
    if chunk_layout == "noncontiguous":
        padded = torch.full(
            (32, 16, 128, 129), -7, dtype=torch.bfloat16, device=data["query"].device
        )
        data["chunk_state"] = padded[..., :128]
    actual_inputs, reference_inputs = clone_test_data(data), clone_test_data(data)
    initial_before = actual_inputs["initial_state"].clone()
    chunk_before = (
        None if data["chunk_state"] is None else actual_inputs["chunk_state"].clone()
    )
    result_ref = chunk_gated_delta_rule(**reference_inputs, impl=impl_ref)
    result = chunk_gated_delta_rule(**actual_inputs, impl=impl)
    assert_accuracy(result, result_ref)
    if chunk_layout != "absent":
        assert_accuracy(actual_inputs["chunk_state"], reference_inputs["chunk_state"])
    assert torch.equal(actual_inputs["initial_state"], initial_before)
    if chunk_layout == "noncontiguous":
        assert torch.equal(actual_inputs["chunk_state"], chunk_before)
