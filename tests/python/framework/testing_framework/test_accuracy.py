"""Real Tensor and nested-output comparison on the selected NPU."""

import pytest

from tests.python.utils import standards
from tests.python.utils.standards import AccuracyStandard


def tensor(runtime, values, dtype="float32", *, device=None):
    torch = runtime.torch
    if device is None:
        device = f"npu:{torch.npu.current_device()}"
    return torch.tensor(values, dtype=getattr(torch, dtype), device=device)


# Exercise absolute and relative tolerances inside, on and outside the boundary.
@pytest.mark.parametrize(
    "dtype, reference, actual, atol, rtol, passes",
    [
        pytest.param("float32", 0, 0, 0, 0, True, id="equal"),
        pytest.param("float32", 0, 0.0625, 0.125, 0, True, id="atol-inside"),
        pytest.param("float32", 0, 0.125, 0.125, 0, True, id="atol-boundary"),
        pytest.param("float32", 0, 0.25, 0.125, 0, False, id="atol-outside"),
        pytest.param("float32", 8, 8.5, 0, 0.125, True, id="rtol-inside"),
        pytest.param("float32", 8, 9, 0, 0.125, True, id="rtol-boundary"),
        pytest.param("float32", 8, 10, 0, 0.125, False, id="rtol-outside"),
        pytest.param("float16", 1, 1.0625, 0.125, 0, True, id="fp16-inside"),
        pytest.param("float16", 1, 1.25, 0.125, 0, False, id="fp16-outside"),
        pytest.param("bfloat16", 1, 1.0625, 0.125, 0, True, id="bf16-inside"),
        pytest.param("bfloat16", 1, 1.25, 0.125, 0, False, id="bf16-outside"),
    ],
)
def test_tensor_tolerance_controls_pass_and_failure(
    npu_runtime, accuracy_tools, dtype, reference, actual, atol, rtol, passes
):
    result = tensor(npu_runtime, [actual], dtype)
    result_ref = tensor(npu_runtime, [reference], dtype)
    override = {f"torch.{dtype}": AccuracyStandard(atol, rtol)}
    if passes:
        accuracy_tools.assert_accuracy(result, result_ref, override)
    else:
        with pytest.raises(AssertionError, match="output:"):
            accuracy_tools.assert_accuracy(result, result_ref, override)


# Compare integer and boolean Tensor outputs exactly under the default policy.
@pytest.mark.parametrize("dtype", ["int8", "int32", "int64", "bool"])
def test_integer_and_bool_tensors_compare_exactly(npu_runtime, accuracy_tools, dtype):
    reference = tensor(npu_runtime, [0, 1], dtype)
    accuracy_tools.assert_accuracy(tensor(npu_runtime, [0, 1], dtype), reference)
    with pytest.raises(AssertionError, match="output:"):
        accuracy_tools.assert_accuracy(tensor(npu_runtime, [1, 1], dtype), reference)


# Reject metadata and Tensor/non-Tensor mismatches before comparing values.
@pytest.mark.parametrize("mismatch", ["shape", "dtype", "device", "tensor-type"])
def test_tensor_metadata_mismatch_is_rejected(npu_runtime, accuracy_tools, mismatch):
    reference = tensor(npu_runtime, [1, 2])
    if mismatch == "shape":
        result = tensor(npu_runtime, [[1, 2]])
        message = "shape mismatch"
    elif mismatch == "dtype":
        result = tensor(npu_runtime, [1, 2], "float16")
        message = "dtype mismatch"
    elif mismatch == "device":
        result = tensor(npu_runtime, [1, 2], device="cpu")
        message = "device mismatch"
    else:
        result = [1, 2]
        message = "Tensor/non-Tensor mismatch"
    with pytest.raises(AssertionError, match=message):
        accuracy_tools.assert_accuracy(result, reference)


# Compare nested outputs regardless of dict order and identify the failing leaf.
def test_nested_outputs_are_compared_with_error_paths(npu_runtime, accuracy_tools):
    reference = {"main": tensor(npu_runtime, [1]), "aux": [None, (2, 3.0)]}
    result = {"aux": [None, (2, 3.0)], "main": tensor(npu_runtime, [1])}
    accuracy_tools.assert_accuracy(result, reference)

    result["aux"][1] = (2, 4.0)
    with pytest.raises(AssertionError) as caught:
        accuracy_tools.assert_accuracy(result, reference)
    assert "output['aux'][1][1]:" in str(caught.value)


# Reject different container types, sequence lengths and dictionary key sets.
@pytest.mark.parametrize(
    "result, reference, message",
    [
        pytest.param([1], (1,), "type mismatch", id="list-vs-tuple"),
        pytest.param([1], [1, 2], "length mismatch", id="list-length"),
        pytest.param((1,), (1, 2), "length mismatch", id="tuple-length"),
        pytest.param({}, {"a": 1}, "dictionary keys mismatch", id="missing-key"),
        pytest.param(
            {"a": 1, "b": 2}, {"a": 1}, "dictionary keys mismatch", id="extra-key"
        ),
        pytest.param(
            {"b": 1}, {"a": 1}, "dictionary keys mismatch", id="different-key"
        ),
    ],
)
def test_container_structure_mismatch_is_rejected(
    accuracy_tools, result, reference, message
):
    with pytest.raises(AssertionError, match=message):
        accuracy_tools.assert_accuracy(result, reference)


# Apply float32 tolerances to float/complex scalars and exact equality to other scalars.
@pytest.mark.parametrize(
    "result, reference, passes",
    [
        pytest.param(1.0625, 1.0, True, id="float-inside"),
        pytest.param(1.25, 1.0, False, id="float-outside"),
        pytest.param(1 + 0.0625j, 1 + 0j, True, id="complex-inside"),
        pytest.param(1 + 0.25j, 1 + 0j, False, id="complex-outside"),
        pytest.param(7, 7, True, id="int-equal"),
        pytest.param(8, 7, False, id="int-different"),
        pytest.param(True, True, True, id="bool-equal"),
        pytest.param(False, True, False, id="bool-different"),
        pytest.param("a", "a", True, id="str-equal"),
        pytest.param("b", "a", False, id="str-different"),
        pytest.param(b"a", b"a", True, id="bytes-equal"),
        pytest.param(b"b", b"a", False, id="bytes-different"),
        pytest.param(None, None, True, id="none-equal"),
        pytest.param(0, None, False, id="none-type-mismatch"),
        pytest.param(True, 1, False, id="bool-is-not-int"),
        pytest.param(1, 1.0, False, id="int-is-not-float"),
    ],
)
def test_python_scalars_follow_type_and_tolerance_policy(
    accuracy_tools, result, reference, passes
):
    override = {"torch.float32": AccuracyStandard(atol=0.125, rtol=0)}
    if passes:
        accuracy_tools.assert_accuracy(result, reference, override)
    else:
        with pytest.raises(AssertionError, match="output:"):
            accuracy_tools.assert_accuracy(result, reference, override)


# Reject NaN even when both outputs contain NaN at the same position.
@pytest.mark.parametrize("as_tensor", [True, False], ids=["tensor", "scalar"])
@pytest.mark.parametrize("reference_is_nan", [True, False], ids=["both-nan", "one-nan"])
def test_nan_values_are_rejected(
    npu_runtime, accuracy_tools, as_tensor, reference_is_nan
):
    result = float("nan")
    reference = float("nan") if reference_is_nan else 0.0
    if as_tensor:
        result = tensor(npu_runtime, [result])
        reference = tensor(npu_runtime, [reference])
    with pytest.raises(AssertionError, match="output:"):
        accuracy_tools.assert_accuracy(result, reference)


# Require a floating Tensor standard and honor a local override when the default is absent.
def test_floating_tensor_requires_an_accuracy_standard(
    npu_runtime, accuracy_tools, monkeypatch
):
    result = tensor(npu_runtime, [1])
    reference = tensor(npu_runtime, [1])
    monkeypatch.delitem(standards.DEFAULT_STANDARDS, "torch.float32")
    with pytest.raises(TypeError, match="no accuracy standard for torch.float32"):
        accuracy_tools.assert_accuracy(result, reference)
    accuracy_tools.assert_accuracy(
        result, reference, {"torch.float32": AccuracyStandard(0, 0)}
    )


# Reject unsupported output objects even if Python considers them equal.
@pytest.mark.parametrize("result", [{1}, object()], ids=["set", "object"])
def test_unsupported_output_type_is_rejected(accuracy_tools, result):
    with pytest.raises(TypeError, match="unsupported output type"):
        accuracy_tools.assert_accuracy(result, result)
