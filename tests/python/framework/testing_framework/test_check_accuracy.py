"""Accuracy orchestration through real registration and lightweight NPU backends."""

from contextlib import nullcontext

import pytest

from tests.python.utils.common import get_skip_acc_check, get_test_impls, use_impls
from tests.python.utils.standards import AccuracyStandard


def assert_calls(case, names, *, offset=1, scale=1):
    assert [call[0] for call in case.calls] == names
    for _, x, actual_offset, actual_scale in case.calls:
        assert x is case.x
        assert actual_offset == offset
        assert actual_scale == scale


# Execute reference then DUT with the bound inputs and return an executor for DUT-only reuse.
def test_accuracy_success_returns_a_bound_executor(accuracy_case):
    case = accuracy_case
    case.add_impl("dut")
    case.add_impl("ref")
    kwargs = {"scale": 3}
    executor = case.check_accuracy(
        case.operator, args=(case.x, 4), kwargs=kwargs, impl="dut", impl_ref="ref"
    )

    assert_calls(case, ["ref", "dut"], offset=4, scale=3)
    assert executor.func is case.operator
    assert executor.impl == "dut"
    assert executor.impl_ref == "ref"
    assert executor.accuracy_checked is True
    assert kwargs == {"scale": 3}

    result = executor()
    assert result.cpu().tolist() == [7.0, 10.0]
    assert_calls(case, ["ref", "dut", "dut"], offset=4, scale=3)
    assert executor.accuracy_checked is True


# Fail on genuinely different Tensor results without replacing the comparison logic.
def test_accuracy_mismatch_fails(accuracy_case):
    case = accuracy_case
    case.add_impl("dut", delta=1)
    case.add_impl("ref")
    with pytest.raises(AssertionError, match="output:"):
        case.check_accuracy(
            case.operator,
            args=(case.x,),
            impl="dut",
            impl_ref="ref",
            standard={"torch.float32": AccuracyStandard(0, 0)},
        )
    assert_calls(case, ["ref", "dut"])


# Forward a custom standard so the same results fail by default and pass with the override.
def test_custom_standard_reaches_comparison(accuracy_case):
    case = accuracy_case
    case.add_impl("dut", delta=0.25)
    case.add_impl("ref")
    with pytest.raises(AssertionError, match="output:"):
        case.check_accuracy(case.operator, args=(case.x,), impl="dut", impl_ref="ref")
    executor = case.check_accuracy(
        case.operator,
        args=(case.x,),
        impl="dut",
        impl_ref="ref",
        standard={"torch.float32": AccuracyStandard(atol=0.25, rtol=0)},
    )
    assert_calls(case, ["ref", "dut", "ref", "dut"])
    assert executor.accuracy_checked is True


# Resolve each omitted or None implementation from context and override explicit sides independently.
@pytest.mark.parametrize(
    "options, expected_dut, expected_ref",
    [
        pytest.param({}, "configured_dut", "configured_ref", id="both-omitted"),
        pytest.param(
            {"impl": None, "impl_ref": None},
            "configured_dut",
            "configured_ref",
            id="both-none",
        ),
        pytest.param(
            {"impl": "explicit_dut"},
            "explicit_dut",
            "configured_ref",
            id="explicit-dut",
        ),
        pytest.param(
            {"impl_ref": "explicit_ref"},
            "configured_dut",
            "explicit_ref",
            id="explicit-reference",
        ),
        pytest.param(
            {"impl": "explicit_dut", "impl_ref": "explicit_ref"},
            "explicit_dut",
            "explicit_ref",
            id="both-explicit",
        ),
        pytest.param(
            {"impl": None, "impl_ref": "explicit_ref"},
            "configured_dut",
            "explicit_ref",
            id="none-dut",
        ),
        pytest.param(
            {"impl": "explicit_dut", "impl_ref": None},
            "explicit_dut",
            "configured_ref",
            id="none-reference",
        ),
    ],
)
def test_impl_arguments_override_each_configured_side(
    accuracy_case, options, expected_dut, expected_ref
):
    case = accuracy_case
    for name in ("configured_dut", "configured_ref", "explicit_dut", "explicit_ref"):
        case.add_impl(name)
    with use_impls("configured_dut", "configured_ref"):
        executor = case.check_accuracy(case.operator, args=(case.x,), **options)
    assert_calls(case, [expected_ref, expected_dut])
    assert executor.impl == expected_dut
    assert executor.impl_ref == expected_ref
    assert executor.accuracy_checked is True


# Reject impl in operator kwargs before either implementation can execute.
@pytest.mark.parametrize("nested_impl", [None, "dut", "other"])
def test_impl_in_operator_kwargs_is_rejected(accuracy_case, nested_impl):
    case = accuracy_case
    case.add_impl("dut")
    case.add_impl("ref")
    with pytest.raises(ValueError, match="Select impl via check_accuracy"):
        case.check_accuracy(
            case.operator,
            args=(case.x,),
            kwargs={"impl": nested_impl},
            impl="dut",
            impl_ref="ref",
        )
    assert_calls(case, [])


# Skip all accuracy computation without requiring a reference and keep later DUT calls unchecked.
def test_skip_accuracy_returns_an_unchecked_executor(accuracy_case):
    case = accuracy_case
    case.add_impl("dut")
    executor = case.check_accuracy(
        case.operator,
        args=(case.x,),
        impl="dut",
        impl_ref="missing_reference",
        skip_acc_check=True,
    )
    assert_calls(case, [])
    assert executor.impl == "dut"
    assert executor.impl_ref == "missing_reference"
    assert executor.accuracy_checked is False
    assert executor().cpu().tolist() == [2.0, 3.0]
    assert_calls(case, ["dut"])
    assert executor.accuracy_checked is False


# Give an explicit skip context priority over code and otherwise retain the code setting.
@pytest.mark.parametrize(
    "context_skip, code_skip, expected_skip",
    [
        pytest.param(True, False, True, id="context-skips"),
        pytest.param(False, True, False, id="context-requires-check"),
        pytest.param(None, True, True, id="code-skips"),
        pytest.param(None, False, False, id="code-requires-check"),
        pytest.param(
            False, "invalid-but-overridden", False, id="validate-effective-value"
        ),
    ],
)
def test_accuracy_skip_context_overrides_code_setting(
    accuracy_case, context_skip, code_skip, expected_skip
):
    case = accuracy_case
    case.add_impl("dut")
    case.add_impl("ref")
    with use_impls("dut", "ref", skip_acc_check=context_skip):
        executor = case.check_accuracy(
            case.operator, args=(case.x,), skip_acc_check=code_skip
        )
    assert_calls(case, [] if expected_skip else ["ref", "dut"])
    assert executor.accuracy_checked is (not expected_skip)


# Reject a non-boolean effective skip value before any backend execution.
@pytest.mark.parametrize("flag", [1, 0, "true", None])
def test_invalid_effective_skip_flag_is_rejected(accuracy_case, flag):
    case = accuracy_case
    case.add_impl("dut")
    case.add_impl("ref")
    with pytest.raises(TypeError, match="skip_acc_check must be bool"):
        case.check_accuracy(
            case.operator,
            args=(case.x,),
            impl="dut",
            impl_ref="ref",
            skip_acc_check=flag,
        )
    assert_calls(case, [])


# Reject the same registered callable selected by the same name or by an alias.
@pytest.mark.parametrize("reference", ["dut", "alias"], ids=["same-name", "alias"])
def test_same_entry_cannot_be_used_as_reference(accuracy_case, reference):
    case = accuracy_case
    backend = case.add_impl("dut")
    case.registry.register_impl(case.op_name, "alias")(backend)
    with pytest.raises(ValueError, match="same registered entry"):
        case.check_accuracy(
            case.operator, args=(case.x,), impl="dut", impl_ref=reference
        )
    assert_calls(case, [])


# Distinguish missing entries, unsupported DUT skips and reference errors before computation.
@pytest.mark.parametrize(
    "skip_accuracy, dut_state, ref_state, expected",
    [
        pytest.param(False, "missing", "valid", "missing-dut", id="unknown-dut"),
        pytest.param(False, "valid", "missing", "missing-ref", id="unknown-reference"),
        pytest.param(False, "unsupported", "valid", "skip", id="unsupported-dut"),
        pytest.param(
            False, "valid", "unsupported", "unsupported-ref", id="unsupported-reference"
        ),
        pytest.param(
            False,
            "unsupported",
            "missing",
            "missing-ref",
            id="reference-error-before-skip",
        ),
        pytest.param(
            False,
            "unsupported",
            "unsupported",
            "unsupported-ref",
            id="both-unsupported",
        ),
        pytest.param(
            True, "missing", "missing", "missing-dut", id="skip-accuracy-unknown-dut"
        ),
        pytest.param(
            True, "unsupported", "missing", "skip", id="skip-accuracy-unsupported-dut"
        ),
    ],
)
def test_implementation_resolution_controls_fail_and_skip(
    accuracy_case, skip_accuracy, dut_state, ref_state, expected
):
    case = accuracy_case
    for name, state in (("dut", dut_state), ("ref", ref_state)):
        if state != "missing":
            case.add_impl(
                name, supported_generations=[] if state == "unsupported" else None
            )

    if expected == "skip":
        exception = pytest.skip.Exception
    elif expected.startswith("missing"):
        exception = case.registry.ImplementationNotFoundError
    else:
        exception = case.registry.UnsupportedGenerationError
    # Catch expected skips so this UT passes only after checking their cause.
    with pytest.raises(exception) as caught:
        case.check_accuracy(
            case.operator,
            args=(case.x,),
            impl="dut",
            impl_ref="ref",
            skip_acc_check=skip_accuracy,
        )
    if expected == "skip":
        assert "[impl=dut]" in str(caught.value)
    elif expected.startswith("missing"):
        missing = "dut" if expected == "missing-dut" else "ref"
        assert f"Unknown impl {missing!r}" in str(caught.value)
    else:
        assert caught.value.impl == "ref"
    assert_calls(case, [])


# Propagate the original backend exception and stop without retrying the failed flow.
@pytest.mark.parametrize("failing_impl", ["ref", "dut"])
def test_backend_failure_stops_accuracy_flow(accuracy_case, failing_impl):
    case = accuracy_case
    error = RuntimeError(f"{failing_impl} failed")
    for name in ("dut", "ref"):
        case.add_impl(name, error=error if name == failing_impl else None)
    with pytest.raises(RuntimeError) as caught:
        case.check_accuracy(case.operator, args=(case.x,), impl="dut", impl_ref="ref")
    assert caught.value is error
    assert_calls(case, ["ref"] if failing_impl == "ref" else ["ref", "dut"])


# Restore nested implementation and skip contexts on both normal and exceptional exits.
@pytest.mark.parametrize("raise_inside", [False, True], ids=["normal", "exception"])
def test_accuracy_configuration_is_restored(raise_inside):
    before = (get_test_impls(), get_skip_acc_check())
    error = RuntimeError("leave inner context")
    with use_impls("outer_dut", "outer_ref", skip_acc_check=True):
        outer = (("outer_dut", "outer_ref"), True)
        assert (get_test_impls(), get_skip_acc_check()) == outer
        with pytest.raises(RuntimeError) if raise_inside else nullcontext() as caught:
            with use_impls("inner_dut", "inner_ref", skip_acc_check=False):
                assert get_test_impls() == ("inner_dut", "inner_ref")
                assert get_skip_acc_check() is False
                if raise_inside:
                    raise error
        if raise_inside:
            assert caught.value is error
        assert (get_test_impls(), get_skip_acc_check()) == outer
    assert (get_test_impls(), get_skip_acc_check()) == before
