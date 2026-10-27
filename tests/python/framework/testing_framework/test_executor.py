"""Executor construction and metadata without operator execution."""

import pytest

from tests.python.utils.common import make_executor, use_impls


@pytest.mark.parametrize(
    "options, expected",
    [
        ({}, ("dut", "ref")),
        ({"impl": "other"}, ("other", "ref")),
        ({"impl_ref": "other"}, ("dut", "other")),
        ({"impl": "a", "impl_ref": "b"}, ("a", "b")),
    ],
)
def test_executor_binds_inputs_and_configured_impls(options, expected):
    calls = []

    def operator(value, *, scale, impl):
        calls.append(impl)
        return value[0] * scale

    value = [2]
    kwargs = {"scale": 3}
    with use_impls("dut", "ref"):
        executor = make_executor(operator, args=(value,), kwargs=kwargs, **options)
    assert calls == []
    assert executor.func is operator
    assert (executor.impl, executor.impl_ref) == expected
    assert executor.accuracy_checked is False
    kwargs["scale"] = 9
    value[0] = 4
    assert executor() == 12
    assert calls == [expected[0]]


def test_executor_supports_inplace_operator_and_manual_accuracy_status():
    def operator(value, *, impl):
        value.append(impl)

    value = []
    executor = make_executor(operator, args=(value,), impl="dut", accuracy_checked=True)
    assert value == []
    assert executor.accuracy_checked is True
    assert executor() is None
    assert value == ["dut"]


@pytest.mark.parametrize("status", [None, 0, 1, "true"])
def test_executor_rejects_invalid_accuracy_status(status):
    with pytest.raises(TypeError, match="accuracy_checked must be bool"):
        make_executor(lambda **kwargs: None, accuracy_checked=status)


def test_executor_rejects_impl_in_operator_kwargs():
    with pytest.raises(ValueError, match="Select impl via"):
        make_executor(lambda **kwargs: None, kwargs={"impl": "dut"})
