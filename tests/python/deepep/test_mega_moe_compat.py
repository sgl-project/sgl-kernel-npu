"""CPU coverage of the operator adapter without importing the native extension."""

import importlib.util
import inspect
from pathlib import Path

import pytest

_PATH = (
    Path(__file__).resolve().parents[3]
    / "python/deep_ep/deep_ep/strategies/_mega_moe_compat.py"
)
_SPEC = importlib.util.spec_from_file_location("_mega_moe_compat", _PATH)
_COMPAT = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_COMPAT)
adapt = _COMPAT.adapt_mega_moe_operator


def test_legacy_plain_swiglu_drops_only_the_unsupported_keyword():
    received = {}

    def operator(x, *, activation="swiglu", l1_bias=None, l2_bias=None):
        received.update(x=x, activation=activation, l1_bias=l1_bias, l2_bias=l2_bias)
        return x

    first, second = object(), object()
    adapted = adapt(operator)
    assert adapted(7, activation_params=None, l1_bias=first, l2_bias=second) == 7
    assert received == dict(x=7, activation="swiglu", l1_bias=first, l2_bias=second)
    assert "activation_params" in inspect.signature(adapted).parameters
    assert adapted.__wrapped__ is operator
    assert adapt(adapted) is adapted


def test_legacy_positional_activation_is_checked():
    def operator(x, activation="swiglu"):
        return x

    adapted = adapt(operator)
    assert adapted(7, "swiglu", activation_params=None) == 7
    with pytest.raises(ValueError, match="only swiglu"):
        adapted(7, "situglu", activation_params=None)


@pytest.mark.parametrize("activation", ["situ", "situglu"])
def test_legacy_rejects_other_activations_before_calling_operator(activation):
    def operator(x, *, activation="swiglu"):
        pytest.fail("Incompatible activation reached the legacy operator")

    with pytest.raises(ValueError, match="only swiglu"):
        adapt(operator)(7, activation=activation)


@pytest.mark.parametrize("parameters", [{}, {"beta": 4.0}])
def test_legacy_rejects_nondefault_activation_parameters(parameters):
    def operator(x, *, activation="swiglu"):
        pytest.fail("Activation parameters were silently discarded")

    with pytest.raises(ValueError, match="activation_params=None"):
        adapt(operator)(7, activation_params=parameters)


def test_modern_operator_is_unchanged():
    def operator(x, *, activation="swiglu", activation_params=None):
        return x, activation, activation_params

    assert adapt(operator) is operator
    assert adapt(operator)(
        7, activation="situglu", activation_params={"beta": 4.0}
    ) == (7, "situglu", {"beta": 4.0})


def test_operator_accepting_extra_keywords_is_unchanged():
    def operator(x, **kwargs):
        return x, kwargs

    assert adapt(operator) is operator
    assert adapt(operator)(7, activation_params={"beta": 4.0}) == (
        7,
        {"activation_params": {"beta": 4.0}},
    )


def test_operator_without_activation_selection_is_rejected():
    def operator(x):
        return x

    with pytest.raises(RuntimeError, match="lacks activation selection"):
        adapt(operator)
