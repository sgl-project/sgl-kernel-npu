"""Real registry integration through a test-only entry and lazy adapters.

The production framework has no common dispatcher. These cases validate the
probe's integration pattern, not production operator adapters or kernel accuracy.
"""

from unittest.mock import Mock, call, sentinel

import pytest


def call_probe(module, **selection):
    return module.probe(sentinel.input, 4, **selection)


# Verify explicit dispatch resolves only the selected backend through the real registry.
@pytest.mark.parametrize("impl", ["backend_a", "backend_b"])
def test_explicit_impl_calls_only_selected_backend(probe_module, kernels, impl):
    result = call_probe(probe_module, impl=impl)

    kernels[impl].assert_called_once_with(sentinel.input, 4, scale=1)
    assert result is kernels[impl].return_value
    for name, kernel in kernels.items():
        if name != impl:
            kernel.assert_not_called()


# Verify omitted impl and explicit auto resolve the probe's declared default alias.
@pytest.mark.parametrize("selection", [{}, {"impl": "auto"}], ids=["default", "auto"])
def test_default_and_auto_select_backend_a(probe_module, kernels, selection):
    result = call_probe(probe_module, **selection)

    kernels["backend_a"].assert_called_once_with(sentinel.input, 4, scale=1)
    kernels["backend_b"].assert_not_called()
    assert result is kernels["backend_a"].return_value


# Verify positional and keyword inputs reach the backend without the impl selector.
@pytest.mark.parametrize("call_style", ["positional", "keyword"])
def test_optional_arguments_are_forwarded_without_impl(
    probe_module, kernels, call_style
):
    values = (sentinel.input, sentinel.offset, sentinel.scale)
    if call_style == "positional":
        result = probe_module.probe(*values, impl="backend_b")
    else:
        result = probe_module.probe(
            x=values[0], offset=values[1], scale=values[2], impl="backend_b"
        )
    kernels["backend_b"].assert_called_once_with(*values[:2], scale=values[2])
    kernels["backend_a"].assert_not_called()
    assert result is kernels["backend_b"].return_value


# Verify unknown impl and explicit None fail in the real registry before backend execution.
@pytest.mark.parametrize("impl", ["missing", None], ids=["unknown", "explicit-none"])
def test_invalid_impl_does_not_call_backends(registry, probe_module, kernels, impl):
    with pytest.raises(registry.ImplementationNotFoundError):
        call_probe(probe_module, impl=impl)
    for kernel in kernels.values():
        kernel.assert_not_called()


# Verify the probe entry accepts impl only by keyword, before calling any backend.
def test_impl_is_keyword_only(probe_module, kernels):
    with pytest.raises(TypeError):
        probe_module.probe(sentinel.input, 4, 1, "backend_a")
    for kernel in kernels.values():
        kernel.assert_not_called()


# Verify a backend error propagates through the probe without another implementation running.
@pytest.mark.parametrize("impl", ["backend_a", "backend_b"])
def test_backend_error_propagates_without_fallback(probe_module, kernels, impl):
    error = RuntimeError("kernel failed")
    kernels[impl].side_effect = error
    with pytest.raises(RuntimeError) as caught:
        call_probe(probe_module, impl=impl)
    assert caught.value is error
    kernels[impl].assert_called_once()
    for name, kernel in kernels.items():
        if name != impl:
            kernel.assert_not_called()


# Verify a new registry implementation is callable through the unchanged probe entry.
def test_new_registered_backend_needs_no_dispatcher_or_directory_change(
    registry, probe_module, kernels
):
    operator = probe_module.probe
    backend = Mock(return_value=sentinel.custom_result)
    registry.register_impl(registry.get_op_name(operator), "test_backend")(backend)

    result = call_probe(probe_module, impl="test_backend")

    backend.assert_called_once_with(sentinel.input, 4, scale=1)
    assert result is sentinel.custom_result
    for kernel in kernels.values():
        kernel.assert_not_called()


# Verify the real registry rejects unsupported probe implementations before execution.
def test_unsupported_impl_is_rejected_by_public_entry(registry, probe_module, kernels):
    backend = Mock()
    registry.register_impl(
        registry.get_op_name(probe_module.probe),
        "unsupported",
        supported_generations=[],
    )(backend)

    with pytest.raises(registry.UnsupportedGenerationError):
        call_probe(probe_module, impl="unsupported")
    backend.assert_not_called()
    for kernel in kernels.values():
        kernel.assert_not_called()


# Verify probe adapter caches preserve routing, fresh inputs and results in both switching orders.
@pytest.mark.parametrize(
    "sequence",
    [
        ("backend_a", "backend_b", "backend_a", "backend_b"),
        ("backend_b", "backend_a", "backend_b", "backend_a"),
    ],
    ids=["backend-a-first", "backend-b-first"],
)
def test_cached_backends_do_not_cross_routes(probe_module, kernels, sequence):
    expected_calls = {name: [] for name in kernels}
    for index, impl in enumerate(sequence):
        values = (
            getattr(sentinel, f"input_{index}"),
            getattr(sentinel, f"offset_{index}"),
            getattr(sentinel, f"scale_{index}"),
        )
        expected_result = object()
        kernels[impl].return_value = expected_result

        assert probe_module.probe(*values, impl=impl) is expected_result

        expected_calls[impl].append(call(*values[:2], scale=values[2]))
        for name, kernel in kernels.items():
            assert kernel.call_args_list == expected_calls[name]
