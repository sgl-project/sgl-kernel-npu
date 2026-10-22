"""Public dispatch contracts, with real adapters and inert kernel leaves."""

from unittest.mock import Mock, call, sentinel

import pytest


def call_rmsnorm(module, **selection):
    return module.rmsnorm_bias(
        sentinel.input, sentinel.weight, sentinel.bias, 1e-6, **selection
    )


# Verify explicit dispatch calls only the selected backend and forwards its result.
@pytest.mark.parametrize("impl", ["triton", "torch_native"])
def test_explicit_impl_calls_only_selected_backend(rmsnorm_module, kernels, impl):
    result = call_rmsnorm(rmsnorm_module, impl=impl)

    kernels[impl].assert_called_once_with(
        sentinel.input, sentinel.weight, sentinel.bias, 1e-6, None, None
    )
    assert result is kernels[impl].return_value
    for name, kernel in kernels.items():
        if name != impl:
            kernel.assert_not_called()


# Verify omitted impl and explicit auto both select the current Triton default.
@pytest.mark.parametrize("selection", [{}, {"impl": "auto"}], ids=["default", "auto"])
def test_default_and_auto_select_triton(rmsnorm_module, kernels, selection):
    result = call_rmsnorm(rmsnorm_module, **selection)

    kernels["triton"].assert_called_once_with(
        sentinel.input, sentinel.weight, sentinel.bias, 1e-6, None, None
    )
    kernels["torch_native"].assert_not_called()
    assert result is kernels["triton"].return_value


# Verify positional and keyword arguments reach the backend without the impl selector.
@pytest.mark.parametrize("call_style", ["positional", "keyword"])
def test_optional_arguments_are_forwarded_without_impl(
    rmsnorm_module, kernels, call_style
):
    values = (
        sentinel.input,
        sentinel.weight,
        sentinel.bias,
        1e-5,
        sentinel.scale,
        sentinel.offset,
    )
    if call_style == "positional":
        result = rmsnorm_module.rmsnorm_bias(*values, impl="torch_native")
    else:
        result = rmsnorm_module.rmsnorm_bias(
            input=values[0],
            norm_weight=values[1],
            norm_bias=values[2],
            eps=values[3],
            quant_scale=values[4],
            quant_offset=values[5],
            impl="torch_native",
        )
    kernels["torch_native"].assert_called_once_with(*values)
    kernels["triton"].assert_not_called()
    assert result is kernels["torch_native"].return_value


# Verify unknown impl and explicit None fail before any backend executes.
@pytest.mark.parametrize("impl", ["missing", None], ids=["unknown", "explicit-none"])
def test_invalid_impl_does_not_call_backends(registry, rmsnorm_module, kernels, impl):
    with pytest.raises(registry.ImplementationNotFoundError):
        call_rmsnorm(rmsnorm_module, impl=impl)
    for kernel in kernels.values():
        kernel.assert_not_called()


# Verify passing impl positionally raises an error before backend execution.
def test_impl_is_keyword_only(rmsnorm_module, kernels):
    with pytest.raises(TypeError):
        rmsnorm_module.rmsnorm_bias(
            sentinel.input, sentinel.weight, sentinel.bias, 1e-6, None, None, "triton"
        )
    for kernel in kernels.values():
        kernel.assert_not_called()


# Verify a selected backend error propagates without invoking another implementation.
@pytest.mark.parametrize("impl", ["triton", "torch_native"])
def test_backend_error_propagates_without_fallback(rmsnorm_module, kernels, impl):
    error = RuntimeError("kernel failed")
    kernels[impl].side_effect = error
    with pytest.raises(RuntimeError) as caught:
        call_rmsnorm(rmsnorm_module, impl=impl)
    assert caught.value is error
    kernels[impl].assert_called_once()
    for name, kernel in kernels.items():
        if name != impl:
            kernel.assert_not_called()


# Verify a newly registered backend works through the unchanged public dispatcher.
def test_new_registered_backend_needs_no_dispatcher_or_directory_change(
    registry, rmsnorm_module, kernels
):
    operator = rmsnorm_module.rmsnorm_bias
    backend = Mock(return_value=sentinel.custom_result)
    # A registration-mechanism test, not an actual AscendC integration test.
    registry.register_impl(registry.get_op_name(operator), "test_backend")(backend)

    result = call_rmsnorm(rmsnorm_module, impl="test_backend")

    backend.assert_called_once_with(
        sentinel.input, sentinel.weight, sentinel.bias, 1e-6, None, None
    )
    assert result is sentinel.custom_result
    for kernel in kernels.values():
        kernel.assert_not_called()


# Verify the public entry rejects an unsupported generation before invoking any backend.
def test_unsupported_impl_is_rejected_by_public_entry(
    registry, rmsnorm_module, kernels
):
    backend = Mock()
    registry.register_impl(
        registry.get_op_name(rmsnorm_module.rmsnorm_bias),
        "unsupported",
        supported_generations=[],
    )(backend)

    with pytest.raises(registry.UnsupportedGenerationError):
        call_rmsnorm(rmsnorm_module, impl="unsupported")
    backend.assert_not_called()
    for kernel in kernels.values():
        kernel.assert_not_called()


# Verify both switching orders preserve per-call routing, fresh arguments, and results.
@pytest.mark.parametrize(
    "sequence",
    [
        ("triton", "torch_native", "triton", "torch_native"),
        ("torch_native", "triton", "torch_native", "triton"),
    ],
    ids=["triton-first", "torch-native-first"],
)
def test_cached_backends_do_not_cross_routes(rmsnorm_module, kernels, sequence):
    expected_calls = {name: [] for name in kernels}
    for index, impl in enumerate(sequence):
        values = (
            getattr(sentinel, f"input_{index}"),
            getattr(sentinel, f"weight_{index}"),
            getattr(sentinel, f"bias_{index}"),
            1e-6 * (index + 1),
            getattr(sentinel, f"scale_{index}"),
            getattr(sentinel, f"offset_{index}"),
        )
        expected_result = object()
        kernels[impl].return_value = expected_result

        assert rmsnorm_module.rmsnorm_bias(*values, impl=impl) is expected_result

        expected_calls[impl].append(call(*values))
        for name, kernel in kernels.items():
            assert kernel.call_args_list == expected_calls[name]
