"""Import-time registration, optional dependencies and per-test isolation."""

import importlib.abc
import sys
from pathlib import Path
from unittest.mock import Mock, sentinel

import pytest


# Verify checkout public exports and operator registration share the checkout registry.
def test_checked_out_public_api_uses_the_same_registry(operator_modules, npu_runtime):
    registry = operator_modules.registry
    rmsnorm_bias = operator_modules.import_module("operators.norm").rmsnorm_bias

    assert (
        Path(registry.__file__).resolve().is_relative_to(npu_runtime.source.resolve())
    )
    assert operator_modules.public_api.get_impl is registry.get_impl
    assert operator_modules.public_api.list_impls is registry.list_impls
    assert registry.get_op_name(rmsnorm_bias) == "norm.rmsnorm_bias"
    assert registry.get_impl(rmsnorm_bias, "auto") is registry.get_impl(
        rmsnorm_bias, "triton"
    )


# Verify public imports register once without importing backend kernel modules.
def test_public_import_registers_without_importing_kernel_leaves(operator_modules):
    norm = operator_modules.import_module("operators.norm")
    module = operator_modules.import_module("operators.norm.rmsnorm_bias")
    registry = operator_modules.registry
    assert norm.rmsnorm_bias is module.rmsnorm_bias
    assert registry.list_impls(norm.rmsnorm_bias) == ["auto", "torch_native", "triton"]
    assert not any(
        name.startswith(f"{operator_modules.namespace}.implementation")
        for name in sys.modules
    )
    again = operator_modules.import_module("operators.norm.rmsnorm_bias")
    assert again is module
    assert registry.list_impls(norm.rmsnorm_bias) == ["auto", "torch_native", "triton"]


# Verify hardware detection failure aborts registry import without caching a partial module.
def test_hardware_detection_failure_aborts_registry_import(
    package_factory, monkeypatch
):
    package = package_factory()
    error = package.generation.HardwareDetectionError("hardware unavailable")
    detector = Mock(side_effect=error)
    monkeypatch.setattr(package.generation, "get_hardware", detector)
    with pytest.raises(package.generation.HardwareDetectionError) as caught:
        package.load()
    assert caught.value is error
    detector.assert_called_once_with()
    assert f"{package.namespace}.operators.utils.registry" not in sys.modules


# Verify private registrations do not affect other registries or the installed package.
def test_private_registrations_do_not_leak_to_other_modules(
    package_factory, npu_runtime
):
    installed_path = list(npu_runtime.package.__path__)
    installed_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "sgl_kernel_npu" or name.startswith("sgl_kernel_npu.")
    }
    first, second = package_factory().load(), package_factory().load()
    backend = Mock()
    name = f"test.{first.namespace}"
    first.registry.register_impl(name, "test_backend")(backend)
    assert first.registry.get_impl_by_name(name, "test_backend") is backend
    assert second.registry.list_impls_by_name(name) == []
    assert list(npu_runtime.package.__path__) == installed_path
    assert {
        name: module
        for name, module in sys.modules.items()
        if name == "sgl_kernel_npu" or name.startswith("sgl_kernel_npu.")
    } == installed_modules


# Verify checkout operators work when the installed runtime has no operators package.
def test_checkout_framework_does_not_require_installed_operators(
    package_factory, monkeypatch
):
    # Reproduce an installed runtime predating the operators refactor. Any
    # attempt to import its public framework must fail; checkout imports work.
    monkeypatch.setitem(sys.modules, "sgl_kernel_npu.operators", None)
    modules = package_factory().load()
    operator = modules.import_module("operators.norm").rmsnorm_bias

    assert modules.public_api.list_impls(operator) == ["auto", "torch_native", "triton"]
    assert modules.public_api.get_impl(operator, "auto") is modules.public_api.get_impl(
        operator, "triton"
    )


# Verify missing dependencies affect only the selected backend and permit a later retry.
@pytest.mark.parametrize("missing_impl", ["triton", "torch_native"])
def test_missing_backend_is_isolated_and_failed_import_can_retry(
    operator_modules, rmsnorm_module, kernels, monkeypatch, missing_impl
):
    name = (
        f"{operator_modules.namespace}.implementation.{missing_impl}"
        ".norm.rmsnorm_bias"
    )
    leaf = sys.modules[name]
    other = "torch_native" if missing_impl == "triton" else "triton"
    error = ModuleNotFoundError("optional backend unavailable", name=name)

    class MissingBackend(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname == name:
                raise error
            return None

    values = (sentinel.input, sentinel.weight, sentinel.bias, 1e-6)
    with monkeypatch.context() as patch:
        patch.delitem(sys.modules, name)
        patch.setattr(sys, "meta_path", [MissingBackend(), *sys.meta_path])
        # An unselected missing dependency must not break the selected route.
        assert (
            rmsnorm_module.rmsnorm_bias(*values, impl=other)
            is kernels[other].return_value
        )
        kernels[other].assert_called_once_with(*values, None, None)
        with pytest.raises(ModuleNotFoundError) as caught:
            rmsnorm_module.rmsnorm_bias(*values, impl=missing_impl)
        assert caught.value is error
        kernels[missing_impl].assert_not_called()
        assert kernels[other].call_count == 1  # no fallback on failure

    assert sys.modules[name] is leaf
    assert (
        rmsnorm_module.rmsnorm_bias(*values, impl=missing_impl)
        is kernels[missing_impl].return_value
    )
    kernels[missing_impl].assert_called_once_with(*values, None, None)
    assert kernels[other].call_count == 1
