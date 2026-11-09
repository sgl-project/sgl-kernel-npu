"""Import-time registration, optional dependencies and per-test isolation."""

import importlib.abc
import sys
from pathlib import Path
from unittest.mock import Mock, sentinel

import pytest


# Verify checkout public exports and operator registration share the checkout registry.
def test_checked_out_public_api_uses_the_same_registry(
    operator_modules, npu_runtime, probe_loader
):
    registry = operator_modules.registry
    module = probe_loader(operator_modules)
    operator = module.probe

    assert (
        Path(registry.__file__).resolve().is_relative_to(npu_runtime.source.resolve())
    )
    assert operator_modules.public_api.get_impl is registry.get_impl
    assert operator_modules.public_api.list_impls is registry.list_impls
    assert registry.get_op_name(operator) == module.OP_NAME
    assert registry.get_impl(operator, "auto") is registry.get_impl(
        operator, "backend_a"
    )


# Verify importing the test-only probe registers once without importing backend leaves.
def test_probe_import_registers_without_importing_backend_leaves(
    operator_modules, probe_loader
):
    module = probe_loader(operator_modules)
    registry = operator_modules.registry
    assert registry.get_op_name(module.probe) == module.OP_NAME
    assert registry.list_impls(module.probe) == ["auto", "backend_a", "backend_b"]
    assert not any(
        name.startswith(f"{operator_modules.namespace}._probe_backends")
        or name.startswith(f"{operator_modules.namespace}.implementation")
        or name.startswith(f"{operator_modules.namespace}.operators.norm")
        for name in sys.modules
    )
    again = probe_loader(operator_modules)
    assert again is module
    assert registry.list_impls(module.probe) == ["auto", "backend_a", "backend_b"]


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
    package_factory, monkeypatch, probe_loader
):
    # Reproduce an installed runtime predating the operators refactor. Any
    # attempt to import its public framework must fail; checkout imports work.
    monkeypatch.setitem(sys.modules, "sgl_kernel_npu.operators", None)
    modules = package_factory().load()
    operator = probe_loader(modules).probe

    assert modules.public_api.list_impls(operator) == ["auto", "backend_a", "backend_b"]
    assert modules.public_api.get_impl(operator, "auto") is modules.public_api.get_impl(
        operator, "backend_a"
    )


# Verify missing dependencies affect only the selected backend and permit a later retry.
@pytest.mark.parametrize("missing_impl", ["backend_a", "backend_b"])
def test_missing_backend_is_isolated_and_failed_import_can_retry(
    operator_modules, probe_module, kernels, monkeypatch, missing_impl
):
    name = f"{operator_modules.namespace}._probe_backends.{missing_impl}"
    leaf = sys.modules[name]
    other = "backend_b" if missing_impl == "backend_a" else "backend_a"
    error = ModuleNotFoundError("optional backend unavailable", name=name)

    class MissingBackend(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname == name:
                raise error
            return None

    values = (sentinel.input, 4)
    with monkeypatch.context() as patch:
        patch.delitem(sys.modules, name)
        patch.setattr(sys, "meta_path", [MissingBackend(), *sys.meta_path])
        # An unselected missing dependency must not break the selected route.
        assert probe_module.probe(*values, impl=other) is kernels[other].return_value
        kernels[other].assert_called_once_with(*values, scale=1)
        with pytest.raises(ModuleNotFoundError) as caught:
            probe_module.probe(*values, impl=missing_impl)
        assert caught.value is error
        kernels[missing_impl].assert_not_called()
        assert kernels[other].call_count == 1  # no fallback on failure

    assert sys.modules[name] is leaf
    assert (
        probe_module.probe(*values, impl=missing_impl)
        is kernels[missing_impl].return_value
    )
    kernels[missing_impl].assert_called_once_with(*values, scale=1)
    assert kernels[other].call_count == 1
