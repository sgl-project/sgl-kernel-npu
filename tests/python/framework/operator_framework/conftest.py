"""Fixtures for operator registration, architecture and probe-entry tests."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest


@pytest.fixture
def registry(operator_modules):
    return operator_modules.registry


@pytest.fixture
def generation_module(package_factory):
    return package_factory().generation


@pytest.fixture
def registry_for_generation(package_factory, monkeypatch):
    """Simulate registration on another generation, not runtime device switching."""

    def create(name):
        package = package_factory()
        generation = package.generation
        arch_id = {"ARCH22": 2201, "ARCH35": 3510}[name]
        hardware = generation.Hardware(arch_id, generation.Generation[name])
        detector = Mock(return_value=hardware)
        monkeypatch.setattr(generation, "get_hardware", detector)
        modules = package.load()
        modules.hardware = hardware
        modules.detector = detector
        return modules

    return create


@pytest.fixture
def probe_loader():
    """Load a test-only public entry under the isolated package, at most once."""

    def load(modules):
        name = f"{modules.namespace}._framework_probe"
        if name in sys.modules:
            return sys.modules[name]
        source = Path(__file__).resolve().parents[1] / "_support" / "probe.py"
        spec = importlib.util.spec_from_file_location(name, source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            del sys.modules[name]
            raise
        return module

    return load


@pytest.fixture
def probe_module(operator_modules, probe_loader):
    return probe_loader(operator_modules)


@pytest.fixture
def kernels(operator_modules, monkeypatch):
    """Supply inert backend leaves without importing any production operator."""
    parent_name = f"{operator_modules.namespace}._probe_backends"
    parent = ModuleType(parent_name)
    parent.__path__ = []
    monkeypatch.setitem(sys.modules, parent_name, parent)
    backends = {}
    for impl in ("backend_a", "backend_b"):
        name = f"{parent_name}.{impl}"
        module = ModuleType(name)
        module.run = Mock(name=f"{impl}_kernel", return_value=object())
        monkeypatch.setitem(sys.modules, name, module)
        backends[impl] = module.run
    return backends
