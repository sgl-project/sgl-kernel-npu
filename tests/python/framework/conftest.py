"""Shared NPU prerequisites and isolated checkout framework modules."""

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from uuid import uuid4

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "npu_integration: collect real NPU profiling data (no profiler stub)"
    )


@pytest.fixture(scope="session", autouse=True)
def npu_runtime(request):
    """Require a working NPU installation; never turn a broken setup into skips."""
    config = request.config
    workers = getattr(config, "workerinput", {}).get(
        "workercount", config.getoption("numprocesses", default=0)
    )
    index = config.getoption("npu_device")
    if index is not None and index < 0:
        raise pytest.UsageError(
            "--npu-device must be a nonnegative visible device index"
        )
    if workers not in (None, 0, 1, "0", "1"):
        raise pytest.UsageError(
            "Framework UT requires a single worker; remove -n or use -n 1."
        )

    import acl
    import torch
    import torch_npu  # noqa: F401 -- registers torch.npu

    if not torch.npu.is_available():
        pytest.fail("Framework UT requires an available Ascend NPU", pytrace=False)
    if index is not None and index >= torch.npu.device_count():
        pytest.fail("--npu-device is outside the visible NPU range", pytrace=False)

    previous_device = torch.npu.current_device()
    try:
        if index is not None:
            torch.npu.set_device(index)
        # Normal import executes the installed package's native-library bootstrap.
        package = importlib.import_module("sgl_kernel_npu")
        source = (
            Path(__file__).resolve().parents[3]
            / "python"
            / "sgl_kernel_npu"
            / "sgl_kernel_npu"
        )
        # The installed package supplies the native runtime, not the Python
        # framework under test. package_factory loads operators from checkout;
        # an older installed package need not contain operators at all.
        yield SimpleNamespace(
            acl=acl,
            torch=torch,
            package=package,
            soc_name=torch.npu.get_device_name(),
            source=source,
        )
    finally:
        if index is not None:
            torch.npu.set_device(previous_device)


@pytest.fixture
def package_factory(npu_runtime):
    """Fresh source modules per case, without altering installed registrations.

    The real package has already loaded above. Private namespaces isolate test
    registrations and lazy-import caches; they are not a CPU compatibility path.
    Generation is loaded first so tests can replace hardware detection BEFORE
    importing registry. The real operators __init__ still runs on load().
    """
    namespaces = []

    def create():
        namespace = f"_sgl_framework_ut_{uuid4().hex}"
        namespaces.append(namespace)
        package = ModuleType(namespace)
        package.__path__ = [str(npu_runtime.source)]
        package.__package__ = namespace
        sys.modules[namespace] = package

        name = f"{namespace}.operators.utils.generation"
        path = npu_runtime.source / "operators" / "utils" / "generation.py"
        spec = importlib.util.spec_from_file_location(name, path)
        generation = importlib.util.module_from_spec(spec)
        sys.modules[name] = generation
        spec.loader.exec_module(generation)

        def import_module(suffix):
            return importlib.import_module(f"{namespace}.{suffix}")

        def load():
            """Finish imports after the caller has configured hardware detection."""
            api = import_module("operators")
            registry = import_module("operators.utils.registry")
            return SimpleNamespace(
                namespace=namespace,
                public_api=api,
                registry=registry,
                generation=generation,
                import_module=import_module,
            )

        # This preparation handle intentionally has no registry/public_api yet.
        return SimpleNamespace(
            namespace=namespace,
            generation=generation,
            import_module=import_module,
            load=load,
        )

    try:
        yield create
    finally:
        for name in list(sys.modules):
            if any(name == ns or name.startswith(f"{ns}.") for ns in namespaces):
                del sys.modules[name]


@pytest.fixture
def operator_modules(package_factory):
    return package_factory().load()
