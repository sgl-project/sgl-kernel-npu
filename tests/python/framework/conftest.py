"""NPU prerequisites and isolated copies of the real framework modules."""

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock
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
            api = import_module("operators")
            registry = import_module("operators.utils.registry")
            return SimpleNamespace(
                namespace=namespace,
                public_api=api,
                registry=registry,
                generation=generation,
                import_module=import_module,
            )

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
def rmsnorm_module(operator_modules):
    return operator_modules.import_module("operators.norm.rmsnorm_bias")


@pytest.fixture
def kernels(operator_modules, monkeypatch):
    """Replace only kernel leaves; keep actual public adapters and dispatcher."""
    backends = {}
    for impl in ("triton", "torch_native"):
        parent = f"implementation.{impl}.norm"
        operator_modules.import_module(parent)
        name = f"{operator_modules.namespace}.{parent}.rmsnorm_bias"
        module = ModuleType(name)
        module.rmsnorm_bias = Mock(name=f"{impl}_kernel", return_value=object())
        monkeypatch.setitem(sys.modules, name, module)
        backends[impl] = module.rmsnorm_bias
    return backends


@pytest.fixture
def accuracy_tools(npu_runtime):
    """Import the checkout's real accuracy helpers after NPU initialization."""
    return importlib.import_module("tests.python.utils.accuracy")


@pytest.fixture
def accuracy_case(operator_modules, accuracy_tools, npu_runtime, monkeypatch):
    """Use real registration and Tensor arithmetic with a per-case call log.

    common.py imports registry by its canonical name. Bind that import to the
    same isolated source module used by this operator, and restore it on teardown.
    No registry lookup, accuracy helper or Tensor comparison is replaced.
    """
    from tests.python.utils.common import use_impls

    registry = operator_modules.registry
    monkeypatch.setitem(
        sys.modules, "sgl_kernel_npu.operators.utils.registry", registry
    )
    calls = []
    op_name = "framework.accuracy_probe"

    @registry.register_operator(op_name)
    def operator(x, offset=1, *, scale=1, impl="auto"):
        return registry.get_impl(operator, impl)(x, offset, scale=scale)

    def add_impl(name, *, delta=0, error=None, supported_generations=None):
        def backend(x, offset=1, *, scale=1):
            calls.append((name, x, offset, scale))
            if error is not None:
                raise error
            return x * scale + offset + delta

        registry.register_impl(
            op_name, name, supported_generations=supported_generations
        )(backend)
        return backend

    torch = npu_runtime.torch
    x = torch.tensor(
        [1.0, 2.0], dtype=torch.float32, device=f"npu:{torch.npu.current_device()}"
    )
    # Reset context defaults explicitly: framework/ is outside operators/ setup.
    with use_impls("auto", "torch_native", skip_acc_check=None):
        yield SimpleNamespace(
            operator=operator,
            add_impl=add_impl,
            calls=calls,
            x=x,
            registry=registry,
            op_name=op_name,
            check_accuracy=accuracy_tools.check_accuracy,
        )


@pytest.fixture
def performance_case(accuracy_case, monkeypatch, tmp_path):
    """Keep real orchestration/reporting; substitute only the measurement boundary."""
    from tests.python.utils import performance

    monkeypatch.chdir(tmp_path)
    case = accuracy_case
    case.add_impl("dut")
    case.executor = case.check_accuracy(
        case.operator, args=(case.x,), impl="dut", impl_ref="ref", skip_acc_check=True
    )
    case.measurements = []

    def benchmark(executor, directory, *, warmup, active):
        directory = Path(directory)
        case.measurements.append((executor, directory, warmup, active))
        directory.mkdir(parents=True)
        (directory / "synthetic-profile.txt").write_text("measurement stub\n")
        return 12.5

    monkeypatch.setattr(performance, "benchmark_npu", benchmark)
    case.performance = performance
    with performance.performance_context("", {}, None):
        yield case
