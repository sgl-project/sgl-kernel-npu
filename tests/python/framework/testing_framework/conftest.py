"""Fixtures for accuracy and performance helpers using a lightweight NPU operator."""

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


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
