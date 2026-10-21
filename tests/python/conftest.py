"""Fixtures for the new public-interface tests; no global device side effects."""

from argparse import ArgumentTypeError
from pathlib import Path

import pytest

from tests.python.utils.common import use_impls
from tests.python.utils.performance import performance_context


def is_public_interface_test(path):
    root = Path(__file__).resolve().parent
    return Path(path).resolve().is_relative_to(root / "operators")


def parse_bool_switch(value):
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    raise ArgumentTypeError("Expected true or false")


def pytest_addoption(parser):
    group = parser.getgroup("operator implementations")
    group.addoption(
        "--prof-dir",
        default=None,
        help="Profiling root; overrides code settings (default: cwd/profilings).",
    )
    group.addoption(
        "--csv-path",
        default=None,
        help="Append performance results to this CSV; overrides code settings.",
    )
    group.addoption(
        "--impl",
        default="auto",
        help="Implementation to validate and benchmark (default: auto).",
    )
    group.addoption(
        "--impl-ref",
        default="torch_native",
        help="Implementation used for accuracy comparison (default: torch_native).",
    )
    group.addoption(
        "--npu-device",
        type=int,
        default=None,
        help="Visible logical NPU index; omitted means keep the current NPU.",
    )
    group.addoption(
        "--enable-perf",
        nargs="?",
        const=True,
        default=None,
        type=parse_bool_switch,
        help="Override all case settings: --enable-perf[=true|false]; omitted uses code settings.",
    )

    group.addoption(
        "--skip-acc-check",
        nargs="?",
        const=True,
        default=None,
        type=parse_bool_switch,
        help="Override accuracy skipping: --skip-acc-check[=true|false]; default: False.",
    )


def pytest_generate_tests(metafunc):
    if (
        is_public_interface_test(metafunc.definition.path)
        and "impl_pair" in metafunc.fixturenames
    ):
        pair = (
            metafunc.config.getoption("impl"),
            metafunc.config.getoption("impl_ref"),
        )
        metafunc.parametrize(
            "impl_pair",
            [pair],
            indirect=True,
            ids=[f"{pair[0]}-vs-{pair[1]}"],
        )


@pytest.fixture
def configure_npu(request):
    from tests.python.utils.device import use_npu_device, validate_npu_options

    config = request.config
    workers = getattr(config, "workerinput", {}).get(
        "workercount",
        config.getoption("numprocesses", default=0),
    )
    index = config.getoption("npu_device")
    try:
        validate_npu_options(index, workers)
    except ValueError as error:
        raise pytest.UsageError(str(error)) from error
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("An Ascend NPU is required")
    if index is not None and index >= torch.npu.device_count():
        raise pytest.UsageError(
            f"--npu-device={index} is outside the visible NPU range"
        )
    with use_npu_device(torch, index):
        yield


@pytest.fixture
def impl_pair(request):
    return getattr(request, "param", None)


@pytest.fixture(autouse=True)
def configure_public_interface_test(request, impl_pair):
    if not is_public_interface_test(request.node.path):
        yield
        return

    # Resolve these fixtures only for new tests, never for legacy tests.
    request.getfixturevalue("configure_npu")
    record_property = request.getfixturevalue("record_property")
    parameters = getattr(getattr(request.node, "callspec", None), "params", {})
    with use_impls(
        *impl_pair,
        skip_acc_check=request.config.getoption("skip_acc_check"),
    ), performance_context(
        request.node.nodeid,
        parameters,
        record_property,
        enable_perf=request.config.getoption("enable_perf"),
        prof_dir=request.config.getoption("prof_dir"),
        csv_path=request.config.getoption("csv_path"),
    ):
        yield
