"""Exercise lazy backend setup and repeated device guards without an NPU."""

import importlib.metadata
import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
HELPER = ROOT / "python/sgl_kernel_npu/sgl_kernel_npu/operators/utils/cannbot_dsl.py"


@pytest.fixture
def helper(monkeypatch):
    spec = importlib.util.spec_from_file_location("_test_cannbot_helper", HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.importlib.metadata, "version", lambda _: "0.8.0")
    return module


@pytest.mark.parametrize("version", [None, "0.7.1", "0.9.0"])
def test_wrong_or_missing_dsl_is_rejected_before_import(helper, monkeypatch, version):
    def installed(name):
        if version is None:
            raise importlib.metadata.PackageNotFoundError(name)
        return version

    monkeypatch.setattr(helper.importlib.metadata, "version", installed)
    monkeypatch.setattr(
        helper.importlib, "import_module", lambda _: pytest.fail("source imported")
    )
    with pytest.raises(RuntimeError, match="cannbot-dsl"):
        helper.load_cannbot_dsl("quant_lightning_indexer")


def test_source_jit_can_run_without_native_bundle(helper):
    helper.initialize_backend()


def test_device_identity_is_cached_but_context_tracks_current_device(
    helper, monkeypatch
):
    queried, switches = [], []
    current = {"index": 0}

    @contextmanager
    def device(index):
        previous = current["index"]
        switches.append(index)
        current["index"] = index
        try:
            yield
        finally:
            current["index"] = previous

    torch = ModuleType("torch")
    torch.npu = SimpleNamespace(
        current_device=lambda: current["index"],
        get_device_name=lambda index: queried.append(index) or "Ascend950PR_9599",
        device=device,
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    tensor = SimpleNamespace(device=SimpleNamespace(type="npu", index=0))
    for _ in range(3):
        helper.require_ascend950(tensor)
        with helper.ascend950_device(tensor):
            assert current["index"] == 0
    assert queried == [0]
    assert switches == []
    current["index"] = 1
    with helper.ascend950_device(tensor):
        assert current["index"] == 0
    assert current["index"] == 1
    assert queried == [0]
    assert switches == [0]


def test_unsupported_device_failure_is_not_cached(helper, monkeypatch):
    model = {"name": "Ascend910_9362"}
    torch = ModuleType("torch")
    torch.npu = SimpleNamespace(
        current_device=lambda: 0, get_device_name=lambda _: model["name"]
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    tensor = SimpleNamespace(device=SimpleNamespace(type="npu", index=0))
    with pytest.raises(RuntimeError, match="Ascend 950"):
        helper.require_ascend950(tensor)
    model["name"] = "Ascend950DT_9591"
    assert helper.require_ascend950(tensor) == 0


@pytest.mark.parametrize("current", [0, 1])
def test_implicit_and_keyword_devices_resolve_to_one_npu(helper, monkeypatch, current):
    queried = []
    torch = ModuleType("torch")
    torch.npu = SimpleNamespace(
        current_device=lambda: queried.append(current) or current,
        get_device_name=lambda _: "Ascend950PR_9599",
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    implicit = SimpleNamespace(device=SimpleNamespace(type="npu", index=None))
    explicit = SimpleNamespace(device=SimpleNamespace(type="npu", index=current))
    assert (
        helper.require_ascend950(implicit, None, out=explicit, metadata=implicit)
        == current
    )
    assert queried == [current]
    other = SimpleNamespace(device=SimpleNamespace(type="npu", index=1 - current))
    with pytest.raises(RuntimeError, match="same NPU device"):
        helper.require_ascend950(implicit, out=other)


@pytest.mark.parametrize("device", [None, SimpleNamespace(type="cpu", index=None)])
def test_invalid_device_in_keyword_is_rejected(helper, monkeypatch, device):
    torch = ModuleType("torch")
    torch.npu = SimpleNamespace(
        current_device=lambda: 0,
        get_device_name=lambda _: pytest.fail("invalid device reached identity lookup"),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    npu = SimpleNamespace(device=SimpleNamespace(type="npu", index=0))
    with pytest.raises(RuntimeError, match="NPU tensors"):
        helper.require_ascend950(npu, out=SimpleNamespace(device=device))


def test_metadata_without_tensor_inputs_uses_current_npu(helper, monkeypatch):
    queried = []
    torch = ModuleType("torch")
    torch.npu = SimpleNamespace(
        current_device=lambda: 1,
        get_device_name=lambda index: queried.append(index) or "Ascend950DT_9591",
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    assert helper.require_ascend950(None, batch_size=2, topk=512) == 1
    assert queried == [1]
