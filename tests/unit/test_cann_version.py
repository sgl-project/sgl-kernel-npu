"""CPU tests for SDK metadata detection and version-selected binding dispatch."""

import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch
from uuid import uuid4

import pytest

pytest.importorskip("torch")

UTILS = (
    Path(__file__).resolve().parents[2]
    / "python/sgl_kernel_npu/sgl_kernel_npu/operators/utils"
)


@pytest.fixture
def runtime():
    # Load only the utility package, avoiding the NPU bootstrap and registry.
    name = "_cann_version_test_" + uuid4().hex
    package = ModuleType(name)
    package.__path__ = [str(UTILS)]
    sys.modules[name] = package
    module = importlib.import_module(name + "._transformer_compat")
    yield module
    for key in list(sys.modules):
        if key == name or key.startswith(name + "."):
            del sys.modules[key]


@pytest.fixture
def bindings(runtime, monkeypatch, tmp_path):
    sdk = tmp_path / "cann-sdk"
    sdk.mkdir()
    monkeypatch.setenv("ASCEND_HOME_PATH", str(sdk))

    def set_version(version):
        (sdk / "version.cfg").write_text(f"version={version}\n")

    set_version("9.2.0")
    calls = {
        ("cann_ops_transformer", "mhc_post"): Mock(),
        ("custom", "npu_hc_post"): Mock(),
    }
    monkeypatch.setattr(
        runtime, "optional_op", lambda namespace, name: calls.get((namespace, name))
    )
    return SimpleNamespace(runtime=runtime, calls=calls, set_version=set_version)


@pytest.mark.parametrize(
    "primary,secondary,expected",
    [
        ("version=9.1.0\n", None, "9.1.0"),
        ("  VERSION_DIR = 9.2.3 \n", None, "9.2.3"),
        (None, "version=9.2.0\r\n", "9.2.0"),
        ("package=cann\n", "version_dir=9.1.1\n", "9.1.1"),
        ("version=9.1.0\n", "version=9.2.0\n", "9.1.0"),
    ],
)
def test_detect_cann_version_from_sdk_files(
    runtime, monkeypatch, tmp_path, primary, secondary, expected
):
    sdk = tmp_path / "sdk"
    sdk.mkdir()
    if primary is not None:
        (sdk / "version.cfg").write_text(primary)
    if secondary is not None:
        (sdk / "compiler").mkdir()
        (sdk / "compiler/version.info").write_text(secondary)
    monkeypatch.setenv("ASCEND_HOME_PATH", str(sdk))
    assert runtime.detect_cann_version() == expected


@pytest.mark.parametrize("location", [None, "missing", "file"])
def test_detect_cann_version_requires_initialized_sdk(
    runtime, monkeypatch, tmp_path, location
):
    if location is None:
        monkeypatch.delenv("ASCEND_HOME_PATH", raising=False)
    else:
        sdk = tmp_path / location
        if location == "file":
            sdk.write_text("version=9.2.0")
        monkeypatch.setenv("ASCEND_HOME_PATH", str(sdk))
    with pytest.raises(RuntimeError, match="ASCEND_HOME_PATH"):
        runtime.detect_cann_version()


@pytest.mark.parametrize(
    "text",
    [
        "",
        "version=9.1",
        "version=9.2.0-beta.1",
        "version=9.2.0 extra",
        "version_dir=/sdk/9.2.0",
    ],
)
def test_detect_cann_version_rejects_inexact_metadata(
    runtime, monkeypatch, tmp_path, text
):
    monkeypatch.setenv("ASCEND_HOME_PATH", str(tmp_path))
    (tmp_path / "version.cfg").write_text(text)
    with pytest.raises(RuntimeError, match="exact CANN release") as error:
        runtime.detect_cann_version()
    assert "version.cfg" in str(error.value) and "version.info" in str(error.value)


def test_detect_cann_version_reports_unreadable_metadata(
    runtime, monkeypatch, tmp_path
):
    monkeypatch.setenv("ASCEND_HOME_PATH", str(tmp_path))
    (tmp_path / "version.cfg").touch()
    with patch.object(runtime.Path, "read_text", side_effect=PermissionError("denied")):
        with pytest.raises(RuntimeError, match="Cannot read CANN metadata") as error:
            runtime.detect_cann_version()
    assert isinstance(error.value.__cause__, PermissionError)


@pytest.mark.parametrize(
    "version,expected",
    [
        ("9.1.0", "legacy"),
        ("9.1.12", "legacy"),
        ("9.2.0", "cann_ops_transformer"),
        ("9.2.9", "cann_ops_transformer"),
    ],
)
def test_cann_patch_release_selects_its_version_family(
    runtime, bindings, version, expected
):
    bindings.set_version(version)
    assert bindings.runtime.cann_backend() == expected


@pytest.mark.parametrize("version", ["8.2.0", "9.0.0", "9.3.0", "9.10.0", "10.1.0"])
def test_unsupported_cann_release_never_probes_bindings(
    runtime, bindings, monkeypatch, version
):
    bindings.set_version(version)
    lookup = Mock(
        side_effect=AssertionError("unsupported release must not resolve ops")
    )
    monkeypatch.setattr(bindings.runtime, "optional_op", lookup)
    with pytest.raises(
        bindings.runtime.OperatorUnavailableError, match="Unsupported CANN release"
    ):
        bindings.runtime.select_family((("new", "op"),), (("old", "op"),))
    lookup.assert_not_called()


@pytest.mark.parametrize(
    "version,namespace", [("9.1.0", "custom"), ("9.2.0", "cann_ops_transformer")]
)
def test_version_selected_backend_does_not_probe_the_other_installation(
    runtime, bindings, monkeypatch, version, namespace
):
    bindings.set_version(version)
    resolved = []

    def lookup(ns, name):
        resolved.append(ns)
        return bindings.calls.get((ns, name))

    monkeypatch.setattr(bindings.runtime, "optional_op", lookup)
    family = bindings.runtime.select_family(
        (("cann_ops_transformer", "mhc_post"),), (("custom", "npu_hc_post"),)
    )
    assert resolved == [namespace]
    assert family.legacy == (version == "9.1.0")


def test_detected_sdk_version_is_cached_before_kv_allocation(runtime, bindings):
    assert bindings.runtime.detect_cann_version() == "9.2.0"
    assert bindings.runtime.cann_backend() == "cann_ops_transformer"
    bindings.set_version("9.1.0")
    assert bindings.runtime.detect_cann_version() == "9.2.0"
    assert bindings.runtime.cann_backend() == "cann_ops_transformer"


@pytest.mark.parametrize(
    "version,namespace,name",
    [("9.1.0", "custom", "npu_hc_post"), ("9.2.0", "cann_ops_transformer", "mhc_post")],
)
def test_missing_version_selected_binding_is_an_error(
    bindings, version, namespace, name
):
    bindings.set_version(version)
    del bindings.calls[namespace, name]
    with pytest.raises(
        bindings.runtime.OperatorUnavailableError, match=f"CANN {version}"
    ):
        bindings.runtime.select_family(
            (("cann_ops_transformer", "mhc_post"),), (("custom", "npu_hc_post"),)
        )
    assert all(not call.called for call in bindings.calls.values())


@pytest.mark.parametrize("version", ["9.1.0", "9.2.0"])
def test_schema_support_is_checked_only_in_selected_version(
    bindings, monkeypatch, version
):
    import torch

    bindings.set_version(version)
    namespace = "_cann_schema_" + uuid4().hex
    library = torch.library.Library(namespace, "DEF")
    library.define("released(Tensor x) -> Tensor")
    library.define("legacy(Tensor x, *, Tensor? input_ids=None) -> Tensor")
    packet = getattr(torch.ops, namespace)
    monkeypatch.setattr(
        bindings.runtime, "optional_op", lambda ns, name: getattr(packet, name)
    )
    args = ((("cann_ops_transformer", "released"),), (("custom", "legacy"),))
    if version == "9.2.0":
        with pytest.raises(
            bindings.runtime.OperatorUnavailableError, match="argument input_ids"
        ):
            bindings.runtime.select_family(*args, required_arguments=("input_ids",))
    else:
        family = bindings.runtime.select_family(
            *args, required_arguments=("input_ids",)
        )
        assert family.legacy


def test_sdk_path_may_be_a_latest_symlink(runtime, monkeypatch, tmp_path):
    sdk = tmp_path / "cann-9.2.0"
    sdk.mkdir()
    (sdk / "version.cfg").write_text("version=9.2.0\n")
    link = tmp_path / "latest"
    link.symlink_to(sdk, target_is_directory=True)
    monkeypatch.setenv("ASCEND_HOME_PATH", str(link))
    assert runtime.detect_cann_version() == "9.2.0"
