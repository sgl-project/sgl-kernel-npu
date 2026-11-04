"""The CI result gate must reject a green pytest run with missing NPU coverage."""

import importlib.util
import json
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/check_cannbotdsl_ci.py"
spec = importlib.util.spec_from_file_location("_cannbotdsl_ci", SCRIPT)
ci = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ci)


def report(tmp_path, change=None):
    root = ET.Element("testsuites")
    suite = ET.SubElement(root, "testsuite")
    for i, name in enumerate(ci.REQUIRED_CASES):
        if change == "missing" and i == 0:
            continue
        case = ET.SubElement(
            suite, "testcase", name=name + ("sample]" if name.endswith("[") else "")
        )
        if i == 0 and change in ("skipped", "failure", "error"):
            ET.SubElement(case, change)
    path = tmp_path / "junit.xml"
    ET.ElementTree(root).write(path)
    return path


def test_all_required_device_cases_pass(tmp_path):
    result = ci.check_junit(report(tmp_path))
    assert result["passed"] == len(ci.REQUIRED_CASES)


@pytest.mark.parametrize("change", ["missing", "skipped", "failure", "error"])
def test_partial_or_skipped_device_suite_fails(tmp_path, change):
    with pytest.raises(RuntimeError, match="Incomplete device validation"):
        ci.check_junit(report(tmp_path, change))


def test_empty_successful_pytest_run_is_not_device_validation(tmp_path):
    path = tmp_path / "empty.xml"
    path.write_text('<testsuites><testsuite tests="0"/></testsuites>')
    with pytest.raises(RuntimeError, match="missing="):
        ci.check_junit(path)


@pytest.fixture
def installed_fixture(tmp_path, monkeypatch):
    source, package = tmp_path / "source", tmp_path / "installed"
    files = {
        "implementation/cannbot-dsl/attention/entry.py": "# fixture source\n",
        "implementation/cannbot-dsl/UPSTREAM.json": json.dumps(
            {"files": [{"file": "entry.py"}]}
        ),
        "operators/indexer/entry.py": "# public entry\n",
        "operators/indexer/__init__.py": "",
        "operators/attention/__init__.py": "",
        "operators/utils/cannbot_dsl.py": "# runtime helper\n",
    }
    for name, text in files.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    shutil.copytree(source, package)
    manifest = json.dumps({"arch_version": "3510"})
    for base in (package, tmp_path / "build/torch/package/sgl_kernel_npu"):
        (base / "lib").mkdir(parents=True)
        (base / "_build_manifest.json").write_text(manifest)
        (base / "lib/libsgl_kernel_npu.so").write_bytes(b"binary fixture")
    module = ModuleType("sgl_kernel_npu")
    module.__file__ = str(package / "__init__.py")
    monkeypatch.setitem(sys.modules, "sgl_kernel_npu", module)
    monkeypatch.setattr(ci, "SOURCE", source)
    monkeypatch.setattr(ci, "ROOT", tmp_path)
    return package


def test_installed_fixture_tracks_the_runtime_helper(installed_fixture):
    result = ci.installed_sources()
    assert "operators/utils/cannbot_dsl.py" in result["source_hashes"]


@pytest.mark.parametrize("change", ["helper", "missing_helper", "library"])
def test_installed_gate_rejects_stale_helper_or_library(installed_fixture, change):
    if change == "library":
        (installed_fixture / "lib/libsgl_kernel_npu.so").write_bytes(b"old binary")
    elif change == "missing_helper":
        (installed_fixture / "operators/utils/cannbot_dsl.py").unlink()
    else:
        (installed_fixture / "operators/utils/cannbot_dsl.py").write_text(
            "# old helper\n"
        )
    with pytest.raises(RuntimeError, match="differs from"):
        ci.installed_sources()
