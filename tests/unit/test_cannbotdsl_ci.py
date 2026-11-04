"""The CI result gate must reject a green pytest run with missing NPU coverage."""

import importlib.util
import xml.etree.ElementTree as ET
from pathlib import Path

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
