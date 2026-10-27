"""Real CPU pytest runs verify complete execution and nonzero failure status."""

import os
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import run_selected_kernel_tests as runner


class CompletionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        environment = patch.dict(
            os.environ,
            {
                "GITHUB_STEP_SUMMARY": "",
                "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
                # CI arguments must override inherited early-stop defaults.
                "PYTEST_ADDOPTS": "-x",
                "PATH": str(Path(sys.executable).parent)
                + os.pathsep
                + os.environ["PATH"],
            },
        )
        environment.start()
        self.addCleanup(environment.stop)

    def write(self, path, content):
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        return path

    def run_selected(self, tests):
        plan = {"modules": {"norm": {"tests": tests}}, "cann_exclusions": {}}
        return runner.run(plan, "norm", self.root, "9.0.0")

    def cases(self, report):
        return ET.parse(self.root / "test-results" / report).findall(".//testcase")

    def test_public_parameter_cases_and_later_files_finish_after_failure(self):
        self.write(
            "conftest.py",
            "def pytest_addoption(parser):\n"
            "    for option in ['impl', 'impl-ref', 'skip-acc-check', 'enable-perf', 'npu-device']:\n"
            "        parser.addoption('--' + option)\n",
        )
        tests = [
            self.write(
                "tests/python/operators/norm/test_first.py",
                "import pytest\n"
                "@pytest.mark.parametrize('value', [0, 1, 2])\n"
                "def test_parameter(value):\n"
                "    assert value != 0\n",
            ),
            self.write(
                "tests/python/operators/norm/test_later.py",
                "def test_later(): pass\n",
            ),
            self.write(
                "tests/python/sgl_kernel_npu/test_legacy.py",
                "def test_legacy(): pass\n",
            ),
        ]
        self.assertEqual(self.run_selected(tests), 1)
        cases = self.cases("public-norm.xml")
        self.assertEqual(len(cases), 4)
        self.assertEqual(sum(c.find("failure") is not None for c in cases), 1)
        self.assertIn("test_parameter[2]", {c.attrib["name"] for c in cases})
        self.assertIn("test_later", {c.attrib["name"] for c in cases})
        self.assertEqual(len(self.cases("legacy-test_legacy.xml")), 1)

    def test_collection_error_does_not_suppress_valid_public_tests(self):
        self.write(
            "conftest.py",
            "def pytest_addoption(parser):\n"
            "    for option in ['impl', 'impl-ref', 'skip-acc-check', 'enable-perf', 'npu-device']:\n"
            "        parser.addoption('--' + option)\n",
        )
        tests = [
            self.write(
                "tests/python/operators/norm/test_broken.py",
                "raise RuntimeError('collection failure')\n",
            ),
            self.write(
                "tests/python/operators/norm/test_valid.py",
                "def test_valid(): pass\n",
            ),
        ]
        self.assertEqual(self.run_selected(tests), 1)
        cases = self.cases("public-norm.xml")
        self.assertTrue(any(c.find("error") is not None for c in cases))
        self.assertTrue(any(c.attrib["name"] == "test_valid" for c in cases))

    def test_legacy_cases_and_later_scripts_finish_after_failure(self):
        tests = [
            self.write(
                "tests/python/sgl_kernel_npu/test_cases.py",
                "def test_first(): assert False\n" "def test_later(): pass\n",
            ),
            self.write(
                "tests/python/sgl_kernel_npu/test_script.py",
                "print('later script ran')\n",
            ),
        ]
        self.assertEqual(self.run_selected(tests), 1)
        cases = self.cases("legacy-test_cases.xml")
        self.assertEqual(len(cases), 2)
        self.assertEqual(sum(c.find("failure") is not None for c in cases), 1)
        self.assertEqual(len(self.cases("legacy-test_script.xml")), 1)

    def test_original_shell_group_finishes_cases_and_files_before_failing(self):
        self.write(
            "tests/python/sgl_kernel_npu/test_rmsnorm_split.py",
            "from pathlib import Path\n"
            "def test_first(): assert False\n"
            "def test_later(): Path('later-case-ran').touch()\n",
        )
        for name in ["rmsnorm_without_weight", "l1_norm", "scale_shift"]:
            self.write(
                f"tests/python/sgl_kernel_npu/test_{name}.py",
                f"from pathlib import Path\nPath('{name}-ran').touch()\n",
            )
        result = subprocess.run(
            ["bash", str(ROOT / "scripts/run_kernel_tests.sh"), "norm"],
            cwd=self.root,
            env=dict(os.environ, GITHUB_WORKSPACE=str(self.root), CANN_VERSION="9.0.0"),
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        folder = self.root / "tests/python/sgl_kernel_npu"
        for marker in [
            "later-case-ran",
            "rmsnorm_without_weight-ran",
            "l1_norm-ran",
            "scale_shift-ran",
        ]:
            self.assertTrue((folder / marker).exists(), result.stdout + result.stderr)
        self.assertIn("Failed: 1", result.stdout)


if __name__ == "__main__":
    unittest.main()
