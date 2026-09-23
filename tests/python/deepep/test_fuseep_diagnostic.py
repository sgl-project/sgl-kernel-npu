"""CPU-only regressions for failure reporting and private-source selection."""

import contextlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "diagnostic", Path(__file__).with_name("diagnose_sglang_fuseep.py")
)
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


class DiagnosticTests(unittest.TestCase):
    def test_no_samples_are_unknown(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "mode2.log").write_text("original full-model assertion failed")
            with contextlib.redirect_stdout(io.StringIO()):
                diagnostic.summarize(root, 1)
            report = json.loads((root / "layer-summary.json").read_text())
            self.assertEqual(report["diagnostic_status"], "no_comparisons")
            self.assertTrue(report["artifacts"]["mode2.log"])
            self.assertEqual(report["original_test_exit_code"], 1)
            for name in ("int32_dequant_swiglu", "bf16_gmm_swiglu"):
                self.assertIsNone(report[name]["all_finite"])
                self.assertIsNone(report[name]["all_recv_counts_equal"])

    def test_real_failure_is_not_hidden_by_unknown_status(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "layers").mkdir()
            value = dict(
                mean_abs=1.0,
                reference_finite=True,
                candidate_finite=False,
                recv_counts_equal=False,
            )
            row = {name: value for name in ("int32_dequant_swiglu", "bf16_gmm_swiglu")}
            (root / "layers/rank0.jsonl").write_text(json.dumps(row) + "\n")
            with contextlib.redirect_stdout(io.StringIO()):
                diagnostic.summarize(root, 1)
            report = json.loads((root / "layer-summary.json").read_text())
            self.assertIs(report["int32_dequant_swiglu"]["all_finite"], False)
            self.assertIs(
                report["int32_dequant_swiglu"]["all_recv_counts_equal"], False
            )

    def test_local_proxy_bypass_preserves_remote_configuration(self):
        env = dict(
            HTTP_PROXY="http://proxy.example:3128",
            NO_PROXY="example.org",
            no_proxy="internal.example",
        )
        diagnostic.bypass_local_proxy(env)
        self.assertEqual(env["HTTP_PROXY"], "http://proxy.example:3128")
        self.assertEqual(env["NO_PROXY"], env["no_proxy"])
        self.assertEqual(
            set(env["NO_PROXY"].split(",")),
            {"example.org", "internal.example", "127.0.0.1", "localhost", "::1"},
        )

    def test_console_is_saved_and_exit_code_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            log = Path(temp) / "console.log"
            console = io.StringIO()
            with contextlib.redirect_stdout(console):
                status = diagnostic.run_test(
                    [
                        sys.executable,
                        "-u",
                        "-c",
                        "print('original failure'); raise SystemExit(7)",
                    ],
                    os.environ.copy(),
                    log,
                )
            self.assertEqual(status, 7)
            self.assertIn("original failure", console.getvalue())
            self.assertIn("original failure", log.read_text())

    def test_test_and_server_use_private_source_despite_shadow_package(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            private, shadow = root / "private", root / "shadow"
            for path in (
                private / "sglang",
                private / "sglang/test",
                private / "sglang/cli",
                shadow / "sglang",
            ):
                path.mkdir(parents=True, exist_ok=True)
                (path / "__init__.py").write_text("")
            (shadow / "sglang/__init__.py").write_text(
                "raise RuntimeError('shadow imported')"
            )
            (private / "sglang/test/test_utils.py").write_text(
                "import subprocess\ndef _launch_server_process(command, env):\n"
                "    return subprocess.run(command, env=env, check=True)\n"
            )
            (private / "sglang/cli/main.py").write_text(
                "import sys\ndef main():\n"
                "    assert sys.argv[1:] == ['serve', '--model-path', 'dummy']\n"
                "    print('PRIVATE_SERVER_OK')\n"
            )
            test = root / "case.py"
            test.write_text(
                "import os\nfrom sglang.test import test_utils\n"
                "test_utils._launch_server_process(['sglang', 'serve', '--model-path', 'dummy'], env=os.environ.copy())\n"
            )
            bootstrap = root / "bootstrap.py"
            bootstrap.write_text(diagnostic.BOOTSTRAP)
            env = {
                **os.environ,
                "PYTHONPATH": str(shadow),
                "SGLANG_FUSEEP_AUDIT_PYTHON": str(private),
            }
            result = subprocess.run(
                [sys.executable, str(bootstrap), "test", str(test)],
                env=env,
                cwd=shadow,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("role=test", result.stdout)
            self.assertIn("role=server", result.stdout)
            self.assertIn("PRIVATE_SERVER_OK", result.stdout)


if __name__ == "__main__":
    unittest.main()
