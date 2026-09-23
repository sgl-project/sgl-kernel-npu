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
from types import SimpleNamespace
from unittest import mock

spec = importlib.util.spec_from_file_location(
    "diagnostic", Path(__file__).with_name("diagnose_sglang_fuseep.py")
)
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


class DiagnosticTests(unittest.TestCase):
    def test_logprob_comparison_checks_alignment(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)

            def save(name, values):
                (root / name).write_text(
                    json.dumps(
                        {
                            "generation": [
                                {"meta_info": {"input_token_logprobs": values}}
                            ]
                        }
                    )
                )

            save("baseline.json", [[None, 1], [-1.0, 2], [-3.0, 3]])
            save("mode2.json", [[None, 1], [-1.5, 2], [-2.0, 3]])
            report = diagnostic.compare_logprobs(root)
            self.assertEqual(report["tokens_compared"], 2)
            self.assertEqual(report["mean_abs_logprob_diff"], 0.75)
            self.assertEqual(report["max_abs_logprob_diff"], 1.0)
            save("mode2.json", [[None, 1], [-1.5, 2], [-2.0, 4]])
            self.assertEqual(
                diagnostic.compare_logprobs(root)["status"], "invalid_results"
            )

    def test_intervention_is_explicitly_labeled_in_report(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "diagnostic-config.json").write_text(
                json.dumps({"model_output": "bf16_gmm_swiglu"})
            )
            console = io.StringIO()
            with contextlib.redirect_stdout(console):
                diagnostic.summarize(root, 0)
            report = json.loads((root / "layer-summary.json").read_text())
            self.assertFalse(report["fused_outputs_preserved"])
            self.assertEqual(report["model_output"], "bf16_gmm_swiglu")
            self.assertIn("does not validate", console.getvalue())

    def test_output_selection_applies_to_decode_and_repeated_prefill(self):
        # Exercise the real audit control flow without CANN. References are mocked;
        # NPU integration separately checks the arithmetic and communication.
        path = Path(__file__).with_name("_fuseep_real_input_reference.py")
        for selected in ("fused", "int32_dequant_swiglu", "bf16_gmm_swiglu"):
            for tokens, seen in ((1, False), (8, False), (8, True)):
                with self.subTest(selected=selected, tokens=tokens, seen=seen):
                    torch = mock.MagicMock()
                    torch.tensor.return_value.item.return_value = tokens
                    torch.equal.return_value = True
                    torch.distributed.get_rank.return_value = 0
                    spec = importlib.util.spec_from_file_location("audit_helper", path)
                    helper = importlib.util.module_from_spec(spec)
                    with mock.patch.dict(
                        sys.modules,
                        {
                            "torch": torch,
                            "torch.distributed": torch.distributed,
                            "torch_npu": mock.MagicMock(),
                            "deep_ep": mock.MagicMock(),
                        },
                    ):
                        spec.loader.exec_module(helper)
                    helper._buffer = object()
                    helper._scales = {0: (None, None)}
                    helper._seen = {(0, tokens)} if seen else set()
                    helper.progress = mock.Mock()
                    helper.metrics = mock.Mock(return_value={"mean_abs": 0.0})
                    reference = mock.MagicMock()
                    chosen_output = object()
                    reference.detach.return_value.clone.return_value = chosen_output
                    helper.reference = mock.Mock(
                        return_value=(reference, mock.MagicMock())
                    )
                    x = mock.MagicMock()
                    x.shape = (tokens, 2)
                    x.numel.return_value = tokens * 2
                    x.detach.return_value.float.return_value.abs.return_value.max.return_value.item.return_value = (
                        1.0
                    )
                    layer = SimpleNamespace(
                        layer_id=0,
                        w13_weight=SimpleNamespace(shape=(2, 2, 4)),
                        w2_weight=SimpleNamespace(shape=(2, 2, 2)),
                    )
                    original = object()
                    with tempfile.TemporaryDirectory() as temp, mock.patch.dict(
                        os.environ,
                        {
                            "SGLANG_FUSEEP_AUDIT_DIR": temp,
                            "SGLANG_FUSEEP_AUDIT_MODEL_OUTPUT": selected,
                            "SGLANG_FUSEEP_AUDIT_MIN_TOKENS": "4",
                        },
                    ), contextlib.redirect_stdout(io.StringIO()):
                        result = helper.audit(
                            layer,
                            mock.MagicMock(),
                            x,
                            mock.MagicMock(),
                            original,
                            mock.MagicMock(),
                        )
                    self.assertIs(
                        result, original if selected == "fused" else chosen_output
                    )
                    capture = tokens >= 4 and not seen
                    expected_calls = 2 if capture else (0 if selected == "fused" else 1)
                    self.assertEqual(helper.reference.call_count, expected_calls)
                    if not capture and selected != "fused":
                        self.assertEqual(helper.reference.call_args.args[-1], selected)

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
