"""CPU-only checks for fixture geometry, CLI isolation and result schema."""

from __future__ import annotations

import array
import contextlib
import io
import json
import subprocess
import sys
import tempfile
import types
import unittest
import venv
import zipfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from bench import _import_shm, _require_kernel_prefix  # noqa: E402
from compare import delta, show_matrix, standalone_accepted  # noqa: E402
from fixture import FixtureSpec, parse_miss_counts  # noqa: E402
from prepare_wheels import unpack  # noqa: E402
from report import _parity  # noqa: E402
from result_contract import timing_stats, validate_result  # noqa: E402
from run_matrix import load_config  # noqa: E402
from runtime_check import check_interpreter  # noqa: E402


class RuntimeCheckTest(unittest.TestCase):
    def test_isolated_runtime_without_torch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            venv.EnvBuilder(with_pip=False).create(root / "venv")
            executable = root / "venv/bin/python"
            if sys.platform == "win32":
                executable = root / "venv/Scripts/python.exe"
            with self.assertRaisesRegex(RuntimeError, "Missing modules: torch"):
                check_interpreter(str(executable))
            config = json.loads((ROOT / "arm-config.example.json").read_text())
            for arm in config["arms"]:
                arm["python"] = str(executable)
            config_path = root / "arms.json"
            config_path.write_text(json.dumps(config))
            output = root / "results"
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "run_matrix.py"),
                    "--arm-config",
                    str(config_path),
                    "--output-root",
                    str(output),
                    "--batch",
                    "1",
                    "--heads",
                    "8",
                    "--index-heads",
                    "32",
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Missing modules: torch", result.stderr)
            self.assertFalse(output.exists())


class TimingComparisonTest(unittest.TestCase):
    def test_standalone_acceptance_requires_its_own_accuracy_and_parity(self):
        result = {
            "arm": {"name": "post_migration"},
            "status": "failed",
            "accuracy": {
                "standalone": {"passed": True},
                "canonical_ids": {"passed": False},
            },
        }
        row = {
            "fixture_equal": True,
            "parity_to_native": {"standalone": {"passed": True}},
        }
        self.assertTrue(standalone_accepted(result, row))
        row["parity_to_native"]["standalone"]["passed"] = False
        self.assertFalse(standalone_accepted(result, row))
        row["parity_to_native"]["standalone"]["passed"] = True
        row["fixture_equal"] = False
        self.assertFalse(standalone_accepted(result, row))

    def test_failed_cases_show_deltas_without_claiming_acceptance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for arm, duration in (("pre_migration", 200), ("post_migration", 250)):
                path = root / "q4-graph" / arm / "result.json"
                path.parent.mkdir(parents=True)
                path.write_text(
                    json.dumps(
                        {
                            "case": {"query_width": 4, "mode": "graph"},
                            "arm": {"name": arm},
                            "status": "failed",
                            "timings": {
                                name: {"mean_us": duration}
                                for name in (
                                    "indexer_or_lim",
                                    "attention_or_copysfa",
                                    "chain",
                                )
                            },
                            "accuracy": {"chain": {"passed": False}},
                        }
                    )
                )
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                show_matrix(root, "mean_us")
            self.assertIn("CopySFA +25.00%", stream.getvalue())
            self.assertIn("NOT ACCEPTED", stream.getvalue())
            self.assertIn("accuracy failed: chain", stream.getvalue())
            self.assertIn("MISSING", stream.getvalue())
            self.assertEqual(delta(0, 1), "n/a")


class ShmCompatibilityTest(unittest.TestCase):
    def test_selects_matching_wheel_allocator(self):
        package = "sgl_kernel_npu.sparsity_driven_kv_offload"
        for modern in (False, True):
            with self.subTest(modern=modern):
                module_name = package + (".fused_shm" if modern else ".ops")
                module = types.ModuleType(module_name)
                create, free = object(), object()
                if modern:
                    module.create_fused_shm_tensor, module.free_fused_shm = create, free
                else:
                    module.create_shm_tensor, module.free_shm = create, free
                spec = object() if modern else None
                with (
                    patch("importlib.util.find_spec", return_value=spec),
                    patch.dict(sys.modules, {module_name: module}),
                ):
                    self.assertEqual(_import_shm(), (create, free))


class WheelPreparationTest(unittest.TestCase):
    def test_private_prefix_preserves_the_supplied_payload_and_refuses_reuse(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            wheel = root / "kernel.whl"
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr("sgl_kernel_npu/__init__.py", "marker=123\n")
            prefix = root / "prefix"
            receipt = unpack(wheel, prefix)
            self.assertEqual(
                (prefix / "sgl_kernel_npu/__init__.py").read_text(), "marker=123\n"
            )
            self.assertEqual(receipt["prefix"], str(prefix.resolve()))
            with self.assertRaisesRegex(ValueError, "overwrite"):
                unpack(wheel, prefix)

    def test_wheel_paths_cannot_escape_the_private_prefix(self):
        for name in ("../outside", "/outside", "C:/outside", "a\\outside"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                wheel = root / "kernel.whl"
                with zipfile.ZipFile(wheel, "w") as archive:
                    archive.writestr("sgl_kernel_npu/__init__.py", "")
                    entry = zipfile.ZipInfo(name)
                    # ZipInfo normalizes Windows separators at construction.
                    entry.filename = name
                    archive.writestr(entry, "bad")
                with self.assertRaisesRegex(ValueError, "unsafe"):
                    unpack(wheel, root / "prefix")
                self.assertFalse((root / "prefix").exists())


class FixtureSpecTest(unittest.TestCase):
    def test_an_installed_kernel_cannot_silently_replace_the_selected_arm(self):
        with tempfile.TemporaryDirectory() as temporary:
            prefix = Path(temporary) / "selected"
            _require_kernel_prefix(str(prefix / "sgl_kernel_npu/__init__.py"), prefix)
            with self.assertRaisesRegex(RuntimeError, "outside --kernel-prefix"):
                _require_kernel_prefix(
                    str(Path(temporary) / "other/__init__.py"), prefix
                )

    def test_q1_q4_miss_distribution_and_fingerprint(self):
        hashes = set()
        for width in (1, 4):
            spec = FixtureSpec(width, 12, 8, 32, 8192, 9216, 9345)
            manifest = spec.manifest()
            self.assertEqual(manifest["miss_counts"], [512, 513] * 6)
            self.assertEqual(manifest["request_slots"], list(range(1, 13)))
            self.assertEqual(len(manifest["selected_prefix_hit_rates"]), 12)
            self.assertGreater(
                manifest["selected_prefix_hit_rates"][0],
                manifest["selected_prefix_hit_rates"][1],
            )
            self.assertEqual(manifest["positions"], list(range(9345 - width, 9345)))
            self.assertTrue(
                all(count <= 2048 for count in manifest["route_prefix_counts"])
            )
            hashes.add(spec.fingerprint())
        self.assertEqual(len(hashes), 2)

    def test_invalid_capacity_is_rejected(self):
        with self.assertRaises(ValueError):
            FixtureSpec(4, 12, 8, 32, 4096, 8320, 8449).validate()

    def test_full_resident_pool_cannot_hide_more_tokens_than_the_prefix_gap(self):
        with self.assertRaisesRegex(ValueError, "prefix cannot fill"):
            FixtureSpec(4, 12, 8, 32, 8192, 8320, 8449).validate()

    def test_miss_profile_parser_supports_threshold_sweeps(self):
        self.assertEqual(parse_miss_counts("512"), (512,))
        self.assertEqual(parse_miss_counts("513"), (513,))
        self.assertEqual(parse_miss_counts("512,513"), (512, 513))
        with self.assertRaises(ValueError):
            parse_miss_counts("")


class ContractTest(unittest.TestCase):
    def test_complete_result_contract(self):
        stats = timing_stats([1.0, 2.0, 3.0])
        result = {
            "schema_version": 1,
            "arm": {"name": "native", "kind": "native"},
            "case": {"mode": "eager"},
            "timer": "npu_event",
            "profiling": {
                "enabled": False,
                "iterations_per_component": 3,
                "scope": "post-timing",
                "components": {},
            },
            "timings": {
                "indexer_or_lim": stats,
                "attention_or_copysfa": stats,
                "chain": stats,
            },
            "fixture": {
                "spec_sha256": "a",
                "tensor_sha256": "b",
                "miss_counts": [512, 513],
            },
            "provenance": {
                "framework_prefix": "/framework",
                "kernel_prefix": "/kernel",
                "schemas": {},
                "libraries": [],
            },
            "accuracy": {"standalone": {}, "chain": {}, "canonical_ids": {}},
            "diagnostics": {
                "fixed_attention_plan": {
                    "topk_misses": [1],
                    "fallback_any": False,
                },
                "live_chain_lim": None,
            },
            "status": "passed",
        }
        validate_result(result)

    def test_direct_parity_uses_strict_legacy_defaults(self):
        passed = _parity(
            array.array("f", [1.006]),
            array.array("f", [1.0]),
            atol=0.002,
            rtol=0.005,
            max_relative_l2=0.08,
        )
        failed = _parity(
            array.array("f", [1.008]),
            array.array("f", [1.0]),
            atol=0.002,
            rtol=0.005,
            max_relative_l2=0.08,
        )
        self.assertTrue(passed["passed"])
        self.assertFalse(failed["passed"])


class RunnerTest(unittest.TestCase):
    def test_three_arms_are_required_and_dry_run_is_explicit_device(self):
        arms = [
            {
                "name": "native",
                "kind": "native",
                "python": "python3",
                "framework_prefix": "/native/framework",
                "kernel_prefix": "/native/kernel",
                "libraries": [],
            },
            {
                "name": "pre_migration",
                "kind": "lim_copysfa",
                "python": "python3",
                "framework_prefix": "/pre/framework",
                "kernel_prefix": "/pre/kernel",
                "libraries": ["/pre/lim.so"],
            },
            {
                "name": "post_migration",
                "kind": "lim_copysfa",
                "python": "python3",
                "framework_prefix": "/post/framework",
                "kernel_prefix": "/post/kernel",
                "libraries": ["/post/lim.so"],
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "arms.json"
            config.write_text(
                json.dumps({"schema_version": 1, "arms": arms}), encoding="utf-8"
            )
            self.assertEqual(len(load_config(config)), 3)
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "run_matrix.py"),
                    "--arm-config",
                    str(config),
                    "--output-root",
                    str(Path(directory) / "output"),
                    "--device",
                    "0",
                    "--batch",
                    "12",
                    "--heads",
                    "8",
                    "--index-heads",
                    "32",
                    "--dry-run",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            commands = json.loads(completed.stdout)["commands"]
            self.assertEqual(len(commands), 12)
            self.assertTrue(all("--profile" not in command for command in commands))
            self.assertTrue(
                all(
                    command[command.index("--device") + 1] == "0"
                    for command in commands
                )
            )
            self.assertTrue(
                all(
                    command[command.index("--golden-atol") + 1] == "0.005"
                    and command[command.index("--golden-rtol") + 1] == "0.02"
                    for command in commands
                )
            )
            self.assertTrue(
                all(
                    command[command.index("--miss-counts") + 1] == "512,513"
                    for command in commands
                )
            )
            self.assertNotIn("ASCEND_RT_VISIBLE_DEVICES", completed.stdout)

            profiled = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "run_matrix.py"),
                    "--arm-config",
                    str(config),
                    "--output-root",
                    str(Path(directory) / "profiled-output"),
                    "--device",
                    "0",
                    "--batch",
                    "12",
                    "--heads",
                    "8",
                    "--index-heads",
                    "32",
                    "--profile",
                    "--profile-iterations",
                    "3",
                    "--dry-run",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            profile_commands = json.loads(profiled.stdout)["commands"]
            self.assertTrue(all("--profile" in command for command in profile_commands))
            self.assertTrue(
                all(
                    command[command.index("--profile-iterations") + 1] == "3"
                    for command in profile_commands
                )
            )


if __name__ == "__main__":
    unittest.main()
