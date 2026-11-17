"""CPU tests for diagnostic oracles; these do not claim A5 device coverage."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fixture import _packed_rows  # noqa: E402
from tensor_trace import TensorTrace, compare, gather_logical, summarize  # noqa: E402
from trace_alignment import (  # noqa: E402
    build_inputs,
    parse_args,
    short_golden,
    validate_ids,
)


class TensorTraceTest(unittest.TestCase):
    def test_short_defaults_and_causal_padding(self):
        with patch.object(
            sys,
            "argv",
            [
                "trace_short_alignment.py",
                "--framework-prefix",
                "/unused",
                "--output",
                "/unused",
            ],
        ):
            args = parse_args(
                {"context": 16, "batch": 1, "heads": 128}, short_only=True
            )
        self.assertEqual(
            (args.context, args.batch, args.heads, args.device), (16, 1, 128, 0)
        )
        for width in (1, 4):
            args.query_width = width
            inputs = build_inputs(torch, args)
            ids = torch.full((width, 2048), -1, dtype=torch.int32)
            for row, position in enumerate(inputs["positions"].tolist()):
                ids[row, : position + 1] = torch.arange(position + 1)
            with tempfile.TemporaryDirectory() as directory:
                trace = TensorTrace(torch, Path(directory), {})
                trace.begin("short")
                validate_ids(torch, trace, ids, args)
            self.assertEqual(
                inputs["locations"].tolist(), list(range(128 + 16 - width, 128 + 16))
            )

    def test_short_golden_excludes_future_kv(self):
        args = SimpleNamespace(
            batch=1, query_width=4, heads=8, context=16, capacity=8192
        )
        inputs = build_inputs(torch, args)
        inputs["packed"] = torch.cat(
            (torch.zeros(128, 656, dtype=torch.int8), _packed_rows(torch, 128))
        ).view(-1, 128, 1, 656)
        before = short_golden(torch, inputs, args)
        flat = inputs["packed"].view(-1, 656)
        flat[128 + 15].copy_(flat[128 + 30])
        after = short_golden(torch, inputs, args)
        self.assertTrue(torch.equal(before[:3], after[:3]))
        self.assertFalse(torch.allclose(before[3], after[3]))

    def test_resolves_request_specific_pages_and_ignores_padding(self):
        packed = (
            torch.arange(6 * 128)
            .remainder(127)
            .to(torch.int8)[:, None]
            .expand(-1, 656)
            .contiguous()
        )
        table = torch.tensor([[3, 1], [5, 2]], dtype=torch.int32)
        ids = torch.tensor([[0, 129, -1], [2, 130, -1], [1, 128, -1], [3, 131, -1]])
        result = gather_logical(torch, packed, table, ids, 2)
        physical = [[384, 129], [386, 130], [641, 256], [643, 259]]
        self.assertTrue(torch.equal(result[:, :2], packed[torch.tensor(physical)]))
        self.assertTrue(
            torch.equal(result[:, 2], torch.zeros(4, 656, dtype=torch.int8))
        )
        with self.assertRaisesRegex(ValueError, "outside packed"):
            gather_logical(torch, packed, table + 10, ids, 2)

    def test_bad_dtype_nonfinite_and_small_relative_error_are_distinct(self):
        self.assertFalse(compare(torch, torch.ones(2), torch.ones(2).int())["passed"])
        self.assertFalse(
            compare(torch, torch.tensor([float("nan")]), torch.zeros(1))["passed"]
        )
        result = compare(torch, torch.tensor([1.001]), torch.ones(1), exact=False)
        self.assertTrue(result["passed"])
        self.assertFalse(compare(torch, torch.ones(2), torch.ones(3))["passed"])

    def test_first_mismatch_partial_run_and_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = TensorTrace(torch, root, {})
            trace.begin("cold")
            trace.check("topk", torch.ones(2), torch.ones(2))
            trace.check("host", torch.zeros(2), torch.ones(2))
            trace.check("output", torch.zeros(2), torch.ones(2))
            trace.flush()
            report = json.loads((root / "report.json").read_text())
            self.assertEqual(
                report["first_mismatch"], {"phase": "cold", "check": "host"}
            )
            self.assertFalse(report["passed"])
            summary = summarize(report)
            self.assertIn("First mismatch: cold / host", summary)
            self.assertIn("index=[0]", summary)
            self.assertLess(len(summary.splitlines()), 8)
            self.assertIn(
                "host.actual", torch.load(root / "cold.pt", weights_only=True)
            )
            trace.report["complete"] = True
            trace.write()
            self.assertFalse(trace.report["passed"])

    def test_success_requires_completed_run(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = TensorTrace(torch, Path(directory), {})
            trace.begin("cold")
            trace.check("kv", torch.ones(2), torch.ones(2))
            self.assertFalse(trace.report["passed"])
            trace.report["complete"] = True
            trace.write()
            self.assertTrue(trace.report["passed"])

    def test_input_mapping_and_bf16_topk_boundaries(self):
        for width in (1, 4):
            args = SimpleNamespace(
                batch=2, query_width=width, heads=8, context=9344, capacity=8192
            )
            inputs = build_inputs(torch, args)
            self.assertTrue((inputs["source_table"] > 0).all())
            self.assertEqual(
                inputs["source_table"].unique().numel(), inputs["source_table"].numel()
            )
            self.assertEqual(inputs["locations"].unique().numel(), 2 * width)
            scores = (
                (inputs["key_scale"].flatten()[128 : 128 + 9344] * 128)
                .bfloat16()
                .float()
            )
            for route in range(width):
                limit = 9344 - width + route + 1
                chosen = torch.topk(scores[:limit], 2048).indices.sort().values
                self.assertTrue(torch.equal(chosen, torch.arange(limit - 2048, limit)))

    def test_invalid_noncausal_and_duplicate_selection_stop_before_native(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = TensorTrace(torch, Path(directory), {})
            trace.begin("cold")
            args = SimpleNamespace(batch=1, query_width=1, context=2048)
            ids = torch.arange(2048).view(1, -1)
            validate_ids(torch, trace, ids, args)
            bad = ids.clone()
            bad[0, 0] = 2048
            with self.assertRaisesRegex(ValueError, "noncausal"):
                validate_ids(torch, trace, bad, args)
            bad[0, 0] = 1
            with self.assertRaisesRegex(ValueError, "duplicate"):
                validate_ids(torch, trace, bad, args)

    def test_native_and_python_output_go_to_log_only(self):
        script = """
import os
from pathlib import Path
from trace_alignment import capture_log
with capture_log(Path('run.log')):
    print('python detail')
    os.write(1, b'native detail\\n')
    os.write(2, b'native warning\\n')
print('parsed result only')
"""
        with tempfile.TemporaryDirectory() as directory:
            environment = dict(__import__("os").environ)
            environment["PYTHONPATH"] = str(Path(__file__).resolve().parent)
            result = subprocess.run(
                [sys.executable, "-c", script],
                cwd=directory,
                env=environment,
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertEqual(result.stdout, "parsed result only\n")
            self.assertEqual(result.stderr, "")
            log = (Path(directory) / "run.log").read_text()
            for line in ("python detail", "native detail", "native warning"):
                self.assertIn(line, log)


if __name__ == "__main__":
    unittest.main()
