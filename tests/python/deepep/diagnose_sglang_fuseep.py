"""Run SGLang #40516 with per-layer, same-input INT8 MoE comparisons.

This diagnostic keeps the original test assertions and fused model outputs.
It instruments a private source copy; no installed package is modified.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def instrument(text):
    changes = {
        "    buf = _get_fuseep_buffer(layer)": (
            "    audit_input = hidden_states\n    buf = _get_fuseep_buffer(layer)"
        ),
        "    hidden_states, _ = buf.fused_deep_moe(": (
            "    hidden_states, audit_counts = buf.fused_deep_moe("
        ),
        "    return hidden_states\n": (
            "    from _fuseep_real_input_reference import audit\n"
            "    audit(layer, buf, audit_input, topk_output, hidden_states, audit_counts)\n"
            "    return hidden_states\n"
        ),
    }
    for old, new in changes.items():
        if text.count(old) != 1:
            raise ValueError(f"Unexpected FuseEP source: expected one {old!r}")
        text = text.replace(old, new)
    compile(text, "instrumented_fuseep.py", "exec")
    return text


def summarize(output, returncode):
    rows = []
    for path in sorted((output / "layers").glob("rank*.jsonl")):
        rows.extend(json.loads(line) for line in path.read_text().splitlines())
    result = {"original_test_exit_code": returncode, "layer_comparisons": len(rows)}
    for variant in ("int32_dequant_swiglu", "bf16_gmm_swiglu"):
        values = [row for row in rows if not row[variant].get("empty")]
        result[variant] = {
            "all_finite": bool(rows)
            and all(
                row[variant]["reference_finite"] and row[variant]["candidate_finite"]
                for row in rows
            ),
            "all_recv_counts_equal": bool(rows)
            and all(row[variant]["recv_counts_equal"] for row in rows),
            "largest_mean_errors": sorted(
                values, key=lambda row: row[variant]["mean_abs"], reverse=True
            )[:8],
        }
    (output / "layer-summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if not isinstance(v, dict)}))
    print(f"Results: {output / 'layer-summary.json'}")
    print("The original full-model thresholds and exit status are unchanged.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sglang-root", type=Path, required=True)
    parser.add_argument(
        "--model", type=Path, help="Override the original test's model path"
    )
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-tokens", type=int, default=4)
    args = parser.parse_args()
    root, output = args.sglang_root.resolve(), args.output_dir.resolve()
    test_path = (
        root / "test/registered/npu/basic_function/parameter/test_npu_fuseep_mode.py"
    )
    module_path = Path("sglang/srt/hardware_backend/npu/moe/fuseep.py")
    if not test_path.is_file() or not (root / "python" / module_path).is_file():
        parser.error(
            "--sglang-root must contain the SGLang #40516 test and FuseEP module"
        )
    devices = args.devices.split(",")
    if (
        len(devices) < 2
        or len(set(devices)) != len(devices)
        or not all(v.isdigit() for v in devices)
    ):
        parser.error("--devices requires at least two distinct numeric device IDs")
    if args.min_tokens < 1 or args.min_tokens > 128:
        parser.error("--min-tokens must be in [1, 128]")
    if output.exists() and any(output.iterdir()):
        parser.error("--output-dir must be new or empty; earlier results are preserved")
    instrumented = instrument((root / "python" / module_path).read_text())
    output.mkdir(parents=True, exist_ok=True)
    private_python = output / "python"
    shutil.copytree(
        root / "python",
        private_python,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    (private_python / module_path).write_text(instrumented)
    helper = Path(__file__).with_name("_fuseep_real_input_reference.py")
    shutil.copy2(helper, private_python / helper.name)
    env = {
        **os.environ,
        "PYTHONPATH": str(private_python)
        + os.pathsep
        + os.environ.get("PYTHONPATH", ""),
        "ASCEND_RT_VISIBLE_DEVICES": args.devices,
        "SGLANG_TEST_TP_SIZE": str(len(devices)),
        "SGLANG_TEST_FUSEEP_MODES": "2",
        "SGLANG_TEST_LOG_DIR": str(output),
        "SGLANG_FUSEEP_AUDIT_DIR": str(output / "layers"),
        "SGLANG_FUSEEP_AUDIT_MIN_TOKENS": str(args.min_tokens),
        "PYTHONUNBUFFERED": "1",
    }
    if args.model is not None:
        env["SGLANG_TEST_MODEL_PATH"] = str(args.model.resolve())
    (output / "diagnostic-config.json").write_text(
        json.dumps(
            {
                "sglang_root": str(root),
                "model_override": env.get("SGLANG_TEST_MODEL_PATH"),
                "devices": args.devices,
                "min_tokens": args.min_tokens,
                "fused_outputs_preserved": True,
            },
            indent=2,
        )
    )
    with (output / "test-console.log").open("w") as log:
        process = subprocess.run(
            [sys.executable, "-u", str(test_path), "-v"],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    summarize(output, process.returncode)
    return process.returncode


if __name__ == "__main__":
    sys.exit(main())
