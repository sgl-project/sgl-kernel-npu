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

BOOTSTRAP = '''"""Select the instrumented source after Python startup hooks run."""
import os
import runpy
import sys
from pathlib import Path

private_python = Path(os.environ["SGLANG_FUSEEP_AUDIT_PYTHON"]).resolve()
sys.path.insert(0, str(private_python))

def main():
    import sglang
    source = Path(sglang.__file__).resolve()
    if not source.is_relative_to(private_python):
        raise RuntimeError(f"Diagnostic source was bypassed: {source}; expected {private_python}")
    role = sys.argv[1]
    print(f"[FUSEEP_DIAGNOSTIC_IMPORT] role={role} source={source} python={sys.executable}", flush=True)
    if role == "server":
        from sglang.cli.main import main as cli_main
        sys.argv = ["sglang", *sys.argv[2:]]
        cli_main()
    elif role == "server-module":
        sys.argv = ["sglang.launch_server", *sys.argv[2:]]
        runpy.run_module("sglang.launch_server", run_name="__main__")
    elif role == "test":
        from sglang.test import test_utils
        launch = test_utils._launch_server_process
        def launch_private(command, *args, **kwargs):
            if command[:2] == ["sglang", "serve"]:
                command = [sys.executable, "-u", __file__, "server", *command[1:]]
            elif len(command) >= 3 and command[1:3] == ["-m", "sglang.launch_server"]:
                command = [sys.executable, "-u", __file__, "server-module", *command[3:]]
            else:
                raise RuntimeError(f"Unrecognized diagnostic server command: {command[:3]}")
            return launch(command, *args, **kwargs)
        test_utils._launch_server_process = launch_private
        sys.argv = sys.argv[2:]
        runpy.run_path(sys.argv[0], run_name="__main__")
    else:
        raise ValueError(f"Unknown diagnostic role: {role}")

if __name__ == "__main__":
    main()
'''


def instrument(text):
    changes = {
        "    buf = _get_fuseep_buffer(layer)": (
            "    from _fuseep_real_input_reference import progress\n"
            "    progress('fused_enter', layer=layer.layer_id, source=__file__)\n"
            "    audit_input = hidden_states\n"
            "    buf = _get_fuseep_buffer(layer)\n"
            "    progress('before_fused', layer=layer.layer_id)"
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
    result["diagnostic_status"] = "comparisons_recorded" if rows else "no_comparisons"
    result["artifacts"] = {
        name: (output / name).exists()
        for name in (
            "test-console.log",
            "baseline.log",
            "baseline.json",
            "mode2.log",
            "mode2.json",
        )
    }
    result["last_progress"] = [
        json.loads(path.read_text())
        for path in sorted((output / "layers").glob("progress-rank*.json"))
    ]
    for variant in ("int32_dequant_swiglu", "bf16_gmm_swiglu"):
        values = [row for row in rows if not row[variant].get("empty")]
        result[variant] = {
            "all_finite": (
                all(
                    row[variant]["reference_finite"]
                    and row[variant]["candidate_finite"]
                    for row in rows
                )
                if rows
                else None
            ),
            "all_recv_counts_equal": (
                all(row[variant]["recv_counts_equal"] for row in rows) if rows else None
            ),
            "largest_mean_errors": sorted(
                values, key=lambda row: row[variant]["mean_abs"], reverse=True
            )[:8],
        }
    (output / "layer-summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if not isinstance(v, dict)}))
    print(f"Results: {output / 'layer-summary.json'}")
    print("The original full-model thresholds and exit status are unchanged.")
    if not rows:
        print(
            "No layer comparisons were collected; numerical checks are unknown, not failed."
        )
        print(f"Inspect {output / 'test-console.log'} and the available server logs.")
        print("Artifacts: " + json.dumps(result["artifacts"]))
        print("Last progress: " + json.dumps(result["last_progress"]))


def bypass_local_proxy(env):
    # SGLang's server-health checks use requests with environment proxies.
    # Keep remote proxy settings, but always contact local test servers directly.
    hosts = dict.fromkeys(
        host.strip()
        for value in (
            env.get("NO_PROXY", ""),
            env.get("no_proxy", ""),
            "127.0.0.1,localhost,::1",
        )
        for host in value.split(",")
        if host.strip()
    )
    env["NO_PROXY"] = env["no_proxy"] = ",".join(hosts)


def run_test(command, env, log_path):
    print(f"Starting test; output is also saved to {log_path}", flush=True)
    with log_path.open("w") as log, subprocess.Popen(
        command,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        bufsize=1,
    ) as process:
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
        return process.wait()


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
    bootstrap = private_python / "_fuseep_diagnostic_bootstrap.py"
    bootstrap.write_text(BOOTSTRAP)
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
        "SGLANG_FUSEEP_AUDIT_PYTHON": str(private_python),
        "SGLANG_FUSEEP_AUDIT_MIN_TOKENS": str(args.min_tokens),
        "PYTHONUNBUFFERED": "1",
    }
    if args.model is not None:
        env["SGLANG_TEST_MODEL_PATH"] = str(args.model.resolve())
    bypass_local_proxy(env)
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
    returncode = run_test(
        [sys.executable, "-u", str(bootstrap), "test", str(test_path), "-v"],
        env,
        output / "test-console.log",
    )
    summarize(output, returncode)
    return returncode


if __name__ == "__main__":
    sys.exit(main())
