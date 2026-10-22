"""Verify CANN checkpoints in order, stopping at the first failure; never launch a Kernel."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys


def run(command, log_path):
    print(f"\n$ {shlex.join(command)}", flush=True)
    with log_path.open("w") as log:
        log.write(shlex.join(command) + "\n")
        log.flush()
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        result = process.wait()
    if result:
        raise RuntimeError(f"Command failed ({result}); stop here and return {log_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compute-unit", required=True, help="Reviewed CANN compute unit, not a guessed chip SKU")
    parser.add_argument("--confirmed-socs", nargs="+", required=True, help="Reviewed compute units for the operator declaration")
    parser.add_argument("--build-dir", type=Path, help="Absolute path to a dedicated build directory")
    parser.add_argument("--through", choices=("codegen", "tiling", "kernel"), default="codegen",
                        help="Last checkpoint to execute; preceding checkpoints must pass first")
    args = parser.parse_args()

    if platform.system() != "Linux":
        parser.error("Run SDK generation on Linux; portable CMake checks can run separately on macOS")
    if not os.environ.get("ASCEND_HOME_PATH"):
        parser.error("Initialize the intended SDK environment before running this check")
    sdk = Path(os.environ["ASCEND_HOME_PATH"]).resolve()
    if not sdk.is_dir():
        parser.error(f"SDK directory does not exist: {sdk}")

    root = Path(__file__).resolve().parents[1]
    build = args.build_dir or root / "build" / "stage2-codegen"
    if not build.is_absolute():
        parser.error("--build-dir must be an absolute path")
    build = build.resolve()
    if build == root or build in root.parents or build.is_relative_to(root / "csrc") or build.is_relative_to(root / "tests"):
        parser.error("Use a separate build directory, not a source directory")

    configuration = {"source": str(root), "sdk": str(sdk), "compute_unit": args.compute_unit,
                     "confirmed_socs": args.confirmed_socs}
    stamp = build / "checkpoint-config.json"
    if build.exists() and any(build.iterdir()) and not stamp.is_file():
        parser.error("Build directory already contains unrelated files; choose a new --build-dir")
    if stamp.is_file() and json.loads(stamp.read_text()) != configuration:
        parser.error("This build directory belongs to a different configuration; choose a new --build-dir")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    logs = build / "logs" / run_id
    # Fresh SDK directories keep these checks independent of unverified SDK incremental-copy rules.
    run_build = build / "runs" / run_id
    logs.mkdir(parents=True, exist_ok=True)
    stamp.write_text(json.dumps(configuration, indent=2) + "\n")
    print(json.dumps(configuration, indent=2), flush=True)

    run(["cmake", "--version"], logs / "00-cmake-version.log")
    run(["cmake", "-S", str(root / "tests/cmake"), "-B", str(build / "rule-tests")],
        logs / "01-rules-configure.log")
    run(["ctest", "--test-dir", str(build / "rule-tests"), "--output-on-failure"],
        logs / "02-rules-test.log")
    portable = run_build / "portable"
    run(["cmake", "-S", str(root / "tests/csrc"), "-B", str(portable)], logs / "03-portable-configure.log")
    run(["cmake", "--build", str(portable)], logs / "04-portable-build.log")
    run(["ctest", "--test-dir", str(portable), "--output-on-failure"], logs / "05-portable-test.log")
    stages = [("codegen", "sgl_bitmask_api_check"), ("tiling", "sgl_cann_tiling"),
              ("kernel", "sgl_bitmask_kernel_check")]
    for stage, target in stages:
        stage_build = run_build / stage
        run(["cmake", "-S", str(root / "csrc"), "-B", str(stage_build),
             "-DSGL_CANN_DECLARATIONS_ONLY=OFF", f"-DSGL_CANN_SOC={args.compute_unit}",
             f"-DSGL_BITMASK_CONFIRMED_SOCS={';'.join(args.confirmed_socs)}",
             f"-DSGL_CANN_BUILD_STAGE={stage.upper()}", "-DSGL_CANN_OPERATORS=apply_token_bitmask"],
            logs / f"{stage}-configure.log")
        run(["cmake", "--build", str(stage_build), "--target", target, "--verbose"],
            logs / f"{stage}-build.log")
        if stage == "kernel":
            binaries = sorted(p for p in (stage_build / "sgl_cann_kernels").rglob("*.o") if p.stat().st_size)
            metadata = sorted((stage_build / "sgl_cann_kernels").rglob("*.json"))
            if not binaries or not metadata:
                raise RuntimeError(f"Kernel target returned success without binary / metadata evidence in {stage_build}")
            (logs / "kernel-artifacts.json").write_text(json.dumps(
                {"binaries": [str(p) for p in binaries], "metadata": [str(p) for p in metadata]}, indent=2) + "\n")
        print(f"\nPASS: {stage} checkpoint. Build: {stage_build}", flush=True)
        if stage == args.through:
            break
    print(f"\nPASS: checkpoints through {args.through}. Logs: {logs}")
    print("No Kernel was launched; numerical correctness, in-place execution, package installation and Torch remain unverified.")
    print(f"Return the logs and generated ACLNN headers from {run_build / 'codegen/autogen'}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        sys.exit(1)
