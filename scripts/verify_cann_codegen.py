"""Verify CANN build and optional private-install checkpoints; never launch a Kernel."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys


def run(command, log_path, *, env=None, cwd=None):
    print(f"\n$ {shlex.join(command)}", flush=True)
    with log_path.open("w") as log:
        log.write(shlex.join(command) + "\n")
        log.flush()
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, env=env, cwd=cwd)
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        result = process.wait()
    if result:
        raise RuntimeError(f"Command failed ({result}); stop here and return {log_path}")


def find_run_package(stage_build):
    packages = sorted((stage_build / "package-stage").glob("*.run"))
    if len(packages) != 1 or not packages[0].is_file() or not packages[0].stat().st_size:
        raise RuntimeError(f"Expected one nonempty .run package in {stage_build / 'package-stage'}; found {packages}")
    return packages[0]


def inspect_installation(install_dir, stage_build, compute_unit):
    vendor = install_dir / "vendors/sgl_kernel_npu"
    tbe = vendor / "op_impl/ai_core/tbe"
    libraries = [vendor / "op_api/lib/libcust_opapi.so"]
    for directory, name in ((tbe / "op_tiling", "libcust_opmaster_rt2.0.so"),
                            (vendor / "op_proto", "libcust_opsproto_rt2.0.so")):
        matches = sorted(directory.rglob(name))
        if len(matches) != 1:
            raise RuntimeError(f"Expected one installed {name} under {directory}; found {matches}")
        libraries.extend(matches)
    for path in [*libraries, vendor / "bin/set_env.bash"]:
        if not path.is_file() or not path.stat().st_size or not path.resolve().is_relative_to(install_dir.resolve()):
            raise RuntimeError(f"Missing, empty or external installed file: {path}")

    headers = sorted((stage_build / "autogen").glob("aclnn*.h"))
    if not headers:
        raise RuntimeError(f"No generated ACLNN headers in {stage_build / 'autogen'}")
    # CMake may rewrite shared-library RPATHs when packaging; load those below.
    pairs = [(path, vendor / "op_api/include" / path.name) for path in headers]
    binary_root = stage_build / "sgl_cann_kernels/binary"
    binaries = sorted((binary_root / compute_unit).rglob("*.o"))
    metadata = sorted((binary_root / compute_unit).rglob("*.json"))
    metadata += sorted((binary_root / "config" / compute_unit).rglob("*.json"))
    if not binaries or not metadata:
        raise RuntimeError(f"Missing compiled Kernel binaries or metadata for {compute_unit}")
    pairs += [(path, tbe / "kernel" / path.relative_to(binary_root)) for path in binaries + metadata]
    info = f"aic-{compute_unit}-ops-info.json"
    pairs.append((stage_build / "sgl_cann_kernels/tbe/op_info_cfg/ai_core" / compute_unit / info,
                  tbe / "config" / compute_unit / info))
    for source, installed in pairs:
        if not source.is_file() or not source.stat().st_size:
            raise RuntimeError(f"Missing or empty build artifact: {source}")
        if not installed.is_file() or not installed.resolve().is_relative_to(install_dir.resolve()):
            raise RuntimeError(f"Missing or external installed artifact: {installed}")
        if installed.read_bytes() != source.read_bytes():
            raise RuntimeError(f"Installed artifact differs from this build: {installed}")
    return vendor, libraries, [installed for _, installed in pairs]


def verify_package(stage_build, run_build, logs, compute_unit):
    run(["cmake", "--build", str(stage_build), "--target", "package", "--verbose"],
        logs / "package-build.log")
    package = find_run_package(stage_build)
    install_dir = run_build / "installed"
    # Always install to a fresh private directory, never the SDK's default OPP tree.
    install_dir.mkdir(exist_ok=False)
    run(["bash", str(package), "--quiet", f"--install-path={install_dir}"],
        logs / "package-install.log", cwd=install_dir)
    vendor, libraries, artifacts = inspect_installation(install_dir, stage_build, compute_unit)
    inventory = [f"sha256 {hashlib.sha256(package.read_bytes()).hexdigest()}  {package}",
                 f"vendor {vendor}", *[str(path) for path in libraries + artifacts]]
    (logs / "package-artifacts.txt").write_text("\n".join(inventory) + "\n")
    print("\n".join(inventory), flush=True)
    env = os.environ.copy()
    env["ASCEND_CUSTOM_OPP_PATH"] = str(vendor)
    env["LD_LIBRARY_PATH"] = str(vendor / "op_api/lib") + ":" + env.get("LD_LIBRARY_PATH", "")
    # A new Host process loads absolute installed paths and resolves both API symbols.
    # It does not call aclInit, choose a device, or execute either ACLNN function.
    loader = """import ctypes, os, sys
libraries = [ctypes.CDLL(path, mode=os.RTLD_NOW) for path in sys.argv[1:]]
for symbol in ('aclnnSglApplyTokenBitMaskGetWorkspaceSize', 'aclnnSglApplyTokenBitMask'):
    getattr(libraries[0], symbol)
    print('Resolved: ' + symbol)
for path in sys.argv[1:]:
    print('Loaded installed library: ' + path)
"""
    run([sys.executable, "-c", loader, *map(str, libraries)],
        logs / "package-load.log", env=env, cwd=install_dir)
    print(f"\nPASS: package checkpoint. Installed vendor: {vendor}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compute-unit", required=True, help="CANN compute unit to build in this run, not a chip SKU")
    parser.add_argument("--confirmed-socs", nargs="+", help="Deprecated; ignored. Support scope comes from the operator declaration")
    parser.add_argument("--build-dir", type=Path, help="Absolute path to a dedicated build directory")
    parser.add_argument("--through", choices=("codegen", "tiling", "kernel", "package"), default="codegen",
                        help="Last checkpoint; package also installs to a fresh private build subdirectory")
    args = parser.parse_args()
    if args.confirmed_socs is not None:
        print("DEPRECATED: --confirmed-socs is ignored; SUPPORTED_SOCS comes from the operator declaration.",
              file=sys.stderr)

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

    configuration = {"source": str(root), "sdk": str(sdk), "compute_unit": args.compute_unit}
    stamp = build / "checkpoint-config.json"
    if build.exists() and any(build.iterdir()) and not stamp.is_file():
        parser.error("Build directory already contains unrelated files; choose a new --build-dir")
    if stamp.is_file():
        previous = json.loads(stamp.read_text())
        previous.pop("confirmed_socs", None)  # Older records included the now-retired declaration override.
        if previous != configuration:
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
    if args.through == "package":
        verify_package(run_build / "kernel", run_build, logs, args.compute_unit)
    print(f"\nPASS: checkpoints through {args.through}. Logs: {logs}")
    print("No Kernel was launched; numerical correctness, in-place execution, performance and Torch remain unverified.")
    if args.through != "package":
        print("Package creation and installation remain unverified.")
    print(f"Return the logs and generated ACLNN headers from {run_build / 'codegen/autogen'}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        sys.exit(1)
