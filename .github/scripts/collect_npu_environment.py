"""Collect comparable NPU toolchain and runtime fingerprints without kernel builds."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path


def command(args):
    try:
        result = subprocess.run(
            args, capture_output=True, text=True, timeout=30, check=False
        )
        output = result.stdout + result.stderr
        output = re.sub(
            r"(?im)^(.*serial\s*number\s*:\s*)(\S+)",
            lambda match: match[1]
            + "sha256:"
            + hashlib.sha256(match[2].encode()).hexdigest()[:12],
            output,
        )
        return {"command": args, "returncode": result.returncode, "output": output}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"command": args, "error": str(error)}


def fingerprint(path):
    path = Path(path).resolve()
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "size": path.stat().st_size, "sha256": digest}


def collect(skip_runtime):
    sdk = Path(os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/cann")).resolve()
    report = {
        "host": {
            "architecture": platform.machine(),
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "libc": platform.libc_ver(),
            "os_release": (
                Path("/etc/os-release").read_text()
                if Path("/etc/os-release").exists()
                else None
            ),
        },
        "sdk": str(sdk),
        "git": command(["git", "rev-parse", "HEAD"]),
        "ci": {
            key: os.environ.get(key)
            for key in ("GITHUB_RUN_ID", "GITHUB_SHA", "RUNNER_NAME", "RUNNER_ARCH")
        },
        "packages": dict(
            sorted(
                (dist.metadata["Name"], dist.version)
                for dist in importlib.metadata.distributions()
                if dist.metadata["Name"]
            )
        ),
        "environment": {
            key: value
            for key, value in sorted(os.environ.items())
            if key.startswith(("TRITON_", "PYTORCH_NPU_"))
            or key
            in {
                "ASCEND_HOME_PATH",
                "ASCEND_CANN_PACKAGE_PATH",
                "ASCEND_OPP_PATH",
                "ASCEND_CUSTOM_OPP_PATH",
                "ASCEND_RT_VISIBLE_DEVICES",
                "ASCEND_VISIBLE_DEVICES",
                "ASCEND_DEVICE_ID",
                "ACL_OP_SELECT_IMPL_MODE",
                "TASK_QUEUE_ENABLE",
                "ASCEND_LAUNCH_BLOCKING",
                "TORCH_DEVICE_BACKEND_AUTOLOAD",
                "LD_LIBRARY_PATH",
                "LD_PRELOAD",
                "PYTHONPATH",
                "PATH",
            }
        },
        "cann_packages": command(["bash", str(sdk / "query_pkg_version.sh")]),
        "host_tools": {
            name: command([name, "--version"]) for name in ("gcc", "cmake", "ldd")
        },
        "npu_smi": [],
        "compilers": {},
        "cann_libraries": {},
        "files": {},
        "errors": [],
    }
    for args in ([], ["-l"], ["-m"]):
        report["npu_smi"].append(command(["npu-smi", "info", *args]))
    cards = re.findall(
        r"(?m)^\s*NPU ID\s*:\s*(\d+)", report["npu_smi"][1].get("output", "")
    )
    for card in sorted(set(cards) | {"0"}, key=int):
        report["npu_smi"].append(
            command(["npu-smi", "info", "-t", "board", "-i", card])
        )
    for component in ("driver", "firmware"):
        path = Path("/usr/local/Ascend") / component / "version.info"
        if path.is_file():
            report[component] = path.read_text()
    try:
        installed_kernel = importlib.metadata.distribution("sgl-kernel-npu")
        report["installed_kernel_wheel"] = installed_kernel.read_text("WHEEL")
    except importlib.metadata.PackageNotFoundError:
        report["installed_kernel_wheel"] = None

    compiler_paths = [
        sdk / "bin/bisheng",
        sdk / "tools/bisheng_compiler/bin/bisheng",
        sdk / "tools/ccec_compiler/bin/bisheng",
    ]
    if compiler := shutil.which("bisheng"):
        compiler_paths.append(Path(compiler))
    for path in compiler_paths:
        if path.is_file() and str(path.resolve()) not in report["compilers"]:
            report["compilers"][str(path.resolve())] = {
                **fingerprint(path),
                "version": command([str(path), "--version"]),
            }

    library_dirs = {
        sdk / "lib64",
        sdk / f"{platform.machine()}-linux/lib64",
        sdk / "opp/built-in/op_api/lib",
        *(Path(path) for path in os.environ.get("LD_LIBRARY_PATH", "").split(":")),
    }
    for directory in sorted(library_dirs):
        if not directory.is_absolute() or not directory.is_dir():
            continue
        for pattern in ("libascendcl.so*", "libnnopbase.so*", "libopapi*.so*"):
            for path in sorted(directory.glob(pattern)):
                key = str(path.resolve())
                if path.is_file() and key not in report["cann_libraries"]:
                    report["cann_libraries"][key] = fingerprint(path)

    for source in (
        "scripts/npu_ci_install_dependency.sh",
        "scripts/prepare_kernel_tests.sh",
        "tests/python/operators/attention/kda/test_chunk_kda_fwd.py",
    ):
        if Path(source).is_file():
            report["files"][source] = fingerprint(source)

    if skip_runtime:
        return report

    try:
        import torch
        import torch_npu
        import triton
        from triton.runtime import driver

        torch.npu.set_device(0)
        properties = torch_npu.npu.get_device_properties(0)
        report["runtime"] = {
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
            "triton": triton.__version__,
            "triton_file": triton.__file__,
            "soc": torch.npu.get_device_name(0),
            "properties": str(properties),
            "target": str(driver.active.get_current_target()),
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        }
        for module, paths in (
            (torch, ("__init__.py", "lib/libtorch_cpu.so")),
            (torch_npu, ("__init__.py", "lib/libtorch_npu.so")),
            (
                triton,
                (
                    "__init__.py",
                    "language/__init__.py",
                    "backends/ascend/compiler.py",
                    "backends/ascend/utils.py",
                    "backends/ascend/bishengir/bin/bishengir-compile",
                ),
            ),
        ):
            root = Path(module.__file__).parent
            for relative in paths:
                path = root / relative
                if path.is_file():
                    report["files"][f"{module.__name__}/{relative}"] = fingerprint(path)
                    if path.name == "bishengir-compile":
                        report["compilers"][str(path.resolve())] = {
                            **fingerprint(path),
                            "version": command([str(path), "--version"]),
                        }
        maps = Path("/proc/self/maps")
        if maps.is_file():
            report["runtime"]["loaded_libraries"] = {}
            for line in maps.read_text().splitlines():
                fields = line.split(maxsplit=5)
                if len(fields) != 6 or not fields[5].startswith("/"):
                    continue
                path = Path(fields[5])
                if (
                    path.name.startswith(
                        (
                            "libascendcl.so",
                            "libnnopbase.so",
                            "libopapi.so",
                            "libascend_hal.so",
                        )
                    )
                    and path.is_file()
                ):
                    dest = report["runtime"]["loaded_libraries"]
                    if str(path.resolve()) not in dest:
                        dest[str(path.resolve())] = fingerprint(path)
    except Exception as error:
        report["errors"].append(f"{type(error).__name__}: {error}")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-runtime", action="store_true")
    args = parser.parse_args()
    report = collect(args.skip_runtime)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    args.output.write_text(rendered)
    print(rendered, end="")
    return int(bool(report["errors"]))


if __name__ == "__main__":
    raise SystemExit(main())
