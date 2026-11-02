"""Build only mlp_lightning_indexer for A5 and run its eager accuracy tests."""

import argparse
import datetime
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import traceback
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OPERATOR = "mlp_lightning_indexer"
TEST = "tests/python/operators/indexer/test_mlp_lightning_indexer.py"

PREFLIGHT = r"""
import importlib
import json
import platform
import sys
import torch
import torch_npu

for name in ("pytest", "numpy", "setuptools", "wheel", "packaging"):
    importlib.import_module(name)
device = int(sys.argv[1])
if not torch.npu.is_available() or not 0 <= device < torch.npu.device_count():
    raise RuntimeError("Selected NPU is unavailable")
torch.npu.set_device(device)
name = torch.npu.get_device_name(device)
if "950" not in name:
    raise RuntimeError("This runner requires Ascend 950; got " + name)
torch.ones(1, device="npu:%d" % device).sum().item()
torch.npu.synchronize()
print(json.dumps(dict(python=sys.executable, version=sys.version,
                     machine=platform.machine(), torch=torch.__version__,
                     torch_npu=torch_npu.__version__, device=device,
                     device_name=name,
                     visible_devices=torch.npu.device_count()), indent=2))
"""

VERIFY = r"""
import json
import sys
from pathlib import Path
import sgl_kernel_npu
import torch
from sgl_kernel_npu.operators.indexer import mlp_lightning_indexer
from sgl_kernel_npu.operators.utils.generation import get_hardware
from sgl_kernel_npu.operators.utils.registry import get_impl

package = Path(sgl_kernel_npu.__file__).resolve().parent
if Path(sys.prefix).resolve() not in package.parents:
    raise RuntimeError("Imported another installation: " + str(package))
manifest = json.loads((package / "_build_manifest.json").read_text())
if manifest["arch_version"] != "3510" or manifest["operators"] != ["mlp_lightning_indexer"]:
    raise RuntimeError("Incorrect installed build manifest: " + str(manifest))
torch.npu.set_device(int(sys.argv[1]))
if get_hardware().arch_id != 3510:
    raise RuntimeError("Detected architecture is not 3510")
for impl in ("auto", "ascendc", "torch_native"):
    get_impl(mlp_lightning_indexer, impl)
getattr(torch.ops.npu, "mlp_lightning_indexer")
print("PACKAGE:", package)
print(json.dumps(manifest, indent=2))
"""


def run(command, log, *, env, cwd=ROOT):
    command = [str(part) for part in command]
    print("$ " + shlex.join(command), flush=True)
    with log.open("w", encoding="utf-8") as stream:
        stream.write("$ " + shlex.join(command) + "\n")
        stream.flush()
        with subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        ) as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                stream.write(line)
            code = process.wait()
        if code:
            raise RuntimeError("Command failed (exit %d), see %s" % (code, log))


def validate_wheel(path):
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("sgl_kernel_npu/_build_manifest.json"))
    if manifest.get("arch_version") != "3510" or manifest.get("operators") != [
        OPERATOR
    ]:
        raise RuntimeError("Expected an arch3510 wheel containing only " + OPERATOR)
    return manifest


def validate_report(path):
    cases = list(ET.parse(path).getroot().iter("testcase"))
    if not cases:
        raise RuntimeError("No test cases executed; this is not an accuracy pass")
    bad = [
        case.attrib.get("name")
        for case in cases
        if any(case.find(tag) is not None for tag in ("skipped", "failure", "error"))
    ]
    if bad:
        raise RuntimeError("Skipped/failed tests are not accepted: " + repr(bad))
    return len(cases)


def record_sdk(sdk, destination):
    # Record the installed declarations, not just the online preview version.
    names = {"kernel_struct_mm.h", "kernel_struct_data_copy.h"}
    declarations = ("LoadDataRepeatParam", "dstStride", "DataCopyCO12DstParams")
    with destination.open("w", encoding="utf-8") as stream:
        for info in (
            sdk / "version.info",
            sdk / "compiler/version.info",
            sdk / "ascend_toolkit_install.info",
        ):
            if info.is_file():
                stream.write(
                    "\nVERSION: %s\n%s\n" % (info, info.read_text(errors="replace"))
                )
        found = set()
        for directory in (
            sdk / "asc/include",
            sdk / "include",
            sdk / "compiler/tikcpp/tikcfw",
        ):
            if not directory.is_dir():
                continue
            for header in sorted(directory.rglob("*.h")):
                if header.name not in names:
                    continue
                found.add(header.name)
                stream.write("\nHEADER: %s\n" % header)
                lines = header.read_text(errors="replace").splitlines()
                selected = set()
                for i, line in enumerate(lines):
                    if any(name in line for name in declarations):
                        selected.update(range(max(0, i - 2), min(len(lines), i + 12)))
                for i in sorted(selected):
                    stream.write("%d: %s\n" % (i + 1, lines[i]))
        if names - found:
            stream.write(
                "\nHeaders not located in probed SDK directories: %s\n"
                % sorted(names - found)
            )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0, help="logical visible NPU ID")
    parser.add_argument("--jobs", type=int, default=8, help="parallel build jobs")
    parser.add_argument(
        "--output-dir", type=Path, help="new directory for all run logs"
    )
    args = parser.parse_args(argv)
    if args.device < 0 or args.jobs < 1:
        parser.error("--device must be nonnegative and --jobs must be positive")

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    out = (args.output_dir or ROOT / "output/light_indexer_a5" / stamp).resolve()
    out.mkdir(parents=True, exist_ok=False)
    summary = dict(status="FAIL", stage="preflight", log_dir=str(out))
    env = os.environ.copy()
    # Prevent a source checkout, editable package, or pytest flags from silently
    # replacing this build, selecting only some cases, or enabling profiling.
    for key in ("PYTHONPATH", "PYTHONHOME", "PYTEST_ADDOPTS", "PYTEST_PLUGINS"):
        env.pop(key, None)
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    try:
        if platform.system() != "Linux":
            raise RuntimeError("Run on the A5 Linux host/container with torch_npu")
        sdk = Path(env.get("ASCEND_HOME_PATH", ""))
        if not env.get("ASCEND_HOME_PATH") or not sdk.is_dir():
            raise RuntimeError(
                "Source your CANN set_env.sh; ASCEND_HOME_PATH is missing"
            )
        summary["sdk"] = str(sdk.resolve())
        run(["git", "rev-parse", "HEAD"], out / "commit.log", env=env)
        run(["git", "diff", "HEAD"], out / "source-diff.log", env=env)
        run(["git", "status", "--short"], out / "source-status.log", env=env)
        run(
            [sys.executable, "-c", PREFLIGHT, args.device],
            out / "environment.log",
            env=env,
        )
        if shutil.which("npu-smi", path=env.get("PATH")):
            run(["npu-smi", "info"], out / "npu-smi.log", env=env)
        if shutil.which("bisheng", path=env.get("PATH")):
            run(["bisheng", "--version"], out / "compiler.log", env=env)
        run(["cmake", "--version"], out / "cmake.log", env=env)
        record_sdk(sdk, out / "sdk-headers.log")

        summary["stage"] = "venv"
        venv = out / "venv"
        run(
            [sys.executable, "-m", "venv", "--system-site-packages", venv],
            out / "venv.log",
            env=env,
        )
        python = venv / "bin/python"
        env["PATH"] = str(venv / "bin") + os.pathsep + env.get("PATH", "")
        env["VIRTUAL_ENV"] = str(venv)
        run(
            [python, "-c", PREFLIGHT, args.device],
            out / "venv-environment.log",
            env=env,
        )

        summary["stage"] = "build"
        run(
            [
                python,
                ROOT / "setup.py",
                "--make_clean",
                "--arch-version",
                "3510",
                "--ops",
                OPERATOR,
                "--artifact",
                "wheel",
                "--jobs",
                args.jobs,
            ],
            out / "build.log",
            env=env,
        )
        wheels = list((ROOT / "output").glob("sgl_kernel_npu-*.whl"))
        if len(wheels) != 1:
            raise RuntimeError("Expected exactly one newly built wheel in output/")
        wheel = out / wheels[0].name
        shutil.copy2(wheels[0], wheel)
        summary["manifest"] = validate_wheel(wheel)

        summary["stage"] = "install"
        run(
            [
                python,
                "-m",
                "pip",
                "--isolated",
                "--disable-pip-version-check",
                "install",
                "--no-index",
                "--no-deps",
                "--no-user",
                "--prefix",
                venv,
                "--force-reinstall",
                wheel,
            ],
            out / "install.log",
            env=env,
        )
        site = subprocess.check_output(
            [
                str(python),
                "-c",
                "import sysconfig; print(sysconfig.get_path('purelib'))",
            ],
            env=env,
            text=True,
        ).strip()
        vendor = Path(site) / "sgl_kernel_npu/_cann/vendors/sgl_kernel_npu"
        # Set before torch_npu initialization in the pytest process as well.
        env["ASCEND_CUSTOM_OPP_PATH"] = (
            str(vendor) + os.pathsep + env.get("ASCEND_CUSTOM_OPP_PATH", "")
        )
        run([python, "-c", VERIFY, args.device], out / "installed-build.log", env=env)

        summary["stage"] = "eager-accuracy"
        report = out / "pytest.xml"
        run(
            [
                python,
                "-m",
                "pytest",
                TEST,
                "--impl",
                "ascendc",
                "--impl-ref",
                "torch_native",
                "--skip-acc-check=false",
                "--enable-perf=false",
                "--npu-device",
                args.device,
                "--junitxml",
                report,
                "-v",
                "-s",
                "-x",
                "-rs",
            ],
            out / "pytest.log",
            env=env,
        )
        summary["passed"] = validate_report(report)
        summary["status"] = "PASS"
        print("PASS: %d eager accuracy cases; logs: %s" % (summary["passed"], out))
        return 0
    except Exception as error:
        summary["error"] = str(error)
        traceback.print_exc()
        print("FAIL at %s; logs: %s" % (summary["stage"], out), file=sys.stderr)
        return 1
    finally:
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    sys.exit(main())
