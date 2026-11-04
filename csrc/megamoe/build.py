#!/usr/bin/env python3
"""Build the MegaMoe source with its pinned ops-transformer MC2 framework.

The resulting installer and Python wheel are standalone artifacts. This script
does not install them or modify the existing sgl-kernel-npu build.
"""

import argparse
import importlib.util
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPOSITORY = "https://gitcode.com/hanggongmao1/ops-transformer.git"
REVISION = "d4c5f65d93591f26895c126fc590f0d622440819"
VENDOR = "sgl_megamoe"
HERE = Path(__file__).resolve().parent
LOCAL_SOURCE = HERE / "source"
OVERLAYS = (
    "mc2/mega_moe/CMakeLists.txt",
    "mc2/mega_moe/op_api",
    "mc2/mega_moe/op_graph",
    "mc2/mega_moe/op_host",
    "mc2/mega_moe/op_kernel",
    "torch_extension/npu_ops_transformer/ops/mega_moe.py",
    "torch_extension/npu_ops_transformer/ops/comm_context.py",
    "torch_extension/npu_ops_transformer/ops/csrc/mega_moe.cpp",
    "torch_extension/npu_ops_transformer/ops/csrc/comm_context.cpp",
    "torch_extension/npu_ops_transformer/ops/graph_convert/graph_convert_mega_moe.py",
)


def run(command, *, cwd=None):
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def overlay_sources(destination):
    """Apply the checked-in files without removing any fetched files."""
    for relative in OVERLAYS:
        local = LOCAL_SOURCE / relative
        if not local.exists() or local.is_symlink():
            raise RuntimeError(f"Missing or symlinked MegaMoe source: {local}")
        target_root = destination / relative
        if local.is_file() and not target_root.is_file():
            raise RuntimeError(f"Pinned framework is missing {target_root}")
        if local.is_dir():
            if not target_root.is_dir() or target_root.is_symlink():
                raise RuntimeError(f"Pinned framework is missing {target_root}")
            local_files = {
                p.relative_to(local) for p in local.rglob("*") if p.is_file()
            }
            target_files = {
                p.relative_to(target_root)
                for p in target_root.rglob("*")
                if p.is_file()
            }
            if not target_files.issubset(local_files):
                raise RuntimeError(
                    f"Source file set differs from pinned framework: {relative}"
                )
        paths = sorted(local.rglob("*")) if local.is_dir() else [local]
        for path in paths:
            if path.is_symlink():
                raise RuntimeError(f"Symlinks are not supported: {path}")
            if path.is_file():
                target = destination / path.relative_to(LOCAL_SOURCE)
                for parent in target.parents:
                    if parent == destination:
                        break
                    if parent.is_symlink():
                        raise RuntimeError(f"Symlinked framework directory: {parent}")
                if target.is_symlink() or (target.exists() and not target.is_file()):
                    raise RuntimeError(f"Pinned framework is missing {target}")
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-j", "--jobs", type=int, default=8)
    parser.add_argument(
        "--work-dir", type=Path, default=HERE.parents[1] / "build" / "megamoe"
    )
    parser.add_argument(
        "--plan", action="store_true", help="show the recipe without building"
    )
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    package = [
        "bash",
        "build.sh",
        "--pkg",
        "--soc=ascend950",
        "--ops=mega_moe",
        f"--vendor_name={VENDOR}",
        f"-j{args.jobs}",
    ]
    wheel = [sys.executable, "-m", "build", "--wheel", "--no-isolation"]
    if args.plan:
        print(f"repository: {REPOSITORY}\nrevision: {REVISION}")
        print("overlay: " + str(LOCAL_SOURCE))
        print("package: " + " ".join(package))
        print("wheel (in torch_extension): " + " ".join(wheel))
        return
    if sys.platform != "linux":
        raise RuntimeError("MegaMoe's Ascend 950 build requires Linux and CANN")
    missing = [
        name
        for name in ("build", "setuptools", "wheel")
        if importlib.util.find_spec(name) is None
    ]
    if missing:
        raise RuntimeError("Install Python build dependencies: " + ", ".join(missing))
    args.work_dir.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="megamoe-", dir=args.work_dir))
    source = work / "ops-transformer"
    run(["git", "clone", "--no-checkout", REPOSITORY, str(source)])
    run(["git", "switch", "--detach", REVISION], cwd=source)
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source, text=True
    ).strip()
    if head != REVISION:
        raise RuntimeError("Unexpected ops-transformer checkout")
    overlay_sources(source)
    run(package, cwd=source)
    # The --ops branch in this revision does not build the Python wheel.
    run(wheel, cwd=source / "torch_extension")
    installers = list((source / "build_out").glob("*.run"))
    wheels = list((source / "torch_extension/dist").glob("*.whl"))
    if len(installers) != 1 or len(wheels) != 1:
        raise RuntimeError("Expected exactly one custom installer and Python wheel")
    print("Installer:", installers[0])
    print("Python wheel:", wheels[0])
    print("Build workspace:", work)


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        sys.exit(1)
