#!/usr/bin/env python3
"""Repackage the metadata size helper using existing package binaries.

Run from the repository root. Native binaries and bundled third-party packages
are retained; the only source update allowed is get_transformer_metadata_size.
"""

import argparse
import ast
import importlib.metadata
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def without_size_helper(text):
    tree = ast.parse(text)
    tree.body = [
        node
        for node in tree.body
        if not (
            isinstance(node, ast.FunctionDef)
            and node.name == "get_transformer_metadata_size"
        )
    ]
    return ast.dump(tree, include_attributes=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--install", action="store_true", help="install the refreshed wheel"
    )
    args = parser.parse_args()
    root = Path.cwd().resolve()
    relative = Path("operators/utils/transformer_compat.py")
    source = root / "python/sgl_kernel_npu/sgl_kernel_npu" / relative
    if not source.is_file():
        raise RuntimeError(
            "Run from the PR #939 repository root; missing: " + str(source)
        )
    text = source.read_text()
    if not any(
        isinstance(node, ast.FunctionDef)
        and node.name == "get_transformer_metadata_size"
        for node in ast.parse(text).body
    ):
        raise RuntimeError(
            "Checkout lacks get_transformer_metadata_size; update PR #939 first"
        )
    package = root / "build/torch/package/sgl_kernel_npu"
    required = (
        relative,
        Path("lib/libsgl_kernel_npu.so"),
        Path("_build_manifest.json"),
    )
    from_installed = not all((package / path).is_file() for path in required)
    if from_installed:
        # Read distribution metadata without importing/initializing the NPU package.
        distribution = importlib.metadata.distribution("sgl_kernel_npu")
        package = Path(distribution.locate_file("sgl_kernel_npu")).resolve()
        for path in required:
            if not (package / path).is_file():
                raise RuntimeError(
                    "Installed package is incomplete; missing: " + str(package / path)
                )
    target = package / relative
    if without_size_helper(text) != without_size_helper(target.read_text()):
        raise RuntimeError(
            "The staged module has other code differences. This script only refreshes "
            "get_transformer_metadata_size; rebuild from the matching checkout instead."
        )
    manifest = json.loads((package / "_build_manifest.json").read_text())
    if args.install:
        distribution = importlib.metadata.distribution("sgl_kernel_npu")
        installed_manifest = Path(
            distribution.locate_file("sgl_kernel_npu/_build_manifest.json")
        )
        if (
            not installed_manifest.is_file()
            or json.loads(installed_manifest.read_text()) != manifest
        ):
            raise RuntimeError(
                "Installed and staged build manifests differ. Use the matching build "
                "directory, or omit --install to create a wheel for manual review."
            )
    if from_installed:
        (root / "build").mkdir(exist_ok=True)
        work = Path(
            tempfile.mkdtemp(prefix="metadata-helper-refresh-", dir=root / "build")
        )
        staged = work / "torch/package/sgl_kernel_npu"
        staged.parent.mkdir(parents=True)
        shutil.copytree(
            package, staged, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
        )
        print("Reusing installed package:", package, flush=True)
        package = staged
        target = package / relative
    backup = root / "build/metadata-helper-refresh/transformer_compat.py.before"
    backup.parent.mkdir(parents=True, exist_ok=True)
    if not backup.exists():
        shutil.copy2(target, backup)
    shutil.copy2(source, target)
    sys.path.insert(0, str(root / "python/sgl_kernel_npu"))
    from build_tools.torch_wheel import build_wheel

    wheel = build_wheel(package, source_root=root)
    output = root / "output" / wheel.name
    output.parent.mkdir(exist_ok=True)
    shutil.copy2(wheel, output)
    print("Refreshed wheel:", output, flush=True)
    if args.install:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--force-reinstall",
                "--no-deps",
                str(output),
            ],
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                "-c",
                "from sgl_kernel_npu.operators.utils.transformer_compat import "
                "get_transformer_metadata_size; "
                "print('Import OK:', get_transformer_metadata_size)",
            ],
            check=True,
        )


if __name__ == "__main__":
    try:
        main()
    except (
        OSError,
        RuntimeError,
        ValueError,
        importlib.metadata.PackageNotFoundError,
        subprocess.CalledProcessError,
    ) as error:
        print("FAIL:", error, file=sys.stderr)
        sys.exit(1)
