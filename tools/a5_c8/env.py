#!/usr/bin/env python3
"""Print shell exports for the installed original OPP without importing torch."""

import argparse
import importlib.util
import os
import shlex
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path)
    args = parser.parse_args()
    root = args.package_root
    if root is None:
        spec = importlib.util.find_spec("sgl_kernel_npu")
        if spec is None or spec.origin is None:
            parser.error("sgl_kernel_npu is not installed")
        root = Path(spec.origin).parent
    helper = root / "_original_opp.py"
    if not helper.is_file():
        parser.error("selected kernel package lacks the original OPP adapter")
    spec = importlib.util.spec_from_file_location("_sgl_original_opp_env", helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifest = module.activate(root)
    if manifest is None:
        parser.error("selected kernel package does not contain the original OPP")
    vendor = root.resolve() / "lib/a5_original_opp" / manifest["vendor"]
    libs = [str(vendor / "op_api/lib")]
    libs.extend(p for p in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep) if p)
    print(
        "export ASCEND_CUSTOM_OPP_PATH="
        + shlex.quote(os.environ["ASCEND_CUSTOM_OPP_PATH"])
    )
    print("export LD_LIBRARY_PATH=" + shlex.quote(os.pathsep.join(dict.fromkeys(libs))))


if __name__ == "__main__":
    main()
