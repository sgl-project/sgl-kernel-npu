"""Coordinate selected REFACTOR packages without changing the default build."""

import argparse
import os
from pathlib import Path
import shutil
import sys

from build_tools.cann import PROJECT_ROOT, _remove_tree, build_cann, parse_operators, run


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-a", "--package", choices=("kernels", "deepep", "deepep2", "memory-saver", "attentions", "all"), default="kernels")
    parser.add_argument("--arch-version", choices=("2201", "3510"))
    parser.add_argument("--ops", help="all (default) or comma-separated migrated operator IDs")
    parser.add_argument("--artifact", choices=("run", "wheel"), default="wheel")
    parser.add_argument("-j", "--jobs", type=int)
    parser.add_argument("--make_clean", action="store_true", help="clear REFACTOR intermediates before building; alone, clean and exit")
    parser.add_argument("--declarations-only", action="store_true", help="validate operator selection without compiling or packaging")
    args = parser.parse_args(argv)
    if not args.arch_version and not args.make_clean:
        parser.error("--arch-version is required")
    if args.jobs is not None and args.jobs < 1:
        parser.error("--jobs must be positive")
    if args.ops is not None and args.package not in ("kernels", "all"):
        parser.error("--ops requires selecting kernels")
    if (args.artifact == "run" or args.declarations_only) and args.package != "kernels":
        parser.error("run and --declarations-only are available only for -a kernels")
    if args.arch_version == "3510" and args.package in ("attentions", "deepep2", "all"):
        parser.error("the existing attentions and deepep2 builds do not support ARCH 3510; select supported packages explicitly")
    try:
        args.operators = parse_operators(args.ops or "all")
    except ValueError as error:
        parser.error(str(error))
    return args


def clear_outputs(output, prefix):
    output.mkdir(parents=True, exist_ok=True)
    for path in output.glob(prefix + "*.whl"):
        path.unlink()


def build_auxiliary(package, arch_version, source_root=PROJECT_ROOT):
    """Reuse independent projects, preserving each DeepEP hardware variant."""
    variants = {"2201": [("a2", "Ascend910B1"), ("a3", "Ascend910_9382")],
                "3510": [("950", "Ascend950")]}
    selected = variants[arch_version] if package == "deepep" else [("a2", "Ascend910B1")] if package == "deepep2" else [(package, "")]
    output = source_root / "output"
    for variant, target in selected:
        destination = output / ("deepep_" + variant) if package.startswith("deepep") else output
        prefix = "deep_ep" if package.startswith("deepep") else "torch_memory_saver" if package == "memory-saver" else "attentions"
        clear_outputs(destination, prefix)
        directory = source_root / "build/independent" / variant
        directory.mkdir(parents=True, exist_ok=True)
        # Values are positional arguments, never interpolated into shell code.
        run(["bash", "-c", 'source "$1/build.sh"; build_refactor_auxiliary "$2" "$3" "$4" "$5"',
             "refactor-auxiliary", source_root, package, target, directory, destination],
            directory / "build.log", cwd=source_root,
            env=dict(os.environ, SGL_BUILD_MODE="REFACTOR"))
        if not list(destination.glob(prefix + "*.whl")):
            raise RuntimeError(f"Independent build produced no {prefix} wheel in {destination}")


def main(argv=None):
    args = arguments(argv)
    if args.make_clean:
        for name in ("cann", "torch", "wheel", "independent"):
            _remove_tree(PROJECT_ROOT / "build" / name)
        if not args.arch_version:
            print("REFACTOR intermediates removed")
            return
    if args.package in ("kernels", "all"):
        output = PROJECT_ROOT / "output"
        if not args.declarations_only:
            output.mkdir(exist_ok=True)
            if args.artifact == "wheel":
                clear_outputs(output, "sgl_kernel_npu")
            else:
                for path in output.glob("sgl_kernel_npu-*.run"):
                    path.unlink()
        result = build_cann(args.arch_version, args.operators,
                            declarations_only=args.declarations_only, jobs=args.jobs)
        if not args.declarations_only:
            if args.artifact == "run":
                destination = output / f"sgl_kernel_npu-{args.arch_version}-{result.selection_kind}.run"
                shutil.copy2(result.package, destination)
            else:
                from build_tools.torch_wheel import build_torch, build_wheel
                package = build_torch(result, args.arch_version, jobs=args.jobs)
                wheel = build_wheel(package)
                destination = output / wheel.name
                shutil.copy2(wheel, destination)
            print(f"ARTIFACT: {destination}")
    packages = ("deepep", "memory-saver", "attentions") if args.package == "all" else (args.package,)
    for package in packages:
        if package != "kernels":
            build_auxiliary(package, args.arch_version)
