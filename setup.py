#!/usr/bin/env python3
"""Build sgl-kernel-npu from selected CANN and host C++ operators and Python sources."""
import argparse
import os
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "python/sgl_kernel_npu"))

from build_tools.cann import CannBuild, build_cann
from build_tools.common import (
    PROJECT_ROOT,
    _cache,
    _remove_tree,
    _selection,
    architecture,
    parse_operators,
    run,
)
from build_tools.host import build_host
from build_tools.torch import build_torch, configure_torch


def configure_native(
    arch_version,
    operators=("ALL",),
    *,
    artifact="wheel",
    source_root=PROJECT_ROOT,
    build_root=None,
):
    """Configure the existing csrc entry once; return CANN configuration and separate HOST IDs.

    HOST-only configuration needs no custom-op SDK or compiler. CANN requests
    configure their real SDK targets here, and build_cann reuses that directory.
    """
    architecture(arch_version, source_root)
    if artifact not in ("run", "wheel"):
        raise ValueError("--artifact must be run or wheel")
    source_root = Path(source_root).resolve()
    build_root = Path(build_root) if build_root is not None else source_root / "build"
    if build_root.is_symlink():
        raise RuntimeError("Refusing a symlinked build root")
    directory = build_root.absolute() / "cann"
    if directory.is_symlink():
        raise RuntimeError("Refusing a symlinked CANN directory")
    env = os.environ.copy()
    sdk = env.get("ASCEND_HOME_PATH", "")
    if sdk:
        sdk = str(Path(sdk).resolve())
        env["ASCEND_HOME_PATH"] = sdk
    expected = {
        "CMAKE_HOME_DIRECTORY": str(source_root / "csrc"),
        "SGL_ARCH_VERSION": str(arch_version),
        "SGL_NATIVE_OPERATORS": ";".join(operators),
        "SGL_NATIVE_ARTIFACT": artifact,
        "SGL_CANN_BUILD_STAGE": "KERNEL",
    }
    cached = _cache(directory)
    # Reuse the existing CANN SDK snapshot only for configurations that need it.
    if cached.get("SGL_CANN_CONFIGURED_OPERATORS"):
        expected["SGL_CANN_SDK_ROOT"] = sdk
    if cached and any(cached.get(key) != value for key, value in expected.items()):
        raise RuntimeError("Build configuration changed; rerun with --make-clean")
    directory.mkdir(parents=True, exist_ok=True)
    run(
        [
            "cmake",
            "-S",
            source_root / "csrc",
            "-B",
            directory,
            "-DSGL_ARCH_VERSION=" + str(arch_version),
            "-DSGL_NATIVE_OPERATORS=" + ";".join(operators),
            "-DSGL_NATIVE_ARTIFACT=" + artifact,
            "-DSGL_CANN_BUILD_STAGE=KERNEL",
        ],
        directory / "configure.log",
        cwd=source_root,
        env=env,
    )
    for name in ("selected-operators.txt", "selected-host-operators.txt"):
        if not (directory / name).is_file():
            raise RuntimeError("CMake did not export native selection: " + name)
    selected = _selection(directory)
    host = _selection(directory, "selected-host-operators.txt")
    if not selected and not host:
        raise RuntimeError("CMake exported an empty native selection")
    if artifact == "run" and host:
        raise RuntimeError("CMake selected HOST operators for a CANN RUN package")
    print("BUILD_PLAN: cann={} host={}".format(selected, host), flush=True)
    configured = None
    if selected:
        configured = CannBuild(
            directory,
            selected,
            arch_version=str(arch_version),
            selection_kind="all" if operators == ("ALL",) else "custom",
        )
    return configured, host


def arguments(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--arch-version",
        choices=("2201", "3510"),
        help="required architecture coverage, except for standalone --make-clean",
    )
    parser.add_argument(
        "--ops",
        default="all",
        help="all supported migrated native operators, or comma-separated operator IDs",
    )
    parser.add_argument(
        "--artifact",
        choices=("run", "wheel"),
        default="wheel",
        help="CANN installer or complete sgl-kernel-npu wheel",
    )
    parser.add_argument("-j", "--jobs", type=int, help="parallel compilation jobs")
    third_party = parser.add_mutually_exclusive_group()
    third_party.add_argument(
        "--third-party",
        help="comma-separated third-party profile filenames without .json; default: all",
    )
    third_party.add_argument(
        "--without-third-party",
        action="store_true",
        help="build only repository-owned operators",
    )
    parser.add_argument(
        "--plan",
        action="store_true",
        help="print third-party build commands without building",
    )
    parser.add_argument(
        "--cann-version",
        help="CANN profile version for --plan only; builds detect the SDK version",
    )
    parser.add_argument(
        "--make-clean",
        action="store_true",
        help="remove this build's intermediates; used alone, clean and exit",
    )
    args = parser.parse_args(argv)
    if args.cann_version and not args.plan:
        parser.error("--cann-version is only available with --plan")
    if not args.arch_version and argv != ["--make-clean"]:
        parser.error("--arch-version is required")
    if args.jobs is not None and args.jobs < 1:
        parser.error("--jobs must be positive")
    if args.third_party is not None:
        if args.artifact != "wheel":
            parser.error("--third-party requires --artifact wheel")
        args.third_party = tuple(name.strip() for name in args.third_party.split(","))
        if any(
            not re.fullmatch(r"[a-z][a-z0-9_]*", name) for name in args.third_party
        ) or len(set(args.third_party)) != len(args.third_party):
            parser.error("--third-party requires distinct, nonempty profile names")
    try:
        args.operators = parse_operators(args.ops)
    except ValueError as error:
        parser.error(str(error))
    return args


def main(argv=None):
    args = arguments(argv)
    if (PROJECT_ROOT / "build").is_symlink():
        raise RuntimeError("Refusing a symlinked build root")
    if args.make_clean:
        for name in ("cann", "torch", "wheel"):
            _remove_tree(PROJECT_ROOT / "build" / name)
        if not args.arch_version:
            print("sgl-kernel-npu intermediates removed")
            return

    from build_tools.third_party import (
        build_dependencies,
        detect_cann_version,
        resolve_plan,
    )

    result = None
    host_operators = ()
    if not args.plan:
        result, host_operators = configure_native(
            args.arch_version, args.operators, artifact=args.artifact
        )

    plan = []
    if not args.without_third_party and args.artifact == "wheel":
        plan = resolve_plan(
            args.cann_version or detect_cann_version(),
            args.arch_version,
            args.jobs,
            third_party=args.third_party,
        )
    if args.plan:
        import json

        print(json.dumps(plan, indent=2))
        return
    dependencies = build_dependencies(plan) if plan else []

    output = PROJECT_ROOT / "output"
    output.mkdir(exist_ok=True)
    extension = "whl" if args.artifact == "wheel" else "run"
    for path in output.glob(f"sgl_kernel_npu-*.{extension}"):
        path.unlink()
    if result is not None:
        result = build_cann(result, jobs=args.jobs)
    if args.artifact == "run":
        destination = (
            output / f"sgl_kernel_npu-{args.arch_version}-{result.selection_kind}.run"
        )
        shutil.copy2(result.package, destination)
    else:
        from build_tools.torch_wheel import build_wheel, stage_package

        configured_torch = configure_torch(
            args.arch_version,
            host_operators,
            cann=result,
            selection_kind="all" if args.operators == ("ALL",) else "custom",
        )
        host = (
            build_host(
                args.arch_version,
                host_operators,
                directory=configured_torch.directory,
                jobs=args.jobs,
            )
            if host_operators
            else None
        )
        torch = build_torch(configured_torch, host=host, jobs=args.jobs)
        package = stage_package(torch, cann=result, dependencies=dependencies)
        wheel = build_wheel(package)
        destination = output / wheel.name
        shutil.copy2(wheel, destination)
    print(f"ARTIFACT: {destination}")


if __name__ == "__main__":
    if os.environ.get("SGL_REFACTOR_STAGED_PACKAGE"):
        from build_tools.torch_wheel import setup_refactor

        setup_refactor()
    else:
        try:
            main()
        except (OSError, RuntimeError, ValueError) as error:
            print(f"FAIL: {error}", file=sys.stderr)
            sys.exit(1)
