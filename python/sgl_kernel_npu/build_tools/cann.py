"""Build a CANN package independently of Torch; reusable by wheel build hooks."""

import json
import os
import platform
import re
import shlex
import shutil
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class CannBuild:
    directory: Path
    operators: tuple
    package: Path = None
    vendor: Path = None
    arch_version: str = ""
    selection_kind: str = "all"


def architecture(arch_version, source_root=PROJECT_ROOT):
    architectures = json.loads(
        (Path(source_root) / "csrc/cmake/architectures.json").read_text()
    )
    if str(arch_version) not in architectures:
        raise ValueError("--arch-version must be 2201 or 3510")
    return architectures[str(arch_version)]


def run(command, log_path, *, cwd=None, env=None):
    command = [str(value) for value in command]
    display = " ".join(shlex.quote(value) for value in command)
    print("\n$ " + display, flush=True)
    with log_path.open("w") as log:
        log.write(display + "\n")
        log.flush()
        with subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        ) as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
            result = process.wait()
    if result:
        raise RuntimeError("Command failed ({}); see {}".format(result, log_path))


def _remove_tree(path):
    if path.is_symlink():
        raise RuntimeError(
            "Refusing to clean a symlinked build directory: {}".format(path)
        )
    if not path.exists():
        return

    # SDK installers may make vendor directories read-only. Do not follow links.
    for directory, _, _ in os.walk(path, followlinks=False):
        entry = Path(directory)
        entry.chmod(entry.stat().st_mode | stat.S_IWUSR | stat.S_IXUSR)
    shutil.rmtree(path)


def _cache(directory):
    values = {}
    path = directory / "CMakeCache.txt"
    if path.is_file():
        for line in path.read_text().splitlines():
            if (
                not line.startswith(("//", "#"))
                and "=" in line
                and ":" in line.split("=", 1)[0]
            ):
                key, value = line.split("=", 1)
                values[key.split(":", 1)[0]] = value
    return values


def _selection(directory):
    path = directory / "selected-operators.txt"
    return tuple(path.read_text().splitlines()) if path.is_file() else ()


def parse_operators(value):
    if value == "all":
        return ("ALL",)
    operators = tuple(part.strip() for part in value.split(","))
    if not operators or any(
        not re.fullmatch(r"[a-z][a-z0-9_]*", item) for item in operators
    ):
        raise ValueError("--ops must be 'all' or comma-separated operator IDs")
    if "all" in operators or len(set(operators)) != len(operators):
        raise ValueError("--ops must not mix 'all' with IDs or repeat an ID")
    return operators


def build_cann(
    arch_version,
    operators=("ALL",),
    *,
    jobs=None,
    source_root=PROJECT_ROOT,
    build_root=None,
):
    """Build and stage a CANN package for the selected operators."""
    compute_units = architecture(arch_version, source_root)["compute_units"]
    if jobs is not None and jobs < 1:
        raise ValueError("--jobs must be positive")
    source_root = Path(source_root).resolve()
    build_root = Path(build_root) if build_root is not None else source_root / "build"
    if build_root.is_symlink():
        raise RuntimeError("Refusing a symlinked build root: {}".format(build_root))
    build_root = build_root.absolute()
    directory = build_root / "cann"
    env = os.environ.copy()
    if platform.system() != "Linux":
        raise RuntimeError("CANN compilation requires Linux")
    sdk_path = Path(env.get("ASCEND_HOME_PATH", ""))
    if not env.get("ASCEND_HOME_PATH") or not sdk_path.is_dir():
        raise RuntimeError(
            "Initialize CANN first: ASCEND_HOME_PATH must point to the SDK"
        )
    sdk = str(sdk_path.resolve())
    env["ASCEND_HOME_PATH"] = sdk

    expected = {
        "CMAKE_HOME_DIRECTORY": str(source_root / "csrc"),
        "SGL_ARCH_VERSION": str(arch_version),
        "SGL_CANN_OPERATORS": ";".join(operators),
        "SGL_CANN_SDK_ROOT": sdk,
    }
    cached = _cache(directory)
    if directory.is_symlink():
        raise RuntimeError("Refusing a symlinked CANN directory: {}".format(directory))
    if cached and any(cached.get(k) != v for k, v in expected.items()):
        raise RuntimeError("Build configuration changed; rerun with --make_clean")
    directory.mkdir(parents=True, exist_ok=True)
    command = [
        "cmake",
        "-S",
        source_root / "csrc",
        "-B",
        directory,
        "-DSGL_ARCH_VERSION=" + str(arch_version),
        "-DSGL_CANN_OPERATORS=" + ";".join(operators),
        "-DSGL_CANN_BUILD_STAGE=KERNEL",
    ]
    run(command, directory / "configure.log", cwd=source_root, env=env)
    selected = _selection(directory)
    if not selected:
        raise RuntimeError("CMake did not produce a nonempty operator selection")
    print("SELECTED_OPERATORS: " + ",".join(selected), flush=True)
    # The SDK's package target does not build device binaries automatically.
    for target, log_name in (
        ("sgl_cann_kernel_check", "kernel-build.log"),
        ("package", "build.log"),
    ):
        build_command = ["cmake", "--build", directory, "--target", target]
        if jobs is not None:
            build_command += ["--parallel", str(jobs)]
        run(build_command, directory / log_name, cwd=source_root, env=env)
    package = find_run_package(directory)
    installed = directory / "installed"
    _remove_tree(installed)
    installed.mkdir()
    run(
        ["bash", package, "--quiet", "--install-path=" + str(installed)],
        directory / "install.log",
        cwd=installed,
        env=env,
    )
    for compute_unit in compute_units:
        vendor, _, _ = inspect_installation(installed, directory, compute_unit)
    print("RUN_PACKAGE: " + str(package), flush=True)
    print("INSTALLED_VENDOR: " + str(vendor), flush=True)
    print("PASS: CANN package built and staged; no NPU execution or wheel build")
    return CannBuild(
        directory,
        selected,
        package,
        vendor,
        str(arch_version),
        "all" if operators == ("ALL",) else "custom",
    )


def _inside(path, directory):
    try:
        path.resolve().relative_to(directory.resolve())
        return True
    except ValueError:
        return False


def find_run_package(stage_build):
    packages = sorted((stage_build / "package-stage").glob("*.run"))
    if (
        len(packages) != 1
        or not packages[0].is_file()
        or not packages[0].stat().st_size
    ):
        raise RuntimeError(
            f"Expected one nonempty .run package in {stage_build / 'package-stage'}; found {packages}"
        )
    return packages[0]


def inspect_installation(install_dir, stage_build, compute_unit):
    vendor = install_dir / "vendors/sgl_kernel_npu"
    tbe = vendor / "op_impl/ai_core/tbe"
    libraries = [vendor / "op_api/lib/libcust_opapi.so"]
    for directory, name in (
        (tbe / "op_tiling", "libcust_opmaster_rt2.0.so"),
        (vendor / "op_proto", "libcust_opsproto_rt2.0.so"),
    ):
        matches = sorted(directory.rglob(name))
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected one installed {name} under {directory}; found {matches}"
            )
        libraries.extend(matches)
    for path in [*libraries, vendor / "bin/set_env.bash"]:
        if (
            not path.is_file()
            or not path.stat().st_size
            or not _inside(path, install_dir)
        ):
            raise RuntimeError(f"Missing, empty or external installed file: {path}")

    headers = sorted((stage_build / "autogen").glob("aclnn*.h"))
    if not headers:
        raise RuntimeError(f"No generated ACLNN headers in {stage_build / 'autogen'}")
    # CMake may rewrite shared-library RPATHs; compare headers and kernels instead.
    pairs = [(path, vendor / "op_api/include" / path.name) for path in headers]
    binary_root = stage_build / "sgl_cann_kernels/binary"
    binaries = sorted((binary_root / compute_unit).rglob("*.o"))
    metadata = sorted((binary_root / compute_unit).rglob("*.json"))
    metadata += sorted((binary_root / "config" / compute_unit).rglob("*.json"))
    if not binaries or not metadata:
        raise RuntimeError(
            f"Missing compiled Kernel binaries or metadata for {compute_unit}"
        )
    pairs += [
        (path, tbe / "kernel" / path.relative_to(binary_root))
        for path in binaries + metadata
    ]
    info = f"aic-{compute_unit}-ops-info.json"
    pairs.append(
        (
            stage_build
            / "sgl_cann_kernels/tbe/op_info_cfg/ai_core"
            / compute_unit
            / info,
            tbe / "config" / compute_unit / info,
        )
    )
    for source, installed in pairs:
        if not source.is_file() or not source.stat().st_size:
            raise RuntimeError(f"Missing or empty build artifact: {source}")
        if not installed.is_file() or not _inside(installed, install_dir):
            raise RuntimeError(f"Missing or external installed artifact: {installed}")
        if installed.read_bytes() != source.read_bytes():
            raise RuntimeError(
                f"Installed artifact differs from this build: {installed}"
            )
    return vendor, libraries, [installed for _, installed in pairs]
