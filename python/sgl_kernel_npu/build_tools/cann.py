"""Build, install and verify CANN operators (AICORE/AICPU), independently of HOST."""

import os
import platform
from dataclasses import dataclass
from pathlib import Path

from build_tools.common import (
    PROJECT_ROOT,
    _cache,
    _cache_list,
    _remove_tree,
    _selection,
    architecture,
    run,
)


@dataclass
class CannBuild:
    directory: Path
    operators: tuple
    package: Path = None
    vendor: Path = None
    arch_version: str = ""
    selection_kind: str = "all"


def build_cann(configured: CannBuild, *, jobs=None, source_root=PROJECT_ROOT):
    """Build and stage CANN artifacts from a required, existing configuration."""
    if configured is None:
        raise ValueError("build_cann requires a configured CANN build")
    if not configured.operators:
        raise ValueError("No CANN operators selected; skip build_cann")
    arch_version = str(configured.arch_version)
    compute_units = architecture(arch_version, source_root)["compute_units"]
    if jobs is not None and jobs < 1:
        raise ValueError("--jobs must be positive")
    source_root = Path(source_root).resolve()
    directory = Path(configured.directory).absolute()
    if directory.parent.is_symlink():
        raise RuntimeError(
            "Refusing a symlinked build root: {}".format(directory.parent)
        )
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
        "SGL_ARCH_VERSION": arch_version,
        "SGL_CANN_BUILD_STAGE": "KERNEL",
        "SGL_CANN_SDK_ROOT": sdk,
        "SGL_CANN_CONFIGURED_OPERATORS": ";".join(configured.operators),
    }
    cached = _cache(directory)
    if directory.is_symlink() or any(
        cached.get(key) != value for key, value in expected.items()
    ):
        raise RuntimeError("Configured CANN build changed; rerun with --make-clean")
    selected = _selection(directory)
    if selected != configured.operators:
        raise RuntimeError(
            "Configured native selection no longer matches CMake outputs"
        )
    aicpu_operators = _cache_list(_cache(directory), "SGL_CANN_AICPU_OPERATORS")
    if not set(aicpu_operators).issubset(selected):
        raise RuntimeError("CMake exported AICPU operators outside this selection")
    has_aicpu = bool(aicpu_operators)
    has_aicore = bool(set(selected).difference(aicpu_operators))
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
    install_command = ["bash", package, "--quiet", "--install-path=" + str(installed)]
    if has_aicpu and not has_aicore:
        # CANN 9.2 requires opmaster even without Core Tiling. --force skips
        # the installer's entire Host-library validation, not file installation.
        install_command.append("--force")
    run(
        install_command,
        directory / "install.log",
        cwd=installed,
        env=env,
    )
    vendor, _, _ = inspect_installation(
        installed, directory, compute_units, aicore=has_aicore, aicpu=has_aicpu
    )
    print("RUN_PACKAGE: " + str(package), flush=True)
    print("INSTALLED_VENDOR: " + str(vendor), flush=True)
    print("PASS: CANN package built and staged; no NPU execution or wheel build")
    return CannBuild(
        directory,
        selected,
        package,
        vendor,
        str(arch_version),
        configured.selection_kind,
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


def inspect_installation(install_dir, stage_build, compute_units, *, aicore, aicpu):
    vendor = install_dir / "vendors/sgl_kernel_npu"
    tbe = vendor / "op_impl/ai_core/tbe"
    cache = _cache(stage_build)
    host_libraries = _cache_list(cache, "SGL_CANN_HOST_LIBRARIES")
    headers = [Path(path) for path in _cache_list(cache, "SGL_CANN_ACLNN_HEADERS")]
    libraries = []
    if "sgl_cann_opapi" in host_libraries:
        libraries.append(vendor / "op_api/lib/libcust_opapi.so")
        if not headers:
            raise RuntimeError("CMake created ACLNN without exporting its headers")
    library_searches = []
    if "sgl_cann_proto" in host_libraries:
        library_searches.append((vendor / "op_proto", "libcust_opsproto_rt2.0.so"))
    if aicore:
        library_searches.append((tbe / "op_tiling", "libcust_opmaster_rt2.0.so"))
    for directory, name in library_searches:
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

    # CMake may rewrite shared-library RPATHs; compare headers and kernels instead.
    pairs = [(path, vendor / "op_api/include" / path.name) for path in headers]
    binary_root = stage_build / "sgl_cann_kernels/binary"
    for compute_unit in compute_units if aicore else ():
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
    if aicpu:
        cpu = vendor / "op_impl/cpu"
        pairs += [
            (
                stage_build / "aicpu/cust_aicpu_kernel.json",
                cpu / "config/cust_aicpu_kernel.json",
            ),
            (
                stage_build / "aicpu/libsgl_kernel_npu_aicpu.so",
                cpu / "aicpu_kernel/impl/libsgl_kernel_npu_aicpu.so",
            ),
        ]
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
