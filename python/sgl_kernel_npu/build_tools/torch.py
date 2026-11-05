"""Configure the shared Torch project and build its registration/link target."""

import importlib.metadata
import json
import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path

from build_tools.common import PROJECT_ROOT, _cache, architecture, run


@dataclass(frozen=True)
class TorchBuild:
    directory: Path
    operators: tuple
    cann_operators: tuple
    arch_version: str
    selection_kind: str
    cann_vendor: Path = None
    library: Path = None


def configure_torch(
    arch_version,
    operators=(),
    *,
    cann=None,
    selection_kind=None,
    source_root=PROJECT_ROOT,
    build_root=None,
):
    """Configure one project for HOST objects and Torch adapters, without compiling."""
    source_root = Path(source_root).resolve()
    arch_version = str(arch_version)
    architecture(arch_version, source_root)
    if cann is not None and cann.vendor is None:
        raise RuntimeError("Torch requires a built and installed CANN package")
    if cann is not None and cann.arch_version != str(arch_version):
        raise ValueError("Torch architecture differs from configured native build")
    cann_ids = cann.operators if cann is not None else ()
    host_ids = tuple(operators)
    if not cann_ids and not host_ids:
        raise ValueError("No native operators selected")
    if selection_kind is None:
        selection_kind = cann.selection_kind if cann is not None else "custom"
    if selection_kind not in ("all", "custom"):
        raise ValueError("Invalid Torch selection kind")
    if cann is not None and cann.selection_kind != selection_kind:
        raise ValueError("Torch selection kind differs from CANN build")
    vendor = cann.vendor if cann is not None else None
    if build_root is None:
        build_root = (
            cann.directory.parent if cann is not None else source_root / "build"
        )
    build_root = Path(build_root)
    if build_root.is_symlink():
        raise RuntimeError("Refusing a symlinked build root")
    directory = build_root.absolute() / "torch"
    if directory.is_symlink():
        raise RuntimeError("Refusing a symlinked Torch build directory")
    expected = {
        "CMAKE_HOME_DIRECTORY": str(source_root / "csrc/aclnn_torch_adapter"),
        "SGL_ARCH_VERSION": arch_version,
        "SGL_CANN_OPERATORS": ";".join(cann_ids),
        "SGL_HOST_OPERATORS": ";".join(host_ids),
        "SGL_CANN_VENDOR": str(vendor) if vendor is not None else "",
        "Python3_EXECUTABLE": sys.executable,
        "SGL_TORCH_LIBRARY_DIR": str(directory / "lib"),
    }
    cached = _cache(directory)
    if cached and any(cached.get(key) != value for key, value in expected.items()):
        raise RuntimeError("Torch build configuration changed; rerun with --make-clean")
    environment = {
        "python": sys.executable,
        "python_version": sys.version,
        "sdk": os.environ.get("ASCEND_HOME_PATH", ""),
        "selection_kind": selection_kind,
    }
    for distribution in ("torch", "torch_npu"):
        try:
            environment[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            environment[distribution] = None  # CMake validates real Torch imports.
    signature = directory / "torch-environment.json"
    if signature.exists() and json.loads(signature.read_text()) != environment:
        raise RuntimeError(
            "Torch/Python/SDK environment changed; rerun with --make-clean"
        )
    directory.mkdir(parents=True, exist_ok=True)
    signature.write_text(json.dumps(environment, sort_keys=True) + "\n")
    run(
        [
            "cmake",
            "-S",
            source_root / "csrc/aclnn_torch_adapter",
            "-B",
            directory,
            "-DCMAKE_BUILD_TYPE=Release",
            "-DPython3_EXECUTABLE=" + sys.executable,
            "-DSGL_ARCH_VERSION=" + arch_version,
            "-DSGL_CANN_OPERATORS=" + ";".join(cann_ids),
            "-DSGL_HOST_OPERATORS=" + ";".join(host_ids),
            "-DSGL_CANN_VENDOR=" + (str(vendor) if vendor is not None else ""),
            "-DSGL_TORCH_LIBRARY_DIR=" + str(directory / "lib"),
        ],
        directory / "configure.log",
        cwd=source_root,
    )
    return TorchBuild(
        directory, host_ids, cann_ids, arch_version, selection_kind, vendor
    )


def build_torch(configured: TorchBuild, *, host=None, jobs=None):
    """Compile selected adapters and link the existing single shared extension."""
    if jobs is not None and jobs < 1:
        raise ValueError("--jobs must be positive")
    if configured.operators:
        if host is None or (
            host.directory != configured.directory
            or host.operators != configured.operators
            or host.arch_version != configured.arch_version
        ):
            raise ValueError("HOST objects differ from the configured Torch extension")
    elif host is not None:
        raise ValueError("No HOST objects expected for this Torch extension")
    directory = configured.directory
    command = ["cmake", "--build", directory, "--target", "sgl_kernel_npu"]
    if jobs is not None:
        command += ["--parallel", str(jobs)]
    run(command, directory / "build.log")
    library = directory / "lib/libsgl_kernel_npu.so"
    if not library.is_file() or not library.stat().st_size:
        raise RuntimeError("Missing Torch library: {}".format(library))
    print("TORCH_LIBRARY: " + str(library), flush=True)
    return replace(configured, library=library)
